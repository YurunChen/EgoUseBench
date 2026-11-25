from __future__ import annotations

import json
import re

import pytest
from PIL import Image

from egocua.answer_validation import AnswerValidationWorkspace
from egocua.human_reference import AnswerCase
from egocua.environment import task_public_site_data, load_site_data
from egocua.protocol import RunSession
from egocua.review_web import create_answer_validation_app
from egocua.timeline import complete_materials_condition
from tests.conditions import CONDITIONS
from tests.helpers import products, task, materialize_site, item_id


@pytest.fixture
def workspace(tmp_path):
    sample = tmp_path / "data/experiments/human-reference.json"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        json.dumps(
            {
                "selection_seed": 10,
                "tasks": [
                    {"task_id": "a", "stratum": "clarify:variant"},
                    {"task_id": "b", "stratum": "composite:CG"},
                ],
            }
        )
    )
    workspace = AnswerValidationWorkspace(tmp_path, "person-01", tmp_path / "records")
    from dataclasses import replace

    for item, track in [(1, "clarify"), (2, "graft")]:
        definition = task(track)
        definition["visual_session"] = {
            "frequency_samples": {
                "window_seconds": [0, 3],
                "timestamps_seconds": [0, 1, 2, 3],
                "frame_paths": ["frame.jpg"] * 4,
            },
            "update_onset_seconds": 2,
        }
        condition = replace(
            CONDITIONS["online"],
            batch_seconds=1,
            frame_interval_seconds=1,
            frame_delivery_steps=1,
            wearer_frames_per_image=4,
            replay_timing="action_aligned",
        )
        materialize_site(products(track), tmp_path)
        public = task_public_site_data(definition, load_site_data(tmp_path, "egomart"))
        # Put hidden metadata into the raw catalog before projection in the dedicated test below.
        for entity in public["entities"]:
            path = tmp_path / entity["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), "green").save(path)
        workspace.materials._cases[item] = AnswerCase(
            definition,
            RunSession(definition, complete_materials_condition(condition), 2, None),
            public,
            tmp_path,
            {"sample": workspace.rows[item - 1], "study": "frozen-original"},
        )
    Image.new("RGB", (32, 24), "blue").save(tmp_path / "frame.jpg")
    return workspace


def payload(workspace, item=1, **updates):
    return {
        "action": "submit",
        "revision": workspace.record(item)["revision"],
        "judgment": "selected",
        "choices": {"0": 1},
        "fields": {},
        "note": "",
        **updates,
    }


def test_independent_choice_is_scored_but_not_exposed_or_mixed_with_model_runs(
    workspace,
):
    workspace.open(1)
    public = workspace.update(1, payload(workspace, choices={"1": 1}))
    assert public["submitted_at"] and "matches_registered_answer" not in public
    record = json.loads((workspace.output / "01.json").read_text())
    assert record["matches_registered_answer"] is True
    assert record["selected_state"]["selected_entity_ids"] == [item_id("clarify", 1)]
    assert not (workspace.root / "experiments").exists()
    assert not (workspace.root / "data/reviews").exists()
    assert workspace.summary()["groups"]["single"]["matched"] == 1
    with pytest.raises(ValueError, match="不能覆盖"):
        workspace.update(1, payload(workspace, choices={"1": 1}))


def test_draft_and_question_survive_reopening_without_revealing_answer_early(workspace):
    initial = workspace.open(1)
    assert not initial["questions"]
    workspace.update(
        1, payload(workspace, action="save", choices={"1": 2}, note="my draft")
    )
    workspace.update(1, {"action": "ask", "revision": 1, "question": "Which one?"})
    assert (
        workspace.record(1)["questions"][0]["answer_event"]["payload"]
        == "The target variant."
    )
    restored = AnswerValidationWorkspace(workspace.root, "person-01", workspace.output)
    restored.materials._cases = workspace.materials._cases
    assert restored.public_record(1)["choices"] == {"1": 2}
    assert restored.public_record(1)["note"] == "my draft"
    with pytest.raises(ValueError, match="已经提问"):
        restored.update(1, {"action": "ask", "revision": 2, "question": "Again?"})
    with pytest.raises(ValueError, match="页面已变化"):
        workspace.update(1, payload(workspace, revision=0))


@pytest.mark.parametrize("judgment", ["no_match", "insufficient"])
def test_uncertain_or_empty_candidate_verdict_is_not_forced_into_a_wrong_answer(
    workspace, judgment
):
    with pytest.raises(ValueError, match="说明"):
        workspace.update(1, payload(workspace, judgment=judgment, choices={}))
    workspace.update(
        1,
        payload(
            workspace, judgment=judgment, choices={}, note="Cannot identify the part"
        ),
    )
    record = workspace.record(1)
    assert record["matches_registered_answer"] is None
    assert workspace.summary()["groups"]["single"][judgment] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"choices": {"99": 1}},
        {"choices": {"0": True}},
        {"choices": {"0": -1}},
        {"choices": {"0": 1.5}},
        {"fields": {"hidden": "x"}},
        {"judgment": "other"},
        {"judgment": {}},
        {"judgment": "insufficient", "note": "x"},
        {"choices": {}},
    ],
)
def test_invalid_choices_do_not_create_a_submission(workspace, changes):
    with pytest.raises(ValueError):
        workspace.update(1, payload(workspace, **changes))
    assert workspace.record(1)["submitted_at"] is None


def test_quantities_extra_entities_and_multiple_required_results_use_shared_checker(
    workspace,
):
    case = workspace.case(2)
    ids = [e["entity_id"] for e in case.public_site["entities"]]
    case.task["goal"]["acceptable_states"] = [
        {"selected_entity_ids": [ids[0], ids[0], ids[1]], "fields": {}}
    ]
    workspace.update(2, payload(workspace, 2, choices={"0": 2, "1": 1}))
    assert workspace.record(2)["matches_registered_answer"] is True
    assert workspace.summary()["groups"]["composite"]["matched"] == 1


def test_participant_app_has_no_answer_bearing_admin_routes_or_arbitrary_file_access(
    workspace,
):
    app = create_answer_validation_app(workspace, preview=True)
    client = app.test_client()
    html = client.get("/task/1").get_data(as_text=True)
    for secret in [
        "最终目标",
        "rival",
        "is_final",
        "The target variant.",
        "matches_registered_answer",
        "canonical_answer",
    ]:
        assert secret not in html
    assert "clarify-target" not in html
    assert client.get("/review").status_code == 404
    assert client.get("/review/asset?path=.env").status_code == 404
    assert client.get("/task/1/asset/0").status_code == 200
    assert client.get("/task/1/frame/0/0").mimetype == "image/jpeg"
    assert client.get("/task/1/frame/999/0").status_code == 404
    assert client.get("/task/1/asset/999").status_code == 400
    assert client.post("/task/1", json=payload(workspace)).status_code == 403
    token = re.search('name="validation-session" content="([^"]+)"', html)[1]
    headers = {"X-Validation-Session": token}
    response = client.post("/task/1", json=payload(workspace), headers=headers)
    assert response.status_code == 200
    exported = client.get("/export").get_json()
    assert "matches_registered_answer" not in str(exported)
    assert "source" not in str(exported)
    assert exported["preview"] is True
    assert len(exported["records"]) == 1


def test_offline_choice_autosave_question_submit_resume_and_export(workspace):
    import threading
    from urllib.parse import urlsplit
    from werkzeug.serving import make_server

    sync_api = pytest.importorskip("playwright.sync_api")
    app = create_answer_validation_app(workspace, preview=True)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_api.sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1100, "height": 900}, accept_downloads=True
            )
            external = []

            def local_only(route):
                if urlsplit(route.request.url).hostname not in {
                    None,
                    "127.0.0.1",
                    "localhost",
                }:
                    external.append(route.request.url)
                    route.abort()
                else:
                    route.continue_()

            context.route("**/*", local_only)
            page = context.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(url + "/task/1")
            assert page.locator("#website").count() == 0
            assert "The target variant." not in page.locator("body").inner_text()
            page.locator(".frame-thumb").first.click()
            page.get_by_role("button", name="关闭", exact=True).click()
            page.locator('[data-option="1"] [type=checkbox]').check()
            page.locator("#note").fill("Independent choice")
            page.get_by_text("已保存到本机", exact=True).wait_for()
            page.reload()
            assert page.locator('[data-option="1"] [type=checkbox]').is_checked()
            assert page.locator("#note").input_value() == "Independent choice"
            page.get_by_text("需要确认指代？向佩戴者提问", exact=True).click()
            page.locator("#question").fill("Which one?")
            page.get_by_role("button", name="提问", exact=True).click()
            page.get_by_text(
                "Which one?\n佩戴者：The target variant.", exact=True
            ).wait_for()
            page.get_by_role("button", name="提交本题", exact=True).click()
            page.get_by_text("首次提交已保存", exact=True).wait_for()
            assert page.locator("#submit").is_disabled()
            assert workspace.record(1)["matches_registered_answer"] is True
            page.reload()
            assert page.locator("#submit").is_disabled()
            page.get_by_role("link", name="任务列表", exact=True).click()
            page.get_by_text("已提交 1 / 2 题", exact=False).wait_for()
            with page.expect_download() as download:
                page.get_by_role("link", name="导出本人作答记录").click()
            exported = json.loads(download.value.path().read_text())
            assert "matches_registered_answer" not in str(exported)
            assert "The target variant." in str(exported)
            assert not external and not errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_two_local_sessions_cannot_replace_the_first_submission(workspace):
    from concurrent.futures import ThreadPoolExecutor

    workspace.open(1)
    other = AnswerValidationWorkspace(
        workspace.root, workspace.participant, workspace.output
    )
    other.materials._cases = workspace.materials._cases

    def submit(owner):
        try:
            owner.update(1, payload(owner, revision=0))
            return "saved"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(submit, [workspace, other]))
    assert sorted(outcomes) == ["rejected", "saved"]
    assert workspace.record(1)["revision"] == 1


def test_different_answer_is_preserved_for_review_without_changing_task_truth(
    workspace,
):
    original = json.dumps(workspace.case(1).task, sort_keys=True)
    workspace.update(1, payload(workspace, choices={"0": 1}, note="I think this fits"))
    assert workspace.record(1)["matches_registered_answer"] is False
    assert workspace.summary()["groups"]["single"]["different"] == 1
    assert json.dumps(workspace.case(1).task, sort_keys=True) == original
