from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from queue import Queue
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from PIL import Image

from egocua.adapters.human import run_task
from egocua.checker import check_run
from egocua.run_ledger import load_run_record
from tests.conditions import CONDITIONS
from tests.helpers import item_id, materialize_site, products, task


def get_json(url):
    return json.load(urlopen(url, timeout=10))


def post(url, data, token):
    return json.load(
        urlopen(
            Request(
                url + "command",
                json.dumps(data).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Human-Session": token,
                },
            ),
            timeout=10,
        )
    )


def command(url, token, action, *, question=""):
    current = get_json(url + "state")
    return post(
        url,
        {"action": action, "revision": current["revision"], "question": question},
        token,
    )


def website_post(site, path, form):
    urlopen(Request(site + path, urlencode(form).encode()), timeout=10).read()


@contextmanager
def human_run(root, *, track="graft", max_steps=20, session_label="", mechanisms="RG"):
    definition = task(track)
    if max_steps == 30:
        from tests.test_composite_execution import task_definition

        definition = task_definition(mechanisms)
    definition["visual_session"] = {
        "update_onset_seconds": 3,
        "frequency_samples": {
            "window_seconds": [0, 6],
            "timestamps_seconds": list(range(7)),
            "frame_paths": [f"frames/{i}.jpg" for i in range(7)],
        },
    }
    if max_steps == 30:
        if "R" in mechanisms:
            definition["visual_session"]["refresh_onset_seconds"] = 2
        if "G" not in mechanisms:
            definition["visual_session"].pop("update_onset_seconds")
    materialize_site(products(track), root)
    (root / "frames").mkdir(exist_ok=True)
    for index in range(7):
        Image.new("RGB", (64, 48), (index * 25, 120, 90)).save(
            root / f"frames/{index}.jpg"
        )
    condition = replace(
        CONDITIONS["online"],
        frame_interval_seconds=1,
        frame_delivery_steps=1,
        batch_seconds=1,
        release_at=None,
        replay_timing="action_aligned",
        wearer_frames_per_image=4,
        visual_history="compressed",
    )
    urls = Queue()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            run_task,
            definition,
            condition,
            evaluator_id="human-a",
            source_max_steps=max_steps,
            default_release_step=2,
            reference_steps=None,
            asset_root=root,
            output_root=root / "experiments",
            experiment_id="human-test",
            session_label=session_label,
            message_writer=lambda message: None,
            browser_opener=urls.put,
        )
        result.add_done_callback(
            lambda done: urls.put(done.exception()) if done.exception() else None
        )
        url = urls.get(timeout=10)
        if isinstance(url, BaseException):
            raise url
        html = urlopen(url).read().decode()
        token = re.search(r'name="human-session" content="([^"]+)"', html)[1]
        site = re.search(r'data-site-url="([^"]+)"', html)[1].rstrip("/")
        try:
            yield url, token, site, result
        finally:
            if not result.done():
                try:
                    if get_json(url + "state")["status"] == "active":
                        command(url, token, "finish")
                except OSError:
                    pass
            result.result(timeout=10)


def test_complete_materials_are_available_without_updates_or_step_limits(tmp_path):
    with human_run(tmp_path) as (url, token, site, result):
        initial = get_json(url + "state")
        assert "Find A" in initial["evidence"] and "Also apply B" in initial["evidence"]
        assert "source t=6 s" in initial["evidence"]
        assert "budget" not in initial and "browser" not in initial
        assert get_json(url + "state")["revision"] == initial["revision"]
        # Normal native actions need no control-panel acknowledgments.
        for _ in range(35):
            urlopen(site + "/search?q=Product", timeout=10).read()
        website_post(site, "/cart/add", {"item_id": item_id("graft", 0)})
        progress = next((tmp_path / "experiments").rglob("progress.json"))
        checkpoint = json.loads(progress.read_text())
        assert len(checkpoint["actions"]) == 36
        assert checkpoint["live_state"]["selected_entity_ids"] == [item_id("graft", 0)]
        ended = command(url, token, "finish")
        record = result.result(timeout=10)
        assert ended["status"] == "finished"
        assert record["meta"]["max_steps"] is None
        assert (
            load_run_record(record["meta"]["artifact_directory"])["live_state"]
            == record["live_state"]
        )
        assert record["meta"]["transport"] == "egocua-human-solvability-v1"
        assert record["condition"]["schedule"] == "human_solvability"
        assert record["condition"]["visual_history"] == "full"
        assert record["trace"]["source_protocol"]["max_steps"] == 20
        assert record["trace"]["elapsed_seconds"] >= 0
        assert record["final_state"]["submission_completed"] is False
        assert check_run(record["task_snapshot"], record)["e2e_success"] is True
        assert not any(
            a["type"] in {"standby", "instruction_injection"} for a in record["actions"]
        )
        assert "task_snapshot" not in ended["record"]


def test_clarification_remains_question_triggered(tmp_path):
    with human_run(tmp_path, track="clarify") as (url, token, site, result):
        assert "The target variant." not in get_json(url + "state")["evidence"]
        answer = command(url, token, "ask", question="Which one?")
        assert "The target variant." in answer["evidence"]
        website_post(site, "/cart/add", {"item_id": item_id("clarify", 1)})
        command(url, token, "finish")
        record = result.result(timeout=10)
        assert check_run(record["task_snapshot"], record)["e2e_success"] is True
        assert record["trace"]["commands"][0]["answer_event"]["valid"] is True


@pytest.mark.parametrize("mechanisms", ["RG", "CG", "CR", "RCG"])
def test_composite_has_complete_materials_and_no_thirty_step_cap(tmp_path, mechanisms):
    with human_run(tmp_path, max_steps=30, mechanisms=mechanisms) as (
        url,
        token,
        site,
        result,
    ):
        evidence = get_json(url + "state")["evidence"]
        assert "source t=6 s" in evidence
        if "C" in mechanisms:
            command(url, token, "ask", question="Which one?")
        ended = command(url, token, "finish")
        record = result.result(timeout=10)
        assert ended["status"] == "finished"
        assert record["meta"]["max_steps"] is None
        assert record["trace"]["source_protocol"]["max_steps"] == 30
        assert check_run(record["task_snapshot"], record)["run_valid"] is True


def test_failed_final_state_is_kept_as_failure(tmp_path):
    with human_run(tmp_path) as (url, token, site, result):
        for index in (0, 1):
            website_post(site, "/cart/add", {"item_id": item_id("graft", index)})
        command(url, token, "finish")
        record = result.result(timeout=10)
        assert record["run_status"]["valid"] is True
        assert check_run(record["task_snapshot"], record)["e2e_success"] is False


def test_old_or_invalid_control_requests_do_not_repeat(tmp_path):
    with human_run(tmp_path) as (url, token, site, result):
        command(url, token, "ask", question="Which item?")
        for data, key, status in [
            ({"action": "finish", "revision": 0}, token, 409),
            ({"action": "finish", "revision": 1}, "wrong-session", 403),
            ({"action": "wait", "revision": 1}, token, 400),
            ({"action": "browser", "revision": 1}, token, 400),
            ({"action": "ask", "revision": 1, "question": ""}, token, 400),
        ]:
            with pytest.raises(HTTPError) as error:
                post(url, data, key)
            assert error.value.code == status
        assert get_json(url + "state")["revision"] == 1


def test_unlimited_budget_is_not_accepted_for_model_records():
    from egocua.execution import RunConfig

    with pytest.raises(ValueError, match="only human"):
        RunConfig(
            agent="browsergym",
            model="fixture",
            max_steps=None,
            default_release_step=2,
            reference_steps=None,
        )
