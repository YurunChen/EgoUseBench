from __future__ import annotations

import json
from pathlib import Path

import pytest

from egocua.review_web import create_review_app
from egocua.review_workspace import (
    AlreadySubmittedError,
    ReviewWorkspace,
    ReviewWorkspaceError,
    StaleReviewError,
)
from tests.helpers import materialize_assets, products, task


def _pending_workspace(root: Path, reviewer: str = "reviewer-a") -> tuple[dict, ReviewWorkspace]:
    benchmark_task = task("clarify")
    materialize_assets(benchmark_task, products("clarify"), root)
    directory = root / "data/reviews/submissions" / benchmark_task["id"]
    for path in directory.glob("*.json"):
        path.unlink()
    return benchmark_task, ReviewWorkspace.open(root, reviewer)


def _response(case: dict, *, failed: str | None = None) -> dict:
    return {
        "decision": "reject" if failed else "admit",
        "note": "The evidence is insufficient." if failed else "",
    }


def test_workspace_derives_one_case_and_personal_progress(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)

    progress = workspace.view()
    case = workspace.view(benchmark_task["id"])

    assert progress["total"] == 1
    assert progress["counts"]["todo"] == 1
    assert case["experiment"]["condition"] == "interactive"
    assert len(case["checks"]) == 3
    assert case["design_profile"] == {
        "initial_options": 2,
        "final_entity_count": 1,
        "required_field_count": 0,
        "minimum_key_actions": 5,
        "change": "确认前有两个合理候选，回答后只剩一个",
    }
    assert case["scenario_evidence"]["label"] == "由原始标注改写"
    assert case["motivation"]["task_risk"] == ""
    assert len(case["scenario_evidence"]["supported_claims"]) == 1
    assert case["own_status"] == "todo"


def test_workspace_refuses_a_paused_legacy_selection(tmp_path: Path) -> None:
    benchmark_task = task("clarify")
    materialize_assets(benchmark_task, products("clarify"), tmp_path)
    selection_path = tmp_path / "data/selection.json"
    selection = json.loads(selection_path.read_text())
    selection["status"] = "paused_for_glass_use_redesign"
    selection_path.write_text(json.dumps(selection))

    with pytest.raises(ReviewWorkspaceError, match="暂停人工审核"):
        ReviewWorkspace.open(tmp_path, "reviewer-a")


def test_open_workspace_stops_when_selection_is_later_paused(tmp_path: Path) -> None:
    _, workspace = _pending_workspace(tmp_path)
    selection_path = tmp_path / "data/selection.json"
    selection = json.loads(selection_path.read_text())
    selection["status"] = "paused_for_glass_use_redesign"
    selection_path.write_text(json.dumps(selection))

    with pytest.raises(ReviewWorkspaceError, match="暂停人工审核"):
        workspace.view()


def test_single_linked_candidate_is_shown_as_a_review_warning(tmp_path: Path) -> None:
    benchmark_task = task("refresh")
    initial = benchmark_task["refresh"]["initial_entity_id"]
    benchmark_task["refresh"]["policy"] = "retain"
    benchmark_task["goal"]["acceptable_states"][0]["selected_entity_ids"] = [initial]
    materialize_assets(benchmark_task, products("refresh"), tmp_path)

    case = ReviewWorkspace.open(tmp_path, "reviewer-a").view(benchmark_task["id"])

    assert len(case["entities"]) == 1
    assert "只关联了一个网页候选" in case["review_warnings"][0]


def test_missing_site_asset_blocks_only_its_case_not_the_review_queue(
    tmp_path: Path,
) -> None:
    benchmark_task, _ = _pending_workspace(tmp_path)
    site_records = products("clarify")
    referenced = set(benchmark_task["clarification"]["candidate_ids"])
    record = next(
        item for item in site_records
        if item.get("entity_id", item.get("item_id")) in referenced
    )
    missing = tmp_path / record["image_path"]
    missing.unlink()

    workspace = ReviewWorkspace.open(tmp_path, "reviewer-b")
    case = workspace.view(benchmark_task["id"])

    assert case["release_blockers"]
    assert any("发布材料不完整" in message for message in case["release_blockers"])


def test_submit_is_atomic_final_and_updates_only_personal_progress(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])

    receipt = workspace.submit(
        benchmark_task["id"], _response(case), expected_revision=case["revision"]
    )

    assert receipt["decision"] == "admit"
    assert workspace.view()["counts"]["submitted"] == 1
    other = ReviewWorkspace.open(tmp_path, "reviewer-b")
    assert other.view()["counts"]["todo"] == 1
    with pytest.raises(AlreadySubmittedError):
        workspace.submit(
            benchmark_task["id"], _response(case), expected_revision=case["revision"]
        )


def test_material_change_rejects_an_open_page_submission(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    spec_path = tmp_path / "data/reviews/construction.json"
    spec = json.loads(spec_path.read_text())
    spec["checks"]["common"][0]["question"] = "Changed after opening?"
    spec_path.write_text(json.dumps(spec))

    with pytest.raises(StaleReviewError):
        workspace.submit(
            benchmark_task["id"], _response(case), expected_revision=case["revision"]
        )


def test_local_web_flow_shows_task_and_ground_truth_for_review(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    source_dir = tmp_path / "reference/fixture"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"fixture-{benchmark_task['track']}.mp4").write_bytes(b"fixture video")
    app = create_review_app(workspace)
    client = app.test_client()

    index = client.get("/review")
    page = client.get(f"/review/task/{benchmark_task['id']}")

    assert index.status_code == 200
    assert "任务审核" in index.get_data(as_text=True)
    assert "平均评分" in page.get_data(as_text=True)
    assert "基准答案" in page.get_data(as_text=True)
    assert "给出整体结论" in page.get_data(as_text=True)
    assert "答案没问题，保留" in page.get_data(as_text=True)
    assert "打开冻结网站核对" in page.get_data(as_text=True)
    assert "结构难度" in page.get_data(as_text=True)
    assert "最少关键网页操作" in page.get_data(as_text=True)
    assert "现实场景依据" in page.get_data(as_text=True)
    assert "先判断它是不是眼镜任务" in page.get_data(as_text=True)
    assert "最关键的反事实检查" in page.get_data(as_text=True)
    assert "画面支持的关键事实" in page.get_data(as_text=True)
    assert "已定位到抽帧前后" in page.get_data(as_text=True)
    assert "先独立判断" not in page.get_data(as_text=True)
    assert client.get("/review/asset?path=../outside").status_code == 404


def test_current_selected_review_pages_have_complete_answer_entities() -> None:
    root = Path(__file__).resolve().parents[1]
    workspace = ReviewWorkspace.open(root, "workspace-integration-audit")
    app = create_review_app(workspace)
    client = app.test_client()

    progress = workspace.view()
    index = client.get("/review")
    index_html = index.get_data(as_text=True)

    assert index.status_code == 200
    assert progress["total"] == 11
    for task_id in workspace.ordered_ids:
        case = workspace.view(task_id)
        page = client.get(f"/review/task/{task_id}")

        assert task_id in index_html
        assert page.status_code == 200
        assert not case["release_blockers"]
        assert len(case["entities"]) >= 2
        assert all(entity["visual_path"] for entity in case["entities"])

    rejected_ids = {
        "candidate-clarify-position-ego2web-02",
        "candidate-clarify-position-ego2web-04",
        "graft-c-02",
        "graft-c-03",
    }
    assert all(task_id not in index_html for task_id in rejected_ids)


def test_current_refresh_retain_page_shows_initial_and_later_sweaters() -> None:
    root = Path(__file__).resolve().parents[1]
    workspace = ReviewWorkspace.open(root, "workspace-refresh-audit")
    task_id = "seed-refresh-retain-egoschema-02"
    case = workspace.view(task_id)
    page = create_review_app(workspace).test_client().get(f"/review/task/{task_id}")
    html = page.get_data(as_text=True)

    initial = next(
        entity for entity in case["entities"] if "V0目标" in entity["roles"]
    )
    later = next(
        entity for entity in case["entities"]
        if "后来出现的对象" in entity["roles"]
    )

    assert page.status_code == 200
    assert initial["entity_id"] == "p_85d682215a07"
    assert later["entity_id"] == "p_d976df782179"
    assert initial["name"] in html
    assert later["name"] in html
    assert "V0目标" in html
    assert "后来出现的对象" in html


def test_web_submission_returns_to_progress(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    app = create_review_app(workspace)
    client = app.test_client()
    data = {
        "revision": case["revision"],
        "decision": "admit",
        "note": "",
    }

    response = client.post(
        f"/review/task/{benchmark_task['id']}/submit",
        data=data,
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "已保留" in response.get_data(as_text=True)


def test_reject_requires_one_problem_note(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    response = {"decision": "reject", "note": ""}

    with pytest.raises(ValueError, match="发现问题时请简要说明"):
        workspace.submit(
            benchmark_task["id"], response, expected_revision=case["revision"]
        )
