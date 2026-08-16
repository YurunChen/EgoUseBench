import json
from collections import Counter
from pathlib import Path

from egocua.schema import load_tasks
from tools.build_task_preaudit import build_preaudit


ROOT = Path(__file__).resolve().parents[1]


def test_task_preaudit_covers_current_selection() -> None:
    selection = json.loads((ROOT / "data/selection.json").read_text())
    selected_ids = [
        task_id
        for task_ids in selection["primary"].values()
        for task_id in task_ids
    ]
    compiled = {
        task["id"]: task for task in load_tasks(ROOT / "data/tasks")
        if task["id"] in selected_ids
    }
    report = build_preaudit(ROOT)

    assert report["summary"]["tasks"] == len(selected_ids)
    assert report["summary"]["decisions"] == {
        "primary_preaudit_pass": len(selected_ids),
    }
    assert report["summary"]["tracks"] == {
        track: len(task_ids)
        for track, task_ids in selection["primary"].items()
        if task_ids
    }
    assert report["summary"]["sites"] == dict(sorted(Counter(
        compiled[task_id]["site"] for task_id in selected_ids
    ).items()))
    assert all(row["structural_pass"] for row in report["tasks"])
    assert all(row["site_realism_ready"] for row in report["tasks"])
    assert all(row["independent_human_review_pending"] for row in report["tasks"])
    assert {row["task_id"] for row in report["tasks"]} == set(selected_ids)
    assert all(
        row["checks"]["supported_claim_frames_resolve"]
        and row["checks"]["scenario_origin_valid"]
        and row["checks"]["natural_user_instruction"]
        for row in report["tasks"]
    )
