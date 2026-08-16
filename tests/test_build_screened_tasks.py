from __future__ import annotations

import json
from pathlib import Path

from tools.build_screened_tasks import build


ROOT = Path(__file__).resolve().parents[1]


def test_builder_constructs_only_the_selection_owned_task_pool() -> None:
    tasks, selection, report = build(ROOT)
    selected = {
        task_id
        for task_ids in selection["primary"].values()
        for task_id in task_ids
    }

    assert {task["id"] for task in tasks} == selected
    assert report["selected_records"] == len(selected)
    assert report["constructed_tasks"] == len(selected)
    assert report["screened_records"] >= report["selected_records"]
    assert report["rescreening_candidates"] == (
        report["screened_records"] - report["selected_records"]
    )


def test_selection_is_the_only_owner_of_active_task_ids() -> None:
    selection = json.loads((ROOT / "data/selection.json").read_text())
    _, returned_selection, _ = build(ROOT)

    assert returned_selection == selection
