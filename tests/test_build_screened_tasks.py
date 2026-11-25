from __future__ import annotations

import json
from pathlib import Path

from egocua.construction import build_selected
from egocua.selection import task_ids
import pytest
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
@pytest.mark.requires_local_data("reference/sources/EgoSchema/repo/uid_to_ego4d.json")
def test_builder_constructs_only_the_selection_owned_task_pool() -> None:
    tasks, selection, report = build_selected(ROOT)
    selected = set(task_ids(selection))

    assert {task["id"] for task in tasks} == selected
    assert report["selected_records"] == len(selected)
    assert report["constructed_tasks"] == len(selected)
    assert report["screened_records"] >= report["selected_records"]
    assert report["rescreening_candidates"] == (
        report["screened_records"] - report["selected_records"]
    )


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
@pytest.mark.requires_local_data("reference/sources/EgoSchema/repo/uid_to_ego4d.json")
def test_selection_is_the_only_owner_of_active_task_ids() -> None:
    selection = json.loads((ROOT / "data/selection.json").read_text())
    _, returned_selection, _ = build_selected(ROOT)

    assert returned_selection == selection
