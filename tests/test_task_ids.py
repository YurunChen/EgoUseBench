from __future__ import annotations

from pathlib import Path

import pytest

from egocua.schema import (
    TASK_ID_PATTERN,
    canonical_task_id,
    load_tasks,
    validate_executable_task_id,
)
from egocua.selection import load_selection, task_ids
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
def test_executable_task_ids_match_origin_site_track_subtype_serial() -> None:
    tasks = load_tasks(ROOT / "data/tasks")
    assert tasks
    for task in tasks:
        expected = canonical_task_id(task)
        assert task["id"] == expected
        assert TASK_ID_PATTERN.fullmatch(task["id"])
        path = ROOT / "data/tasks" / task["track"] / f"{task['id']}.json"
        assert path.is_file()
    ids = [task["id"] for task in tasks]
    assert len(ids) == len(set(ids))


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
def test_review_queue_uses_canonical_task_ids() -> None:
    selection = load_selection(ROOT)
    queue = task_ids(selection)
    assert queue
    assert all(TASK_ID_PATTERN.fullmatch(task_id) for task_id in queue)


def test_validate_executable_task_id_skips_test_fixtures() -> None:
    fixture = {"id": "test-graft", "track": "graft", "site": "egomart"}
    validate_executable_task_id(fixture)


def test_executable_task_id_allows_serials_above_99() -> None:
    task = {
        "id": "candidate-media-graft-c-100",
        "track": "graft",
        "site": "egomedia",
        "graft": {"transition": "refine"},
    }

    validate_executable_task_id(task)

    assert canonical_task_id(task) == task["id"]
    assert TASK_ID_PATTERN.fullmatch(task["id"])
