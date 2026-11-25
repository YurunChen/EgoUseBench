import json
from pathlib import Path

import pytest

from egocua.selection import (
    SelectionError,
    load_selection,
    task_ids,
    validate_selection,
)


def test_selection_rejects_duplicate_keys_instead_of_replacing_members(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data/selection.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"task_set": {"graft": ["selected"]}, "task_set": {}}')

    with pytest.raises(SelectionError, match="duplicate JSON key 'task_set'"):
        load_selection(tmp_path)


def test_selection_rejects_task_file_with_another_id(tmp_path: Path) -> None:
    manifest = tmp_path / "data/sources/manifests/sample.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "selected",
                        "status": "video_screened",
                        "family": "graft",
                        "source": {"original_video_id": "recording"},
                    }
                ]
            }
        )
    )
    selection_path = tmp_path / "data/selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "task_set": {"graft": ["selected"], "refresh": [], "clarify": []},
                "source_manifests": [manifest.relative_to(tmp_path).as_posix()],
            }
        )
    )
    task_path = tmp_path / "data/tasks/graft/selected.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(
        json.dumps({"id": "another", "track": "graft", "site": "egomart"})
    )

    with pytest.raises(SelectionError, match="executable ID does not match selection"):
        validate_selection("data/selection.json", root=tmp_path)


def test_task_set_is_the_review_queue() -> None:
    selection = {
        "task_set": {
            "graft": [],
            "refresh": [],
            "clarify": ["seed-clarify"],
        }
    }

    assert task_ids(selection) == ["seed-clarify"]
    assert task_ids(selection) == ["seed-clarify"]


def test_review_queue_preserves_track_order() -> None:
    selection = {
        "task_set": {
            "graft": ["candidate-mart-graft-p-06", "candidate-mart-graft-c-03"],
            "refresh": ["seed-mart-refresh-replace-01"],
            "clarify": ["seed-mart-clarify-sequence-01"],
        },
    }

    assert task_ids(selection) == [
        "candidate-mart-graft-p-06",
        "candidate-mart-graft-c-03",
        "seed-mart-refresh-replace-01",
        "seed-mart-clarify-sequence-01",
    ]


def test_review_queue_requires_all_three_tracks() -> None:
    selection = {
        "task_set": {"graft": ["candidate-mart-graft-c-03"]},
    }

    with pytest.raises(SelectionError, match="track names are invalid"):
        task_ids(selection)


def test_old_release_set_name_is_rejected() -> None:
    selection: dict[str, object] = {
        "release_set": {"graft": [], "refresh": [], "clarify": []},
    }

    with pytest.raises(SelectionError, match="task_set"):
        task_ids(selection)
