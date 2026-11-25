"""Every released frame belongs to a screening record, and every record to a manifest.

data/observations grows one directory per screened clip and never shrinks on its own,
so when a task is retired its frames are what get left behind. A stray directory is
not harmless: it is indistinguishable from a live task's evidence when auditing what
the benchmark actually released.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from egocua.schema import load_tasks
from egocua.selection import load_selection, task_ids
from egocua.sources import iter_records, screened_records

ROOT = Path(__file__).resolve().parents[1]


def _record_observation_directories() -> set[str]:
    directories: set[str] = set()
    for _, record in iter_records(ROOT):
        for packet in record.get("packets", []):
            for frame in packet.get("frame_paths", []):
                parts = Path(str(frame)).parts
                if parts[:2] == ("data", "observations") and len(parts) > 2:
                    directories.add(parts[2])
    return directories


@pytest.mark.requires_local_data("data/observations")
def test_every_observation_directory_belongs_to_a_screening_record() -> None:
    directories = {
        path.name for path in (ROOT / "data/observations").iterdir() if path.is_dir()
    }
    assert directories, "no observation directories found"
    assert not directories - _record_observation_directories()


@pytest.mark.requires_local_data("data/observations")
def test_every_live_task_has_its_frames() -> None:
    for task in load_tasks(ROOT / "data/tasks"):
        for packet in task["visual_packets"]:
            for frame in packet["frame_paths"]:
                assert (ROOT / frame).is_file(), f"{task['id']}: missing {frame}"


def test_selected_tasks_compile_from_screened_sources() -> None:
    live = {task["id"] for task in load_tasks(ROOT / "data/tasks")}
    screened = {str(record.get("task_id", "")) for _, record in screened_records(ROOT)}
    selected = set(task_ids(load_selection(ROOT)))
    assert live == selected
    assert selected <= screened


def test_a_record_awaiting_its_curve_says_so() -> None:
    for _, record in iter_records(ROOT, status="needs_curve"):
        task_id = str(record.get("task_id", ""))
        assert "decision" not in (record.get("construct") or {}), task_id
        assert record.get("curve_note"), task_id


def test_selection_registers_every_manifest_it_draws_from() -> None:
    selection = json.loads((ROOT / "data/selection.json").read_text())
    registered = set(selection.get("source_manifests", []))
    queued = {task_id for pool in selection["task_set"].values() for task_id in pool}
    for manifest_path, record in iter_records(ROOT, status="video_screened"):
        if str(record.get("task_id")) in queued:
            relative = manifest_path.relative_to(ROOT).as_posix()
            assert relative in registered, f"{relative} is not in source_manifests"
