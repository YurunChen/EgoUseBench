"""The benchmark task membership and its source-manifest invariants."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .environment import SITES
from .lineage import validate_independent_recordings
from .schema import TRACKS, TRACK_ORDER
from .workspace import read_json_strict, resolve_within

SELECTION_PATH = "data/selection.json"


class SelectionError(ValueError):
    """Raised when the task selection cannot be used as a queue."""


def load_selection(root: str | Path, path: str = SELECTION_PATH) -> dict[str, Any]:
    """Read the selection object from the workspace."""

    workspace = Path(root).resolve()
    try:
        selection_path = resolve_within(workspace, path, context="task selection")
    except (ValueError, FileNotFoundError) as exc:
        raise SelectionError(str(exc)) from exc
    return _read_json(selection_path, path)


def task_ids(selection: Mapping[str, Any]) -> list[str]:
    """Return every selected task ID, graft then refresh then clarify."""

    task_set = selection.get("task_set")
    if not isinstance(task_set, dict):
        raise SelectionError("task_set track names are invalid")
    return _track_ids(task_set, label="task_set", complete=True)


def validate_selection(
    selection_path: str | Path, *, root: str | Path = "."
) -> dict[str, Any]:
    """Check that the selection is internally consistent with its source manifests."""

    workspace = Path(root).resolve()
    try:
        path = resolve_within(workspace, selection_path, context="pilot selection")
    except (ValueError, FileNotFoundError) as exc:
        raise SelectionError(str(exc)) from exc
    selection = _read_json(path, str(selection_path))
    if selection.get("schema_version") != 3:
        raise SelectionError("selection schema_version must be 3")
    allowed_keys = {
        "schema_version",
        "task_set",
        "source_manifests",
        "replacement_history",
    }
    unknown_keys = set(selection) - allowed_keys
    if unknown_keys:
        raise SelectionError(
            f"selection contains derived or unknown fields: {sorted(unknown_keys)}"
        )

    tasks = _task_index(selection, workspace)
    selected = task_ids(selection)

    track_counts, site_counts = _validate_block(
        selection["task_set"],
        tasks,
        workspace,
        label="task_set",
        require_files=True,
        complete=True,
    )
    try:
        recording_summary = validate_independent_recordings(
            [tasks[task_id] for task_id in selected],
            root=workspace,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SelectionError(str(exc)) from exc
    return {
        "status": "valid",
        "tasks": len(selected),
        **recording_summary,
        "track_counts": dict(sorted(track_counts.items())),
        "site_counts": site_counts,
    }


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = read_json_strict(path)
    except (OSError, ValueError) as exc:
        raise SelectionError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelectionError(f"{context}: top level must be an object")
    return payload


def _track_ids(
    block: Mapping[str, Any],
    *,
    label: str,
    complete: bool = False,
) -> list[str]:
    if not isinstance(block, dict):
        raise SelectionError(f"{label} track names are invalid")
    unknown = set(block) - TRACKS
    if unknown or (complete and set(block) != TRACKS):
        raise SelectionError(f"{label} track names are invalid")

    selected: list[str] = []
    for family in TRACK_ORDER:
        if family not in block:
            continue
        task_ids = block.get(family)
        if not isinstance(task_ids, list) or not all(
            isinstance(task_id, str) and task_id for task_id in task_ids
        ):
            raise SelectionError(f"{label}/{family}: task IDs must be strings")
        selected.extend(task_ids)
    # An empty pool is a legitimate state between corpus archival and the next
    # synthesis round; downstream consumers decide whether emptiness is an error.
    if len(selected) != len(set(selected)):
        raise SelectionError(f"a {label} task is selected more than once")
    return selected


def _task_index(
    selection: Mapping[str, Any], workspace: Path
) -> dict[str, tuple[Path, dict[str, Any]]]:
    manifests = selection.get("source_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise SelectionError("source_manifests must be a non-empty list")

    tasks: dict[str, tuple[Path, dict[str, Any]]] = {}
    for manifest_name in manifests:
        if not isinstance(manifest_name, str):
            raise SelectionError("source_manifests entries must be paths")
        try:
            manifest_path = resolve_within(
                workspace, manifest_name, context="selection source manifest"
            )
        except (ValueError, FileNotFoundError) as exc:
            raise SelectionError(str(exc)) from exc
        manifest = _read_json(manifest_path, manifest_name)
        records = manifest.get("tasks")
        if not isinstance(records, list):
            raise SelectionError(f"{manifest_name}: tasks must be a list")
        for record in records:
            if not isinstance(record, dict):
                raise SelectionError(f"{manifest_name}: task records must be objects")
            task_id = record.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise SelectionError(f"{manifest_name}: task_id must be non-empty")
            if task_id in tasks:
                raise SelectionError(f"duplicate task_id across manifests: {task_id}")
            tasks[task_id] = (manifest_path, record)
    return tasks


def _validate_block(
    block: Mapping[str, Any],
    tasks: Mapping[str, tuple[Path, dict[str, Any]]],
    workspace: Path,
    *,
    label: str,
    require_files: bool,
    complete: bool = True,
) -> tuple[dict[str, int], dict[str, int]]:
    _track_ids(block, label=label, complete=complete)
    site_counts = {site: 0 for site in SITES}
    track_counts = {track: 0 for track in TRACKS}
    for family, task_ids in block.items():
        for task_id in task_ids:
            entry = tasks.get(task_id)
            if entry is None:
                raise SelectionError(f"unknown {label} task: {task_id}")
            _, task = entry
            if task.get("status") != "video_screened":
                raise SelectionError(f"{task_id}: {label} task is not video_screened")
            if task.get("family") != family:
                raise SelectionError(
                    f"{task_id}: selected under {family} but source says "
                    f"{task.get('family')}"
                )
            if not require_files:
                continue
            task_path = workspace / "data" / "tasks" / family / f"{task_id}.json"
            executable = _read_json(task_path, task_id)
            if executable.get("id") != task_id:
                raise SelectionError(
                    f"{task_id}: executable ID does not match selection"
                )
            if executable.get("track") != family:
                raise SelectionError(
                    f"{task_id}: executable track does not match selection"
                )
            site = executable.get("site")
            if site not in SITES:
                raise SelectionError(f"{task_id}: unsupported task site {site!r}")
            site_counts[site] += 1
            track_counts[family] += 1
    return track_counts, site_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the active evidence-screened task selection."
    )
    parser.add_argument("selection")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(validate_selection(args.selection, root=args.root), indent=2))


if __name__ == "__main__":
    main()
