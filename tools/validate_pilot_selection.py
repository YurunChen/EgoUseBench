from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from egocua.workspace import resolve_within


TRACKS = {"graft", "refresh", "clarify"}
SITES = {"egomart", "egomedia", "egomaps"}


class SelectionError(ValueError):
    """Raised when the pilot selection is inconsistent with its source manifests."""


def _read_json(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SelectionError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SelectionError(f"{context}: top level must be an object")
    return payload


def _task_index(
    selection: Mapping[str, Any], workspace: Path
) -> dict[str, dict[str, Any]]:
    manifests = selection.get("source_manifests")
    if not isinstance(manifests, list) or not manifests:
        raise SelectionError("source_manifests must be a non-empty list")

    tasks: dict[str, dict[str, Any]] = {}
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
            tasks[task_id] = record
    return tasks


def _primary_ids(selection: Mapping[str, Any]) -> list[str]:
    primary = selection.get("primary")
    if not isinstance(primary, dict) or set(primary) != TRACKS:
        raise SelectionError("primary track names are invalid")

    selected: list[str] = []
    for family in sorted(TRACKS):
        task_ids = primary.get(family)
        if not isinstance(task_ids, list) or not all(
            isinstance(task_id, str) and task_id for task_id in task_ids
        ):
            raise SelectionError(f"{family}: task IDs must be strings")
        if not task_ids:
            raise SelectionError(f"{family}: must contain at least one task")
        selected.extend(task_ids)
    if len(selected) != len(set(selected)):
        raise SelectionError("a primary task is selected more than once")
    return selected


def _source_cluster(task: Mapping[str, Any]) -> str:
    source = task.get("source")
    if not isinstance(source, dict):
        raise SelectionError(f"{task.get('task_id')}: source must be an object")
    cluster = source.get("original_video_id") or source.get("media_id")
    if not isinstance(cluster, str) or not cluster:
        raise SelectionError(f"{task.get('task_id')}: source cluster is missing")
    return cluster


def validate_selection(
    selection_path: str | Path, *, root: str | Path = "."
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    try:
        path = resolve_within(workspace, selection_path, context="pilot selection")
    except (ValueError, FileNotFoundError) as exc:
        raise SelectionError(str(exc)) from exc
    selection = _read_json(path, str(selection_path))
    if selection.get("schema_version") != 1:
        raise SelectionError("selection schema_version must be 1")

    tasks = _task_index(selection, workspace)
    selected = _primary_ids(selection)
    if selection.get("target_size") != len(selected):
        raise SelectionError("target_size does not match the primary task count")

    clusters: dict[str, str] = {}
    site_counts = {site: 0 for site in SITES}
    track_counts = {track: 0 for track in TRACKS}
    primary = selection["primary"]
    for family, task_ids in primary.items():
        for task_id in task_ids:
            task = tasks.get(task_id)
            if task is None:
                raise SelectionError(f"unknown primary task: {task_id}")
            if task.get("status") != "video_screened":
                raise SelectionError(f"{task_id}: primary task is not video_screened")
            if task.get("family") != family:
                raise SelectionError(
                    f"{task_id}: selected under {family} but source says "
                    f"{task.get('family')}"
                )
            cluster = _source_cluster(task)
            if cluster in clusters:
                raise SelectionError(
                    f"primary source cluster {cluster!r} is shared by "
                    f"{clusters[cluster]} and {task_id}"
                )
            clusters[cluster] = task_id

            task_path = workspace / "data" / "tasks" / family / f"{task_id}.json"
            executable = _read_json(task_path, task_id)
            if executable.get("track") != family:
                raise SelectionError(f"{task_id}: executable track does not match selection")
            site = executable.get("site")
            if site not in SITES:
                raise SelectionError(f"{task_id}: unsupported task site {site!r}")
            site_counts[site] += 1
            track_counts[family] += 1

    return {
        "status": "valid",
        "primary_tasks": len(selected),
        "source_clusters": len(clusters),
        "track_counts": dict(sorted(track_counts.items())),
        "site_counts": site_counts,
    }


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
