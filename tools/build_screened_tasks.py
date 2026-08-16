from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from egocua.construction import ConstructionError, compile_task_from_site
from egocua.environment import load_site_data
from egocua.lineage import build_source_identity, discover_frame_provenance
from egocua.schema import load_tasks, validate_dataset
from egocua.workspace import write_json_atomic


TRACKS = ("graft", "refresh", "clarify")


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: JSON root must be an object")
    return payload


def _selected_ids(selection: dict[str, Any]) -> list[str]:
    primary = selection.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("selection.primary must be an object")
    ids = [
        str(task_id)
        for track in TRACKS
        for task_id in primary.get(track, [])
    ]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("selection.primary task IDs must be non-empty and unique")
    return ids


def _source_cluster(record: dict[str, Any]) -> str:
    source = record.get("source", {})
    return str(source.get("original_video_id") or source.get("media_id") or "")


def _screened_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    records: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for manifest in sorted((root / "data/sources").glob("*screening_manifest.json")):
        payload = json.loads(manifest.read_text())
        for record in payload.get("tasks", []):
            if record.get("status") != "video_screened":
                continue
            task_id = str(record.get("task_id", ""))
            if not task_id or task_id in seen:
                raise ValueError(f"duplicate or missing screened task id: {task_id!r}")
            seen.add(task_id)
            records.append((manifest, record))
    return records


def build(root: str | Path = ".") -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    workspace = Path(root).resolve()
    selection = _read_object(workspace / "data/selection.json", "task selection")
    selected_ids = _selected_ids(selection)
    allowed_manifests = set(selection.get("source_manifests", []))
    screened = _screened_records(workspace)
    records_by_id = {
        str(record["task_id"]): (manifest, record)
        for manifest, record in screened
    }
    missing = sorted(set(selected_ids) - set(records_by_id))
    if missing:
        raise ValueError(f"selected source records are missing: {missing}")
    site_cache: dict[str, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    selected_clusters: dict[str, str] = {}

    for task_id in selected_ids:
        manifest, record = records_by_id[task_id]
        task_id = str(record["task_id"])
        dossier_reference = manifest.relative_to(workspace).as_posix()
        if dossier_reference not in allowed_manifests:
            raise ValueError(
                f"{task_id}: selected source manifest is not registered by selection.json"
            )
        cluster = _source_cluster(record)
        if not cluster:
            raise ValueError(f"{task_id}: source cluster is missing")
        if cluster in selected_clusters:
            raise ValueError(
                f"source cluster {cluster!r} is shared by "
                f"{selected_clusters[cluster]} and {task_id}"
            )
        provenance_reference, _ = discover_frame_provenance(
            dossier_reference, root=workspace
        )
        source_identity = build_source_identity(
            record,
            dossier_reference=dossier_reference,
            provenance_reference=provenance_reference,
            root=workspace,
        )
        blueprint = record.get("construct", {}).get("task_blueprint", {})
        site = str(blueprint.get("site") or "egomart")
        if site not in site_cache:
            site_cache[site] = load_site_data(workspace, site)
        site_data = site_cache[site]
        try:
            task = compile_task_from_site(
                record,
                source_identity=source_identity,
                site_data=site_data,
                root=workspace,
            )
        except ConstructionError as exc:
            raise ValueError(f"{task_id}: selected task cannot be constructed: {exc}") from exc
        tasks.append(task)
        selected_clusters[cluster] = task_id

    tasks.sort(key=lambda task: (TRACKS.index(task["track"]), task["id"]))
    validate_dataset(tasks, root=workspace, require_assets=True)
    clusters: dict[str, str] = {}
    record_by_id = {task_id: records_by_id[task_id][1] for task_id in selected_ids}
    for task in tasks:
        cluster = _source_cluster(record_by_id[task["id"]])
        if not cluster:
            raise ValueError(f"{task['id']}: source cluster is missing")
        if cluster in clusters:
            raise ValueError(
                f"source cluster {cluster!r} is shared by {clusters[cluster]} and {task['id']}"
            )
        clusters[cluster] = task["id"]

    report = {
        "schema_version": 1,
        "screened_records": len(screened),
        "selected_records": len(selected_ids),
        "constructed_tasks": len(tasks),
        "rescreening_candidates": len(screened) - len(selected_ids),
        "source_clusters": len(clusters),
        "by_track": dict(sorted(Counter(task["track"] for task in tasks).items())),
        "by_site": dict(sorted(Counter(task["site"] for task in tasks).items())),
    }
    return tasks, selection, report


def write_build(root: str | Path = ".") -> dict[str, Any]:
    workspace = Path(root).resolve()
    tasks, selection, report = build(workspace)
    task_root = workspace / "data/tasks"
    expected: set[Path] = set()
    for task in tasks:
        path = task_root / task["track"] / f"{task['id']}.json"
        write_json_atomic(path, task)
        expected.add(path)
    for stale in task_root.glob("*/*.json"):
        if stale not in expected:
            stale.unlink()
    write_json_atomic(workspace / "artifacts/reports/task-build.json", report)
    return report


def check_build(root: str | Path = ".") -> dict[str, Any]:
    workspace = Path(root).resolve()
    tasks, _, report = build(workspace)
    current_tasks = load_tasks(workspace / "data/tasks")
    expected_by_id = {task["id"]: task for task in tasks}
    current_by_id = {task["id"]: task for task in current_tasks}
    if expected_by_id != current_by_id:
        raise ValueError("data/tasks is stale; run tools.build_screened_tasks")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build every source-screened task that has a frozen site mapping."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = check_build(args.root) if args.check else write_build(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
