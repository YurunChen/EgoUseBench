"""Screening manifests and task → source resolution, in one place.

A screening manifest is a staged manifest under ``data/sources/manifests/`` or
an explicitly registered ``selection.source_manifests`` path, with a ``tasks``
list of records. A compiled task points back at its record through
``task.source.dossier`` (manifest path) and ``task.source.task_id``. Every tool
that needs to walk manifests or find a task's record goes through this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Mapping

from .workspace import read_json_strict, resolve_within

MANIFEST_GLOB = "*screening_manifest.json"
SOURCES_DIR = "data/sources/manifests"


class SourceError(ValueError):
    """Raised when a source record or manifest is invalid or missing."""


def iter_manifests(root: str | Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield (manifest_path, payload) for every screening manifest, sorted by path."""

    workspace = Path(root).resolve()
    paths = set((workspace / SOURCES_DIR).glob(MANIFEST_GLOB))
    selection_path = workspace / "data/selection.json"
    if selection_path.is_file():
        selection = read_json_strict(selection_path)
        paths.update(
            resolve_within(workspace, reference, context="registered source manifest")
            for reference in selection["source_manifests"]
        )
    for manifest_path in sorted(paths):
        payload = read_json_strict(manifest_path)
        if not isinstance(payload, dict):
            raise SourceError(f"{manifest_path}: manifest must be a JSON object")
        records = payload.get("tasks")
        if not isinstance(records, list):
            raise SourceError(f"{manifest_path}: tasks must be a list")
        if any(not isinstance(record, dict) for record in records):
            raise SourceError(f"{manifest_path}: task records must be objects")
        yield manifest_path, payload


def iter_records(
    root: str | Path,
    *,
    status: str | None = None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield (manifest_path, record) across all manifests, optionally filtered by status."""

    for manifest_path, payload in iter_manifests(root):
        for record in payload["tasks"]:
            if status is not None and record.get("status") != status:
                continue
            yield manifest_path, record


def find_record(
    dossier: Mapping[str, Any],
    task_id: str,
    *,
    context: str,
) -> dict[str, Any]:
    """Return the single record with ``task_id`` inside a loaded dossier."""

    records = dossier.get("tasks")
    if not isinstance(records, list):
        raise SourceError(f"{context}: dossier tasks must be a list")
    matches = [
        record
        for record in records
        if isinstance(record, dict) and record.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise SourceError(
            f"{context}: expected one source record for {task_id!r}, found {len(matches)}"
        )
    return matches[0]


def _source_refs(task: Mapping[str, Any]) -> Mapping[str, Any]:
    source = task.get("source")
    if not isinstance(source, Mapping):
        raise SourceError(f"{task.get('id')}: task has no source block")
    return source


def load_dossier(
    task: Mapping[str, Any], *, root: str | Path
) -> tuple[Path, dict[str, Any]]:
    """Resolve and load the manifest a task was compiled from."""

    source = _source_refs(task)
    dossier_path = resolve_within(
        root,
        str(source.get("dossier", "")),
        context=f"{task['id']} source dossier",
    )
    payload = read_json_strict(dossier_path)
    if not isinstance(payload, dict):
        raise SourceError(f"{task['id']}: source dossier must be a JSON object")
    return dossier_path, payload


def source_record(task: Mapping[str, Any], *, root: str | Path) -> dict[str, Any]:
    """The screening record a compiled task was built from; raises if not exactly one."""

    _, dossier = load_dossier(task, root=root)
    record_task_id = str(_source_refs(task).get("task_id") or task["id"])
    return find_record(dossier, record_task_id, context=f"{task['id']} source record")


def task_detail_entity_ids(
    task: Mapping[str, Any],
    *,
    root: str | Path,
    include_related: bool = False,
) -> list[str]:
    """Return decision entities, optionally including the related comparison ring."""

    from .schema import semantic_entity_ids

    record = source_record(task, root=root)
    construct = record.get("construct")
    resolution = (
        construct.get("catalog_resolution") if isinstance(construct, Mapping) else None
    )
    if not isinstance(resolution, Mapping):
        raise SourceError(f"{task['id']}: source record has no catalog_resolution")
    core = resolution.get("core_neighbor_entity_ids")
    if not isinstance(core, list) or any(
        not isinstance(entity_id, str) or not entity_id for entity_id in core
    ):
        raise SourceError(f"{task['id']}: source record has invalid core neighbors")
    related = resolution.get("related_entity_ids") if include_related else []
    if not isinstance(related, list) or any(
        not isinstance(entity_id, str) or not entity_id for entity_id in related
    ):
        raise SourceError(f"{task['id']}: source record has invalid related entities")
    return list(dict.fromkeys([*semantic_entity_ids(task), *core, *related]))


def frame_provenance_path(task: Mapping[str, Any], *, root: str | Path) -> Path:
    source = _source_refs(task)
    return resolve_within(
        root,
        str(source.get("frame_provenance", "")),
        context=f"{task['id']} frame provenance",
    )


def screened_records(root: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    """Screened source records; formal membership belongs to selection.json."""

    candidates = list(iter_records(root, status="video_screened"))
    seen: set[str] = set()
    for _, record in candidates:
        task_id = str(record.get("task_id", ""))
        if not task_id or task_id in seen:
            raise SourceError(f"duplicate or missing screened task id: {task_id!r}")
        seen.add(task_id)
    return candidates
