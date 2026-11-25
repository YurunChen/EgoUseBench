"""Render one audit page per task plus a catalog-level diagnostic index.

A site audit shows the catalog entities one task can actually use. Task artifacts
bind to that task-local execution surface; unrelated catalog or provenance changes
do not invalidate them. The index still records whole-catalog counts and hashes as
diagnostics, without using those hashes as per-task validity gates.

Usage:
    PYTHONPATH=. uv run python tools/render/render_all_site_audits.py --root .
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from egocua.environment import load_site_data
from egocua.run_ledger import (
    canonical_sha256,
    task_sha256,
    task_site_execution_sha256,
)
from egocua.schema import load_tasks
from egocua.workspace import resolve_within, write_json_atomic
from tools.render.render_site_audit import referenced_entity_ids, render_site_audit


def _render_ids(
    tasks: Sequence[Mapping[str, object]],
    *,
    requested_ids: set[str],
    requested_sites: set[str],
) -> set[str]:
    """Return exactly the requested incremental scope, or every task for a full run."""

    if not requested_ids and not requested_sites:
        return {str(task["id"]) for task in tasks}
    return requested_ids | {
        str(task["id"]) for task in tasks if str(task["site"]) in requested_sites
    }


def _exact_duplicate_groups(
    products: list[dict[str, object]],
    key: str,
) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for product in products:
        if key == "image_sha256":
            provenance = product.get("image_provenance", {})
            value = str(
                provenance.get("sha256", "") if isinstance(provenance, dict) else ""
            ).strip()
        else:
            value = str(product.get(key, "")).strip()
        if key == "name":
            value = " ".join(value.casefold().split())
        grouped[value].append(str(product["item_id"]))
    return {value: ids for value, ids in grouped.items() if value and len(ids) > 1}


def _egomaps_photo_duplicate_groups(
    entities: list[dict[str, object]],
    *,
    workspace: Path,
) -> dict[str, list[str]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for place in entities:
        place_id = str(place.get("entity_id") or place.get("place_id") or "")
        photo_path = str(place.get("photo_path") or "")
        if not place_id or not photo_path:
            continue
        try:
            photo = resolve_within(workspace, photo_path, context=f"{place_id} photo")
        except (ValueError, FileNotFoundError):
            continue
        if not photo.is_file():
            continue
        grouped[hashlib.sha256(photo.read_bytes()).hexdigest()].append(place_id)
    return {digest: ids for digest, ids in grouped.items() if len(ids) > 1}


def render_all(
    tasks_path: str,
    output_path: str,
    *,
    root: str = ".",
    task_ids: list[str] | None = None,
    sites: list[str] | None = None,
) -> dict[str, object]:
    workspace = Path(root).resolve()
    task_root = resolve_within(workspace, tasks_path, context="tasks", must_exist=False)
    if not task_root.is_dir():
        raise ValueError(f"tasks directory does not exist: {task_root}")
    output_root = resolve_within(
        workspace, output_path, context="site audits", must_exist=False
    )
    tasks = load_tasks(task_root)
    discovered_ids = {str(task["id"]) for task in tasks}
    known_sites = {str(task["site"]) for task in tasks}
    requested_ids = {str(task_id) for task_id in (task_ids or [])}
    requested_sites = {str(site) for site in (sites or [])}
    unknown_ids = sorted(requested_ids - discovered_ids)
    unknown_sites = sorted(requested_sites - known_sites)
    if unknown_ids:
        raise ValueError(f"unknown task IDs: {unknown_ids}")
    if unknown_sites:
        raise ValueError(f"unknown sites: {unknown_sites}")
    incremental = bool(requested_ids or requested_sites)
    site_data = {
        site: load_site_data(workspace, site)
        for site in sorted({task["site"] for task in tasks})
    }
    egomart_entities = site_data.get("egomart", {}).get("entities", [])
    egomaps_entities = site_data.get("egomaps", {}).get("entities", [])
    duplicate_images = _exact_duplicate_groups(egomart_entities, "image_sha256")
    duplicate_names = _exact_duplicate_groups(egomart_entities, "name")
    duplicate_map_photos = _egomaps_photo_duplicate_groups(
        egomaps_entities,
        workspace=workspace,
    )
    image_group_by_id = {
        item_id: ids for ids in duplicate_images.values() for item_id in ids
    }
    name_group_by_id = {
        item_id: ids for ids in duplicate_names.values() for item_id in ids
    }
    referenced_map_ids = {
        entity_id
        for task in tasks
        if task["site"] == "egomaps"
        for entity_id in referenced_entity_ids(task)
    }
    referenced_map_photo_groups = {
        digest: [place_id for place_id in ids if place_id in referenced_map_ids]
        for digest, ids in duplicate_map_photos.items()
    }
    referenced_map_photo_groups = {
        digest: ids
        for digest, ids in referenced_map_photo_groups.items()
        if len(ids) > 1
    }
    referenced_map_photo_group_by_id = {
        place_id: ids
        for ids in referenced_map_photo_groups.values()
        for place_id in ids
    }
    index_path = output_root / "index.json"
    artifacts: dict[str, dict[str, object]] = {}
    if incremental:
        if not index_path.is_file():
            raise ValueError(
                "incremental site-audit rendering requires an existing complete index"
            )
        previous = json.loads(index_path.read_text())
        previous_tasks = previous.get("tasks") if isinstance(previous, dict) else None
        if not isinstance(previous_tasks, dict):
            raise ValueError("existing site-audit index has no task artifacts")
        artifacts = {
            str(task_id): dict(artifact)
            for task_id, artifact in previous_tasks.items()
            if str(task_id) in discovered_ids and isinstance(artifact, dict)
        }
        for artifact in artifacts.values():
            artifact.pop("site_data_sha256", None)

    render_ids = _render_ids(
        tasks,
        requested_ids=requested_ids,
        requested_sites=requested_sites,
    )

    expected_images = {f"{task['id']}.jpg" for task in tasks}
    for task in tasks:
        if str(task["id"]) not in render_ids:
            continue
        task_file = task_root / task["track"] / f"{task['id']}.json"
        destination = output_root / f"{task['id']}.jpg"
        render_site_audit(
            task_file.relative_to(workspace),
            destination.relative_to(workspace),
            root=workspace,
        )
        referenced_ids = referenced_entity_ids(task)
        exact_duplicates = [
            {
                "item_id": item_id,
                "same_image_item_ids": image_group_by_id.get(item_id, []),
                "same_name_item_ids": name_group_by_id.get(item_id, []),
                "same_photo_place_ids": referenced_map_photo_group_by_id.get(
                    item_id, []
                ),
            }
            for item_id in referenced_ids
            if item_id in image_group_by_id
            or item_id in name_group_by_id
            or item_id in referenced_map_photo_group_by_id
        ]
        artifacts[task["id"]] = {
            "audit_path": destination.relative_to(workspace).as_posix(),
            "audit_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "task_sha256": task_sha256(task),
            "site": task["site"],
            "task_site_execution_sha256": task_site_execution_sha256(
                task, site_data[str(task["site"])]
            ),
            "referenced_entity_ids": referenced_ids,
            "exact_duplicate_references": exact_duplicates,
        }
    for stale_image in output_root.glob("*.jpg"):
        if stale_image.name not in expected_images:
            stale_image.unlink()
    missing_artifacts = sorted(discovered_ids - set(artifacts))
    if missing_artifacts:
        raise ValueError(
            f"site-audit index is missing unchanged tasks: {missing_artifacts}"
        )
    tasks_with_duplicates = sum(
        bool(artifact.get("exact_duplicate_references"))
        for artifact in artifacts.values()
    )
    maps_tasks_with_duplicates = sum(
        artifact.get("site") == "egomaps"
        and bool(artifact.get("exact_duplicate_references"))
        for artifact in artifacts.values()
    )
    write_json_atomic(
        index_path,
        {
            "schema_version": 4,
            "render_scope": {
                "mode": "incremental" if incremental else "full",
                "requested_task_ids": sorted(requested_ids),
                "requested_sites": sorted(requested_sites),
                "rendered_task_ids": sorted(render_ids),
                "preserved_task_count": len(tasks) - len(render_ids),
            },
            "sites": {
                site: {
                    "entities": len(data["entities"]),
                    "site_data_sha256": canonical_sha256(data),
                }
                for site, data in site_data.items()
            },
            "egomart_duplicate_audit": {
                "distinct_images": len(
                    {
                        str(entity.get("image_path"))
                        for entity in egomart_entities
                        if entity.get("image_path")
                    }
                ),
                "exact_duplicate_image_groups": len(duplicate_images),
                "listings_in_duplicate_image_groups": sum(
                    map(len, duplicate_images.values())
                ),
                "exact_duplicate_name_groups": len(duplicate_names),
                "listings_in_duplicate_name_groups": sum(
                    map(len, duplicate_names.values())
                ),
                "tasks_with_exact_duplicate_references": tasks_with_duplicates,
            },
            "egomaps_duplicate_audit": {
                "referenced_place_ids": sorted(referenced_map_ids),
                "exact_duplicate_photo_groups": len(referenced_map_photo_groups),
                "referenced_places_in_duplicate_photo_groups": sum(
                    map(len, referenced_map_photo_groups.values())
                ),
                "tasks_with_exact_duplicate_references": maps_tasks_with_duplicates,
            },
            "tasks": artifacts,
        },
    )
    return {
        "rendered": len(render_ids),
        "task_ids": sorted(render_ids),
        "preserved": len(tasks) - len(render_ids),
        "output": output_root.as_posix(),
        "index": index_path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render observation/site-product audits for every task."
    )
    parser.add_argument("--tasks", default="data/tasks")
    parser.add_argument("--output", default="artifacts/reports/site-audits")
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--site", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            render_all(
                args.tasks,
                args.output,
                root=args.root,
                task_ids=args.task_id,
                sites=args.site,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
