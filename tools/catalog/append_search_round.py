"""Append one real upstream search round to an existing task's result window.

Use this when a task's frozen queries already name the category its scene should
establish. The new capture keeps the reviewed rounds, rings and rejections; every
added entity starts in the ambient ring and is listed for Coding-Agent judgement.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from egocua.environment import load_site_data
from egocua.sources import iter_manifests
from egocua.workspace import write_json_atomic
from tools.catalog import capture_amazon_search_windows as amazon
from tools.catalog import capture_youtube_search_windows as youtube
from tools.catalog.build_task_search_neighborhoods import (
    SNAPSHOT_DIR,
    UPSTREAM_CAPTURES,
    _blueprint_final_options,
    _entity_id,
    _snapshot_rows,
)
from tools.catalog.materialize_upstream_search_windows import ENTITY_INDEX, materialize

Capture = Callable[[str, str, str], tuple[str, str, dict[str, Any]]]
CAPTURES: dict[str, tuple[Path, Capture]] = {
    "egomart": (amazon.OUTPUT_DIR, amazon._capture_with_reader),
    "egomedia": (youtube.OUTPUT_DIR, youtube._capture),
}
RING_FIELDS = ("core_neighbor_entity_ids", "related_entity_ids", "ambient_entity_ids")


def _source_record(root: Path, task_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    for path, payload in iter_manifests(root):
        for record in payload.get("tasks", []):
            if isinstance(record, dict) and record.get("task_id") == task_id:
                return path, payload, record
    raise ValueError(f"{task_id}: no source record")


def _upstream_capture(root: Path, site: str, task_id: str, stage: str, query: str) -> dict[str, Any]:
    directory, capture = CAPTURES[site]
    path = root / directory / f"{task_id}--{stage}.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("query") != query:
            raise ValueError(f"{path}: already holds a different query")
        return existing
    _, _, payload = capture(task_id, stage, query)
    write_json_atomic(path, payload)
    return payload


def append_round(root: Path, *, task_id: str, stage: str, query: str) -> dict[str, Any]:
    """Capture, materialize and append one round; return the entities needing judgement."""

    manifest_path, manifest, record = _source_record(root, task_id)
    site = str(record["construct"]["task_blueprint"]["site"])
    resolution = record["construct"]["catalog_resolution"]
    if any(row.get("stage") == stage for row in resolution["query_rounds"]):
        raise ValueError(f"{task_id}: stage {stage!r} is already declared")

    capture = _upstream_capture(root, site, task_id, stage, query)
    materialize(root, site=site, task_ids={task_id})
    upstream_key = UPSTREAM_CAPTURES[site][1]
    upstream_index = json.loads((root / ENTITY_INDEX).read_text())["sites"][site]
    observed: dict[str, dict[str, Any]] = {}
    for row in capture["results"]:
        observed.setdefault(str(upstream_index[str(row[upstream_key])]), row)
    result_ids = list(observed)

    by_id = {_entity_id(entity): entity for entity in load_site_data(root, site)["entities"]}
    snapshot_path = SNAPSHOT_DIR / f"{task_id}.json"
    if (root / snapshot_path).is_file():
        snapshot = json.loads((root / snapshot_path).read_text())
    else:
        snapshot = {
            "schema_version": 1,
            "task_id": task_id,
            "site": site,
            "provider": capture["provider"],
            "method": "Rounds appended to this task after its earlier rounds, which stay in their recorded snapshot_path.",
            "rounds": [],
        }
    snapshot["rounds"].append(
        {
            "stage": stage,
            "query": query,
            "provider": capture["provider"],
            "results": _snapshot_rows(result_ids, by_id, site, observed_rows=observed),
        }
    )
    snapshot["retrieved_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json_atomic(root / snapshot_path, snapshot)

    resolution["query_rounds"].append(
        {
            "stage": stage,
            "query": query,
            "provider": capture["provider"],
            "snapshot_path": snapshot_path.as_posix(),
            "result_entity_ids": result_ids,
        }
    )
    window = resolution["window_entity_ids"]
    added = [entity_id for entity_id in result_ids if entity_id not in window]
    window.extend(added)
    final_ids = set().union(*_blueprint_final_options(record))
    ringed = {entity_id for field in RING_FIELDS for entity_id in resolution.get(field, [])}
    pending = [entity_id for entity_id in added if entity_id not in final_ids | ringed]
    resolution.setdefault("ambient_entity_ids", []).extend(pending)
    write_json_atomic(manifest_path, manifest)
    return {
        "task_id": task_id,
        "stage": stage,
        "results": len(result_ids),
        "added_to_window": len(added),
        "pending_judgement": [
            {"entity_id": entity_id, "title": observed[entity_id].get("title")}
            for entity_id in pending
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    result = append_round(
        Path(args.root).resolve(), task_id=args.task_id, stage=args.stage, query=args.query
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
