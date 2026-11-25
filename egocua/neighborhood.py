"""Validation for source-backed task search neighborhoods."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .environment import entity_source_url, entity_visual_path
from .workspace import read_json_strict

NEIGHBORHOOD_POLICY_VERSION = "real-search-window-v1"
TARGET_RELATED_RESULTS = 16
REQUIRED_STAGES = {"before_change", "after_change"}
DETAIL_PREFIX = {"egomart": "product", "egomedia": "watch", "egomaps": "place"}


class NeighborhoodError(ValueError):
    """Raised when a task exposes an incomplete or fabricated result window."""


def _unique_ids(value: Any, *, task_id: str, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise NeighborhoodError(f"{task_id}: {field} must contain unique entity IDs")
    return list(value)


def _snapshot_round(
    *,
    task_id: str,
    site: str,
    stage: str,
    query: str,
    provider: str,
    snapshot_path: Path,
    available_entities: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    snapshot = read_json_strict(snapshot_path)
    if (
        snapshot.get("schema_version") != 1
        or snapshot.get("task_id") != task_id
        or snapshot.get("site") != site
        or not str(snapshot.get("retrieved_at") or "").strip()
        or not str(snapshot.get("method") or "").strip()
    ):
        raise NeighborhoodError(
            f"{task_id}: query snapshot identity mismatch for {stage}"
        )
    rounds = snapshot.get("rounds")
    if not isinstance(rounds, list):
        raise NeighborhoodError(f"{task_id}: query snapshot has no rounds")
    matches = [
        row for row in rounds if isinstance(row, Mapping) and row.get("stage") == stage
    ]
    if len(matches) != 1 or matches[0].get("query") != query:
        raise NeighborhoodError(
            f"{task_id}: query snapshot does not contain the declared {stage} query"
        )
    snapshot_provider = matches[0].get("provider", snapshot.get("provider"))
    if (
        snapshot_provider != provider
        and snapshot.get("provider") != "real-retail-union-v1"
    ):
        raise NeighborhoodError(
            f"{task_id}: query snapshot provider mismatch for {stage}"
        )
    result_rows = matches[0].get("results")
    if not isinstance(result_rows, list):
        raise NeighborhoodError(
            f"{task_id}: query snapshot {stage} results must be a list"
        )
    ids = _unique_ids(
        [
            row.get("entity_id") if isinstance(row, Mapping) else None
            for row in result_rows
        ],
        task_id=task_id,
        field=f"snapshot/{stage}/results",
    )
    for rank, (entity_id, row) in enumerate(
        zip(ids, result_rows, strict=True), start=1
    ):
        if not isinstance(row, Mapping):
            raise NeighborhoodError(
                f"{task_id}: snapshot {stage} row {rank} is invalid"
            )
        entity = available_entities.get(entity_id)
        if not isinstance(entity, Mapping):
            raise NeighborhoodError(
                f"{task_id}: snapshot {stage} entity {entity_id} is not in the Mock catalog"
            )
        expected_visual = entity_visual_path(site, dict(entity))
        expected_detail = f"/{DETAIL_PREFIX[site]}/{entity_id}"
        source_url = str(row.get("source_url") or "")
        if row.get("rank") != rank:
            raise NeighborhoodError(
                f"{task_id}: snapshot {stage} ranks must be consecutive"
            )
        if not str(row.get("title") or "").strip():
            raise NeighborhoodError(
                f"{task_id}: snapshot {stage} title is missing for {entity_id}"
            )
        if (
            source_url != entity_source_url(entity)
            or urlparse(source_url).scheme != "https"
        ):
            raise NeighborhoodError(
                f"{task_id}: snapshot {stage} source URL disagrees with {entity_id}"
            )
        if row.get("visual_path") != expected_visual:
            raise NeighborhoodError(
                f"{task_id}: snapshot {stage} visual disagrees with {entity_id}"
            )
        if row.get("mock_detail_path") != expected_detail:
            raise NeighborhoodError(
                f"{task_id}: snapshot {stage} detail route disagrees with {entity_id}"
            )
    return ids


def search_evidence_sha256(
    *,
    task_id: str,
    site: str,
    resolution: Mapping[str, Any],
    root: Path,
) -> str:
    """Hash the exact external search evidence referenced by one task."""

    rounds = resolution.get("query_rounds")
    if not isinstance(rounds, list):
        raise NeighborhoodError(f"{task_id}: query rounds are missing")
    evidence: list[dict[str, Any]] = []
    workspace = root.resolve()
    for row in rounds:
        if not isinstance(row, Mapping):
            raise NeighborhoodError(f"{task_id}: query round must be an object")
        relative = str(row.get("snapshot_path") or "").strip()
        path = (workspace / relative).resolve()
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise NeighborhoodError(f"{task_id}: snapshot escapes workspace") from exc
        if not path.is_file():
            raise NeighborhoodError(f"{task_id}: missing query snapshot {relative}")
        evidence.append(
            {
                "stage": row.get("stage"),
                "query": row.get("query"),
                "provider": row.get("provider"),
                "result_entity_ids": row.get("result_entity_ids"),
                "snapshot_path": relative,
                "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload = {"task_id": task_id, "site": site, "query_rounds": evidence}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_search_neighborhood(
    *,
    task_id: str,
    site: str,
    resolution: Mapping[str, Any],
    available_entities: Mapping[str, Mapping[str, Any]],
    final_ids: set[str],
    root: Path,
) -> list[str]:
    """Validate and return the exact task-local public result window."""

    if resolution.get("policy_version") != NEIGHBORHOOD_POLICY_VERSION:
        raise NeighborhoodError(
            f"{task_id}: catalog_resolution must use {NEIGHBORHOOD_POLICY_VERSION}"
        )
    rounds = resolution.get("query_rounds")
    if not isinstance(rounds, list) or len(rounds) < 2:
        raise NeighborhoodError(
            f"{task_id}: before/after query coverage references are required"
        )
    stages: set[str] = set()
    round_ids: list[str] = []
    for index, row in enumerate(rounds):
        if not isinstance(row, Mapping):
            raise NeighborhoodError(f"{task_id}: query round {index} must be an object")
        stage = row.get("stage")
        query = str(row.get("query") or "").strip()
        provider = str(row.get("provider") or "").strip()
        snapshot = str(row.get("snapshot_path") or "").strip()
        if not isinstance(stage, str) or not stage.strip() or stage in stages:
            raise NeighborhoodError(
                f"{task_id}: invalid or duplicate query stage {stage!r}"
            )
        if not query or not provider or not snapshot:
            raise NeighborhoodError(
                f"{task_id}: query round {stage} lacks query/provider/snapshot"
            )
        snapshot_path = (root / snapshot).resolve()
        try:
            snapshot_path.relative_to(root.resolve())
        except ValueError as exc:
            raise NeighborhoodError(f"{task_id}: snapshot escapes workspace") from exc
        if not snapshot_path.is_file():
            raise NeighborhoodError(f"{task_id}: missing query snapshot {snapshot}")
        ids = _unique_ids(
            row.get("result_entity_ids"),
            task_id=task_id,
            field=f"query_rounds[{index}].result_entity_ids",
        )
        if not ids:
            raise NeighborhoodError(f"{task_id}: query round {stage} is empty")
        snapshot_ids = _snapshot_round(
            task_id=row.get("snapshot_task_id", task_id),
            site=site,
            stage=row.get("snapshot_stage", stage),
            query=query,
            provider=provider,
            snapshot_path=snapshot_path,
            available_entities=available_entities,
        )
        if ids != snapshot_ids:
            raise NeighborhoodError(
                f"{task_id}: query round {stage} disagrees with its frozen snapshot"
            )
        stages.add(stage)
        round_ids.extend(ids)
    if not REQUIRED_STAGES <= stages:
        raise NeighborhoodError(
            f"{task_id}: before/after query coverage references are both required"
        )

    window_ids = _unique_ids(
        resolution.get("window_entity_ids"),
        task_id=task_id,
        field="window_entity_ids",
    )
    expected_window = list(dict.fromkeys(round_ids))
    if window_ids != expected_window:
        raise NeighborhoodError(
            f"{task_id}: window_entity_ids must preserve the declared query union order"
        )
    missing = sorted(set(window_ids) - set(available_entities))
    if missing:
        raise NeighborhoodError(
            f"{task_id}: search window entities are missing: {missing}"
        )
    core_ids = _unique_ids(
        resolution.get("core_neighbor_entity_ids"),
        task_id=task_id,
        field="core_neighbor_entity_ids",
    )
    related_ids = _unique_ids(
        resolution.get("related_entity_ids"),
        task_id=task_id,
        field="related_entity_ids",
    )
    ambient_ids = _unique_ids(
        resolution.get("ambient_entity_ids"),
        task_id=task_id,
        field="ambient_entity_ids",
    )
    ring_sets = [set(core_ids), set(related_ids), set(ambient_ids)]
    if any(
        ring_sets[left] & ring_sets[right]
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise NeighborhoodError(f"{task_id}: result rings must be disjoint")
    if set(core_ids) & final_ids:
        raise NeighborhoodError(f"{task_id}: a final answer cannot be a core neighbor")
    classified = set().union(*ring_sets)
    # Earlier semantic entities are legitimate final-state negatives: stale Graft
    # goals, the unchosen Clarify branch, and tempting Refresh mappings. Rings
    # classify every row that is not accepted by the final checker.
    expected_classified = set(window_ids) - final_ids
    if classified != expected_classified:
        raise NeighborhoodError(
            f"{task_id}: result rings must classify every non-answer window entity exactly once"
        )

    rejected = resolution.get("rejected")
    if not isinstance(rejected, list):
        raise NeighborhoodError(f"{task_id}: rejected must be a list")
    by_id = {
        str(row.get("entity_id")): row
        for row in rejected
        if isinstance(row, Mapping) and row.get("entity_id")
    }
    for entity_id in core_ids:
        row = by_id.get(entity_id)
        if not isinstance(row, Mapping):
            raise NeighborhoodError(
                f"{task_id}: core neighbor {entity_id} lacks a rejection record"
            )
        failure_predicates = row.get("failure_predicates")
        if (
            row.get("ring") != "core"
            or not str(row.get("reason") or "").strip()
            or not str(row.get("why_plausible") or "").strip()
            or not isinstance(failure_predicates, list)
            or not failure_predicates
            or any(
                not isinstance(predicate, str) or not predicate.strip()
                for predicate in failure_predicates
            )
        ):
            raise NeighborhoodError(
                f"{task_id}: core neighbor {entity_id} requires rejection facts "
                "and failure predicates"
            )
    detail_fields = resolution.get("detail_decision_fields")
    if (
        not isinstance(detail_fields, list)
        or not detail_fields
        or any(not isinstance(field, str) or not field for field in detail_fields)
    ):
        raise NeighborhoodError(
            f"{task_id}: website detail decision fields are required"
        )
    return window_ids
