"""Screening-record and frozen-site fixtures for construction tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from egocua.construction import opaque_item_id


def curve(
    task_id: str,
    track: str,
    candidates: dict[str, list[str]],
    settles_on: str,
    narrowed_by: str,
) -> dict[str, Any]:
    """A survival curve over the fixture's frames: everything live, then the utterance."""

    counts = (("V0", 4), ("V1", 4)) if track == "refresh" else (("V0", 8),)
    frames = [f"{packet}:{index}" for packet, count in counts for index in range(count)]
    live = sorted(candidates)
    observed: list[dict[str, Any]] = [
        {"frame": frame, "live": live} for frame in frames
    ]
    if track != "clarify":
        # Only Clarify may go unsettled by its frames; the others settle on one.
        observed[-1] = {
            "frame": frames[-1],
            "live": [settles_on],
            "narrowed_by": narrowed_by,
            "why": "the later packet shows which one is in use",
        }
    else:
        observed.append(
            {
                "frame": "utterance",
                "live": [settles_on],
                "narrowed_by": narrowed_by,
                "why": "the user says which one",
            }
        )
    return {
        "candidates": {
            role: [opaque_item_id(task_id, entity) for entity in entities]
            for role, entities in candidates.items()
        },
        "observed": observed,
        "established_by": "fixture",
        "established_at": "2026-08-23",
    }


def frozen_site(root: Path, task_id: str, roles: list[str]) -> dict[str, Any]:
    """A frozen EgoMart-shaped site holding the role entities plus distractors."""

    entities = []
    for index, role in enumerate(
        [*roles, *[f"distractor_{i}" for i in range(32 - len(roles))]]
    ):
        path = Path("product-images") / f"image-{index}.jpg"
        image = root / path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        entity_id = opaque_item_id(task_id, role)
        entities.append(
            {
                "item_id": entity_id,
                "entity_id": entity_id,
                "name": f"Product {index}",
                "price": f"${index + 1}.00",
                "image_path": path.as_posix(),
                "image_provenance": {
                    "source_url": f"https://example.invalid/products/{index}",
                    "image_url": f"https://example.invalid/images/{index}.jpg",
                    "retrieved_at": "2026-08-10T00:00:00Z",
                    "license_or_access_note": "Unit-test fixture.",
                },
                "attributes": {"variant": f"Variant {index}"},
            }
        )
    return {
        "schema_version": 1,
        "site": "egomart",
        "supported_fields": [],
        "entities": entities,
    }


def catalog_resolution(
    root: Path,
    task_id: str,
    roles: list[str],
    final_roles: list[str] | None = None,
) -> dict[str, Any]:
    final = {
        opaque_item_id(task_id, role)
        for role in (final_roles if final_roles is not None else roles)
    }
    named_roles = [role for role in roles if not role.startswith("distractor_")]
    window = [
        opaque_item_id(task_id, role)
        for role in [
            *named_roles,
            *[f"distractor_{i}" for i in range(32 - len(named_roles))],
        ]
    ]
    candidates = [entity_id for entity_id in window if entity_id not in final]
    core = candidates[4:8]
    related = candidates[8:16]
    ambient = [entity_id for entity_id in candidates if entity_id not in core + related]
    snapshot = Path("snapshots") / f"{task_id}.json"
    path = root / snapshot
    path.parent.mkdir(parents=True, exist_ok=True)

    def result_row(entity_id: str, rank: int) -> dict[str, Any]:
        index = window.index(entity_id)
        return {
            "rank": rank,
            "entity_id": entity_id,
            "title": f"Product {index}",
            "source_url": f"https://example.invalid/products/{index}",
            "visual_path": f"product-images/image-{index}.jpg",
            "mock_detail_path": f"/product/{entity_id}",
        }

    snapshot_payload = {
        "schema_version": 1,
        "task_id": task_id,
        "site": "egomart",
        "provider": "fixture-real-catalog",
        "retrieved_at": "2026-08-10T00:00:00Z",
        "method": "Unit-test frozen result window.",
        "rounds": [
            {
                "stage": "before_change",
                "query": "ordinary broad query",
                "results": [
                    result_row(entity_id, rank)
                    for rank, entity_id in enumerate(window[:24], start=1)
                ],
            },
            {
                "stage": "after_change",
                "query": "ordinary changed query",
                "results": [
                    result_row(entity_id, rank)
                    for rank, entity_id in enumerate(window[8:], start=1)
                ],
            },
        ],
    }
    path.write_text(json.dumps(snapshot_payload))
    predicates = [
        "scene_evidence_mismatch",
        "late_change_mismatch",
        "website_detail_mismatch",
        "scene_evidence_mismatch",
    ]
    return {
        "policy_version": "real-search-window-v1",
        "query_rounds": [
            {
                "stage": "before_change",
                "query": "ordinary broad query",
                "provider": "fixture-real-catalog",
                "snapshot_path": snapshot.as_posix(),
                "result_entity_ids": window[:24],
            },
            {
                "stage": "after_change",
                "query": "ordinary changed query",
                "provider": "fixture-real-catalog",
                "snapshot_path": snapshot.as_posix(),
                "result_entity_ids": window[8:],
            },
        ],
        "window_entity_ids": window,
        "core_neighbor_entity_ids": core,
        "related_entity_ids": related,
        "ambient_entity_ids": ambient,
        "detail_decision_fields": ["name", "attributes"],
        "rejected": [
            {
                "entity_id": entity_id,
                "ring": "core",
                "reason": f"Fixture rejection {index}.",
                "why_plausible": "Same category with one intentionally mismatched condition.",
                "failure_predicates": [predicates[index]],
                "semantic_review": "coding_agent_accepted",
            }
            for index, entity_id in enumerate(core)
        ],
    }


def frame_packets(root: Path, track: str) -> list[dict[str, Any]]:
    counts = (("V0", 4), ("V1", 4)) if track == "refresh" else (("V0", 8),)
    packets = []
    for packet_id, count in counts:
        paths = []
        for index in range(count):
            path = Path("frames") / f"{track}-{packet_id}-{index}.jpg"
            frame = root / path
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(f"frame-{track}-{packet_id}-{index}".encode())
            paths.append(path.as_posix())
        packets.append({"id": packet_id, "frame_paths": paths})
    return packets


def screening_record(root: Path, track: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = f"seed-{track}"
    common: dict[str, Any] = {
        "task_id": task_id,
        "status": "video_screened",
        "family": track,
        "packets": frame_packets(root, track),
    }
    if track == "graft":
        common.update(
            {
                "subtype": "augment",
                "construct": {
                    "decision": curve(
                        task_id,
                        "graft",
                        {
                            "base_product": ["base_product", "second_product"],
                            "distractor_0": ["distractor_0"],
                        },
                        settles_on="base_product",
                        narrowed_by="scene_appearance",
                    ),
                    "task_blueprint": {
                        "instructions": {
                            "a": "Find A",
                            "b": "Also add B",
                            "bundled": "Find A and B",
                        },
                        "initial_goal_roles": [["base_product"]],
                        "final_goal_roles": [["base_product", "second_product"]],
                        "target_granularity": "variant",
                    },
                },
            }
        )
        roles = ["base_product", "second_product"]
        final_roles = ["base_product", "second_product"]
    elif track == "refresh":
        common.update(
            {
                "subtype": "replace",
                "construct": {
                    "decision": curve(
                        task_id,
                        "refresh",
                        {
                            "first_product": ["first_product"],
                            "second_product": ["second_product"],
                        },
                        settles_on="second_product",
                        narrowed_by="action_order",
                    ),
                    "task_blueprint": {
                        "instruction": "Find the product I switched to",
                        "initial_target_role": "first_product",
                        "final_target_role": "second_product",
                        "target_granularity": "variant",
                    },
                },
            }
        )
        roles = ["first_product", "second_product"]
        final_roles = ["second_product"]
    else:
        common.update(
            {
                "subtype": "variant",
                "construct": {
                    "decision": curve(
                        task_id,
                        "clarify",
                        {
                            "red_product": ["red_product"],
                            "blue_product": ["blue_product"],
                        },
                        settles_on="blue_product",
                        narrowed_by="scene_appearance",
                    ),
                    "task_blueprint": {
                        "ambiguous_instruction": "Find this product",
                        "oracle_instruction": "Find the blue product",
                        "candidate_roles": ["red_product", "blue_product"],
                        "hidden_target_role": "blue_product",
                        "target_granularity": "variant",
                        "valid_slot": "variant",
                        "canonical_question": "Do you mean the red or blue product?",
                        "canonical_answer": "The blue one.",
                    },
                },
            }
        )
        roles = ["red_product", "blue_product"]
        final_roles = ["blue_product"]
    semantic_roles = [
        *roles,
        *(("distractor_0",) if track == "graft" else ()),
    ]
    common["construct"]["catalog_resolution"] = catalog_resolution(
        root,
        task_id,
        semantic_roles,
        final_roles,
    )
    common["construct"]["task_blueprint"]["site"] = "egomart"
    return common, frozen_site(root, task_id, roles)


def source_identity(record: dict[str, Any]) -> dict[str, str]:
    return {
        "dossier": "data/source.json",
        "dossier_sha256": "1" * 64,
        "task_id": str(record["task_id"]),
        "frame_provenance": "data/source_frame_provenance.json",
        "frame_provenance_sha256": "3" * 64,
    }
