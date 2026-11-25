from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from egocua.environment import load_site_data
from egocua.neighborhood import search_evidence_sha256
from egocua.sources import source_record
from egocua.construct_validity import (
    ADMISSION_GATES,
    COUNTEREXAMPLE_CHECKS,
    EVIDENCE_DIMENSIONS,
    PASS_NAMES,
    POLICY,
)
from egocua.confidence import route_assessment
from egocua.confidence_packets import CurrentReviewMaterials, answer_boundary
from egocua.construct_validity import CONFIDENCE_POLICY_VERSION
from egocua.experiment import ExperimentPlan
from egocua.review_workspace import ReviewWorkspace
from egocua.run_ledger import (
    RUN_ARTIFACT_SCHEMA_VERSION,
    build_run_meta,
    canonical_sha256,
    task_sha256,
    task_site_execution_sha256,
)
from egocua.schema import goal_for_condition
from egocua.synthesis_reports import REPORTS
from tests.conditions import CONDITIONS


@pytest.fixture
def unchanged_task_owners(monkeypatch):
    """Keep lifecycle unit fixtures focused on review and execution evidence."""
    from egocua.construction import CurrentConstruction

    monkeypatch.setattr(CurrentConstruction, "matches", lambda self, task: True)


def coding_agent_assessment(
    definition: dict[str, Any], confidence: str = "high"
) -> dict:
    verdict = {
        "decision": "admit",
        "confidence": confidence,
        "gates": {name: "pass" for name in EVIDENCE_DIMENSIONS},
        "evidence": {
            name: [f"{definition['id']}:{name}"] for name in EVIDENCE_DIMENSIONS
        },
        "uncertainty_reasons": ["fixture uncertainty"] if confidence == "low" else [],
        "issues": [],
    }
    passes: list[dict[str, Any]] = [
        {
            "index": index,
            "pass": name,
            "status": "completed",
            "verdict": deepcopy(verdict),
        }
        for index, name in enumerate(PASS_NAMES)
    ]
    passes[0]["verdict"]["derived_final_state"] = deepcopy(
        definition["goal"]["acceptable_states"][0]
    )
    passes[1]["verdict"]["counterexample_checks"] = {
        name: "pass" for name in COUNTEREXAMPLE_CHECKS
    }
    return {
        "task_id": definition["id"],
        "task_sha256": task_sha256(definition),
        "answer_boundary": answer_boundary(definition),
        "status": "completed",
        "assessments": passes,
    }


PLAN = ExperimentPlan(
    "test-plan",
    {
        "graft": ("initial_only", "upfront", "online"),
        "refresh": ("initial_only", "online", "upfront"),
        "clarify": ("upfront", "online", "no_ask"),
    },
    (0, 1, 2),
    "minimal-loop",
    "scripted-fixture",
    "scripted-fixture",
    40,
    7,
    "online",
    conditions=CONDITIONS,
    default_release_step=2,
    reference_steps=None,
    track_probe_conditions={
        "graft": ("first_frame", "text_only"),
        "refresh": ("first_frame", "text_only"),
        "clarify": ("first_frame", "text_only"),
    },
)


def complete_run_meta(**overrides: Any) -> dict[str, Any]:
    return {
        "agent": "test",
        "model": "test",
        "provider": "test",
        "model_version": "test",
        "transport": "test",
        "replicate_id": 0,
        "schedule_index": None,
        "provider_seed": None,
        "randomness_control": "uncontrolled",
        "max_steps": PLAN.max_steps,
        "task_sha256": "test",
        "site": "egomart",
        "site_data_sha256": "test",
        "asset_sha256": {},
        "runtime": {},
        "operator_id": None,
        "default_release_step": PLAN.default_release_step,
        "reference_steps": PLAN.reference_steps,
        "started_at": 0.0,
        "duration_seconds": 0.0,
        "event_delivery_steps": [],
        "run_id": None,
        "artifact_directory": None,
        **overrides,
    }


def complete_run_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "live_state": {
            "selected_entity_ids": deepcopy(
                record.get("final_state", {}).get("selected_entity_ids", [])
            ),
            "fields": deepcopy(record.get("final_state", {}).get("fields", {})),
        },
        "task_id": "test-record",
        "track": "graft",
        "condition": CONDITIONS["online"].as_dict(),
        "run_status": {"valid": True, "reason": None},
        "initial_state": {
            "selected_entity_ids": [],
            "fields": {},
            "submission_completed": False,
        },
        "steps": 0,
        "actions": [],
        "final_state": {
            "selected_entity_ids": [],
            "fields": {},
            "submission_completed": False,
        },
        "trace": [],
        **record,
        "meta": complete_run_meta(**record.get("meta", {})),
    }


def plan_payload(plan: ExperimentPlan = PLAN) -> dict[str, Any]:
    """The fixture plan as it is written to disk. One definition, so a new plan
    field cannot be declared in the object and forgotten in the JSON."""

    return {
        "schema_version": 3,
        "experiment_id": plan.experiment_id,
        "default_release_step": plan.default_release_step,
        "reference_steps": plan.reference_steps,
        "reference_condition": plan.reference_condition_name,
        "conditions": [condition.as_dict() for condition in plan.conditions.values()],
        "tracks": {
            track: list(condition_names)
            for track, condition_names in plan.track_conditions.items()
        },
        **({"probes": {
            track: list(condition_names)
            for track, condition_names in plan.track_probe_conditions.items()
        }} if plan.track_probe_conditions else {}),
        "replicate_ids": list(plan.replicate_ids),
        "runtime": {
            "adapter": plan.adapter,
            "model": plan.model,
            "model_version": plan.model_version,
            "max_steps": plan.max_steps,
        },
        "scheduler_seed": plan.scheduler_seed,
    }


def _run_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        **meta,
        "default_release_step": PLAN.default_release_step,
        "reference_steps": PLAN.reference_steps,
    }


def _write_run_artifact(
    directory: Path,
    condition_name: str,
    record: dict[str, Any],
    *,
    experiment_id: str = "test-fixture",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    run_id = f"fixture-{directory.name}"
    record = complete_run_record(record)
    record.pop("archive", None)
    record["meta"].update(
        {
            "run_id": run_id,
            "artifact_directory": directory.as_posix(),
        }
    )
    (directory / "run.json").write_text(json.dumps(record))
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
                "run_id": run_id,
                "experiment_id": experiment_id,
                "task_id": record["task_id"],
                "condition": CONDITIONS[condition_name].as_dict(),
                "replicate_id": record["meta"]["replicate_id"],
                "agent": record["meta"]["agent"],
                "status": "completed",
                "started_at_utc": "2026-08-20T00:00:00Z",
                "finished_at_utc": "2026-08-20T00:00:01Z",
                "record_sha256": canonical_sha256(record),
                "provenance": record["meta"],
            }
        )
    )


def item_id(track: str, index: int) -> str:
    digest = hashlib.sha256(f"fixture:{track}:{index}".encode()).hexdigest()[:12]
    return f"p_{digest}"


def clarification_question(slot: str) -> str:
    return {
        "position": "Which position do you mean?",
        "appearance": "Which appearance do you mean?",
        "brand": "Which brand do you mean?",
        "variant": "Which variant do you mean?",
        "sequence": "Which one in the sequence do you mean?",
    }[slot]


def _tier(index: int) -> str:
    if index == 0:
        return "target"
    if index < 4:
        return "hard"
    if index < 8:
        return "medium"
    return "easy"


def products(track: str) -> list[dict[str, Any]]:
    return [
        {
            "item_id": item_id(track, index),
            "name": f"{track} Product {index}",
            "price": f"${index + 1}.00",
            "rating": 4.5,
            "ratings_count": 100 + index,
            "image_path": f"products/{item_id(track, index)}.jpg",
            "image_provenance": {
                "source_url": f"https://example.invalid/products/{item_id(track, index)}",
                "image_url": f"https://example.invalid/images/{item_id(track, index)}.jpg",
                "retrieved_at": "2026-08-09T00:00:00Z",
                "license_or_access_note": "Synthetic unit-test fixture; not benchmark data.",
                "sha256": hashlib.sha256(
                    f"asset:{item_id(track, index)}".encode()
                ).hexdigest(),
            },
            "tier": _tier(index),
            "attributes": {"variant": "target" if index < 2 else f"decoy-{index}"},
        }
        for index in range(16)
    ]


def site_data(track: str) -> dict[str, Any]:
    return {
        "site": "egomart",
        "entities": [
            {**product, "entity_id": product["item_id"]} for product in products(track)
        ],
    }


def task(track: str) -> dict[str, Any]:
    base: dict[str, Any] = {
        "schema_version": 2,
        "id": f"test-{track}",
        "track": track,
        "site": "egomart",
        "source": {"dataset": "fixture", "id": 1},
    }
    if track == "graft":
        base.update(
            {
                "instructions": {
                    "a": "Find A",
                    "b": "Also apply B",
                    "bundled": "Find A and apply B",
                },
                "visual_packets": [
                    {
                        "id": "V0",
                        "frame_paths": [f"frames/graft-{i}.jpg" for i in range(8)],
                    }
                ],
                "graft": {
                    "transition": "refine",
                    "initial_goal": {
                        "target_granularity": "variant",
                        "acceptable_states": [
                            {
                                "selected_entity_ids": [item_id("graft", index)],
                                "fields": {},
                                "submission_completed": True,
                            }
                            for index in (0, 1)
                        ],
                    },
                },
                "goal": {
                    "target_granularity": "variant",
                    "acceptable_states": [
                        {
                            "selected_entity_ids": [item_id("graft", 0)],
                            "fields": {},
                            "submission_completed": True,
                        }
                    ],
                },
            }
        )
    elif track == "refresh":
        base.update(
            {
                "instructions": {"base": "Find the item selected last"},
                "visual_packets": [
                    {
                        "id": "V0",
                        "frame_paths": [f"frames/refresh-v0-{i}.jpg" for i in range(4)],
                    },
                    {
                        "id": "V1",
                        "frame_paths": [f"frames/refresh-v1-{i}.jpg" for i in range(4)],
                    },
                ],
                "refresh": {
                    "policy": "replace",
                    "initial_entity_id": item_id("refresh", 0),
                },
                "truth": {
                    "initial_acceptable_states": [
                        {"selected_entity_ids": [item_id("refresh", 0)], "fields": {}}
                    ],
                },
                "goal": {
                    "target_granularity": "variant",
                    "acceptable_states": [
                        {
                            "selected_entity_ids": [item_id("refresh", 1)],
                            "fields": {},
                            "submission_completed": True,
                        }
                    ],
                },
            }
        )
    else:
        base.update(
            {
                "instructions": {
                    "ambiguous": "Find this product",
                    "oracle": "Find the target variant",
                },
                "visual_packets": [
                    {
                        "id": "V0",
                        "frame_paths": [f"frames/clarify-{i}.jpg" for i in range(8)],
                    }
                ],
                "clarification": {
                    "candidate_ids": [item_id("clarify", 0), item_id("clarify", 1)],
                    "valid_slot": "variant",
                    "canonical_answer": "The target variant.",
                    "max_questions": 1,
                },
                "goal": {
                    "target_granularity": "variant",
                    "acceptable_states": [
                        {
                            "selected_entity_ids": [item_id("clarify", 1)],
                            "fields": {},
                            "submission_completed": True,
                        }
                    ],
                },
            }
        )
    return base


def release_ready_task(track: str) -> dict[str, Any]:
    """Synthetic task fixture with explicitly registered diagnostic truth."""

    fixture = task(track)
    final_ids = {
        entity_id
        for state in fixture["goal"]["acceptable_states"]
        for entity_id in state["selected_entity_ids"]
    }
    if track == "graft":
        initial_ids = {
            entity_id
            for state in fixture["graft"]["initial_goal"]["acceptable_states"]
            for entity_id in state["selected_entity_ids"]
        }
        tempting = sorted(initial_ids - final_ids)
    elif track == "refresh":
        tempting = sorted({fixture["refresh"]["initial_entity_id"]} - final_ids)
    else:
        fixture["clarification"]["deixis_level"] = 1
        tempting = sorted(set(fixture["clarification"]["candidate_ids"]) - final_ids)
    fixture.setdefault("truth", {})["tempting_referent_ids"] = tempting
    return fixture


def materialize_site(items: list[dict[str, Any]], root: Path) -> None:
    path = root / "sites/egomart/products.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.is_file():
        existing = json.loads(path.read_text())["products"]
    by_id = {item["item_id"]: item for item in [*existing, *items]}
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site": "egomart",
                "products": list(by_id.values()),
            }
        )
    )


def materialize_assets(
    task_definition: dict[str, Any], items: list[dict[str, Any]], root: Path
) -> None:
    config_path = root / "configs/pilot.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(plan_payload()))
    formal_path = root / "configs/benchmark.json"
    if not formal_path.exists():
        formal_path.symlink_to(config_path.name)
    materialize_site(items, root)
    frame_paths = [
        frame
        for packet in task_definition["visual_packets"]
        for frame in packet["frame_paths"]
    ]
    for relative_path in frame_paths:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frame:{relative_path}".encode())
    _materialize_source_lineage(task_definition, root)
    for item in items:
        path = root / item["image_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"asset:{item['item_id']}".encode())
    for condition_name in PLAN.conditions_for(task_definition["track"]):
        condition = PLAN.condition(condition_name)
        if (
            task_definition["track"] == "refresh"
            and task_definition["refresh"]["policy"] == "replace"
            and condition.scope == "initial_only"
        ):
            selected_ids = [task_definition["refresh"]["initial_entity_id"]]
        elif task_definition["track"] == "clarify" and condition.ask == "denied":
            selected_ids = [task_definition["clarification"]["candidate_ids"][0]]
        else:
            selected_ids = goal_for_condition(
                task_definition,
                condition,
            )["acceptable_states"][0]["selected_entity_ids"]
        actions: list[dict[str, Any]] = []
        step = 1
        if (
            task_definition["track"] == "graft"
            and condition.schedule == "online"
            and condition.scope == "full"
        ):
            actions.append(
                {
                    "type": "instruction_injection",
                    "step": 2,
                    "trigger_reason": "task_timeline",
                }
            )
            step = 5
        elif (
            task_definition["track"] == "refresh"
            and condition.schedule == "online"
            and condition.scope == "full"
        ):
            actions.append(
                {
                    "type": "visual_context_update",
                    "step": 2,
                    "trigger_reason": "task_timeline",
                }
            )
            step = 5
        elif (
            task_definition["track"] == "clarify"
            and condition.schedule == "online"
            and condition.scope == "full"
            and condition.ask == "allowed"
        ):
            clarification = task_definition["clarification"]
            actions.append(
                {
                    "type": "clarification_answer",
                    "step": 1,
                    "slot": clarification["valid_slot"],
                    "question": clarification_question(clarification["valid_slot"]),
                    "valid": True,
                    "payload": clarification["canonical_answer"],
                }
            )
            step = 2
        actions.extend(
            {"type": "select", "step": step, "item_id": item} for item in selected_ids
        )
        human_meta = _run_meta(
            build_run_meta(
                task_definition,
                asset_root=root,
                agent="human",
                model="human",
                provider="human",
                model_version="human",
                transport="exact-harness",
                replicate_id=0,
                provider_seed=None,
                max_steps=PLAN.max_steps,
            )
        )
        for evaluator in ("human-1", "human-2"):
            evaluator_meta = {**human_meta, "operator_id": evaluator}
            record = {
                "task_id": task_definition["id"],
                "task_snapshot": deepcopy(task_definition),
                "track": task_definition["track"],
                "condition": condition.as_dict(),
                "run_status": {"valid": True, "reason": None},
                "steps": step,
                "actions": actions,
                "final_state": {
                    "selected_entity_ids": selected_ids,
                    "fields": {},
                    "submission_completed": True,
                },
                "meta": evaluator_meta,
            }
            artifact = (
                root
                / "artifacts"
                / "runs"
                / "human"
                / task_definition["track"]
                / condition_name
                / evaluator
            )
            _write_run_artifact(artifact, condition_name, record)
    reference_condition = PLAN.reference_condition_for(task_definition["track"])
    reference_ids = goal_for_condition(
        task_definition,
        reference_condition,
    )["acceptable_states"][0]["selected_entity_ids"]
    replay_specs: dict[str, tuple[str, list[str], list[dict[str, Any]]]] = {}
    reference_actions: list[dict[str, Any]] = []
    reference_step = 1
    if task_definition["track"] == "graft":
        reference_actions.append(
            {
                "type": "instruction_injection",
                "step": 2,
                "trigger_reason": "task_timeline",
            }
        )
        reference_step = 5
    elif task_definition["track"] == "refresh":
        reference_actions.append(
            {
                "type": "visual_context_update",
                "step": 2,
                "trigger_reason": "task_timeline",
            }
        )
        reference_step = 5
    elif task_definition["track"] == "clarify":
        clarification = task_definition["clarification"]
        reference_actions.append(
            {
                "type": "clarification_answer",
                "step": 1,
                "slot": clarification["valid_slot"],
                "question": clarification_question(clarification["valid_slot"]),
                "valid": True,
                "payload": clarification["canonical_answer"],
            }
        )
        reference_step = 2
    reference_actions.extend(
        {"type": "select", "step": reference_step, "item_id": item}
        for item in reference_ids
    )
    replay_specs["alternative_correct_path"] = (
        reference_condition.name,
        reference_ids,
        reference_actions,
    )
    replay_specs["extra_item_rejected"] = (
        reference_condition.name,
        [*reference_ids, item_id(task_definition["track"], 15)],
        reference_actions,
    )
    replay_specs["duplicate_quantity_rejected"] = (
        reference_condition.name,
        [*reference_ids, reference_ids[0]],
        reference_actions,
    )
    if task_definition["track"] in {"graft", "refresh"}:
        dynamic_condition = "online"
        event_type = (
            "instruction_injection"
            if task_definition["track"] == "graft"
            else "visual_context_update"
        )
        dynamic_ids = goal_for_condition(
            task_definition,
            PLAN.condition(dynamic_condition),
        )["acceptable_states"][0]["selected_entity_ids"]
        replay_specs["early_submission_rejected"] = (
            dynamic_condition,
            dynamic_ids,
            [
                {"type": "submit", "step": 1, "item_ids": dynamic_ids},
                {"type": event_type, "step": 2, "trigger_reason": "task_timeline"},
            ],
        )
        replay_specs["missing_event_rejected"] = (
            dynamic_condition,
            dynamic_ids,
            [{"type": "submit", "step": 2, "item_ids": dynamic_ids}],
        )
    else:
        target = reference_ids[0]
        wrong = next(
            candidate
            for candidate in task_definition["clarification"]["candidate_ids"]
            if candidate != target
        )
        answer = task_definition["clarification"]["canonical_answer"]
        slot = task_definition["clarification"]["valid_slot"]
        question = clarification_question(slot)
        wrong_slot = next(
            candidate_slot
            for candidate_slot in (
                "position",
                "appearance",
                "brand",
                "variant",
                "sequence",
            )
            if candidate_slot != slot
        )
        replay_specs.update(
            {
                "invalid_slot_detected": (
                    "online",
                    [target],
                    [
                        {
                            "type": "clarification_answer",
                            "step": 1,
                            "slot": wrong_slot,
                            "question": clarification_question(wrong_slot),
                            "valid": False,
                            "payload": "That question did not distinguish the candidates.",
                        },
                        {"type": "select", "step": 2, "item_id": target},
                    ],
                ),
                "second_question_limit_detected": (
                    "online",
                    [target],
                    [
                        {
                            "type": "clarification_answer",
                            "step": 1,
                            "slot": slot,
                            "question": question,
                            "valid": True,
                            "payload": answer,
                        },
                        {
                            "type": "clarification_answer",
                            "step": 2,
                            "slot": slot,
                            "limit_reached": True,
                        },
                        {"type": "select", "step": 3, "item_id": target},
                    ],
                ),
                "late_question_detected": (
                    "online",
                    [target],
                    [
                        {"type": "select", "step": 1, "item_id": target},
                        {
                            "type": "clarification_answer",
                            "step": 2,
                            "slot": slot,
                            "question": question,
                            "valid": True,
                            "payload": answer,
                        },
                    ],
                ),
                "ignored_answer_detected": (
                    "online",
                    [wrong],
                    [
                        {
                            "type": "clarification_answer",
                            "step": 1,
                            "slot": slot,
                            "question": question,
                            "valid": True,
                            "payload": answer,
                        },
                        {"type": "select", "step": 2, "item_id": wrong},
                    ],
                ),
            }
        )
    evaluator_meta = _run_meta(
        build_run_meta(
            task_definition,
            asset_root=root,
            agent="evaluator-replay",
            model="deterministic",
            provider="local",
            model_version="deterministic",
            transport="replay",
            replicate_id=0,
            provider_seed=None,
            max_steps=PLAN.max_steps,
        )
    )
    for case, (condition_name, selected_ids, actions) in replay_specs.items():
        condition = PLAN.condition(condition_name)
        record = {
            "task_id": task_definition["id"],
            "task_snapshot": deepcopy(task_definition),
            "track": task_definition["track"],
            "condition": condition.as_dict(),
            "run_status": {"valid": True, "reason": None},
            "steps": max((action.get("step", 0) for action in actions), default=1),
            "actions": actions,
            "final_state": {
                "selected_entity_ids": selected_ids,
                "fields": {},
                "submission_completed": True,
            },
            "meta": evaluator_meta,
        }
        artifact = (
            root / "artifacts" / "runs" / "evaluator" / task_definition["track"] / case
        )
        _write_run_artifact(artifact, condition_name, record)
    from egocua.development_gate import POLICY, REVIEWS, material_identity

    review_path = root / REVIEWS
    development_reviews = (
        json.loads(review_path.read_text()) if review_path.is_file() else {"tasks": {}}
    )
    development_reviews.setdefault("legacy_baseline", {"policy": POLICY, "tasks": {}})[
        "tasks"
    ][task_definition["id"]] = material_identity(task_definition, root)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(development_reviews))
    with patch("egocua.construction.CurrentConstruction.matches", return_value=True):
        _materialize_construction_review(task_definition, root)
    for condition_name in PLAN.conditions_for(task_definition["track"]):
        condition = PLAN.condition(condition_name)
        model_actions: list[dict[str, Any]] = []
        if (
            task_definition["track"] == "graft"
            and condition.schedule == "online"
            and condition.scope == "full"
        ):
            model_actions.append(
                {
                    "type": "instruction_injection",
                    "step": 2,
                    "trigger_reason": "task_timeline",
                }
            )
        elif (
            task_definition["track"] == "refresh"
            and condition.schedule == "online"
            and condition.scope == "full"
        ):
            model_actions.append(
                {
                    "type": "visual_context_update",
                    "step": 2,
                    "trigger_reason": "task_timeline",
                }
            )
        meta = _run_meta(
            build_run_meta(
                task_definition,
                asset_root=root,
                agent=PLAN.adapter,
                model=PLAN.model,
                provider="fixture-provider",
                model_version=PLAN.model_version,
                transport="fixture-transport",
                replicate_id=0,
                provider_seed=None,
                max_steps=PLAN.max_steps,
            )
        )
        meta["schedule_index"] = None
        record = {
            "task_id": task_definition["id"],
            "task_snapshot": deepcopy(task_definition),
            "track": task_definition["track"],
            "condition": condition.as_dict(),
            "run_status": {"valid": True, "reason": None},
            "steps": 8,
            "actions": model_actions,
            "final_state": {
                "selected_entity_ids": [],
                "fields": {},
                "submission_completed": True,
            },
            "meta": meta,
        }
        artifact = (
            root
            / "artifacts"
            / "runs"
            / "fixture"
            / task_definition["track"]
            / condition_name
            / "0"
        )
        _write_run_artifact(artifact, condition_name, record)


def _materialize_source_lineage(task_definition: dict[str, Any], root: Path) -> None:
    track = task_definition["track"]
    dossier_reference = f"data/source/{track}_screening_manifest.json"
    provenance_reference = f"data/source/{track}_frame_provenance.json"
    source_packets: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    packet_sampling: list[dict[str, Any]] = []
    for packet in task_definition["visual_packets"]:
        timestamps = [float(index + 1) for index in range(len(packet["frame_paths"]))]
        source_packets.append(
            {
                "id": packet["id"],
                "timestamps_seconds": timestamps,
                "frame_paths": list(packet["frame_paths"]),
            }
        )
        packet_sampling.append(
            {
                "task_id": task_definition["id"],
                "packet_id": packet["id"],
                "sampling_method": "uniform_inclusive_endpoints",
                "frame_count": len(timestamps),
                "start_seconds": timestamps[0],
                "end_seconds": timestamps[-1],
                "inter_frame_seconds": 1.0,
                "effective_hz": 1.0,
            }
        )
        for index, (timestamp, frame_path) in enumerate(
            zip(timestamps, packet["frame_paths"], strict=True)
        ):
            frame_records.append(
                {
                    "task_id": task_definition["id"],
                    "packet_id": packet["id"],
                    "frame_index": index,
                    "media_id": f"fixture-{track}",
                    "timestamp_seconds": timestamp,
                    "media_timestamp_seconds": timestamp,
                    "frame_path": frame_path,
                    "sha256": hashlib.sha256(
                        (root / frame_path).read_bytes()
                    ).hexdigest(),
                }
            )
    record = {
        "task_id": task_definition["id"],
        "status": "video_screened",
        "family": track,
        "source": {
            "record_ids": [1],
            "media_id": f"fixture-{track}",
            "original_annotations": ["Unit-test fixture."],
        },
        "packets": source_packets,
        "supported_claims": [
            {
                "text": "The released fixture frames support the test referent.",
                "supported_by": [
                    f"{packet['id']}:{index}"
                    for packet in source_packets
                    for index, _ in enumerate(packet["frame_paths"])
                ],
            }
        ],
        "unresolved": [],
        "blocking_checks": [],
        "construct": {
            "event_origin": (
                "observed"
                if track == "refresh"
                else "annotation_derived"
                if track == "clarify"
                else "counterfactual_recombination"
            ),
        },
        "rejection_reason": None,
        "stream": {
            "session_seconds": [
                source_packets[0]["timestamps_seconds"][0],
                source_packets[-1]["timestamps_seconds"][-1],
            ],
            **(
                {"update_onset_seconds": source_packets[0]["timestamps_seconds"][-1]}
                if track == "graft"
                else {}
            ),
            "review_evidence": "fixture-stream-window-review.json",
        },
    }
    dossier_path = root / dossier_reference
    dossier_path.parent.mkdir(parents=True, exist_ok=True)
    dossier_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset": {
                    "name": "fixture",
                    "source_root": "reference/fixture",
                    "license_or_access_note": "Unit-test fixture.",
                },
                "tasks": [record],
            }
        )
    )
    provenance_path = root / provenance_reference
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest": dossier_reference,
                "source_dataset": "fixture",
                "videos": {f"fixture-{track}": {"sha256": "4" * 64}},
                "packet_sampling": packet_sampling,
                "frames": frame_records,
            }
        )
    )
    task_definition["source"] = {
        "dossier": dossier_reference,
        "dossier_sha256": hashlib.sha256(dossier_path.read_bytes()).hexdigest(),
        "task_id": task_definition["id"],
        "frame_provenance": provenance_reference,
        "frame_provenance_sha256": hashlib.sha256(
            provenance_path.read_bytes()
        ).hexdigest(),
    }


def _materialize_construction_review(
    task_definition: dict[str, Any],
    root: Path,
) -> None:
    task_id = task_definition["id"]
    track = task_definition["track"]
    task_path = root / f"data/tasks/{track}/{task_id}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(json.dumps(task_definition))
    review_spec = {
        "schema_version": 1,
        "reviewer_policy": {
            "required_independent_reviews": 1,
            "applies_to": "coding_agent_low_confidence_only",
            "reviewer_must_not_see_coding_agent_output": True,
            "admit_outcome": "admit",
            "reject_outcome": "repair",
        },
        "confidence_policy": {
            "version": CONFIDENCE_POLICY_VERSION,
            "method": "blind_reconstruction_then_adversarial_falsification",
            "high_confidence_route": "automatic_construction_admission",
            "low_confidence_route": "human_review_pending",
            "high_confidence_failure_route": "repair",
            "assessment_error_route": "retry_coding_agent",
            "numeric_probability_required": False,
        },
        "checks": {
            "common": [
                {"id": "source_supported", "question": "Source supported?"},
                {"id": "observation_sufficient", "question": "Observation sufficient?"},
            ],
            "graft": [
                {"id": "update_changes_goal", "question": "Update changes goal?"}
            ],
            "refresh": [
                {"id": "refresh_policy_correct", "question": "Policy correct?"}
            ],
            "clarify": [
                {"id": "one_answer_is_enough", "question": "One answer enough?"}
            ],
        },
    }
    spec_path = root / "data/reviews/construction.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(review_spec))
    selected: dict[str, list[str]] = {}
    for path in (root / "data/tasks").glob("*/*.json"):
        current = json.loads(path.read_text())
        selected.setdefault(current["track"], []).append(current["id"])
    selection_path = root / "data/selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "task_set": {
                    track_name: sorted(selected.get(track_name, []))
                    for track_name in ("graft", "refresh", "clarify")
                },
            }
        )
    )
    confidence_path = root / "artifacts/reports/task-confidence.json"
    confidence_path.parent.mkdir(parents=True, exist_ok=True)
    acceptance_path = root / "artifacts/reports/task-acceptance.json"
    selected_tasks = [
        json.loads(path.read_text()) for path in (root / "data/tasks").glob("*/*.json")
    ]
    deterministic_rows = {
        current["id"]: {
            "task_sha256": task_sha256(current),
            "task_site_execution_sha256": task_site_execution_sha256(
                current, load_site_data(root, current["site"])
            ),
            "search_evidence_sha256": (
                search_evidence_sha256(
                    task_id=current["id"],
                    site=current["site"],
                    resolution=source_record(current, root=root)["construct"][
                        "catalog_resolution"
                    ],
                    root=root,
                )
                if "catalog_resolution"
                in source_record(current, root=root).get("construct", {})
                else None
            ),
            "status": "candidate_ready",
        }
        for current in selected_tasks
    }
    acceptance_path.write_text(
        json.dumps(
            {
                "schema_version": REPORTS["acceptance"].schema_version,
                "tasks": deterministic_rows,
            }
        )
    )
    smoke_rows = {
        task_id: {**row, "status": "pass"}
        for task_id, row in deterministic_rows.items()
    }
    (root / "artifacts/reports/task-smoke.json").write_text(
        json.dumps(
            {
                "schema_version": REPORTS["smoke"].schema_version,
                "tasks": smoke_rows,
            }
        )
    )
    (root / "artifacts/reports/role-recoverability.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "tasks": {
                    current["id"]: {
                        "task_sha256": task_sha256(current),
                        "status": "pass",
                    }
                    for current in selected_tasks
                },
            }
        )
    )
    model_assessment_path = (
        root / "artifacts/reports/coding-agent-task-assessments.json"
    )
    materials = CurrentReviewMaterials(root)
    model_assessment_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "prompt_version": POLICY["assessment"]["prompt_version"],
                "assessor": "skill_executing_coding_agent",
                "pass_names": ["blind_reconstruction", "adversarial_falsification"],
                "materials_policy": "blind_release_only_then_expected_state",
                "agent": "fixture-coding-agent",
                "assessments_per_task": 2,
                "results": [
                    {
                        **coding_agent_assessment(current),
                        "materials_sha256": materials.packet(current)[
                            "materials_sha256"
                        ],
                    }
                    for current in selected_tasks
                ],
            }
        )
    )
    confidence_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "policy": {
                    "source": "egocua.construct_validity.POLICY",
                    "sha256": canonical_sha256(POLICY),
                    "version": CONFIDENCE_POLICY_VERSION,
                    "admission_gates": list(ADMISSION_GATES),
                    "evidence_dimensions": list(EVIDENCE_DIMENSIONS),
                },
                "coding_agent_assessment_report": {
                    "path": "artifacts/reports/coding-agent-task-assessments.json",
                    "sha256": hashlib.sha256(
                        model_assessment_path.read_bytes()
                    ).hexdigest(),
                    "agent": "fixture-coding-agent",
                    "prompt_version": POLICY["assessment"]["prompt_version"],
                },
                "tasks": {
                    current["id"]: route_assessment(
                        task_sha256(current),
                        coding_agent_assessment(current),
                        expected_goal=current["goal"],
                    )
                    for path in (root / "data/tasks").glob("*/*.json")
                    for current in [json.loads(path.read_text())]
                },
            }
        )
    )
    (root / "data/reviews/submissions").mkdir(parents=True, exist_ok=True)
    for reviewer_id in ("annotator-1",):
        workspace = ReviewWorkspace.open(root, reviewer_id, task_ids=[task_id])
        case = workspace.view(task_id)
        workspace.submit(
            task_id,
            {"decision": "admit", "note": ""},
            expected_revision=case["revision"],
        )


def materialize_task_set(tasks: list[dict[str, Any]], root: Path) -> Path:
    task_root = root / "tasks"
    for task_definition in tasks:
        path = task_root / task_definition["track"] / f"{task_definition['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(task_definition))
    return task_root


def live_corpus_empty() -> bool:
    """Return whether the checkout has no executable tasks."""

    root = Path(__file__).resolve().parents[1]
    return not any((root / "data" / "tasks").rglob("*.json"))


ARCHIVED_CORPUS = "this checkout has no executable task corpus"
