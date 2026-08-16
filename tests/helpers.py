from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from egocua.experiment import ExperimentPlan, PRIMARY_CONDITION
from egocua.environment import load_site_data
from egocua.reviews import construction_evidence_from_reviews
from egocua.review_workspace import ReviewWorkspace
from egocua.run_ledger import build_run_meta, canonical_sha256
from egocua.schema import goal_for_arm


PLAN = ExperimentPlan(
    "test-plan",
    {
        "graft": ("a_only", "bundled", "injected"),
        "refresh": ("stale", "stream", "oracle_latest"),
        "clarify": ("oracle", "interactive", "forced_execute"),
    },
    (0, 1, 2),
    "minimal-loop",
    "scripted-fixture",
    "scripted-fixture",
    40,
    7,
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
            "image_path": f"products/{item_id(track, index)}.jpg",
            "image_provenance": {
                "source_url": f"https://example.invalid/{item_id(track, index)}.jpg",
                "retrieved_at": "2026-08-09T00:00:00Z",
                "license_or_access_note": "Synthetic unit-test fixture; not benchmark data.",
                "sha256": hashlib.sha256(f"asset:{item_id(track, index)}".encode()).hexdigest(),
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
            {**product, "entity_id": product["item_id"]}
            for product in products(track)
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
        base.update({
            "instructions": {"a": "Find A", "b": "Also apply B", "bundled": "Find A and apply B"},
            "visual_packets": [{"id": "V0", "frame_paths": [f"frames/graft-{i}.jpg" for i in range(8)]}],
            "graft": {
                "integration_type": "C",
                "initial_goal": {
                    "target_granularity": "variant",
                    "acceptable_states": [
                        {"selected_entity_ids": [item_id("graft", index)], "fields": {}, "submission_completed": True}
                        for index in (0, 1)
                    ],
                },
            },
            "goal": {
                "target_granularity": "variant",
                "acceptable_states": [{"selected_entity_ids": [item_id("graft", 0)], "fields": {}, "submission_completed": True}],
            },
        })
    elif track == "refresh":
        base.update({
            "instructions": {"base": "Find the item selected last"},
            "visual_packets": [
                {"id": "V0", "frame_paths": [f"frames/refresh-v0-{i}.jpg" for i in range(4)]},
                {"id": "V1", "frame_paths": [f"frames/refresh-v1-{i}.jpg" for i in range(4)]},
            ],
            "refresh": {"policy": "replace", "initial_entity_id": item_id("refresh", 0)},
            "goal": {
                "target_granularity": "variant",
                "acceptable_states": [{"selected_entity_ids": [item_id("refresh", 1)], "fields": {}, "submission_completed": True}],
            },
        })
    else:
        base.update({
            "instructions": {"ambiguous": "Find this product", "oracle": "Find the target variant"},
            "visual_packets": [{"id": "V0", "frame_paths": [f"frames/clarify-{i}.jpg" for i in range(8)]}],
            "clarification": {
                "candidate_ids": [item_id("clarify", 0), item_id("clarify", 1)],
                "valid_slot": "variant",
                "canonical_answer": "The target variant.",
                "max_questions": 1,
            },
            "goal": {
                "target_granularity": "variant",
                "acceptable_states": [{"selected_entity_ids": [item_id("clarify", 1)], "fields": {}, "submission_completed": True}],
            },
        })
    return base


def validation_evidence(
    track: str,
    status: str = "approved",
    *,
    task_definition: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    fixture = task_definition or task(track)
    reviewed_sets = fixture["goal"]["acceptable_states"]
    reference_arm = PRIMARY_CONDITION[track]
    full_choice = goal_for_arm(fixture, reference_arm)["acceptable_states"][0]["selected_entity_ids"]
    counterfactual = {
        "graft": {"b_is_essential": True},
        "refresh": {"v0_v1_identifies_latest": True, "v0_only_identifies_latest": False},
        "clarify": {"ambiguous_without_answer": True, "unique_with_answer": True},
    }[track]
    replay_arms = {
        "alternative_correct_path": reference_arm,
        "extra_item_rejected": reference_arm,
        "duplicate_quantity_rejected": reference_arm,
    }
    if track in {"graft", "refresh"}:
        dynamic_arm = "injected" if track == "graft" else "stream"
        replay_arms.update({
            "early_submission_rejected": dynamic_arm,
            "missing_event_rejected": dynamic_arm,
        })
    else:
        replay_arms.update({
            "invalid_slot_detected": "interactive",
            "second_question_limit_detected": "interactive",
            "late_question_detected": "interactive",
            "ignored_answer_detected": "interactive",
        })
    if status in {"constructed", "human_verified", "approved"}:
        if root is None or task_definition is None:
            raise ValueError(
                "admitted validation evidence requires the materialized task and workspace root"
            )
        construction = construction_evidence_from_reviews(fixture, root=root)
    else:
        construction = {}
    return {
        "status": status,
        "construction": construction,
        "human": {
            "exact_harness": True,
            "arm_runs": {
                arm: [
                    {
                        "evaluator": evaluator,
                        "completion_seconds": 42.0 + index,
                        "run_artifact": f"artifacts/runs/human/{track}/{arm}/{evaluator}/run.json",
                    }
                    for index, evaluator in enumerate(("human-1", "human-2"))
                ]
                for arm in PLAN.arms_for(track)
            },
            "visual_necessity": {
                "reference_arm": reference_arm,
                "reviews": [
                    {
                        "reviewer": evaluator,
                        "text_only_choice_ids": [item_id(track, 0), item_id(track, 1)],
                        "full_evidence_choice_ids": full_choice,
                    }
                    for evaluator in ("human-1", "human-2")
                ],
            },
            "counterfactual_checks": [
                {"reviewer": evaluator, **counterfactual}
                for evaluator in ("human-1", "human-2")
            ],
        },
        "evaluator": {
            "reviewers": ["evaluator-reviewer-1"],
            "replay_cases": {
                case: {
                    "arm": arm,
                    "run_artifact": f"artifacts/runs/evaluator/{track}/{case}/run.json",
                }
                for case, arm in replay_arms.items()
            },
        },
        "model": {
            "execution_mode": "real_model",
            "provider": "fixture-provider",
            "model": "scripted-fixture",
            "model_version": "scripted-fixture",
            "adapter": "minimal-loop",
            "max_steps": 40,
            "exact_harness": True,
            "vision_input_used": True,
            "arm_runs": {
                arm: [{
                    "run_artifact": f"artifacts/runs/fixture/{track}/{arm}/0",
                    "protocol_valid": True,
                    "e2e_success": False,
                }]
                for arm in PLAN.arms_for(track)
            },
            "unresolved_operational_issues": [],
            "used_for_task_selection": False,
        },
    }


def materialize_site(items: list[dict[str, Any]], root: Path) -> None:
    path = root / "sites/egomart/products.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if path.is_file():
        existing = json.loads(path.read_text())["products"]
    by_id = {item["item_id"]: item for item in [*existing, *items]}
    path.write_text(json.dumps({
        "schema_version": 1,
        "site": "egomart",
        "products": list(by_id.values()),
    }))


def materialize_assets(task_definition: dict[str, Any], items: list[dict[str, Any]], root: Path) -> None:
    materialize_site(items, root)
    frozen_site = load_site_data(root, "egomart")
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
    for arm in PLAN.arms_for(task_definition["track"]):
        if (
            task_definition["track"] == "refresh"
            and task_definition["refresh"]["policy"] == "replace"
            and arm == "stale"
        ):
            selected_ids = [task_definition["refresh"]["initial_entity_id"]]
        elif task_definition["track"] == "clarify" and arm == "forced_execute":
            selected_ids = [task_definition["clarification"]["candidate_ids"][0]]
        else:
            selected_ids = goal_for_arm(task_definition, arm)["acceptable_states"][0]["selected_entity_ids"]
        actions: list[dict[str, Any]] = []
        step = 1
        if (task_definition["track"], arm) == ("graft", "injected"):
            actions.append({
                "type": "instruction_injection",
                "step": 2,
                "trigger_reason": "task_timeline",
            })
            step = 5
        elif (task_definition["track"], arm) == ("refresh", "stream"):
            actions.append({
                "type": "visual_context_update",
                "step": 2,
                "trigger_reason": "task_timeline",
            })
            step = 5
        elif (task_definition["track"], arm) == ("clarify", "interactive"):
            clarification = task_definition["clarification"]
            actions.append({
                "type": "clarification_answer",
                "step": 1,
                "slot": clarification["valid_slot"],
                "question": clarification_question(clarification["valid_slot"]),
                "valid": True,
                "payload": clarification["canonical_answer"],
            })
            step = 2
        actions.extend(
            {"type": "select", "step": step, "item_id": item}
            for item in selected_ids
        )
        human_meta = build_run_meta(
            task_definition,
            frozen_site,
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
        for evaluator in ("human-1", "human-2"):
            evaluator_meta = {**human_meta, "operator_id": evaluator}
            record = {
                "task_id": task_definition["id"],
                "track": task_definition["track"],
                "arm": arm,
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
            artifact = root / "artifacts" / "runs" / "human" / task_definition["track"] / arm / evaluator
            artifact.mkdir(parents=True, exist_ok=True)
            (artifact / "run.json").write_text(json.dumps(record))
    reference_arm = PRIMARY_CONDITION[task_definition["track"]]
    reference_ids = goal_for_arm(task_definition, reference_arm)["acceptable_states"][0]["selected_entity_ids"]
    replay_specs: dict[str, tuple[str, list[str], list[dict[str, Any]]]] = {}
    reference_actions: list[dict[str, Any]] = []
    reference_step = 1
    if task_definition["track"] == "graft":
        reference_actions.append({
            "type": "instruction_injection",
            "step": 2,
            "trigger_reason": "task_timeline",
        })
        reference_step = 5
    elif task_definition["track"] == "refresh":
        reference_actions.append({
            "type": "visual_context_update",
            "step": 2,
            "trigger_reason": "task_timeline",
        })
        reference_step = 5
    elif task_definition["track"] == "clarify":
        clarification = task_definition["clarification"]
        reference_actions.append({
            "type": "clarification_answer",
            "step": 1,
            "slot": clarification["valid_slot"],
            "question": clarification_question(clarification["valid_slot"]),
            "valid": True,
            "payload": clarification["canonical_answer"],
        })
        reference_step = 2
    reference_actions.extend(
        {"type": "select", "step": reference_step, "item_id": item}
        for item in reference_ids
    )
    replay_specs["alternative_correct_path"] = (
        reference_arm, reference_ids, reference_actions,
    )
    replay_specs["extra_item_rejected"] = (
        reference_arm,
        [*reference_ids, item_id(task_definition["track"], 15)],
        reference_actions,
    )
    replay_specs["duplicate_quantity_rejected"] = (
        reference_arm,
        [*reference_ids, reference_ids[0]],
        reference_actions,
    )
    if task_definition["track"] in {"graft", "refresh"}:
        dynamic_arm = "injected" if task_definition["track"] == "graft" else "stream"
        event_type = (
            "instruction_injection" if task_definition["track"] == "graft"
            else "visual_context_update"
        )
        dynamic_ids = goal_for_arm(task_definition, dynamic_arm)["acceptable_states"][0]["selected_entity_ids"]
        replay_specs["early_submission_rejected"] = (
            dynamic_arm,
            dynamic_ids,
            [
                {"type": "submit", "step": 1, "item_ids": dynamic_ids},
                {"type": event_type, "step": 2, "trigger_reason": "task_timeline"},
            ],
        )
        replay_specs["missing_event_rejected"] = (
            dynamic_arm,
            dynamic_ids,
            [{"type": "submit", "step": 2, "item_ids": dynamic_ids}],
        )
    else:
        target = reference_ids[0]
        wrong = next(
            candidate for candidate in task_definition["clarification"]["candidate_ids"]
            if candidate != target
        )
        answer = task_definition["clarification"]["canonical_answer"]
        slot = task_definition["clarification"]["valid_slot"]
        question = clarification_question(slot)
        wrong_slot = next(
            candidate_slot for candidate_slot in
            ("position", "appearance", "brand", "variant", "sequence")
            if candidate_slot != slot
        )
        replay_specs.update({
            "invalid_slot_detected": (
                "interactive",
                [target],
                [
                    {
                        "type": "clarification_answer", "step": 1, "slot": wrong_slot,
                        "question": clarification_question(wrong_slot), "valid": False,
                        "payload": "That question did not distinguish the candidates.",
                    },
                    {"type": "select", "step": 2, "item_id": target},
                ],
            ),
            "second_question_limit_detected": (
                "interactive",
                [target],
                [
                    {
                        "type": "clarification_answer", "step": 1, "slot": slot,
                        "question": question, "valid": True, "payload": answer,
                    },
                    {"type": "clarification_answer", "step": 2, "slot": slot, "limit_reached": True},
                    {"type": "select", "step": 3, "item_id": target},
                ],
            ),
            "late_question_detected": (
                "interactive",
                [target],
                [
                    {"type": "select", "step": 1, "item_id": target},
                    {
                        "type": "clarification_answer", "step": 2, "slot": slot,
                        "question": question, "valid": True, "payload": answer,
                    },
                ],
            ),
            "ignored_answer_detected": (
                "interactive",
                [wrong],
                [
                    {
                        "type": "clarification_answer", "step": 1, "slot": slot,
                        "question": question, "valid": True, "payload": answer,
                    },
                    {"type": "select", "step": 2, "item_id": wrong},
                ],
            ),
        })
    evaluator_meta = build_run_meta(
        task_definition,
        frozen_site,
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
    for case, (arm, selected_ids, actions) in replay_specs.items():
        record = {
            "task_id": task_definition["id"],
            "track": task_definition["track"],
            "arm": arm,
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
        artifact = root / "artifacts" / "runs" / "evaluator" / task_definition["track"] / case
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "run.json").write_text(json.dumps(record))
    _materialize_construction_review(task_definition, root)
    for arm in PLAN.arms_for(task_definition["track"]):
        model_actions: list[dict[str, Any]] = []
        if (task_definition["track"], arm) == ("graft", "injected"):
            model_actions.append({
                "type": "instruction_injection",
                "step": 2,
                "trigger_reason": "task_timeline",
            })
        elif (task_definition["track"], arm) == ("refresh", "stream"):
            model_actions.append({
                "type": "visual_context_update",
                "step": 2,
                "trigger_reason": "task_timeline",
            })
        meta = build_run_meta(
            task_definition,
            frozen_site,
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
        meta["schedule_index"] = None
        record = {
            "task_id": task_definition["id"],
            "track": task_definition["track"],
            "arm": arm,
            "run_status": {"valid": True, "reason": None},
            "steps": 8,
            "actions": model_actions,
            "final_state": {"selected_entity_ids": [], "fields": {}, "submission_completed": True},
            "meta": meta,
        }
        artifact = root / "artifacts" / "runs" / "fixture" / task_definition["track"] / arm / "0"
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "run.json").write_text(json.dumps(record))


def _materialize_source_lineage(task_definition: dict[str, Any], root: Path) -> None:
    track = task_definition["track"]
    dossier_reference = f"data/source/{track}_screening_manifest.json"
    provenance_reference = f"data/source/{track}_frame_provenance.json"
    source_packets: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    packet_sampling: list[dict[str, Any]] = []
    for packet in task_definition["visual_packets"]:
        timestamps = [float(index + 1) for index in range(len(packet["frame_paths"]))]
        source_packets.append({
            "id": packet["id"],
            "timestamps_seconds": timestamps,
            "frame_paths": list(packet["frame_paths"]),
        })
        packet_sampling.append({
            "task_id": task_definition["id"],
            "packet_id": packet["id"],
            "sampling_method": "uniform_inclusive_endpoints",
            "frame_count": len(timestamps),
            "start_seconds": timestamps[0],
            "end_seconds": timestamps[-1],
            "inter_frame_seconds": 1.0,
            "effective_hz": 1.0,
        })
        for index, (timestamp, frame_path) in enumerate(
            zip(timestamps, packet["frame_paths"], strict=True)
        ):
            frame_records.append({
                "task_id": task_definition["id"],
                "packet_id": packet["id"],
                "frame_index": index,
                "media_id": f"fixture-{track}",
                "timestamp_seconds": timestamp,
                "media_timestamp_seconds": timestamp,
                "frame_path": frame_path,
                "sha256": hashlib.sha256((root / frame_path).read_bytes()).hexdigest(),
            })
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
        "supported_claims": [{
            "text": "The released fixture frames support the test referent.",
            "supported_by": [
                f"{packet['id']}:{index}"
                for packet in source_packets
                for index, _ in enumerate(packet["frame_paths"])
            ],
        }],
        "unresolved": [],
        "blocking_checks": [],
        "construct": {
            "event_origin": (
                "observed" if track == "refresh" else
                "annotation_derived" if track == "clarify" else
                "counterfactual_recombination"
            ),
        },
        "rejection_reason": None,
    }
    dossier_path = root / dossier_reference
    dossier_path.parent.mkdir(parents=True, exist_ok=True)
    dossier_path.write_text(json.dumps({
        "schema_version": 2,
        "dataset": {
            "name": "fixture",
            "source_root": "reference/fixture",
            "license_or_access_note": "Unit-test fixture.",
        },
        "tasks": [record],
    }))
    provenance_path = root / provenance_reference
    provenance_path.write_text(json.dumps({
        "schema_version": 2,
        "manifest": dossier_reference,
        "source_dataset": "fixture",
        "videos": {f"fixture-{track}": {"sha256": "4" * 64}},
        "packet_sampling": packet_sampling,
        "frames": frame_records,
    }))
    task_definition["source"] = {
        "dossier": dossier_reference,
        "dossier_sha256": hashlib.sha256(dossier_path.read_bytes()).hexdigest(),
        "task_id": task_definition["id"],
        "record_sha256": canonical_sha256(record),
        "frame_provenance": provenance_reference,
        "frame_provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
    }


def _materialize_construction_review(
    task_definition: dict[str, Any], root: Path,
) -> None:
    task_id = task_definition["id"]
    track = task_definition["track"]
    task_path = root / f"data/tasks/{track}/{task_id}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(json.dumps(task_definition))
    review_spec = {
        "schema_version": 1,
        "reviewer_policy": {
            "required_independent_reviews": 2,
            "reviewers_must_not_see_each_others_submission": True,
            "disagreement_outcome": "reject",
            "model_output_may_influence_decision": False,
        },
        "checks": {
            "common": [
                {"id": "source_supported", "question": "Source supported?"},
                {"id": "observation_sufficient", "question": "Observation sufficient?"},
            ],
            "graft": [{"id": "update_changes_goal", "question": "Update changes goal?"}],
            "refresh": [{"id": "refresh_policy_correct", "question": "Policy correct?"}],
            "clarify": [{"id": "one_answer_is_enough", "question": "One answer enough?"}],
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
    selection_path.write_text(json.dumps({
        "schema_version": 1,
        "primary": {track_name: sorted(ids) for track_name, ids in selected.items()},
    }))
    for reviewer in ("annotator-1", "annotator-2"):
        workspace = ReviewWorkspace.open(root, reviewer)
        case = workspace.view(task_id)
        current = {"decision": "admit", "note": ""}
        workspace.submit(task_id, current, expected_revision=case["revision"])


def materialize_task_set(tasks: list[dict[str, Any]], root: Path) -> Path:
    task_root = root / "tasks"
    for task_definition in tasks:
        path = task_root / task_definition["track"] / f"{task_definition['id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(task_definition))
    return task_root
