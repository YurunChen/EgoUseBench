from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from egocua.confidence import (
    confidence_review_required,
    high_confidence_sample,
    resolve_human_review,
    route_assessment,
    unavailable_assessment_confidence,
)
from egocua.confidence_packets import MATERIALS_HASH_POLICY, answer_boundary
from egocua.construct_validity import CONFIDENCE_POLICY_VERSION, COUNTEREXAMPLE_CHECKS
from egocua.run_ledger import canonical_sha256, task_sha256
from tests.helpers import task
from egocua import confidence_report as build_task_confidence


def _sample_population(count: int) -> tuple[dict, dict]:
    definitions = {}
    rows = {}
    for index in range(count):
        definition = task(("graft", "refresh", "clarify")[index % 3])
        definition["id"] = f"sample-{index}"
        definition["site"] = ("egomart", "egomedia", "egomaps")[(index // 3) % 3]
        definitions[definition["id"]] = definition
        rows[definition["id"]] = {
            "level": "high",
            "route": "automatic_construction_admission",
            "human_review_required": False,
            "policy_version": CONFIDENCE_POLICY_VERSION,
            "task_sha256": task_sha256(definition),
        }
    return definitions, rows


@pytest.mark.parametrize(
    ("population", "expected"), [(0, 0), (4, 4), (30, 10), (101, 21)]
)
def test_high_confidence_sample_has_an_independent_budget(
    population: int, expected: int
) -> None:
    definitions, rows = _sample_population(population)
    baseline = high_confidence_sample(definitions, rows)
    for index in range(102):
        definition = task("clarify")
        definition["id"] = f"low-{index}"
        definitions[definition["id"]] = definition
        rows[definition["id"]] = {"level": "low"}
    result = high_confidence_sample(definitions, rows)
    assert result == baseline
    assert result["population"] == population
    assert result["sample_size"] == expected
    assert len(set(result["task_ids"])) == expected


def test_high_confidence_sample_covers_strata_and_ignores_input_order() -> None:
    definitions, rows = _sample_population(90)
    result = high_confidence_sample(definitions, rows)
    strata = {
        (definitions[key]["track"], definitions[key]["site"])
        for key in result["task_ids"]
    }
    assert len(strata) == 9
    assert (
        high_confidence_sample(dict(reversed(list(definitions.items()))), rows)
        == result
    )


def test_high_confidence_sample_does_not_reuse_a_stale_task_decision() -> None:
    definitions, rows = _sample_population(2)
    definitions["sample-0"]["instructions"]["a"] = "A new request"
    result = high_confidence_sample(definitions, rows)
    assert result["task_ids"] == ["sample-1"]
    assert rows["sample-1"]["human_review_required"] is False


def test_quality_control_approval_does_not_create_an_admission_requirement() -> None:
    definition = task("clarify")
    row = route_assessment(task_sha256(definition), _result(definition, _verdict()))
    admitted = resolve_human_review(
        row, {"status": "admit", "purpose": "quality_control"}
    )
    assert admitted["level"] == "high"
    assert not confidence_review_required(admitted)
    assert admitted["quality_control_status"] == "admit"
    assert "quality_control_status" not in resolve_human_review(
        row, {"status": "admit"}
    )
    rejected = resolve_human_review(row, {"status": "reject"})
    assert rejected["route"] == "repair"
    assert not confidence_review_required(rejected)


def _verdict(*, decision: str = "admit", confidence: str = "high") -> dict:
    verdict = {
        "decision": decision,
        "confidence": confidence,
        "gates": {
            "visual_evidence": "pass",
            "causal_contract": "pass",
            "information_partition": "pass",
            "site_mapping": "pass",
            "answer_boundary": "pass",
            "user_instruction": "pass",
        },
        "uncertainty_reasons": [],
        "evidence": {
            "visual_evidence": ["frame:V0:0"],
            "causal_contract": ["instruction:initial", "event:update"],
            "information_partition": ["counterfactual:no_scene"],
            "site_mapping": ["entity:task-specific"],
            "answer_boundary": ["window:complete-equivalent-closure"],
            "user_instruction": ["instruction:released"],
        },
        "issues": [],
    }
    if decision == "repair":
        verdict["issues"] = ["A concrete semantic gate failed."]
    return verdict


def _result(definition: dict, verdict: dict) -> dict:
    assessments: list[dict[str, Any]] = [
        {
            "index": index,
            "pass": ("blind_reconstruction", "adversarial_falsification")[index],
            "status": "completed",
            "verdict": deepcopy(verdict),
        }
        for index in range(2)
    ]
    assessments[0]["verdict"]["derived_final_state"] = deepcopy(
        definition["goal"]["acceptable_states"][0]
    )
    assessments[1]["verdict"]["counterexample_checks"] = {
        name: "pass" for name in COUNTEREXAMPLE_CHECKS
    }
    return {
        "task_id": definition["id"],
        "task_sha256": task_sha256(definition),
        "answer_boundary": answer_boundary(definition),
        "status": "completed",
        "assessments": assessments,
    }


def _resolved_review(definition: dict) -> dict:
    result = _result(definition, _verdict())
    blind, second = (item["verdict"] for item in result["assessments"])
    blind.update(decision="repair", confidence="low")
    blind["gates"]["visual_evidence"] = "uncertain"
    blind["uncertainty_reasons"] = [
        "The pan body colour is unclear in the sampled frames."
    ]
    blind["issues"] = ["Continuous frames remain unchecked."]
    second["reviewer"] = {
        "agent_id": "independent",
        "context_id": "independent-context",
    }
    second["evidence"]["visual_evidence"] = [
        "Frames 29-34 show a consistently charcoal pan body; the warm edge is reflection."
    ]
    second["blind_resolutions"] = [
        {
            "concern": concern,
            "original": original,
            "evidence": ["evidence.visual_evidence.0"],
            "reason": "The continuous sequence resolves body colour and completes inspection.",
        }
        for concern, original in {
            "gates.visual_evidence": "uncertain",
            "uncertainty_reasons.0": blind["uncertainty_reasons"][0],
            "issues.0": blind["issues"][0],
        }.items()
    ]
    return result


@pytest.mark.parametrize("fault", ["none", "wrong_blind", "uncertain_counterexample"])
def test_human_route_preserves_all_distinct_unresolved_questions(fault):
    definition = task("graft")
    result = _result(definition, _verdict(confidence="low"))
    blind, second = (phase["verdict"] for phase in result["assessments"])
    blind["uncertainty_reasons"] = ["The package count is unreadable."]
    second["uncertainty_reasons"] = [
        "The package count is unreadable.",
        "The other listing's material remains unverified.",
    ]
    if fault == "wrong_blind":
        blind["derived_final_state"]["selected_entity_ids"] = ["wrong"]
    elif fault == "uncertain_counterexample":
        second["counterexample_checks"]["undeclared_equivalent"] = "uncertain"

    routed = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )

    assert routed["route"] == "human_review_pending"
    assert "The package count is unreadable." in routed["uncertainty_reasons"]
    assert (
        "The other listing's material remains unverified."
        in routed["uncertainty_reasons"]
    )
    assert routed["uncertainty_reasons"].count("The package count is unreadable.") == 1


def test_explicit_second_pass_resolves_blind_uncertainty_without_rewriting_it():
    definition = task("clarify")
    result = _resolved_review(definition)
    before = deepcopy(result)
    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )
    assert row["level"] == "high"
    assert set(row["admission_gates"].values()) == {"pass"}
    assert result == before
    assert row["assessment_count"] == 2


def test_partial_resolutions_remove_only_the_resolved_reasons_and_gate():
    definition = task("clarify")
    result = _resolved_review(definition)
    blind, second = (entry["verdict"] for entry in result["assessments"])
    second["blind_resolutions"].pop()
    second["gates"]["answer_boundary"] = "uncertain"
    second["uncertainty_reasons"] = ["A different listing's package is unreadable."]
    before = deepcopy(result)

    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )

    assert row["level"] == "low"
    assert row["admission_gates"]["effectiveness"] == "uncertain"
    assert blind["uncertainty_reasons"][0] not in row["uncertainty_reasons"]
    assert blind["issues"][0] in row["uncertainty_reasons"]
    assert second["uncertainty_reasons"][0] in row["uncertainty_reasons"]
    assert result == before


def test_partial_resolution_does_not_change_a_wrong_primary():
    definition = task("clarify")
    result = _resolved_review(definition)
    blind = result["assessments"][0]["verdict"]
    blind["derived_final_state"]["selected_entity_ids"] = ["wrong"]
    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )
    assert row["level"] == "low"
    assert blind["uncertainty_reasons"][0] in row["uncertainty_reasons"]


def test_resolved_blind_is_not_reopened_by_a_different_second_pass_issue():
    definition = task("clarify")
    result = _resolved_review(definition)
    second = result["assessments"][1]["verdict"]
    second["issues"] = ["A newly added listing's material remains unverified."]
    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )
    assert row["level"] == "low"
    assert row["uncertainty_reasons"] == second["issues"]


@pytest.mark.parametrize(
    ("actions", "route", "human"),
    [
        (["repair"], "repair", False),
        (["collect_evidence"], "evidence_collection_pending", False),
        (["human_review"], "human_review_pending", True),
        (["human_review", "collect_evidence"], "evidence_collection_pending", False),
        (["collect_evidence", "repair"], "repair", False),
    ],
)
def test_named_follow_up_routes_evidence_without_interpreting_issue_text(
    actions, route, human
):
    definition = task("clarify")
    result = _result(definition, _verdict(confidence="low"))
    second = result["assessments"][1]["verdict"]
    second["reviewer"] = {"agent_id": "reviewer", "context_id": "current"}
    second["follow_up"] = [
        {
            "action": action,
            "evidence": ["evidence.answer_boundary.0"],
            "reason": f"Specific {action} evidence.",
        }
        for action in actions
    ]
    if "repair" in actions:
        second["counterexample_checks"]["undeclared_equivalent"] = "fail"

    row = route_assessment(task_sha256(definition), result)

    assert row["route"] == route
    assert confidence_review_required(row) is human
    assert row["follow_up"] == second["follow_up"]
    if not human:
        assert resolve_human_review(row, {"status": "admit"}) == row


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "unknown_action",
        "missing_reviewer",
        "blind_reference",
        "missing_reference",
    ],
)
def test_malformed_follow_up_does_not_route_or_admit(fault):
    definition = task("clarify")
    result = _result(definition, _verdict())
    second = result["assessments"][1]["verdict"]
    second["reviewer"] = {"agent_id": "reviewer", "context_id": "current"}
    second["follow_up"] = [
        {
            "action": "repair",
            "reason": "A supported ending is rejected.",
            "evidence": ["evidence.answer_boundary.0"],
        }
    ]
    if fault == "empty":
        second["follow_up"] = []
    elif fault == "unknown_action":
        second["follow_up"][0]["action"] = "guess_from_chinese"
    elif fault == "missing_reviewer":
        second.pop("reviewer")
    elif fault == "blind_reference":
        second["follow_up"][0]["evidence"] = [
            "blind_reconstruction.evidence.answer_boundary.0"
        ]
    else:
        second["follow_up"][0]["evidence"] = ["evidence.answer_boundary.9"]
    assert route_assessment(task_sha256(definition), result)["level"] == "unassessed"


@pytest.mark.parametrize(
    "case", ["legal_alternatives", "one_wrong", "empty", "malformed"]
)
def test_resolved_blind_alternatives_must_all_be_complete_legal_endings(case):
    definition = task("clarify")
    expected = definition["goal"]["acceptable_states"]
    expected.append({"selected_entity_ids": ["alternative"], "fields": {}})
    result = _resolved_review(definition)
    endings = deepcopy(expected)
    if case == "one_wrong":
        endings.append({"selected_entity_ids": ["wrong"], "fields": {}})
    elif case == "empty":
        endings = []
    elif case == "malformed":
        endings.append({"fields": {}})
    result["assessments"][0]["verdict"]["derived_final_state"] = {
        "acceptable_states": endings,
        "semantics": "Each state is an alternative complete ending, not a combined cart.",
    }
    before = deepcopy(result)
    row = route_assessment(task_sha256(definition), result, expected_goal={"acceptable_states": expected})
    assert (row["level"] == "high") is (case == "legal_alternatives")
    assert result == before


@pytest.mark.parametrize("required_fields", [{}, {"destination": "museum"}])
def test_review_can_omit_only_empty_fields_without_changing_its_original(
    required_fields,
):
    definition = task("clarify")
    definition["goal"]["acceptable_states"][0]["fields"] = required_fields
    result = _resolved_review(definition)
    result["assessments"][0]["verdict"]["derived_final_state"].pop("fields")
    before = deepcopy(result)

    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )

    assert (row["level"] == "high") is (not required_fields)
    assert result == before


@pytest.mark.parametrize(
    "problem",
    [
        "no_resolution",
        "partial",
        "wrong_blind_state",
        "missing_blind_state",
        "missing_expected_states",
        "remaining_issue",
        "failed_counterexample",
        "low_second",
        "false_reference",
        "original_mismatch",
        "reference_to_blind",
        "wrong_reviewer",
    ],
)
def test_concern_resolution_does_not_bypass_unresolved_evidence(problem):
    definition = task("clarify")
    result = _resolved_review(definition)
    blind, second = (item["verdict"] for item in result["assessments"])
    states = definition["goal"]["acceptable_states"]
    resolutions = second["blind_resolutions"]
    if problem == "no_resolution":
        second.pop("blind_resolutions")
    elif problem == "partial":
        resolutions.pop()
    elif problem == "wrong_blind_state":
        blind["derived_final_state"]["selected_entity_ids"] = ["wrong"]
    elif problem == "missing_blind_state":
        blind.pop("derived_final_state")
    elif problem == "missing_expected_states":
        states = None
    elif problem == "remaining_issue":
        second["issues"] = ["A competing colour remains unresolved."]
    elif problem == "failed_counterexample":
        second["counterexample_checks"]["undeclared_equivalent"] = "fail"
    elif problem == "low_second":
        second["confidence"] = "low"
    elif problem == "false_reference":
        resolutions[0]["evidence"] = ["evidence.visual_evidence.90"]
    elif problem == "original_mismatch":
        resolutions[0]["original"] = "pass"
    elif problem == "reference_to_blind":
        second["evidence"]["visual_evidence"][0] = {
            "ref": "blind_reconstruction.evidence.visual_evidence"
        }
    elif problem == "wrong_reviewer":
        result["resolution_reviews"] = [
            {"reviewer": {}, "blind_resolutions": resolutions}
        ]
    row = route_assessment(task_sha256(definition), result, expected_goal={"acceptable_states": states})
    assert row["level"] != "high"


def test_latest_resolution_review_controls_current_concerns_and_preserves_history():
    definition = task("clarify")
    result = _resolved_review(definition)
    second = result["assessments"][1]["verdict"]
    review = {
        "reviewer": second["reviewer"],
        "blind_resolutions": second.pop("blind_resolutions"),
    }
    result["resolution_reviews"] = [review]
    assert (
        route_assessment(
            task_sha256(definition),
            result,
            expected_goal=definition["goal"],
        )["level"]
        == "high"
    )
    newer = deepcopy(review)
    newer["blind_resolutions"].pop()
    result["resolution_reviews"].append(newer)
    assert (
        route_assessment(
            task_sha256(definition),
            result,
            expected_goal=definition["goal"],
        )["level"]
        == "low"
    )
    assert len(result["resolution_reviews"][0]["blind_resolutions"]) == 3


def test_high_confidence_model_admit_skips_human_review() -> None:
    definition = task("clarify")
    row = route_assessment(
        task_sha256(definition),
        _result(definition, _verdict()),
    )

    assert row["level"] == "high"
    assert row["route"] == "automatic_construction_admission"
    assert row["admission_gates"] == {
        "utility": "pass",
        "effectiveness": "pass",
        "continuous_change": "pass",
        "verifiability": "pass",
    }
    assert confidence_review_required(row) is False


def test_two_phases_can_share_concrete_evidence_and_one_reviewer() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    blind, adversarial = (entry["verdict"] for entry in result["assessments"])
    blind["evidence"]["user_instruction"] = [{"ref": "evidence.causal_contract"}]
    adversarial["evidence"] = {
        name: [{"ref": f"blind_reconstruction.evidence.{name}"}]
        for name in blind["evidence"]
    }
    for verdict in (blind, adversarial):
        verdict["reviewer"] = {
            "agent_id": "independent-reviewer",
            "context_id": "fresh-context",
        }

    row = route_assessment(task_sha256(definition), result)

    assert row["route"] == "automatic_construction_admission"


def test_legacy_review_without_identity_remains_reusable_without_inventing_identity() -> (
    None
):
    definition = task("clarify")
    result = _result(definition, _verdict())
    original = deepcopy(result)

    row = route_assessment(task_sha256(definition), result)

    assert row["route"] == "automatic_construction_admission"
    assert result == original
    assert all("reviewer" not in entry["verdict"] for entry in result["assessments"])


@pytest.mark.parametrize(
    ("evidence_case", "reason"),
    [
        ("missing", "未找到已完成的两阶段具体评语"),
        ("historical", "尚无依据证明其审核材料与当前材料等价"),
        ("invalid", "评语有效性或答案边界尚未通过核验"),
        ("reopened", "已明确重开或正在进行"),
    ],
)
def test_unavailable_review_distinguishes_missing_evidence_from_unverified_materials(
    evidence_case: str, reason: str
) -> None:
    definition = task("clarify")
    historical = _result(definition, _verdict())
    historical["materials_sha256"] = "old-materials"
    current = {
        "task_id": definition["id"],
        "status": "pending",
        "pending_reason": "review_materials_changed",
        "assessments": [],
        "history": [] if evidence_case == "missing" else [historical],
    }
    if evidence_case == "reopened":
        current["pending_reason"] = "explicit_reassessment"
    material_hash = (
        "old-materials" if evidence_case == "invalid" else "current-materials"
    )
    original = deepcopy(current)

    row = unavailable_assessment_confidence(
        definition, current, {"materials_sha256": material_hash}
    )

    assert row["level"] == "unassessed"
    assert row["route"] == "coding_agent_retry_pending"
    assert reason in row["assessment_errors"][0]
    assert current == original
    if evidence_case == "missing":
        assert "historical_assessments" not in row
    else:
        assert row["historical_assessments"][0][
            "assessments_sha256"
        ] == canonical_sha256(historical["assessments"])


@pytest.mark.parametrize(
    "reference",
    [
        "evidence.unknown",
        "expected_states",
        "blind_reconstruction.evidence.visual_evidence",
    ],
)
def test_blind_phase_rejects_missing_or_answer_phase_evidence_references(
    reference,
) -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    result["assessments"][0]["verdict"]["evidence"]["visual_evidence"] = [
        {"ref": reference}
    ]

    assert route_assessment(task_sha256(definition), result)["level"] == "unassessed"


def test_evidence_reference_cycle_is_not_accepted() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    evidence = result["assessments"][0]["verdict"]["evidence"]
    evidence["visual_evidence"] = [{"ref": "evidence.site_mapping"}]
    evidence["site_mapping"] = [{"ref": "evidence.visual_evidence"}]

    assert route_assessment(task_sha256(definition), result)["level"] == "unassessed"


def test_reviewer_identity_cannot_disguise_copied_assessment_body() -> None:
    first = _result(task("clarify"), _verdict())
    second = deepcopy(first)
    second["task_id"] = "different-task"
    for index, result in enumerate((first, second)):
        for assessment in result["assessments"]:
            assessment["verdict"]["reviewer"] = {
                "agent_id": f"reviewer-{index}",
                "context_id": f"context-{index}",
            }
    first["history"] = [deepcopy(first)]

    counts = build_task_confidence.assessment_fingerprints([first, second])

    assert counts[canonical_sha256(first["assessments"])] == 2
    assert counts[canonical_sha256(second["assessments"])] == 2


def test_reused_evidence_does_not_override_counterexample_failure() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    second = result["assessments"][1]["verdict"]
    second["evidence"]["visual_evidence"] = [
        {"ref": "blind_reconstruction.evidence.visual_evidence"}
    ]
    second["counterexample_checks"]["no_scene"] = "fail"

    assert (
        route_assessment(task_sha256(definition), result)["route"]
        == "coding_agent_retry_pending"
    )


def test_model_uncertainty_is_the_only_human_review_route() -> None:
    definition = task("clarify")
    verdict = _verdict(confidence="low")
    verdict["gates"]["site_mapping"] = "uncertain"
    verdict["uncertainty_reasons"] = ["The exact site entity may not match the scene."]

    row = route_assessment(
        task_sha256(definition),
        _result(definition, verdict),
    )

    assert row["level"] == "low"
    assert row["route"] == "human_review_pending"
    assert row["human_review_required"] is True
    assert confidence_review_required(row) is True


@pytest.mark.parametrize("status", ["pending", "admit", "reject", "invalid"])
def test_human_review_resolves_current_semantic_uncertainty(status: str) -> None:
    definition = task("clarify")
    verdict = _verdict(confidence="low")
    verdict["uncertainty_reasons"] = ["The public description is ambiguous."]
    row = route_assessment(task_sha256(definition), _result(definition, verdict))

    resolved = resolve_human_review(row, {"status": status})

    expected = {
        "pending": ("human_review_pending", True),
        "admit": ("human_review_complete", False),
        "reject": ("repair", False),
        "invalid": ("human_review_pending", True),
    }
    assert (resolved["route"], resolved["human_review_required"]) == expected[status]
    assert confidence_review_required(resolved) is expected[status][1]


def test_high_confidence_model_failure_goes_to_repair_not_human_review() -> None:
    definition = task("clarify")
    verdict = _verdict(decision="repair")
    verdict["gates"]["visual_evidence"] = "fail"

    row = route_assessment(
        task_sha256(definition),
        _result(definition, verdict),
    )

    assert row["level"] == "rejected"
    assert row["route"] == "repair"


def test_high_confidence_repair_ignores_unrelated_unresolved_counterexamples() -> None:
    definition = task("clarify")
    verdict = _verdict(decision="repair")
    verdict["gates"]["visual_evidence"] = "fail"
    result = _result(definition, verdict)
    result["assessments"][1]["verdict"]["counterexample_checks"][
        "undeclared_equivalent"
    ] = "uncertain"

    row = route_assessment(
        task_sha256(definition),
        result,
    )

    assert row["level"] == "rejected"
    assert row["agent_verdict"] == "repair"
    assert row["agent_confidence"] == "high"
    assert row["human_review_required"] is False
    assert row["route"] == "repair"
    assert confidence_review_required(row) is False


def test_assessment_is_not_invalidated_by_task_hash_change() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    result["task_sha256"] = "0" * 64

    row = route_assessment(task_sha256(definition), result)

    assert row["level"] == "high"
    assert row["route"] == "automatic_construction_admission"


def test_shared_high_confidence_failure_survives_documented_other_uncertainty() -> None:
    definition = task("refresh")
    verdict = _verdict(decision="repair")
    verdict["gates"]["causal_contract"] = "fail"
    result = _result(definition, verdict)
    adversarial = result["assessments"][1]["verdict"]
    adversarial["counterexample_checks"]["early_commit"] = "uncertain"
    adversarial["uncertainty_reasons"] = [
        "The full timing replay is unverified; the absent physical change is certain."
    ]

    row = route_assessment(task_sha256(definition), result)

    assert row["route"] == "repair"
    assert row["human_review_required"] is False
    assert row["admission_gates"]["continuous_change"] == "fail"


def test_two_repairs_without_a_shared_failed_gate_are_not_consensus() -> None:
    definition = task("refresh")
    verdict = _verdict(decision="repair")
    verdict["gates"]["causal_contract"] = "fail"
    result = _result(definition, verdict)
    adversarial = result["assessments"][1]["verdict"]
    adversarial["gates"]["causal_contract"] = "pass"
    adversarial["gates"]["site_mapping"] = "fail"

    row = route_assessment(task_sha256(definition), result)

    assert row["route"] == "repair"
    assert row["human_review_required"] is False


def test_assessment_is_not_bound_to_review_packet_hash() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    result["review_packet"] = {"sha256": "old-review-packet"}

    row = route_assessment(
        task_sha256(definition),
        result,
    )

    assert row["level"] == "high"
    assert row["route"] == "automatic_construction_admission"


def test_answer_bearing_site_audit_waits_for_retry() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    result["site_audit"] = {"sha256": "answer-bearing-render"}

    row = route_assessment(task_sha256(definition), result)

    assert row["route"] == "coding_agent_retry_pending"
    assert row["assessment_errors"] == [
        "assessment record uses an answer-bearing site audit"
    ]


def test_coding_agent_call_failure_waits_for_retry() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    result["assessments"][1] = {
        "index": 1,
        "status": "error",
        "error": "insufficient balance",
    }

    row = route_assessment(task_sha256(definition), result)

    assert row["level"] == "unassessed"
    assert row["route"] == "coding_agent_retry_pending"
    assert row["human_review_required"] is False


def test_reused_assessment_body_waits_for_coding_agent_retry() -> None:
    definition = task("clarify")
    row = route_assessment(
        task_sha256(definition),
        _result(definition, _verdict()),
        assessment_body_reused=True,
    )

    assert row["level"] == "unassessed"
    assert row["route"] == "coding_agent_retry_pending"


def test_answer_bearing_roles_are_not_valid_blind_materials() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    result["proposed_entities"] = [{"roles": ["final_acceptable"]}]

    row = route_assessment(task_sha256(definition), result)

    assert row["level"] == "unassessed"
    assert row["route"] == "coding_agent_retry_pending"


def test_blind_reconstruction_mismatch_routes_to_human_review() -> None:
    definition = task("clarify")
    result = _result(definition, _verdict())
    result["assessments"][0]["verdict"]["derived_final_state"] = {
        "selected_entity_ids": ["wrong"],
        "fields": {},
        "submission_completed": True,
    }

    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )

    assert row["level"] == "low"
    assert row["route"] == "human_review_pending"
    assert row["human_review_required"] is True


@pytest.mark.parametrize(
    "acceptance_status", ["candidate_ready", "needs_source_review"]
)
@pytest.mark.parametrize("stale_input", ["acceptance", "review_packet"])
def test_builder_uses_model_report_as_confidence_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acceptance_status: str,
    stale_input: str,
) -> None:
    definitions = []
    for index in range(3):
        definition = deepcopy(task("clarify"))
        definition["id"] = f"model-route-{index}"
        definitions.append(definition)
    acceptance_path = tmp_path / "artifacts/reports/task-acceptance.json"
    acceptance_path.parent.mkdir(parents=True)
    acceptance_path.write_text(
        json.dumps(
            {
                "tasks": {
                    definition["id"]: {
                        "task_sha256": task_sha256(definition),
                        "status": acceptance_status,
                        "integrity_pass": True,
                        "failed_checks": [],
                    }
                    for definition in definitions
                }
            }
        )
    )
    verdicts = [
        _verdict(),
        _verdict(confidence="low"),
        _verdict(decision="repair"),
    ]
    verdicts[1]["gates"]["site_mapping"] = "uncertain"
    verdicts[1]["uncertainty_reasons"] = ["Site mapping is unclear."]
    verdicts[2]["gates"]["causal_contract"] = "fail"
    packet_dir = tmp_path / "artifacts/reports/confidence-review-packets"
    packet_dir.mkdir()
    packet_bytes = b"blind-review-packet"
    packet_hash = hashlib.sha256(packet_bytes).hexdigest()
    results = [
        _result(definition, verdict)
        for definition, verdict in zip(definitions, verdicts, strict=True)
    ]
    packet_rows = {}
    for definition, result in zip(definitions, results, strict=True):
        packet_path = packet_dir / f"{definition['id']}.json"
        packet_path.write_bytes(packet_bytes)
        packet_rows[definition["id"]] = {
            "path": packet_path.relative_to(tmp_path).as_posix(),
            "sha256": packet_hash,
            "task_sha256": task_sha256(definition),
            "materials_sha256": "a" * 64,
        }
        result["materials_sha256"] = "a" * 64
    (packet_dir / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "materials_policy": "blind_release_only_then_expected_state",
                "materials_hash_policy": MATERIALS_HASH_POLICY,
                "tasks": packet_rows,
            }
        )
    )
    model_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    model_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "prompt_version": "task-confidence-blind-reconstruction-v4",
                "assessor": "skill_executing_coding_agent",
                "pass_names": ["blind_reconstruction", "adversarial_falsification"],
                "materials_policy": "blind_release_only_then_expected_state",
                "agent": "fixture-coding-agent",
                "assessments_per_task": 2,
                "refresh_policy": "explicit_task_ids",
                "freshness_policy": MATERIALS_HASH_POLICY,
                "results": results,
            }
        )
    )
    monkeypatch.setattr(build_task_confidence, "load_construction_tasks", lambda _, task_ids=None: definitions)
    current_hashes = {definition["id"]: "a" * 64 for definition in definitions}
    monkeypatch.setattr(
        build_task_confidence.CurrentReviewMaterials,
        "packet",
        lambda self, definition: {"materials_sha256": current_hashes[definition["id"]]},
    )
    monkeypatch.setattr(
        build_task_confidence,
        "audit_construction_reviews",
        lambda **_: {
            "tasks": {
                definition["id"]: {"status": "pending"} for definition in definitions
            }
        },
    )

    report = build_task_confidence.build_confidence_report(tmp_path)

    assert report["summary"]["levels"] == {
        "high": 1,
        "low": 1,
        "rejected": 1,
        "unassessed": 0,
    }
    assert report["summary"]["human_review_queue"] == 1
    assert report["summary"]["coding_agent_retry_queue"] == 0
    assert report["quality_control"]["task_ids"] == [definitions[0]["id"]]
    assert report["quality_control"]["population"] == 1
    assert (
        report["coding_agent_assessment_report"]["sha256"]
        == hashlib.sha256(model_path.read_bytes()).hexdigest()
    )

    changed_id = definitions[0]["id"]
    input_path = (
        acceptance_path if stale_input == "acceptance" else packet_dir / "index.json"
    )
    stale_payload = json.loads(input_path.read_text())
    stale_payload["tasks"][changed_id]["task_sha256"] = "0" * 64
    input_path.write_text(json.dumps(stale_payload))
    if stale_input == "review_packet":
        current_hashes[changed_id] = "changed-public-content"
    original_assessments = model_path.read_bytes()

    refreshed = build_task_confidence.build_confidence_report(tmp_path)

    if stale_input == "acceptance":
        assert refreshed["tasks"][changed_id] == report["tasks"][changed_id]
        assert refreshed["quality_control"] == report["quality_control"]
    else:
        assert refreshed["tasks"][changed_id]["level"] == "unassessed"
        assert refreshed["quality_control"]["task_ids"] == []
    for definition in definitions[1:]:
        assert refreshed["tasks"][definition["id"]] == report["tasks"][definition["id"]]
    assert model_path.read_bytes() == original_assessments
