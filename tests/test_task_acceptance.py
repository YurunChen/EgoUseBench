from __future__ import annotations

from pathlib import Path

import pytest

from egocua.schema import load_tasks
from tools.audit_task_acceptance import (
    STATUS_VALUES,
    _checker_check,
    _construct_check,
    _instruction_check,
    build_acceptance_report,
)

from .helpers import site_data, task


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_production_checker_contract_replays_every_arm_and_rejects_negatives(
    track: str,
) -> None:
    result = _checker_check(task(track), site_data(track))

    assert result["status"] == "pass"
    assert result["details"]["positive_paths"] == 3
    assert result["details"]["adversarial_paths"] >= 15


def test_instruction_check_rejects_internal_language_and_opaque_ids() -> None:
    clean = task("clarify")
    leaking = task("clarify")
    leaking["instructions"]["ambiguous"] = (
        "Use V0 and select p_0123456789ab as the correct answer."
    )

    assert _instruction_check(clean)["status"] == "pass"
    rejected = _instruction_check(leaking)
    assert rejected["status"] == "fail"
    assert rejected["details"]["leakage_matches"]["ambiguous"]


def test_construct_check_flags_static_replacement_without_claiming_human_pass() -> None:
    benchmark_task = task("clarify")
    record = {
        "source": {"original_annotations": ["The wearer handles two objects."]},
        "supported_claims": [{"text": "Two objects remain visible."}],
        "construct": {
            "research_question": "Should the Agent ask before choosing?",
            "divergence_source": "grounding",
            "event_origin": "annotation_derived",
            "belief_before": "Either candidate is possible.",
            "belief_after": "One candidate is intended.",
            "required_behavior_change": "Ask and then select.",
            "no_update_failure": "Guessing.",
            "overupdate_failure": "Asking unnecessarily.",
            "visual_dependency_claim": "The candidates come from the scene.",
            "task_blueprint": {"candidate_roles": ["a", "b"]},
        },
    }

    result = _construct_check(benchmark_task, record, is_primary=True)

    assert result["status"] == "pass"
    assert result["details"]["human_decision_required"] is True
    assert result["details"]["construct_risk_reasons"]


def test_sequence_clarification_is_not_reduced_to_one_static_photo() -> None:
    benchmark_task = task("clarify")
    benchmark_task["clarification"]["valid_slot"] = "sequence"
    record = {
        "source": {"original_annotations": ["The wearer examines two posters in order."]},
        "supported_claims": [{"text": "The first and second posters are distinct."}],
        "construct": {
            "research_question": "Should the Agent ask which occurrence?",
            "divergence_source": "grounding",
            "event_origin": "annotation_derived",
            "belief_before": "Either occurrence is possible.",
            "belief_after": "The second occurrence is intended.",
            "required_behavior_change": "Ask and then select the second.",
            "no_update_failure": "Guessing.",
            "overupdate_failure": "Asking unnecessarily.",
            "visual_dependency_claim": "The ordered scene establishes the occurrence.",
            "task_blueprint": {"candidate_roles": ["a", "b"]},
        },
    }

    result = _construct_check(benchmark_task, record, is_primary=True)

    assert result["status"] == "pass"
    assert result["details"]["construct_risk_reasons"] == []


def test_full_report_has_one_fail_closed_row_per_constructed_task() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {task["id"] for task in load_tasks(root / "data/tasks")}

    report = build_acceptance_report(root, verify_source_frames=False)

    assert set(report["tasks"]) == expected
    assert report["summary"]["tasks"] == len(expected)
    assert sum(report["summary"]["statuses"].values()) == len(expected)
    assert all(row["status"] in STATUS_VALUES for row in report["tasks"].values())
    assert all(
        row["task_acceptance_ready"] is False
        for row in report["tasks"].values()
        if row["status"] != "task_verified"
    )
    assert all(
        set(row["checks"]) == {
            "schema_and_frames",
            "source_lineage",
            "glass_use_construct_prerequisites",
            "continuous_video",
            "site_entities_images_and_answers",
            "compiled_from_source_truth",
            "production_checker",
            "instruction_form_and_leakage",
        }
        for row in report["tasks"].values()
    )
