from copy import deepcopy

import pytest

from egocua.checker import _collateral_damage
from egocua.controlled_metrics import (
    PairedComparison,
    controlled_run_metrics,
    paired_source_statistics,
    rule_baseline_scores,
)


def state(*ids, fields=None):
    return {"selected_entity_ids": list(ids), "fields": fields or {}}


def record():
    old, empty, correct = state("old", "keep"), state("keep"), state("new", "keep")
    return {
        "task_snapshot": {
            "id": "test-metrics",
            "track": "graft",
            "goal": {"acceptable_states": [correct]},
            "graft": {
                "transition": "replace",
                "initial_goal": {"acceptable_states": [old]},
            },
        },
        "run_status": {"valid": True},
        "condition": {"schedule": "online"},
        "initial_state": state(),
        "live_state": correct,
        "steps": 6,
        "initial_context": {
            "observations": [{"frame_paths": ["first.jpg", "second.jpg"]}]
        },
        "actions": [
            {
                "type": "select_entity",
                "step": 1,
                "state_before": state(),
                "state_after": old,
            },
            {
                "type": "instruction_injection",
                "step": 2,
                "payload": "replace old; keep keep",
            },
            {
                "type": "deselect_entity",
                "step": 3,
                "state_before": old,
                "state_after": empty,
            },
            {
                "type": "select_entity",
                "step": 5,
                "state_before": empty,
                "state_after": correct,
            },
        ],
    }


def test_recovery_requires_whole_state_and_counts_only_delivered_frames():
    run = record()
    run["task_snapshot"]["visual_packets"] = [{"frame_paths": ["never_seen.jpg"]}]
    result = controlled_run_metrics(run)
    assert result["delivered_frame_count"] == 2
    assert result["first_full_recovery_step"] == 5
    assert result["recovery_steps_after_release"] == 3
    assert result["regressed_after_recovery"] is False
    assert result["obsolete_residual_count"] == 0
    assert result["valid_item_loss_count"] == 0
    assert result["obsolete_residual_opportunity"] is True
    assert result["valid_item_loss_opportunity"] is True
    assert result["selection_held_at_update"] is True


def test_recovery_rejects_stale_extra_item_until_removed():
    run = record()
    intermediate = state("old", "keep", "new")
    run["actions"][2].update(type="select_entity", state_after=intermediate)
    run["actions"][3].update(type="deselect_entity", state_before=intermediate)
    assert controlled_run_metrics(run)["first_full_recovery_step"] == 5


def test_regression_is_recorded_even_when_later_repaired():
    run = record()
    correct = run["live_state"]
    run["actions"].extend(
        [
            {
                "type": "deselect_entity",
                "step": 6,
                "state_before": correct,
                "state_after": state("new"),
            },
            {
                "type": "select_entity",
                "step": 7,
                "state_before": state("new"),
                "state_after": correct,
            },
        ]
    )
    run["steps"] = 7
    assert controlled_run_metrics(run)["regressed_after_recovery"] is True


def test_wrong_quantity_or_fields_do_not_count_as_recovery():
    run = record()
    incorrect = state("new", "keep", "keep", fields={"mode": "wrong"})
    run["actions"][-1]["state_after"] = incorrect
    run["live_state"] = incorrect
    assert controlled_run_metrics(run)["recovery_status"] == "not_recovered"


def test_missing_state_does_not_fall_back_to_submission_snapshot():
    run = record()
    run["final_state"] = run.pop("live_state")
    result = controlled_run_metrics(run)
    assert result["first_full_recovery_step"] is None
    assert result["recovery_status"] == "incomplete_state_history"


def test_broken_snapshot_chain_cannot_establish_first_recovery():
    run = record()
    run["actions"][2]["state_before"] = state("not_logged")
    assert controlled_run_metrics(run)["recovery_status"] == "incomplete_state_history"


def test_invalid_run_and_update_not_delivered_are_distinct():
    run = record()
    run["run_status"]["valid"] = False
    assert controlled_run_metrics(run)["recovery_status"] == "invalid_run"
    run["run_status"]["valid"] = True
    del run["actions"][1]
    assert controlled_run_metrics(run)["recovery_status"] == "update_not_delivered"


def test_actual_visual_release_and_upfront_frame_counts():
    run = record()
    run["actions"][1].update(
        type="visual_context_update", payload={"frame_paths": ["later.jpg"]}
    )
    assert controlled_run_metrics(run)["delivered_frame_count"] == 3
    run["condition"]["schedule"] = "upfront"
    del run["actions"][1]
    result = controlled_run_metrics(run)
    assert result["release_step"] == 0
    assert result["delivered_frame_count"] == 2
    del run["initial_context"]
    assert controlled_run_metrics(run)["delivered_frame_count"] is None


def test_multigoal_ambiguity_does_not_pick_union_or_first_answer():
    run = record()
    run["task_snapshot"]["goal"]["acceptable_states"].append(state("old", "other"))
    run["actions"] = run["actions"][:2]
    run["live_state"] = state("old", "keep")
    result = controlled_run_metrics(run)
    assert result["obsolete_residual_count"] is None
    assert result["obsolete_residual_opportunity"] is True
    assert result["state_change_status"] == "ambiguous_acceptable_states"


def test_any_acceptable_complete_answer_is_successful_recovery():
    run = record()
    alternative = state("old", "keep")
    run["task_snapshot"]["goal"]["acceptable_states"].append(alternative)
    run["actions"] = run["actions"][:2]
    run["live_state"] = alternative
    result = controlled_run_metrics(run)
    assert result["first_full_recovery_step"] == 2
    assert result["obsolete_residual_count"] == 0


def test_initially_wrong_selection_is_not_called_obsolete():
    run = record()
    run["task_snapshot"]["graft"]["initial_goal"]["acceptable_states"] = [
        state("different")
    ]
    run["actions"] = run["actions"][:2]
    run["live_state"] = state("old", "keep")
    result = controlled_run_metrics(run)
    assert result["state_change_status"] == "known"
    assert result["obsolete_residual_count"] == 0
    assert result["valid_item_loss_count"] == 0
    assert result["obsolete_residual_opportunity"] is False
    assert result["valid_item_loss_opportunity"] is False


def test_partial_initial_selection_still_supports_obsolete_evidence():
    run = record()
    run["actions"] = run["actions"][:2]
    run["actions"][0]["state_after"] = state("old")
    run["live_state"] = state("old")
    result = controlled_run_metrics(run)
    assert result["state_change_status"] == "known"
    assert result["obsolete_residual_count"] == 1
    assert result["valid_item_loss_count"] == 0


def test_wrong_and_right_initial_items_preserve_right_item_evidence():
    run = record()
    before = state("old", "keep", "wrong")
    run["actions"] = run["actions"][:2] + [
        {
            "type": "deselect_entity",
            "step": 3,
            "state_before": before,
            "state_after": state("old", "wrong"),
        },
    ]
    run["actions"][0]["state_after"] = before
    run["live_state"] = state("old", "wrong")
    result = controlled_run_metrics(run)
    assert result["obsolete_residual_count"] == 1
    assert result["valid_item_loss_count"] == 1


def test_collateral_damage_reads_lost_valid_items_and_stays_unknown_without_update():
    clean, damaged, undelivered = record(), record(), record()
    before = state("old", "keep")
    damaged["actions"] = damaged["actions"][:2] + [
        {
            "type": "deselect_entity",
            "step": 3,
            "state_before": before,
            "state_after": state("old"),
        },
    ]
    damaged["live_state"] = state("old")
    undelivered["actions"] = undelivered["actions"][:1]
    assert _collateral_damage(clean) is False
    assert _collateral_damage(damaged) is True
    assert _collateral_damage(undelivered) is None


def test_unmatched_initial_state_leaves_ambiguous_item_status_unknown():
    run = record()
    run["task_snapshot"]["graft"]["initial_goal"]["acceptable_states"].append(
        state("alternative", "keep")
    )
    run["actions"] = run["actions"][:2]
    run["actions"][0]["state_after"] = state("old", "keep", "wrong")
    run["live_state"] = state("old", "keep", "wrong")
    result = controlled_run_metrics(run)
    assert result["obsolete_residual_count"] is None
    assert result["valid_item_loss_count"] == 0
    assert result["state_change_status"] == "ambiguous_acceptable_states"


def test_matching_initial_answer_confirms_actual_selection_despite_alternatives():
    run = record()
    run["task_snapshot"]["graft"]["initial_goal"]["acceptable_states"].append(
        state("alternative", "keep")
    )
    run["actions"] = run["actions"][:2]
    run["live_state"] = state("old", "keep")
    result = controlled_run_metrics(run)
    assert result["obsolete_residual_count"] == 1
    assert result["obsolete_residual_opportunity"] is True


def test_missing_initial_goal_does_not_infer_legality_from_final_goal():
    run = record()
    run["task_snapshot"]["graft"] = {}
    run["actions"] = run["actions"][:2]
    run["live_state"] = state("old", "keep")
    result = controlled_run_metrics(run)
    assert result["state_change_status"] == "missing_initial_goal"
    assert result["obsolete_residual_count"] is None


def test_known_stale_item_and_retained_item_loss():
    run = record()
    run["actions"] = run["actions"][:2] + [
        {
            "type": "deselect_entity",
            "step": 3,
            "state_before": state("old", "keep"),
            "state_after": state("old"),
        },
    ]
    run["live_state"] = state("old")
    result = controlled_run_metrics(run)
    assert result["obsolete_residual_count"] == 1
    assert result["valid_item_loss_count"] == 1


def outcome(task, arm, success, replicate=0, valid=True):
    return {
        "task_id": task,
        "condition": {"name": arm},
        "e2e_success": success,
        "replicate_id": replicate,
        "run_valid": valid,
    }


def test_both_sides_and_difference_use_only_matching_replicates():
    rows = [
        outcome("t", "a", True, 0),
        outcome("t", "a", False, 1),
        outcome("t", "b", True, 9),
        outcome("u", "a", False),
        outcome("u", "b", False),
    ]
    result = paired_source_statistics(
        rows, PairedComparison("a", "b", {"t": "s1", "u": "s2"})
    )
    assert result["difference"] == 0
    assert result["left_success"] == result["right_success"] == 0
    assert result["joint_success"] == 0
    assert result["difference_ci95"] is None
    assert result["coverage"]["paired_sources"] == 1
    assert result["coverage"]["joint_paired_sources"] == 1
    assert result["coverage"]["paired_replicates"] == 1
    assert result["coverage"]["unpaired_replicates"] == 3


def test_quadrants_and_adaptability_read_left_as_reference():
    outcomes = {"t0": (True, True), **{f"t{i}": (True, False) for i in range(1, 6)}}
    rows = [
        outcome(task, arm, success)
        for task, pair in outcomes.items()
        for arm, success in zip(["a", "b"], pair, strict=True)
    ]
    result = paired_source_statistics(
        rows, PairedComparison("a", "b", {task: task for task in outcomes})
    )
    assert result["outcome_quadrants"] == {"S/S": 1, "S/F": 5, "F/S": 0, "F/F": 0}
    assert result["adaptability"] == 1 / 6
    assert result["adaptability_denominator"] == 6
    assert "mcnemar_p" not in result


def test_no_reference_success_leaves_adaptability_undefined():
    rows = [outcome("t", "a", False), outcome("t", "b", False)]
    result = paired_source_statistics(rows, PairedComparison("a", "b", {"t": "s"}))
    assert result["outcome_quadrants"]["F/F"] == 1
    assert result["adaptability"] is None
    assert result["adaptability_denominator"] == 0


def test_multiple_tasks_in_one_source_do_not_inflate_source_weight():
    rows = [outcome(task, arm, arm == "b") for task in ["t", "v"] for arm in ["a", "b"]]
    rows += [outcome("u", "a", True), outcome("u", "b", False)]
    result = paired_source_statistics(
        rows, PairedComparison("a", "b", {"t": "s1", "v": "s1", "u": "s2"})
    )
    assert result["difference"] == 0
    assert result["coverage"]["paired_pairs"] == 3
    assert result["coverage"]["paired_sources"] == 2


def test_invalid_and_missing_arms_report_coverage_not_failures():
    rows = [
        outcome("t", "a", True),
        outcome("t", "b", True),
        outcome("u", "a", False),
        outcome("u", "b", False, valid=False),
    ]
    result = paired_source_statistics(
        rows, PairedComparison("a", "b", {"t": "s1", "u": "s2", "absent": "s3"})
    )
    assert result["joint_success"] == 1
    assert result["difference_ci95"] is None
    assert result["coverage"]["invalid_runs"] == 1
    assert result["coverage"]["source_fraction"] == 1 / 3
    assert result["coverage"]["unpaired_pairs"] == 1


def test_distinct_task_variants_can_use_explicit_pair_id():
    rows = [outcome("t1", "a", True), outcome("t2", "b", True)]
    for row in rows:
        row["pair_id"] = "variant-pair"
    result = paired_source_statistics(
        rows, PairedComparison("a", "b", {"t1": "s", "t2": "s"})
    )
    assert result["joint_success"] == 1


def test_unvalidated_and_duplicate_records_fail_loudly():
    comparison = PairedComparison("a", "b", {"t": "s"})
    with pytest.raises(ValueError, match="unvalidated"):
        paired_source_statistics([outcome("unknown", "a", True)], comparison)
    row = outcome("t", "a", True)
    with pytest.raises(ValueError, match="duplicate"):
        paired_source_statistics([row, deepcopy(row)], comparison)


def test_no_complete_pair_has_no_estimate():
    result = paired_source_statistics(
        [outcome("t", "a", True)], PairedComparison("a", "b", {"t": "s"})
    )
    assert result["difference"] is None
    assert result["joint_success"] is None


@pytest.mark.parametrize("ending, obsolete", [(state("old"), 1), (state("new"), 0)])
def test_refresh_replace_counts_only_old_target_residual(ending, obsolete):
    run = record()
    run["task_snapshot"] = {
        "id": "test-metrics-refresh",
        "track": "refresh",
        "refresh": {"policy": "replace", "initial_entity_id": "old"},
        "truth": {"initial_acceptable_states": [state("old")]},
        "goal": {"acceptable_states": [state("new")]},
    }
    run["actions"] = [
        {
            "type": "select_entity",
            "step": 1,
            "state_before": state(),
            "state_after": state("old"),
        },
        {
            "type": "visual_context_update",
            "step": 2,
            "payload": {"frame_paths": ["later.jpg"]},
        },
        {
            "type": "select_entity",
            "step": 3,
            "state_before": state("old"),
            "state_after": ending,
        },
    ]
    run["live_state"] = ending
    result = controlled_run_metrics(run)
    assert result["state_change_status"] == "known"
    assert result["obsolete_residual_count"] == obsolete
    assert result["valid_item_loss_count"] == 0


@pytest.mark.parametrize("ending, loss", [(state("other"), 1), (state("old"), 0)])
def test_refresh_retain_counts_valid_target_loss(ending, loss):
    run = record()
    run["task_snapshot"] = {
        "id": "test-metrics-refresh",
        "track": "refresh",
        "refresh": {"policy": "retain", "initial_entity_id": "old"},
        "truth": {"initial_acceptable_states": [state("old")]},
        "goal": {"acceptable_states": [state("old")]},
    }
    run["actions"] = [
        {
            "type": "select_entity",
            "step": 1,
            "state_before": state(),
            "state_after": state("old"),
        },
        {
            "type": "visual_context_update",
            "step": 2,
            "payload": {"frame_paths": ["later.jpg"]},
        },
        {
            "type": "select_entity",
            "step": 3,
            "state_before": state("old"),
            "state_after": ending,
        },
    ]
    run["live_state"] = ending
    result = controlled_run_metrics(run)
    assert result["obsolete_residual_count"] == 0
    assert result["valid_item_loss_count"] == loss


def test_refresh_cannot_lose_a_target_that_was_never_selected():
    run = record()
    run["task_snapshot"] = {
        "id": "test-metrics-refresh",
        "track": "refresh",
        "refresh": {"policy": "retain", "initial_entity_id": "old"},
        "truth": {"initial_acceptable_states": [state("old")]},
        "goal": {"acceptable_states": [state("old")]},
    }
    run["actions"] = [
        {
            "type": "visual_context_update",
            "step": 2,
            "payload": {"frame_paths": ["later.jpg"]},
        },
    ]
    run["live_state"] = state()
    assert controlled_run_metrics(run)["valid_item_loss_count"] == 0


def test_refresh_missing_initial_facts_are_unknown_and_alternatives_are_not_guessed():
    run = record()
    run["task_snapshot"] = {
        "id": "test-metrics-refresh",
        "track": "refresh",
        "refresh": {"policy": "replace", "initial_entity_id": "old"},
        "goal": {"acceptable_states": [state("new")]},
    }
    run["actions"] = [
        {
            "type": "select_entity",
            "step": 1,
            "state_before": state(),
            "state_after": state("alternative"),
        },
        {"type": "visual_context_update", "step": 2},
    ]
    run["steps"] = 2
    run["live_state"] = state("alternative")
    historical = controlled_run_metrics(run)
    assert historical["state_change_status"] == "missing_initial_goal"
    assert historical["obsolete_residual_count"] is None
    run["task_snapshot"]["truth"] = {
        "initial_acceptable_states": [state("old"), state("alternative")],
    }
    current = controlled_run_metrics(run)
    assert current["state_change_status"] == "known"
    assert current["obsolete_residual_count"] == 1


def test_clarify_state_change_metrics_are_not_applicable_even_on_success():
    run = record()
    del run["task_snapshot"]["graft"]
    run["task_snapshot"]["track"] = "clarify"
    result = controlled_run_metrics(run)
    assert result["state_change_status"] == "not_applicable"
    assert result["obsolete_residual_count"] is None
    assert result["valid_item_loss_count"] is None


def test_joint_success_measures_real_matching_outcomes_not_product_of_means():
    rows = [
        outcome("t", "a", True, 0),
        outcome("t", "b", True, 0),
        outcome("t", "a", False, 1),
        outcome("t", "b", False, 1),
    ]
    result = paired_source_statistics(rows, PairedComparison("a", "b", {"t": "s"}))
    assert result["joint_success"] == 0.5
    assert result["coverage"]["paired_replicates"] == 2


def test_opposite_matching_outcomes_have_zero_joint_success():
    rows = [
        outcome("t", "a", True, 0),
        outcome("t", "b", False, 0),
        outcome("t", "a", False, 1),
        outcome("t", "b", True, 1),
    ]
    result = paired_source_statistics(rows, PairedComparison("a", "b", {"t": "s"}))
    assert result["joint_success"] == 0


def test_joint_success_gives_sources_equal_weight_not_variant_count():
    rows = [
        outcome(task, arm, task != "u")
        for task in ["t", "v", "u"]
        for arm in ["a", "b"]
    ]
    result = paired_source_statistics(
        rows, PairedComparison("a", "b", {"t": "s1", "v": "s1", "u": "s2"})
    )
    assert result["joint_success"] == 0.5
    assert result["joint_success_ci95"] == [0, 1]


def test_no_matching_replicates_has_no_effect_estimate():
    rows = [outcome("t", "a", False, 0), outcome("t", "b", True, 1)]
    result = paired_source_statistics(rows, PairedComparison("a", "b", {"t": "s"}))
    assert result["difference"] is None
    assert result["joint_success"] is None
    assert result["joint_success_ci95"] is None
    assert result["coverage"]["unpaired_replicates"] == 2


def test_invalid_matching_replicate_does_not_enter_joint_denominator():
    rows = [
        outcome("t", "a", True, 0),
        outcome("t", "b", True, 0),
        outcome("t", "a", True, 1),
        outcome("t", "b", True, 1, valid=False),
    ]
    result = paired_source_statistics(rows, PairedComparison("a", "b", {"t": "s"}))
    assert result["joint_success"] == 1
    assert result["coverage"]["paired_replicates"] == 1
    assert result["coverage"]["unpaired_replicates"] == 1
    assert result["coverage"]["invalid_runs"] == 1


def test_three_arm_comparison_counts_coverage_before_filtering_unscorable_runs():
    rows = [
        outcome("t", "a", True),
        outcome("t", "b", True),
        outcome("t", "c", None),
        outcome("u", "a", True),
        outcome("u", "b", False),
        outcome("u", "c", True),
    ]
    result = paired_source_statistics(
        rows, PairedComparison("a", "b", {"t": "s1", "u": "s2"}, required_conditions=("c",))
    )
    assert result["coverage"]["attempted_pairs"] == 2
    assert result["coverage"]["paired_pairs"] == 1
    assert result["coverage"]["unpaired_pairs"] == 1
    assert result["difference"] == result["right_success"] - result["left_success"] == -1


def test_keep_initial_grades_each_distinct_physical_cart_without_using_final_to_choose():
    task = record()["task_snapshot"]
    task["graft"]["initial_goal"]["acceptable_states"] = [
        state("old", "keep"), state("keep", "new")
    ]
    assert rule_baseline_scores(task, state()) == {"no_action": 0, "keep_initial": 0.5}
    assert rule_baseline_scores(task, state("new", "keep"))["no_action"] == 1


def test_rule_baseline_leaves_unresolved_initial_target_inapplicable():
    for track in ("clarify", "refresh"):
        task = record()["task_snapshot"]
        task["track"] = track
        task["refresh"] = {"policy": "replace"}
        assert rule_baseline_scores(task, state()) == {"no_action": 0, "keep_initial": None}


def test_same_pair_slot_cannot_be_duplicated_under_different_task_ids():
    rows = [outcome("t", "a", True), outcome("u", "a", False)]
    for row in rows:
        row["pair_id"] = "same-pair"
    with pytest.raises(ValueError, match="duplicate"):
        paired_source_statistics(rows, PairedComparison("a", "b", {"t": "s", "u": "s"}))


def test_trajectory_measures_initial_and_final_goals_with_full_checker():
    from egocua.controlled_metrics import trajectory_mechanism

    run = record()
    result = trajectory_mechanism(run)
    assert result["reason"] == "adjustment_required"
    assert result["initial_goal_satisfied"] is True
    assert result["native_state_changed_after_update"] is True
    run["task_snapshot"]["graft"]["initial_goal"]["acceptable_states"] = [
        state("old", "keep", "keep")
    ]
    assert trajectory_mechanism(run)["initial_goal_satisfied"] is False
    run["task_snapshot"]["graft"]["initial_goal"]["acceptable_states"] = [
        state("old", "keep", fields={"route": "walking"})
    ]
    assert trajectory_mechanism(run)["initial_goal_satisfied"] is False


def test_trajectory_distinguishes_retention_from_changes_and_repaired_errors():
    from egocua.controlled_metrics import trajectory_mechanism

    run = record()
    run["task_snapshot"]["goal"]["acceptable_states"].append(state("old", "keep"))
    assert trajectory_mechanism(run)["reason"] == "final_goal_already_satisfied"
    run["actions"] = run["actions"][:2]
    run["live_state"] = state("old", "keep")
    assert trajectory_mechanism(run)["reason"] == "valid_selection_retained"
    assert controlled_run_metrics(run)["recovery_status"] == "recovered"
    run["actions"].extend(
        [
            {
                "type": "deselect_entity",
                "step": 3,
                "state_before": state("old", "keep"),
                "state_after": state("old"),
            },
            {
                "type": "select_entity",
                "step": 4,
                "state_before": state("old"),
                "state_after": state("old", "keep"),
            },
        ]
    )
    result = trajectory_mechanism(run)
    assert result["reason"] == "final_goal_already_satisfied"
    assert result["native_state_changed_after_update"] is True
    assert controlled_run_metrics(run)["recovery_status"] == "recovered"


def test_trajectory_keeps_missing_evidence_distinct_from_negative_facts():
    from egocua.controlled_metrics import trajectory_mechanism

    run = record()
    run["actions"][2].pop("state_after")
    result = trajectory_mechanism(run)
    assert result["status"] == "unknown"
    assert result["initial_goal_satisfied"] is None
    assert result["native_state_changed_after_update"] is None
    assert result["update_delivered"] is True
    run = record()
    del run["actions"][1]
    result = trajectory_mechanism(run)
    assert result["reason"] == "update_not_delivered"
    assert result["update_delivered"] is False


def test_trajectory_clarification_delivery_is_not_a_success_requirement():
    from egocua.controlled_metrics import trajectory_mechanism

    run = record()
    run["task_snapshot"].update(
        track="clarify", clarification={"canonical_answer": "The one on the left."}
    )
    run["actions"] = []
    assert trajectory_mechanism(run)["reason"] == "answer_not_delivered"
    run["actions"] = [
        {
            "type": "clarification_answer",
            "step": 2,
            "valid": True,
            "payload": "The one on the left.",
        }
    ]
    assert trajectory_mechanism(run)["answer_delivered"] is True
    run["actions"][0]["valid"] = False
    assert trajectory_mechanism(run)["answer_delivered"] is False
    run["condition"]["schedule"] = "upfront"
    run["task_snapshot"]["instructions"] = {"oracle": "Pick the left tool."}
    run["initial_context"]["instruction"] = "Pick the left tool."
    result = trajectory_mechanism(run)
    assert result["reason"] == "upfront_information"
    assert result["update_delivered"] is True
    assert result["answer_delivered"] is False


@pytest.mark.parametrize(("replacement", "expected"), [("wrong", True), ("alternate", False)])
def test_cue_retain_regression_distinguishes_invalid_and_legal_replacements(
    replacement, expected
):
    run = record()
    goal = {"acceptable_states": [state("old"), state("alternate")]}
    run["task_snapshot"] = {"track": "refresh", "refresh": {"policy": "retain"}, "goal": goal}
    run["initial_state"] = state("old")
    run["live_state"] = state("old")
    run["actions"] = [
        {"type": "visual_context_update", "update": True, "step": 2},
        {"type": "select_entity", "step": 3,
         "state_before": state("old"), "state_after": state(replacement)},
        {"type": "select_entity", "step": 4,
         "state_before": state(replacement), "state_after": state("old")},
    ]
    assert controlled_run_metrics(run)["retain_selection_regression"] is expected


def test_arrival_summary_keeps_undelivered_terminal_success():
    from egocua.controlled_metrics import trajectory_mechanism
    from egocua.study_summary import _arm_summary

    base = {
        "variable": "arrival",
        "track": "graft",
        "run_valid": True,
        "process": {"release_step": None, "delivered_frame_count": 8, "obsolete_residual_count": None,
                    "valid_item_loss_count": None, "obsolete_residual_opportunity": None,
                    "valid_item_loss_opportunity": None},
        "trajectory": trajectory_mechanism({}),
        "key_evidence_visibility": "unreviewed",
        "usage": None,
        "request_inputs": None,
    }
    rows = [
        {**base, "woken_from_standby": None, "e2e_success": True},
        {**base, "woken_from_standby": True, "e2e_success": True},
        {**base, "woken_from_standby": False, "e2e_success": False},
    ]
    summary = _arm_summary(rows)
    assert summary["success"]["rate"] == 2 / 3
    assert summary["arrival"]["undelivered"]["rate"] == 1 / 3
    assert summary["arrival"]["delivered_and_success"]["rate"] == 1 / 3
    assert summary["arrival"]["delivery_mode"] == {"undelivered": 1, "standby": 1, "active": 1}


def test_commitment_groups_do_not_confuse_holding_with_a_past_action():
    from egocua.study_summary import _commitment_summaries

    base = {
        "variable": "formal", "level": "online", "track": "graft", "run_valid": True,
        "e2e_success": True, "update_after_commitment": True,
        "trajectory": {"update_delivered": True},
        "process": {
            "selection_held_at_update": False,
            "obsolete_residual_count": 0, "valid_item_loss_count": 0,
            "obsolete_residual_opportunity": False, "valid_item_loss_opportunity": False,
        },
    }
    summary = _commitment_summaries([
        base,
        {**base, "update_after_commitment": False, "e2e_success": False},
        {**base, "trajectory": {"update_delivered": False}},
        {**base, "process": {**base["process"], "selection_held_at_update": None}},
    ])["by_track"]["graft"]["groups"]
    assert {name: value["observed"] for name, value in summary.items()} == {
        "before_commitment": 1, "after_commitment": 1, "undelivered": 1, "unknown": 1
    }
    assert summary["after_commitment"]["selection_held_at_update"]["rate"] == 0
    assert summary["before_commitment"]["success"]["rate"] == 0
    assert summary["after_commitment"]["obsolete_residual"]["opportunities"]["count"] == 0


def test_paired_costs_use_common_triplets_source_weights_and_known_usage():
    from egocua.study_summary import _paired_costs
    from egocua.controlled_metrics import PairedComparison
    rows = []
    for task_id, _source, values, success in (
        ('a', 's1', (2, 4), True), ('b', 's1', (4, 8), True),
        ('c', 's2', (10, 10), False),
    ):
        for name, cost in zip(('offline', 'online'), values, strict=True):
            rows.append({'task_id': task_id, 'replicate_id': 0, 'condition': {'name': name},
                         'run_valid': True, 'e2e_success': success,
                         'costs': {'operations': cost, 'tokens': None}})
        rows.append({'task_id': task_id, 'replicate_id': 0, 'condition': {'name': 'computer'},
                     'run_valid': True, 'e2e_success': True, 'costs': {}})
    protocol = PairedComparison('offline', 'online', {'a': 's1', 'b': 's1', 'c': 's2'},
                                required_conditions=('computer',))
    result = _paired_costs(rows, protocol)
    assert result['all_valid']['metrics']['operations']['difference'] == 1.5
    assert result['both_successful']['metrics']['operations']['difference'] == 3
    assert result['all_valid']['metrics']['tokens']['missing_pairs'] == 3
    assert result['all_valid']['metrics']['tokens']['left_mean'] is None
    rows.pop()
    assert _paired_costs(rows, protocol)['all_valid']['paired_runs'] == 2


def test_run_costs_count_failed_browser_operations_not_controls_or_requests():
    from egocua.study_summary import _run_costs
    history = [{'model_output': {'action': [a]}, 'result': [{'error': 'error'}]} for a in (
        {'browser_action': {'code': "click('12')"}}, {'browser_action': {'code': 'wait'}},
        {'ask_user': {'question': 'Which?'}}, {'done': {}},
    )]
    assert _run_costs({'steps': 4, 'meta': {'agent': 'browsergym'}, 'trace': {'history': history}})['browser_operations'] == 1
    assert _run_costs({'meta': {'agent': 'browser-use'}, 'trace': {'history': history}})['browser_operations'] is None
    assert _run_costs({})['browser_operations'] is None


@pytest.mark.parametrize("duration", [0, 12.5])
def test_run_costs_preserve_recorded_duration_and_count_logical_requests(duration):
    from egocua.study_summary import _run_costs

    request = {
        "success": True,
        "wearer_image_count": 2,
        "website_screenshot_count": 1,
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }
    record = {
        "meta": {"duration_seconds": duration},
        "trace": {
            "llm_requests": [request, {**request, "retries": [{"attempt": 1}]}],
            "usage": {"total_prompt_tokens": 200, "total_completion_tokens": 20},
        },
    }

    costs = _run_costs(record)

    assert costs["wall_time_seconds"] == duration
    assert costs["model_requests"] == 2
    assert costs["input_tokens"] == 200
    assert costs["output_tokens"] == 20


def test_run_costs_distinguish_missing_records_from_zero_requests():
    from egocua.study_summary import _run_costs

    missing = _run_costs({})
    no_requests = _run_costs({"trace": {"llm_requests": []}})

    assert missing["wall_time_seconds"] is None
    assert missing["model_requests"] is None
    assert no_requests["model_requests"] == 0
    assert no_requests["input_tokens"] is None
    assert no_requests["wall_time_seconds"] is None


def test_paired_costs_exclude_missing_durations_without_replacing_them_with_zero():
    from egocua.controlled_metrics import PairedComparison
    from egocua.study_summary import _paired_costs

    rows = []
    for task_id, durations in (("a", (10.5, 11)), ("b", (20, 24)), ("c", (None, 30))):
        for condition, duration in zip(("offline", "online"), durations, strict=True):
            rows.append({
                "task_id": task_id,
                "replicate_id": 0,
                "condition": {"name": condition},
                "run_valid": True,
                "e2e_success": True,
                "costs": {"wall_time_seconds": duration},
            })
    protocol = PairedComparison("offline", "online", {"a": "s1", "b": "s1", "c": "s2"})

    summary = _paired_costs(rows, protocol)["all_valid"]
    timing = summary["metrics"]["wall_time_seconds"]

    assert summary["paired_runs"] == 3
    assert timing["paired_runs"] == 2
    assert timing["missing_pairs"] == 1
    assert timing["sources"] == 1
    assert timing["left_mean"] == 15.25
    assert timing["right_mean"] == 17.5
    assert timing["difference"] == 2.25
    assert timing["difference_ci95"] is None


def test_empty_prior_selection_has_no_change_error_opportunity():
    from egocua.controlled_metrics import _state_change_errors

    task = record()["task_snapshot"]
    task["graft"]["initial_goal"]["acceptable_states"] = [
        state("old", f"initial-{index}") for index in range(100)
    ]
    task["goal"]["acceptable_states"] = [
        state("new", f"final-{index}") for index in range(100)
    ]
    assert _state_change_errors(task, state(), state("unrelated")) == {
        "obsolete_residual_count": 0,
        "valid_item_loss_count": 0,
        "obsolete_residual_opportunity": False,
        "valid_item_loss_opportunity": False,
        "state_change_status": "known",
    }


def test_change_errors_normalize_final_alternatives_only_once(monkeypatch):
    from egocua import controlled_metrics

    task = record()["task_snapshot"]
    task["graft"]["initial_goal"]["acceptable_states"] = [
        state("old", f"initial-{index}") for index in range(30)
    ]
    task["goal"]["acceptable_states"] = [
        state("new", f"final-{index}") for index in range(50)
    ]
    original = controlled_metrics.normalized_entity_counts
    calls = []

    def counted(goal, ids):
        calls.append((goal, ids))
        return original(goal, ids)

    monkeypatch.setattr(controlled_metrics, "normalized_entity_counts", counted)
    result = controlled_metrics._state_change_errors(task, state("old"), state("old"))
    assert result["obsolete_residual_count"] == 1
    assert result["valid_item_loss_count"] == 0
    assert result["obsolete_residual_opportunity"] is True
    assert result["valid_item_loss_opportunity"] is False
    assert len(calls) == 3 + 30 + 50
