from __future__ import annotations

import unittest
from typing import Any

from egocua.checker import check_run
from egocua.metrics import aggregate, bootstrap_intervals
from tests.helpers import item_id, task


def record(task_id: str, track: str, arm: str, selected: list[str], actions: list[dict] | None = None) -> dict:
    return {
        "task_id": task_id,
        "track": track,
        "arm": arm,
        "steps": 5,
        "actions": actions or [],
        "final_state": {
            "selected_entity_ids": selected,
            "fields": {},
            "submission_completed": True,
        },
    }


def goal(*entity_options: list[str]) -> dict[str, Any]:
    return {
        "target_granularity": "variant",
        "acceptable_states": [
            {"selected_entity_ids": entities, "fields": {}, "submission_completed": True}
            for entities in entity_options
        ],
    }


def clarification_answer(
    clarify: dict[str, Any], step: int, *, question: str = "Which variant do you mean?"
) -> dict[str, Any]:
    return {
        "type": "clarification_answer",
        "step": step,
        "slot": clarify["clarification"]["valid_slot"],
        "question": question,
        "valid": True,
        "payload": clarify["clarification"]["canonical_answer"],
    }


class CheckerMetricsTest(unittest.TestCase):
    def test_workload_is_derived_from_domain_actions(self) -> None:
        graft = task("graft")
        checked = check_run(
            graft,
            record(
                graft["id"],
                "graft",
                "a_only",
                [item_id("graft", 0)],
                [
                    {"type": "search", "query": "blocks"},
                    {"type": "search", "query": "wooden blocks"},
                    {"type": "open_product", "item_id": item_id("graft", 1)},
                    {"type": "open_product", "item_id": item_id("graft", 0)},
                    {"type": "add_to_cart", "item_id": item_id("graft", 1)},
                    {"type": "remove_from_cart", "item_id": item_id("graft", 1)},
                    {"type": "add_to_cart", "item_id": item_id("graft", 0)},
                ],
            ),
        )

        self.assertEqual(checked["search_actions"], 2)
        self.assertEqual(checked["distinct_search_queries"], 2)
        self.assertEqual(checked["entity_detail_opens"], 2)
        self.assertEqual(checked["distinct_entities_opened"], 2)
        self.assertEqual(checked["selection_mutations"], 3)
        self.assertEqual(checked["selection_corrections"], 1)
        self.assertEqual(aggregate([checked])["common"]["selection_corrections"], 1.0)

    def test_duplicate_quantity_fails_exact_goal_state_validation(self) -> None:
        graft = task("graft")
        checked = check_run(
            graft,
            record(graft["id"], "graft", "a_only", [item_id("graft", 0), item_id("graft", 0)]),
        )

        self.assertFalse(checked["goal_state_correct"])

    def test_exact_goal_state_rejects_partial_extra_and_unknown_submissions(self) -> None:
        graft = task("graft")
        graft["graft"]["integration_type"] = "P"
        graft["graft"]["initial_goal"] = goal([item_id("graft", 0)])
        graft["goal"] = goal([item_id("graft", 0), item_id("graft", 1)])

        for submitted in ([item_id("graft", 0)], [item_id("graft", 0), item_id("graft", 1), item_id("graft", 2)], ["unknown-id"]):
            with self.subTest(submitted=submitted):
                checked = check_run(
                    graft, record(graft["id"], "graft", "bundled", list(submitted))
                )
                self.assertFalse(checked["goal_state_correct"])

    def test_incomplete_final_state_is_unknown_and_not_a_valid_failure(self) -> None:
        graft = task("graft")
        incomplete = record(graft["id"], "graft", "a_only", [item_id("graft", 0)])
        incomplete["final_state"].pop("submission_completed")

        checked = check_run(graft, incomplete)

        self.assertFalse(checked["run_valid"])
        self.assertEqual(checked["evidence_status"], "UNKNOWN")
        self.assertIsNone(checked["e2e_success"])

    def test_dynamic_arm_requires_exactly_one_timeline_event(self) -> None:
        graft = task("graft")
        missing = check_run(
            graft, record(graft["id"], "graft", "injected", [item_id("graft", 0)])
        )
        delivered = check_run(
            graft,
            record(
                graft["id"], "graft", "injected", [item_id("graft", 0)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {"type": "instruction_injection", "step": 2, "trigger_reason": "task_timeline"},
                ],
            ),
        )
        early = check_run(
            graft,
            record(
                graft["id"], "graft", "injected", [item_id("graft", 0)],
                [{"type": "instruction_injection", "step": 1, "trigger_reason": "task_timeline"}],
            ),
        )
        duplicated = check_run(
            graft,
            record(
                graft["id"], "graft", "injected", [item_id("graft", 0)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {"type": "instruction_injection", "step": 2, "trigger_reason": "task_timeline"},
                    {"type": "instruction_injection", "step": 3, "trigger_reason": "task_timeline"},
                ],
            ),
        )

        self.assertFalse(missing["run_valid"])
        self.assertTrue(delivered["run_valid"])
        self.assertFalse(early["run_valid"])
        self.assertFalse(duplicated["run_valid"])

    def test_timeline_delivery_is_valid_for_dynamic_runs(self) -> None:
        graft = task("graft")
        checked = check_run(
            graft,
            record(
                graft["id"], "graft", "injected", [item_id("graft", 0)],
                [{
                    "type": "instruction_injection",
                    "step": 2,
                    "trigger_reason": "task_timeline",
                }],
            ),
        )

        self.assertTrue(checked["run_valid"])
        self.assertEqual(checked["event_trigger_reason"], "task_timeline")

    def test_submission_before_a_required_event_is_protocol_invalid(self) -> None:
        graft = task("graft")
        checked = check_run(
            graft,
            record(
                graft["id"], "graft", "injected", [item_id("graft", 0)],
                [
                    {"type": "submit", "step": 1, "item_ids": [item_id("graft", 0)]},
                    {"type": "instruction_injection", "step": 2, "trigger_reason": "task_timeline"},
                ],
            ),
        )

        self.assertFalse(checked["run_valid"])
        self.assertIn("submission_before_required_event", checked["run_invalid_reasons"])

    def test_retain_false_update_distinguishes_switching_from_no_submission(self) -> None:
        refresh = task("refresh")
        refresh["refresh"]["policy"] = "retain"
        refresh["goal"] = goal([item_id("refresh", 0)])

        no_submission = check_run(
            refresh, record(refresh["id"], "refresh", "stream", [])
        )
        switched = check_run(
            refresh, record(refresh["id"], "refresh", "stream", [item_id("refresh", 2)])
        )
        self.assertFalse(no_submission["false_update"])
        self.assertTrue(switched["false_update"])

    def test_retain_false_update_is_only_defined_for_stream(self) -> None:
        refresh = task("refresh")
        refresh["refresh"]["policy"] = "retain"
        refresh["goal"] = goal([item_id("refresh", 0)])

        for arm in ("stale", "oracle_latest"):
            checked = check_run(
                refresh,
                record(refresh["id"], "refresh", arm, [item_id("refresh", 2)]),
            )
            self.assertIsNone(checked["false_update"])

    def test_answer_utilization_requires_the_first_post_answer_candidate_action(self) -> None:
        clarify = task("clarify")
        asked_success = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [item_id("clarify", 1)],
                [
                    clarification_answer(clarify, 2),
                    {"type": "open_product", "step": 3, "item_id": item_id("clarify", 1)},
                ],
            ),
        )
        no_ask_failure = check_run(
            clarify,
            record(clarify["id"], "clarify", "interactive", [item_id("clarify", 0)]),
        )
        report = aggregate([asked_success, no_ask_failure])
        self.assertEqual(report["clarify"]["answer_utilization"], 1.0)

    def test_interactive_success_requires_a_relevant_question_and_delivered_answer(self) -> None:
        clarify = task("clarify")
        target = item_id("clarify", 1)
        unrelated = clarification_answer(
            clarify, 1, question="What is the weather?"
        )
        correct_state_after_bad_question = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [target],
                [unrelated, {"type": "select", "step": 2, "item_id": target}],
            ),
        )
        missing_answer = clarification_answer(clarify, 1)
        missing_answer.pop("payload")
        correct_state_without_answer = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [target],
                [missing_answer, {"type": "select", "step": 2, "item_id": target}],
            ),
        )

        self.assertFalse(correct_state_after_bad_question["valid_question"])
        self.assertFalse(correct_state_after_bad_question["e2e_success"])
        self.assertFalse(correct_state_without_answer["answer_delivered"])
        self.assertFalse(correct_state_without_answer["e2e_success"])

    def test_interactive_success_requires_timely_question_and_answer_use(self) -> None:
        clarify = task("clarify")
        target = item_id("clarify", 1)
        late = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [target],
                [
                    {"type": "select", "step": 1, "item_id": target},
                    clarification_answer(clarify, 2),
                ],
            ),
        )
        submitted_before_ask = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [target],
                [
                    {"type": "submit", "step": 1},
                    clarification_answer(clarify, 2),
                ],
            ),
        )
        ignored = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [target],
                [clarification_answer(clarify, 1)],
            ),
        )
        successful = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [target],
                [
                    clarification_answer(clarify, 1),
                    {"type": "open_product", "step": 2, "item_id": target},
                ],
            ),
        )

        self.assertFalse(late["timely_question"])
        self.assertFalse(late["e2e_success"])
        self.assertFalse(submitted_before_ask["timely_question"])
        self.assertFalse(submitted_before_ask["e2e_success"])
        self.assertFalse(ignored["answer_used"])
        self.assertFalse(ignored["e2e_success"])
        self.assertTrue(successful["answer_used"])
        self.assertTrue(successful["e2e_success"])

    def test_oracle_unnecessary_question_is_diagnostic_not_an_e2e_failure(self) -> None:
        clarify = task("clarify")
        target = item_id("clarify", 1)
        checked = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "oracle", [target],
                [clarification_answer(clarify, 1)],
            ),
        )

        self.assertTrue(checked["unnecessary_ask"])
        self.assertTrue(checked["e2e_success"])

    def test_second_clarification_attempt_is_auditable(self) -> None:
        clarify = task("clarify")
        checked = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [item_id("clarify", 1)],
                [
                    clarification_answer(clarify, 1),
                    {"type": "clarification_answer", "step": 2, "limit_reached": True},
                    {"type": "select", "step": 3, "item_id": item_id("clarify", 1)},
                ],
            ),
        )

        self.assertEqual(checked["question_attempts"], 2)
        self.assertTrue(checked["question_limit_exceeded"])

    def test_answer_utilization_excludes_questions_after_a_candidate_commitment(self) -> None:
        clarify = task("clarify")
        timely_failure = check_run(
            clarify,
            record(
                "c1", "clarify", "interactive", [item_id("clarify", 0)],
                [
                    clarification_answer(clarify, 2),
                    {"type": "open_product", "step": 3, "item_id": item_id("clarify", 0)},
                ],
            ),
        )
        late_success = check_run(
            clarify,
            record(
                "c2", "clarify", "interactive", [item_id("clarify", 1)],
                [
                    {"type": "select", "step": 2, "item_id": item_id("clarify", 1)},
                    clarification_answer(clarify, 3),
                ],
            ),
        )

        report = aggregate([timely_failure, late_success])

        self.assertEqual(report["clarify"]["answer_utilization"], 0.0)

    def test_arm_effects_pair_tasks_not_uncontrolled_seed_labels(self) -> None:
        rows = [
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "bundled", "seed": 11, "replicate_id": 0, "run_valid": True, "e2e_success": True},
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "injected", "seed": 29, "replicate_id": 1, "run_valid": True, "e2e_success": False},
            {"task_id": "g2", "track": "graft", "subtype": "C", "arm": "bundled", "seed": 7, "replicate_id": 0, "run_valid": True, "e2e_success": False},
        ]
        report = aggregate(rows)
        self.assertEqual(report["graft"]["diagnostic_interruption_cost"], 1.0)

    def test_invalid_runs_are_reported_but_excluded_from_agent_success(self) -> None:
        rows = [
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": True, "e2e_success": True},
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": False, "e2e_success": None},
        ]

        report = aggregate(rows)

        self.assertEqual(report["common"]["valid_run_rate"], 0.5)
        self.assertEqual(report["common"]["e2e_success"], 1.0)
        self.assertNotIn("capability_filtered", report)

    def test_clarification_policy_reports_balanced_ask_f1(self) -> None:
        clarify = task("clarify")
        rows: list[dict[str, Any]] = []
        for task_id, arm, selected, actions in (
            ("c1", "interactive", [item_id("clarify", 1)], [clarification_answer(clarify, 2)]),
            ("c2", "interactive", [item_id("clarify", 0)], []),
            ("c1", "oracle", [item_id("clarify", 1)], []),
            ("c2", "oracle", [item_id("clarify", 1)], [clarification_answer(clarify, 2)]),
        ):
            current = dict(clarify)
            current["id"] = task_id
            rows.append(check_run(current, record(task_id, "clarify", arm, selected, actions)))

        policy = aggregate(rows)["clarify"]

        self.assertEqual(policy["ask_recall"], 0.5)
        self.assertEqual(policy["diagnostic_ask_precision"], 0.5)
        self.assertEqual(policy["diagnostic_ask_f1"], 0.5)

    def test_clarification_policy_handles_an_oracle_only_slice(self) -> None:
        clarify = task("clarify")
        checked = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "oracle", [item_id("clarify", 1)],
                [clarification_answer(clarify, 2)],
            ),
        )

        policy = aggregate([checked])["clarify"]

        self.assertEqual(policy["diagnostic_ask_precision"], 0.0)
        self.assertIsNone(policy["ask_recall"])
        self.assertIsNone(policy["diagnostic_ask_f1"])

    def test_refresh_latency_reports_nonresponses_and_post_event_success_curve(self) -> None:
        refresh = task("refresh")
        success = check_run(
            refresh,
            record(
                "r1", "refresh", "stream", [item_id("refresh", 1)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {"type": "visual_context_update", "step": 2, "trigger_reason": "task_timeline"},
                    {"type": "select", "step": 4, "item_id": item_id("refresh", 1)},
                ],
            ),
        )
        failure = check_run(
            refresh,
            record(
                "r2", "refresh", "stream", [],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {"type": "visual_context_update", "step": 2, "trigger_reason": "task_timeline"},
                ],
            ),
        )

        metrics = aggregate([success, failure])["refresh"]

        self.assertEqual(metrics["post_update_target_action_response_rate"], 0.5)
        self.assertEqual(metrics["post_update_target_action_latency_on_response"], 2.0)
        self.assertEqual(metrics["post_update_sr_at_3"], 0.5)

    def test_refresh_response_metrics_only_use_replace_tasks(self) -> None:
        refresh = task("refresh")
        replace_failure = check_run(
            refresh,
            record(
                "replace-1", "refresh", "stream", [],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {"type": "visual_context_update", "step": 2, "trigger_reason": "task_timeline"},
                ],
            ),
        )
        retain = task("refresh")
        retain["id"] = "retain-1"
        retain["refresh"]["policy"] = "retain"
        retain["goal"] = goal([item_id("refresh", 0)])
        retain_response = check_run(
            retain,
            record(
                "retain-1", "refresh", "stream", [item_id("refresh", 0)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {"type": "visual_context_update", "step": 2, "trigger_reason": "task_timeline"},
                    {"type": "select", "step": 3, "item_id": item_id("refresh", 0)},
                ],
            ),
        )

        metrics = aggregate([replace_failure, retain_response])["refresh"]

        self.assertEqual(metrics["post_update_target_action_response_rate"], 0.0)
        self.assertEqual(metrics["post_update_sr_at_3"], 0.0)

    def test_graft_retention_is_only_defined_when_final_goal_requires_a(self) -> None:
        graft = task("graft")
        graft["graft"]["integration_type"] = "R"
        graft["graft"]["initial_goal"] = goal([item_id("graft", 0)])
        graft["goal"] = goal([item_id("graft", 1)])

        checked = check_run(
            graft,
            record(
                graft["id"], "graft", "injected", [item_id("graft", 1)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {"type": "instruction_injection", "step": 2, "trigger_reason": "task_timeline"},
                ],
            ),
        )

        self.assertFalse(checked["final_state_requires_a"])
        self.assertIsNone(checked["final_state_contains_a"])

    def test_common_success_macro_averages_tasks_before_replicates(self) -> None:
        rows = [
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": True, "e2e_success": True},
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": True, "e2e_success": True},
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": True, "e2e_success": True},
            {"task_id": "g2", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": True, "e2e_success": False},
        ]

        report = aggregate(rows)

        self.assertEqual(report["common"]["e2e_success"], 0.5)

    def test_evidence_status_reports_supported_success_bounds(self) -> None:
        rows = [
            {"task_id": "g1", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": True, "evidence_status": "PASS", "e2e_success": True},
            {"task_id": "g2", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": True, "evidence_status": "FAIL", "e2e_success": False},
            {"task_id": "g3", "track": "graft", "subtype": "C", "arm": "a_only", "run_valid": False, "evidence_status": "UNKNOWN", "e2e_success": None},
        ]

        evidence = aggregate(rows)["common"]["evidence"]

        self.assertEqual(evidence["unknown_rate"], 1 / 3)
        self.assertEqual(evidence["success_lower_bound"], 1 / 3)
        self.assertEqual(evidence["success_upper_bound"], 2 / 3)

    def test_three_replicate_reliability_is_reported_separately_from_pass_at_one(self) -> None:
        rows: list[dict[str, Any]] = []
        for task_id, outcomes in (("g1", [True, True, True]), ("g2", [True, True, False])):
            rows.extend(
                {
                    "task_id": task_id,
                    "track": "graft",
                    "subtype": "C",
                    "arm": "a_only",
                    "replicate_id": replicate_id,
                    "run_valid": True,
                    "e2e_success": outcome,
                }
                for replicate_id, outcome in enumerate(outcomes)
            )

        reliability = aggregate(rows)["common"]["reliability"]

        self.assertAlmostEqual(reliability["pass_at_1"], 5 / 6)
        self.assertEqual(reliability["pass_3"], 0.5)

    def test_clarification_utility_reports_predeclared_ask_cost_sensitivity(self) -> None:
        clarify = task("clarify")
        interactive = check_run(
            clarify,
            record(
                clarify["id"], "clarify", "interactive", [item_id("clarify", 1)],
                [
                    clarification_answer(clarify, 2),
                    {"type": "select", "step": 3, "item_id": item_id("clarify", 1)},
                ],
            ),
        )
        forced = check_run(
            clarify, record(clarify["id"], "clarify", "forced_execute", [item_id("clarify", 0)])
        )

        sensitivity = aggregate([interactive, forced])["clarify"]["diagnostic_utility_sensitivity"]

        self.assertAlmostEqual(sensitivity["lambda_0.10"], 0.9)

    def test_track_metrics_are_arm_differences(self) -> None:
        graft = task("graft")
        refresh = task("refresh")
        clarify = task("clarify")
        checked = []
        for arm, selected in (("a_only", [item_id("graft", 0)]), ("bundled", [item_id("graft", 0)]), ("injected", [])):
            actions = ([
                {"type": "search", "step": 1, "results": 4},
                {"type": "instruction_injection", "step": 2, "trigger_reason": "task_timeline"},
            ] if arm == "injected" else [])
            checked.append(check_run(graft, record(graft["id"], "graft", arm, selected, actions)))
        for arm, selected in (("stale", [item_id("refresh", 0)]), ("stream", [item_id("refresh", 1)]), ("oracle_latest", [item_id("refresh", 1)])):
            actions = ([
                {"type": "search", "step": 1, "results": 4},
                {"type": "visual_context_update", "step": 2, "trigger_reason": "task_timeline"},
                {"type": "select", "step": 4, "item_id": item_id("refresh", 1)},
            ] if arm == "stream" else [])
            checked.append(check_run(refresh, record(refresh["id"], "refresh", arm, selected, actions)))
        for arm, selected in (("oracle", [item_id("clarify", 1)]), ("interactive", [item_id("clarify", 1)]), ("forced_execute", [item_id("clarify", 0)])):
            actions = ([
                clarification_answer(clarify, 2),
                {"type": "open_product", "step": 3, "item_id": item_id("clarify", 1)},
                {"type": "submit", "step": 4, "item_ids": [item_id("clarify", 1)]},
            ] if arm == "interactive" else [])
            checked.append(check_run(clarify, record(clarify["id"], "clarify", arm, selected, actions)))

        report = aggregate(checked)
        self.assertEqual(report["graft"]["diagnostic_interruption_cost"], 1.0)
        self.assertEqual(report["refresh"]["diagnostic_replace_adaptation_benefit"], 1.0)
        self.assertEqual(report["refresh"]["diagnostic_replace_online_refresh_cost"], 0.0)
        self.assertEqual(report["clarify"]["diagnostic_clarification_gain"], 1.0)
        self.assertEqual(report["clarify"]["valid_question_rate"], 1.0)
        self.assertEqual(report["breakdown"]["refresh"]["replace"]["runs"], 3)
        intervals = bootstrap_intervals(checked, samples=20)
        self.assertIn("graft.diagnostic_interruption_cost", intervals)


if __name__ == "__main__":
    unittest.main()
