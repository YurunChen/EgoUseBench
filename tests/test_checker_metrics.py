from __future__ import annotations

import unittest
from typing import Any

from egocua.checker import behavioral_failure_mode, behavioral_symptoms, check_run
from egocua.metrics import aggregate, source_intervals
from egocua.timeline import Condition
from tests.conditions import CONDITIONS
from tests.helpers import complete_run_record, item_id, task


def record(
    task_id: str,
    track: str,
    condition_name: str,
    selected: list[str],
    actions: list[dict] | None = None,
) -> dict:
    return complete_run_record(
        {
            "task_id": task_id,
            "track": track,
            "condition": CONDITIONS[condition_name].as_dict(),
            "meta": {"default_release_step": 2, "reference_steps": None},
            "steps": 5,
            "actions": actions or [],
            "final_state": {
                "selected_entity_ids": selected,
                "fields": {},
                "submission_completed": True,
            },
        }
    )


def goal(*entity_options: list[str]) -> dict[str, Any]:
    return {
        "target_granularity": "variant",
        "acceptable_states": [
            {
                "selected_entity_ids": entities,
                "fields": {},
                "submission_completed": True,
            }
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


def retained_refresh(*choices: list[str]) -> dict[str, Any]:
    definition = task("refresh")
    definition["refresh"]["policy"] = "retain"
    definition["goal"] = goal(*(choices or ([item_id("refresh", 0)],)))
    return definition


def selection_change(before: list[str], after: list[str], step: int) -> dict[str, Any]:
    return {
        "type": "select_entity" if after else "deselect_entity",
        "step": step,
        "entity_id": (after or before)[-1],
        "state_before": {"selected_entity_ids": before, "fields": {}},
        "state_after": {"selected_entity_ids": after, "fields": {}},
    }


def refresh_release() -> dict[str, Any]:
    return {
        "type": "visual_context_update",
        "step": 2,
        "trigger_reason": "task_timeline",
    }


class CheckerMetricsTest(unittest.TestCase):
    def test_unsubmitted_record_without_live_state_does_not_infer_selection(
        self,
    ) -> None:
        definition = task("graft")
        run = record(definition["id"], "graft", "upfront", [])
        run["final_state"]["submission_completed"] = False
        run.pop("live_state")

        checked = check_run(definition, run)

        self.assertIsNone(checked["goal_state_correct"])
        self.assertNotIn("wrong_final_state", behavioral_symptoms(checked))

    def test_unsubmitted_state_diagnostics_require_live_evidence(self) -> None:
        definition = task("refresh")
        old, target = item_id("refresh", 0), item_id("refresh", 1)
        run = record(
            definition["id"],
            "refresh",
            "online",
            [],
            [
                {"type": "select_entity", "step": 1, "entity_id": old},
                {
                    "type": "visual_context_update",
                    "step": 2,
                    "trigger_reason": "task_timeline",
                },
                {"type": "deselect_entity", "step": 3, "entity_id": old},
                {"type": "select_entity", "step": 4, "entity_id": target},
            ],
        )
        run["final_state"]["submission_completed"] = False
        run.pop("live_state")

        unknown = check_run(definition, run)
        for field in (
            "false_update",
            "stale_final",
            "missing_after_update",
            "revised",
            "final_in_tempting",
        ):
            self.assertIsNone(unknown[field], field)

        run["live_state"] = {"selected_entity_ids": [target], "fields": {}}
        correct = check_run(definition, run)
        self.assertIsNone(correct["false_update"])
        self.assertFalse(correct["stale_final"])
        self.assertFalse(correct["missing_after_update"])
        self.assertTrue(correct["revised"])

        run["live_state"] = {"selected_entity_ids": [old], "fields": {}}
        wrong = check_run(definition, run)
        self.assertTrue(wrong["stale_final"])
        self.assertTrue(wrong["missing_after_update"])
        self.assertFalse(wrong["revised"])
        self.assertIn("wrong_final_state", behavioral_symptoms(wrong))

        run["final_state"].update(
            {"selected_entity_ids": [target], "submission_completed": True}
        )
        committed = check_run(definition, run)
        self.assertFalse(committed["e2e_success"])
        self.assertTrue(committed["stale_final"])
        self.assertTrue(committed["missing_after_update"])

    def test_live_edits_after_submission_determine_outcome(self) -> None:
        definition = task("graft")
        target = definition["goal"]["acceptable_states"][0]["selected_entity_ids"]
        run = record(definition["id"], "graft", "upfront", target)
        run["live_state"] = {"selected_entity_ids": [], "fields": {}}

        checked = check_run(definition, run)

        self.assertFalse(checked["goal_state_correct"])
        self.assertFalse(checked["e2e_success"])

    def test_missing_commitment_release_is_invalid_after_a_delivery_opportunity(
        self,
    ) -> None:
        definition = task("graft")
        run = record(
            definition["id"],
            "graft",
            "online",
            [],
            [
                {"type": "select_entity", "step": 1, "entity_id": item_id("graft", 0)},
                {"type": "search", "step": 2, "query": "another item"},
            ],
        )
        run["meta"]["runtime"]["release_trigger"] = "first_native_commitment"

        checked = check_run(definition, run)

        self.assertFalse(checked["run_valid"])
        self.assertIn(
            "expected_one_instruction_injection", checked["run_invalid_reasons"]
        )

    def test_retain_sufficiency_stays_at_start_under_commitment_release(self) -> None:
        definition = task("refresh")
        definition["refresh"]["policy"] = "retain"
        run = record(
            definition["id"],
            "refresh",
            "online",
            [],
            [
                {
                    "type": "select_entity",
                    "step": 1,
                    "entity_id": item_id("refresh", 0),
                },
                {
                    "type": "visual_context_update",
                    "step": 2,
                    "trigger_reason": "first_native_commitment",
                },
            ],
        )
        run["meta"]["runtime"]["release_trigger"] = "first_native_commitment"

        self.assertEqual(check_run(definition, run)["sufficiency_step"], 0)

    def test_workload_is_derived_from_domain_actions(self) -> None:
        graft = task("graft")
        checked = check_run(
            graft,
            record(
                graft["id"],
                "graft",
                "initial_only",
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
            record(
                graft["id"],
                "graft",
                "initial_only",
                [item_id("graft", 0), item_id("graft", 0)],
            ),
        )

        self.assertFalse(checked["goal_state_correct"])

    def test_exact_goal_state_rejects_partial_extra_and_unknown_submissions(
        self,
    ) -> None:
        graft = task("graft")
        graft["graft"]["transition"] = "augment"
        graft["graft"]["initial_goal"] = goal([item_id("graft", 0)])
        graft["goal"] = goal([item_id("graft", 0), item_id("graft", 1)])

        for submitted in (
            [item_id("graft", 0)],
            [item_id("graft", 0), item_id("graft", 1), item_id("graft", 2)],
            ["unknown-id"],
        ):
            with self.subTest(submitted=submitted):
                checked = check_run(
                    graft, record(graft["id"], "graft", "upfront", list(submitted))
                )
                self.assertFalse(checked["goal_state_correct"])

    def test_incomplete_final_state_is_unknown_and_not_a_valid_failure(self) -> None:
        graft = task("graft")
        incomplete = record(graft["id"], "graft", "initial_only", [item_id("graft", 0)])
        incomplete.pop("live_state")

        checked = check_run(graft, incomplete)

        self.assertFalse(checked["run_valid"])
        self.assertEqual(checked["evidence_status"], "UNKNOWN")
        self.assertIsNone(checked["e2e_success"])

    def test_progressive_condition_requires_exactly_one_timeline_event(self) -> None:
        graft = task("graft")
        missing = check_run(
            graft, record(graft["id"], "graft", "online", [item_id("graft", 0)])
        )
        delivered = check_run(
            graft,
            record(
                graft["id"],
                "graft",
                "online",
                [item_id("graft", 0)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "instruction_injection",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                ],
            ),
        )
        early = check_run(
            graft,
            record(
                graft["id"],
                "graft",
                "online",
                [item_id("graft", 0)],
                [
                    {
                        "type": "instruction_injection",
                        "step": 1,
                        "trigger_reason": "task_timeline",
                    }
                ],
            ),
        )
        duplicated = check_run(
            graft,
            record(
                graft["id"],
                "graft",
                "online",
                [item_id("graft", 0)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "instruction_injection",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                    {
                        "type": "instruction_injection",
                        "step": 3,
                        "trigger_reason": "task_timeline",
                    },
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
                graft["id"],
                "graft",
                "online",
                [item_id("graft", 0)],
                [
                    {
                        "type": "instruction_injection",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    }
                ],
            ),
        )

        self.assertTrue(checked["run_valid"])
        self.assertEqual(checked["event_trigger_reason"], "task_timeline")

    def test_submission_before_a_required_event_is_measured_not_invalid(self) -> None:
        graft = task("graft")
        checked = check_run(
            graft,
            record(
                graft["id"],
                "graft",
                "online",
                [item_id("graft", 0)],
                [
                    {"type": "submit", "step": 1, "item_ids": [item_id("graft", 0)]},
                    {
                        "type": "instruction_injection",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                ],
            ),
        )

        self.assertTrue(checked["run_valid"])
        self.assertTrue(checked["premature_submission"])
        self.assertEqual(checked["sufficiency_step"], 2)

    def test_commitment_triggered_release_uses_actual_delivery_as_sufficiency(
        self,
    ) -> None:
        graft = task("graft")
        run = record(
            graft["id"],
            graft["track"],
            "online",
            [item_id("graft", 0)],
            [
                {"type": "add_to_cart", "step": 2, "item_id": item_id("graft", 1)},
                {
                    "type": "instruction_injection",
                    "step": 3,
                    "trigger_reason": "first_native_commitment",
                },
                {"type": "submit", "step": 3, "item_ids": [item_id("graft", 0)]},
            ],
        )
        run["meta"]["default_release_step"] = 12
        run["meta"]["runtime"]["release_trigger"] = "first_native_commitment"

        checked = check_run(graft, run)

        self.assertTrue(checked["run_valid"])
        self.assertEqual(checked["sufficiency_step"], 3)
        self.assertFalse(checked["premature_submission"])

    def test_no_commitment_before_conditional_release_is_a_valid_failure(self) -> None:
        refresh = task("refresh")
        run = record(
            refresh["id"],
            refresh["track"],
            "online",
            [],
            [{"type": "search", "step": 1, "query": "wooden spoon"}],
        )
        run["final_state"]["submission_completed"] = False
        run["meta"]["default_release_step"] = 12
        run["meta"]["runtime"]["release_trigger"] = "first_native_commitment"

        checked = check_run(refresh, run)

        self.assertTrue(checked["run_valid"])
        self.assertFalse(checked["e2e_success"])
        self.assertFalse(checked["release_delivered"])

    def test_commitment_triggered_release_may_follow_the_default_timeline_step(
        self,
    ) -> None:
        graft = task("graft")
        run = record(
            graft["id"],
            graft["track"],
            "online",
            [item_id("graft", 0)],
            [
                {"type": "search", "step": 2, "query": "item"},
                {"type": "open_entity", "step": 3, "item_id": item_id("graft", 1)},
                {"type": "add_to_cart", "step": 4, "item_id": item_id("graft", 1)},
                {
                    "type": "instruction_injection",
                    "step": 5,
                    "trigger_reason": "first_native_commitment",
                },
                {"type": "submit", "step": 6, "item_ids": [item_id("graft", 0)]},
            ],
        )
        run["meta"]["default_release_step"] = 2
        run["meta"]["runtime"]["release_trigger"] = "first_native_commitment"

        checked = check_run(graft, run)

        self.assertTrue(checked["run_valid"])
        self.assertEqual(checked["sufficiency_step"], 5)

    def test_retain_terminal_state_alone_does_not_prove_an_update(self) -> None:
        refresh = retained_refresh()
        for selected in ([], [item_id("refresh", 0)], [item_id("refresh", 2)]):
            with self.subTest(selected=selected):
                checked = check_run(
                    refresh, record(refresh["id"], "refresh", "online", selected)
                )
                self.assertIsNone(checked["false_update"])

    def test_retain_wrong_choice_before_release_then_done_is_not_false_update(
        self,
    ) -> None:
        refresh = retained_refresh()
        wrong = [item_id("refresh", 2)]
        checked = check_run(
            refresh,
            record(
                refresh["id"],
                "refresh",
                "online",
                wrong,
                [
                    selection_change([], wrong, 1),
                    refresh_release(),
                    {"type": "done", "step": 3},
                ],
            ),
        )

        self.assertTrue(checked["run_valid"])
        self.assertFalse(checked["false_update"])
        self.assertFalse(checked["e2e_success"])
        self.assertNotIn("incorrect_change_after_update", behavioral_symptoms(checked))

    def test_retain_delivered_release_without_mutations_is_false(self) -> None:
        refresh = retained_refresh()
        checked = check_run(
            refresh,
            record(refresh["id"], "refresh", "online", [], [refresh_release()]),
        )

        self.assertFalse(checked["false_update"])

    def test_retain_post_release_mutation_without_state_evidence_is_unknown(
        self,
    ) -> None:
        refresh = retained_refresh()
        wrong = item_id("refresh", 2)
        checked = check_run(
            refresh,
            record(
                refresh["id"],
                "refresh",
                "online",
                [wrong],
                [
                    refresh_release(),
                    {"type": "select_entity", "step": 3, "entity_id": wrong},
                ],
            ),
        )

        self.assertIsNone(checked["false_update"])

    def test_retain_legal_state_changed_after_release_is_false_update(self) -> None:
        refresh = retained_refresh()
        correct, wrong = [item_id("refresh", 0)], [item_id("refresh", 2)]
        checked = check_run(
            refresh,
            record(
                refresh["id"],
                "refresh",
                "online",
                wrong,
                [
                    selection_change([], correct, 1),
                    refresh_release(),
                    selection_change(correct, wrong, 3),
                ],
            ),
        )

        self.assertTrue(checked["false_update"])
        self.assertIn("incorrect_change_after_update", behavioral_symptoms(checked))

    def test_retain_repaired_false_update_is_an_event_not_a_failure(self) -> None:
        refresh = retained_refresh()
        correct, wrong = [item_id("refresh", 0)], [item_id("refresh", 2)]
        checked = check_run(
            refresh,
            record(
                refresh["id"],
                "refresh",
                "online",
                correct,
                [
                    selection_change([], correct, 1),
                    refresh_release(),
                    selection_change(correct, wrong, 3),
                    selection_change(wrong, correct, 4),
                ],
            ),
        )

        self.assertTrue(checked["false_update"])
        self.assertTrue(checked["e2e_success"])
        self.assertEqual(behavioral_failure_mode(checked), "success")
        self.assertNotIn("incorrect_change_after_update", behavioral_symptoms(checked))

    def test_retain_false_update_uses_event_order_within_one_step(self) -> None:
        refresh = retained_refresh()
        correct, wrong = [item_id("refresh", 0)], [item_id("refresh", 2)]
        change = selection_change(correct, wrong, 2)
        for actions, expected in (
            ([change, refresh_release()], False),
            ([refresh_release(), change], True),
        ):
            with self.subTest(expected=expected):
                checked = check_run(
                    refresh,
                    record(refresh["id"], "refresh", "online", wrong, actions),
                )
                self.assertIs(checked["false_update"], expected)

    def test_retain_false_update_detects_removal_duplicates_and_field_edits(
        self,
    ) -> None:
        refresh = retained_refresh()
        correct = [item_id("refresh", 0)]
        field_edit = selection_change(correct, correct, 3)
        field_edit["type"] = "set_field"
        field_edit["state_after"]["fields"] = {"unrequested": "value"}
        for change in (
            selection_change(correct, [], 3),
            selection_change(correct, correct * 2, 3),
            field_edit,
        ):
            with self.subTest(change=change):
                run = record(
                    refresh["id"],
                    "refresh",
                    "online",
                    change["state_after"]["selected_entity_ids"],
                    [refresh_release(), change],
                )
                run["live_state"] = change["state_after"]
                self.assertTrue(check_run(refresh, run)["false_update"])

    def test_retain_noop_or_already_wrong_edit_does_not_prove_legal_state_loss(
        self,
    ) -> None:
        refresh = retained_refresh()
        wrong = [item_id("refresh", 2)]
        for after in (wrong, [item_id("refresh", 3)]):
            with self.subTest(after=after):
                checked = check_run(
                    refresh,
                    record(
                        refresh["id"],
                        "refresh",
                        "online",
                        after,
                        [refresh_release(), selection_change(wrong, after, 3)],
                    ),
                )
                self.assertFalse(checked["false_update"])

    def test_retain_state_gaps_remain_unknown(self) -> None:
        refresh = retained_refresh()
        correct, wrong = [item_id("refresh", 0)], [item_id("refresh", 2)]
        for actions in (
            [selection_change([], correct, 1), refresh_release()],
            [
                selection_change([], correct, 1),
                refresh_release(),
                selection_change(wrong, wrong, 3),
            ],
            [
                refresh_release(),
                {
                    "type": "select_entity",
                    "step": 3,
                    "entity_id": wrong[0],
                    "state_after": {"selected_entity_ids": wrong, "fields": {}},
                },
            ],
        ):
            with self.subTest(actions=actions):
                checked = check_run(
                    refresh,
                    record(refresh["id"], "refresh", "online", wrong, actions),
                )
                self.assertIsNone(checked["false_update"])

    def test_retain_missing_terminal_state_keeps_diagnostic_unknown(self) -> None:
        refresh = retained_refresh()
        run = record(refresh["id"], "refresh", "online", [], [refresh_release()])
        run.pop("live_state")

        self.assertIsNone(check_run(refresh, run)["false_update"])

    def test_retain_accepts_either_verified_alternative(self) -> None:
        alternatives = [item_id("refresh", 0), item_id("refresh", 2)]
        refresh = retained_refresh(*[[entity_id] for entity_id in alternatives])

        for entity_id in alternatives:
            with self.subTest(entity_id=entity_id):
                checked = check_run(
                    refresh,
                    record(
                        refresh["id"],
                        "refresh",
                        "online",
                        [entity_id],
                        [
                            refresh_release(),
                            selection_change([alternatives[0]], [entity_id], 3),
                        ],
                    ),
                )

                self.assertFalse(checked["false_update"])

    def test_retain_reports_an_entity_outside_all_verified_alternatives(self) -> None:
        refresh = retained_refresh([item_id("refresh", 0)], [item_id("refresh", 2)])

        checked = check_run(
            refresh,
            record(
                refresh["id"],
                "refresh",
                "online",
                [item_id("refresh", 3)],
                [
                    refresh_release(),
                    selection_change(
                        [item_id("refresh", 0)], [item_id("refresh", 3)], 3
                    ),
                ],
            ),
        )

        self.assertTrue(checked["false_update"])

    def test_retain_does_not_accept_all_alternatives_as_one_combined_choice(
        self,
    ) -> None:
        alternatives = [item_id("refresh", 0), item_id("refresh", 2)]
        refresh = retained_refresh(*[[entity_id] for entity_id in alternatives])

        checked = check_run(
            refresh,
            record(
                refresh["id"],
                "refresh",
                "online",
                alternatives,
                [
                    refresh_release(),
                    selection_change([alternatives[0]], alternatives, 3),
                ],
            ),
        )

        self.assertTrue(checked["false_update"])

    def test_answer_utilization_requires_the_first_post_answer_candidate_action(
        self,
    ) -> None:
        clarify = task("clarify")
        asked_success = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [item_id("clarify", 1)],
                [
                    clarification_answer(clarify, 2),
                    {
                        "type": "open_product",
                        "step": 3,
                        "item_id": item_id("clarify", 1),
                    },
                ],
            ),
        )
        no_ask_failure = check_run(
            clarify,
            record(clarify["id"], "clarify", "online", [item_id("clarify", 0)]),
        )
        report = aggregate([asked_success, no_ask_failure])
        self.assertEqual(report["clarification_policy"]["answer_utilization"], 1.0)

    def test_recorded_rejection_and_missing_answer_are_diagnostic(self) -> None:
        clarify = task("clarify")
        target = item_id("clarify", 1)
        historical_rejection = {
            "type": "clarification_answer",
            "step": 1,
            "question": "What is the weather?",
            "valid": False,
            "payload": "No answer was given.",
        }
        correct_state_after_bad_question = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [target],
                [historical_rejection, {"type": "select", "step": 2, "item_id": target}],
            ),
        )
        missing_answer = clarification_answer(clarify, 1)
        missing_answer.pop("payload")
        correct_state_without_answer = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [target],
                [missing_answer, {"type": "select", "step": 2, "item_id": target}],
            ),
        )

        self.assertFalse(correct_state_after_bad_question["valid_question"])
        self.assertTrue(correct_state_after_bad_question["e2e_success"])
        self.assertFalse(correct_state_without_answer["answer_delivered"])
        self.assertTrue(correct_state_without_answer["e2e_success"])

    def test_question_timing_and_answer_use_are_diagnostic(self) -> None:
        clarify = task("clarify")
        target = item_id("clarify", 1)
        late = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [target],
                [
                    {"type": "select", "step": 1, "item_id": target},
                    clarification_answer(clarify, 2),
                ],
            ),
        )
        submitted_before_ask = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [target],
                [
                    {"type": "submit", "step": 1},
                    clarification_answer(clarify, 2),
                ],
            ),
        )
        ignored = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [target],
                [clarification_answer(clarify, 1)],
            ),
        )
        successful = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [target],
                [
                    clarification_answer(clarify, 1),
                    {"type": "open_product", "step": 2, "item_id": target},
                ],
            ),
        )

        self.assertFalse(late["timely_question"])
        self.assertTrue(late["e2e_success"])
        self.assertFalse(submitted_before_ask["timely_question"])
        self.assertTrue(submitted_before_ask["e2e_success"])
        self.assertFalse(ignored["answer_used"])
        self.assertTrue(ignored["e2e_success"])
        self.assertTrue(successful["answer_used"])
        self.assertTrue(successful["e2e_success"])

    def test_clarify_answer_use_accepts_a_declared_site_equivalent(self) -> None:
        clarify = task("clarify")
        equivalent = item_id("clarify", 2)
        clarify["goal"] = goal(
            [item_id("clarify", 1)],
            [equivalent],
        )

        checked = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [equivalent],
                [
                    clarification_answer(clarify, 1),
                    {"type": "open_product", "step": 2, "item_id": equivalent},
                ],
            ),
        )

        self.assertTrue(checked["answer_used"])
        self.assertTrue(checked["e2e_success"])

    def test_site_equivalent_commitment_before_clarification_is_late(self) -> None:
        clarify = task("clarify")
        equivalent = item_id("clarify", 2)
        clarify["goal"] = goal(
            [item_id("clarify", 1)],
            [equivalent],
        )

        checked = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [equivalent],
                [
                    {"type": "select", "step": 1, "item_id": equivalent},
                    clarification_answer(clarify, 2),
                ],
            ),
        )

        self.assertFalse(checked["timely_question"])
        self.assertTrue(checked["e2e_success"])

    def test_nonrepresentative_choice_before_clarification_is_late(self) -> None:
        clarify = task("clarify")
        checked = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [item_id("clarify", 1)],
                [
                    {"type": "select", "step": 1, "item_id": "other_site_knife"},
                    clarification_answer(clarify, 2),
                    {
                        "type": "open_product",
                        "step": 3,
                        "item_id": item_id("clarify", 1),
                    },
                ],
            ),
        )
        self.assertFalse(checked["timely_question"])
        self.assertFalse(checked["answer_used"])
        self.assertTrue(checked["e2e_success"])

    def test_wrong_physical_candidate_after_answer_is_not_hidden_by_equivalent(
        self,
    ) -> None:
        clarify = task("clarify")
        equivalent = item_id("clarify", 2)
        clarify["goal"] = goal(
            [item_id("clarify", 1)],
            [equivalent],
        )

        checked = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [equivalent],
                [
                    clarification_answer(clarify, 1),
                    {
                        "type": "select",
                        "step": 2,
                        "item_id": item_id("clarify", 0),
                    },
                    {"type": "open_product", "step": 3, "item_id": equivalent},
                ],
            ),
        )

        self.assertFalse(checked["answer_used"])
        self.assertTrue(checked["e2e_success"])

    def test_oracle_unnecessary_question_is_diagnostic_not_an_e2e_failure(self) -> None:
        clarify = task("clarify")
        target = item_id("clarify", 1)
        checked = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "upfront",
                [target],
                [clarification_answer(clarify, 1)],
            ),
        )

        self.assertTrue(checked["over_ask"])
        self.assertTrue(checked["e2e_success"])

    def test_unneeded_question_rate_counts_runs_that_did_not_ask(self) -> None:
        graft = task("graft")
        rows = []
        for ask, actions in (
            ("open", []),
            ("open", [{"type": "clarification_answer", "step": 1, "valid": False}]),
            ("allowed", []),
        ):
            run = record(graft["id"], "graft", "online", [])
            run["condition"] = {**run["condition"], "ask": ask}
            run["actions"] = actions
            rows.append(check_run(graft, run)["over_ask"])

        self.assertEqual(rows, [False, True, None])

    def test_second_clarification_attempt_is_auditable(self) -> None:
        clarify = task("clarify")
        checked = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [item_id("clarify", 1)],
                [
                    clarification_answer(clarify, 1),
                    {"type": "clarification_answer", "step": 2, "limit_reached": True},
                    {"type": "select", "step": 3, "item_id": item_id("clarify", 1)},
                ],
            ),
        )

        self.assertEqual(checked["question_attempts"], 2)
        self.assertTrue(checked["question_limit_exceeded"])

    def test_answer_utilization_excludes_questions_after_a_candidate_commitment(
        self,
    ) -> None:
        clarify = task("clarify")
        timely_failure = check_run(
            clarify,
            record(
                "c1",
                "clarify",
                "online",
                [item_id("clarify", 0)],
                [
                    clarification_answer(clarify, 2),
                    {
                        "type": "open_product",
                        "step": 3,
                        "item_id": item_id("clarify", 0),
                    },
                ],
            ),
        )
        late_success = check_run(
            clarify,
            record(
                "c2",
                "clarify",
                "online",
                [item_id("clarify", 1)],
                [
                    {"type": "select", "step": 2, "item_id": item_id("clarify", 1)},
                    clarification_answer(clarify, 3),
                ],
            ),
        )

        report = aggregate([timely_failure, late_success])

        self.assertEqual(report["clarification_policy"]["answer_utilization"], 0.0)

    def test_condition_effects_pair_tasks_not_uncontrolled_seed_labels(self) -> None:
        complete_at_start = Condition(
            "complete_at_start",
            "upfront",
            "full",
            "allowed",
            None,
        )
        progressive = Condition(
            "progressive",
            "online",
            "full",
            "allowed",
            None,
        )
        timing_probe = Condition(
            "timing_probe",
            "online",
            "full",
            "allowed",
            0.5,
        )
        rows = [
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": complete_at_start.as_dict(),
                "seed": 11,
                "replicate_id": 0,
                "run_valid": True,
                "e2e_success": True,
            },
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": progressive.as_dict(),
                "seed": 29,
                "replicate_id": 1,
                "run_valid": True,
                "e2e_success": False,
            },
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": timing_probe.as_dict(),
                "seed": 37,
                "replicate_id": 2,
                "run_valid": True,
                "e2e_success": True,
            },
            {
                "task_id": "g2",
                "track": "graft",
                "subtype": "refine",
                "condition": complete_at_start.as_dict(),
                "seed": 7,
                "replicate_id": 0,
                "run_valid": True,
                "e2e_success": False,
            },
        ]
        report = aggregate(rows)
        self.assertEqual(
            report["paired_effects"]["graft"]["refine"]["upfront_minus_online"],
            1.0,
        )

    def test_invalid_runs_are_reported_but_excluded_from_agent_success(self) -> None:
        rows = [
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": True,
                "e2e_success": True,
            },
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": False,
                "e2e_success": None,
            },
        ]

        report = aggregate(rows)

        self.assertEqual(report["common"]["valid_run_rate"], 0.5)
        self.assertEqual(report["common"]["e2e_success"], 1.0)
        self.assertNotIn("capability_filtered", report)

    def test_clarification_policy_reports_balanced_ask_f1(self) -> None:
        clarify = task("clarify")
        rows: list[dict[str, Any]] = []
        for task_id, condition_name, selected, actions in (
            (
                "c1",
                "online",
                [item_id("clarify", 1)],
                [clarification_answer(clarify, 2)],
            ),
            ("c2", "online", [item_id("clarify", 0)], []),
            ("c1", "upfront", [item_id("clarify", 1)], []),
            (
                "c2",
                "upfront",
                [item_id("clarify", 1)],
                [clarification_answer(clarify, 2)],
            ),
        ):
            current = dict(clarify)
            current["id"] = task_id
            rows.append(
                check_run(
                    current,
                    record(task_id, "clarify", condition_name, selected, actions),
                )
            )

        policy = aggregate(rows)["clarification_policy"]

        self.assertEqual(policy["ask_recall"], 0.5)
        self.assertEqual(policy["ask_precision"], 0.5)
        self.assertEqual(policy["ask_f1"], 0.5)

    def test_clarification_policy_handles_an_oracle_only_slice(self) -> None:
        clarify = task("clarify")
        checked = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "upfront",
                [item_id("clarify", 1)],
                [clarification_answer(clarify, 2)],
            ),
        )

        policy = aggregate([checked])["clarification_policy"]

        self.assertEqual(policy["ask_precision"], 0.0)
        self.assertIsNone(policy["ask_recall"])
        self.assertIsNone(policy["ask_f1"])

    def test_refresh_latency_reports_nonresponses_and_post_event_success_curve(
        self,
    ) -> None:
        refresh = task("refresh")
        success = check_run(
            refresh,
            record(
                "r1",
                "refresh",
                "online",
                [item_id("refresh", 1)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "visual_context_update",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                    {"type": "select", "step": 4, "item_id": item_id("refresh", 1)},
                ],
            ),
        )
        failure = check_run(
            refresh,
            record(
                "r2",
                "refresh",
                "online",
                [],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "visual_context_update",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                ],
            ),
        )

        metrics = aggregate([success, failure])["breakdown"]["refresh"]["replace"]

        self.assertEqual(metrics["post_release_recovery_latency"], 2.0)
        self.assertEqual(metrics["post_release_sr_at_3"], 0.5)

    def test_refresh_response_metrics_only_use_replace_tasks(self) -> None:
        refresh = task("refresh")
        replace_failure = check_run(
            refresh,
            record(
                "replace-1",
                "refresh",
                "online",
                [],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "visual_context_update",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
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
                "retain-1",
                "refresh",
                "online",
                [item_id("refresh", 0)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "visual_context_update",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                    {"type": "select", "step": 3, "item_id": item_id("refresh", 0)},
                ],
            ),
        )

        metrics = aggregate([replace_failure, retain_response])["breakdown"]["refresh"][
            "replace"
        ]

        self.assertIsNone(metrics["post_release_recovery_latency"])
        self.assertEqual(metrics["post_release_sr_at_3"], 0.0)

    def test_checker_emits_only_unified_diagnostics(self) -> None:
        graft = task("graft")
        graft["graft"]["transition"] = "replace"
        graft["graft"]["initial_goal"] = goal([item_id("graft", 0)])
        graft["goal"] = goal([item_id("graft", 1)])

        checked = check_run(
            graft,
            record(
                graft["id"],
                "graft",
                "online",
                [item_id("graft", 1)],
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "instruction_injection",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                ],
            ),
        )

        self.assertNotIn("final_state_requires_a", checked)
        self.assertNotIn("final_state_contains_a", checked)

    def test_common_success_macro_averages_tasks_before_replicates(self) -> None:
        rows = [
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": True,
                "e2e_success": True,
            },
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": True,
                "e2e_success": True,
            },
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": True,
                "e2e_success": True,
            },
            {
                "task_id": "g2",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": True,
                "e2e_success": False,
            },
        ]

        report = aggregate(rows)

        self.assertEqual(report["common"]["e2e_success"], 0.5)

    def test_evidence_status_reports_supported_success_bounds(self) -> None:
        rows = [
            {
                "task_id": "g1",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": True,
                "evidence_status": "PASS",
                "e2e_success": True,
            },
            {
                "task_id": "g2",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": True,
                "evidence_status": "FAIL",
                "e2e_success": False,
            },
            {
                "task_id": "g3",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS["initial_only"].as_dict(),
                "run_valid": False,
                "evidence_status": "UNKNOWN",
                "e2e_success": None,
            },
        ]

        evidence = aggregate(rows)["common"]["evidence"]

        self.assertEqual(evidence["unknown_rate"], 1 / 3)
        self.assertEqual(evidence["success_lower_bound"], 1 / 3)
        self.assertEqual(evidence["success_upper_bound"], 2 / 3)

    def test_three_replicate_reliability_is_reported_separately_from_pass_at_one(
        self,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for task_id, outcomes in (
            ("g1", [True, True, True]),
            ("g2", [True, True, False]),
        ):
            rows.extend(
                {
                    "task_id": task_id,
                    "track": "graft",
                    "subtype": "refine",
                    "condition": CONDITIONS["initial_only"].as_dict(),
                    "replicate_id": replicate_id,
                    "run_valid": True,
                    "e2e_success": outcome,
                }
                for replicate_id, outcome in enumerate(outcomes)
            )

        reliability = aggregate(rows)["common"]["reliability"]

        self.assertAlmostEqual(reliability["pass_at_1"], 5 / 6)
        self.assertEqual(reliability["pass_3"], 0.5)

    def test_clarification_gain_is_a_condition_pair(self) -> None:
        clarify = task("clarify")
        interactive = check_run(
            clarify,
            record(
                clarify["id"],
                "clarify",
                "online",
                [item_id("clarify", 1)],
                [
                    clarification_answer(clarify, 2),
                    {"type": "select", "step": 3, "item_id": item_id("clarify", 1)},
                ],
            ),
        )
        forced = check_run(
            clarify, record(clarify["id"], "clarify", "no_ask", [item_id("clarify", 0)])
        )

        effect = aggregate([interactive, forced])["paired_effects"]["clarify"][
            "resolve"
        ]

        self.assertEqual(effect["allowed_minus_denied"], 1.0)

    def test_track_metrics_are_condition_differences(self) -> None:
        graft = task("graft")
        refresh = task("refresh")
        clarify = task("clarify")
        checked = []
        for condition_name, selected in (
            ("initial_only", [item_id("graft", 0)]),
            ("upfront", [item_id("graft", 0)]),
            ("online", []),
        ):
            actions = (
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "instruction_injection",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                ]
                if condition_name == "online"
                else []
            )
            checked.append(
                check_run(
                    graft,
                    record(graft["id"], "graft", condition_name, selected, actions),
                )
            )
        for condition_name, selected in (
            ("initial_only", [item_id("refresh", 0)]),
            ("online", [item_id("refresh", 1)]),
            ("upfront", [item_id("refresh", 1)]),
        ):
            actions = (
                [
                    {"type": "search", "step": 1, "results": 4},
                    {
                        "type": "visual_context_update",
                        "step": 2,
                        "trigger_reason": "task_timeline",
                    },
                    {"type": "select", "step": 4, "item_id": item_id("refresh", 1)},
                ]
                if condition_name == "online"
                else []
            )
            checked.append(
                check_run(
                    refresh,
                    record(refresh["id"], "refresh", condition_name, selected, actions),
                )
            )
        for condition_name, selected in (
            ("upfront", [item_id("clarify", 1)]),
            ("online", [item_id("clarify", 1)]),
            ("no_ask", [item_id("clarify", 0)]),
        ):
            actions = (
                [
                    clarification_answer(clarify, 2),
                    {
                        "type": "open_product",
                        "step": 3,
                        "item_id": item_id("clarify", 1),
                    },
                    {"type": "submit", "step": 4, "item_ids": [item_id("clarify", 1)]},
                ]
                if condition_name == "online"
                else []
            )
            checked.append(
                check_run(
                    clarify,
                    record(clarify["id"], "clarify", condition_name, selected, actions),
                )
            )

        report = aggregate(checked)
        self.assertEqual(
            report["paired_effects"]["graft"]["refine"]["upfront_minus_online"], 1.0
        )
        self.assertEqual(
            report["paired_effects"]["refresh"]["replace"]["full_minus_initial_only"],
            1.0,
        )
        self.assertEqual(
            report["paired_effects"]["refresh"]["replace"]["upfront_minus_online"], 0.0
        )
        self.assertEqual(
            report["paired_effects"]["clarify"]["resolve"]["allowed_minus_denied"], 1.0
        )
        self.assertEqual(report["clarification_policy"]["valid_question_rate"], 1.0)
        self.assertEqual(report["breakdown"]["refresh"]["replace"]["runs"], 3)
        intervals = source_intervals(
            checked, {row["task_id"]: row["task_id"] for row in checked}
        )
        graft = intervals["graft"]["online_minus_upfront"]
        self.assertEqual(graft["difference"], -1.0)
        self.assertEqual(graft["outcome_quadrants"], {"S/S": 0, "S/F": 1, "F/S": 0, "F/F": 0})
        self.assertEqual(graft["adaptability"], 0.0)
        self.assertNotIn("online_minus_initial_only", intervals["refresh"])
        self.assertEqual(
            intervals["refresh"]["by_subtype"]["replace"]["online_minus_initial_only"][
                "difference"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
