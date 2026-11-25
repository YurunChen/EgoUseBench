from __future__ import annotations

import unittest
from typing import Any

from egocua.checker import check_run
from egocua.timeline import Condition, Timeline
from egocua.truth import Truth
from tests.conditions import CONDITIONS
from tests.helpers import complete_run_record, item_id, task


def _record(
    condition_name: str,
    actions: list[dict[str, Any]],
    submitted: list[str],
) -> dict[str, Any]:
    condition = CONDITIONS.get(
        condition_name,
        Condition(condition_name, "online", "full", "allowed", None),
    )
    return complete_run_record(
        {
            "condition": condition.as_dict(),
            "run_status": {"valid": True, "reason": None},
            "steps": max((int(a.get("step", 0)) for a in actions), default=0),
            "actions": actions,
            "final_state": {
                "selected_entity_ids": submitted,
                "fields": {},
                "submission_completed": True,
            },
            "meta": {
                "replicate_id": 0,
                "default_release_step": 2,
                "reference_steps": None,
            },
        }
    )


def _v1_event(step: int) -> dict[str, Any]:
    timeline = Timeline.compile(
        task("refresh"),
        CONDITIONS["online"],
        default_release_step=step,
        reference_steps=None,
    )
    return timeline.releases[0].as_event()


def _b_event(step: int) -> dict[str, Any]:
    return {
        "type": "instruction_injection",
        "step": step,
        "payload": "Also apply B",
        "trigger_reason": "task_timeline",
    }


class TruthTest(unittest.TestCase):
    def test_refresh_initial_alternatives_share_one_truth(self) -> None:
        definition = task("refresh")
        alternative = item_id("refresh", 2)
        definition["truth"]["initial_acceptable_states"].append(
            {"selected_entity_ids": [alternative], "fields": {}}
        )
        truth = Truth.of(definition)
        self.assertEqual(
            truth.initial_referent_ids, {item_id("refresh", 0), alternative}
        )
        self.assertEqual(truth.invalidated_referent_ids, truth.initial_referent_ids)
        self.assertTrue(truth.initial_goal_differs)
        actions = [
            _v1_event(2),
            {"type": "select_entity", "step": 3, "entity_id": alternative},
        ]
        result = check_run(definition, _record("online", actions, [alternative]))
        self.assertTrue(result["stale_final"])
        self.assertFalse(result["e2e_success"])

    def test_missing_historical_initial_truth_does_not_guess_or_change_e2e(
        self,
    ) -> None:
        definition = task("refresh")
        definition["truth"].pop("initial_acceptable_states")
        truth = Truth.of(definition)
        self.assertIsNone(truth.initial_acceptable_states)
        self.assertIsNone(truth.initial_referent_ids)
        self.assertIsNone(truth.invalidated_referent_ids)
        self.assertIsNone(truth.required_after_update_ids)
        actions = [
            _v1_event(2),
            {"type": "select_entity", "step": 3, "entity_id": item_id("refresh", 1)},
        ]
        record = _record("online", actions, [item_id("refresh", 1)])
        record["task_id"] = definition["id"]
        record["task_snapshot"] = definition
        checked = check_run(task("refresh"), record)
        self.assertTrue(checked["e2e_success"])
        self.assertIsNone(checked["stale_final"])
        self.assertIsNone(checked["missing_after_update"])

    def test_refresh_replace_tempting_is_the_initial_referent(self) -> None:
        truth = Truth.of(task("refresh"))
        self.assertEqual(truth.initial_referent_ids, {item_id("refresh", 0)})
        self.assertEqual(truth.tempting_referent_ids, {item_id("refresh", 0)})
        self.assertTrue(truth.later_release_changes_referent)

    def test_graft_constraint_invalidates_the_other_initial_option(self) -> None:
        truth = Truth.of(task("graft"))
        self.assertEqual(truth.transition, "refine")
        self.assertEqual(truth.invalidated_referent_ids, {item_id("graft", 1)})
        self.assertEqual(truth.tempting_referent_ids, {item_id("graft", 1)})
        self.assertTrue(truth.later_release_changes_referent)

    def test_clarify_tempting_is_the_other_candidate(self) -> None:
        truth = Truth.of(task("clarify"))
        self.assertEqual(truth.tempting_referent_ids, {item_id("clarify", 0)})
        self.assertEqual(
            truth.candidate_ids, {item_id("clarify", 0), item_id("clarify", 1)}
        )


class TimingDiagnosticsTest(unittest.TestCase):
    def test_refresh_premature_commit_to_tempting_then_revised(self) -> None:
        old, new = item_id("refresh", 0), item_id("refresh", 1)
        actions = [
            {"type": "search", "step": 1, "query": "item"},
            _v1_event(2),
            {"type": "select_entity", "step": 1, "entity_id": old},
            {"type": "deselect_entity", "step": 3, "entity_id": old},
            {"type": "select_entity", "step": 4, "entity_id": new},
            {"type": "submit", "step": 5, "entity_ids": [new], "fields": {}},
        ]
        actions.sort(key=lambda a: a["step"])
        result = check_run(task("refresh"), _record("online", actions, [new]))
        self.assertEqual(result["condition"]["name"], "online")
        self.assertEqual(result["sufficiency_step"], 2)
        self.assertTrue(result["premature_commitment"])
        self.assertTrue(result["premature_to_tempting"])
        self.assertTrue(result["revised"])
        self.assertFalse(result["stale_final"])
        self.assertFalse(result["premature_submission"])
        self.assertEqual(result["productive_actions_before_sufficiency"], 1)
        self.assertTrue(result["e2e_success"])

    def test_refresh_stale_final_when_old_referent_survives(self) -> None:
        old = item_id("refresh", 0)
        actions = [
            {"type": "search", "step": 1, "query": "item"},
            _v1_event(2),
            {"type": "select_entity", "step": 3, "entity_id": old},
            {"type": "submit", "step": 4, "entity_ids": [old], "fields": {}},
        ]
        result = check_run(task("refresh"), _record("online", actions, [old]))
        self.assertFalse(result["premature_commitment"])
        self.assertTrue(result["stale_final"])
        self.assertFalse(result["e2e_success"])

    def test_graft_augment_early_a_is_not_premature(self) -> None:
        a_target = item_id("graft", 0)
        actions = [
            {"type": "search", "step": 1, "query": "A"},
            {"type": "select_entity", "step": 1, "entity_id": a_target},
            _b_event(2),
            {"type": "submit", "step": 3, "entity_ids": [a_target], "fields": {}},
        ]
        result = check_run(task("graft"), _record("online", actions, [a_target]))
        self.assertEqual(result["sufficiency_step"], 2)
        self.assertFalse(result["premature_commitment"])
        self.assertFalse(result["stale_final"])

    def test_graft_augment_reports_an_update_that_was_not_integrated(self) -> None:
        graft_task = task("graft")
        first, second = item_id("graft", 0), item_id("graft", 2)
        graft_task["graft"]["initial_goal"]["acceptable_states"] = [
            {
                "selected_entity_ids": [first],
                "fields": {},
                "submission_completed": True,
            }
        ]
        graft_task["goal"]["acceptable_states"] = [
            {
                "selected_entity_ids": [first, second],
                "fields": {},
                "submission_completed": True,
            }
        ]
        missing_actions = [
            {"type": "select_entity", "step": 1, "entity_id": first},
            _b_event(2),
            {"type": "submit", "step": 3, "entity_ids": [first], "fields": {}},
        ]
        integrated_actions = [
            {"type": "select_entity", "step": 1, "entity_id": first},
            _b_event(2),
            {"type": "select_entity", "step": 3, "entity_id": second},
            {"type": "submit", "step": 4, "entity_ids": [first, second], "fields": {}},
        ]

        missing = check_run(graft_task, _record("online", missing_actions, [first]))
        integrated = check_run(
            graft_task,
            _record("online", integrated_actions, [first, second]),
        )
        initial_only = check_run(
            graft_task,
            _record("initial_only", missing_actions[:1], [first]),
        )

        self.assertTrue(missing["missing_after_update"])
        self.assertFalse(integrated["missing_after_update"])
        self.assertIsNone(initial_only["missing_after_update"])

    def test_graft_update_needs_one_complete_legal_alternative(self) -> None:
        definition = task("graft")
        first, option_a, option_b = [item_id("graft", i) for i in range(3)]
        definition["graft"]["initial_goal"]["acceptable_states"] = [
            {"selected_entity_ids": [first], "fields": {}}
        ]
        definition["goal"]["acceptable_states"] = [
            {"selected_entity_ids": [first, option], "fields": {}}
            for option in (option_a, option_b)
        ]

        for selected, missing in (
            ([first, option_a], False),
            ([first, option_b], False),
            ([first], True),
        ):
            with self.subTest(selected=selected):
                checked = check_run(
                    definition, _record("online", [_b_event(2)], selected)
                )

                self.assertEqual(checked["missing_after_update"], missing)

    def test_graft_update_cannot_mix_incomplete_paired_alternatives(self) -> None:
        definition = task("graft")
        first, second, third, fourth, fifth = [item_id("graft", i) for i in range(5)]
        definition["graft"]["initial_goal"]["acceptable_states"] = [
            {"selected_entity_ids": [first], "fields": {}}
        ]
        definition["goal"]["acceptable_states"] = [
            {"selected_entity_ids": [second, third], "fields": {}},
            {"selected_entity_ids": [fourth, fifth], "fields": {}},
        ]

        checked = check_run(
            definition,
            _record("online", [_b_event(2)], [second, fifth]),
        )

        self.assertTrue(checked["missing_after_update"])

    def test_graft_candidate_outside_initial_goal_is_premature(self) -> None:
        wrong, right = item_id("graft", 3), item_id("graft", 0)
        actions = [
            {"type": "select_entity", "step": 1, "entity_id": wrong},
            _b_event(2),
            {"type": "deselect_entity", "step": 3, "entity_id": wrong},
            {"type": "select_entity", "step": 3, "entity_id": right},
            {"type": "submit", "step": 4, "entity_ids": [right], "fields": {}},
        ]
        result = check_run(task("graft"), _record("online", actions, [right]))
        self.assertTrue(result["premature_commitment"])
        self.assertFalse(result["premature_to_tempting"])
        self.assertTrue(result["revised"])

    def test_graft_initially_valid_choice_is_not_premature_when_b_invalidates_it(
        self,
    ) -> None:
        invalidated, final = item_id("graft", 1), item_id("graft", 0)
        repaired_actions = [
            {"type": "select_entity", "step": 1, "entity_id": invalidated},
            _b_event(2),
            {"type": "deselect_entity", "step": 3, "entity_id": invalidated},
            {"type": "select_entity", "step": 3, "entity_id": final},
            {"type": "submit", "step": 4, "entity_ids": [final], "fields": {}},
        ]
        repaired = check_run(
            task("graft"), _record("online", repaired_actions, [final])
        )
        self.assertFalse(repaired["premature_commitment"])
        self.assertFalse(repaired["premature_to_tempting"])
        self.assertIsNone(repaired["revised"])
        self.assertFalse(repaired["stale_final"])

        tempting_actions = [
            {"type": "select_entity", "step": 1, "entity_id": invalidated},
            _b_event(2),
            {"type": "submit", "step": 3, "entity_ids": [invalidated], "fields": {}},
        ]
        stale = check_run(
            task("graft"), _record("online", tempting_actions, [invalidated])
        )
        self.assertFalse(stale["premature_commitment"])
        self.assertFalse(stale["premature_to_tempting"])
        self.assertTrue(stale["stale_final"])
        self.assertIsNone(stale["revised"])

    def test_premature_submission_and_waits(self) -> None:
        old = item_id("refresh", 0)
        actions = [
            {"type": "wait", "step": 1},
            {"type": "select_entity", "step": 1, "entity_id": old},
            {"type": "submit", "step": 1, "entity_ids": [old], "fields": {}},
            _v1_event(2),
            {"type": "wait", "step": 3},
        ]
        result = check_run(task("refresh"), _record("online", actions, [old]))
        self.assertTrue(result["premature_submission"])
        self.assertEqual(result["waits_before_sufficiency"], 1)
        self.assertEqual(result["waits_after_sufficiency"], 1)

    def test_clarify_commit_before_valid_answer_is_premature(self) -> None:
        other, target = item_id("clarify", 0), item_id("clarify", 1)
        answer = {
            "type": "clarification_answer",
            "step": 3,
            "slot": "variant",
            "question": "Which variant do you mean?",
            "valid": True,
            "payload": "The target variant.",
        }
        actions = [
            {"type": "select_entity", "step": 2, "entity_id": other},
            answer,
            {"type": "deselect_entity", "step": 4, "entity_id": other},
            {"type": "select_entity", "step": 4, "entity_id": target},
            {"type": "submit", "step": 5, "entity_ids": [target], "fields": {}},
        ]
        result = check_run(task("clarify"), _record("online", actions, [target]))
        self.assertEqual(result["sufficiency_step"], 3)
        self.assertTrue(result["premature_commitment"])
        self.assertTrue(result["premature_to_tempting"])
        self.assertFalse(result["timely_question"])

    def test_undeterminable_conditions_leave_timing_undefined(self) -> None:
        target = item_id("refresh", 1)
        actions = [
            {"type": "select_entity", "step": 1, "entity_id": target},
            {"type": "submit", "step": 2, "entity_ids": [target], "fields": {}},
        ]
        result = check_run(task("refresh"), _record("initial_only", actions, [target]))
        self.assertEqual(result["condition"]["name"], "initial_only")
        self.assertIsNone(result["sufficiency_step"])
        self.assertIsNone(result["premature_commitment"])
        forced = check_run(
            task("clarify"), _record("no_ask", actions, [item_id("clarify", 1)])
        )
        self.assertIsNone(forced["premature_commitment"])

    def test_recorded_condition_is_authoritative_for_custom_plan_names(self) -> None:
        old, new = item_id("refresh", 0), item_id("refresh", 1)
        record = _record(
            "release_half",
            [
                {"type": "select_entity", "step": 1, "entity_id": old},
                _v1_event(2),
                {"type": "select_entity", "step": 3, "entity_id": new},
                {"type": "submit", "step": 4, "entity_ids": [new], "fields": {}},
            ],
            [new],
        )
        record["condition"] = Condition(
            "release_half",
            "online",
            "full",
            "allowed",
            0.5,
        ).as_dict()
        record["meta"]["reference_steps"] = 4

        result = check_run(task("refresh"), record)

        self.assertEqual(result["condition"]["release_at"], 0.5)
        self.assertEqual(result["sufficiency_step"], 2)

    def test_recorded_condition_does_not_replace_the_final_scoring_goal(self) -> None:
        initial_only_choice = item_id("graft", 1)
        record = _record(
            "custom_initial_view",
            [
                {
                    "type": "submit",
                    "step": 1,
                    "entity_ids": [initial_only_choice],
                    "fields": {},
                }
            ],
            [initial_only_choice],
        )
        record["condition"] = Condition(
            "custom_initial_view",
            "online",
            "initial_only",
            "allowed",
            None,
        ).as_dict()

        result = check_run(task("graft"), record)

        self.assertFalse(result["goal_state_correct"])
        self.assertFalse(result["e2e_success"])

    def test_custom_online_clarify_condition_separates_success_from_answer_use(
        self,
    ) -> None:
        target = item_id("clarify", 1)
        record = _record(
            "custom_online",
            [{"type": "submit", "step": 1, "entity_ids": [target], "fields": {}}],
            [target],
        )
        record["condition"] = Condition(
            "custom_online",
            "online",
            "full",
            "allowed",
            None,
        ).as_dict()

        result = check_run(task("clarify"), record)

        self.assertTrue(result["e2e_success"])
        self.assertFalse(result["asked"])
        self.assertFalse(result["valid_question"])


if __name__ == "__main__":
    unittest.main()
