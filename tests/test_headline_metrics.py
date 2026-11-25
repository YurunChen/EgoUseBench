from __future__ import annotations

import unittest
from typing import Any

import pytest

from egocua.checker import check_run
from egocua.metrics import HEADLINE_LEGEND, _post_release_sr, aggregate
from egocua.timeline import Timeline
from tests.conditions import CONDITIONS
from tests.helpers import complete_run_record, item_id, task


def _record(
    condition_name: str,
    actions: list[dict[str, Any]],
    submitted: list[str],
) -> dict[str, Any]:
    return complete_run_record(
        {
            "condition": CONDITIONS[condition_name].as_dict(),
            "run_status": {"valid": True, "reason": None},
            "steps": max((int(a.get("step", 0)) for a in actions), default=0),
            "actions": actions,
            "final_state": {
                "selected_entity_ids": submitted,
                "fields": {},
                "submission_completed": True,
            },
            "live_state": {
                "selected_entity_ids": submitted,
                "fields": {},
            },
            "meta": {
                "replicate_id": 0,
                "default_release_step": 2,
                "reference_steps": None,
            },
        }
    )


class HeadlineTest(unittest.TestCase):
    def test_question_compliance_does_not_change_paired_success_effect(self) -> None:
        rows = [
            {
                "task_id": "same-target",
                "track": "clarify",
                "subtype": "resolve",
                "condition": CONDITIONS[name].as_dict(),
                "run_valid": True,
                "e2e_success": True,
                "asked": False,
                "valid_question": False,
                "timely_question": False,
                "policy_pass": False,
            }
            for name in ("online", "no_ask")
        ]
        report = aggregate(rows)
        self.assertEqual(report["common"]["e2e_success"], 1.0)
        self.assertEqual(
            report["paired_effects"]["clarify"]["resolve"]["allowed_minus_denied"],
            0.0,
        )
        self.assertEqual(report["clarification_policy"]["ask_recall"], 0.0)

    def test_visual_effects_compare_same_task_final_success(self) -> None:
        rows = [
            {
                "task_id": "same-target",
                "track": "graft",
                "subtype": "refine",
                "condition": CONDITIONS[name].as_dict(),
                "run_valid": True,
                "e2e_success": success,
            }
            for name, success in (
                ("online", True),
                ("first_frame", True),
                ("text_only", False),
            )
        ]
        contrasts = aggregate(rows)["paired_effects"]["graft"]["refine"]
        self.assertEqual(contrasts["full_minus_first_frame"], 0.0)
        self.assertEqual(contrasts["full_minus_text_only"], 1.0)

    def test_recovery_rate_excludes_unknown_live_state(self) -> None:
        unknown = {
            "task_id": "unknown",
            "release_delivered": True,
            "post_release_latency": 1,
            "goal_state_correct": None,
        }
        success = {**unknown, "task_id": "success", "goal_state_correct": True}
        failure = {**unknown, "task_id": "failure", "goal_state_correct": False}
        self.assertIsNone(_post_release_sr([unknown], 3))
        self.assertEqual(_post_release_sr([unknown, success, failure], 3), 0.5)

    def test_headline_has_one_row_per_track_condition_and_legend_covers_columns(
        self,
    ) -> None:
        old, new = item_id("refresh", 0), item_id("refresh", 1)
        release = Timeline.compile(
            task("refresh"),
            CONDITIONS["online"],
            default_release_step=2,
            reference_steps=None,
        ).releases[0]
        stream = check_run(
            task("refresh"),
            _record(
                "online",
                [
                    {"type": "search", "step": 1, "query": "item"},
                    release.as_event(),
                    {"type": "select_entity", "step": 3, "entity_id": new},
                    {"type": "submit", "step": 4, "entity_ids": [new], "fields": {}},
                ],
                [new],
            ),
        )
        stale = check_run(
            task("refresh"),
            _record(
                "initial_only",
                [
                    {"type": "select_entity", "step": 1, "entity_id": old},
                    {"type": "submit", "step": 2, "entity_ids": [old], "fields": {}},
                ],
                [old],
            ),
        )
        report = aggregate([stream, stale])
        rows = {(row["track"], row["condition"]): row for row in report["headline"]}
        self.assertEqual(
            set(rows), {("refresh", "online"), ("refresh", "initial_only")}
        )
        online = rows[("refresh", "online")]
        self.assertEqual(online["success"], 1.0)
        self.assertEqual(online["stale"], 0.0)
        self.assertEqual(report["common"]["timing"]["post_release_sr_at_3"], 1.0)
        self.assertEqual(report["common"]["steps_on_success"], 4.0)
        metric_columns = set(online) - {"track", "condition", "runs"}
        self.assertEqual(metric_columns, set(HEADLINE_LEGEND))


@pytest.mark.parametrize(
    "changed_factor", [{"wearer_frames_per_image": 4}, {"wearer_jpeg_quality": 70}]
)
def test_standard_effect_excludes_other_image_conditions(changed_factor: dict) -> None:
    rows = [
        {
            "task_id": "same-task",
            "track": "graft",
            "subtype": "replace",
            "run_valid": True,
            "e2e_success": name == "upfront",
            "condition": CONDITIONS[name].as_dict(),
        }
        for name in ("online", "upfront")
    ]
    rows[1]["condition"].update(name="custom-upfront", **changed_factor)

    assert aggregate(rows)["paired_effects"] == {}

    rows[1]["condition"] = {**CONDITIONS["upfront"].as_dict(), "name": "alias"}
    assert aggregate(rows)["paired_effects"]["graft"]["replace"] == {
        "upfront_minus_online": 1.0
    }


def test_missing_run_validity_is_not_counted_as_valid_success() -> None:
    row = {
        "task_id": "unchecked",
        "track": "graft",
        "subtype": "replace",
        "condition": CONDITIONS["online"].as_dict(),
        "e2e_success": True,
    }

    report = aggregate([row])

    assert report["common"]["valid_run_rate"] == 0.0
    assert report["common"]["e2e_success"] is None
    assert report["headline"] == []


@pytest.mark.parametrize("upfront_quality,paired", [(85, True), (95, False)])
def test_paired_effect_matches_actual_image_configuration(
    upfront_quality: int, paired: bool
) -> None:
    rows = [
        {
            "task_id": "same-task",
            "track": "clarify",
            "subtype": "resolve",
            "run_valid": True,
            "e2e_success": name == "upfront",
            "asked": name == "online",
            "condition": {
                **CONDITIONS[name].as_dict(),
                "wearer_frames_per_image": 4,
                "wearer_jpeg_quality": quality,
            },
        }
        for name, quality in (("online", 85), ("upfront", upfront_quality))
    ]

    report = aggregate(rows)

    assert report["clarification_policy"]["ask_recall"] == 1.0
    online = next(row for row in report["headline"] if row["condition"] == "online")
    assert online["success"] == 0.0
    assert report["clarification_policy"]["ask_f1"] == 1.0
    assert report["paired_effects"] == (
        {"clarify": {"resolve": {"upfront_minus_online": 1.0}}} if paired else {}
    )


if __name__ == "__main__":
    unittest.main()
