"""A plan may not declare a condition that no schedule will ever run.

An unreachable condition is worse than a missing one: it reads like coverage, so
the gap only shows up when someone goes looking for runs that were never emitted.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from egocua.experiment import (
    ExperimentPlanError,
    expand_probe_runs,
    expand_runs,
    load_experiment_plan,
    scheduled_indexes,
    validate_run_records,
)
from egocua.schema import TRACK_ORDER
from tests.helpers import PLAN, plan_payload, release_ready_task


def _load(payload: dict[str, Any]) -> Any:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "plan.json"
        path.write_text(json.dumps(payload))
        return load_experiment_plan(path)


class ExperimentPlanTest(unittest.TestCase):
    def test_plan_round_trip_preserves_all_runtime_and_probe_fields(self) -> None:
        plan = replace(
            PLAN,
            composite_max_steps=30,
            frozen_asset_root="/frozen/inputs",
            site_snapshots={"egomart": "/frozen/site.json"},
            initial_site_states={"task": {"selected_entity_ids": []}},
        )
        self.assertEqual(_load(plan.as_dict()), plan)

    def test_new_runtime_fields_are_validated_at_plan_loading(self) -> None:
        for field, values in {
            "composite_max_steps": (True, 0, -1, "30", 30.5),
            "frozen_asset_root": (False, 1, "", "   ", []),
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    payload = plan_payload()
                    payload["runtime"][field] = value
                    with self.assertRaisesRegex(ExperimentPlanError, field):
                        _load(payload)

    def test_declared_condition_no_track_runs_is_rejected(self) -> None:
        payload = plan_payload()
        payload.pop("probes")
        with self.assertRaises(ExperimentPlanError) as caught:
            _load(payload)
        self.assertIn("first_frame", str(caught.exception))

    def test_condition_cannot_be_both_treatment_and_probe(self) -> None:
        payload = plan_payload()
        payload["probes"]["graft"] = [*payload["probes"]["graft"], "online"]
        with self.assertRaises(ExperimentPlanError) as caught:
            _load(payload)
        self.assertIn("treatment and a probe", str(caught.exception))

    def test_probes_are_optional(self) -> None:
        payload = plan_payload()
        payload.pop("probes")
        payload["conditions"] = [
            condition
            for condition in payload["conditions"]
            if condition["name"] not in {"first_frame", "text_only"}
        ]
        plan = _load(payload)
        self.assertEqual(plan.track_probe_conditions, {})
        self.assertEqual(plan.probe_conditions_for("graft"), ())

    def test_probe_conditions_for_rejects_an_unplanned_track(self) -> None:
        with self.assertRaises(ExperimentPlanError):
            PLAN.probe_conditions_for("no-such-track")

    def test_probe_schedule_covers_every_task_and_replicate(self) -> None:
        tasks = [release_ready_task(track) for track in TRACK_ORDER]
        probe_runs = expand_probe_runs(tasks, PLAN)
        expected = sum(
            len(PLAN.probe_conditions_for(task["track"])) * len(PLAN.replicate_ids)
            for task in tasks
        )
        self.assertEqual(len(probe_runs), expected)
        self.assertEqual(
            sorted(run["schedule_index"] for run in probe_runs), list(range(expected))
        )

    def test_probe_schedule_is_disjoint_from_the_treatment_schedule(self) -> None:
        tasks = [release_ready_task(track) for track in TRACK_ORDER]

        def keys(runs: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
            return {
                (run["task_id"], run["condition"], run["replicate_id"]) for run in runs
            }

        self.assertFalse(
            keys(expand_runs(tasks, PLAN)) & keys(expand_probe_runs(tasks, PLAN))
        )


class RunProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = replace(
            PLAN, replicate_ids=(0,), release_trigger="first_native_commitment"
        )
        self.task = release_ready_task("graft")
        schedule_index = scheduled_indexes([self.task], self.plan)[
            (self.task["id"], "online", 0)
        ]
        self.record = {
            "task_id": self.task["id"],
            "condition": self.plan.condition("online").as_dict(),
            "steps": 0,
            "meta": {
                "replicate_id": 0,
                "schedule_index": schedule_index,
                "agent": self.plan.adapter,
                "model": self.plan.model,
                "model_version": self.plan.model_version,
                "max_steps": self.plan.max_steps,
                "default_release_step": self.plan.default_release_step,
                "reference_steps": self.plan.reference_steps,
                "runtime": {"release_trigger": self.plan.release_trigger},
            },
        }

    def test_complete_registered_protocol_is_accepted(self) -> None:
        validate_run_records([self.task], [self.record], self.plan)

    def test_different_release_mechanism_is_rejected(self) -> None:
        self.record["meta"]["runtime"]["release_trigger"] = "timeline_step"
        with self.assertRaisesRegex(ExperimentPlanError, "wrong_release_trigger"):
            validate_run_records([self.task], [self.record], self.plan)

    def test_missing_runtime_is_rejected(self) -> None:
        del self.record["meta"]["runtime"]
        with self.assertRaisesRegex(ExperimentPlanError, "missing_runtime"):
            validate_run_records([self.task], [self.record], self.plan)

    def test_missing_release_mechanism_is_rejected(self) -> None:
        self.record["meta"]["runtime"] = {}
        with self.assertRaisesRegex(ExperimentPlanError, "missing_release_trigger"):
            validate_run_records([self.task], [self.record], self.plan)

    def test_missing_nullable_protocol_field_is_rejected(self) -> None:
        del self.record["meta"]["reference_steps"]
        with self.assertRaisesRegex(ExperimentPlanError, "missing_reference_steps"):
            validate_run_records([self.task], [self.record], self.plan)


if __name__ == "__main__":
    unittest.main()
