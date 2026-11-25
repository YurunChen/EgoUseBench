from __future__ import annotations

import unittest
from pathlib import Path

from egocua.experiment import load_experiment_plan
from egocua.timeline import STANDARD_CONDITIONS

ROOT = Path(__file__).resolve().parents[1]


class StandardConditionOwnershipTest(unittest.TestCase):
    def test_pilot_plan_does_not_redefine_a_standard_condition(self) -> None:
        plan = load_experiment_plan(ROOT / "configs" / "pilot.json")
        for name, condition in plan.conditions.items():
            standard = STANDARD_CONDITIONS.get(name)
            if standard is not None:
                self.assertEqual(
                    condition,
                    standard,
                    f"plan condition {name!r} disagrees with STANDARD_CONDITIONS",
                )

    def test_every_pilot_condition_is_reachable_from_some_track(self) -> None:
        """Defining a condition is not enough — a schedule has to run it.

        text_only and first_frame were declared for months while no track listed
        them, so `egocua expand` never emitted a single probe run.
        """

        plan = load_experiment_plan(ROOT / "configs" / "pilot.json")
        reachable = {
            name
            for track in plan.track_conditions
            for name in plan.enabled_conditions_for(track)
        }
        self.assertEqual(set(plan.conditions), reachable)

    def test_necessity_probes_are_scheduled_but_not_compared(self) -> None:
        plan = load_experiment_plan(ROOT / "configs" / "pilot.json")
        for track in plan.track_conditions:
            probes = plan.probe_conditions_for(track)
            self.assertIn("text_only", probes)
            self.assertIn("first_frame", probes)
            self.assertFalse(set(probes) & set(plan.conditions_for(track)))


if __name__ == "__main__":
    unittest.main()
