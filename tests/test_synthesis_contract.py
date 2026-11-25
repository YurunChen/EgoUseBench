from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from egocua.construction import (
    ConstructionError,
    compile_task_from_site,
    opaque_item_id,
)
from tests.construction_fixtures import screening_record, source_identity
from tools.audit.report_necessity import build_necessity_report


class DecisionPointContractTest(unittest.TestCase):
    def _compile(self, record, site, root):
        return compile_task_from_site(
            record,
            source_identity=source_identity(record),
            site_data=site,
            root=root,
        )

    def test_a_record_without_a_curve_is_refused(self) -> None:
        """The curve is the claim. Nothing else can stand in for it."""

        with TemporaryDirectory() as directory:
            record, site = screening_record(Path(directory), "graft")
            del record["construct"]["decision"]
            with self.assertRaisesRegex(ConstructionError, "no construct.decision"):
                self._compile(record, site, Path(directory))

    def test_declaring_what_the_curve_derives_is_refused(self) -> None:
        for field, value in (
            ("release_anchor_seconds", 42.0),
            ("tempting_role", "distractor_0"),
            ("deixis_level", 2),
            ("neighbor_role_ids", ["distractor_0"]),
            ("integration_type", "C"),
        ):
            with self.subTest(field=field), TemporaryDirectory() as directory:
                record, site = screening_record(Path(directory), "graft")
                record["construct"][field] = value
                with self.assertRaisesRegex(ConstructionError, "remove them"):
                    self._compile(record, site, Path(directory))

    def test_transition_and_tempting_are_compiled_from_the_source(self) -> None:
        with TemporaryDirectory() as directory:
            record, site = screening_record(Path(directory), "graft")
            task = self._compile(record, site, Path(directory))
            self.assertEqual(task["graft"]["transition"], "augment")
            self.assertEqual(
                task["truth"]["tempting_referent_ids"],
                [opaque_item_id(record["task_id"], "distractor_0")],
            )

    def test_a_refresh_settled_only_by_words_is_refused(self) -> None:
        """Its later packet is then not doing the work the track claims."""

        with TemporaryDirectory() as directory:
            record, site = screening_record(Path(directory), "refresh")
            observed = record["construct"]["decision"]["observed"]
            settled = observed[-1]
            observed[-1] = {
                "frame": settled["frame"],
                "live": sorted(record["construct"]["decision"]["candidates"]),
            }
            observed.append(
                {
                    "frame": "utterance",
                    "live": [settled["live"][0]],
                    "narrowed_by": "user_utterance",
                    "why": "the user says which one",
                }
            )
            with self.assertRaisesRegex(ConstructionError, "not doing the work"):
                self._compile(record, site, Path(directory))

    def test_a_clarify_the_frames_settle_is_refused(self) -> None:
        """If the frames answer it, there is nothing left to ask."""

        with TemporaryDirectory() as directory:
            record, site = screening_record(Path(directory), "clarify")
            observed = record["construct"]["decision"]["observed"]
            del observed[-1]
            observed[-1] = {
                "frame": observed[-1]["frame"],
                "live": ["blue_product"],
                "narrowed_by": "scene_appearance",
                "why": "the blue one is in shot alone",
            }
            with self.assertRaisesRegex(ConstructionError, "nothing is left to ask"):
                self._compile(record, site, Path(directory))

    def test_a_curve_settling_where_the_goal_does_not_accept_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            record, site = screening_record(Path(directory), "refresh")
            observed = record["construct"]["decision"]["observed"]
            observed[-1]["live"] = ["first_product"]
            with self.assertRaisesRegex(ConstructionError, "the goal does not accept"):
                self._compile(record, site, Path(directory))

    def test_a_candidate_the_site_lacks_is_refused(self) -> None:
        with TemporaryDirectory() as directory:
            record, site = screening_record(Path(directory), "refresh")
            site["entities"] = [
                entity
                for entity in site["entities"]
                if entity["entity_id"]
                != opaque_item_id(record["task_id"], "first_product")
            ]
            with self.assertRaises(ConstructionError):
                self._compile(record, site, Path(directory))

    def test_rates_are_grouped_by_task_and_condition(self) -> None:
        def row(task_id, condition, success, tempting):
            return {
                "task_id": task_id,
                "condition": {"name": condition},
                "e2e_success": success,
                "final_in_tempting": tempting,
            }

        report = build_necessity_report(
            [
                {
                    "runs": [
                        row("t1", "text_only", True, None),
                        row("t1", "text_only", False, None),
                        row("t1", "initial_only", False, True),
                        row("t1", "initial_only", False, True),
                        row("t1", "initial_only", True, False),
                        row("t2", "online", None, None),
                    ]
                }
            ]
        )
        t1 = report["tasks"]["t1"]
        self.assertEqual(
            t1["text_only"],
            {
                "runs": 2,
                "e2e_success_rate": 0.5,
                "tempting_pick_rate": None,
            },
        )
        self.assertEqual(t1["initial_only"]["tempting_pick_rate"], 2 / 3)
        self.assertEqual(report["tasks"]["t2"]["online"]["e2e_success_rate"], None)


if __name__ == "__main__":
    unittest.main()
