from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from egocua.experiment import expand_runs, validate_run_records
from egocua.environment import EnvironmentError, load_site_data
from egocua.gates import validate_release, validate_task_for_model_run
from egocua.protocol import ProtocolError, RunSession
from egocua.schema import (
    PILOT_DISTRIBUTION,
    ValidationError,
    load_tasks,
    validate_dataset,
    validate_pilot_distribution,
    validate_task,
)
from tests.helpers import (
    PLAN,
    products,
    site_data,
    item_id,
    materialize_assets,
    task,
    validation_evidence,
)


class SchemaProtocolTest(unittest.TestCase):
    def test_tracks_and_experiment_arms_are_separate(self) -> None:
        tasks = [task("graft"), task("refresh"), task("clarify")]
        site_products = [
            product
            for track in ("graft", "refresh", "clarify")
            for product in products(track)
        ]
        self.assertEqual(
            validate_dataset(tasks, site_data_by_site={
                "egomart": {
                    "site": "egomart",
                    "entities": [
                        {**product, "entity_id": product["item_id"]}
                        for product in site_products
                    ],
                }
            }),
            {"clarify": 1, "graft": 1, "refresh": 1},
        )
        self.assertTrue(all("conditions" not in current for current in tasks))
        runs = expand_runs(tasks, PLAN)
        self.assertEqual(len(runs), 27)
        self.assertEqual({run["replicate_id"] for run in runs}, {0, 1, 2})
        self.assertEqual({run["arm"] for run in runs if run["track"] == "graft"}, set(PLAN.arms_for("graft")))

    def test_dataset_rejects_reused_instruction_text(self) -> None:
        first = task("graft")
        second = task("graft")
        second["id"] = "test-graft-second"
        second["source"] = {"dataset": "fixture", "id": 2}
        second["visual_packets"][0]["frame_paths"] = [
            f"frames/graft-second-{index}.jpg" for index in range(8)
        ]

        with self.assertRaisesRegex(ValidationError, "duplicate instruction text"):
            validate_dataset([first, second], site_data_by_site={"egomart": site_data("graft")})

    def test_dataset_rejects_reused_released_frame(self) -> None:
        first = task("graft")
        second = task("graft")
        second["id"] = "test-graft-second"
        second["source"] = {"dataset": "fixture", "id": 2}
        second["instructions"] = {
            key: f"{value} for the second task"
            for key, value in second["instructions"].items()
        }
        second["visual_packets"][0]["frame_paths"] = [
            first["visual_packets"][0]["frame_paths"][0],
            *[f"frames/graft-second-{index}.jpg" for index in range(1, 8)],
        ]

        with self.assertRaisesRegex(ValidationError, "duplicate released frame"):
            validate_dataset([first, second], site_data_by_site={"egomart": site_data("graft")})

    def test_directory_is_the_physical_task_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for track in ("graft", "refresh", "clarify"):
                path = root / track / f"{track}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(task(track)))
            self.assertEqual([current["track"] for current in load_tasks(root)], ["clarify", "graft", "refresh"])
            wrong = task("graft")
            wrong["track"] = "refresh"
            (root / "graft" / "graft.json").write_text(json.dumps(wrong))
            with self.assertRaisesRegex(ValidationError, "directory class"):
                load_tasks(root)

    def test_run_record_matrix_must_be_complete_and_unique(self) -> None:
        graft = task("graft")
        schedule = expand_runs([graft], PLAN)
        records = [
            {
                "task_id": run["task_id"],
                "arm": run["arm"],
                "meta": {
                    "replicate_id": run["replicate_id"],
                    "schedule_index": run["schedule_index"],
                    "agent": PLAN.adapter,
                    "model": PLAN.model,
                    "model_version": PLAN.model_version,
                    "max_steps": PLAN.max_steps,
                },
            }
            for run in schedule
        ]
        validate_run_records([graft], records, PLAN)
        with self.assertRaises(ValueError):
            validate_run_records([graft], records[:-1], PLAN)
        with self.assertRaises(ValueError):
            validate_run_records([graft], records + [records[0]], PLAN)

    def test_invalid_clarification_fields_are_rejected(self) -> None:
        clarify = task("clarify")
        clarify["clarification"]["valid_slot"] = "color"
        with self.assertRaises(ValidationError):
            validate_task(clarify, site_data("clarify"))
        clarify = task("clarify")
        clarify["clarification"]["candidate_ids"].append(item_id("clarify", 2))
        with self.assertRaisesRegex(ValidationError, "exactly 2"):
            validate_task(clarify, site_data("clarify"))

    def test_clarification_instruction_must_not_permit_an_arbitrary_candidate(self) -> None:
        clarify = task("clarify")
        clarify["instructions"]["ambiguous"] = "Find one of the two products."
        with self.assertRaisesRegex(ValidationError, "underspecify one intended target"):
            validate_task(clarify, site_data("clarify"))

    def test_clarification_slot_is_the_subtype_source_of_truth(self) -> None:
        for slot in ("appearance", "brand", "variant"):
            with self.subTest(slot=slot):
                clarify = task("clarify")
                clarify["clarification"]["valid_slot"] = slot
                validate_task(clarify, site_data("clarify"))

        clarify = task("clarify")
        clarify["clarification"]["valid_slot"] = "sequence"
        validate_task(clarify, site_data("clarify"))

    def test_only_current_task_schema_is_accepted(self) -> None:
        graft = task("graft")
        graft.pop("schema_version")
        with self.assertRaisesRegex(ValidationError, "schema_version"):
            validate_task(graft, site_data("graft"))
        clarify = task("clarify")
        clarify["clarification"]["canonical_answer"] = "  "
        with self.assertRaises(ValidationError):
            validate_task(clarify, site_data("clarify"))

    def test_subtype_must_match_track_specific_label(self) -> None:
        for track, subtype in (("graft", "P"), ("refresh", "retain"), ("clarify", "position")):
            with self.subTest(track=track):
                current = task(track)
                current["subtype"] = subtype
                with self.assertRaises(ValidationError):
                    validate_task(current, site_data(track))

    def test_graft_has_one_initial_goal_and_one_updated_goal(self) -> None:
        unchanged = task("graft")
        unchanged["graft"]["initial_goal"] = unchanged["goal"]
        with self.assertRaisesRegex(ValidationError, "narrow alternatives or add exact fields"):
            validate_task(unchanged, site_data("graft"))

    def test_goal_requires_a_source_supported_target_granularity(self) -> None:
        graft = task("graft")
        graft["goal"].pop("target_granularity")
        with self.assertRaisesRegex(ValidationError, "target_granularity"):
            validate_task(graft, site_data("graft"))

        graft = task("graft")
        graft["graft"]["initial_goal"]["target_granularity"] = "category"
        with self.assertRaisesRegex(ValidationError, "one granularity"):
            validate_task(graft, site_data("graft"))

    def test_released_instruction_rejects_internal_frozen_wording(self) -> None:
        graft = task("graft")
        graft["instructions"]["b"] = "Use the frozen average rating."

        with self.assertRaisesRegex(ValidationError, "benchmark-internal wording"):
            validate_task(graft, site_data("graft"))

    def test_graft_relations_are_derived_from_initial_and_updated_goals(self) -> None:
        graft = task("graft")
        graft["graft"]["target_a_id"] = item_id("graft", 0)
        with self.assertRaisesRegex(ValidationError, "derived from initial_goal and goal"):
            validate_task(graft, site_data("graft"))

        graft = task("graft")
        graft["graft"]["integration_type"] = "P"
        graft["graft"]["initial_goal"] = {
            "target_granularity": "variant",
            "acceptable_states": [{
                "selected_entity_ids": [item_id("graft", 0)],
                "fields": {},
                "submission_completed": True,
            }],
        }
        graft["goal"] = {
            "target_granularity": "variant",
            "acceptable_states": [{
                "selected_entity_ids": [item_id("graft", 1)],
                "fields": {},
                "submission_completed": True,
            }],
        }
        with self.assertRaisesRegex(ValidationError, "retain an initial entity and add an entity"):
            validate_task(graft, site_data("graft"))

    def test_removed_sources_of_truth_are_rejected(self) -> None:
        for field, value in (
            ("conditions", ["a_only"]),
            ("expectations", {"a_only": {"acceptable_sets": [[item_id("graft", 0)]]}}),
            ("catalog", products("graft")),
            ("environment", {"site": "egomart"}),
            ("review", {}),
        ):
            with self.subTest(field=field):
                current = task("graft")
                current[field] = value
                with self.assertRaisesRegex(ValidationError, "removed fields"):
                    validate_task(current, site_data("graft"))

    def test_visual_packet_contract(self) -> None:
        graft = task("graft")
        graft["visual_packets"].append({
            "id": "VB",
            "frame_paths": [f"frames/graft-b-{index}.jpg" for index in range(4)],
        })
        with self.assertRaisesRegex(ValidationError, "visual packet ids"):
            validate_task(graft, site_data("graft"))
        refresh = task("refresh")
        refresh["visual_packets"][0]["target_state"] = "A"
        with self.assertRaisesRegex(ValidationError, "target_state"):
            validate_task(refresh, site_data("refresh"))

    def test_non_pilot_distribution_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            validate_pilot_distribution([task("graft"), task("refresh"), task("clarify")])

    def test_current_pilot_distribution_is_accepted(self) -> None:
        self.assertEqual(
            PILOT_DISTRIBUTION,
            {
                "graft": {"C": 55, "P": 9, "R": 10},
                "refresh": {"retain": 7, "replace": 10},
                "clarify": {"variant": 6, "sequence": 4, "position": 4},
            },
        )
        tasks = []
        for track, subtypes in PILOT_DISTRIBUTION.items():
            for subtype, count in subtypes.items():
                for _ in range(count):
                    current = task(track)
                    if track == "graft":
                        current["graft"]["integration_type"] = subtype
                        if subtype in {"P", "R"}:
                            current["graft"]["initial_goal"]["acceptable_states"] = [
                                current["graft"]["initial_goal"]["acceptable_states"][0]
                            ]
                            current["goal"]["acceptable_states"] = [{
                                "selected_entity_ids": [
                                    item_id("graft", 0), item_id("graft", 1)
                                ],
                                "fields": {},
                                "submission_completed": True,
                            }]
                    elif track == "refresh":
                        current["refresh"]["policy"] = subtype
                        if subtype == "retain":
                            current["goal"]["acceptable_states"][0][
                                "selected_entity_ids"
                            ] = [current["refresh"]["initial_entity_id"]]
                    else:
                        current["clarification"]["valid_slot"] = (
                            subtype if subtype != "variant" else "variant"
                        )
                    tasks.append(current)
        validate_pilot_distribution(tasks)

    def test_release_gate_uses_external_evidence(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            evidence = {
                graft["id"]: validation_evidence(
                    "graft", task_definition=graft, root=root
                )
            }
            self.assertEqual(
                validate_release([graft], evidence, PLAN, root=root),
                {"clarify": 0, "graft": 1, "refresh": 0},
            )
            incomplete = validation_evidence(
                "graft", task_definition=graft, root=root
            )
            incomplete["human"]["arm_runs"]["bundled"].pop()
            with self.assertRaisesRegex(ValidationError, "two independent human runs"):
                validate_release([graft], {graft["id"]: incomplete}, PLAN, root=root)

    def test_release_products_require_independent_hashed_image_provenance(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            store = root / "sites/egomart/products.json"
            payload = json.loads(store.read_text())
            payload["products"][0].pop("image_provenance")
            store.write_text(json.dumps(payload))
            with self.assertRaisesRegex(EnvironmentError, "image provenance missing"):
                load_site_data(root, "egomart", require_assets=True)

    def test_fixed_site_rejects_multiple_offers_using_one_listing_image(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            store = root / "sites/egomart/products.json"
            payload = json.loads(store.read_text())
            payload["products"][1]["image_path"] = payload["products"][0]["image_path"]
            payload["products"][1]["image_provenance"]["sha256"] = payload["products"][0]["image_provenance"]["sha256"]
            store.write_text(json.dumps(payload))
            with self.assertRaisesRegex(EnvironmentError, "duplicate product image"):
                load_site_data(root, "egomart", require_assets=True)

    def test_release_rejects_a_changed_source_dossier(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            evidence = validation_evidence(
                "graft", task_definition=graft, root=root
            )
            dossier = root / graft["source"]["dossier"]
            dossier.write_text(dossier.read_text() + "\n")
            with self.assertRaisesRegex(ValidationError, "source dossier SHA-256 mismatch"):
                validate_release([graft], {graft["id"]: evidence}, PLAN, root=root)

    def test_release_rejects_a_changed_released_frame(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            evidence = validation_evidence(
                "graft", task_definition=graft, root=root
            )
            frame = root / graft["visual_packets"][0]["frame_paths"][0]
            frame.write_bytes(b"changed frame")
            with self.assertRaisesRegex(ValidationError, "released frame SHA-256 mismatch"):
                validate_release([graft], {graft["id"]: evidence}, PLAN, root=root)

    def test_release_rejoins_source_and_frame_timestamps(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            evidence = validation_evidence(
                "graft", task_definition=graft, root=root
            )
            provenance_path = root / graft["source"]["frame_provenance"]
            provenance = json.loads(provenance_path.read_text())
            provenance["frames"][0]["timestamp_seconds"] += 0.5
            provenance_path.write_text(json.dumps(provenance))
            graft["source"]["frame_provenance_sha256"] = hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(ValidationError, "provenance timestamp mismatch"):
                validate_release([graft], {graft["id"]: evidence}, PLAN, root=root)

    def test_release_replays_the_referenced_run_artifact(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            artifact = root / "artifacts/runs/fixture/graft/a_only/0/run.json"
            record = json.loads(artifact.read_text())
            record["meta"]["model"] = "different-model"
            artifact.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValidationError, "identity or hashes"):
                validate_release(
                    [graft],
                    {
                        graft["id"]: validation_evidence(
                            "graft", task_definition=graft, root=root
                        )
                    },
                    PLAN,
                    root=root,
                )

    def test_evaluator_evidence_must_reference_real_replay_artifacts(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            evidence = validation_evidence(
                "graft", task_definition=graft, root=root
            )
            evidence["evaluator"]["replay_cases"]["extra_item_rejected"] = True
            with self.assertRaisesRegex(ValidationError, "needs an arm and artifact"):
                validate_release([graft], {graft["id"]: evidence}, PLAN, root=root)

    def test_model_preflight_requires_external_human_verified_evidence(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            evidence = validation_evidence(
                "graft", status="human_verified", task_definition=graft, root=root
            )
            validate_task_for_model_run(graft, evidence, PLAN, root=root)
            evidence["construction"]["reviews"] = []
            with self.assertRaisesRegex(ValidationError, "does not match current reviews"):
                validate_task_for_model_run(graft, evidence, PLAN, root=root)

    def test_human_gate_replays_artifacts_and_allows_underinformed_controls(self) -> None:
        refresh = task("refresh")
        items = products("refresh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(refresh, items, root)
            evidence = validation_evidence(
                "refresh", status="human_verified", task_definition=refresh, root=root
            )
            validate_task_for_model_run(refresh, evidence, PLAN, root=root)
            artifact = root / "artifacts/runs/human/refresh/stream/human-1/run.json"
            record = json.loads(artifact.read_text())
            record["final_state"]["selected_entity_ids"] = [
                refresh["refresh"]["initial_entity_id"]
            ]
            artifact.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValidationError, "information-sufficient human run must succeed"):
                validate_task_for_model_run(refresh, evidence, PLAN, root=root)

    def test_visual_necessity_and_counterfactual_checks_are_enforced(self) -> None:
        clarify = task("clarify")
        items = products("clarify")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(clarify, items, root)
            evidence = validation_evidence(
                "clarify", status="human_verified", task_definition=clarify, root=root
            )
            target = clarify["goal"]["acceptable_states"][0]["selected_entity_ids"]
            for review in evidence["human"]["visual_necessity"]["reviews"]:
                review["text_only_choice_ids"] = target
            with self.assertRaisesRegex(ValidationError, "text-only evidence stably identifies"):
                validate_task_for_model_run(clarify, evidence, PLAN, root=root)
            evidence = validation_evidence(
                "clarify", status="human_verified", task_definition=clarify, root=root
            )
            evidence["human"]["counterfactual_checks"][0]["unique_with_answer"] = False
            with self.assertRaisesRegex(ValidationError, "counterfactual check failed"):
                validate_task_for_model_run(clarify, evidence, PLAN, root=root)

    def test_annotation_placeholders_fail_before_human_review(self) -> None:
        graft = task("graft")
        graft["instructions"]["b"] = "[ANNOTATE] Add the reviewed requirement"
        validate_task(graft, site_data("graft"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "annotation placeholders"):
                materialize_assets(graft, products("graft"), root)

    def test_refresh_policy_must_match_the_canonical_goal(self) -> None:
        refresh = task("refresh")
        refresh["refresh"]["policy"] = "retain"
        with self.assertRaisesRegex(ValidationError, "policy disagrees"):
            validate_task(refresh, site_data("refresh"))

    def test_graft_injects_on_the_task_timeline_once(self) -> None:
        session = RunSession(task("graft"), "injected")
        self.assertEqual(session.advance(1), [])
        event = session.advance(2)[0]
        self.assertEqual(event["type"], "instruction_injection")
        self.assertEqual(event["trigger_reason"], "task_timeline")
        self.assertEqual(session.advance(5), [])

    def test_graft_arms_share_the_exact_same_visual_observation(self) -> None:
        graft = task("graft")
        payloads = {
            arm: RunSession(graft, arm).initial_payload()
            for arm in PLAN.arms_for("graft")
        }
        self.assertEqual(payloads["a_only"]["observations"], payloads["bundled"]["observations"])
        self.assertEqual(payloads["bundled"]["observations"], payloads["injected"]["observations"])

    def test_refresh_releases_on_the_task_timeline(self) -> None:
        session = RunSession(task("refresh"), "stream")
        self.assertEqual(session.advance(1), [])
        event = session.advance(2)[0]
        self.assertEqual(event["type"], "visual_context_update")
        self.assertEqual(event["trigger_reason"], "task_timeline")

    def test_released_observation_is_allowlisted(self) -> None:
        refresh = task("refresh")
        refresh["visual_packets"][0]["target_state"] = "hidden-marker"
        payload = RunSession(refresh, "stream").initial_payload()
        self.assertEqual(set(payload), {"instruction", "observations", "clarification_available"})
        self.assertEqual(set(payload["observations"][0]), {"frame_paths"})
        self.assertNotIn("hidden-marker", repr(payload))

    def test_clarification_is_deterministic_and_limited(self) -> None:
        clarify = task("clarify")
        session = RunSession(clarify, "interactive")
        invalid = session.ask_user("variant", "What is the weather?", 2)
        self.assertFalse(invalid["valid"])
        self.assertNotEqual(invalid["payload"], clarify["clarification"]["canonical_answer"])
        self.assertTrue(session.ask_user("variant", "Which variant?", 3)["limit_reached"])

        valid = RunSession(clarify, "interactive").ask_user(
            "variant", "Which variant do you mean?", 2
        )
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["payload"], clarify["clarification"]["canonical_answer"])
        with self.assertRaises(ProtocolError):
            RunSession(clarify, "forced_execute").ask_user("variant", "Which variant?", 1)

    def test_natural_generic_disambiguation_uses_the_structured_slot(self) -> None:
        clarify = task("clarify")
        event = RunSession(clarify, "interactive").ask_user(
            "variant", "Which one do you mean?", 1
        )

        self.assertTrue(event["valid"])
        self.assertEqual(event["payload"], clarify["clarification"]["canonical_answer"])

    def test_natural_candidate_names_do_not_need_abstract_slot_words(self) -> None:
        clarify = task("clarify")
        event = RunSession(clarify, "interactive").ask_user(
            "variant", "Do you mean the rabbit puppet or the pig puppet?", 1
        )

        self.assertTrue(event["valid"])

    def test_question_with_another_explicit_dimension_is_rejected(self) -> None:
        clarify = task("clarify")
        event = RunSession(clarify, "interactive").ask_user(
            "variant", "Do you mean the one on the left or the right?", 1
        )

        self.assertFalse(event["valid"])
        self.assertNotIn(clarify["clarification"]["canonical_answer"], event["payload"])

    def test_question_validation_supports_natural_chinese_slot_questions(self) -> None:
        clarify = task("clarify")
        event = RunSession(clarify, "interactive").ask_user(
            "variant", "你指的是哪个版本？", 1
        )

        self.assertTrue(event["valid"])


if __name__ == "__main__":
    unittest.main()
