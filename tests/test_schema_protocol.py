from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import pytest

from egocua.experiment import expand_runs, validate_run_records
from egocua.environment import EnvironmentError, load_site_data
from egocua.gates import validate_task_for_human_run, validate_task_for_model_run
from egocua.lineage import LineageError, validate_source_lineage
from egocua.protocol import ProtocolError, RunSession
from egocua.timeline import Condition
from tests.conditions import CONDITIONS
from egocua.schema import (
    ValidationError,
    load_tasks,
    semantic_visual_representative_ids,
    validate_dataset,
    validate_task,
)
from tests.helpers import (
    PLAN,
    products,
    release_ready_task,
    site_data,
    item_id,
    materialize_assets,
    task,
)


def _run_session(task_definition: dict, condition: Condition) -> RunSession:
    return RunSession(
        task_definition,
        condition,
        event_step=PLAN.default_release_step,
        reference_steps=PLAN.reference_steps,
    )


@pytest.mark.parametrize(
    "failure", ["empty", "outside_scope", "policy", "representative"]
)
def test_refresh_initial_truth_validates_complete_exposed_alternatives(
    failure: str,
) -> None:
    definition = task("refresh")
    initial = definition["truth"]["initial_acceptable_states"]
    if failure == "empty":
        definition["truth"]["initial_acceptable_states"] = []
    elif failure == "outside_scope":
        initial.append({"selected_entity_ids": [item_id("refresh", 2)], "fields": {}})
        definition["truth"]["site_scope_entity_ids"] = [
            item_id("refresh", 0),
            item_id("refresh", 1),
        ]
    elif failure == "policy":
        definition["refresh"]["policy"] = "retain"
    else:
        initial[0]["selected_entity_ids"] = [item_id("refresh", 2)]
    with pytest.raises(ValidationError):
        validate_task(definition, site_data("refresh"))


def test_historical_refresh_without_initial_truth_remains_readable() -> None:
    definition = task("refresh")
    definition["truth"].pop("initial_acceptable_states")
    validate_task(definition, site_data("refresh"))


def test_refresh_replace_allows_a_listing_valid_at_both_stages() -> None:
    definition = task("refresh")
    definition["truth"]["initial_acceptable_states"].append(
        {"selected_entity_ids": [item_id("refresh", 1)], "fields": {}}
    )
    validate_task(definition, site_data("refresh"))


def test_refresh_replace_can_keep_its_initial_representative_as_a_legal_answer() -> (
    None
):
    definition = task("refresh")
    definition["goal"]["acceptable_states"].append(
        {"selected_entity_ids": [item_id("refresh", 0)], "fields": {}}
    )

    validate_task(definition, site_data("refresh"))


def test_refresh_replace_rejects_identical_initial_and_final_sets() -> None:
    definition = task("refresh")
    final = definition["goal"]["acceptable_states"][0]
    definition["truth"]["initial_acceptable_states"].append(final.copy())
    definition["goal"]["acceptable_states"].append(
        {"selected_entity_ids": [item_id("refresh", 0)], "fields": {}}
    )
    with pytest.raises(ValidationError, match="policy disagrees"):
        validate_task(definition, site_data("refresh"))


@pytest.mark.parametrize(
    "scenario",
    [
        ("drop", [[0, 1], [0, 2]], [[0]], (1, 2)),
        ("drop", [[0, 1, 2], [0, 1, 3]], [[0, 1]], (2, 3)),
        ("augment", [[0]], [[0, 1], [0, 2]], (1, 2)),
        ("drop", [[0, 2], [1, 2]], [[0], [1]], (0, 1)),
    ],
)
def test_graft_same_role_equivalents_may_share_public_images(
    tmp_path: Path,
    scenario: tuple,
) -> None:
    transition, initial, final, shared = scenario
    definition = task("graft")
    definition["graft"]["transition"] = transition
    for goal, options in (
        (definition["graft"]["initial_goal"], initial),
        (definition["goal"], final),
    ):
        goal["acceptable_states"] = [
            {
                "selected_entity_ids": [item_id("graft", index) for index in ids],
                "fields": {},
                "submission_completed": True,
            }
            for ids in options
        ]
    materialize_assets(definition, products("graft"), tmp_path)
    site = site_data("graft")
    site["entities"][shared[1]]["image_path"] = site["entities"][shared[0]][
        "image_path"
    ]

    validate_task(definition, site, root=tmp_path, require_assets=True)


@pytest.mark.parametrize(
    "scenario",
    [
        ("drop", [[0, 1], [0, 2]], [[0]], (0, 1)),
        ("refine", [[0], [1]], [[0]], (0, 1)),
        ("augment", [[0]], [[0, 1], [0, 2]], (0, 1)),
        ("augment", [[0], [2]], [[0, 1], [2, 3]], (1, 3)),
    ],
)
def test_graft_different_roles_must_keep_distinct_public_images(
    tmp_path: Path,
    scenario: tuple,
) -> None:
    transition, initial, final, shared = scenario
    definition = task("graft")
    definition["graft"]["transition"] = transition
    for goal, options in (
        (definition["graft"]["initial_goal"], initial),
        (definition["goal"], final),
    ):
        goal["acceptable_states"] = [
            {
                "selected_entity_ids": [item_id("graft", index) for index in ids],
                "fields": {},
                "submission_completed": True,
            }
            for ids in options
        ]
    materialize_assets(definition, products("graft"), tmp_path)
    site = site_data("graft")
    site["entities"][shared[1]]["image_path"] = site["entities"][shared[0]][
        "image_path"
    ]

    with pytest.raises(ValidationError, match="share the same visual asset"):
        validate_task(definition, site, root=tmp_path, require_assets=True)


def test_retail_size_variants_can_share_the_same_published_product_image(
    tmp_path: Path,
) -> None:
    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    site = site_data("graft")
    original, larger = site["entities"][:2]
    larger["image_path"] = original["image_path"]
    larger["image_provenance"]["image_url"] = original["image_provenance"]["image_url"]
    original["attributes"] = {"diameter": "20 cm"}
    larger["attributes"] = {"diameter": "26 cm"}
    validate_task(definition, site, root=tmp_path, require_assets=True)


def test_visual_equivalence_preserves_native_field_requirements(tmp_path: Path) -> None:
    definition = task("refresh")
    materialize_assets(definition, products("refresh"), tmp_path)
    definition["goal"]["acceptable_states"] = [
        {
            "selected_entity_ids": [item_id("refresh", index)],
            "fields": {"note": str(index)},
            "submission_completed": True,
        }
        for index in (1, 2)
    ]
    entities = site_data("refresh")["entities"]
    entities[2]["image_path"] = entities[1]["image_path"]

    representatives = semantic_visual_representative_ids(
        definition,
        {entity["entity_id"]: entity for entity in entities},
        root=tmp_path,
    )

    assert set(representatives) == {item_id("refresh", index) for index in (0, 1, 2)}


class SchemaProtocolTest(unittest.TestCase):
    def test_tasks_and_experiment_conditions_are_separate(self) -> None:
        tasks = [task("graft"), task("refresh"), task("clarify")]
        site_products = [
            product
            for track in ("graft", "refresh", "clarify")
            for product in products(track)
        ]
        self.assertEqual(
            validate_dataset(
                tasks,
                site_data_by_site={
                    "egomart": {
                        "site": "egomart",
                        "entities": [
                            {**product, "entity_id": product["item_id"]}
                            for product in site_products
                        ],
                    }
                },
            ),
            {"clarify": 1, "graft": 1, "refresh": 1},
        )
        self.assertTrue(all("conditions" not in current for current in tasks))
        runs = expand_runs(tasks, PLAN)
        self.assertEqual(len(runs), 27)
        self.assertEqual({run["replicate_id"] for run in runs}, {0, 1, 2})
        self.assertEqual(
            {run["condition"] for run in runs if run["track"] == "graft"},
            set(PLAN.conditions_for("graft")),
        )

    def test_distinct_recordings_can_share_a_natural_request(self) -> None:
        first = task("graft")
        second = task("graft")
        second["id"] = "test-graft-second"
        second["source"] = {"dataset": "fixture", "id": 2}
        second["visual_packets"][0]["frame_paths"] = [
            f"frames/graft-second-{index}.jpg" for index in range(8)
        ]

        validate_dataset(
            [first, second], site_data_by_site={"egomart": site_data("graft")}
        )

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
            validate_dataset(
                [first, second], site_data_by_site={"egomart": site_data("graft")}
            )

    def test_directory_is_the_physical_task_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for track in ("graft", "refresh", "clarify"):
                path = root / track / f"{track}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(task(track)))
            self.assertEqual(
                [current["track"] for current in load_tasks(root)],
                ["clarify", "graft", "refresh"],
            )
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
                "condition": PLAN.condition(run["condition"]).as_dict(),
                "meta": {
                    "replicate_id": run["replicate_id"],
                    "schedule_index": run["schedule_index"],
                    "agent": PLAN.adapter,
                    "model": PLAN.model,
                    "model_version": PLAN.model_version,
                    "max_steps": PLAN.max_steps,
                    "reference_steps": PLAN.reference_steps,
                    "default_release_step": PLAN.default_release_step,
                    "runtime": {"release_trigger": PLAN.release_trigger},
                },
                "steps": 0,
            }
            for run in schedule
        ]
        validate_run_records([graft], records, PLAN)
        with self.assertRaises(ValueError):
            validate_run_records([graft], records[:-1], PLAN)
        with self.assertRaises(ValueError):
            validate_run_records([graft], records + [records[0]], PLAN)
        changed = json.loads(json.dumps(records))
        changed[0]["condition"]["scope"] = "none"
        with self.assertRaisesRegex(ValueError, "schedule/runtime"):
            validate_run_records([graft], changed, PLAN)

    def test_invalid_clarification_fields_are_rejected(self) -> None:
        clarify = task("clarify")
        clarify["clarification"]["valid_slot"] = "color"
        with self.assertRaises(ValidationError):
            validate_task(clarify, site_data("clarify"))
        clarify = task("clarify")
        clarify["clarification"]["candidate_ids"].append(item_id("clarify", 2))
        with self.assertRaisesRegex(ValidationError, "exactly 2"):
            validate_task(clarify, site_data("clarify"))

    def test_clarification_schema_does_not_judge_intent_by_substrings(self) -> None:
        for instruction in (
            "I'd like to make another one of these. Find a making tutorial.",
            "Find a replacement for one of the tools I'm using here.",
            "Find one of the two products.",
        ):
            with self.subTest(instruction=instruction):
                clarify = task("clarify")
                clarify["instructions"]["ambiguous"] = instruction
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
        for track, subtype in (
            ("graft", "augment"),
            ("refresh", "retain"),
            ("clarify", "resolve"),
        ):
            with self.subTest(track=track):
                current = task(track)
                current["subtype"] = subtype
                with self.assertRaises(ValidationError):
                    validate_task(current, site_data(track))

    def test_graft_has_one_initial_goal_and_one_updated_goal(self) -> None:
        unchanged = task("graft")
        unchanged["graft"]["initial_goal"] = unchanged["goal"]
        with self.assertRaisesRegex(ValidationError, "must change the final goal"):
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

    def test_exact_instance_cannot_hide_multiple_acceptable_answers(self) -> None:
        graft = task("graft")
        graft["goal"]["target_granularity"] = "exact_instance"
        graft["graft"]["initial_goal"]["target_granularity"] = "exact_instance"
        graft["goal"]["acceptable_states"].append(
            {
                "selected_entity_ids": [item_id("graft", 1)],
                "fields": {},
                "submission_completed": True,
            }
        )

        with self.assertRaisesRegex(ValidationError, "one exact acceptable state"):
            validate_task(graft, site_data("graft"))

    def test_equivalent_answers_can_use_different_packaging_quantities(self) -> None:
        graft = task("graft")
        graft["graft"]["transition"] = "replace"
        graft["goal"]["acceptable_states"].append(
            {
                "selected_entity_ids": [item_id("graft", 0), item_id("graft", 1)],
                "fields": {},
                "submission_completed": True,
            }
        )

        validate_task(graft, site_data("graft"))

    def test_acceptable_answer_can_encode_required_quantity(self) -> None:
        graft = task("graft")
        for goal in (graft["goal"], graft["graft"]["initial_goal"]):
            for state in goal["acceptable_states"]:
                state["selected_entity_ids"] *= 2
        validate_task(graft, site_data("graft"))

    def test_released_instruction_rejects_internal_frozen_wording(self) -> None:
        graft = task("graft")
        graft["instructions"]["b"] = "Use the frozen average rating."

        with self.assertRaisesRegex(ValidationError, "benchmark-internal wording"):
            validate_task(graft, site_data("graft"))

    def test_released_instruction_flags_a_quoted_catalog_entry(self) -> None:
        graft = task("graft")
        graft["instructions"]["b"] = "Keep only the Acme Graft Product Number 0."
        catalog = site_data("graft")
        catalog["entities"][0]["name"] = "Acme Graft Product Number 0"

        from egocua.schema import catalog_title_warnings

        validate_task(graft, catalog)
        self.assertIn(
            "quotes the catalog entry", catalog_title_warnings(graft, catalog)[0]
        )

    def test_released_instruction_allows_describing_what_sets_the_target_apart(
        self,
    ) -> None:
        graft = task("graft")
        graft["instructions"]["b"] = "Keep only the graft product that ships free."

        validate_task(graft, site_data("graft"))

    def test_released_instruction_allows_a_generic_three_word_category(self) -> None:
        graft = task("graft")
        graft["instructions"]["b"] = "Keep only the black frying pan."
        catalog = site_data("graft")
        catalog["entities"][0]["name"] = "Acme 8 Inch Black Frying Pan Skillet"

        validate_task(graft, catalog)

    def test_graft_relations_are_derived_from_initial_and_updated_goals(self) -> None:
        graft = task("graft")
        graft["graft"]["target_a_id"] = item_id("graft", 0)
        with self.assertRaisesRegex(
            ValidationError, "transition and initial_goal only"
        ):
            validate_task(graft, site_data("graft"))

        graft = task("graft")
        graft["graft"]["transition"] = "augment"
        graft["graft"]["initial_goal"] = {
            "target_granularity": "variant",
            "acceptable_states": [
                {
                    "selected_entity_ids": [item_id("graft", 0)],
                    "fields": {},
                    "submission_completed": True,
                }
            ],
        }
        graft["goal"] = {
            "target_granularity": "variant",
            "acceptable_states": [
                {
                    "selected_entity_ids": [item_id("graft", 1)],
                    "fields": {},
                    "submission_completed": True,
                }
            ],
        }
        with self.assertRaisesRegex(ValidationError, "augment must preserve"):
            validate_task(graft, site_data("graft"))

    def test_removed_sources_of_truth_are_rejected(self) -> None:
        for field, value in (
            ("conditions", ["initial_only"]),
            (
                "expectations",
                {"initial_only": {"acceptable_sets": [[item_id("graft", 0)]]}},
            ),
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
        graft["visual_packets"].append(
            {
                "id": "VB",
                "frame_paths": [f"frames/graft-b-{index}.jpg" for index in range(4)],
            }
        )
        with self.assertRaisesRegex(ValidationError, "visual packet ids"):
            validate_task(graft, site_data("graft"))
        refresh = task("refresh")
        refresh["visual_packets"][0]["target_state"] = "A"
        with self.assertRaisesRegex(ValidationError, "target_state"):
            validate_task(refresh, site_data("refresh"))

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

    def test_release_rejects_a_source_video_crop_as_a_product_image(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            store = root / "sites/egomart/products.json"
            payload = json.loads(store.read_text())
            provenance = payload["products"][0]["image_provenance"]
            provenance.pop("source_url")
            provenance.pop("image_url")
            provenance.update(
                {
                    "source_recording": "episode-1",
                    "retrieval_method": "crop_from_source_recording",
                    "timestamp_seconds": 12.5,
                }
            )
            store.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                EnvironmentError, "independent web page missing"
            ):
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
            payload["products"][1]["image_provenance"]["sha256"] = payload["products"][
                0
            ]["image_provenance"]["sha256"]
            store.write_text(json.dumps(payload))
            with self.assertRaisesRegex(EnvironmentError, "duplicate product image"):
                load_site_data(root, "egomart", require_assets=True)

    def test_release_ignores_a_shared_dossier_file_hash_change(self) -> None:
        graft = release_ready_task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            dossier = root / graft["source"]["dossier"]
            dossier.write_text(dossier.read_text() + "\n")
            validate_source_lineage(graft, root=root)

    def test_release_reuses_unchanged_content_after_source_review_notes(self) -> None:
        from egocua.confidence_packets import CurrentReviewMaterials

        graft = release_ready_task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            original_task = json.dumps(graft, sort_keys=True)
            original_materials = CurrentReviewMaterials(root).packet(graft)[
                "materials_sha256"
            ]
            dossier = root / graft["source"]["dossier"]
            payload = json.loads(dossier.read_text())
            payload["tasks"][0]["repair_history"] = [
                {"reviewer": "named-reviewer", "reason": "Closed the prior concern."}
            ]
            dossier.write_text(json.dumps(payload))
            validate_source_lineage(graft, root=root)
            validate_source_lineage(graft, root=root)
            assert json.dumps(graft, sort_keys=True) == original_task
            assert (
                CurrentReviewMaterials(root).packet(graft)["materials_sha256"]
                == original_materials
            )

    def test_release_rejects_a_changed_source_record_identity(self) -> None:
        graft = release_ready_task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, products("graft"), root)
            dossier = root / graft["source"]["dossier"]
            payload = json.loads(dossier.read_text())
            payload["tasks"][0]["family"] = "refresh"
            dossier.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                LineageError, "source record identity mismatch"
            ):
                validate_source_lineage(graft, root=root)

    def test_release_rejects_a_source_media_reassignment(self) -> None:
        graft = release_ready_task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, products("graft"), root)
            dossier = root / graft["source"]["dossier"]
            payload = json.loads(dossier.read_text())
            payload["tasks"][0]["source"]["media_id"] = "different-recording"
            dossier.write_text(json.dumps(payload))
            with self.assertRaisesRegex(
                LineageError, "source media_id disagrees with frame provenance"
            ):
                validate_source_lineage(graft, root=root)

    def test_release_rejects_a_changed_released_frame(self) -> None:
        graft = release_ready_task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            frame = root / graft["visual_packets"][0]["frame_paths"][0]
            frame.write_bytes(b"changed frame")
            with self.assertRaisesRegex(
                LineageError, "released frame SHA-256 mismatch"
            ):
                validate_source_lineage(graft, root=root)

    def test_release_rejoins_source_and_frame_timestamps(self) -> None:
        graft = release_ready_task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            provenance_path = root / graft["source"]["frame_provenance"]
            provenance = json.loads(provenance_path.read_text())
            provenance["frames"][0]["timestamp_seconds"] += 0.5
            provenance_path.write_text(json.dumps(provenance))
            graft["source"]["frame_provenance_sha256"] = hashlib.sha256(
                provenance_path.read_bytes()
            ).hexdigest()
            with self.assertRaisesRegex(
                LineageError, "provenance timestamp mismatch"
            ):
                validate_source_lineage(graft, root=root)

    def test_human_run_requires_the_workflow_ready_status(self) -> None:
        graft = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, products("graft"), root)
            for status, admitted in (("ready", True), ("needs_review", False)):
                row = {"status": status, "next_action": "fixture"}
                workflow = unittest.mock.Mock()
                workflow.return_value.status.return_value = {"tasks": {graft["id"]: row}}
                with unittest.mock.patch("egocua.gates.SynthesisWorkflow", workflow):
                    if admitted:
                        validate_task_for_human_run(graft, root=root)
                    else:
                        with self.assertRaisesRegex(ValidationError, "ready task"):
                            validate_task_for_human_run(graft, root=root)

    def test_model_preflight_does_not_require_human_reviews(self) -> None:
        graft = task("graft")
        items = products("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, items, root)
            validate_task_for_model_run(graft, root=root)
            del graft["instructions"]["b"]
            with self.assertRaisesRegex(ValidationError, "missing 'b'"):
                validate_task_for_model_run(graft, root=root)

    def test_model_preflight_uses_source_access_not_stored_hashes(self) -> None:
        graft = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(graft, products("graft"), root)
            dossier = root / graft["source"]["dossier"]
            payload = json.loads(dossier.read_text())
            payload["tasks"][0]["construct"]["event_origin"] = "changed"
            dossier.write_text(json.dumps(payload))
            validate_task_for_model_run(graft, root=root)
            dossier.unlink()
            with self.assertRaisesRegex(ValidationError, "graft_screening_manifest"):
                validate_task_for_model_run(graft, root=root)

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

    def test_refresh_equivalent_answers_may_share_a_public_image(self) -> None:
        refresh = task("refresh")
        refresh["goal"]["acceptable_states"].append(
            {
                "selected_entity_ids": [item_id("refresh", 2)],
                "fields": {},
                "submission_completed": True,
            }
        )
        items = products("refresh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(refresh, items, root)
            entities = [
                {**product, "entity_id": product["item_id"]} for product in items
            ]
            entities[2]["image_path"] = entities[1]["image_path"]
            validate_task(
                refresh,
                {"site": "egomart", "entities": entities},
                root=root,
                require_assets=True,
            )

    def test_refresh_old_and_final_targets_must_remain_visually_distinct(self) -> None:
        refresh = task("refresh")
        items = products("refresh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(refresh, items, root)
            entities = [
                {**product, "entity_id": product["item_id"]} for product in items
            ]
            entities[1]["image_path"] = entities[0]["image_path"]
            with self.assertRaisesRegex(ValidationError, "share the same visual asset"):
                validate_task(
                    refresh,
                    {"site": "egomart", "entities": entities},
                    root=root,
                    require_assets=True,
                )

    def test_graft_injects_on_the_task_timeline_once(self) -> None:
        session = _run_session(task("graft"), CONDITIONS["online"])
        self.assertEqual(session.advance(1), [])
        event = session.advance(2)[0]
        self.assertEqual(event["type"], "instruction_injection")
        self.assertEqual(event["trigger_reason"], "task_timeline")
        self.assertEqual(session.advance(5), [])

    def test_graft_conditions_share_the_exact_same_visual_observation(self) -> None:
        graft = task("graft")
        payloads = {
            name: _run_session(graft, PLAN.condition(name)).initial_payload()
            for name in PLAN.conditions_for("graft")
        }
        self.assertEqual(
            payloads["initial_only"]["observations"],
            payloads["upfront"]["observations"],
        )
        self.assertEqual(
            payloads["upfront"]["observations"], payloads["online"]["observations"]
        )

    def test_refresh_releases_on_the_task_timeline(self) -> None:
        session = _run_session(task("refresh"), CONDITIONS["online"])
        self.assertEqual(session.advance(1), [])
        event = session.advance(2)[0]
        self.assertEqual(event["type"], "visual_context_update")
        self.assertEqual(event["trigger_reason"], "task_timeline")

    def test_released_observation_is_allowlisted(self) -> None:
        refresh = task("refresh")
        refresh["visual_packets"][0]["target_state"] = "hidden-marker"
        payload = _run_session(refresh, CONDITIONS["online"]).initial_payload()
        self.assertEqual(
            set(payload), {"instruction", "observations", "clarification_available"}
        )
        self.assertEqual(set(payload["observations"][0]), {"frame_paths"})
        self.assertNotIn("hidden-marker", repr(payload))

    def test_clarification_is_deterministic_and_limited(self) -> None:
        clarify = task("clarify")
        session = _run_session(clarify, CONDITIONS["online"])
        answer = session.ask_user("Can you clarify?", 2)
        self.assertTrue(answer["valid"])
        self.assertEqual(answer["payload"], clarify["clarification"]["canonical_answer"])
        self.assertTrue(session.ask_user("Which variant?", 3)["limit_reached"])
        with self.assertRaises(ProtocolError):
            _run_session(clarify, CONDITIONS["no_ask"]).ask_user("Which variant?", 1)


if __name__ == "__main__":
    unittest.main()


def test_relax_keeps_old_answers_and_adds_optional_alternatives():
    from egocua.schema import _validate_graft_relation
    from egocua.truth import Truth

    definition = task("graft")
    definition["graft"]["transition"] = "relax"
    initial = definition["graft"]["initial_goal"]
    definition["goal"]["acceptable_states"] = initial["acceptable_states"] + [
        {
            "selected_entity_ids": ["new-optional"],
            "fields": {},
            "submission_completed": True,
        }
    ]
    definition["goal"]["target_granularity"] = "category"
    _validate_graft_relation(definition, initial)
    truth = Truth.of(definition)
    assert truth.required_after_update_ids == frozenset()
    assert truth.invalidated_referent_ids == frozenset()
    assert not truth.later_release_changes_referent


@pytest.mark.parametrize(
    "final",
    [
        [(["a"], {})],
        [(["b"], {})],
        [(["a", "b"], {})],
        [(["a"], {"mode": "changed"}), (["b"], {})],
    ],
)
def test_relax_rejects_equal_or_replaced_or_augmented_states(final):
    from egocua.schema import _validate_graft_relation

    initial = {
        "acceptable_states": [
            {"selected_entity_ids": ["a"], "fields": {}, "submission_completed": True}
        ]
    }
    definition = {
        "id": "relax-task",
        "graft": {"transition": "relax"},
        "goal": {
            "acceptable_states": [
                {
                    "selected_entity_ids": ids,
                    "fields": fields,
                    "submission_completed": True,
                }
                for ids, fields in final
            ]
        },
    }
    with pytest.raises(ValidationError):
        _validate_graft_relation(definition, initial)
