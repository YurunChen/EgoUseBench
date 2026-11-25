from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from egocua.construction import (
    ConstructionError,
    _decision_curve,
    _map_sets,
    compile_task_from_site,
    opaque_item_id,
    resolve_authored_record,
    check_selected,
    CurrentConstruction,
)
from egocua.truth import Truth
from egocua.final_state import evaluate_final_state
from egocua.lineage import build_source_identity
from egocua.run_ledger import task_sha256
from tests.construction_fixtures import screening_record, source_identity
from tests.construction_fixtures import catalog_resolution
from tests.helpers import plan_payload
from tools.audit import report_decision_curves, report_task_difficulty


def _owner_workspace(root):
    record, site = screening_record(root, "graft")
    definition = _compile(record, site, root)
    dossier_path = root / definition["source"]["dossier"]
    dossier_path.parent.mkdir(parents=True, exist_ok=True)
    dossier_path.write_text(json.dumps({"tasks": [record]}))
    authored_path = root / "data/instructions/authored.json"
    authored_path.parent.mkdir(parents=True, exist_ok=True)
    authored_path.write_text(
        json.dumps(
            {
                "instructions": {
                    definition["id"]: record["construct"]["task_blueprint"][
                        "instructions"
                    ],
                }
            }
        )
    )
    site_path = root / "sites/egomart/products.json"
    site_path.parent.mkdir(parents=True, exist_ok=True)
    site_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site": "egomart",
                "products": site["entities"],
            }
        )
    )
    task_path = root / "data/tasks/graft" / f"{definition['id']}.json"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(json.dumps(definition))
    (root / "data/selection.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source_manifests": [],
                "task_set": {"graft": [definition["id"]], "refresh": [], "clarify": []},
            }
        )
    )
    return definition, record, authored_path


def test_incremental_compile_keeps_one_current_file_when_track_changes(
    tmp_path: Path, monkeypatch
) -> None:
    from egocua import construction

    definition, _, _ = _owner_workspace(tmp_path)
    old_path = tmp_path / "data/tasks/refresh" / f"{definition['id']}.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text(json.dumps({**definition, "track": "refresh"}))
    unrelated = tmp_path / "data/tasks/refresh/unrelated.json"
    unrelated.write_text('{"unchanged": true}')
    monkeypatch.setattr(
        construction,
        "build_selected",
        lambda *args, **kwargs: ([definition], {}, {"constructed": 1}),
    )

    construction.write_selected(tmp_path, task_ids=[definition["id"]])

    paths = list((tmp_path / "data/tasks").glob(f"*/{definition['id']}.json"))
    assert paths == [tmp_path / "data/tasks/graft" / f"{definition['id']}.json"]
    assert json.loads(paths[0].read_text()) == definition
    assert unrelated.read_text() == '{"unchanged": true}'


def test_scoped_build_rejects_recording_already_used_by_another_member(
    tmp_path, monkeypatch
):
    from egocua import construction

    ids = ["selected", "other"]
    manifest = tmp_path / "source.json"
    records = [
        (
            manifest,
            {
                "task_id": task_id,
                "source": {"media_id": "clip", "original_video_id": "same-recording"},
            },
        )
        for task_id in ids
    ]
    manifest.write_text(json.dumps({"dataset": {"name": "HD-EPIC"}}))
    monkeypatch.setattr(
        construction,
        "load_selection",
        lambda _: {
            "task_set": {"graft": ids, "refresh": [], "clarify": []},
            "source_manifests": ["source.json"],
        },
    )
    monkeypatch.setattr(construction, "_screened_records", lambda _: records)

    with pytest.raises(ConstructionError, match="source cluster.*shared by"):
        construction.build_selected(tmp_path, task_ids=["selected"])


@pytest.mark.parametrize("owner", ["authored", "blueprint"])
def test_status_detects_uncompiled_owner_before_opening_old_packet(
    tmp_path, monkeypatch, owner
):
    from egocua.synthesis_workflow import SynthesisWorkflow
    from egocua.synthesis_reports import REPORTS

    definition, record, authored_path = _owner_workspace(tmp_path)
    if owner == "authored":
        authored = json.loads(authored_path.read_text())
        authored["instructions"][definition["id"]]["a"] = (
            "Choose the first required object."
        )
        authored_path.write_text(json.dumps(authored))
    else:
        record["construct"]["task_blueprint"]["target_granularity"] = "category"
        (tmp_path / definition["source"]["dossier"]).write_text(
            json.dumps({"tasks": [record]})
        )
    report_path = tmp_path / REPORTS["assessment"].path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": REPORTS["assessment"].schema_version,
                "results": [
                    {
                        "task_id": definition["id"],
                        "status": "completed",
                        "assessments": [{}, {}],
                    }
                ],
            }
        )
    )
    original = report_path.read_bytes()

    def old_packet(*args):
        raise AssertionError(
            "uncompiled owners must be detected before opening old materials"
        )

    monkeypatch.setattr(
        "egocua.synthesis_workflow.CurrentReviewMaterials.packet", old_packet
    )
    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["status"] == "needs_review"
    assert row["next_action"] == "compile_task"
    assert report_path.read_bytes() == original


def test_next_compiles_only_affected_task_and_retains_review(tmp_path, monkeypatch):
    from egocua import construction
    from egocua.synthesis_workflow import SynthesisWorkflow

    definition, _, authored_path = _owner_workspace(tmp_path)
    plan = tmp_path / "configs/benchmark.json"
    plan.parent.mkdir()
    plan.write_text(json.dumps(plan_payload()))
    authored = json.loads(authored_path.read_text())
    authored["instructions"][definition["id"]]["a"] = (
        "Choose the first required object."
    )
    authored_path.write_text(json.dumps(authored))
    other_path = tmp_path / "data/tasks/graft/unrelated.json"
    other_path.write_text('{"preserve":"other task"}')
    review = tmp_path / "data/reviews/retained.json"
    review.parent.mkdir(parents=True)
    review.write_text('{"preserve":"specific review"}')
    original_review = review.read_bytes()
    original_other = other_path.read_bytes()
    compiled = deepcopy(definition)
    compiled["instructions"] = authored["instructions"][definition["id"]]

    def selected(root, *, task_ids):
        assert task_ids == [definition["id"]]
        return [compiled], {}, {"constructed_tasks": 1}

    monkeypatch.setattr(construction, "build_selected", selected)
    monkeypatch.setattr(
        "egocua.construction.load_tasks",
        lambda _: [
            json.loads(
                (tmp_path / "data/tasks/graft" / f"{definition['id']}.json").read_text()
            )
        ],
    )
    item = SynthesisWorkflow(tmp_path).next_work_item(definition["id"])

    assert item["compiled_task"] == definition["id"]
    assert item["kind"] == "run_deterministic_audit"
    assert other_path.read_bytes() == original_other
    assert review.read_bytes() == original_review


def test_current_construction_reuses_source_site_and_authored_reads(
    tmp_path, monkeypatch
):
    from egocua import construction

    definition, _, _ = _owner_workspace(tmp_path)
    reads = {"source": 0, "site": 0, "authored": 0}
    for name, attribute in [
        ("source", "load_dossier"),
        ("site", "load_site_data"),
        ("authored", "_authored_overlay"),
    ]:
        actual = getattr(construction, attribute)

        def counted(*args, _actual=actual, _name=name, **kwargs):
            reads[_name] += 1
            return _actual(*args, **kwargs)

        monkeypatch.setattr(construction, attribute, counted)
    current = CurrentConstruction(tmp_path)

    assert current.matches(definition)
    assert current.matches(definition)
    assert reads == {"source": 1, "site": 1, "authored": 1}


def test_current_construction_exposes_invalid_owner(tmp_path):
    definition, _, authored_path = _owner_workspace(tmp_path)
    authored_path.write_text(
        json.dumps({"instructions": {definition["id"]: {"a": "Missing B"}}})
    )

    with pytest.raises(ConstructionError, match="missing"):
        CurrentConstruction(tmp_path).matches(definition)


def test_unrelated_source_record_does_not_require_recompilation(tmp_path):
    definition, record, _ = _owner_workspace(tmp_path)
    dossier = tmp_path / definition["source"]["dossier"]
    dossier.write_text(
        json.dumps({"tasks": [record, {"task_id": "unrelated", "note": "changed"}]})
    )

    assert CurrentConstruction(tmp_path).matches(definition)


def _rebuild_owner(root: Path, definition: dict) -> dict:
    source = definition["source"]
    raw = json.loads((root / source["dossier"]).read_text())["tasks"][0]
    identity = build_source_identity(
        raw,
        dossier_reference=source["dossier"],
        provenance_reference=source["frame_provenance"],
        root=root,
    )
    return compile_task_from_site(
        resolve_authored_record(raw, root=root),
        source_identity=identity,
        site_data=CurrentConstruction(root).site("egomart"),
        root=root,
    )


@pytest.mark.parametrize("change", ["unresolved", "repair_history", "named_review"])
def test_review_notes_do_not_change_actual_compiled_content(
    tmp_path, monkeypatch, change
):
    definition, record, _ = _owner_workspace(tmp_path)
    (tmp_path / definition["source"]["frame_provenance"]).write_text("{}")
    definition = _rebuild_owner(tmp_path, definition)
    path = tmp_path / "data/tasks/graft" / f"{definition['id']}.json"
    path.write_text(json.dumps(definition))
    original = path.read_bytes()
    original_hash = task_sha256(definition)
    if change == "named_review":
        record["construct"]["catalog_resolution"]["semantic_review"] = {
            "reviewer": "current-named-reviewer",
            "reason": "Unchanged public candidates checked again.",
        }
    else:
        record[change] = ["A concrete task-local review note, not an execution input."]
    (tmp_path / definition["source"]["dossier"]).write_text(
        json.dumps({"tasks": [record]})
    )
    rebuilt = _rebuild_owner(tmp_path, definition)
    monkeypatch.setattr(
        "egocua.construction.build_selected",
        lambda root: ([rebuilt], {}, {"constructed_tasks": 1}),
    )

    assert "record_sha256" not in rebuilt["source"]
    assert task_sha256(rebuilt) == original_hash
    assert CurrentConstruction(tmp_path).matches(definition)
    assert check_selected(tmp_path) == {"constructed_tasks": 1}
    assert path.read_bytes() == original


@pytest.mark.parametrize("change", ["instructions", "frames", "answers"])
def test_compile_check_compares_actual_changed_owner_content(
    tmp_path, monkeypatch, change
):
    definition, record, authored_path = _owner_workspace(tmp_path)
    (tmp_path / definition["source"]["frame_provenance"]).write_text("{}")
    definition = _rebuild_owner(tmp_path, definition)
    path = tmp_path / "data/tasks/graft" / f"{definition['id']}.json"
    path.write_text(json.dumps(definition))
    if change == "instructions":
        authored = json.loads(authored_path.read_text())
        authored["instructions"][definition["id"]]["a"] = (
            "Choose another required outcome."
        )
        authored_path.write_text(json.dumps(authored))
    elif change == "frames":
        record["packets"][0]["frame_paths"].reverse()
    else:
        catalog = record["construct"]["catalog_resolution"]
        equivalent = catalog["ambient_entity_ids"].pop()
        record["construct"]["task_blueprint"]["equivalent_role_ids"] = {
            "second_product": [equivalent]
        }
    if change != "instructions":
        (tmp_path / definition["source"]["dossier"]).write_text(
            json.dumps({"tasks": [record]})
        )
    rebuilt = _rebuild_owner(tmp_path, definition)
    monkeypatch.setattr(
        "egocua.construction.build_selected", lambda root: ([rebuilt], {}, {})
    )

    assert task_sha256(rebuilt) != task_sha256(definition)
    assert not CurrentConstruction(tmp_path).matches(definition)
    with pytest.raises(ConstructionError, match="stale"):
        check_selected(tmp_path)


def test_compile_check_ignores_changes_to_unrelated_source_records(
    monkeypatch, tmp_path
):
    compiled = {
        "id": "test-one",
        "source": {
            "dossier": "source.json",
            "dossier_sha256": "new-batch",
            "frame_provenance_sha256": "new-sidecar",
        },
        "instructions": {"a": "Buy this"},
    }
    previous = {
        **compiled,
        "source": {
            **compiled["source"],
            "dossier_sha256": "old-batch",
            "frame_provenance_sha256": "old-sidecar",
        },
    }
    monkeypatch.setattr(
        "egocua.construction.build_selected",
        lambda root: ([compiled], {}, {"constructed_tasks": 1}),
    )
    monkeypatch.setattr("egocua.construction.load_tasks", lambda root: [previous])
    assert check_selected(tmp_path) == {"constructed_tasks": 1}


@pytest.mark.parametrize("change", ["instructions", "visual_packets", "goal"])
def test_compile_check_still_rejects_task_local_changes(monkeypatch, tmp_path, change):
    compiled = {
        "id": "test-one",
        "source": {"dossier": "source.json"},
        "instructions": {"a": "Buy this"},
        "visual_packets": [{"id": "V0", "frame_paths": ["new.jpg"]}],
        "goal": {"acceptable_states": [{"selected_entity_ids": ["new"], "fields": {}}]},
    }
    previous = {**compiled, "source": dict(compiled["source"])}
    if change == "instructions":
        previous["instructions"] = {"a": "Buy something else"}
    elif change == "visual_packets":
        previous["visual_packets"] = [{"id": "V0", "frame_paths": ["old.jpg"]}]
    else:
        previous["goal"] = {
            "acceptable_states": [{"selected_entity_ids": ["old"], "fields": {}}]
        }
    monkeypatch.setattr(
        "egocua.construction.build_selected", lambda root: ([compiled], {}, {})
    )
    monkeypatch.setattr("egocua.construction.load_tasks", lambda root: [previous])
    with pytest.raises(ConstructionError, match="stale"):
        check_selected(tmp_path)


@pytest.mark.parametrize(
    "initial",
    [
        [
            {"selected_entity_ids": ["a"], "fields": {}, "submission_completed": True},
            {"selected_entity_ids": ["b"], "fields": {}, "submission_completed": True},
        ],
        [
            {
                "selected_entity_ids": ["a", "b"],
                "fields": {"route": "cycling"},
                "submission_completed": True,
            }
        ],
        [
            {
                "selected_entity_ids": ["a", "b"],
                "fields": {},
                "submission_completed": True,
            },
            {"selected_entity_ids": ["a"], "fields": {}, "submission_completed": True},
        ],
    ],
)
def test_goal_reference_cannot_hide_wrong_alternatives_or_field_values(
    initial: list[dict],
) -> None:
    task = {
        "track": "graft",
        "visual_packets": [{"id": "V0", "frame_paths": ["frame.jpg"]}],
        "graft": {"initial_goal": {"acceptable_states": initial}},
        "goal": {
            "acceptable_states": [
                {
                    "selected_entity_ids": ["a", "b"],
                    "fields": {},
                    "submission_completed": True,
                }
            ]
        },
    }
    record = {
        "construct": {
            "decision": {
                "candidates": {
                    "before": {"goal_ref": "initial"},
                    "after": {"goal_ref": "final"},
                },
                "observed": [
                    {"frame": "V0:0", "live": ["before", "after"]},
                    {
                        "frame": "utterance",
                        "live": ["before"],
                        "narrowed_by": "user_utterance",
                        "why": "The user keeps the earlier goal.",
                    },
                ],
                "established_by": "test",
            }
        }
    }
    site = {"entities": [{"entity_id": "a"}, {"entity_id": "b"}]}

    with pytest.raises(ConstructionError, match="complete acceptable goal"):
        _decision_curve(record, task, site, "test-goal-reference")


def _compile(
    record: dict[str, Any], site: dict[str, Any], root: Path
) -> dict[str, Any]:
    return compile_task_from_site(
        record,
        source_identity=source_identity(record),
        site_data=site,
        root=root,
    )


def _entity_named(site: dict[str, Any], name: str) -> str:
    return next(
        entity["entity_id"] for entity in site["entities"] if entity["name"] == name
    )


def _rewrite_snapshot(
    record: dict[str, Any],
    root: Path,
    field: str,
    value: Any,
) -> None:
    resolution = record["construct"]["catalog_resolution"]
    path = root / resolution["query_rounds"][0]["snapshot_path"]
    snapshot = json.loads(path.read_text())
    snapshot["rounds"][0]["results"][0][field] = value
    path.write_text(json.dumps(snapshot))


def test_map_sets_expands_role_equivalents_without_changing_state_shape() -> None:
    states = _map_sets(
        [["knife", "produce"]],
        {"knife": "knife-a", "produce": "pepper"},
        "test/final_goal_roles",
        {"knife": ["knife-b", "knife-c"]},
    )

    assert states == [
        ["knife-a", "pepper"],
        ["knife-b", "pepper"],
        ["knife-c", "pepper"],
    ]


def test_map_sets_accepts_cross_role_equivalent_as_an_initial_answer() -> None:
    states = _map_sets(
        [["making"]],
        {"making": "making-video"},
        "initial",
        {"making": ["both-video"]},
    )

    assert states == [["making-video"], ["both-video"]]


def test_map_sets_preserves_all_distinct_cross_role_pairs() -> None:
    states = _map_sets(
        [["making", "layout"]],
        {"making": "making-video", "layout": "both-video"},
        "final",
        {"making": ["both-video"], "layout": ["layout-video"]},
    )

    assert states == [
        ["making-video", "both-video"],
        ["making-video", "layout-video"],
        ["both-video", "layout-video"],
    ]


def test_map_sets_deduplicates_alternative_role_assignments() -> None:
    states = _map_sets(
        [["making", "layout"]],
        {"making": "both-a", "layout": "both-b"},
        "final",
        {"making": ["both-b"], "layout": ["both-a"]},
    )

    assert states == [["both-a", "both-b"]]


def test_map_sets_rejects_roles_that_only_share_one_entity() -> None:
    with pytest.raises(ConstructionError, match="no distinct-entity assignment"):
        _map_sets(
            [["making", "layout"]],
            {"making": "both-video", "layout": "both-video"},
            "final",
        )


@pytest.mark.parametrize(
    ("selected_roles", "expected"),
    [
        (["base_product", "second_product"], True),
        (["second_product"], False),
        (["second_product", "second_product"], False),
    ],
)
def test_compiled_augment_requires_two_distinct_entities_with_overlapping_roles(
    tmp_path: Path,
    selected_roles: list[str],
    expected: bool,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    blueprint = record["construct"]["task_blueprint"]
    blueprint["target_granularity"] = "category"
    blueprint["equivalent_role_ids"] = {
        "base_product": [opaque_item_id(record["task_id"], "second_product")],
    }

    task = _compile(record, site, tmp_path)
    submitted = {
        "selected_entity_ids": [
            opaque_item_id(record["task_id"], role) for role in selected_roles
        ],
        "fields": {},
        "submission_completed": True,
    }

    assert evaluate_final_state(task["goal"], submitted) is expected


def test_authored_wording_overlays_without_mutating_source_record(
    tmp_path: Path,
) -> None:
    record, _ = screening_record(tmp_path, "graft")
    original = dict(record["construct"]["task_blueprint"]["instructions"])
    authored = {
        "a": "Add the first object.",
        "b": "Also add the later object, and keep the first.",
        "bundled": "Add both objects.",
    }
    path = tmp_path / "data" / "instructions" / "authored.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"instructions": {record["task_id"]: authored}}))

    resolved = resolve_authored_record(record, root=tmp_path)

    assert record["construct"]["task_blueprint"]["instructions"] == original
    assert resolved["construct"]["task_blueprint"]["instructions"] == authored


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_compile_task_from_site_produces_a_valid_executable_task(
    track: str, tmp_path: Path
) -> None:
    record, site = screening_record(tmp_path, track)

    task = _compile(record, site, tmp_path)

    assert task["track"] == track
    assert "environment" not in task
    assert task["source"] == source_identity(record)
    assert set(task) <= {
        "schema_version",
        "id",
        "track",
        "site",
        "source",
        "instructions",
        "visual_packets",
        "goal",
        "graft",
        "refresh",
        "clarification",
        "truth",
    }
    if track == "graft":
        assert task["graft"]["transition"] == "augment"
        assert task["truth"]["tempting_referent_ids"] == [
            opaque_item_id(record["task_id"], "distractor_0")
        ]
    if track == "clarify":
        # Derived, not declared: the fixture curve narrows on how the scene looks,
        # which a sentence carries, so the deixis level is the shallow one.
        assert task["clarification"]["deixis_level"] == 1


@pytest.mark.parametrize(
    ("reporter", "field", "expected"),
    [
        (report_decision_curves, "settled_at", "V0:0"),
        (report_task_difficulty, "subtype", "retain"),
    ],
)
def test_initially_clear_refresh_retain_compiles_and_reports(
    tmp_path: Path,
    monkeypatch,
    reporter,
    field: str,
    expected: str,
) -> None:
    record, site = screening_record(tmp_path, "refresh")
    record["subtype"] = "retain"
    blueprint = record["construct"]["task_blueprint"]
    blueprint["initial_target_role"] = "second_product"
    blueprint["instruction"] = (
        "Find the original product; the later tool is only a helper."
    )
    curve = record["construct"]["decision"]
    curve["observed"] = [
        {"frame": observation["frame"], "live": ["second_product"]}
        for observation in curve["observed"]
    ]
    curve["observed"][0].update(
        {
            "narrowed_by": "scene_appearance",
            "why": "The original product is already clear in the initial frame.",
        }
    )
    compiled = _compile(record, site, tmp_path)
    monkeypatch.setattr(reporter, "load_tasks", lambda _: [compiled])
    monkeypatch.setattr(reporter, "load_site_data", lambda *_: site)
    monkeypatch.setattr(reporter, "source_record", lambda *_, **__: record)

    report = reporter.build_report(tmp_path)

    assert report["tasks"][compiled["id"]][field] == expected


def test_catalog_resolution_compiles_a_task_scoped_realistic_neighborhood(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    neighbor_id = record["construct"]["catalog_resolution"]["core_neighbor_entity_ids"][
        0
    ]

    compiled = _compile(record, site, tmp_path)

    assert neighbor_id in compiled["truth"]["site_scope_entity_ids"]
    assert set(compiled["truth"]["site_scope_entity_ids"]) >= set(
        compiled["truth"]["tempting_referent_ids"]
    )


def test_compilation_rejects_answer_classified_as_core_neighbor(tmp_path: Path) -> None:
    record, site = screening_record(tmp_path, "graft")
    answer_id = _compile(record, site, tmp_path)["goal"]["acceptable_states"][0][
        "selected_entity_ids"
    ][-1]
    resolution = record["construct"]["catalog_resolution"]
    resolution["core_neighbor_entity_ids"][0] = answer_id
    resolution["rejected"][0]["entity_id"] = answer_id

    with pytest.raises(
        ConstructionError, match="also classified as rejected neighbors"
    ):
        _compile(record, site, tmp_path)


def test_catalog_neighbor_requires_a_recorded_discovery_query(tmp_path: Path) -> None:
    record, site = screening_record(tmp_path, "graft")
    del record["construct"]["catalog_resolution"]["query_rounds"][0]["query"]

    with pytest.raises(ConstructionError, match="lacks query"):
        _compile(record, site, tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("rank", 2, "ranks must be consecutive"),
        ("title", "", "title is missing"),
        ("source_url", "https://example.invalid/fake", "source URL disagrees"),
        ("visual_path", "product-images/fake.jpg", "visual disagrees"),
        ("mock_detail_path", "/product/fake", "detail route disagrees"),
    ],
)
def test_search_snapshot_rows_must_match_the_real_mock_entity(
    field: str,
    value: Any,
    error: str,
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    _rewrite_snapshot(record, tmp_path, field, value)

    with pytest.raises(ConstructionError, match=error):
        _compile(record, site, tmp_path)


def test_catalog_neighbor_requires_a_source_grounded_rejection(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    record["construct"]["catalog_resolution"]["rejected"] = []

    with pytest.raises(ConstructionError, match="lacks a rejection record"):
        _compile(record, site, tmp_path)


@pytest.mark.parametrize(
    "field", ["reason", "why_plausible", "failure_predicates", "semantic_review"]
)
def test_core_neighbor_compilation_checks_facts_not_review_admission(
    field: str,
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    del record["construct"]["catalog_resolution"]["rejected"][0][field]

    if field == "semantic_review":
        assert _compile(record, site, tmp_path)["id"] == record["task_id"]
        return

    with pytest.raises(
        ConstructionError, match="rejection facts and failure predicates"
    ):
        _compile(record, site, tmp_path)


def test_compilation_always_scopes_runtime_to_task_semantics(tmp_path: Path) -> None:
    record, site = screening_record(tmp_path, "graft")

    compiled = _compile(record, site, tmp_path)

    scope = set(compiled["truth"]["site_scope_entity_ids"])
    assert scope
    assert scope >= set(compiled["truth"]["tempting_referent_ids"])
    assert scope >= {
        entity_id
        for state in compiled["goal"]["acceptable_states"]
        for entity_id in state["selected_entity_ids"]
    }


def test_compile_task_from_site_refuses_a_missing_role_entity(tmp_path: Path) -> None:
    record, site = screening_record(tmp_path, "refresh")
    site["entities"] = [
        entity
        for entity in site["entities"]
        if entity["entity_id"] != opaque_item_id(record["task_id"], "first_product")
    ]

    with pytest.raises(ConstructionError, match="missing referenced entities"):
        _compile(record, site, tmp_path)


def test_refresh_goal_is_derived_from_its_single_final_target_role(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "refresh")

    task = _compile(record, site, tmp_path)

    assert task["goal"]["acceptable_states"][0]["selected_entity_ids"] == [
        _entity_named(site, "Product 1")
    ]
    assert task["refresh"]["initial_entity_id"] == _entity_named(site, "Product 0")


def test_refresh_may_register_a_later_distractor_without_changing_the_goal(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "refresh")
    record["construct"]["task_blueprint"]["later_distractor_role"] = "distractor_0"

    task = _compile(record, site, tmp_path)

    assert task["goal"]["acceptable_states"][0]["selected_entity_ids"] == [
        _entity_named(site, "Product 1")
    ]


def test_clarify_hidden_target_must_be_one_of_exactly_two_candidates(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "clarify")
    record["construct"]["task_blueprint"]["hidden_target_role"] = "distractor_0"

    with pytest.raises(ConstructionError, match="must be one of the candidates"):
        _compile(record, site, tmp_path)


def test_clarify_compiler_rejects_the_obsolete_nested_blueprint(tmp_path: Path) -> None:
    record, site = screening_record(tmp_path, "clarify")
    blueprint = record["construct"]["task_blueprint"]
    blueprint["instructions"] = {
        "ambiguous": blueprint.pop("ambiguous_instruction"),
        "oracle": blueprint.pop("oracle_instruction"),
    }

    with pytest.raises(ConstructionError, match="obsolete fields"):
        _compile(record, site, tmp_path)


def test_graft_compiler_rejects_obsolete_b_essential_witness(tmp_path: Path) -> None:
    record, site = screening_record(tmp_path, "graft")
    record["construct"]["task_blueprint"]["b_essential_witness"] = ["base_product"]

    with pytest.raises(ConstructionError, match="obsolete field 'b_essential_witness'"):
        _compile(record, site, tmp_path)


def test_graft_augment_has_a_failure_surface_even_without_a_stale_entity(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    first = opaque_item_id(record["task_id"], "base_product")
    second = opaque_item_id(record["task_id"], "second_product")
    record["construct"]["decision"]["candidates"] = {
        "first_only": [first],
        "both": [first, second],
    }
    for observation in record["construct"]["decision"]["observed"]:
        observation["live"] = ["first_only", "both"]
    record["construct"]["decision"]["observed"][-1].update(
        {
            "live": ["both"],
            "narrowed_by": "user_utterance",
            "why": "the later instruction adds the second item",
        }
    )
    record["construct"]["catalog_resolution"] = catalog_resolution(
        tmp_path,
        record["task_id"],
        ["base_product", "second_product"],
        ["base_product", "second_product"],
    )

    compiled = _compile(record, site, tmp_path)

    assert compiled["truth"]["tempting_referent_ids"] == []
    assert Truth.of(compiled).required_after_update_ids == {second}


@pytest.mark.parametrize(
    "invalid_state", ["partial", "mixed_alternatives", "duplicate"]
)
def test_compiler_rejects_curve_that_does_not_settle_on_a_complete_goal(
    tmp_path: Path,
    invalid_state: str,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    first = opaque_item_id(record["task_id"], "base_product")
    second = opaque_item_id(record["task_id"], "second_product")
    settled = [first]
    if invalid_state == "mixed_alternatives":
        record["subtype"] = "drop"
        blueprint = record["construct"]["task_blueprint"]
        blueprint["initial_goal_roles"] = [["base_product", "second_product"]]
        blueprint["final_goal_roles"] = [["base_product"], ["second_product"]]
        settled = [first, second]
    elif invalid_state == "duplicate":
        settled = [first, second, second]
    record["construct"]["decision"]["candidates"]["base_product"] = settled

    with pytest.raises(ConstructionError, match="complete acceptable goal"):
        _compile(record, site, tmp_path)


@pytest.mark.parametrize("settled_role", ["base_product", "distractor_1"])
def test_graft_compiles_declared_category_equivalents_as_acceptable_states(
    tmp_path: Path,
    settled_role: str,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    blueprint = record["construct"]["task_blueprint"]
    blueprint["target_granularity"] = "category"
    blueprint["environment_role_ids"] = {
        "base_product": opaque_item_id(record["task_id"], "base_product"),
        "base_alternative": opaque_item_id(record["task_id"], "distractor_1"),
        "second_product": opaque_item_id(record["task_id"], "second_product"),
    }
    blueprint["initial_goal_roles"] = [
        ["base_product"],
        ["base_alternative"],
    ]
    blueprint["final_goal_roles"] = [
        ["base_product", "second_product"],
        ["base_alternative", "second_product"],
    ]
    record["construct"]["catalog_resolution"] = catalog_resolution(
        tmp_path,
        record["task_id"],
        ["base_product", "distractor_1", "second_product", "distractor_0"],
        ["base_product", "distractor_1", "second_product"],
    )
    record["construct"]["decision"]["candidates"]["base_product"] = [
        opaque_item_id(record["task_id"], settled_role),
        opaque_item_id(record["task_id"], "second_product"),
    ]

    compiled = _compile(record, site, tmp_path)

    assert len(compiled["graft"]["initial_goal"]["acceptable_states"]) == 2
    assert len(compiled["goal"]["acceptable_states"]) == 2


def test_graft_drop_compiles_initial_only_equivalents_with_the_same_image(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "graft")
    record["subtype"] = "drop"
    blueprint = record["construct"]["task_blueprint"]
    blueprint["initial_goal_roles"] = [["base_product", "second_product"]]
    blueprint["final_goal_roles"] = [["base_product"]]
    record["construct"]["decision"]["candidates"]["base_product"] = [
        opaque_item_id(record["task_id"], "base_product"),
    ]
    second = opaque_item_id(record["task_id"], "second_product")
    alternative = opaque_item_id(record["task_id"], "distractor_0")
    blueprint["equivalent_role_ids"] = {"second_product": [alternative]}
    record["construct"]["catalog_resolution"] = catalog_resolution(
        tmp_path,
        record["task_id"],
        ["base_product", "second_product"],
        ["base_product"],
    )
    entities = {entity["entity_id"]: entity for entity in site["entities"]}
    (tmp_path / entities[alternative]["image_path"]).write_bytes(
        (tmp_path / entities[second]["image_path"]).read_bytes()
    )

    compiled = _compile(record, site, tmp_path)

    assert len(compiled["graft"]["initial_goal"]["acceptable_states"]) == 2


def test_refresh_compiles_declared_category_equivalents_as_acceptable_states(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "refresh")
    blueprint = record["construct"]["task_blueprint"]
    blueprint["target_granularity"] = "category"
    equivalent = opaque_item_id(record["task_id"], "distractor_0")
    blueprint["equivalent_role_ids"] = {
        "second_product": [equivalent],
    }
    record["construct"]["catalog_resolution"] = catalog_resolution(
        tmp_path,
        record["task_id"],
        ["first_product", "second_product"],
        ["second_product", "distractor_0"],
    )

    compiled = _compile(record, site, tmp_path)

    assert compiled["goal"]["acceptable_states"] == [
        {
            "selected_entity_ids": [
                opaque_item_id(record["task_id"], "second_product")
            ],
            "fields": {},
        },
        {
            "selected_entity_ids": [equivalent],
            "fields": {},
        },
    ]


def test_refresh_compiles_complete_initial_alternatives_from_source(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "refresh")
    blueprint = record["construct"]["task_blueprint"]
    blueprint["target_granularity"] = "category"
    alternative = opaque_item_id(record["task_id"], "distractor_0")
    blueprint["equivalent_role_ids"] = {"first_product": [alternative]}
    record["construct"]["catalog_resolution"] = catalog_resolution(
        tmp_path,
        record["task_id"],
        ["first_product", "second_product"],
        ["second_product"],
    )
    compiled = _compile(record, site, tmp_path)
    initial_ids = {
        state["selected_entity_ids"][0]
        for state in compiled["truth"]["initial_acceptable_states"]
    }
    assert initial_ids == {
        opaque_item_id(record["task_id"], "first_product"),
        alternative,
    }
    assert Truth.of(compiled).initial_referent_ids == initial_ids


def test_clarify_compiles_site_equivalents_without_changing_physical_candidates(
    tmp_path: Path,
) -> None:
    record, site = screening_record(tmp_path, "clarify")
    blueprint = record["construct"]["task_blueprint"]
    blueprint["target_granularity"] = "category"
    equivalent = opaque_item_id(record["task_id"], "distractor_0")
    blueprint["equivalent_role_ids"] = {
        "blue_product": [equivalent],
    }
    record["construct"]["catalog_resolution"] = catalog_resolution(
        tmp_path,
        record["task_id"],
        ["red_product", "blue_product"],
        ["blue_product", "distractor_0"],
    )

    compiled = _compile(record, site, tmp_path)

    assert compiled["clarification"]["candidate_ids"] == [
        opaque_item_id(record["task_id"], "red_product"),
        opaque_item_id(record["task_id"], "blue_product"),
    ]
    assert compiled["goal"]["acceptable_states"] == [
        {
            "selected_entity_ids": [opaque_item_id(record["task_id"], "blue_product")],
            "fields": {},
        },
        {
            "selected_entity_ids": [equivalent],
            "fields": {},
        },
    ]


def test_relax_compiles_and_accepts_old_or_new_choice_but_not_both(tmp_path):
    record, site = screening_record(tmp_path, "graft")
    record["subtype"] = "relax"
    blueprint = record["construct"]["task_blueprint"]
    blueprint["initial_goal_roles"] = [["base_product"]]
    blueprint["final_goal_roles"] = [["base_product"], ["second_product"]]
    blueprint["target_granularity"] = "category"
    record["construct"]["decision"] = {
        "recording": "key_events",
        "candidates": {
            "restricted": {"goal_ref": "initial"},
            "relaxed": {"goal_ref": "final"},
        },
        "established_by": "fixture",
        "observed": [
            {"frame": "V0:0", "live": ["restricted", "relaxed"]},
            {
                "frame": "utterance",
                "live": ["relaxed"],
                "narrowed_by": "user_utterance",
                "why": "The brand constraint is removed.",
            },
        ],
    }
    record["construct"]["catalog_resolution"] = catalog_resolution(
        tmp_path,
        record["task_id"],
        ["base_product", "second_product"],
        ["base_product", "second_product"],
    )
    compiled = _compile(record, site, tmp_path)
    first = opaque_item_id(record["task_id"], "base_product")
    second = opaque_item_id(record["task_id"], "second_product")
    assert evaluate_final_state(
        compiled["goal"], {"selected_entity_ids": [first], "fields": {}}
    )
    assert evaluate_final_state(
        compiled["goal"], {"selected_entity_ids": [second], "fields": {}}
    )
    assert not evaluate_final_state(
        compiled["goal"], {"selected_entity_ids": [first, second], "fields": {}}
    )
    assert Truth.of(compiled).required_after_update_ids == frozenset()


def test_candidate_compiles_without_changing_formal_members(tmp_path, monkeypatch):
    from egocua import construction

    definition, record, _ = _owner_workspace(tmp_path)
    (tmp_path / 'data/tasks/graft' / f'{definition["id"]}.json').unlink()
    monkeypatch.setattr(construction, 'load_tasks', lambda _: [])
    path = tmp_path / definition['source']['dossier']
    selection = (tmp_path / 'data/selection.json').read_bytes()
    monkeypatch.setattr(construction, '_screened_records', lambda _: [(path, record)])
    monkeypatch.setattr(construction, 'discover_frame_provenance', lambda *args, **kwargs: (definition['source']['frame_provenance'], {}))
    monkeypatch.setattr(construction, 'build_source_identity', lambda *args, **kwargs: definition['source'])

    candidates = construction.load_construction_tasks(tmp_path, [definition['id']])

    assert candidates == [definition]
    assert not list((tmp_path / 'data/tasks').glob('*/*.json'))
    assert (tmp_path / 'data/selection.json').read_bytes() == selection


def test_unknown_candidate_is_not_silently_skipped(tmp_path):
    from egocua.construction import load_construction_tasks

    _owner_workspace(tmp_path)
    with pytest.raises(ConstructionError, match='unknown construction candidates'):
        load_construction_tasks(tmp_path, ['not-a-source-candidate'])
