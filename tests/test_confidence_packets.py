from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

import pytest

from egocua.confidence_packets import (
    CONTINUOUS_VIDEO_REQUIREMENT,
    MATERIALS_HASH_POLICY,
    ConfidencePacketError,
    _continuous_video_ref,
    answer_boundary,
    review_materials_sha256,
    validate_review_packet,
)


def _packet() -> dict:
    materials = {
        "task_id": "blind-task",
        "task_sha256": "0" * 64,
        "track": "graft",
        "site": "egomart",
        "initial": {"instruction": "Find the object.", "observations": []},
        "scheduled_releases": [],
        "public_site": {"path": "public.json", "sha256": "1" * 64},
        "available_actions": ["search", "submit"],
    }
    return {
        "schema_version": 1,
        "materials_policy": "blind_release_only_then_expected_state",
        "materials_hash_policy": MATERIALS_HASH_POLICY,
        "materials_sha256": review_materials_sha256(materials),
        "materials": materials,
    }


def test_answer_free_packet_is_accepted() -> None:
    validate_review_packet(_packet())


@pytest.mark.parametrize("field", ["goal", "initial_acceptable_states", "answer_boundary", "initial_goal", "final_goal"])
def test_hidden_goal_is_rejected(field: str) -> None:
    packet = deepcopy(_packet())
    packet["materials"][field] = {"acceptable_states": []}
    packet["materials_sha256"] = review_materials_sha256(packet["materials"])

    with pytest.raises(ConfidencePacketError, match="hidden keys"):
        validate_review_packet(packet)


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_answer_boundary_ignores_order_but_preserves_quantities_and_fields(track):
    from tests.helpers import task

    definition = task(track)
    state = definition["goal"]["acceptable_states"][0]
    state["selected_entity_ids"] = ["b", "a", "a"]
    definition["goal"]["acceptable_states"].append(deepcopy(state))
    original = answer_boundary(definition)
    state["selected_entity_ids"].reverse()
    definition["goal"]["acceptable_states"].reverse()
    assert answer_boundary(definition) == original
    assert len(original["final_goal"]["acceptable_states"]) == 1
    definition["goal"]["acceptable_states"][0]["selected_entity_ids"].remove("a")
    assert answer_boundary(definition) != original
    definition["goal"]["acceptable_states"][0]["fields"]["note"] = "needed"
    assert answer_boundary(definition) != original


def test_material_hash_must_match() -> None:
    packet = _packet()
    packet["materials"]["initial"]["instruction"] = "Changed after hashing."

    with pytest.raises(ConfidencePacketError, match="hash mismatch"):
        validate_review_packet(packet)


def _located_materials():
    return {
        "task_id": "blind-task",
        "initial": {
            "observations": [
                {
                    "frames": [
                        {
                            "path": "frames/first.jpg",
                            "sha256": "a" * 64,
                            "timestamp_seconds": 1,
                        },
                        {
                            "path": "frames/second.jpg",
                            "sha256": "b" * 64,
                            "timestamp_seconds": 2,
                        },
                    ]
                }
            ]
        },
        "scheduled_releases": [
            {
                "payload": {
                    "frames": [
                        {
                            "path": "frames/later.jpg",
                            "sha256": "c" * 64,
                            "timestamp_seconds": 5,
                        }
                    ]
                }
            }
        ],
        "continuous_source_video": {
            "path": "videos/source.mp4",
            "sha256": "d" * 64,
            "source_interval_seconds": [1, 5],
        },
        "public_site": {"path": "public.json", "content_sha256": "e" * 64},
    }


def test_relocating_identical_evidence_preserves_material_identity():
    materials = _located_materials()
    moved = deepcopy(materials)
    moved["initial"]["observations"][0]["frames"][0]["path"] = "moved/first.jpg"
    moved["scheduled_releases"][0]["payload"]["frames"][0]["path"] = "moved/later.jpg"
    moved["continuous_source_video"]["path"] = "moved/source.mp4"
    moved["public_site"]["path"] = "moved/public.json"

    assert review_materials_sha256(moved) == review_materials_sha256(materials)


@pytest.mark.parametrize("change", ["bytes", "timestamp", "order", "interval"])
def test_material_identity_keeps_deciding_evidence(change):
    materials = _located_materials()
    changed = deepcopy(materials)
    frames = changed["initial"]["observations"][0]["frames"]
    if change == "bytes":
        frames[0]["sha256"] = "f" * 64
    elif change == "timestamp":
        frames[0]["timestamp_seconds"] = 3
    elif change == "order":
        frames.reverse()
    else:
        changed["continuous_source_video"]["source_interval_seconds"] = [2, 5]

    assert review_materials_sha256(changed) != review_materials_sha256(materials)


def test_unhashed_location_is_not_assumed_to_have_identical_contents():
    assert review_materials_sha256({"reference": {"path": "a"}}) != (
        review_materials_sha256({"reference": {"path": "b"}})
    )


def test_continuous_video_is_exposed_only_when_source_requires_it(tmp_path) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video = video_root / "recording.mp4"
    video.write_bytes(b"fixture")
    task = {"id": "blind-task"}
    record: dict[str, Any] = {
        "construct": {"evidence_requirements": [CONTINUOUS_VIDEO_REQUIREMENT]},
        "source": {
            "media_id": "recording",
            "media_start_seconds": 90,
            "media_sha256": "a" * 64,
        },
        "packets": [
            {"id": "V0", "timestamps_seconds": [94, 96]},
            {"id": "V1", "timestamps_seconds": [101, 104]},
        ],
    }

    ref = _continuous_video_ref(
        task,
        record,
        {"source_root": "videos"},
        tmp_path,
    )

    assert ref == {
        "path": "videos/recording.mp4",
        "sha256": "a" * 64,
        "playback_interval_seconds": [4.0, 14.0],
        "source_interval_seconds": [94.0, 104.0],
        "inspection_requirement": "inspect_the_complete_interval_in_order",
    }
    static_record = deepcopy(record)
    static_record["construct"].pop("evidence_requirements")
    assert (
        _continuous_video_ref(
            task,
            static_record,
            {"source_root": "videos"},
            tmp_path,
        )
        is None
    )


def test_available_continuous_video_does_not_require_a_recorded_digest(tmp_path) -> None:
    video = tmp_path / "videos/recording.mp4"
    video.parent.mkdir()
    video.write_bytes(b"fixture")
    reference = _continuous_video_ref(
        {"id": "blind-task"},
        {
            "construct": {"evidence_requirements": [CONTINUOUS_VIDEO_REQUIREMENT]},
            "source": {"media_id": "recording"},
            "packets": [{"id": "V0", "timestamps_seconds": [1, 2]}],
        },
        {"source_root": "videos"},
        tmp_path,
    )

    assert reference["path"] == "videos/recording.mp4"
    assert reference["sha256"] is None
    assert reference["playback_interval_seconds"] == [1.0, 2.0]



@pytest.mark.parametrize("offset", [True, False])
def test_legacy_continuous_source_rejects_boolean_offset(tmp_path, offset):
    video = tmp_path / "videos/recording.mp4"
    video.parent.mkdir()
    video.write_bytes(b"fixture")
    record = {
        "construct": {"evidence_requirements": [CONTINUOUS_VIDEO_REQUIREMENT]},
        "source": {"media_id": "recording", "media_start_seconds": offset},
        "packets": [{"id": "V0", "timestamps_seconds": [2, 3]}],
    }
    with pytest.raises(ConfidencePacketError, match="media_start_seconds.*numeric"):
        _continuous_video_ref({"id": "test"}, record, {"source_root": "videos"}, tmp_path)



def test_legacy_continuous_source_rejects_string_offset(tmp_path):
    video = tmp_path / "videos/recording.mp4"
    video.parent.mkdir()
    video.write_bytes(b"fixture")
    record = {
        "construct": {"evidence_requirements": [CONTINUOUS_VIDEO_REQUIREMENT]},
        "source": {"media_id": "recording", "media_start_seconds": "2.0"},
        "packets": [{"id": "V0", "timestamps_seconds": [3, 4]}],
    }
    with pytest.raises(ConfidencePacketError, match="media_start_seconds.*numeric"):
        _continuous_video_ref({"id": "test"}, record, {"source_root": "videos"}, tmp_path)

def test_required_continuous_video_must_resolve_to_one_real_file(tmp_path) -> None:
    record = {
        "construct": {"evidence_requirements": [CONTINUOUS_VIDEO_REQUIREMENT]},
        "source": {"media_id": "missing", "media_sha256": "a" * 64},
        "packets": [{"id": "V0", "timestamps_seconds": [1, 2]}],
    }

    with pytest.raises(ConfidencePacketError, match="source video root is missing"):
        _continuous_video_ref(
            {"id": "blind-task"},
            record,
            {"source_root": "videos"},
            tmp_path,
        )


def test_packet_builder_preserves_unindexed_historical_packets(monkeypatch, tmp_path) -> None:
    from egocua import confidence_packets as builder

    output = tmp_path / "packets"
    output.mkdir()
    stale = output / "removed-task.json"
    stale.write_text("{}")
    monkeypatch.setattr(builder, "load_construction_tasks", lambda path, task_ids=None: [])
    monkeypatch.setattr(
        builder,
        "load_experiment_plan",
        lambda path: object(),
    )
    plan = tmp_path / "pilot.json"
    plan.write_text("{}")

    report = builder.build_packets(
        tmp_path,
        output_dir="packets",
        plan_path="pilot.json",
    )

    assert report["tasks"] == {}
    assert stale.read_text() == "{}"
    assert (output / "index.json").exists()


def test_packet_builder_writes_one_task_local_public_catalog(
    monkeypatch,
    tmp_path,
) -> None:
    from egocua import confidence_packets as builder

    task: dict[str, Any] = {
        "id": "candidate-media-graft-c-01",
        "site": "egomedia",
        "track": "graft",
        "graft": {"initial_goal": {"acceptable_states": []}},
        "goal": {"acceptable_states": []},
    }
    other = {**task, "id": "candidate-media-graft-c-02"}
    monkeypatch.setattr(builder, "load_construction_tasks", lambda path, task_ids=None: [task, other])
    monkeypatch.setattr(
        builder,
        "load_site_data",
        lambda root, site: {
            "entities": [{"entity_id": "visible"}, {"entity_id": "hidden"}]
        },
    )
    monkeypatch.setattr(
        builder,
        "task_public_site_data",
        lambda current, data: {"entities": [data["entities"][0]]},
    )
    monkeypatch.setattr(builder, "load_experiment_plan", lambda path: object())
    monkeypatch.setattr(builder, "validate_review_packet", lambda packet: None)

    def fake_packet(current, **kwargs):
        return {
            "materials": {"task_sha256": "0" * 64},
            "materials_sha256": "1" * 64,
        }

    monkeypatch.setattr(builder, "build_review_packet", fake_packet)
    (tmp_path / "pilot.json").write_text("{}")

    report = builder.build_packets(
        tmp_path,
        output_dir="packets",
        plan_path="pilot.json",
    )

    public_path = tmp_path / "packets/public-sites/candidate-media-graft-c-01.json"
    assert json.loads(public_path.read_text())["entities"] == [{"entity_id": "visible"}]
    assert not (tmp_path / "packets/public-sites/egomedia.json").exists()
    assert report["scope_policy"] == "task_local_public_catalog"
    untouched_path = tmp_path / "packets/candidate-media-graft-c-02.json"
    untouched_bytes = untouched_path.read_bytes()
    old_plan = report["tasks"][other["id"]]["plan_sha256"]
    (tmp_path / "pilot.json").write_text('{"changed": true}')

    scoped = builder.build_packets(
        tmp_path,
        output_dir="packets",
        plan_path="pilot.json",
        task_ids=[task["id"]],
    )

    assert untouched_path.read_bytes() == untouched_bytes
    assert scoped["tasks"][other["id"]] == report["tasks"][other["id"]]
    assert scoped["tasks"][task["id"]]["plan_sha256"] != old_plan

    def load_after_candidate_withdrawal(path, task_ids=None):
        assert task_ids == [task["id"]]
        return [task]

    monkeypatch.setattr(builder, "load_construction_tasks", load_after_candidate_withdrawal)
    scoped = builder.build_packets(
        tmp_path,
        output_dir="packets",
        plan_path="pilot.json",
        task_ids=[task["id"]],
    )
    assert set(scoped["tasks"]) == {task["id"]}
    assert set(scoped["public_sites"]) == {task["id"]}
    assert scoped["summary"]["tasks"] == 1
    assert untouched_path.read_bytes() == untouched_bytes
    assert (tmp_path / "packets/public-sites" / f"{other['id']}.json").exists()


def test_continuous_repair_selection_requires_two_high_confidence_passes() -> None:
    from tools.audit.apply_continuous_evidence_repairs import continuous_repair_ids

    def assessment(
        decision: str,
        confidence: str,
        *,
        interval: str | None = None,
    ) -> dict[str, Any]:
        verdict: dict[str, Any] = {"decision": decision, "confidence": confidence}
        if interval is not None:
            verdict["counterexample_checks"] = {"continuous_interval": interval}
        return {"verdict": verdict}

    report = {
        "results": [
            {
                "task_id": "repair-me",
                "status": "completed",
                "assessments": [
                    assessment("repair", "high"),
                    assessment("repair", "high", interval="fail"),
                ],
            },
            {
                "task_id": "uncertain",
                "status": "completed",
                "assessments": [
                    assessment("repair", "high"),
                    assessment("repair", "low", interval="fail"),
                ],
            },
            {
                "task_id": "different-problem",
                "status": "completed",
                "assessments": [
                    assessment("repair", "high"),
                    assessment("repair", "high", interval="pass"),
                ],
            },
        ],
    }

    assert continuous_repair_ids(report) == ["repair-me"]


def test_continuous_evidence_repair_does_not_rewrite_compiled_task(tmp_path):
    from egocua.lineage import file_sha256
    from tools.audit.apply_continuous_evidence_repairs import apply
    from tests.helpers import materialize_assets, products, task

    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    task_path = tmp_path / "data/tasks/graft" / f"{definition['id']}.json"
    original = task_path.read_bytes()
    dossier_path = tmp_path / definition["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())
    dossier["tasks"].append({"task_id": "unrelated", "note": "Preserve this record."})
    dossier_path.write_text(json.dumps(dossier))
    video = tmp_path / "reference/fixture/fixture-graft.mp4"
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"same recorded video")
    verdict = {"decision": "repair", "confidence": "high"}
    report = {
        "results": [
            {
                "task_id": definition["id"],
                "status": "completed",
                "assessments": [
                    {"verdict": verdict},
                    {
                        "verdict": {
                            **verdict,
                            "counterexample_checks": {"continuous_interval": "fail"},
                        }
                    },
                ],
            }
        ]
    }
    review_path = tmp_path / "continuous-review.json"
    review_path.write_text(json.dumps(report))

    result = apply(tmp_path, report_path="continuous-review.json", expected_count=1)

    updated = json.loads(dossier_path.read_text())
    assert updated["tasks"][0]["construct"]["evidence_requirements"] == [
        CONTINUOUS_VIDEO_REQUIREMENT
    ]
    assert updated["tasks"][0]["source"]["media_sha256"] == file_sha256(video)
    assert updated["tasks"][1] == dossier["tasks"][1]
    assert task_path.read_bytes() == original
    assert result == {"repairs": 1, "source_manifests": 1, "added_video_hashes": 1}
    assert (
        apply(tmp_path, report_path="continuous-review.json")["added_video_hashes"] == 0
    )
    assert task_path.read_bytes() == original
