from __future__ import annotations

import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from egocua.schema import load_tasks
from tools.audit.audit_task_acceptance import (
    STATUS_VALUES,
    MediaVerificationCache,
    _extract_frame_sha256,
    _checker_check,
    _construct_check,
    _instruction_check,
    build_acceptance_report,
)

from .helpers import PLAN, materialize_assets, products, site_data, task


@pytest.mark.parametrize("change", ["owner", "fingerprint", "report_version"])
def test_incremental_acceptance_only_checks_requested_tasks(tmp_path, change):
    definitions = [task("graft"), task("refresh")]
    for definition in definitions:
        materialize_assets(definition, products(definition["track"]), tmp_path)
    previous = build_acceptance_report(tmp_path, verify_source_frames=False)
    other_id = definitions[1]["id"]
    if change == "owner":
        dossier = tmp_path / definitions[1]["source"]["dossier"]
        payload = json.loads(dossier.read_text())
        record = next(row for row in payload["tasks"] if row["task_id"] == other_id)
        record["unresolved"] = ["unrelated owner changed"]
        dossier.write_text(json.dumps(payload))
    elif change == "fingerprint":
        previous["tasks"][other_id]["task_sha256"] = "different-record"
        previous["tasks"][other_id]["task_site_execution_sha256"] = (
            "different-site-record"
        )
    else:
        previous["schema_version"] = 999

    result = build_acceptance_report(
        tmp_path,
        verify_source_frames=False,
        task_ids=[definitions[0]["id"]],
        previous_report=previous,
    )

    assert result["audit_scope"]["audited_task_ids"] == [definitions[0]["id"]]
    assert result["tasks"][other_id] == previous["tasks"][other_id]
    assert result["audit_scope"]["preserved_task_count"] == 1


def test_incremental_acceptance_keeps_unrequested_candidate_rows(tmp_path):
    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    previous = build_acceptance_report(tmp_path, verify_source_frames=False)
    candidate = {**previous["tasks"][definition["id"]], "task_id": "staged-candidate"}
    previous["tasks"]["staged-candidate"] = candidate

    result = build_acceptance_report(
        tmp_path,
        verify_source_frames=False,
        task_ids=[definition["id"]],
        previous_report=previous,
    )

    assert result["tasks"]["staged-candidate"] == candidate
    assert result["audit_scope"]["audited_task_ids"] == [definition["id"]]
    assert result["audit_scope"]["preserved_task_count"] == 1


def test_media_verification_cache_reuses_only_unchanged_file_and_toolchain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"first-video-bytes")
    probe_calls = 0

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        nonlocal probe_calls
        if command[1] == "-version":
            return SimpleNamespace(stdout=f"{command[0]} version stable\n")
        probe_calls += 1
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "format": {"duration": "3.0"},
                    "streams": [{"width": 640, "height": 480}],
                }
            )
        )

    monkeypatch.setattr(
        "tools.audit.audit_task_acceptance.subprocess.run",
        fake_run,
    )
    cache_path = tmp_path / "media-cache.json"
    cache = MediaVerificationCache(cache_path)
    first = cache.inspect_video(video)
    cache.save()

    reloaded = MediaVerificationCache(cache_path)
    second = reloaded.inspect_video(video)
    assert probe_calls == 1
    assert second == first

    video.write_bytes(b"changed-video-bytes")
    changed = reloaded.inspect_video(video)
    assert probe_calls == 2
    assert changed["sha256"] != first["sha256"]

    extracted = []

    def extract(video_path, timestamp, jpeg_qscale):
        extracted.append((video_path, timestamp, jpeg_qscale))
        return f"frame-{jpeg_qscale}"

    monkeypatch.setattr(
        "tools.audit.audit_task_acceptance._extract_frame_sha256", extract
    )
    assert reloaded.frame_sha256(video, 1, 2) == "frame-2"
    assert reloaded.frame_sha256(video, 1, None) == "frame-None"
    assert reloaded.frame_sha256(video, 1, 2) == "frame-2"
    assert extracted == [(video, 1, 2), (video, 1, None)]
    reloaded.save()
    assert (
        MediaVerificationCache(cache_path).frame_sha256(video, 1, None) == "frame-None"
    )
    assert len(extracted) == 2


@pytest.mark.parametrize("jpeg_qscale", [2, 5, None])
def test_frame_reextraction_uses_the_recorded_encoder_setting(
    tmp_path, monkeypatch, jpeg_qscale
):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[-1]).write_bytes(b"encoded-frame")

    monkeypatch.setattr("tools.audit.audit_task_acceptance.subprocess.run", run)

    digest = _extract_frame_sha256(tmp_path / "video.mp4", 64, jpeg_qscale)

    assert digest == hashlib.sha256(b"encoded-frame").hexdigest()
    assert commands[0][commands[0].index("-ss") + 1] == "64.000"
    if jpeg_qscale is None:
        assert "-q:v" not in commands[0]
    else:
        assert commands[0][commands[0].index("-q:v") + 1] == str(jpeg_qscale)


@pytest.mark.parametrize("jpeg_qscale", [0, 32, True, "5"])
def test_frame_reextraction_rejects_invalid_encoder_setting(tmp_path, jpeg_qscale):
    with pytest.raises(ValueError, match="jpeg_qscale"):
        _extract_frame_sha256(tmp_path / "video.mp4", 64, jpeg_qscale)


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_production_checker_contract_replays_every_condition_and_rejects_negatives(
    track: str,
) -> None:
    result = _checker_check(task(track), site_data(track), PLAN)

    assert result["status"] == "pass"
    assert result["details"]["positive_paths"] == (
        2
        * len(PLAN.conditions_for(track))
        * len(task(track)["goal"]["acceptable_states"])
        + (1 if track == "clarify" else 0)
    )
    assert result["details"]["adversarial_paths"] >= 15


def test_instruction_check_rejects_internal_language_and_opaque_ids() -> None:
    clean = task("clarify")
    leaking = task("clarify")
    leaking["instructions"]["ambiguous"] = (
        "Use V0 and select p_0123456789ab as the correct answer."
    )

    assert _instruction_check(clean)["status"] == "pass"
    rejected = _instruction_check(leaking)
    assert rejected["status"] == "fail"
    assert rejected["details"]["leakage_matches"]["ambiguous"]


@pytest.mark.parametrize("phrase", ["in this sequence", "my video", "this clip"])
def test_instruction_wording_flags_do_not_decide_semantic_validity(phrase: str) -> None:
    definition = task("graft")
    definition["instructions"]["a"] = f"Find a guide for the part I use {phrase}."

    result = _instruction_check(definition)

    assert result["status"] == "pass"
    assert result["details"]["wording_flags"]["a"] == [phrase]


def test_instruction_check_rejects_scene_equivalence_as_procedure_motivation() -> None:
    benchmark_task = task("graft")
    benchmark_task["instructions"]["a"] = "Save a beginner tutorial."
    benchmark_task["instructions"]["b"] = (
        "Make it the one that matches how I do this from start to finish."
    )
    source_record = {"construct": {"evidence_type": "procedure_stage"}}

    rejected = _instruction_check(benchmark_task, source_record)

    assert rejected["status"] == "fail"
    assert rejected["details"]["utility_gate"]["benchmark_equivalence_only"] is True

    benchmark_task["instructions"]["b"] = (
        "The joint keeps slipping, so choose one that explains this step clearly."
    )
    accepted = _instruction_check(benchmark_task, source_record)
    assert accepted["status"] == "pass"


def test_construct_check_flags_static_replacement_without_claiming_human_pass() -> None:
    benchmark_task = task("clarify")
    record = {
        "source": {"original_annotations": ["The wearer handles two objects."]},
        "supported_claims": [{"text": "Two objects remain visible."}],
        "construct": {
            "research_question": "Should the Agent ask before choosing?",
            "divergence_source": "grounding",
            "event_origin": "annotation_derived",
            "belief_before": "Either candidate is possible.",
            "belief_after": "One candidate is intended.",
            "required_behavior_change": "Ask and then select.",
            "no_update_failure": "Guessing.",
            "overupdate_failure": "Asking unnecessarily.",
            "visual_dependency_claim": "The candidates come from the scene.",
            "task_blueprint": {"candidate_roles": ["a", "b"]},
        },
    }

    result = _construct_check(benchmark_task, record, is_primary=True)

    assert result["status"] == "pass"
    assert result["details"]["human_review_recommended"] is True
    assert result["details"]["construct_risk_reasons"]
    assert "diagnostic, not a defect" in result["details"]["admission_test"]


def test_sequence_clarification_is_not_reduced_to_one_static_photo() -> None:
    benchmark_task = task("clarify")
    benchmark_task["clarification"]["valid_slot"] = "sequence"
    record = {
        "source": {
            "original_annotations": ["The wearer examines two posters in order."]
        },
        "supported_claims": [{"text": "The first and second posters are distinct."}],
        "construct": {
            "research_question": "Should the Agent ask which occurrence?",
            "divergence_source": "grounding",
            "event_origin": "annotation_derived",
            "belief_before": "Either occurrence is possible.",
            "belief_after": "The second occurrence is intended.",
            "required_behavior_change": "Ask and then select the second.",
            "no_update_failure": "Guessing.",
            "overupdate_failure": "Asking unnecessarily.",
            "visual_dependency_claim": "The ordered scene establishes the occurrence.",
            "task_blueprint": {"candidate_roles": ["a", "b"]},
        },
    }

    result = _construct_check(benchmark_task, record, is_primary=True)

    assert result["status"] == "pass"
    assert result["details"]["construct_risk_reasons"] == []


def test_full_report_has_one_fail_closed_row_per_constructed_task(
    tmp_path: Path,
) -> None:
    for track in ("graft", "refresh", "clarify"):
        materialize_assets(task(track), products(track), tmp_path)
    expected = {definition["id"] for definition in load_tasks(tmp_path / "data/tasks")}

    report = build_acceptance_report(tmp_path, verify_source_frames=False)

    assert report["summary"]["tracks"] == {"graft": 1, "refresh": 1, "clarify": 1}
    assert set(report["tasks"]) == expected
    assert report["summary"]["tasks"] == len(expected)
    assert sum(report["summary"]["statuses"].values()) == len(expected)
    assert all(row["status"] in STATUS_VALUES for row in report["tasks"].values())
    assert all(
        "task_acceptance_ready" not in row and "benchmark_release_ready" not in row
        for row in report["tasks"].values()
    )
    assert all(
        set(row["checks"])
        == {
            "schema_and_frames",
            "source_lineage",
            "glass_use_construct_prerequisites",
            "continuous_video",
            "site_entities_images_and_answers",
            "compiled_from_source_truth",
            "production_checker",
            "instruction_form_and_leakage",
            "text_pinning",
            "answer_closure",
            "necessity",
        }
        for row in report["tasks"].values()
    )


def _catalog_task(a: str, b: str, accepted: list[str]) -> dict:
    return {
        "id": "pinning-fixture",
        "track": "graft",
        "site": "fixture",
        "instructions": {"a": a, "b": b},
        "goal": {"acceptable_states": [{"selected_entity_ids": accepted}]},
    }


@pytest.fixture
def plain_entity_text(monkeypatch: pytest.MonkeyPatch) -> None:
    import egocua.answer_closure as closure

    monkeypatch.setattr(closure, "entity_text", lambda _site, entity: entity["text"])


def test_text_pinning_warns_when_each_target_is_pinned_by_its_own_words(
    plain_entity_text: None,
) -> None:
    from tools.audit.audit_task_acceptance import _text_pinning_check

    catalog = {
        "entities": [
            {"entity_id": "strainer", "text": "steel strainer 15cm fine mesh"},
            {"entity_id": "corn", "text": "white whole kernel corn 15.25 ounce"},
            {"entity_id": "coarse", "text": "steel strainer 20cm coarse"},
            {"entity_id": "yellow", "text": "yellow cream corn 15.25 ounce"},
        ]
    }
    task = _catalog_task(
        "Add a 15-centimeter fine-mesh version of the strainer.",
        "Also add one 15¼-ounce can of white whole-kernel corn.",
        ["strainer", "corn"],
    )

    result = _text_pinning_check(task, catalog)

    assert result["status"] == "warn"
    assert "errors" not in result
    assert result["details"]["pinned"] == ["corn", "strainer"]
    assert "15cm" in result["details"]["targets"]["strainer"]["words"]


def test_text_pinning_passes_when_the_scene_must_pick_among_matches(
    plain_entity_text: None,
) -> None:
    from tools.audit.audit_task_acceptance import _text_pinning_check

    catalog = {
        "entities": [
            {"entity_id": "bar-a", "text": "chocolate snack bar"},
            {"entity_id": "bar-b", "text": "lemon snack bar"},
            {"entity_id": "bar-c", "text": "peanut snack bar"},
        ]
    }
    task = _catalog_task(
        "Add the snack bar I reached for first.",
        "Also add the snack bar I reached for next.",
        ["bar-a", "bar-b"],
    )

    assert _text_pinning_check(task, catalog)["status"] == "pass"


def test_answer_closure_warns_on_an_undecided_window_listing_with_the_same_words(
    plain_entity_text: None,
) -> None:
    from tools.audit.audit_task_acceptance import _answer_closure_check

    catalog = {
        "entities": [
            {"entity_id": "pan-a", "text": "black frying pan 8 inch"},
            {"entity_id": "pan-b", "text": "black nonstick frying pan 20 cm"},
            {"entity_id": "pan-c", "text": "black frying pan 28 cm"},
            {"entity_id": "pan-d", "text": "red frying pan 20 cm"},
        ]
    }
    task = {
        "id": "closure-fixture",
        "track": "clarify",
        "site": "fixture",
        "instructions": {"ambiguous": "Find the black frying pan I'm holding."},
        "clarification": {"canonical_answer": "The one about 20 centimeters across."},
        "goal": {"acceptable_states": [{"selected_entity_ids": ["pan-a"]}]},
    }
    record: dict[str, Any] = {
        "construct": {
            "catalog_resolution": {
                "window_entity_ids": ["pan-a", "pan-b", "pan-c", "pan-d"],
                "rejected": [],
            }
        }
    }

    result = _answer_closure_check(task, catalog, record)

    assert result["status"] == "warn"
    assert "errors" not in result
    assert list(result["details"]["undecided"]) == ["pan-b"]

    record["construct"]["catalog_resolution"]["rejected"] = [
        {"entity_id": "pan-b", "reason": "public photo shows a wooden handle"}
    ]
    assert _answer_closure_check(task, catalog, record)["status"] == "pass"


def test_one_released_word_never_pins_or_opens_closure(plain_entity_text: None) -> None:
    from tools.audit.audit_task_acceptance import (
        _answer_closure_check,
        _text_pinning_check,
    )

    catalog = {
        "entities": [
            {"entity_id": "vase-a", "text": "silver vase"},
            {"entity_id": "vase-b", "text": "glass vase"},
        ]
    }
    task = _catalog_task("Add the vase I hold.", "Keep it.", ["vase-a"])
    record = {
        "construct": {"catalog_resolution": {"window_entity_ids": ["vase-a", "vase-b"]}}
    }

    assert _text_pinning_check(task, catalog)["status"] == "pass"
    assert _answer_closure_check(task, catalog, record)["status"] == "pass"


def test_unresolved_answer_boundary_is_a_semantic_warning(
    plain_entity_text: None,
) -> None:
    from tools.audit.audit_task_acceptance import _answer_closure_check

    task = _catalog_task("Add the vase I hold.", "Keep it.", ["vase-a"])
    record = {
        "construct": {
            "catalog_resolution": {
                "window_entity_ids": ["vase-a"],
                "answer_boundary_review": {
                    "status": "hold_answer_closure_unresolved",
                    "unresolved_candidates": [{"entity_id": "vase-b"}],
                },
            }
        }
    }

    result = _answer_closure_check(task, {"entities": []}, record)

    assert result["status"] == "warn"
    assert "errors" not in result
    assert result["details"]["unresolved_candidates"] == [{"entity_id": "vase-b"}]
