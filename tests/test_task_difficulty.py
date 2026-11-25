from __future__ import annotations

import asyncio
import json
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from egocua.checker import check_run
from egocua.decision import DecisionCurve, DecisionState, Observation, UTTERANCE_ROW
from egocua.difficulty import reviewed_difficulty, structural_difficulty
from egocua.environment import load_site_data
from egocua.execution import run_isolated
from egocua.development_gate import DevelopmentEvidence, record_mismatch
from egocua.qualification import (
    failure_diagnostic,
    qualification_family,
    qualification_family_key,
    qualification_outcome,
    trace_has_model_action,
    trace_has_provider_refusal,
)
from egocua.run_ledger import (
    SITE_RUNTIME_CONTRACTS,
    load_run_record,
    task_sha256,
)
from tests.helpers import (
    _write_run_artifact,
    item_id,
    materialize_assets,
    plan_payload,
    products,
    task,
)
from egocua.qualification_report import (
    _failure_surface,
    _should_replace_trial,
    build_report,
)
from tools.run_task_qualification import (
    _blocking_task_ids,
    _select_tasks,
)


def _curve(*, cause: str, final_role: str = "final") -> DecisionCurve:
    return DecisionCurve(
        candidates={
            "initial": (DecisionState(("entity-a",)),),
            "final": (DecisionState(("entity-b",)),),
        },
        observations=(
            Observation("V0:0", ("initial", "final"), None, None),
            Observation(
                UTTERANCE_ROW if cause == "user_utterance" else "V0:1",
                (final_role,),
                cause,
                "The released evidence rules the other candidate out.",
            ),
        ),
        established_by="test",
    )


def _source_with_resolution() -> dict:
    return {
        "construct": {
            "catalog_resolution": {
                "query_rounds": [
                    {
                        "query": "before",
                        "snapshot_path": "before.json",
                        "result_entity_ids": ["a"],
                    },
                    {
                        "query": "after",
                        "snapshot_path": "after.json",
                        "result_entity_ids": ["b"],
                    },
                ],
                "core_neighbor_entity_ids": ["n1", "n2", "n3", "n4"],
                "detail_decision_fields": ["title", "attributes"],
                "rejected": [
                    {
                        "entity_id": "n1",
                        "why_plausible": "Same object category but a different method.",
                        "failure_predicates": ["scene_evidence_mismatch"],
                    },
                    {
                        "entity_id": "n2",
                        "why_plausible": "Same method but a different object.",
                        "failure_predicates": ["late_change_mismatch"],
                    },
                    {
                        "entity_id": "n3",
                        "why_plausible": "Matching product type with a different size.",
                        "failure_predicates": ["website_detail_mismatch"],
                    },
                    {
                        "entity_id": "n4",
                        "why_plausible": "Same brand and category but different material.",
                        "failure_predicates": ["scene_evidence_mismatch"],
                    },
                ],
            }
        }
    }


def test_structural_difficulty_is_a_profile_not_a_hand_written_score() -> None:
    definition = task("graft")
    profile = structural_difficulty(definition, _curve(cause="user_utterance"))

    assert profile.band == "unclassified"
    assert "integrate_late_instruction" in profile.challenges
    assert "score" not in profile.as_dict()


def test_many_challenges_do_not_automatically_assign_a_level() -> None:
    definition = task("graft")
    final_id = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    definition["truth"] = {
        "site_scope_entity_ids": [final_id, "neighbor-a", "neighbor-b"],
    }

    with_neighborhood = structural_difficulty(
        definition,
        _curve(cause="action_order"),
        _source_with_resolution(),
    )
    without_neighborhood = structural_difficulty(
        task("graft"),
        _curve(cause="user_utterance"),
    )

    assert with_neighborhood.band == without_neighborhood.band == "unclassified"
    assert with_neighborhood.selection_key < without_neighborhood.selection_key


def test_indexical_grounding_adds_an_independent_challenge_axis() -> None:
    definition = task("graft")
    direct = structural_difficulty(definition, _curve(cause="scene_appearance"))
    temporal = structural_difficulty(definition, _curve(cause="action_order"))

    assert "temporal_or_wearer_relative_grounding" not in direct.challenges
    assert "temporal_or_wearer_relative_grounding" in temporal.challenges
    assert temporal.selection_key == direct.selection_key


def _reviewed_level(level: str) -> dict[str, Any]:
    conditions = [
        {
            "id": "object",
            "requirement": "Choose the observed object",
            "independent_counterexample": "The similar object fails only this condition",
        }
    ]
    if level != "L1":
        conditions.append(
            {
                "id": "compatibility",
                "requirement": "Keep the compatible accessory",
                "independent_counterexample": "The wrong connector fails only this condition",
            }
        )
    return {
        "difficulty": {
            "level": level,
            "reviewed_by": "test-reviewer",
            "evidence_refs": ["V0:0", "entity:accessory"],
            "simplest_solution": "Identify the object and apply every required condition",
            "no_simpler_solution": "Each independent counterexample remains otherwise acceptable",
            "necessary_conditions": conditions,
            "dependencies": (
                [
                    {
                        "from": "object",
                        "to": "compatibility",
                        "proof": "Changing the object changes which accessory is compatible",
                    }
                ]
                if level == "L3"
                else []
            ),
        }
    }


@pytest.mark.parametrize("level", ["L1", "L2", "L3"])
def test_level_requires_a_reviewed_minimum_solution(level: str) -> None:
    band, evidence = reviewed_difficulty(_reviewed_level(level))
    assert band == level
    assert evidence["method"] == "coding_agent_semantic_review"


@pytest.mark.parametrize(
    "missing",
    [
        "reviewed_by",
        "simplest_solution",
        "no_simpler_solution",
        "evidence_refs",
        "necessary_conditions",
        "dependencies",
    ],
)
def test_incomplete_level_review_stays_unclassified(missing: str) -> None:
    construct = _reviewed_level("L3")
    del construct["difficulty"][missing]
    assert reviewed_difficulty(construct)[0] == "unclassified"


def test_counting_independent_conditions_cannot_prove_dependency() -> None:
    construct = _reviewed_level("L2")
    construct["difficulty"]["level"] = "L3"
    assert reviewed_difficulty(construct)[0] == "unclassified"


def test_missing_dependency_endpoint_does_not_prove_l3() -> None:
    construct = _reviewed_level("L3")
    construct["difficulty"]["dependencies"][0]["to"] = "unknown"
    assert reviewed_difficulty(construct)[0] == "unclassified"


def test_duplicate_condition_is_not_an_independent_requirement() -> None:
    construct = _reviewed_level("L2")
    construct["difficulty"]["necessary_conditions"][1]["id"] = "object"
    with pytest.raises(ValueError, match="unique"):
        reviewed_difficulty(construct)


def test_unknown_level_fails_instead_of_silently_relabeling() -> None:
    construct = _reviewed_level("L1")
    construct["difficulty"]["level"] = "hard"
    with pytest.raises(ValueError, match="Unknown difficulty"):
        reviewed_difficulty(construct)


def test_ordered_source_relation_adds_temporal_challenge_axis() -> None:
    definition = task("graft")
    source_record = {
        "source": {
            "observed_events": [
                {"role": "ordered_pair", "description": "first then later"},
            ],
        },
    }

    profile = structural_difficulty(
        definition,
        _curve(cause="user_utterance"),
        source_record,
    )

    assert "temporal_or_wearer_relative_grounding" in profile.challenges
    assert profile.evidence["temporal_source_relation"] is True


def test_public_non_answer_adds_target_discrimination_axis() -> None:
    definition = task("graft")
    final_id = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    definition["truth"] = {
        "site_scope_entity_ids": [final_id, "near-neighbor"],
    }

    profile = structural_difficulty(definition, _curve(cause="user_utterance"))

    assert "target_discrimination" in profile.challenges
    assert profile.web_execution["public_scope_entity_count"] == 2
    assert profile.web_execution["public_non_answer_count"] == 1
    assert "limited_candidate_neighborhood" in profile.risks


def test_four_core_neighbors_add_compositional_discrimination_burden() -> None:
    definition = task("graft")
    final_id = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    definition["truth"] = {
        "site_scope_entity_ids": [
            final_id,
            "neighbor-a",
            "neighbor-b",
            "neighbor-c",
            "neighbor-d",
        ],
    }

    profile = structural_difficulty(
        definition,
        _curve(cause="user_utterance"),
        _source_with_resolution(),
    )

    assert "compositional_target_discrimination" in profile.challenges
    assert profile.evidence["independent_failure_predicates"] == 3
    assert "website_evidence_integration" in profile.burdens
    assert "limited_candidate_neighborhood" not in profile.risks


def test_search_and_detail_are_one_independent_web_burden() -> None:
    profile = structural_difficulty(
        task("graft"),
        _curve(cause="user_utterance"),
        _source_with_resolution(),
    )

    assert profile.burdens.count("website_evidence_integration") == 1


def test_one_real_predicate_is_counted_without_marking_the_window_incomplete():
    source = _source_with_resolution()
    resolution = source["construct"]["catalog_resolution"]
    for row in resolution["rejected"]:
        row["failure_predicates"] = ["not_drinking_cup"]
    resolution["rejected"].append(
        {"entity_id": "unrelated", "failure_predicates": ["not_a_core_reason"]}
    )
    profile = structural_difficulty(
        task("graft"), _curve(cause="user_utterance"), source
    )
    assert profile.evidence["independent_failure_predicates"] == 1
    assert "limited_candidate_neighborhood" not in profile.risks
    assert "compositional_target_discrimination" not in profile.challenges
    assert "minimum_failure_predicate_density" not in profile.opportunities
    assert profile.band == "unclassified"


def test_refresh_replace_reports_a_guaranteed_native_repair() -> None:
    definition = task("refresh")
    profile = structural_difficulty(
        definition,
        _curve(cause="scene_appearance"),
        _source_with_resolution(),
    )

    assert profile.state_transition["minimum_total_edits"] == 2
    assert "guaranteed_native_state_repair" in profile.challenges


def test_refresh_initial_alternatives_and_missing_history_are_explicit() -> None:
    definition = task("refresh")
    definition["truth"]["initial_acceptable_states"].append(
        {"selected_entity_ids": [item_id("refresh", 2)], "fields": {}}
    )
    profile = structural_difficulty(definition, _curve(cause="scene_appearance"))
    assert profile.state_transition["initial_entities"] == 2
    assert profile.state_transition["minimum_total_edits"] == 2
    definition["truth"].pop("initial_acceptable_states")
    historical = structural_difficulty(definition, _curve(cause="scene_appearance"))
    assert historical.band == "unclassified"
    assert historical.state_transition["initial_entities"] is None
    assert historical.state_transition["minimum_total_edits"] is None
    assert "missing_initial_goal" in historical.risks


def test_rank_only_core_reason_is_an_opportunity_not_a_fake_failure() -> None:
    source = _source_with_resolution()
    source["construct"]["catalog_resolution"]["rejected"][0]["why_plausible"] = (
        "It appears in the ranked result list."
    )

    profile = structural_difficulty(
        task("graft"),
        _curve(cause="user_utterance"),
        source,
    )

    assert "rank_only_core_plausibility" in profile.opportunities
    assert "rank_only_core_plausibility" not in profile.risks


def test_ranked_neighbor_with_named_semantic_overlap_is_not_rank_only() -> None:
    source = _source_with_resolution()
    source["construct"]["catalog_resolution"]["rejected"][0]["why_plausible"] = (
        "A Morton salt canister ranked highly and matches the later container."
    )

    profile = structural_difficulty(
        task("graft"),
        _curve(cause="user_utterance"),
        source,
    )

    assert "rank_only_core_plausibility" not in profile.opportunities


def test_clarify_two_candidate_protocol_is_not_a_limited_catalog_risk() -> None:
    definition = task("clarify")

    profile = structural_difficulty(
        definition,
        _curve(cause="user_utterance"),
        _source_with_resolution(),
    )

    assert "clarification_protocol" in profile.burdens
    assert "limited_candidate_neighborhood" not in profile.risks
    assert "shallow_native_commitment" not in profile.risks


def test_one_model_failure_routes_to_triage_not_rejection() -> None:
    outcome = qualification_outcome(
        {
            "e2e_success": False,
            "run_valid": True,
            "steps": 20,
            "premature_commitment": True,
        }
    )

    assert outcome["status"] == "triage_required"
    assert outcome["next_action"] == "classify_task_defect_vs_valid_model_failure"


def test_successful_trial_still_requires_trajectory_review() -> None:
    outcome = qualification_outcome(
        {"e2e_success": True, "run_valid": True, "steps": 8}
    )

    assert outcome["status"] == "trajectory_review_required"
    assert outcome["next_action"] == "review_development_trajectory"


def test_invalid_run_is_not_mislabeled_as_model_failure() -> None:
    outcome = qualification_outcome({"e2e_success": False, "run_valid": False})

    assert outcome["status"] == "invalid_run"


def test_provider_failure_without_a_model_action_is_not_a_valid_trial() -> None:
    trace = {
        "history": [
            {
                "metadata": {"step_number": 0},
                "model_output": {"action": [{"navigate": {"url": "http://site"}}]},
            },
            {
                "metadata": {"step_number": 1},
                "model_output": None,
                "result": [{"error": "403 insufficient_balance"}],
            },
        ]
    }

    assert trace_has_model_action(trace) is False


def test_post_navigation_model_action_makes_the_trial_screenable() -> None:
    trace = {
        "history": [
            {
                "metadata": {"step_number": 1},
                "model_output": {"action": [{"input": {"index": 3, "text": "knife"}}]},
            }
        ]
    }

    assert trace_has_model_action(trace) is True


def test_provider_refusal_after_model_actions_invalidates_the_trial() -> None:
    trace = {
        "history": [
            {
                "metadata": {"step_number": 1},
                "model_output": {"action": [{"input": {"index": 3, "text": "knife"}}]},
            },
            {
                "metadata": {"step_number": 2},
                "model_output": None,
                "result": [{"error": "403 insufficient_balance"}],
            },
        ]
    }

    assert trace_has_model_action(trace) is True
    assert trace_has_provider_refusal(trace) is True


@pytest.fixture
def development_trial(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    source = tmp_path / "artifacts/runs/human/graft/online/human-1/run.json"
    record = json.loads(source.read_text())
    record["meta"].update(
        agent="browsergym",
        model="test-model",
        model_version="test-model",
        transport="browsergym-openai-vision-v1",
    )
    record["trace"] = {"llm_requests": [{"success": True}]}
    artifact = tmp_path / "artifacts/runs/development/test"
    artifact.mkdir(parents=True)
    snapshot = artifact / "site.json"
    snapshot.write_text(json.dumps(load_site_data(tmp_path, "egomart")))
    archive = artifact / "visual-inputs"
    for packet in definition["visual_packets"]:
        for path in packet["frame_paths"]:
            archived_frame = archive / path
            archived_frame.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp_path / path, archived_frame)
    record["meta"]["runtime"].update(
        site_snapshot=str(snapshot),
        site_contract=SITE_RUNTIME_CONTRACTS[definition["site"]],
        visual_asset_root=str(archive),
    )
    _write_run_artifact(artifact, "online", record, experiment_id="development")
    (tmp_path / "data/reviews/task-qualification.json").write_text(
        json.dumps({"tasks": {}})
    )
    return definition, artifact, load_run_record(artifact)


@pytest.mark.parametrize("model", ["gpt-5.4-mini", "qwen3.5-flash"])
def test_shared_development_protocol_accepts_different_real_models(
    tmp_path: Path, development_trial, model: str
) -> None:
    definition, _, record = development_trial
    record["meta"].update(model=model, model_version=model)

    assert record_mismatch(definition, record, tmp_path) is None


def test_shared_development_protocol_rejects_legacy_dynamic_release_semantics(
    tmp_path: Path, development_trial
) -> None:
    definition, _, record = development_trial
    plan_path = tmp_path / "configs/pilot.json"
    plan = json.loads(plan_path.read_text())
    plan["runtime"]["release_trigger"] = "first_native_commitment"
    plan_path.write_text(json.dumps(plan))
    record["meta"]["runtime"]["release_trigger"] = "first_native_commitment"
    record["meta"]["runtime"].pop("release_semantics", None)

    assert record_mismatch(definition, record, tmp_path) == "legacy_release_semantics"


@pytest.mark.parametrize("review_artifact_matches", [True, False])
@pytest.mark.parametrize("successful", [True, False])
def test_report_binds_trajectory_review_to_the_actual_run(
    tmp_path: Path, development_trial, review_artifact_matches: bool, successful: bool
) -> None:
    definition, artifact, record = development_trial
    artifact_reference = artifact.relative_to(tmp_path).as_posix()
    if not successful:
        record["live_state"]["selected_entity_ids"] = []
        record["final_state"]["selected_entity_ids"] = []
        _write_run_artifact(artifact, "online", record, experiment_id="development")
    verdict = "valid_model_success" if successful else "valid_model_failure"
    review = {
        "reviewer": "coding-agent",
        "task_sha256": "old-source-digest",
        "run_artifact": artifact_reference if review_artifact_matches else "other/run",
        "verdict": verdict,
        "evidence": ["Read native additions and the actual final cart."],
        "mechanism_coverage": {"observed": True},
    }
    (tmp_path / "data/reviews/task-qualification.json").write_text(
        json.dumps({"tasks": {definition["id"]: review}})
    )

    report = build_report(tmp_path, experiment_ids=("development",))
    screen = report["tasks"][definition["id"]]["development_screen"]

    assert screen["status"] == (
        "qualified_after_triage"
        if review_artifact_matches
        else "trajectory_review_required"
        if successful
        else "triage_required"
    ), screen
    if review_artifact_matches:
        assert screen["coding_agent_triage"]["verdict"] == verdict
    assert record["task_snapshot"] == load_run_record(artifact)["task_snapshot"]


def test_shared_trajectory_review_rejects_changed_scoring_evidence(
    tmp_path: Path, development_trial
) -> None:
    definition, _, _ = development_trial
    review = {
        "reviewer": "coding-agent",
        "task_sha256": task_sha256(definition),
        "verdict": "valid_model_success",
        "evidence": ["Evidence for the prior answer."],
        "mechanism_coverage": {"observed": True},
    }
    repaired = deepcopy(definition)
    repaired["goal"]["acceptable_states"] = [{"selected_entity_ids": [], "fields": {}}]
    checked = {"run_valid": True, "e2e_success": False}

    decision = DevelopmentEvidence(tmp_path).trajectory_decision(
        repaired, checked, review
    )

    assert decision.action == "review_development_trial"


@pytest.mark.parametrize(
    ("verdict", "status"),
    [("task_or_site_defect", "repair_required"), ("invalid_run", "invalid_run")],
)
def test_report_does_not_override_shared_trajectory_faults_with_legacy_admission(
    tmp_path: Path,
    development_trial,
    monkeypatch: pytest.MonkeyPatch,
    verdict: str,
    status: str,
) -> None:
    definition, artifact, _ = development_trial
    review = {
        "reviewer": "coding-agent",
        "task_sha256": task_sha256(definition),
        "run_artifact": artifact.relative_to(tmp_path).as_posix(),
        "verdict": verdict,
        "evidence": ["The recorded fault is still unresolved."],
        "mechanism_coverage": {"observed": True},
    }
    (tmp_path / "data/reviews/task-qualification.json").write_text(
        json.dumps({"tasks": {definition["id"]: review}})
    )
    monkeypatch.setattr("egocua.development_gate.grandfathered", lambda *_: True)

    report = build_report(tmp_path, experiment_ids=("development",))

    assert report["tasks"][definition["id"]]["development_screen"]["status"] == status


@pytest.mark.parametrize(
    "retention", ["grandfathered", "retained_development_evidence"]
)
def test_report_keeps_registered_admission_without_a_new_model_score(
    tmp_path: Path, development_trial, monkeypatch: pytest.MonkeyPatch, retention: str
) -> None:
    definition, _, _ = development_trial
    monkeypatch.setattr(f"egocua.development_gate.{retention}", lambda *_: True)

    report = build_report(tmp_path, experiment_ids=("no-new-trial",))
    screen = report["tasks"][definition["id"]]["development_screen"]

    assert screen["status"] == "executable_demonstrated_once"
    assert screen["required"] is False
    assert screen["e2e_success"] is None
    assert screen["evidence_basis"] == "retained_development_admission"
    assert report["valid_as_held_out_difficulty_evidence"] is False


@pytest.mark.parametrize("current_hash", [True, False])
def test_report_projects_shared_defect_decision_without_an_accepted_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, current_hash: bool
) -> None:
    from egocua import qualification_report

    definition = task("graft")
    review = {
        "task_sha256": task_sha256(definition) if current_hash else "previous-key",
        "verdict": "task_or_site_defect",
        "evidence": ["Concrete current defect still needs resolution."],
    }
    path = tmp_path / "data/reviews/task-qualification.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"tasks": {definition["id"]: review}}))
    plan_path = tmp_path / "configs/benchmark.json"
    plan_path.parent.mkdir()
    plan_path.write_text(json.dumps(plan_payload()))
    monkeypatch.setattr(qualification_report, "load_construction_tasks", lambda _, task_ids=None: [definition])
    monkeypatch.setattr(qualification_report, "source_record", lambda *_, **__: {})

    report = build_report(tmp_path)
    screen = report["tasks"][definition["id"]]["development_screen"]

    assert screen["status"] == (
        "repair_required" if current_hash else "evidence_review_required"
    )
    assert screen["e2e_success"] is None
    assert screen["required"] is True
    assert report["summary"]["screened"] == 0


def test_report_scores_an_accepted_run_only_once(
    tmp_path: Path, development_trial, monkeypatch: pytest.MonkeyPatch
) -> None:
    from egocua import development_gate

    _, artifact, _ = development_trial
    archived = (artifact / "run.json").read_bytes()
    original_rescore = development_gate.rescore_development_run
    calls = []

    def count_rescore(definition, record):
        calls.append(record["task_id"])
        return original_rescore(definition, record)

    monkeypatch.setattr("egocua.qualification_report.rescore_development_run", count_rescore)

    report = build_report(tmp_path, experiment_ids=("development",))

    assert len(calls) == report["summary"]["screened"] == 1
    assert len(report["failure_attributions"]) == 1
    assert (artifact / "run.json").read_bytes() == archived


def test_report_default_discovery_and_explicit_diagnostic_scope(
    tmp_path: Path, development_trial
) -> None:
    definition, _, _ = development_trial

    discovered = build_report(tmp_path)
    scoped = build_report(tmp_path, experiment_ids=("no-new-trial",))

    assert discovered["summary"]["screened"] == 1
    assert scoped["summary"]["screened"] == 0
    assert (
        scoped["tasks"][definition["id"]]["development_screen"]["status"] == "not_run"
    )


def test_completed_audit_is_not_overwritten_by_a_new_model_trial() -> None:
    completed = {"status": "qualified_after_triage", "model": "gpt-5.4-mini"}
    unresolved = {"status": "triage_required", "model": "qwen3.5-flash"}

    assert (
        _should_replace_trial(
            "2026-08-27T08:00:00Z",
            completed,
            "2026-08-27T09:00:00Z",
            unresolved,
        )
        is False
    )


def test_completed_audit_replaces_an_unresolved_trial_regardless_of_model() -> None:
    unresolved = {"status": "triage_required", "model": "qwen3.5-flash"}
    completed = {"status": "executable_demonstrated_once", "model": "other-model"}

    assert (
        _should_replace_trial(
            "2026-08-27T09:00:00Z",
            unresolved,
            "2026-08-27T08:00:00Z",
            completed,
        )
        is True
    )


def test_no_question_is_the_primary_clarify_failure_surface() -> None:
    screen = {"status": "triage_required", "asked": False, "false_update": True}

    assert _failure_surface(screen, "clarify") == "clarification_or_grounding"
    assert _failure_surface(screen, "graft") == "overreaction_to_update"


def test_bounded_qualification_blocks_before_the_next_batch() -> None:
    report = {
        "tasks": {
            "task-a": {"development_screen": {"status": "qualified_after_triage"}},
            "task-b": {"development_screen": {"status": "triage_required"}},
            "task-c": {"development_screen": {"status": "repair_required"}},
        }
    }

    assert _blocking_task_ids(report) == ["task-b", "task-c"]
    assert _blocking_task_ids(report, {"task-a"}) == []
    assert _blocking_task_ids(report, {"task-c"}) == ["task-c"]


@pytest.mark.parametrize(
    "status",
    [
        "triage_required",
        "trajectory_review_required",
        "evidence_review_required",
        "repair_required",
    ],
)
def test_inherited_qualification_does_not_block_a_new_batch(status: str) -> None:
    report = {
        "tasks": {
            "legacy": {"development_screen": {"status": status, "required": False}},
            "new": {"development_screen": {"status": status, "required": True}},
        }
    }

    assert _blocking_task_ids(report) == ["new"]
    assert _blocking_task_ids(report, {"legacy"}) == []


@pytest.mark.parametrize("historical_success", [True, False])
def test_qualification_diagnostics_use_the_replacement_answer(
    tmp_path: Path, historical_success: bool
) -> None:
    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    source = tmp_path / "artifacts/runs/human/graft/online/human-1/run.json"
    record = json.loads(source.read_text())
    record["meta"].update(
        agent="browsergym",
        model="test-model",
        model_version="test-model",
        transport="browsergym-openai-vision-v1",
        task_sha256=task_sha256(definition),
    )
    record["trace"] = {"llm_requests": [{"success": True}]}
    if not historical_success:
        record["final_state"]["selected_entity_ids"] = []
        record["live_state"]["selected_entity_ids"] = []
    artifact = tmp_path / "artifacts/runs/development/test"
    artifact.mkdir(parents=True)
    snapshot = artifact / "site.json"
    snapshot.write_text(json.dumps(load_site_data(tmp_path, "egomart")))
    archive = artifact / "visual-inputs"
    for packet in definition["visual_packets"]:
        for path in packet["frame_paths"]:
            archived_frame = archive / path
            archived_frame.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tmp_path / path, archived_frame)
    record["meta"]["runtime"].update(
        site_snapshot=str(snapshot),
        site_contract=SITE_RUNTIME_CONTRACTS[definition["site"]],
        visual_asset_root=str(archive),
    )
    _write_run_artifact(artifact, "online", record)
    archived = (artifact / "run.json").read_bytes()
    definition["goal"]["acceptable_states"] = [
        {"selected_entity_ids": [], "fields": {}}
    ]
    task_path = tmp_path / "data/tasks/graft" / f"{definition['id']}.json"
    task_path.write_text(json.dumps(definition))

    report = build_report(tmp_path, experiment_ids=("test-fixture",))
    assert report["tasks"][definition["id"]]["development_screen"]["e2e_success"] is (
        not historical_success
    )
    attribution = report["failure_attributions"][0]
    assert (attribution["run_verdict"] == "success") is (not historical_success)
    if historical_success:
        diagnostic = report["diagnostics"][definition["id"]]
        assert diagnostic["checker"]["e2e_success"] is False
        assert diagnostic["expected_states"] == definition["goal"]["acceptable_states"]
    else:
        assert definition["id"] not in report["diagnostics"]
    assert (artifact / "run.json").read_bytes() == archived
    assert check_run(definition, load_run_record(artifact))["e2e_success"] is (
        historical_success
    )


def test_unclassified_tasks_are_canaried_one_at_a_time() -> None:
    tasks = [{"id": "task-a"}, {"id": "task-b"}]
    report = {
        "tasks": {
            "task-a": {"development_screen": {"status": "not_run"}},
            "task-b": {"development_screen": {"status": "not_run"}},
        }
    }

    assert _select_tasks(tasks, report, None, concurrency=4) == [tasks[0]]


def test_bounded_qualification_canaries_distinct_unseen_families() -> None:
    tasks = [task("graft"), task("refresh"), task("clarify")]
    for index, definition in enumerate(tasks):
        definition["id"] = f"task-{index}"
    graft_family = qualification_family(tasks[0])
    refresh_family = qualification_family(tasks[1])
    clarify_family = qualification_family(tasks[2])
    report = {
        "tasks": {
            tasks[0]["id"]: {
                "qualification_family": {
                    "key": qualification_family_key(graft_family),
                    **graft_family,
                },
                "development_screen": {"status": "executable_demonstrated_once"},
            },
            tasks[1]["id"]: {
                "qualification_family": {
                    "key": qualification_family_key(refresh_family),
                    **refresh_family,
                },
                "development_screen": {"status": "not_run"},
            },
            tasks[2]["id"]: {
                "qualification_family": {
                    "key": qualification_family_key(clarify_family),
                    **clarify_family,
                },
                "development_screen": {"status": "not_run"},
            },
        }
    }

    assert _select_tasks(tasks, report, None, concurrency=4) == tasks[1:]


def test_bounded_qualification_fills_four_slots_after_family_canaries() -> None:
    tasks = [task("graft") for _ in range(5)]
    for index, definition in enumerate(tasks):
        definition["id"] = f"task-{index}"
    family = qualification_family(tasks[0])
    family_row = {"key": qualification_family_key(family), **family}
    report = {
        "tasks": {
            definition["id"]: {
                "qualification_family": family_row,
                "development_screen": {
                    "status": "executable_demonstrated_once"
                    if index == 0
                    else "not_run"
                },
            }
            for index, definition in enumerate(tasks)
        }
    }

    assert _select_tasks(tasks, report, None, concurrency=4) == tasks[1:5]


def test_unseen_family_canary_shares_slots_with_screened_family_members() -> None:
    tasks = [task("graft"), task("graft"), task("refresh"), task("refresh")]
    for index, definition in enumerate(tasks):
        definition["id"] = f"task-{index}"
    graft = qualification_family(tasks[0])
    refresh = qualification_family(tasks[2])
    rows = {
        "task-0": (graft, "executable_demonstrated_once"),
        "task-1": (graft, "not_run"),
        "task-2": (refresh, "not_run"),
        "task-3": (refresh, "not_run"),
    }
    report = {
        "tasks": {
            task_id: {
                "qualification_family": {"key": qualification_family_key(family), **family},
                "development_screen": {"status": status},
            }
            for task_id, (family, status) in rows.items()
        }
    }

    assert _select_tasks(tasks, report, None, concurrency=4) == [tasks[2], tasks[1]]


def test_explicit_task_override_stays_single_even_with_four_workers() -> None:
    tasks = [{"id": "task-a"}, {"id": "task-b"}]
    report = {
        "tasks": {
            "task-a": {"development_screen": {"status": "not_run"}},
            "task-b": {"development_screen": {"status": "not_run"}},
        }
    }

    assert _select_tasks(tasks, report, "task-b", concurrency=4) == [tasks[1]]


def test_named_qualification_batch_runs_only_its_runnable_tasks() -> None:
    tasks = [{"id": key} for key in ("screened", "unrelated", "task-a", "task-b")]
    report = {
        "tasks": {
            definition["id"]: {
                "qualification_family": {"key": "existing"},
                "development_screen": {
                    "status": "executable_demonstrated_once"
                    if definition["id"] == "screened"
                    else "not_run",
                },
            }
            for definition in tasks
        }
    }

    assert _select_tasks(
        tasks,
        report,
        None,
        concurrency=2,
        requested_task_ids=frozenset({"screened", "task-a", "task-b"}),
    ) == tasks[2:]


def test_empty_repair_scope_never_falls_back_to_all_tasks() -> None:
    tasks = [{"id": "task-a"}, {"id": "task-b"}]
    report = {
        "tasks": {
            "task-a": {"development_screen": {"status": "not_run"}},
            "task-b": {"development_screen": {"status": "not_run"}},
        }
    }

    assert (
        _select_tasks(
            tasks,
            report,
            None,
            concurrency=4,
            requested_task_ids=frozenset(),
        )
        == []
    )


def test_failed_trial_produces_a_compact_task_specific_diagnostic() -> None:
    definition = task("graft")
    family = qualification_family(definition)
    record = {
        "actions": [
            {"type": "search", "step": 1, "query": "product", "results": 3},
            {"type": "select_entity", "step": 2, "entity_id": "wrong"},
        ],
        "final_state": {
            "selected_entity_ids": ["wrong"],
            "fields": {},
            "submission_completed": True,
        },
    }
    checked = {
        "track": "graft",
        "run_valid": True,
        "e2e_success": False,
        "goal_state_correct": False,
        "policy_pass": True,
    }
    entities: list[Mapping[str, Any]] = [
        {"entity_id": item_id("graft", 0), "name": "Expected"},
        {"entity_id": "wrong", "name": "Wrong"},
    ]

    packet = failure_diagnostic(
        definition,
        record,
        checked,
        family={"key": qualification_family_key(family), **family},
        related_task_ids=["sibling"],
        site_entities=entities,
        run_artifact="artifacts/runs/test",
    )

    assert packet["observed_symptom"] == "final_state_mismatch"
    assert packet["actual_final_state"] is None
    assert packet["end_state_source"] == "missing"
    assert packet["related_task_ids"] == ["sibling"]
    assert packet["actions"][0]["query"] == "product"
    assert packet["task_context"]["instructions"] == definition["instructions"]
    assert packet["task_context"]["visual_packets"] == definition["visual_packets"]
    assert packet["trajectory_reference"] == "artifacts/runs/test"


def test_development_trials_freeze_the_formal_online_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from egocua import controlled_experiments
    from egocua.experiment import load_experiment_plan

    monkeypatch.setattr(
        controlled_experiments, "prepare_formal_inputs", lambda tasks, plan, root: tasks
    )
    root = Path(__file__).parents[1]
    formal = load_experiment_plan(root / "configs/benchmark.json")
    definition = task("graft")

    (arm,) = controlled_experiments.load_validation_study(root, [definition])

    assert arm.name == arm.variable == "validation"
    assert arm.plan.conditions == {"online": formal.condition("online")}
    assert (arm.plan.model, arm.plan.max_steps, arm.plan.release_trigger) == (
        formal.model,
        formal.max_steps,
        formal.release_trigger,
    )
    assert arm.plan.replicate_ids == (0,)
    assert controlled_experiments.output_labels([arm])["category"] == "任务试跑"


def test_isolated_trial_cancels_leftover_background_tasks() -> None:
    cleaned_up: list[bool] = []

    async def background() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cleaned_up.append(True)

    async def trial() -> str:
        asyncio.create_task(background())
        await asyncio.sleep(0)
        return "finished"

    assert run_isolated(trial()) == "finished"
    assert cleaned_up == [True]


def test_task_hash_ignores_source_review_and_shared_file_digests() -> None:
    definition = task("graft")
    rebuilt = deepcopy(definition)
    rebuilt["source"]["dossier_sha256"] = "a" * 64
    rebuilt["source"]["frame_provenance_sha256"] = "b" * 64

    assert task_sha256(rebuilt) == task_sha256(definition)

    rebuilt["source"]["record_sha256"] = "c" * 64
    assert task_sha256(rebuilt) == task_sha256(definition)
    rebuilt["instructions"]["a"] = "Choose a different required outcome."
    assert task_sha256(rebuilt) != task_sha256(definition)


def test_qualification_report_loads_each_catalog_once_per_report(tmp_path, monkeypatch):
    from collections import Counter
    from egocua import environment, qualification_report

    loads = []
    monkeypatch.setattr(qualification_report, "load_construction_tasks", lambda _, task_ids=None: [])

    def load_catalog(root, site, **_):
        loads.append(site)
        return {"site": site, "entities": []}

    def screen_records(root, *_):
        assert environment.load_site_data(
            root, "egomedia"
        ) is environment.load_site_data(root, "egomedia")
        return {}, Counter(), {}, {}

    monkeypatch.setattr(environment, "_load_site_data", load_catalog)
    monkeypatch.setattr(qualification_report, "_screen_records", screen_records)

    build_report(tmp_path)
    assert loads == ["egomedia"]
    build_report(tmp_path)
    assert loads == ["egomedia", "egomedia"]


def test_scoped_qualification_scans_only_requested_tasks_and_retains_others(tmp_path, monkeypatch):
    from collections import Counter
    from types import SimpleNamespace
    from egocua import qualification_report as report_module

    first, second = task("graft"), task("refresh")
    monkeypatch.setattr(report_module, "load_construction_tasks", lambda *args: [first, second])
    monkeypatch.setattr(report_module, "source_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(report_module, "DevelopmentEvidence", lambda root: SimpleNamespace(
        inherited_admission=lambda definition: False,
    ))
    scanned = []

    def screen(root, tasks, *args):
        scanned.append(set(tasks))
        return {}, Counter(), {}, {}

    monkeypatch.setattr(report_module, "_screen_records", screen)
    previous_row = {
        "qualification_family": {"key": "existing-family"},
        "development_screen": {
            "status": "qualified_after_triage", "model": "existing-model",
            "run_artifact": "experiments/existing/runs/one",
        },
    }
    path = tmp_path / report_module.REPORT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 2,
        "experiments": ["existing"],
        "tasks": {second["id"]: previous_row},
        "diagnostics": {second["id"]: {"evidence": "keep"}},
        "failure_attributions": [{"task_id": second["id"], "evidence": "keep"}],
    }))
    result = build_report(tmp_path, task_ids=[first["id"]])
    assert scanned == [{first["id"]}]
    assert result["tasks"][second["id"]] == previous_row
    assert result["diagnostics"][second["id"]] == {"evidence": "keep"}
    assert result["failure_attributions"] == [{"task_id": second["id"], "evidence": "keep"}]
    assert result["summary"]["tasks"] == 2
    assert result["summary"]["screened"] == 1
    assert result["summary"]["selected_models"] == {"existing-model": 1}
    assert result["experiments"] == ["existing"]
    assert result["audit_scope"]["task_ids"] == [first["id"]]
