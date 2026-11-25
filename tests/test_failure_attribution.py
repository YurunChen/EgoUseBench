from __future__ import annotations

from pathlib import Path
from typing import Any
from copy import deepcopy

import pytest

from egocua.checker import EVALUATOR_VERSION, check_run, rescore_run
from egocua.failure_attribution import (
    PULSE_STAGES,
    _released_visual_packets,
    historical_review_context,
    attribute_run,
)
from egocua.judge_validation import select_reliability_sample
from egocua.failure_report import build_failure_report
from egocua.run_ledger import RunLedger, load_run_record, task_sha256
from tests.conditions import CONDITIONS
from tests.helpers import (
    complete_run_meta,
    complete_run_record,
    release_ready_task,
    task,
)
from tools.audit.report_failure_attribution import _benchmark_source


COUNTERFACTUAL_POLICY_VERSION = "artifact-boundary-intervention-v3"


def _checked(**overrides: object) -> dict[str, object]:
    return {
        "task_id": "task-1",
        "run_id": "run-1",
        "model": "model-a",
        "track": "graft",
        "site": "egomart",
        "evaluator": {
            "version": "3.0",
            "acceptable_states": [{"selected_entity_ids": ["target"], "fields": {}}],
        },
        "run_valid": True,
        "e2e_success": False,
        "goal_state_correct": False,
        "submission_completed": True,
        "target_opened": False,
        "target_selected": False,
        "steps": 12,
        **overrides,
    }


def _record(*actions: dict[str, object], **meta: object) -> dict[str, object]:
    return {"actions": list(actions), "steps": 12, "meta": dict(meta)}


def _source_row(task: dict[str, Any]) -> dict[str, Any]:
    return attribute_run(
        _checked(task_id=task["id"]),
        _record(
            run_id="run-1",
            model="model-a",
            task_sha256=task_sha256(task),
            site_data_sha256="site-v1",
        ),
    )


def _write_probe_run(
    root: Path,
    task: dict[str, Any],
    *,
    pair_id: str,
    stage: str,
    intervention_kind: str,
    intervention_step: int | None,
    success: bool,
    index: int,
    completed: bool | None = None,
) -> str:
    root = root.resolve()
    ledger = RunLedger.start(
        root / "artifacts",
        experiment_id=f"pulse-{stage.lower()}",
        task_id=task["id"],
        condition=CONDITIONS["online"],
        replicate_id=index,
        agent="test",
    )
    target = task["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    completed = success if completed is None else completed
    selected = [target] if success else []
    actions: list[dict[str, Any]] = [
        {
            "type": "instruction_injection",
            "step": 2,
            "trigger_reason": "task_timeline",
        }
    ]
    if stage != "control":
        actions.append(
            {
                "type": "diagnostic_intervention",
                "step": intervention_step,
                "stage": stage,
                "intervention_kind": intervention_kind,
                "source_run_id": "run-1",
                "pair_id": pair_id,
            }
        )
    if success:
        actions.append({"type": "select", "step": 5, "item_id": target})
    meta = complete_run_meta(
        agent="test",
        model="model-a",
        model_version="v1",
        provider_seed=index,
        replicate_id=index,
        task_sha256=task_sha256(task),
        site_data_sha256="site-v1",
        run_id=ledger.run_id,
        artifact_directory=ledger.directory.as_posix(),
        diagnostic_intervention={
            "policy_version": COUNTERFACTUAL_POLICY_VERSION,
            "source_run_id": "run-1",
            "pair_id": pair_id,
            "stage": stage,
            "intervention_kind": intervention_kind,
            "intervention_step": intervention_step,
        },
    )
    ledger.complete(
        {
            "task_id": task["id"],
            "track": task["track"],
            "condition": CONDITIONS["online"].as_dict(),
            "run_status": {"valid": True, "reason": None},
            "initial_state": {
                "selected_entity_ids": [],
                "fields": {},
                "submission_completed": False,
            },
            "steps": 5,
            "actions": actions,
            "final_state": {
                "selected_entity_ids": selected if completed else [],
                "fields": {},
                "submission_completed": completed,
            },
            "live_state": {"selected_entity_ids": selected, "fields": {}},
            "task_snapshot": task,
            "trace": {
                "history": [
                    {
                        "metadata": {"step_number": 6},
                        "model_output": {"action": [{"done": {"success": True}}]},
                        "result": [{"is_done": True}],
                        "state_message": "<user_request>Save this</user_request><agent_history>private summary</agent_history><browser_state>Saved items</browser_state>",
                    }
                ]
            },
            "meta": meta,
        }
    )
    return ledger.directory.relative_to(root).as_posix()


def _trajectory_workspace(root: Path, stage: str = "L") -> tuple[dict, dict]:
    task = release_ready_task("graft")
    reference = _write_probe_run(
        root,
        task,
        pair_id="trace",
        stage="control",
        intervention_kind="none",
        intervention_step=None,
        success=False,
        index=0,
    )
    record = load_run_record(root / reference)
    row = attribute_run(
        _checked(run_id=record["meta"]["run_id"], task_id=task["id"]),
        record,
    )
    row["run_artifact"] = reference
    mechanisms = {
        "P": "object_grounding",
        "U": "update_integration",
        "L": "candidate_discrimination",
        "S": "state_repair",
        "E": "completion_judgment",
    }
    assessment = {
        "policy_version": "pulse-v6",
        "checker_version": "3.0",
        "run_id": row["run_id"],
        "reviewer_kind": "coding_agent",
        "candidate_stage": stage,
        "candidate_mechanism": mechanisms[stage],
        "candidate_step": 2,
        "candidate_confidence": "high",
        "candidate_unknown_boundary": None,
        "observable_facts": ["task update released at step 2"],
        "trajectory_review": {
            "reason": "Synthetic fixture: classify the recorded failure.",
            "earlier_stages": "Synthetic fixture: earlier stages were checked.",
            "failure_effect": "Synthetic fixture: this error prevented completion.",
            "citations": [{"action_index": 0, "action": record["actions"][0]}],
        },
    }
    return {"failure_attributions": [row]}, {"assessments": [assessment]}


def test_pulse_has_one_stable_five_stage_vocabulary() -> None:
    assert PULSE_STAGES == ("P", "U", "L", "S", "E")


@pytest.mark.parametrize("reason", ["stage_boundary", "evidence_gap"])
def test_unknown_review_preserves_specific_uncertainty(
    tmp_path: Path, reason: str
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    assessment = evidence["assessments"][0]
    assessment.update(
        candidate_stage=None,
        candidate_mechanism=None,
        candidate_step=None,
        candidate_confidence="low",
        candidate_unknown_boundary="P/L",
    )
    assessment["trajectory_review"]["uncertainty_reason"] = reason

    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)

    row = report["attributions"][0]
    assert row["pulse_stage"] is None
    assert row["rule_evidence"]["uncertainty_reason"] == reason
    assert report["statistics"]["summary"]["unknown_reasons"] == {reason: 1}
    assert row["review_required"] is (reason == "stage_boundary")
    assert bool(report["human_review"]["packets"]) is (reason == "stage_boundary")
    assert report["workflow"]["processing_complete"]
    assert not report["workflow"]["resolved_complete"]


@pytest.mark.parametrize("reason", ["inferred_thought", "stage_boundary"])
def test_review_rejects_invalid_or_resolved_uncertainty(
    tmp_path: Path, reason: str
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    evidence["assessments"][0]["trajectory_review"]["uncertainty_reason"] = reason

    with pytest.raises(ValueError, match="uncertainty_reason"):
        build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)


def test_attribution_uses_existing_shared_review_registry() -> None:
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH

    assert FAILURE_EVIDENCE_PATH == Path("data/reviews/failure-attribution.json")


def test_old_unknown_review_does_not_require_reopening(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    evidence["assessments"][0].update(
        candidate_stage=None,
        candidate_mechanism=None,
        candidate_step=None,
        candidate_confidence="low",
        candidate_unknown_boundary="P/L",
    )

    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)

    assert report["workflow"]["processing_complete"]
    assert report["statistics"]["summary"]["unknown_reasons"] == {"unrecorded": 1}


@pytest.mark.parametrize("verdict", ["success", "task_or_site_defect"])
def test_result_adjudication_precedes_historical_failure_label(
    tmp_path: Path, verdict: str
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    previous = deepcopy(evidence)
    old = source["failure_attributions"][0]
    record = load_run_record(tmp_path / old["run_artifact"])
    checked = _checked(
        run_id=old["run_id"],
        task_id=old["task_id"],
        e2e_success=verdict == "success",
    )
    row = attribute_run(
        checked,
        record,
        qualification={"verdict": verdict} if verdict != "success" else None,
    )
    row["run_artifact"] = old["run_artifact"]

    report = build_failure_report(
        {"failure_attributions": [row]}, evidence, root=tmp_path, bootstrap_samples=0
    )

    assert evidence == previous
    assert report["attributions"][0]["pulse_stage"] is None
    assert report["attributions"][0]["attribution_status"] == "excluded"
    assert report["statistics"]["summary"]["valid_model_failures"] == 0
    assert (
        report["statistics"]["summary"][
            "successes" if verdict == "success" else "unresolved_run_validity"
        ]
        == 1
    )
    if verdict == "task_or_site_defect":
        assert not report["workflow"]["resolved_complete"]
        assert report["workflow"]["qualification_pending_run_ids"] == [row["run_id"]]
        assert "task_validity_pending" in report["attributions"][0]["observable_facts"]
        assert [packet["run_id"] for packet in report["human_review"]["packets"]] == [row["run_id"]]
        assert report["human_review"]["reliability_sample_run_ids"] == []


def test_invalid_run_and_success_are_excluded_before_pulse() -> None:
    invalid = attribute_run(_checked(run_valid=False, e2e_success=None), _record())
    success = attribute_run(_checked(e2e_success=True), _record())

    assert invalid["run_verdict"] == "invalid_run"
    assert success["run_verdict"] == "success"
    assert invalid["pulse_stage"] is None
    assert success["pulse_stage"] is None


def test_benchmark_fault_requires_a_registered_certificate() -> None:
    result = attribute_run(
        _checked(),
        _record(),
        qualification={
            "verdict": "benchmark_fault",
            "policy_version": "benchmark-fault-certificate-v1",
            "predicate": "checker_replay_mismatch",
            "facts": {
                "input_sha256": "abc123",
                "recorded_verdict": "fail",
                "replayed_verdict": "pass",
            },
            "evidence_refs": ["artifacts/checker-replay.json"],
        },
    )

    assert result["run_verdict"] == "benchmark_fault"
    assert result["qualification_evidence"]["predicate"] == "checker_replay_mismatch"


def test_uncertified_benchmark_fault_opinion_is_not_a_model_label() -> None:
    result = attribute_run(_checked(), _record(), qualification="task_or_site_defect")

    assert result["run_verdict"] == "unknown"
    assert result["pulse_stage"] is None
    report = build_failure_report(
        {"failure_attributions": [result]}, bootstrap_samples=0
    )
    assert report["statistics"]["summary"]["unresolved_run_validity"] == 1
    assert report["statistics"]["summary"]["valid_model_failures"] == 0


def test_checker_symptoms_do_not_infer_failure_categories() -> None:
    result = attribute_run(
        _checked(track="clarify", asked=False),
        _record({"type": "search", "step": 1}),
    )

    assert result["pulse_stage"] is None
    assert result["candidate_stage"] is None
    assert result["attribution_status"] == "coding_agent_pending"


def test_correct_state_without_submission_is_not_a_failure() -> None:
    result = attribute_run(
        _checked(e2e_success=True, goal_state_correct=True, submission_completed=False),
        _record({"type": "select_entity", "step": 7, "entity_id": "target"}),
    )

    assert result["pulse_stage"] is None
    assert result["candidate_stage"] is None
    assert result["attribution_status"] == "excluded"


def test_unknown_unsubmitted_state_does_not_invent_a_state_failure_candidate() -> None:
    definition = task("refresh")
    target = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": "refresh",
            "steps": 3,
            "actions": [
                {
                    "type": "visual_context_update",
                    "step": 2,
                    "trigger_reason": "task_timeline",
                },
                {"type": "select_entity", "step": 3, "entity_id": target},
            ],
        }
    )
    run.pop("live_state")

    result = attribute_run(check_run(definition, run), run)

    assert result["candidate_stage"] is None
    assert result["candidate_mechanism"] is None
    assert result["run_verdict"] == "invalid_run"
    assert result["secondary_symptoms"] == []


def test_report_keeps_coding_agent_candidate_out_of_formal_labels() -> None:
    source = {"failure_attributions": [attribute_run(_checked(), _record())]}
    evidence = {
        "assessments": [
            {
                "policy_version": "pulse-v6",
                "checker_version": "3.0",
                "run_id": "run-1",
                "reviewer_kind": "coding_agent",
                "candidate_stage": "L",
                "candidate_mechanism": "candidate_discrimination",
                "candidate_step": 3,
                "observable_facts": ["wrong candidate opened"],
                "candidate_confidence": "low",
                "candidate_unknown_boundary": "L/S",
            }
        ]
    }

    report = build_failure_report(source, evidence, bootstrap_samples=20)
    row = report["attributions"][0]

    assert row["candidate_stage"] == "L"
    assert row["candidate_unknown_boundary"] == "L/S"
    assert row["pulse_stage"] is None
    assert report["workflow"]["coding_agent_pending_run_ids"] == ["run-1"]


@pytest.mark.parametrize("stage", PULSE_STAGES)
def test_coding_agent_trace_review_resolves_without_rerun(
    tmp_path: Path, stage: str
) -> None:
    source, evidence = _trajectory_workspace(tmp_path, stage)
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=20)
    row = report["attributions"][0]
    assert (row["pulse_stage"], row["attribution_method"]) == (
        stage,
        "trajectory_review",
    )


@pytest.mark.parametrize("change", ["citation", "step", "reason", "run_id"])
def test_trace_review_rejects_bad_evidence(tmp_path: Path, change: str) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    assessment = evidence["assessments"][0]
    if change == "citation":
        assessment["trajectory_review"]["citations"][0]["action"] = {}
    elif change == "step":
        assessment["candidate_step"] = 100
    elif change == "reason":
        assessment["trajectory_review"]["reason"] = ""
    else:
        source["failure_attributions"][0]["run_id"] = "different"
        assessment["run_id"] = "different"
    with pytest.raises(ValueError):
        build_failure_report(source, evidence, root=tmp_path)


def test_unknown_trace_stays_unknown(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    assessment = evidence["assessments"][0]
    assessment.update(
        candidate_stage=None, candidate_mechanism=None, candidate_confidence="low"
    )
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=20)
    assert report["attributions"][0]["attribution_status"] == "evidence_insufficient"


def _browser_review(root: Path) -> tuple[dict, dict]:
    source, evidence = _trajectory_workspace(root, "E")
    assessment = evidence["assessments"][0]
    assessment["candidate_step"] = 6
    assessment["trajectory_review"]["citations"] = []
    assessment["trajectory_review"]["trace_citations"] = [
        {
            "history_index": 0,
            "step": 6,
            "actions": [{"done": {"success": True}}],
            "result": [{"is_done": True}],
        }
    ]
    return source, evidence


def test_browser_done_is_citable_without_a_native_action(tmp_path: Path) -> None:
    source, evidence = _browser_review(tmp_path)
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    row = report["attributions"][0]
    assert (row["pulse_stage"], row["first_decisive_step"]) == ("E", 6)
    rebuilt = build_failure_report(
        {"failure_attributions": report["attributions"]},
        root=tmp_path,
        bootstrap_samples=0,
    )
    assert rebuilt["attributions"] == report["attributions"]


def test_benchmark_source_uses_only_the_requested_experiment(tmp_path: Path) -> None:
    original, evidence = _trajectory_workspace(tmp_path)
    row = original["failure_attributions"][0]
    evidence["assessments"].append({"run_id": "historical-run"})
    summary = {
        "tasks": {
            row["task_id"]: {
                "artifact": row["run_artifact"],
                "checked": _checked(task_id=row["task_id"], run_id=row["run_id"]),
            }
        }
    }
    source, scoped = _benchmark_source(summary, evidence, tmp_path)
    assert len(scoped["assessments"]) == 1
    report = build_failure_report(source, scoped, root=tmp_path, bootstrap_samples=0)
    assert report["attributions"][0]["pulse_stage"] == "L"
    summary["tasks"][row["task_id"]]["checked"]["run_id"] = "another-run"
    with pytest.raises(ValueError, match="summary does not match"):
        _benchmark_source(summary, evidence, tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("history_index", -1),
        ("history_index", True),
        ("history_index", 1),
        ("step", 5),
        ("actions", []),
        ("result", []),
        ("extra", "unsupported"),
    ],
)
def test_browser_citation_rejects_changed_evidence(
    tmp_path: Path, field: str, value: Any
) -> None:
    source, evidence = _browser_review(tmp_path)
    evidence["assessments"][0]["trajectory_review"]["trace_citations"][0][field] = value
    with pytest.raises(ValueError, match="trace citation"):
        build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)


def test_diagnostic_reruns_cannot_assign_pulse_labels() -> None:
    with pytest.raises(ValueError, match="diagnostic reruns are separate"):
        build_failure_report(
            {"failure_attributions": []}, {"counterfactuals": {"r": {}}}
        )


def test_legacy_diagnostic_history_does_not_assign_labels() -> None:
    row = attribute_run(_checked(), _record())
    report = build_failure_report(
        {"failure_attributions": [row]},
        {"legacy_counterfactuals": {"run-1": {"pulse_stage": "P"}}},
        bootstrap_samples=0,
    )
    assert report["attributions"][0]["attribution_status"] == "coding_agent_pending"


def test_only_final_report_owns_the_pulse_matrices() -> None:
    rows = [
        attribute_run(_checked(run_id="r1", e2e_success=True), _record()),
        attribute_run(
            _checked(
                run_id="r2",
                goal_state_correct=True,
                submission_completed=False,
            ),
            _record(),
        ),
        attribute_run(
            _checked(run_id="r3", model="model-b", site="egomedia"),
            _record(),
        ),
    ]

    report = build_failure_report({"failure_attributions": rows}, bootstrap_samples=20)

    assert set(report["statistics"]["matrices"]) == {"model", "track", "site", "setting", "model_setting"}
    assert report["statistics"]["summary"]["valid_model_failures"] == 2
    assert report["statistics"]["summary"]["pending_attributions"] == 2


def test_human_agreement_audits_the_trace_label(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path, "E")
    run_id = source["failure_attributions"][0]["run_id"]
    evidence["ratings"] = [
        {"run_id": run_id, "rater_id": "a", "pulse_stage": "P"},
        {"run_id": run_id, "rater_id": "b", "pulse_stage": "P"},
    ]
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=20)
    assert report["attributions"][0]["pulse_stage"] == "E"
    assert report["reliability"]["automatic_vs_consensus_agreement"] == 0.0


def test_review_packet_does_not_reveal_automatic_label(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)

    report = build_failure_report(source, evidence, root=tmp_path)
    packet = report["human_review"]["packets"][0]

    assert "pulse_stage" not in packet
    assert "candidate_stage" not in packet
    assert "first_decisive_step" not in packet
    assert "rule_evidence" not in packet
    assert "observable_facts" not in packet


@pytest.mark.parametrize("boundary", ["P/L", "U/S"])
def test_ambiguous_boundaries_preserve_review_for_humans(
    tmp_path: Path, boundary: str
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    assessment = evidence["assessments"][0]
    assessment.update(candidate_confidence="low", candidate_unknown_boundary=boundary)
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    row = report["attributions"][0]
    assert row["pulse_stage"] is None
    assert row["rule_evidence"]["reason"] == assessment["trajectory_review"]["reason"]
    assert row["review_required"] is True


def test_unknown_without_decisive_step_is_supported(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    assessment = evidence["assessments"][0]
    assessment.update(
        candidate_stage=None,
        candidate_mechanism=None,
        candidate_confidence="low",
        candidate_step=None,
    )
    assessment["trajectory_review"]["citations"] = []
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    assert report["attributions"][0]["attribution_status"] == "evidence_insufficient"


@pytest.mark.parametrize("field", ["task_id", "track", "model"])
def test_review_cannot_be_attached_to_another_context(
    tmp_path: Path, field: str
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    source["failure_attributions"][0][field] = "another"
    with pytest.raises(ValueError, match=f"{field} mismatch"):
        build_failure_report(source, evidence, root=tmp_path)


def test_duplicate_source_cannot_double_count_failures() -> None:
    row = attribute_run(_checked(), _record())
    with pytest.raises(ValueError, match="duplicate failure source"):
        build_failure_report({"failure_attributions": [row, row]})


def test_rebuilding_preserves_existing_trace_reviews(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    first = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    second = build_failure_report(
        {"failure_attributions": first["attributions"]},
        root=tmp_path,
        bootstrap_samples=0,
    )
    assert second["attributions"] == first["attributions"]


def test_failure_label_does_not_imply_stage_reachability(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path, "U")
    source["failure_attributions"].append(attribute_run(_checked(), _record()))
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    stages = report["statistics"]["summary"]["stages"]
    assert all(value["conditional_failure_rate"] is None for value in stages.values())
    assert stages["U"]["share_of_failures"] == 0.5


def test_single_class_ratings_do_not_define_kappa(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    run_id = source["failure_attributions"][0]["run_id"]
    evidence["ratings"] = [
        {"run_id": run_id, "rater_id": rater, "pulse_stage": "L"}
        for rater in ("a", "b")
    ]
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    assert report["reliability"]["raw_agreement"] == 1
    assert report["reliability"]["cohen_kappa"] is None
    assert "cohen_kappa" not in report["reliability"]["acceptance"]


def test_partial_reviews_do_not_pass_reliability_acceptance(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    source["failure_attributions"].append(attribute_run(_checked(), _record()))
    run_id = source["failure_attributions"][0]["run_id"]
    evidence["ratings"] = [
        {"run_id": run_id, "rater_id": rater, "pulse_stage": "L"}
        for rater in ("a", "b")
    ]
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    assert not any(report["reliability"]["acceptance"].values())


def test_low_confidence_does_not_displace_high_confidence_sample() -> None:
    low = [
        {
            "run_id": f"low-{index}",
            "run_verdict": "valid_model_failure",
            "confidence": "low",
            "attribution_status": "evidence_insufficient",
        }
        for index in range(102)
    ]
    high = [
        {
            "run_id": f"high-{index}",
            "run_verdict": "valid_model_failure",
            "confidence": "high",
            "attribution_status": "resolved",
            "pulse_stage": PULSE_STAGES[index % 5],
            "model": "model",
            "track": "graft",
            "site": "egomart",
        }
        for index in range(32)
    ]
    selected = select_reliability_sample(low + high)
    assert len(selected) == 112
    assert len([row for row in selected if row["confidence"] == "high"]) == 10
    assert {row.get("pulse_stage") for row in selected} >= set(PULSE_STAGES)


def test_one_human_completes_sample_without_inventing_double_rating(
    tmp_path: Path,
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    run_id = source["failure_attributions"][0]["run_id"]
    evidence["ratings"] = [{"run_id": run_id, "rater_id": "human", "pulse_stage": "L"}]
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    reliability = report["reliability"]
    assert reliability["sample_complete"]
    assert reliability["automatic_vs_human_agreement"] == 1
    assert reliability["cohen_kappa"] is None
    assert reliability["raw_agreement"] is None
    assert reliability["double_rated_cases"] == 0


def test_human_adjudication_resolves_unknown_without_relabeling_judge(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    assessment = evidence["assessments"][0]
    assessment.update(candidate_stage=None, candidate_mechanism=None,
                      candidate_step=None, candidate_confidence="low")
    assessment["trajectory_review"]["uncertainty_reason"] = "stage_boundary"
    run_id = assessment["run_id"]
    evidence["adjudications"] = {run_id: {
        "stage": "L", "reviewer_id": "user", "reason": "Human decision from the case summary."
    }}
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    row = report["attributions"][0]
    assert row["pulse_stage"] == "L"
    assert row["candidate_source"] == "human"
    assert row["confidence"] == "adjudicated"
    assert row["rule_evidence"]["original_evidence"]["uncertainty_reason"] == "stage_boundary"
    assert report["statistics"]["summary"]["unknown_failures"] == 0


def test_human_adjudication_cannot_override_a_resolved_judge(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    run_id = evidence["assessments"][0]["run_id"]
    evidence["adjudications"] = {run_id: {
        "stage": "S", "reviewer_id": "user", "reason": "Human decision."
    }}
    with pytest.raises(ValueError, match="invalid human adjudication"):
        build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)


def test_packet_reads_historical_snapshot_and_omits_model_self_report(
    tmp_path: Path,
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    packet = report["human_review"]["packets"][0]
    assert "diagnostic" not in packet
    diagnostic = historical_review_context(packet, tmp_path)
    assert diagnostic["task_context"]["source"] == "recorded_task_snapshot"
    assert diagnostic["end_state_source"] == "recorded_live_state"
    assert diagnostic["actual_final_state"] == {"selected_entity_ids": [], "fields": {}}
    message = diagnostic["web_observations"][0]["state_message"]
    assert "Saved items" in message and "Save this" in message
    assert "private summary" not in message


def test_historical_scoring_cannot_be_used_as_new_attribution() -> None:
    with pytest.raises(ValueError, match="final-state Checker"):
        attribute_run(_checked(evaluator={"version": "2.0"}), _record())


def test_current_checker_output_can_be_attributed_without_version_rewriting() -> None:
    definition = task("clarify")
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": definition["track"],
            "condition": CONDITIONS["online"].as_dict(),
            "steps": 1,
            "actions": [],
            "live_state": {"selected_entity_ids": [], "fields": {}},
        }
    )

    checked = check_run(definition, run)
    attribution = attribute_run(checked, run)

    assert checked["evaluator"]["version"] == EVALUATOR_VERSION
    assert attribution["checker_version"] == EVALUATOR_VERSION
    assert attribution["run_verdict"] == "valid_model_failure"


def test_rescored_attribution_uses_scoring_goal_and_keeps_historical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import egocua.failure_attribution as attribution

    historical = task("graft")
    revised = deepcopy(historical)
    revised["goal"]["acceptable_states"] = [
        {"selected_entity_ids": ["revised-target"], "fields": {}}
    ]
    record = complete_run_record(
        {
            "task_id": historical["id"],
            "track": historical["track"],
            "condition": CONDITIONS["upfront"].as_dict(),
            "task_snapshot": historical,
            "meta": {"run_id": "rescored-run"},
            "steps": 1,
            "actions": [],
            "live_state": {"selected_entity_ids": [], "fields": {}},
        }
    )
    checked = rescore_run(revised, record)
    row = attribute_run(checked, record)
    row["run_artifact"] = "artifacts/run"
    monkeypatch.setattr(attribution, "load_run_record", lambda _: record)
    monkeypatch.setattr(attribution, "_released_visual_packets", lambda *_: ([], []))

    diagnostic = historical_review_context(row, tmp_path)

    assert diagnostic["expected_states"] == revised["goal"]["acceptable_states"]
    assert diagnostic["scoring_goal_source"] == "checker_evaluator"
    assert diagnostic["task_context"]["instructions"] == historical["instructions"]
    assert diagnostic["task_context"]["reference"]["goal"] == historical["goal"]
    expected_states = deepcopy(revised["goal"]["acceptable_states"])
    checked["evaluator"]["acceptable_states"].clear()
    assert row["scoring_goal"]["acceptable_states"] == expected_states
    row.pop("scoring_goal")
    missing_goal = historical_review_context(row, tmp_path)
    assert missing_goal["expected_states"] is None
    assert "checker_scoring_goal" in missing_goal["missing_evidence"]


def test_attribution_rejects_checker_result_without_scoring_goal() -> None:
    with pytest.raises(ValueError, match="acceptable_states"):
        attribute_run(_checked(evaluator={"version": EVALUATOR_VERSION}), _record())


def test_new_hash_free_sampling_run_retains_complete_historical_materials(
    tmp_path: Path,
) -> None:
    import json

    from egocua.environment import load_site_data
    from egocua.execution import RunConfig, RunContext, run_experiment
    from egocua.timeline import Condition
    from tests.helpers import materialize_site, products

    definition = task("refresh")
    materialize_site(products("refresh"), tmp_path)
    site_snapshot = tmp_path / "study/sites/egomart.json"
    site_snapshot.parent.mkdir(parents=True)
    site_snapshot.write_text(json.dumps(load_site_data(tmp_path, "egomart")))
    for packet in definition["visual_packets"]:
        pool = [f"frames/{packet['id']}-{index}.jpg" for index in range(3)]
        packet["frequency_samples"] = {
            "window_seconds": [0, 2],
            "timestamps_seconds": [0, 1, 2],
            "frame_paths": pool,
        }
        for relative in [*packet["frame_paths"], *pool]:
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"historical image")

    def adapter(context: RunContext) -> dict:
        context.advance()
        context.advance()
        return {
            "history": [
                {"metadata": {"step_number": 2}, "state_message": "Recorded page"}
            ]
        }

    record = run_experiment(
        definition,
        Condition(
            "sparse_repeated",
            "online",
            "full",
            "allowed",
            None,
            frame_interval_seconds=2,
            frame_repeat_to_interval_seconds=1,
        ),
        RunConfig(
            agent="offline-fixture",
            model="offline-fixture",
            default_release_step=2,
            reference_steps=None,
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            site_snapshot=site_snapshot,
        ),
        adapter,
    )
    assert (
        not {"task_sha256", "site_data_sha256", "asset_sha256"} & record["meta"].keys()
    )
    row = attribute_run(check_run(definition, record), record)
    row["run_artifact"] = (
        Path(record["meta"]["artifact_directory"])
        .relative_to(tmp_path.resolve())
        .as_posix()
    )
    definition["instructions"]["base"] = "Changed current request"
    materialize_site(
        [{**products("refresh")[0], "name": "Changed current product"}], tmp_path
    )

    diagnostic = historical_review_context(row, tmp_path)

    assert diagnostic["missing_evidence"] == []
    assert (
        diagnostic["task_context"]["instructions"]["base"] != "Changed current request"
    )
    packets = diagnostic["task_context"]["visual_packets"]
    assert [(packet["id"], packet["released_at_step"]) for packet in packets] == [
        ("V0", 0),
        ("V1", 2),
    ]
    assert [packet["frame_paths"] for packet in packets] == [
        [
            f"frames/{packet_id}-0.jpg",
            f"frames/{packet_id}-0.jpg",
            f"frames/{packet_id}-2.jpg",
        ]
        for packet_id in ("V0", "V1")
    ]
    assert [packet["timestamps_seconds"] for packet in packets] == [
        [0, 0, 2],
        [0, 0, 2],
    ]
    assert (
        diagnostic["site_context"]["entities"][0]["name"] != "Changed current product"
    )


def test_current_task_file_cannot_fill_missing_historical_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import egocua.failure_attribution as attribution

    current = task("graft")
    path = tmp_path / f"data/tasks/graft/{current['id']}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(current))
    record = {"task_id": current["id"], "meta": {"run_id": "run"}, "trace": []}
    monkeypatch.setattr(attribution, "load_run_record", lambda _: record)

    diagnostic = historical_review_context(
        {
            "task_id": current["id"],
            "run_id": "run",
            "run_artifact": "artifacts/run",
        },
        tmp_path,
    )

    assert "historical_task_snapshot" in diagnostic["missing_evidence"]
    assert "checker_scoring_goal" in diagnostic["missing_evidence"]
    assert diagnostic["expected_states"] is None
    assert "instructions" not in diagnostic["task_context"]


@pytest.mark.parametrize("source_change", ["replaced", "removed", "archive_missing"])
def test_historical_review_serves_archived_released_frames(
    tmp_path: Path, source_change: str
) -> None:
    from egocua.execution import RunConfig, RunContext, run_experiment
    from egocua.review_web import create_review_app
    from egocua.review_workspace import ReviewWorkspace
    from tests.helpers import materialize_assets, products

    definition = task("refresh")
    materialize_assets(definition, products("refresh"), tmp_path)

    def adapter(context: RunContext) -> dict:
        context.observation_images(context.initial_payload["observations"][0])
        return {"history": [{"state_message": "Recorded page"}]}

    record = run_experiment(
        definition,
        CONDITIONS["online"],
        RunConfig(
            agent="offline-fixture",
            model="offline-fixture",
            default_release_step=2,
            reference_steps=None,
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
        ),
        adapter,
    )
    directory = Path(record["meta"]["artifact_directory"])
    archive = Path(record["meta"]["runtime"]["visual_asset_root"])
    first = definition["visual_packets"][0]["frame_paths"][0]
    original = (archive / first).read_bytes()
    if source_change == "removed":
        (tmp_path / first).unlink()
    elif source_change == "replaced":
        (tmp_path / first).write_bytes(b"different image never shown to the model")
    else:
        (archive / first).unlink()

    diagnostic = historical_review_context(
        {
            **attribute_run(check_run(definition, record), record),
            "run_artifact": directory.relative_to(tmp_path).as_posix(),
        },
        tmp_path,
    )

    packets = diagnostic["task_context"]["visual_packets"]
    assert [packet["id"] for packet in packets] == ["V0"]
    assert not any(
        (archive / frame).is_file()
        for frame in definition["visual_packets"][1]["frame_paths"]
    )
    archived_reference = (archive / first).relative_to(tmp_path).as_posix()
    if source_change == "archive_missing":
        assert diagnostic["missing_evidence"] == [f"historical_frame:{first}"]
        assert archived_reference not in packets[0]["frame_paths"]
        assert first not in packets[0]["frame_paths"]
        return

    assert diagnostic["missing_evidence"] == []
    assert packets[0]["frame_paths"][0] == archived_reference
    workspace = ReviewWorkspace.open(tmp_path, "reviewer", task_ids=[definition["id"]])
    client = create_review_app(workspace).test_client()
    response = client.get("/review/asset", query_string={"path": archived_reference})
    assert response.status_code == 200
    assert response.data == original


def test_recorded_frame_must_belong_to_its_historical_packet(tmp_path: Path) -> None:
    (tmp_path / "later.jpg").write_bytes(b"later")
    definition = {
        "visual_packets": [
            {"id": "V0", "frame_paths": ["initial.jpg"]},
            {"id": "V1", "frame_paths": ["later.jpg"]},
        ]
    }
    record = {
        "meta": {
            "released_visual_inputs": [
                {"id": "V0", "step": 0, "frame_paths": ["later.jpg"]},
            ]
        }
    }

    packets, missing = _released_visual_packets(record, definition, tmp_path)

    assert packets == []
    assert missing == ["frame_not_in_historical_task:later.jpg"]


def test_unknown_completed_review_is_not_resolved(tmp_path: Path) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    evidence["assessments"][0].update(candidate_confidence="low")
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    assert report["workflow"]["processing_complete"]
    assert not report["workflow"]["resolved_complete"]
    assert report["statistics"]["summary"]["resolved_coverage"] == 0


def test_fault_evidence_must_exist_and_belong_to_historical_run(tmp_path: Path) -> None:
    source, _ = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    record = load_run_record(tmp_path / row["run_artifact"])
    certificate = {
        "verdict": "benchmark_fault",
        "policy_version": "benchmark-fault-certificate-v1",
        "predicate": "task_invariant_failure",
        "source_run_id": row["run_id"],
        "review_method": "coding_agent",
        "reviewer_id": "independent-reviewer",
        "facts": {
            "check_name": "loading_configuration",
            "check_status": "fail",
            "expected": "front",
            "observed": "lid",
        },
        "evidence_refs": [row["run_artifact"] + "/run.json"],
    }
    fault = attribute_run(
        _checked(run_id=row["run_id"], task_id=row["task_id"]),
        record,
        qualification=certificate,
    )
    fault["run_artifact"] = row["run_artifact"]
    report = build_failure_report(
        {"failure_attributions": [fault]}, root=tmp_path, bootstrap_samples=0
    )
    assert report["statistics"]["summary"]["benchmark_faults"] == 1
    fault["qualification_evidence"]["review_method"] = "deterministic_check"
    with pytest.raises(ValueError, match="no deterministic validator"):
        build_failure_report(
            {"failure_attributions": [fault]}, root=tmp_path, bootstrap_samples=0
        )
    fault["qualification_evidence"]["review_method"] = "coding_agent"
    fault["qualification_evidence"]["source_run_id"] = "other-run"
    with pytest.raises(ValueError, match="source_run_id"):
        build_failure_report(
            {"failure_attributions": [fault]}, root=tmp_path, bootstrap_samples=0
        )
    fault["qualification_evidence"]["source_run_id"] = row["run_id"]
    fault["qualification_evidence"]["evidence_refs"] = ["missing.json"]
    with pytest.raises((ValueError, FileNotFoundError)):
        build_failure_report(
            {"failure_attributions": [fault]}, root=tmp_path, bootstrap_samples=0
        )


def test_fault_strings_alone_do_not_publish_a_confirmed_fault(tmp_path: Path) -> None:
    source, _ = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    record = load_run_record(tmp_path / row["run_artifact"])
    fault = attribute_run(
        _checked(),
        record,
        qualification={
            "verdict": "benchmark_fault",
            "policy_version": "benchmark-fault-certificate-v1",
            "predicate": "task_invariant_failure",
            "facts": {
                "check_name": "made_up",
                "check_status": "fail",
                "expected": "x",
                "observed": "y",
            },
            "evidence_refs": [row["run_artifact"] + "/run.json"],
        },
    )
    fault["run_artifact"] = row["run_artifact"]
    with pytest.raises(ValueError, match="source_run_id"):
        build_failure_report(
            {"failure_attributions": [fault]}, root=tmp_path, bootstrap_samples=0
        )


@pytest.mark.parametrize("archived", [False, True])
def test_visual_context_never_shows_future_unreleased_packet(
    tmp_path: Path, archived: bool
) -> None:
    asset_root = tmp_path / "artifacts/visual-inputs" if archived else tmp_path
    asset_root.mkdir(parents=True, exist_ok=True)
    frames = {"old.jpg": b"old", "new.jpg": b"new"}
    for name, value in frames.items():
        (asset_root / name).write_bytes(value)
    definition = {
        "visual_packets": [
            {"id": "V0", "frame_paths": ["old.jpg"]},
            {"id": "V1", "frame_paths": ["new.jpg"]},
        ]
    }
    record: dict[str, Any] = {
        "condition": {"scope": "full", "schedule": "online"},
        "actions": [],
        "meta": {"runtime": {"visual_asset_root": str(asset_root)}} if archived else {},
    }
    packets, missing = _released_visual_packets(record, definition, tmp_path)
    assert not missing
    assert [(p["id"], p["released_at_step"], p["frame_paths"]) for p in packets] == [
        ("V0", 0, [(asset_root / "old.jpg").relative_to(tmp_path).as_posix()])
    ]
    record["actions"] = [
        {
            "type": "visual_context_update",
            "step": 4,
            "payload": {"frame_paths": ["new.jpg"]},
        }
    ]
    packets, _ = _released_visual_packets(record, definition, tmp_path)
    assert [(p["id"], p["released_at_step"]) for p in packets] == [("V0", 0), ("V1", 4)]
    record["initial_context"] = {"observations": []}
    packets, _ = _released_visual_packets(record, definition, tmp_path)
    assert [(p["id"], p["released_at_step"]) for p in packets] == [("V1", 4)]


@pytest.mark.parametrize(
    "trace", [[], [{"state_message": "Actual page", "result": []}], None]
)
def test_historical_context_accepts_list_or_missing_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trace: Any
) -> None:
    import egocua.failure_attribution as attribution

    record = {"task_id": "task", "meta": {"run_id": "run"}, "trace": trace}
    monkeypatch.setattr(attribution, "load_run_record", lambda _: record)
    packet = historical_review_context(
        {"run_id": "run", "task_id": "task", "run_artifact": "artifacts/run"}, tmp_path
    )
    assert packet["task_context"]["visual_packets"] == []
    assert "historical_task_snapshot" in packet["missing_evidence"]
    if trace:
        assert packet["web_observations"][0]["state_message"] == "Actual page"
    else:
        assert "historical_web_observations" in packet["missing_evidence"]


@pytest.mark.parametrize(
    "symptoms",
    [
        {"asked": False, "valid_question": False, "track": "clarify"},
        {"false_update": True},
        {"premature_submission": True},
        {"target_selected": True, "goal_state_correct": False, "stale_final": True},
    ],
)
def test_checker_never_guesses_candidate_category(symptoms: dict) -> None:
    result = attribute_run(_checked(**symptoms), _record())
    assert result["candidate_stage"] is None
    assert result["candidate_source"] is None


def test_unscorable_source_is_visible_without_failure_label(tmp_path: Path) -> None:
    original, _ = _trajectory_workspace(tmp_path)
    row = original["failure_attributions"][0]
    summary = {
        "tasks": {
            row["task_id"]: {
                "artifact": row["run_artifact"],
                "run_id": row["run_id"],
                "status": "not_rescorable",
                "reasons": ["missing_terminal_live_state"],
            }
        }
    }
    source, evidence = _benchmark_source(summary, {}, tmp_path)
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    assert report["attributions"] == []
    assert report["source_summary"] == {
        "runs": 1,
        "scored_runs": 0,
        "not_rescorable": 1,
    }
    assert report["source_exclusions"][0]["reasons"] == ["missing_terminal_live_state"]
    assert not report["workflow"]["processing_complete"]
    assert not report["workflow"]["resolved_complete"]
    assert not report["workflow"]["attribution_complete"]
    assert report["statistics"]["summary"]["valid_model_failures"] == 0
    assert report["statistics"]["summary"]["successes"] == 0


def test_empty_source_is_not_reported_as_completed() -> None:
    report = build_failure_report({"failure_attributions": []}, bootstrap_samples=0)
    assert not report["workflow"]["processing_complete"]
    assert not report["workflow"]["resolved_complete"]


def test_save_assessment_reuses_existing_and_archives_explicit_replacement(
    tmp_path: Path,
) -> None:
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH, save_assessment
    import json

    source, evidence = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    assessment = evidence["assessments"][0]
    first = save_assessment(tmp_path, row, assessment)
    changed = deepcopy(assessment)
    changed["trajectory_review"]["reason"] = (
        "New source evidence changes interpretation."
    )
    assert save_assessment(tmp_path, row, changed) == first
    replacement = save_assessment(tmp_path, row, changed, affected=True)
    assert replacement["review_history"] == [first]
    saved = json.loads((tmp_path / FAILURE_EVIDENCE_PATH).read_text())
    assert len(saved["assessments"]) == 1
    assert saved["assessments"][0] == replacement


def test_save_assessment_requires_matching_original_run_and_citations(
    tmp_path: Path,
) -> None:
    from egocua.failure_attribution import save_assessment

    source, evidence = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    assessment = evidence["assessments"][0]
    assessment["run_id"] = "another-run"
    with pytest.raises(ValueError, match="run_id mismatch"):
        save_assessment(tmp_path, row, assessment)
    assessment["run_id"] = row["run_id"]
    assessment["trajectory_review"]["citations"][0]["action"] = {}
    with pytest.raises(ValueError, match="source action"):
        save_assessment(tmp_path, row, assessment)


def _llm_assessment(coding: dict) -> dict:
    return {
        "run_id": coding["run_id"],
        "policy_version": coding["policy_version"],
        "checker_version": coding["checker_version"],
        "reviewer_kind": "llm_judge",
        "judge_model": "test-judge",
        "prompt_version": "test-v1",
        "usage": {},
        "judgment": {
            "stage": "L",
            "step": 2,
            "confidence": "high",
            "reason": "The chosen webpage object does not meet the requirement.",
            "evidence": [{"kind": "action", "index": 0}],
            "uncertainty": None,
        },
    }


def test_report_accepts_llm_judge_source_without_reopening_old_reviews(
    tmp_path: Path,
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    evidence["assessments"][0] = _llm_assessment(evidence["assessments"][0])
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    assert report["attributions"][0]["candidate_source"] == "llm_judge"
    assert report["attributions"][0]["pulse_stage"] == "L"


def test_task_issue_overrides_previous_model_label_without_changing_source(
    tmp_path: Path,
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    run_id = source["failure_attributions"][0]["run_id"]
    evidence["run_qualifications"] = {
        run_id: {
            "verdict": "task_or_site_defect",
            "reason": "Another answer may satisfy the requirement.",
        }
    }
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    assert source["failure_attributions"][0]["run_verdict"] == "valid_model_failure"
    assert report["attributions"][0]["run_verdict"] == "unknown"
    assert report["attributions"][0]["pulse_stage"] is None
    assert report["statistics"]["summary"]["valid_model_failures"] == 0


def test_judge_validation_separates_recordings_and_uses_fifty_existing_runs() -> None:
    from egocua.judge_validation import select_judge_validation_sample

    rows = [
        attribute_run(_checked(run_id=f"run-{i}", task_id=f"task-{i}"), _record())
        for i in range(60)
    ]
    groups = {row["run_id"]: f"recording-{i // 2}" for i, row in enumerate(rows)}
    selected = select_judge_validation_sample(rows, groups)
    assert len(selected["calibration_run_ids"]) == 10
    assert len(selected["validation_run_ids"]) == 40
    assert not (
        {groups[run_id] for run_id in selected["calibration_run_ids"]}
        & {groups[run_id] for run_id in selected["validation_run_ids"]}
    )
    with pytest.raises(ValueError, match="source-separated"):
        select_judge_validation_sample(rows[:30], groups)


def test_human_step_comparison_is_separate_from_category_agreement(
    tmp_path: Path,
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    run_id = source["failure_attributions"][0]["run_id"]
    evidence["ratings"] = [
        {
            "run_id": run_id,
            "rater_id": "human",
            "pulse_stage": "L",
            "first_decisive_step": 3,
        }
    ]
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    reliability = report["reliability"]
    assert reliability["classification_agreement"] == 1
    assert reliability["decisive_step_exact_agreement"] == 0
    assert reliability["decisive_step_within_one_agreement"] == 1
    assert reliability["high_confidence_error_rate"] == 0


def test_llm_unknown_can_save_only_end_state_evidence_and_task_issue(
    tmp_path: Path,
) -> None:
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH, save_assessment
    import json

    source, evidence = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    assessment = _llm_assessment(evidence["assessments"][0])
    assessment["judgment"].update(
        stage=None,
        step=None,
        confidence="low",
        evidence=[{"kind": "state", "field": "actual_final_state"}],
        uncertainty="task_validity",
    )
    save_assessment(tmp_path, row, assessment)
    saved = json.loads((tmp_path / FAILURE_EVIDENCE_PATH).read_text())
    assert (
        saved["run_qualifications"][row["run_id"]]["verdict"] == "task_or_site_defect"
    )
    report = build_failure_report(source, saved, root=tmp_path, bootstrap_samples=0)
    assert report["attributions"][0]["run_verdict"] == "unknown"
    assert not report["attributions"][0]["pulse_stage"]
    assert report["statistics"]["summary"]["benchmark_faults"] == 0


def test_llm_save_rejects_nonexistent_evidence_and_missing_judge_identity(
    tmp_path: Path,
) -> None:
    from egocua.failure_attribution import save_assessment

    source, evidence = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    assessment = _llm_assessment(evidence["assessments"][0])
    assessment["judgment"]["evidence"] = [{"kind": "action", "index": 999}]
    with pytest.raises(ValueError, match="not in this historical run"):
        save_assessment(tmp_path, row, assessment)
    assessment["judgment"]["evidence"] = [{"kind": "action", "index": 0}]
    del assessment["judge_model"]
    with pytest.raises(ValueError, match="judge_model"):
        save_assessment(tmp_path, row, assessment)


def test_unknown_human_without_step_does_not_count_as_localization_error(
    tmp_path: Path,
) -> None:
    source, evidence = _trajectory_workspace(tmp_path)
    run_id = source["failure_attributions"][0]["run_id"]
    evidence["ratings"] = [
        {
            "run_id": run_id,
            "rater_id": "human",
            "pulse_stage": None,
            "first_decisive_step": None,
        }
    ]
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    reliability = report["reliability"]
    assert reliability["decisive_step_cases"] == 0
    assert reliability["decisive_step_exact_agreement"] is None
    assert reliability["decisive_step_within_one_agreement"] is None
    assert (
        reliability["decisive_step_comparison"]
        == "explicit_human_steps_independent_of_category"
    )


def _judge_validation_fixture(root: Path) -> tuple[list[dict], dict, dict]:
    from egocua.judge_validation import register_judge_validation

    rows = []
    judgments = {}
    for index in range(50):
        source, evidence = _trajectory_workspace(root / f"case-{index}")
        row = source["failure_attributions"][0]
        row["run_artifact"] = f"case-{index}/{row['run_artifact']}"
        assessment = _llm_assessment(evidence["assessments"][0])
        rows.append(row)
        judgments[row["run_id"]] = assessment
    selection = {
        "calibration_run_ids": [row["run_id"] for row in rows[:10]],
        "validation_run_ids": [row["run_id"] for row in rows[10:]],
        "source_groups": {
            row["run_id"]: f"recording-{index}" for index, row in enumerate(rows)
        },
    }
    register_judge_validation(
        root, selection, model="test-judge", prompt_version="test-v1"
    )
    return rows, judgments, selection


def test_validation_results_do_not_replace_current_assessments_and_exclude_calibration(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import (
        FAILURE_EVIDENCE_PATH,
        save_assessment,
    )
    from egocua.judge_validation import (
        judge_validation_result,
        save_judge_validation_assessment,
    )

    rows, judgments, selection = _judge_validation_fixture(tmp_path)
    _, old_evidence = _trajectory_workspace(tmp_path / "original-coding")
    old = old_evidence["assessments"][0]
    old["run_id"] = rows[0]["run_id"]
    save_assessment(tmp_path, rows[0], old)
    from egocua.judge_validation import begin_judge_validation

    begin_judge_validation(tmp_path)
    for row in rows:
        save_judge_validation_assessment(tmp_path, row, judgments[row["run_id"]])
    path = tmp_path / FAILURE_EVIDENCE_PATH
    evidence = json.loads(path.read_text())
    assert evidence["assessments"] == [old]
    assert len(evidence["judge_validation"]["assessments"]) == 50
    evidence["ratings"] = [
        {
            "run_id": row["run_id"],
            "rater_id": "human",
            "pulse_stage": "P"
            if row["run_id"] in selection["calibration_run_ids"]
            else "L",
            "first_decisive_step": 2,
        }
        for row in rows
    ]
    result = judge_validation_result(evidence, rows, tmp_path)
    assert result["automatic_cases"] == 40
    assert result["classification_cases"] == 40
    assert result["automatic_vs_human_agreement"] == 1
    assert result["accepted"]
    assert result["model"] == "test-judge"
    assert result["prompt_version"] == "test-v1"
    changed = deepcopy(judgments[rows[0]["run_id"]])
    changed["judgment"]["reason"] = "Should not overwrite validation evidence."
    assert (
        save_judge_validation_assessment(tmp_path, rows[0], changed)["judgment"][
            "reason"
        ]
        != changed["judgment"]["reason"]
    )
    changed["judge_model"] = "another-judge"
    with pytest.raises(ValueError, match="model/prompt_version"):
        save_judge_validation_assessment(tmp_path, rows[0], changed)
    low = deepcopy(evidence)
    for assessment in low["judge_validation"]["assessments"][19:]:
        assessment["judgment"]["confidence"] = "low"
    result = judge_validation_result(low, rows, tmp_path)
    assert result["automatic_vs_human_cases"] == 9
    assert not result["accepted"]


def test_validation_registration_binds_model_once_and_rejects_stale_qualification(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import (
        register_judge_validation,
        save_judge_validation_assessment,
    )

    rows, judgments, selection = _judge_validation_fixture(tmp_path)
    path = tmp_path / FAILURE_EVIDENCE_PATH
    evidence = json.loads(path.read_text())
    evidence["judge_validation"]["model"] = None
    path.write_text(json.dumps(evidence))
    assert (
        register_judge_validation(
            tmp_path, selection, model=None, prompt_version="test-v1"
        )["model"]
        is None
    )
    not_bound = deepcopy(judgments[rows[0]["run_id"]])
    not_bound["judge_model"] = None
    with pytest.raises(ValueError, match="model/prompt_version"):
        save_judge_validation_assessment(tmp_path, rows[0], not_bound)
    assert (
        register_judge_validation(
            tmp_path, selection, model="test-judge", prompt_version="test-v1"
        )["model"]
        == "test-judge"
    )
    evidence = json.loads(path.read_text())
    evidence["run_qualifications"][rows[0]["run_id"]] = {
        "verdict": "task_or_site_defect",
        "reason": "Unresolved admissible answer.",
    }
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="currently valid"):
        save_judge_validation_assessment(
            tmp_path, rows[0], judgments[rows[0]["run_id"]]
        )


def test_study_source_reuses_archives_and_rejects_conflicts_or_missing_paths(
    tmp_path: Path,
) -> None:
    import json
    from egocua.attribution_sources import study_source

    source, evidence = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    checked = _checked(run_id=row["run_id"], task_id=row["task_id"])
    for name in ("study-a", "study-b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "study.json").write_text(
            json.dumps(
                {
                    "execution_results": [
                        {
                            "run_id": row["run_id"],
                            "artifact_directory": row["run_artifact"],
                        }
                    ]
                }
            )
        )
        (directory / "results.json").write_text(json.dumps({"runs": [checked]}))
    partial_path = tmp_path / "study-a/study.json"
    partial = json.loads(partial_path.read_text())
    partial["execution_results"].append(
        {
            "execution_index": 9,
            "artifact_directory": None,
            "error": {"type": "CancelledError"},
        }
    )
    partial_path.write_text(json.dumps(partial))
    loaded, scoped = study_source(["study-a", "study-b"], evidence, tmp_path)
    assert len(loaded["failure_attributions"]) == 1
    assert loaded["source_studies"][row["run_id"]] == ["study-a", "study-b"]
    assert scoped["assessments"] == evidence["assessments"]
    partial["execution_results"][-1]["valid"] = True
    partial_path.write_text(json.dumps(partial))
    with pytest.raises(ValueError, match="missing run_id"):
        study_source(["study-a"], evidence, tmp_path)
    partial["execution_results"][-1]["valid"] = False
    partial_path.write_text(json.dumps(partial))
    changed = {**checked, "e2e_success": True}
    (tmp_path / "study-b/results.json").write_text(json.dumps({"runs": [changed]}))
    with pytest.raises(ValueError, match="conflicting recorded grades"):
        study_source(["study-a", "study-b"], evidence, tmp_path)
    (tmp_path / "study-b/results.json").write_text(json.dumps({"runs": [checked]}))
    (tmp_path / "study-b/study.json").write_text(
        json.dumps(
            {
                "execution_results": [
                    {"run_id": row["run_id"], "artifact_directory": "artifacts/missing"}
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="conflicting artifact"):
        study_source(["study-a", "study-b"], evidence, tmp_path)
    with pytest.raises(
        (FileNotFoundError, ValueError), match="not found|does not exist|missing"
    ):
        study_source(["study-b"], evidence, tmp_path)


@pytest.mark.parametrize(
    "condition_name,instruction,clarification",
    [
        ("online", "Save the item shown here.", True),
        ("upfront", "Save the item shown here, specifically the second tool.", False),
    ],
)
def test_historical_context_keeps_actual_initial_input_separate_from_task_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    condition_name: str,
    instruction: str,
    clarification: bool,
) -> None:
    import egocua.failure_attribution as attribution

    definition = task("graft")
    record = complete_run_record(
        {
            "task_id": definition["id"],
            "track": definition["track"],
            "task_snapshot": definition,
            "condition": CONDITIONS[condition_name].as_dict(),
            "meta": {"run_id": "input-run", "model": "model-a"},
            "steps": 1,
            "initial_context": {
                "instruction": instruction,
                "clarification_available": clarification,
                "observations": [],
            },
            "actions": [],
            "live_state": {"selected_entity_ids": [], "fields": {}},
        }
    )
    row = attribute_run(_checked(task_id=definition["id"], run_id="input-run"), record)
    row["run_artifact"] = "artifacts/run"
    monkeypatch.setattr(attribution, "load_run_record", lambda _: record)
    diagnostic = historical_review_context(row, tmp_path)
    assert diagnostic["actual_initial_input"] == {
        "instruction": instruction,
        "clarification_available": clarification,
    }
    assert diagnostic["condition"] == record["condition"]
    assert "initial_instruction" not in diagnostic["missing_evidence"]
    del record["initial_context"]["instruction"]
    diagnostic = historical_review_context(row, tmp_path)
    assert "initial_instruction" in diagnostic["missing_evidence"]
    assert "instruction" not in diagnostic["actual_initial_input"]


def test_report_keeps_only_small_material_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import egocua.failure_attribution as attribution

    source, evidence = _trajectory_workspace(tmp_path)

    def forbidden_material_load(*args: object) -> None:
        raise AssertionError(
            "report must not load or embed historical material packets"
        )

    monkeypatch.setattr(
        attribution, "historical_review_context", forbidden_material_load
    )
    report = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    packet = report["human_review"]["packets"][0]
    assert set(packet) == set(attribution.REVIEW_PACKET_FIELDS)
    assert "diagnostic" not in packet
    assert "site_context" not in packet
    assert len(json.dumps(report)) < 30_000


def test_rare_stage_source_is_reserved_for_heldout_instead_of_calibration() -> None:
    from egocua.judge_validation import select_judge_validation_sample

    rows = [
        attribute_run(_checked(run_id=f"rare-{index}"), _record())
        for index in range(70)
    ]
    for index, row in enumerate(rows):
        row["pulse_stage"] = "E" if index < 2 else "L"
    groups = {row["run_id"]: f"source-{index // 2}" for index, row in enumerate(rows)}
    selection = select_judge_validation_sample(rows, groups)
    assert {"rare-0", "rare-1"} & set(selection["validation_run_ids"])
    assert not {"rare-0", "rare-1"} & set(selection["calibration_run_ids"])
    assert not {groups[run_id] for run_id in selection["calibration_run_ids"]} & {
        groups[run_id] for run_id in selection["validation_run_ids"]
    }


def test_unstarted_validation_preparation_can_be_corrected_but_human_review_freezes_it(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import register_judge_validation

    selection = {
        "calibration_run_ids": [f"r-{i}" for i in range(10)],
        "validation_run_ids": [f"r-{i}" for i in range(10, 50)],
        "source_groups": {f"r-{i}": f"g-{i}" for i in range(50)},
    }
    register_judge_validation(tmp_path, selection, model=None, prompt_version="test-v1")
    corrected = deepcopy(selection)
    corrected["calibration_run_ids"][0], corrected["validation_run_ids"][0] = (
        corrected["validation_run_ids"][0],
        corrected["calibration_run_ids"][0],
    )
    assert (
        register_judge_validation(
            tmp_path, corrected, model=None, prompt_version="test-v1"
        )["validation_run_ids"]
        == corrected["validation_run_ids"]
    )
    path = tmp_path / FAILURE_EVIDENCE_PATH
    evidence = json.loads(path.read_text())
    evidence["ratings"] = [{"run_id": "r-0", "rater_id": "human", "pulse_stage": None}]
    path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="different conditions"):
        register_judge_validation(
            tmp_path, selection, model=None, prompt_version="test-v1"
        )


def test_calibration_can_update_prompt_and_keeps_previous_calibration_judgments(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import (
        register_judge_validation,
        save_judge_validation_assessment,
    )

    rows, judgments, selection = _judge_validation_fixture(tmp_path)
    old = save_judge_validation_assessment(
        tmp_path, rows[0], judgments[rows[0]["run_id"]]
    )
    updated = register_judge_validation(
        tmp_path, selection, model="test-judge", prompt_version="test-v2"
    )
    assert updated["prompt_version"] == "test-v2"
    assert updated["assessments"] == []
    assert updated["calibration_history"] == [
        {"model": "test-judge", "prompt_version": "test-v1", "assessments": [old]}
    ]
    assert (
        json.loads((tmp_path / FAILURE_EVIDENCE_PATH).read_text())["assessments"] == []
    )
    with pytest.raises(ValueError, match="model/prompt_version"):
        save_judge_validation_assessment(tmp_path, rows[0], old)


@pytest.mark.parametrize("started_by", ["api_boundary", "human"])
def test_heldout_start_or_human_rating_freezes_model_and_prompt(
    tmp_path: Path, started_by: str
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import (
        begin_judge_validation,
        register_judge_validation,
    )

    rows, _, selection = _judge_validation_fixture(tmp_path)
    if started_by == "api_boundary":
        begin_judge_validation(tmp_path)
    else:
        path = tmp_path / FAILURE_EVIDENCE_PATH
        evidence = json.loads(path.read_text())
        evidence["ratings"] = [
            {"run_id": rows[10]["run_id"], "rater_id": "human", "pulse_stage": "L"}
        ]
        path.write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="frozen"):
        register_judge_validation(
            tmp_path, selection, model="test-judge", prompt_version="test-v2"
        )


def test_new_validation_keeps_history_and_requires_unexposed_sources(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import (
        begin_judge_validation,
        start_new_judge_validation,
    )

    _, _, selection = _judge_validation_fixture(tmp_path)
    previous = begin_judge_validation(tmp_path)
    with pytest.raises(ValueError, match="not exposed"):
        start_new_judge_validation(tmp_path, selection)
    fresh = {
        "calibration_run_ids": [f"fresh-{index}" for index in range(10)],
        "validation_run_ids": [f"fresh-{index}" for index in range(10, 50)],
        "source_groups": {
            f"fresh-{index}": f"unseen-source-{index}" for index in range(50)
        },
    }
    current = start_new_judge_validation(tmp_path, fresh)
    assert current["model"] is None
    assert not current["validation_started"]
    assert current["assessments"] == []
    evidence = json.loads((tmp_path / FAILURE_EVIDENCE_PATH).read_text())
    assert evidence["judge_validation_history"] == [previous]
    assert evidence["assessments"] == []


def test_judge_stores_one_six_field_judgment_and_rebuilds_without_legacy_review(
    tmp_path: Path,
) -> None:
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH, save_assessment
    import json

    source, old = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    assessment = _llm_assessment(old["assessments"][0])
    save_assessment(tmp_path, row, assessment)
    evidence = json.loads((tmp_path / FAILURE_EVIDENCE_PATH).read_text())
    stored = evidence["assessments"][0]
    assert set(stored) == {
        "run_id",
        "policy_version",
        "checker_version",
        "reviewer_kind",
        "judge_model",
        "prompt_version",
        "usage",
        "judgment",
    }
    assert set(stored["judgment"]) == {
        "stage",
        "step",
        "confidence",
        "reason",
        "evidence",
        "uncertainty",
    }
    first = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    result = first["attributions"][0]
    assert result["mechanism"] is None and result["candidate_mechanism"] is None
    assert set(result["rule_evidence"]) == {
        "policy_version",
        "reason",
        "evidence_refs",
        "uncertainty_reason",
    }
    rebuilt = build_failure_report(
        {"failure_attributions": first["attributions"]},
        root=tmp_path,
        bootstrap_samples=0,
    )
    assert rebuilt["attributions"] == first["attributions"]


def test_six_field_judgment_checks_step_frames_and_uncertainty_without_mechanisms() -> (
    None
):
    from egocua.failure_attribution import validate_judgment

    context = {
        "actions": [{"step": 2, "type": "search"}],
        "web_observations": [],
        "task_context": {
            "visual_packets": [{"frame_paths": ["frame.jpg"], "released_at_step": 0}]
        },
    }
    judgment = {
        "stage": "P",
        "step": 2,
        "confidence": "high",
        "reason": "Search uses an object contradicted by the delivered frame.",
        "evidence": [
            {"kind": "action", "index": 0},
            {"kind": "frame", "packet_index": 0, "index": 0},
        ],
        "uncertainty": None,
    }
    validate_judgment(context, judgment)
    with pytest.raises(ValueError, match="exactly"):
        validate_judgment(context, {**judgment, "mechanism": "object_grounding"})
    context["task_context"]["visual_packets"][0]["released_at_step"] = 3
    with pytest.raises(ValueError, match="frames delivered"):
        validate_judgment(context, judgment)
    context["task_context"]["visual_packets"][0]["released_at_step"] = 0
    with pytest.raises(ValueError, match="corresponding action"):
        validate_judgment(context, {**judgment, "step": 8})
    with pytest.raises(ValueError, match="null stage"):
        validate_judgment(context, {**judgment, "uncertainty": "task_validity"})


def test_judge_rejects_forged_model_identity_and_an_invalid_source_archive(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import save_assessment

    source, old = _trajectory_workspace(tmp_path)
    row = source["failure_attributions"][0]
    assessment = _llm_assessment(old["assessments"][0])
    with pytest.raises(ValueError, match="identity mismatch"):
        save_assessment(tmp_path, {**row, "model": "another-model"}, assessment)
    archive = tmp_path / row["run_artifact"] / "run.json"
    record = json.loads(archive.read_text())
    record["run_status"] = {"valid": False, "reason": "browser_failure"}
    archive.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="valid archived run"):
        save_assessment(tmp_path, row, assessment)


def test_empty_preparation_prompt_update_does_not_create_calibration_history(
    tmp_path: Path,
) -> None:
    from egocua.judge_validation import register_judge_validation

    selection = {
        "calibration_run_ids": [f"r-{i}" for i in range(10)],
        "validation_run_ids": [f"r-{i}" for i in range(10, 50)],
        "source_groups": {f"r-{i}": f"g-{i}" for i in range(50)},
    }
    register_judge_validation(
        tmp_path, selection, model=None, prompt_version="trajectory-judge-v1"
    )
    prepared = register_judge_validation(
        tmp_path, selection, model=None, prompt_version="trajectory-judge-v2"
    )
    assert "calibration_history" not in prepared
    assert prepared["assessments"] == []


def test_experimental_task_validity_does_not_change_global_qualifications(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import save_judge_validation_assessment

    rows, assessments, _ = _judge_validation_fixture(tmp_path)
    row = rows[0]
    assessment = deepcopy(assessments[row["run_id"]])
    assessment["judgment"].update(
        stage=None,
        step=None,
        confidence="low",
        uncertainty="task_validity",
        evidence=[{"kind": "state", "field": "actual_final_state"}],
    )
    save_judge_validation_assessment(tmp_path, row, assessment)
    evidence = json.loads((tmp_path / FAILURE_EVIDENCE_PATH).read_text())
    assert evidence["run_qualifications"] == {}
    assert evidence["assessments"] == []
    assert (
        evidence["judge_validation"]["assessments"][0]["judgment"]["uncertainty"]
        == "task_validity"
    )


def test_judgment_rejects_copied_content_but_shared_old_reference_value_check_remains() -> (
    None
):
    from egocua.failure_attribution import (
        validate_evidence_references,
        validate_judgment,
    )

    action = {"step": 2, "type": "search", "query": "wrong object"}
    context = {"actions": [action]}
    judgment = {
        "stage": "L",
        "step": 2,
        "confidence": "high",
        "reason": "The selected entity does not meet the requirement.",
        "evidence": [{"kind": "action", "index": 0, "actions": action}],
        "uncertainty": None,
    }
    with pytest.raises(ValueError, match="pointer-only"):
        validate_judgment(context, judgment)
    old_ref = {"kind": "action", "index": 0, "value": action}
    validate_evidence_references(context, [old_ref])
    with pytest.raises(ValueError, match="recorded fact"):
        validate_evidence_references(
            context, [{**old_ref, "value": {"type": "another action"}}]
        )
    with pytest.raises(ValueError, match="pointer-only"):
        validate_judgment(context, {**judgment, "evidence": [old_ref]})


def test_judgment_resolves_each_reference_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import egocua.failure_attribution as attribution

    context = {
        "actions": [{"step": 2, "type": "search"}],
        "task_context": {
            "visual_packets": [{"frame_paths": ["frame.jpg"], "released_at_step": 0}]
        },
    }
    judgment = {
        "stage": "P",
        "step": 2,
        "confidence": "high",
        "reason": "The search contradicts the visible object.",
        "evidence": [
            {"kind": "action", "index": 0},
            {"kind": "frame", "packet_index": 0, "index": 0},
        ],
        "uncertainty": None,
    }
    calls = []
    resolve = attribution.resolve_evidence_reference

    def counted(context: dict, reference: dict) -> tuple:
        calls.append(reference)
        return resolve(context, reference)

    monkeypatch.setattr(attribution, "resolve_evidence_reference", counted)
    attribution.validate_judgment(context, judgment)
    assert calls == judgment["evidence"]


def test_report_verifies_the_current_assessment_not_the_superseded_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import egocua.failure_attribution as attribution

    source, evidence = _trajectory_workspace(tmp_path)
    first = build_failure_report(source, evidence, root=tmp_path, bootstrap_samples=0)
    loads = []
    load = attribution.load_run_record

    def counted(directory: Path) -> dict:
        loads.append(directory)
        return load(directory)

    monkeypatch.setattr(attribution, "load_run_record", counted)
    second = build_failure_report(
        {"failure_attributions": first["attributions"]},
        evidence,
        root=tmp_path,
        bootstrap_samples=0,
    )
    assert second["attributions"] == first["attributions"]
    assert len(loads) == 1


def test_heldout_statistics_do_not_reload_calibration_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import egocua.failure_attribution as attribution
    import egocua.judge_validation as validation

    rows, judgments, selection = _judge_validation_fixture(tmp_path)
    validation.begin_judge_validation(tmp_path)
    for row in rows:
        validation.save_judge_validation_assessment(
            tmp_path, row, judgments[row["run_id"]]
        )
    evidence = json.loads((tmp_path / attribution.FAILURE_EVIDENCE_PATH).read_text())
    reviewed = []
    verify = attribution.verify_judge_assessment

    def counted(row: dict, assessment: dict, root: Path) -> dict:
        reviewed.append(row["run_id"])
        return verify(row, assessment, root)

    monkeypatch.setattr(attribution, "verify_judge_assessment", counted)
    result = validation.judge_validation_result(evidence, rows, tmp_path)
    assert result["automatic_cases"] == 40
    assert len(reviewed) == 40
    assert set(reviewed) == set(selection["validation_run_ids"])


def test_dispute_note_and_recheck_preserve_frozen_results_and_formal_reviews(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import (
        FAILURE_EVIDENCE_PATH,
        save_assessment,
    )
    from egocua.judge_validation import (
        begin_judge_validation,
        judge_validation_result,
        save_judge_dispute_review,
        save_judge_recheck,
        save_judge_validation_assessment,
    )

    rows, judgments, _ = _judge_validation_fixture(tmp_path)
    row = rows[10]
    _, coding = _trajectory_workspace(tmp_path / "formal-review")
    formal = coding["assessments"][0]
    formal["run_id"] = row["run_id"]
    save_assessment(tmp_path, row, formal)
    begin_judge_validation(tmp_path)
    original = save_judge_validation_assessment(tmp_path, row, judgments[row["run_id"]])
    path = tmp_path / FAILURE_EVIDENCE_PATH
    evidence = json.loads(path.read_text())
    evidence["ratings"] = [
        {"run_id": row["run_id"], "rater_id": "human", "pulse_stage": "L"}
    ]
    path.write_text(json.dumps(evidence))
    baseline = judge_validation_result(evidence, rows, tmp_path)
    review = {
        "verdict": "both_revised",
        "reviewer_kind": "coding_agent",
        "judgment": {
            **deepcopy(original["judgment"]),
            "stage": "U",
            "reason": "The delivered update was not used.",
        },
    }
    assert save_judge_dispute_review(tmp_path, row, review) == review
    assert save_judge_dispute_review(tmp_path, row, {}) == review
    recheck = deepcopy(original)
    recheck["prompt_version"] = "test-v2"
    recheck["judgment"]["stage"] = "S"
    assert save_judge_recheck(tmp_path, row, recheck) == recheck
    changed = deepcopy(recheck)
    changed["judgment"]["evidence"] = []
    assert save_judge_recheck(tmp_path, row, changed) == recheck
    annotated = {**deepcopy(recheck), "dispute_review": deepcopy(review)}
    modified = deepcopy(annotated)
    modified["judgment"]["stage"] = "E"
    with pytest.raises(ValueError, match="cannot change the original"):
        save_judge_recheck(tmp_path, row, modified)
    invalid_note = deepcopy(annotated)
    invalid_note["dispute_review"]["judgment"]["evidence"] = [
        {"kind": "action", "index": 999}
    ]
    with pytest.raises(ValueError, match="not in this historical run"):
        save_judge_recheck(tmp_path, row, invalid_note)
    assert save_judge_recheck(tmp_path, row, annotated) == annotated
    assert save_judge_recheck(tmp_path, row, modified) == annotated
    saved = json.loads(path.read_text())
    assert saved["assessments"] == [formal]
    assert saved["ratings"] == evidence["ratings"]
    assert saved["run_qualifications"] == evidence["run_qualifications"]
    validation = saved["judge_validation"]
    assert validation["model"] == "test-judge"
    assert validation["prompt_version"] == "test-v1"
    assert validation["assessments"] == [{**original, "dispute_review": review}]
    assert validation["rechecks"] == [annotated]
    result = judge_validation_result(saved, rows, tmp_path)
    assert result["dispute_review_summary"] == {
        "reviewed": 1,
        "verdict_counts": {"both_revised": 1},
    }
    assert result["recheck_cases"] == 1
    for field in (
        "automatic_cases",
        "reviewed_cases",
        "automatic_vs_human_agreement",
        "accepted",
    ):
        assert result[field] == baseline[field]


@pytest.mark.parametrize("operation", ["dispute", "recheck"])
def test_judge_followup_rejects_bad_references_and_unregistered_runs(
    tmp_path: Path, operation: str
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import (
        save_judge_dispute_review,
        save_judge_recheck,
        save_judge_validation_assessment,
    )

    rows, judgments, _ = _judge_validation_fixture(tmp_path)
    row = rows[0]
    assessment = judgments[row["run_id"]]
    save_judge_validation_assessment(tmp_path, row, assessment)
    changed = deepcopy(assessment)
    changed["prompt_version"] = "test-v2"
    changed["judgment"]["evidence"] = [{"kind": "action", "index": 999}]
    if operation == "dispute":
        save = save_judge_dispute_review
        followup = {
            "verdict": "old_supported",
            "reviewer_kind": "coding_agent",
            "judgment": changed["judgment"],
        }
    else:
        save = save_judge_recheck
        followup = changed
    path = tmp_path / FAILURE_EVIDENCE_PATH
    before = json.loads(path.read_text())
    with pytest.raises(ValueError, match="not in this historical run"):
        save(tmp_path, row, followup)
    outside = {**row, "run_id": "not-registered"}
    with pytest.raises(ValueError, match="outside the registered"):
        save(tmp_path, outside, followup)
    assert json.loads(path.read_text()) == before


def test_dispute_review_requires_current_judge_and_coding_agent_source(
    tmp_path: Path,
) -> None:
    from egocua.judge_validation import (
        save_judge_dispute_review,
        save_judge_validation_assessment,
    )

    rows, judgments, _ = _judge_validation_fixture(tmp_path)
    row = rows[0]
    review = {
        "verdict": "old_supported",
        "reviewer_kind": "human",
        "judgment": judgments[row["run_id"]]["judgment"],
    }
    with pytest.raises(ValueError, match="existing Judge"):
        save_judge_dispute_review(tmp_path, row, review)
    save_judge_validation_assessment(tmp_path, row, judgments[row["run_id"]])
    with pytest.raises(ValueError, match="Coding Agent dispute"):
        save_judge_dispute_review(tmp_path, row, review)


def test_judge_output_coverage_counts_each_run_once_without_promoting_rechecks(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.judge_validation import (
        begin_judge_validation,
        judge_validation_result,
        save_judge_recheck,
    )

    rows, judgments, _ = _judge_validation_fixture(tmp_path)
    begin_judge_validation(tmp_path)
    path = tmp_path / FAILURE_EVIDENCE_PATH
    evidence = json.loads(path.read_text())
    evidence["judge_validation"]["assessments"] = [
        judgments[row["run_id"]] for row in rows[:-1]
    ]
    path.write_text(json.dumps(evidence))
    before = judge_validation_result(evidence, rows, tmp_path)
    assert before["output_coverage"] == {
        "selected_cases": 50,
        "original_output_cases": 49,
        "rechecked_only_cases": 0,
        "covered_cases": 49,
        "missing_run_ids": [rows[-1]["run_id"]],
    }
    for prompt in ("test-v2", "test-v3"):
        assessment = {**judgments[rows[-1]["run_id"]], "prompt_version": prompt}
        save_judge_recheck(tmp_path, rows[-1], assessment)
    saved = json.loads(path.read_text())
    result = judge_validation_result(saved, rows, tmp_path)
    assert result["output_coverage"] == {
        "selected_cases": 50,
        "original_output_cases": 49,
        "rechecked_only_cases": 1,
        "covered_cases": 50,
        "missing_run_ids": [],
    }
    assert result["recheck_cases"] == 2
    assert result["automatic_cases"] == before["automatic_cases"] == 39
    assert result["reviewed_cases"] == 0
    assert not result["accepted"]
    assert saved["judge_validation"]["prompt_version"] == "test-v1"
    saved["judge_validation"]["rechecks"][0]["run_id"] = "unregistered"
    with pytest.raises(ValueError, match="unregistered"):
        judge_validation_result(saved, rows, tmp_path)


def test_judge_validation_keeps_original_prediction_after_task_validity_review(
    tmp_path: Path,
) -> None:
    import json
    from egocua.failure_attribution import (
        FAILURE_EVIDENCE_PATH,
        qualify_attribution_row,
    )
    from egocua.judge_validation import (
        begin_judge_validation,
        judge_validation_result,
        save_judge_validation_assessment,
    )

    rows, judgments, _ = _judge_validation_fixture(tmp_path)
    row = rows[10]
    begin_judge_validation(tmp_path)
    save_judge_validation_assessment(tmp_path, row, judgments[row["run_id"]])
    evidence = json.loads((tmp_path / FAILURE_EVIDENCE_PATH).read_text())
    qualification = {
        "verdict": "task_or_site_defect",
        "reason": "The selected tutorial may be a valid omitted answer.",
    }
    evidence["run_qualifications"][row["run_id"]] = qualification
    evidence["ratings"] = [
        {
            "run_id": row["run_id"],
            "rater_id": "human",
            "pulse_stage": None,
            "run_verdict": "task_or_site_defect",
        }
    ]
    qualified = qualify_attribution_row(row, qualification, tmp_path)
    current_rows = [
        qualified if item["run_id"] == row["run_id"] else item for item in rows
    ]
    original_rows = deepcopy(current_rows)
    result = judge_validation_result(evidence, current_rows, tmp_path)
    assert result["classification_agreement"] == 0
    assert result["high_confidence_error_rate"] == 1
    assert result["automatic_confusion_matrix"]["TaskIssue"]["L"] == 1
    assert result["automatic_vs_human_cases"] == 1
    report = build_failure_report(
        {"failure_attributions": rows}, evidence, root=tmp_path, bootstrap_samples=0
    )
    formal = next(
        item for item in report["attributions"] if item["run_id"] == row["run_id"]
    )
    assert formal["run_verdict"] == "unknown"
    assert formal["pulse_stage"] is None
    assert report["statistics"]["summary"]["valid_model_failures"] == 49
    assert report["judge_validation"]["classification_agreement"] == 0
    assert current_rows == original_rows
    assert evidence["judge_validation"]["assessments"] == [judgments[row["run_id"]]]
    judge_task_issue = deepcopy(evidence)
    judge_task_issue["judge_validation"]["assessments"][0]["judgment"].update(
        stage=None, step=None, confidence="low", uncertainty="task_validity"
    )
    result = judge_validation_result(judge_task_issue, current_rows, tmp_path)
    assert result["automatic_confusion_matrix"]["TaskIssue"]["TaskIssue"] == 1
    assert result["classification_agreement"] == 1


@pytest.mark.parametrize("delivery", ["recorded_inputs", "initial", "update"])
@pytest.mark.parametrize("has_packing_metadata", [True, False])
def test_released_packets_preserve_only_recorded_image_packing_settings(
    tmp_path: Path, delivery: str, has_packing_metadata: bool
) -> None:
    paths = ["second.jpg", "first.jpg"]
    for frame in paths:
        (tmp_path / frame).write_bytes(b"recorded frame")
    definition = {
        "visual_packets": [
            {
                "id": "V0",
                "frame_paths": paths,
                "wearer_frames_per_image": 8,
                "wearer_jpeg_quality": 10,
            }
        ]
    }
    payload = {
        "id": "V0",
        "frame_paths": paths,
        "timestamps_seconds": [12, 14],
        "frame_interval_seconds": 2,
    }
    if has_packing_metadata:
        payload.update(wearer_frames_per_image=4, wearer_jpeg_quality=85,
                       wearer_max_image_edge=1280)
    record: dict[str, Any] = {
        "initial_context": {"observations": []},
        "condition": {"wearer_frames_per_image": 1, "wearer_jpeg_quality": 95},
    }
    if delivery == "recorded_inputs":
        record["meta"] = {"released_visual_inputs": [{**payload, "step": 3}]}
        step = 3
    elif delivery == "initial":
        record["initial_context"]["observations"] = [payload]
        step = 0
    else:
        record["actions"] = [
            {"type": "visual_context_update", "step": 5, "payload": payload}
        ]
        step = 5
    packets, missing = _released_visual_packets(record, definition, tmp_path)
    assert not missing
    assert len(packets) == 1
    packet = packets[0]
    assert packet["frame_paths"] == paths
    assert packet["timestamps_seconds"] == [12, 14]
    assert packet["frame_interval_seconds"] == 2
    assert packet["released_at_step"] == step
    if has_packing_metadata:
        assert packet["wearer_frames_per_image"] == 4
        assert packet["wearer_jpeg_quality"] == 85
        assert packet["wearer_max_image_edge"] == 1280
    else:
        assert "wearer_frames_per_image" not in packet
        assert "wearer_jpeg_quality" not in packet
        assert "wearer_max_image_edge" not in packet


def test_stream_review_uses_archived_pool_but_only_delivered_frames(tmp_path):
    for name in ("opening.jpg", "received.jpg", "future.jpg", "unregistered.jpg"):
        (tmp_path / name).write_bytes(b"recorded frame")
    task = {
        "visual_packets": [{"id": "V0", "frame_paths": ["opening.jpg"]}],
        "visual_session": {"frequency_samples": {
            "frame_paths": ["received.jpg", "future.jpg"],
        }},
    }
    record = {"meta": {"released_visual_inputs": [{
        "step": 3, "frame_paths": ["received.jpg", "unregistered.jpg"],
        "timestamps_seconds": [12, 13], "wearer_frames_per_image": 4,
        "wearer_jpeg_quality": 85, "wearer_max_image_edge": 1280,
    }]}}
    packets, missing = _released_visual_packets(record, task, tmp_path)
    assert missing == ["frame_not_in_historical_task:unregistered.jpg"]
    assert len(packets) == 1
    assert packets[0]["id"] == "stream"
    assert packets[0]["frame_paths"] == ["received.jpg"]
    assert packets[0]["timestamps_seconds"] == [12]
    assert packets[0]["released_at_step"] == 3
    assert packets[0]["wearer_max_image_edge"] == 1280


@pytest.mark.parametrize("reason", ["stage_boundary", "evidence_gap"])
def test_all_uncertain_llm_judgments_are_routed_to_review(tmp_path: Path, reason: str):
    from egocua.failure_attribution import apply_assessments

    rows, judgments, _ = _judge_validation_fixture(tmp_path)
    row = rows[0]
    judgment = deepcopy(judgments[row["run_id"]])
    judgment["judgment"].update(stage=None, step=None, confidence="low", uncertainty=reason)
    result = apply_assessments([row], [judgment], tmp_path)[0]
    assert result["review_required"] is True
    assert result["pulse_stage"] is None
    assert result["rule_evidence"]["uncertainty_reason"] == reason


def test_historical_visual_archive_can_be_workspace_root(tmp_path: Path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"recorded-frame")
    packet = {"id": "V0", "frame_paths": ["frame.jpg"]}
    record = {
        "meta": {"runtime": {"visual_asset_root": str(tmp_path.resolve())}},
        "initial_context": {"observations": [packet]}, "actions": [],
    }
    released, missing = _released_visual_packets(record, {"visual_packets": [packet]}, tmp_path)
    assert missing == []
    assert released[0]["frame_paths"] == ["frame.jpg"]


def test_judge_holdout_balances_models_and_settings_without_source_overlap():
    from egocua.judge_validation import select_judge_validation_sample

    rows, groups = [], {}
    for model in ("model-a", "model-b", "model-c"):
        for setting in ("computer_use", "offline_ego_use", "online"):
            for index in range(12):
                run_id = f"{model}-{setting}-{index}"
                rows.append({"run_id": run_id, "run_verdict": "valid_model_failure",
                             "model": model, "setting": setting,
                             "track": "graft" if index % 2 else "clarify", "site": "egomart"})
                groups[run_id] = f"recording-{index}"
    selection = select_judge_validation_sample(rows, groups)
    by_id = {row["run_id"]: row for row in rows}
    cells = {(by_id[run_id]["model"], by_id[run_id]["setting"])
             for run_id in selection["validation_run_ids"]}
    assert len(cells) == 9
    assert {by_id[r]["model"] for r in selection["calibration_run_ids"]} == {"model-a", "model-b", "model-c"}
    assert {groups[r] for r in selection["calibration_run_ids"]}.isdisjoint(
        {groups[r] for r in selection["validation_run_ids"]}
    )


def test_judge_validation_uses_first_blind_rating_not_later_correction(tmp_path: Path):
    from egocua.judge_validation import begin_judge_validation, judge_validation_result, save_judge_validation_assessment
    from egocua.failure_attribution import FAILURE_EVIDENCE_PATH
    from egocua.workspace import read_json_strict

    rows, judgments, _ = _judge_validation_fixture(tmp_path)
    begin_judge_validation(tmp_path)
    row = rows[10]
    assessment = judgments[row["run_id"]]
    save_judge_validation_assessment(tmp_path, row, assessment)
    evidence = read_json_strict(tmp_path / FAILURE_EVIDENCE_PATH)
    predicted = assessment["judgment"]["stage"]
    original = {"run_id": row["run_id"], "rater_id": "human", "pulse_stage": "P" if predicted != "P" else "L"}
    evidence["ratings"] = [{**original, "pulse_stage": predicted, "review_history": [original]}]
    result = judge_validation_result(evidence, rows, tmp_path)
    assert result["rating_basis"] == "first_independent_submission"
    assert result["automatic_vs_human_agreement"] == 0


def test_historical_context_preserves_public_invalid_action_response_and_protocol(tmp_path, monkeypatch):
    import egocua.failure_attribution as attribution
    from tests.helpers import complete_run_record, task

    definition = task("graft")
    record = complete_run_record({
        "task_id": definition["id"], "track": definition["track"],
        "task_snapshot": definition, "meta": {"run_id": "public-output", "model": "model-a"},
        "condition": CONDITIONS["online"].as_dict(),
        "trace": {"agent_settings": {"system_prompt": "Return one valid action JSON.", "model": "model-a"},
                  "llm_requests": [{"step": 1, "response_finish_reason": "length", "input": "duplicate input"}],
                  "history": [{"metadata": {"step_number": 1}, "model_output": None,
                               "response_text": '{"action": "click"', "result": [{"error": "Invalid JSON"}]}]},
        "actions": [], "initial_context": {"instruction": "Save the item.", "observations": []},
        "live_state": {"selected_entity_ids": [], "fields": {}},
    })
    monkeypatch.setattr(attribution, "load_run_record", lambda path: record)
    row = {"run_id": "public-output", "task_id": definition["id"], "run_artifact": "artifacts/run"}
    context = historical_review_context(row, tmp_path)
    assert context["web_observations"][0]["response_text"] == '{"action": "click"'
    assert context["agent_protocol"] == {"system_prompt": "Return one valid action JSON."}
    assert context["request_outcomes"] == [{"step": 1, "response_finish_reason": "length"}]
