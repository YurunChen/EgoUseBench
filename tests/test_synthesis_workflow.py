from __future__ import annotations

import json
import hashlib
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from tests.helpers import unchanged_task_owners as unchanged_task_owners

from egocua.cli import main
from egocua.confidence import (
    resolve_human_review,
    route_assessment,
)
from egocua.confidence_packets import (
    CurrentReviewMaterials,
    answer_boundary,
    build_review_packet,
    review_materials_sha256,
)
from egocua.confidence_session import (
    ConfidenceSessionError,
    record_assessment,
    register_answer_boundary_baselines,
    start_protocol_recheck,
    start_review,
)
from egocua.construct_validity import CONFIDENCE_POLICY_VERSION
from egocua.experiment import load_experiment_plan
from egocua.protocol import RELEASE_DELIVERY_DESCRIPTIONS, first_native_commitment_step
from egocua.environment import load_site_data
from egocua.run_ledger import canonical_sha256, task_sha256, task_site_execution_sha256
from egocua.synthesis_workflow import SynthesisWorkflow, SynthesisWorkflowError
from egocua.synthesis_reports import REPORTS
from egocua.sources import find_record, load_dossier
from tests.helpers import (
    coding_agent_assessment,
    materialize_assets,
    plan_payload,
    products,
    task,
)
from egocua.confidence_packets import build_packets
from egocua.confidence_report import build_confidence_report

pytestmark = pytest.mark.usefixtures("unchanged_task_owners")


def _workspace(tmp_path: Path) -> tuple[dict, str]:
    definition = task("graft")
    task_id = str(definition["id"])
    materialize_assets(definition, products("graft"), tmp_path)
    shutil.rmtree(tmp_path / "artifacts/reports")
    shutil.rmtree(tmp_path / "data/reviews/submissions")
    selection_path = tmp_path / "data/selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "source_manifests": ["data/source/graft_screening_manifest.json"],
                "task_set": {
                    "graft": [task_id],
                    "refresh": [],
                    "clarify": [],
                },
            }
        )
    )
    return definition, task_sha256(definition)


def _report(tmp_path: Path, name: str, rows: dict | list) -> None:
    path = tmp_path / f"artifacts/reports/{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    schema_versions = {
        "task-acceptance": REPORTS["acceptance"].schema_version,
        "task-smoke": REPORTS["smoke"].schema_version,
        "role-recoverability": 3,
        "task-confidence": 4,
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_versions.get(name, 1),
                "tasks": rows,
            }
        )
    )


def _deterministic_reports(tmp_path: Path, definition: dict, digest: str) -> None:
    task_id = str(definition["id"])
    site_digest = task_site_execution_sha256(
        definition, load_site_data(tmp_path, str(definition["site"]))
    )
    _report(
        tmp_path,
        "task-acceptance",
        {
            task_id: {
                "task_sha256": digest,
                "task_site_execution_sha256": site_digest,
                "search_evidence_sha256": None,
                "status": "candidate_ready",
            }
        },
    )
    _report(
        tmp_path,
        "task-smoke",
        {
            task_id: {
                "task_sha256": digest,
                "task_site_execution_sha256": site_digest,
                "status": "pass",
            }
        },
    )
    _report(
        tmp_path,
        "role-recoverability",
        {
            task_id: {
                "task_sha256": digest,
                "status": "pass",
            }
        },
    )


def _high_confidence(digest: str) -> dict:
    return route_assessment(digest, coding_agent_assessment(task("graft")))


@pytest.mark.parametrize("current_failure", [True, False])
@pytest.mark.parametrize("failure_status", ["fail", "failed"])
def test_recorded_failure_is_not_cleared_by_metadata_or_missing_audit(
    tmp_path: Path, current_failure: bool, failure_status: str
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    acceptance_path = tmp_path / "artifacts/reports/task-acceptance.json"
    report = json.loads(acceptance_path.read_text())
    row = report["tasks"][definition["id"]]
    row["status"] = failure_status
    if not current_failure:
        row["task_sha256"] = "older-task"
    acceptance_path.write_text(json.dumps(report))
    (tmp_path / "artifacts/reports/task-smoke.json").unlink()

    status = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert status["status"] == "needs_repair"


def _low_confidence(digest: str) -> dict:
    return route_assessment(digest, coding_agent_assessment(task("graft"), "low"))


def test_status_and_release_report_route_confidence_identically(tmp_path):
    from egocua.confidence_report import build_confidence_report

    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)

    status = SynthesisWorkflow(tmp_path).status()
    report = build_confidence_report(tmp_path)

    assert status["review_evidence"] == report["tasks"]


def test_status_resolves_review_evidence_once_without_reading_confidence_cache(
    tmp_path, monkeypatch
):
    import egocua.synthesis_workflow as synthesis_workflow

    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    cache = tmp_path / REPORTS["confidence"].path
    cache.write_text("invalid unused cache")
    resolve = synthesis_workflow.task_confidence
    calls = []

    def counted_resolve(task, **kwargs):
        calls.append(task["id"])
        return resolve(task, **kwargs)

    monkeypatch.setattr(synthesis_workflow, "task_confidence", counted_resolve)
    result = SynthesisWorkflow(tmp_path).status()

    assert result["tasks"][definition["id"]]["status"] == "ready"
    assert calls == [definition["id"]]
    assert cache.read_text() == "invalid unused cache"


def test_public_status_has_one_three_value_contract(tmp_path: Path) -> None:
    definition, _ = _workspace(tmp_path)

    report = SynthesisWorkflow(tmp_path).status()

    assert report["summary"] == {
        "tasks": 1,
        "status_counts": {"needs_repair": 0, "needs_review": 1, "ready": 0},
        "ready": 0,
        "human_review": 0,
        "in_progress": 1,
        "remaining": 1,
        "next_actions": {"run_deterministic_audit": 1},
        "pending_actions": {"run_deterministic_audit": 1},
    }
    assert report["tasks"][definition["id"]] == {
        "task_id": definition["id"],
        "status": "needs_review",
        "status_label": "处理中",
        "next_action": "run_deterministic_audit",
        "remaining_actions": ["run_deterministic_audit"],
        "reason": "deterministic evidence is missing: acceptance, role_recoverability, smoke",
    }


def test_reviewed_stream_window_clears_only_its_remaining_action(tmp_path: Path) -> None:
    definition, _ = _workspace(tmp_path)
    dossier_path, dossier = load_dossier(definition, root=tmp_path)
    record = find_record(dossier, definition["source"]["task_id"], context="fixture")
    record["stream"] = {"session_seconds": [0.0, 60.0], "update_onset_seconds": 20.0}
    dossier_path.write_text(json.dumps(dossier))
    unreviewed = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    record["stream"]["review_evidence"] = "reviews/window.json"
    dossier_path.write_text(json.dumps(dossier))
    reviewed = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert "review_stream_window" in unreviewed["remaining_actions"]
    assert "review_stream_window" not in reviewed["remaining_actions"]
    assert reviewed["status"] == unreviewed["status"]
    assert reviewed["next_action"] == unreviewed["next_action"]


def test_an_unreviewed_stream_window_holds_an_otherwise_ready_task(tmp_path, monkeypatch):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    from egocua.development_gate import DevelopmentDecision

    monkeypatch.setattr(
        "egocua.synthesis_workflow.DevelopmentEvidence.decision",
        lambda *args: DevelopmentDecision(None, "verified"),
    )
    dossier_path, dossier = load_dossier(definition, root=tmp_path)
    find_record(dossier, definition["source"]["task_id"], context="fixture").pop("stream")
    dossier_path.write_text(json.dumps(dossier))

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["status"] == "needs_review"
    assert row["next_action"] == "review_stream_window"


def test_status_uses_one_source_snapshot_for_search_and_quality_checks(
    tmp_path: Path, monkeypatch
) -> None:
    import egocua.synthesis_workflow as workflow_module

    definition, _ = _workspace(tmp_path)
    reads = []
    read_source = workflow_module.CurrentConstruction.source

    def counted_source(construction, task):
        reads.append(task["id"])
        return read_source(construction, task)

    monkeypatch.setattr(workflow_module.CurrentConstruction, "source", counted_source)

    SynthesisWorkflow(tmp_path).status([definition["id"]])

    assert reads == [definition["id"]]


def test_malformed_audit_report_raises_instead_of_becoming_pending(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    _report(tmp_path, "task-smoke", [])

    with pytest.raises(SynthesisWorkflowError, match="tasks must be an object"):
        SynthesisWorkflow(tmp_path).status()


def test_malformed_task_evidence_raises_instead_of_becoming_missing(
    tmp_path: Path,
) -> None:
    definition, _ = _workspace(tmp_path)
    _report(tmp_path, "task-smoke", {definition["id"]: []})

    with pytest.raises(SynthesisWorkflowError, match="task evidence must be an object"):
        SynthesisWorkflow(tmp_path).status()


def test_duplicate_active_assessments_cannot_overwrite_the_task_evidence(
    tmp_path: Path,
) -> None:
    definition, _ = _workspace(tmp_path)
    path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    completed = coding_agent_assessment(definition)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"schema_version": 4, "results": [completed, deepcopy(completed)]})
    )

    with pytest.raises(SynthesisWorkflowError, match="duplicate task IDs"):
        SynthesisWorkflow(tmp_path).status()


@pytest.mark.parametrize("retain_history", [True, False])
@pytest.mark.parametrize("materials_unchanged", [True, False])
def test_status_uses_verified_completed_history_without_reopening_review(
    tmp_path: Path, retain_history: bool, materials_unchanged: bool
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    completed = coding_agent_assessment(definition)
    completed["materials_sha256"] = CurrentReviewMaterials(tmp_path).packet(definition)[
        "materials_sha256"
    ]
    if not materials_unchanged:
        completed["materials_sha256"] = "unverified-historical-materials"
    active = {
        **completed,
        "status": "pending",
        "pending_reason": "review_materials_changed",
        "assessments": [],
        "history": [completed] if retain_history else [],
    }
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    assessment_path.write_text(json.dumps({"schema_version": 4, "results": [active]}))
    confidence = _high_confidence(digest)
    confidence.update(
        agent_assessments_sha256=canonical_sha256([]),
        source_assessments_sha256=canonical_sha256(completed["assessments"]),
    )
    _report(tmp_path, "task-confidence", {definition["id"]: confidence})
    original = assessment_path.read_bytes()

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["status"] == (
        "ready" if retain_history and materials_unchanged else "needs_review"
    )
    if not retain_history:
        assert "未找到已完成的两阶段具体评语" in row["reason"]
    elif not materials_unchanged:
        assert "尚无依据证明其审核材料与当前材料等价" in row["reason"]
    assert assessment_path.read_bytes() == original


@pytest.mark.parametrize(
    ("current_review", "expected_status"),
    [("admit", "ready"), ("reject", "needs_repair"), ("deleted", "ready")],
)
def test_history_reuse_reads_current_human_review_after_cached_rejection(
    tmp_path: Path, monkeypatch, current_review: str, expected_status: str
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = definition["id"]
    _deterministic_reports(tmp_path, definition, digest)
    completed = coding_agent_assessment(definition)
    completed["materials_sha256"] = CurrentReviewMaterials(tmp_path).packet(definition)[
        "materials_sha256"
    ]
    active = {
        **completed,
        "status": "pending",
        "pending_reason": "review_materials_changed",
        "assessments": [],
        "history": [completed],
    }
    assessment_path = tmp_path / REPORTS["assessment"].path
    assessment_path.write_text(json.dumps({"schema_version": 4, "results": [active]}))
    confidence = _high_confidence(digest)
    confidence.update(
        agent_assessments_sha256=canonical_sha256([]),
        source_assessments_sha256=canonical_sha256(completed["assessments"]),
    )
    cached = resolve_human_review(
        confidence, {"status": "reject", "purpose": "quality_control"}
    )
    _report(tmp_path, "task-confidence", {task_id: cached})
    if current_review != "deleted":
        submission_dir = tmp_path / "data/reviews/submissions" / task_id
        submission_dir.mkdir(parents=True)
        (submission_dir / "reviewer-a.json").write_text("{}")
    monkeypatch.setattr(
        "egocua.confidence_report.audit_construction_reviews",
        lambda **_kwargs: {
            "tasks": {task_id: {"status": current_review, "purpose": "quality_control"}}
        },
    )

    workflow = SynthesisWorkflow(tmp_path)
    report = workflow.status()
    row = report["tasks"][task_id]
    assert row["status"] == expected_status
    if expected_status == "ready":
        released = workflow.release_evidence([task_id])
        assert (
            released["confidence"]["current_review_evidence"]
            == report["review_evidence"]
        )
        assert released["confidence"]["routes"] == {
            "automatic_construction_admission": 1
        }
    else:
        with pytest.raises(SynthesisWorkflowError, match="not release-ready"):
            workflow.release_evidence([task_id])
    cached_report = json.loads((tmp_path / REPORTS["confidence"].path).read_text())
    assert cached_report["tasks"][task_id] == cached


@pytest.mark.parametrize("ready", [False, True])
def test_development_admission_uses_current_task_state(
    tmp_path: Path, ready: bool
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    if ready:
        _completed_review_reports(tmp_path, definition)
        SynthesisWorkflow(tmp_path).require_development_inputs([definition["id"]])
    else:
        with pytest.raises(
            SynthesisWorkflowError, match="prerequisites are incomplete"
        ):
            SynthesisWorkflow(tmp_path).require_development_inputs([definition["id"]])


@pytest.mark.parametrize("change", ["explicit_invalidation", "in_progress"])
def test_cached_history_cannot_override_active_review_invalidation(
    tmp_path: Path, change: str
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = definition["id"]
    _deterministic_reports(tmp_path, definition, digest)
    completed = _completed_review_reports(tmp_path, definition)["results"][0]
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(assessment_path.read_text())
    active = report["results"][0]
    active.update(
        status="pending",
        pending_reason="review_materials_changed",
        assessments=[],
        history=[deepcopy(completed)],
    )
    assessment_path.write_text(json.dumps(report))
    confidence_path = tmp_path / "artifacts/reports/task-confidence.json"
    confidence_path.write_text(json.dumps(build_confidence_report(tmp_path)))
    workflow = SynthesisWorkflow(tmp_path)
    assert workflow.status([task_id])["tasks"][task_id]["status"] == "ready"

    if change == "explicit_invalidation":
        active["pending_reason"] = "user_requested_recheck"
    else:
        active["status"] = "in_progress"
    assessment_path.write_text(json.dumps(report))
    original_confidence = confidence_path.read_bytes()
    original_assessment = assessment_path.read_bytes()

    row = workflow.status([task_id])["tasks"][task_id]

    assert row["status"] == "needs_review"
    assert row["next_action"] == "coding_agent_assess"
    assert confidence_path.read_bytes() == original_confidence
    assert assessment_path.read_bytes() == original_assessment


@pytest.mark.parametrize("change", ["explicit_invalidation", "in_progress"])
def test_cached_protocol_recheck_does_not_override_a_reopened_review(
    tmp_path: Path, change: str
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = definition["id"]
    _deterministic_reports(tmp_path, definition, digest)
    completed = _completed_review_reports(tmp_path, definition)["results"][0]
    completed["answer_boundary"]["final_goal"]["acceptable_states"] = []
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(assessment_path.read_text())
    active = report["results"][0]
    active.update(
        status="pending",
        pending_reason="review_materials_changed",
        assessments=[],
        history=[completed],
    )
    assessment_path.write_text(json.dumps(report))
    confidence_path = tmp_path / "artifacts/reports/task-confidence.json"
    confidence_path.write_text(json.dumps(build_confidence_report(tmp_path)))
    workflow = SynthesisWorkflow(tmp_path)
    assert (
        workflow.status([task_id])["tasks"][task_id]["next_action"]
        == "review_protocol_counterexamples"
    )

    if change == "explicit_invalidation":
        active["pending_reason"] = "user_requested_recheck"
    else:
        active["status"] = "in_progress"
    assessment_path.write_text(json.dumps(report))

    row = workflow.status([task_id])["tasks"][task_id]

    assert row["status"] == "needs_review"
    assert row["next_action"] == "coding_agent_assess"
    assert "recheck_passes" not in row


def _assessment_report(
    tmp_path: Path, definition: dict, confidence: str = "high"
) -> None:
    path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "results": [
                    {
                        **coding_agent_assessment(definition, confidence),
                        "materials_sha256": CurrentReviewMaterials(tmp_path).packet(
                            definition
                        )["materials_sha256"],
                    }
                ],
            }
        )
    )


def _completed_review_reports(tmp_path: Path, definition: dict) -> dict:
    plan = tmp_path / "configs/pilot.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(plan_payload()))
    index = build_packets(tmp_path)
    report = start_review(tmp_path)
    result = coding_agent_assessment(definition)
    result["materials_sha256"] = index["tasks"][definition["id"]]["materials_sha256"]
    report["results"] = [result]
    path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    path.write_text(json.dumps(report))
    confidence = build_confidence_report(tmp_path)
    (tmp_path / "artifacts/reports/task-confidence.json").write_text(
        json.dumps(confidence)
    )
    return report


@pytest.mark.parametrize(
    ("action", "status", "next_action"),
    [
        ("repair", "needs_repair", "repair_task"),
        ("collect_evidence", "needs_review", "resolve_source_review"),
        ("human_review", "needs_review", "human_review"),
    ],
)
def test_named_follow_up_controls_the_single_lifecycle(
    tmp_path, action, status, next_action
):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    report = _completed_review_reports(tmp_path, definition)
    second = report["results"][0]["assessments"][1]["verdict"]
    second["reviewer"] = {"agent_id": "current-reviewer", "context_id": "targeted"}
    second["confidence"] = "low"
    second["follow_up"] = [
        {
            "action": action,
            "evidence": ["evidence.answer_boundary.0"],
            "reason": "The named listing needs its package contents checked.",
        }
    ]
    (tmp_path / REPORTS["assessment"].path).write_text(json.dumps(report))

    result = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert (result["status"], result["next_action"]) == (status, next_action)
    confidence = build_confidence_report(tmp_path)["tasks"][definition["id"]]
    assert confidence["human_review_required"] is (action == "human_review")


def test_explicit_targeted_second_recheck_archives_the_old_second(tmp_path):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    report = _completed_review_reports(tmp_path, definition)
    task_id = definition["id"]
    original = deepcopy(report["results"][0]["assessments"])
    with pytest.raises(ConfidenceSessionError, match="unchanged blind-review"):
        start_protocol_recheck(task_id, root=tmp_path)
    start_protocol_recheck(
        task_id,
        root=tmp_path,
        reason="Only classify the named remaining evidence gap; retain the original scene review.",
    )
    second = deepcopy(original[1]["verdict"])
    second["reviewer"] = {"agent_id": "targeted-reviewer", "context_id": "current"}
    second["follow_up"] = [
        {
            "action": "collect_evidence",
            "evidence": ["evidence.answer_boundary.0"],
            "reason": "The named listing's package contents are not disclosed.",
        }
    ]

    row = record_assessment(task_id, "adversarial_falsification", second, root=tmp_path)

    assert row["assessments"][0] == original[0]
    assert row["assessments"][1]["verdict"] == second
    assert any(history["assessments"] == original for history in row["history"])
    assert row["recheck_reason"].startswith("Only classify")


def test_partial_human_reuse_is_visible_without_clearing_new_questions(
    tmp_path, monkeypatch
):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    report = _completed_review_reports(tmp_path, definition)
    task_id = definition["id"]
    second = report["results"][0]["assessments"][1]["verdict"]
    second["confidence"] = "low"
    second["uncertainty_reasons"] = ["Only the new listing's colour is ambiguous."]
    (tmp_path / REPORTS["assessment"].path).write_text(json.dumps(report))
    directory = tmp_path / "data/reviews/submissions" / task_id
    directory.mkdir(parents=True)
    (directory / "reviewer-a.json").write_text("{}")
    reuse = [
        {
            "retained_conclusions": ["The old scene referent remains valid."],
            "affected_conclusions": ["Only the new listing's colour is ambiguous."],
        }
    ]

    def audit(**kwargs):
        return {"tasks": {task_id: {"status": "pending", "human_review_reuse": reuse}}}

    monkeypatch.setattr("egocua.reviews.audit_construction_reviews", audit)
    monkeypatch.setattr("egocua.confidence_report.audit_construction_reviews", audit)

    result = SynthesisWorkflow(tmp_path).status()

    row = result["tasks"][task_id]
    assert row["next_action"] == "human_review"
    assert (
        row["reason"]
        == "旧人工结论已保留；仅受影响的问题待判：Only the new listing's colour is ambiguous."
    )
    evidence = result["review_evidence"][task_id]
    assert evidence["human_review_required"] is True
    assert (
        build_confidence_report(tmp_path)["tasks"][task_id]["human_review_reuse"]
        == reuse
    )


@pytest.mark.parametrize("audit_state", ["missing", "stale"])
@pytest.mark.parametrize("current_materials", [True, False])
def test_current_semantic_defect_precedes_report_refresh(
    tmp_path: Path, audit_state: str, current_materials: bool
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    report = _completed_review_reports(tmp_path, definition)
    verdict = report["results"][0]["assessments"][1]["verdict"]
    verdict.update(decision="repair", confidence="high")
    verdict["gates"]["answer_boundary"] = "fail"
    verdict["counterexample_checks"]["undeclared_equivalent"] = "fail"
    verdict["evidence"]["answer_boundary"] = [
        "The public listing supports an equivalent missing from the accepted states."
    ]
    verdict["issues"] = [
        "A publicly supported equivalent is missing from the accepted states."
    ]
    (tmp_path / REPORTS["assessment"].path).write_text(json.dumps(report))
    smoke_path = tmp_path / REPORTS["smoke"].path
    if audit_state == "missing":
        smoke_path.unlink()
    else:
        smoke = json.loads(smoke_path.read_text())
        smoke["tasks"][definition["id"]]["task_sha256"] = "previous-task"
        smoke_path.write_text(json.dumps(smoke))
    if not current_materials:
        definition["instructions"]["a"] += " for a different use"
        path = next((tmp_path / "data/tasks").glob("*/*.json"))
        path.write_text(json.dumps(definition))

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["next_action"] == (
        "repair_task"
        if current_materials
        else "run_deterministic_audit"
        if audit_state == "missing"
        else "coding_agent_assess"
    )


@pytest.mark.parametrize(
    "scenario",
    [
        ("missing", "completed"),
        ("missing", "history_only"),
        ("stale_assessment", "completed"),
        ("stale_assessment", "history_only"),
        ("metadata_only", "completed"),
        ("metadata_only", "history_only"),
    ],
)
def test_next_refreshes_derived_confidence_without_repeating_completed_review(
    tmp_path: Path, monkeypatch, scenario: tuple[str, str]
) -> None:
    cached_report, active_state = scenario
    definition, digest = _workspace(tmp_path)
    task_id = definition["id"]
    _deterministic_reports(tmp_path, definition, digest)
    original = _completed_review_reports(tmp_path, definition)["results"][0]
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(assessment_path.read_text())
    if active_state == "history_only":
        report["results"][0].update(
            status="pending",
            assessments=[],
            pending_reason="review_materials_changed",
            history=[deepcopy(original)],
        )
        assessment_path.write_text(json.dumps(report))
    original_active = deepcopy(report["results"][0])
    confidence_path = tmp_path / "artifacts/reports/task-confidence.json"
    if cached_report == "missing":
        confidence_path.unlink()
    elif cached_report == "metadata_only":
        definition["source"]["dossier_sha256"] = "c" * 64
        (tmp_path / f"data/tasks/graft/{task_id}.json").write_text(
            json.dumps(definition)
        )
        _deterministic_reports(tmp_path, definition, task_sha256(definition))
    else:
        confidence = json.loads(confidence_path.read_text())
        confidence["tasks"][task_id]["agent_assessments_sha256"] = "older-assessment"
        confidence_path.write_text(json.dumps(confidence))
    before = confidence_path.read_bytes() if confidence_path.exists() else None

    workflow = SynthesisWorkflow(tmp_path)
    pending = workflow.status([task_id])["tasks"][task_id]

    assert pending["status"] == "ready"
    assert pending["next_action"] == "build_release"
    assert workflow.status([task_id])["review_evidence"][task_id][
        "task_sha256"
    ] == task_sha256(definition)
    assert (
        confidence_path.read_bytes() if confidence_path.exists() else None
    ) == before

    def unexpected_review_open(*args, **kwargs):
        raise AssertionError("completed semantic review must not reopen")

    monkeypatch.setattr("egocua.synthesis_workflow.open_pass", unexpected_review_open)
    work_item = workflow.next_work_item(task_id)
    current = json.loads(assessment_path.read_text())["results"][0]

    assert work_item["kind"] == "build_release"
    assert current["assessments"] == original_active["assessments"]
    assert current.get("history", []) == original_active.get("history", [])


def test_lifecycle_never_skips_pending_coding_agent_review(
    tmp_path: Path, monkeypatch
) -> None:
    definition, digest = _workspace(tmp_path)

    def unexpected_packet(*args, **kwargs):
        raise AssertionError("no completed review requires no material comparison")

    monkeypatch.setattr(CurrentReviewMaterials, "packet", unexpected_packet)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _report(
        tmp_path,
        "task-confidence",
        {
            task_id: {
                "task_sha256": digest,
                "policy_version": CONFIDENCE_POLICY_VERSION,
                "level": "unassessed",
                "route": "coding_agent_retry_pending",
                "human_review_required": False,
            }
        },
    )
    row = SynthesisWorkflow(tmp_path).status()["tasks"][task_id]

    assert row["status"] == "needs_review"
    assert row["next_action"] == "coding_agent_assess"


def _prior_review_fixture(tmp_path: Path) -> tuple[dict, Path]:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    build_packets(tmp_path)
    start_review(tmp_path)
    source = tmp_path / "prior-review.json"
    source.write_text(
        json.dumps({"tasks": [{"task_id": definition["id"], "fact": "observed pan"}]})
    )
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(report_path.read_text())
    report["results"][0]["prior_review_evidence"] = [
        {
            "path": "prior-review.json",
            "pointer": "/tasks/0",
            "reviewer": None,
            "checked": ["observed pan"],
            "remaining": ["confirm equivalent listings"],
        }
    ]
    report_path.write_text(json.dumps(report))
    return definition, report_path


def test_next_reuses_specific_prior_findings_without_reopening_blind(
    tmp_path, monkeypatch
):
    definition, report_path = _prior_review_fixture(tmp_path)
    original = report_path.read_bytes()

    def unexpected_open(*args, **kwargs):
        raise AssertionError("prior evidence must be read before another blind pass")

    monkeypatch.setattr("egocua.synthesis_workflow.open_pass", unexpected_open)
    workflow = SynthesisWorkflow(tmp_path)
    item = workflow.next_work_item(definition["id"])

    assert item["kind"] == "reuse_review_evidence"
    prior = item["work_item"]["prior_review_evidence"][0]
    assert prior["body"]["fact"] == "observed pan"
    assert prior["remaining"] == ["confirm equivalent listings"]
    assert workflow.status()["tasks"][definition["id"]]["status"] == "needs_review"
    assert report_path.read_bytes() == original


@pytest.mark.parametrize("partial", [False, True])
def test_next_exposes_history_before_restarting_blind(tmp_path, monkeypatch, partial):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(report_path.read_text())
    source = deepcopy(report["results"][0])
    source["materials_sha256"] = "old-materials"
    if partial:
        source.update(status="in_progress", assessments=source["assessments"][:1])
    report["results"][0].update(
        status="pending",
        assessments=[],
        pending_reason="review_materials_changed",
        history=[source],
    )
    report_path.write_text(json.dumps(report))
    original = report_path.read_bytes()

    def unexpected_open(*args, **kwargs):
        raise AssertionError("history must be checked before another blind review")

    monkeypatch.setattr("egocua.synthesis_workflow.open_pass", unexpected_open)
    item = SynthesisWorkflow(tmp_path).next_work_item(definition["id"])

    assert item["kind"] == "reuse_review_evidence"
    history = item["work_item"]["prior_review_evidence"][0]
    assert history["body"] == source
    assert history["pointer"] == "/results/0/history/0"
    assert report_path.read_bytes() == original


def test_changed_materials_expose_completed_current_review_before_new_blind(
    tmp_path, monkeypatch
):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    report_path = tmp_path / REPORTS["assessment"].path
    original = report_path.read_bytes()
    site_path = tmp_path / "sites/egomart/products.json"
    site = json.loads(site_path.read_text())
    site["products"][0]["name"] = "Changed relevant product"
    site_path.write_text(json.dumps(site))

    monkeypatch.setattr(
        "egocua.synthesis_workflow.open_pass",
        lambda *_args, **_kwargs: pytest.fail(
            "must reuse completed review before a new blind solve"
        ),
    )
    item = SynthesisWorkflow(tmp_path).next_work_item(definition["id"])

    assert item["kind"] == "reuse_review_evidence"
    reference = item["work_item"]["prior_review_evidence"][0]
    assert reference["pointer"] == "/results/0"
    assert reference["body"]["status"] == "completed"
    assert report_path.read_bytes() == original


@pytest.mark.parametrize("broken", ["missing_file", "missing_pointer", "wrong_task"])
def test_next_exposes_unreadable_or_unrelated_prior_evidence(tmp_path, broken):
    definition, _ = _prior_review_fixture(tmp_path)
    source = tmp_path / "prior-review.json"
    if broken == "missing_file":
        source.unlink()
    elif broken == "missing_pointer":
        source.write_text(json.dumps({"tasks": []}))
    else:
        source.write_text(json.dumps({"tasks": [{"task_id": "another-task"}]}))

    with pytest.raises((ValueError, FileNotFoundError)):
        SynthesisWorkflow(tmp_path).next_work_item(definition["id"])


def test_prior_findings_do_not_interrupt_started_or_completed_review(tmp_path):
    definition, report_path = _prior_review_fixture(tmp_path)
    report = json.loads(report_path.read_text())
    report["results"][0].update(
        status="in_progress", assessments=[{"pass": "blind_reconstruction"}]
    )
    report_path.write_text(json.dumps(report))
    from egocua.confidence_session import prior_review_evidence

    assert prior_review_evidence(tmp_path, definition["id"]) == []
    references = report["results"][0]["prior_review_evidence"]
    report_path.unlink()
    _completed_review_reports(tmp_path, definition)
    completed = json.loads(report_path.read_text())
    completed["results"][0]["prior_review_evidence"] = references
    report_path.write_text(json.dumps(completed))
    assert (
        SynthesisWorkflow(tmp_path).next_work_item(definition["id"])["kind"]
        == "build_release"
    )


def _source_pending(
    tmp_path: Path,
    definition: dict,
    monkeypatch: pytest.MonkeyPatch,
    checks: list[str],
    *,
    unresolved: list[str] | None = None,
    acceptance_status: str = "needs_source_review",
) -> None:
    path = tmp_path / "artifacts/reports/task-acceptance.json"
    report = json.loads(path.read_text())
    row = report["tasks"][definition["id"]]
    row.update(status=acceptance_status, integrity_pass=True, failed_checks=[])
    path.write_text(json.dumps(report))
    source = {
        "blocking_checks": checks,
        "unresolved": unresolved or [],
        "stream": {"session_seconds": [1.0, 2.0], "review_evidence": "fixture"},
    }
    monkeypatch.setattr(
        "egocua.synthesis_workflow.CurrentConstruction.source", lambda *a, **kw: source
    )


def test_confirmed_source_defect_remains_blocked_when_audit_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _source_pending(
        tmp_path, definition, monkeypatch, ["confirmed_scene_method_mismatch"]
    )
    (tmp_path / "artifacts/reports/role-recoverability.json").unlink()

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert (row["status"], row["next_action"]) == ("needs_repair", "repair_task")


@pytest.mark.parametrize(
    "acceptance_status", ["needs_source_review", "candidate_ready"]
)
def test_source_review_pending_can_enter_coding_agent_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acceptance_status: str,
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _source_pending(
        tmp_path,
        definition,
        monkeypatch,
        ["independent_blind_review_pending", "source_redistribution_review_pending"],
        acceptance_status=acceptance_status,
    )

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["next_action"] == "coding_agent_assess"


@pytest.mark.parametrize(
    "checks,unresolved",
    [
        (["source_redistribution_permission_pending"], []),
        (["independent_blind_review_pending"], []),
        ([], ["The current source evidence still requires a decision."]),
    ],
)
@pytest.mark.parametrize(
    "acceptance_status", ["needs_source_review", "candidate_ready"]
)
def test_completed_review_cannot_release_unresolved_source_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    checks: list[str],
    unresolved: list[str],
    acceptance_status: str,
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    _source_pending(
        tmp_path,
        definition,
        monkeypatch,
        checks,
        unresolved=unresolved,
        acceptance_status=acceptance_status,
    )

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["status"] == "needs_review"
    assert row["next_action"] == "resolve_source_review"


def test_cached_source_pending_does_not_reopen_cleared_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    _source_pending(tmp_path, definition, monkeypatch, [])
    assessment_path = tmp_path / REPORTS["assessment"].path
    original = assessment_path.read_bytes()

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["status"] == "ready"
    assert row["next_action"] == "build_release"
    assert assessment_path.read_bytes() == original


@pytest.mark.parametrize(
    "check",
    [
        "confirmed_source_contradiction_first_opened_spread",
        "catalog_title_repair_pending",
        "unknown_source_blocker",
    ],
)
@pytest.mark.parametrize("completed", [False, True])
def test_source_defects_block_review_and_cannot_be_hidden_by_high_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    check: str,
    completed: bool,
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    if completed:
        _completed_review_reports(tmp_path, definition)
    _source_pending(tmp_path, definition, monkeypatch, [check])

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert (row["status"], row["reason"]) == (
        "needs_repair",
        f"source evidence requires repair: {check}",
    )


@pytest.mark.parametrize(
    "integrity,failed_checks",
    [
        (False, ["production_checker"]),
        (False, []),
        (True, ["source_lineage"]),
    ],
)
def test_source_pending_does_not_bypass_failed_integrity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    integrity: bool,
    failed_checks: list[str],
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _source_pending(
        tmp_path, definition, monkeypatch, ["independent_blind_review_pending"]
    )
    path = tmp_path / "artifacts/reports/task-acceptance.json"
    report = json.loads(path.read_text())
    report["tasks"][definition["id"]].update(
        integrity_pass=integrity,
        failed_checks=failed_checks,
    )
    path.write_text(json.dumps(report))

    row = SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]

    assert row["reason"] == "deterministic evidence failed: acceptance"


def test_source_clearance_preserves_completed_blind_review(tmp_path: Path) -> None:
    definition, _ = _workspace(tmp_path)
    task_id = definition["id"]
    plan = tmp_path / "configs/pilot.json"
    plan.parent.mkdir(parents=True, exist_ok=True)
    plan.write_text(json.dumps(plan_payload()))
    dossier_path = tmp_path / definition["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())

    def update_source(checks: list[str]) -> None:
        record = dossier["tasks"][0]
        record["blocking_checks"] = checks
        dossier_path.write_text(json.dumps(dossier))
        _deterministic_reports(tmp_path, definition, task_sha256(definition))
        if checks:
            path = tmp_path / "artifacts/reports/task-acceptance.json"
            report = json.loads(path.read_text())
            report["tasks"][task_id].update(
                status="needs_source_review",
                integrity_pass=True,
                failed_checks=[],
            )
            path.write_text(json.dumps(report))

    update_source(
        [
            "independent_blind_review_pending",
            "source_redistribution_review_pending",
        ]
    )
    workflow = SynthesisWorkflow(tmp_path)
    first = workflow.next_work_item(task_id)
    assert first["pass"] == "blind_reconstruction"
    for assessment in coding_agent_assessment(definition)["assessments"]:
        workflow.decide(
            task_id,
            {
                **assessment["verdict"],
                "reviewer": {
                    "agent_id": "test-reviewer",
                    "context_id": "test-independent-context",
                },
            },
        )
    blocked = workflow.status()["tasks"][task_id]
    assert blocked["status"] == "needs_review"
    assert blocked["next_action"] == "resolve_source_review"
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    completed = json.loads(assessment_path.read_text())["results"][0]
    old_digest = task_sha256(definition)

    update_source([])
    workflow._prepare_confidence_materials(task_id)
    refreshed = json.loads(assessment_path.read_text())["results"][0]

    assert task_sha256(definition) == old_digest
    assert refreshed["materials_sha256"] == completed["materials_sha256"]
    assert refreshed["assessments"] == completed["assessments"]
    assert workflow.status()["tasks"][task_id]["status"] == "ready"


@pytest.mark.parametrize(
    "current_state", ["missing", "pending", "in_progress", "changed"]
)
def test_status_rejects_cached_admit_after_assessment_changes(
    tmp_path: Path,
    current_state: str,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    assessment_report = _completed_review_reports(tmp_path, definition)
    completed = assessment_report["results"][0]
    if current_state == "changed":
        completed["assessments"][1]["verdict"]["issues"] = ["New source ambiguity"]
        completed["assessments"][1]["verdict"]["confidence"] = "low"
        completed["assessments"][1]["verdict"]["uncertainty_reasons"] = [
            "New source ambiguity"
        ]
    elif current_state != "missing":
        completed["history"] = [deepcopy(completed)]
        completed.update(status=current_state, assessments=[])
    assessment_report["results"] = [] if current_state == "missing" else [completed]
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    assessment_path.write_text(json.dumps(assessment_report))
    confidence_path = tmp_path / "artifacts/reports/task-confidence.json"
    original_confidence = confidence_path.read_bytes()

    row = SynthesisWorkflow(tmp_path).status([task_id])["tasks"][task_id]

    assert row["status"] == "needs_review"
    if current_state == "changed":
        assert row["next_action"] == "human_review"
    assert confidence_path.read_bytes() == original_confidence


def test_preparing_changed_display_packet_keeps_actual_review_evidence(
    tmp_path: Path,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    index_path = tmp_path / "artifacts/reports/confidence-review-packets/index.json"
    index = json.loads(index_path.read_text())
    packet_row = index["tasks"][task_id]
    packet_path = tmp_path / packet_row["path"]
    packet = json.loads(packet_path.read_text())
    packet["materials"]["initial"]["instruction"] = (
        "A corrected release-only instruction"
    )
    packet["materials_sha256"] = review_materials_sha256(packet["materials"])
    packet_path.write_text(json.dumps(packet))
    packet_row["sha256"] = hashlib.sha256(packet_path.read_bytes()).hexdigest()
    packet_row["materials_sha256"] = packet["materials_sha256"]
    index_path.write_text(json.dumps(index))

    SynthesisWorkflow(tmp_path)._prepare_confidence_materials(task_id)

    report = json.loads(
        (tmp_path / "artifacts/reports/task-confidence.json").read_text()
    )
    assert report["tasks"][task_id]["route"] == "automatic_construction_admission"


def test_review_packet_distinguishes_commitment_and_scheduled_release(
    tmp_path: Path,
) -> None:
    definition, _ = _workspace(tmp_path)
    plan = load_experiment_plan(tmp_path / "configs/pilot.json")
    packets = [
        build_review_packet(
            definition,
            root=tmp_path,
            plan=replace(plan, release_trigger=trigger),
            public_site_path="public.json",
            public_site_sha256="a" * 64,
        )
        for trigger in ("timeline_step", "first_native_commitment")
    ]

    assert packets[0]["materials_sha256"] != packets[1]["materials_sha256"]
    assert packets[1]["materials"]["release_trigger"] == "first_native_commitment"
    assert (
        "waiting alone does not release" in packets[1]["materials"]["release_delivery"]
    )
    for packet in packets:
        materials = packet["materials"]
        assert (
            materials["release_delivery"]
            == RELEASE_DELIVERY_DESCRIPTIONS[materials["release_trigger"]]
        )
    assert (
        "even if later undone or submitted"
        in packets[1]["materials"]["release_delivery"]
    )
    assert (
        first_native_commitment_step(
            [
                {"type": "add_to_cart", "step": 1},
                {"type": "remove_from_cart", "step": 1},
                {"type": "submit_cart", "step": 1},
            ]
        )
        == 1
    )


def test_goal_change_reopens_only_adversarial_review(tmp_path: Path) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = definition["id"]
    _deterministic_reports(tmp_path, definition, digest)
    completed = _completed_review_reports(tmp_path, definition)["results"][0]
    alternative = deepcopy(definition["graft"]["initial_goal"]["acceptable_states"][1])
    definition["goal"]["acceptable_states"].append(alternative)
    (tmp_path / f"data/tasks/graft/{task_id}.json").write_text(json.dumps(definition))
    _deterministic_reports(tmp_path, definition, task_sha256(definition))

    workflow = SynthesisWorkflow(tmp_path)
    work = workflow.next_work_item(task_id)

    assert work["pass"] == "adversarial_falsification"
    assert (
        work["work_item"]["expected_goal"] == answer_boundary(definition)["final_goal"]
    )
    assert "expected_goal" not in work["work_item"]["review_packet"]["materials"]
    path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    refreshed = json.loads(path.read_text())["results"][0]
    assert refreshed["materials_sha256"] == completed["materials_sha256"]
    assert refreshed["assessments"] == completed["assessments"][:1]
    assert refreshed["history"][-1]["assessments"] == completed["assessments"]
    assert refreshed["status"] == "in_progress"
    assert workflow.status([task_id])["tasks"][task_id]["status"] != "ready"


def test_confidence_rejects_changed_answers_without_session_refresh(
    tmp_path: Path,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = definition["id"]
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    definition["goal"]["acceptable_states"].append(
        deepcopy(definition["graft"]["initial_goal"]["acceptable_states"][1])
    )
    (tmp_path / f"data/tasks/graft/{task_id}.json").write_text(json.dumps(definition))
    _deterministic_reports(tmp_path, definition, task_sha256(definition))
    build_packets(tmp_path)

    assert (
        build_confidence_report(tmp_path)["tasks"][task_id]["route"]
        == "coding_agent_retry_pending"
    )


def test_explicit_answer_boundary_baseline_preserves_sealed_reviews(
    tmp_path: Path,
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    report = _completed_review_reports(tmp_path, definition)
    previous = report["results"][0]
    previous.pop("answer_boundary")
    path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    path.write_text(json.dumps(report))

    register_answer_boundary_baselines(
        [
            {
                "task_id": definition["id"],
                "source_assessments": previous["assessments"],
                "source_materials_sha256": previous["materials_sha256"],
                "answer_boundary": answer_boundary(definition),
            }
        ],
        root=tmp_path,
    )

    refreshed = json.loads(path.read_text())["results"][0]
    assert refreshed["assessments"] == previous["assessments"]
    assert refreshed["answer_boundary"] == answer_boundary(definition)
    assert (
        refreshed["answer_boundary_registration"]["purpose"]
        == "forward_change_tracking"
    )
    assert refreshed["history"][-1]["assessments"] == previous["assessments"]


def test_other_task_assessment_changes_do_not_stale_current_admission(
    tmp_path: Path,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    assessment_report = _completed_review_reports(tmp_path, definition)
    assessment_report["results"].append(
        {
            "task_id": "unrelated-review",
            "status": "pending",
            "assessments": [],
        }
    )
    path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    path.write_text(json.dumps(assessment_report))

    row = SynthesisWorkflow(tmp_path).status([task_id])["tasks"][task_id]

    assert row["status"] == "ready"


@pytest.mark.parametrize(
    "change",
    [
        "site_visible",
        "site_unrelated",
        "plan_format",
        "plan_runtime",
        "plan_release",
        "frame",
    ],
)
def test_status_and_report_check_actual_task_local_review_materials(
    tmp_path, monkeypatch, change
):
    from egocua.development_gate import DevelopmentDecision

    definition, _ = _workspace(tmp_path)
    task_id = definition["id"]
    catalog_path = tmp_path / "sites/egomart/products.json"
    catalog = json.loads(catalog_path.read_text())
    definition.setdefault("truth", {})["site_scope_entity_ids"] = [
        item["item_id"] for item in catalog["products"][:-1]
    ]
    (tmp_path / f"data/tasks/graft/{task_id}.json").write_text(json.dumps(definition))
    _deterministic_reports(tmp_path, definition, task_sha256(definition))
    _completed_review_reports(tmp_path, definition)
    monkeypatch.setattr(
        "egocua.synthesis_workflow.DevelopmentEvidence.decision",
        lambda *args: DevelopmentDecision(None, "isolated semantic gate test"),
    )
    assessment_path = tmp_path / REPORTS["assessment"].path
    confidence_path = tmp_path / REPORTS["confidence"].path
    originals = [path.read_bytes() for path in (assessment_path, confidence_path)]
    if change.startswith("site_"):
        catalog["products"][0 if change == "site_visible" else -1]["name"] = (
            "Changed tutorial content"
        )
        catalog_path.write_text(json.dumps(catalog))
    elif change == "frame":
        (tmp_path / definition["visual_packets"][0]["frame_paths"][0]).write_bytes(
            b"changed actual frame"
        )
    else:
        plan_path = tmp_path / "configs/pilot.json"
        plan = json.loads(plan_path.read_text())
        if change == "plan_release":
            plan["runtime"]["release_trigger"] = "first_native_commitment"
        elif change == "plan_runtime":
            plan["runtime"]["model"] = "another-model"
        plan_path.write_text(json.dumps(plan, indent=4, sort_keys=True))
    _deterministic_reports(tmp_path, definition, task_sha256(definition))

    changed = change in {"site_visible", "plan_release", "frame"}
    row = SynthesisWorkflow(tmp_path).status([task_id])["tasks"][task_id]
    report = build_confidence_report(tmp_path)

    assert row["next_action"] == ("coding_agent_assess" if changed else "build_release")
    assert report["tasks"][task_id]["level"] == ("unassessed" if changed else "high")
    assert [
        path.read_bytes() for path in (assessment_path, confidence_path)
    ] == originals
    assert (
        CurrentReviewMaterials(tmp_path).stored_packet_matches(definition)
        is not changed
    )


@pytest.mark.parametrize("missing", ["materials_sha256", "answer_boundary"])
def test_missing_review_binding_is_not_inferred_from_cached_task(tmp_path, missing):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    report = _completed_review_reports(tmp_path, definition)
    report["results"][0].pop(missing)
    path = tmp_path / REPORTS["assessment"].path
    path.write_text(json.dumps(report))
    original = path.read_bytes()

    assert (
        build_confidence_report(tmp_path)["tasks"][definition["id"]]["level"]
        == "unassessed"
    )
    assert (
        SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]["next_action"]
        == "coding_agent_assess"
    )
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "report_name",
    [
        "task-acceptance",
        "task-smoke",
        "role-recoverability",
        "coding-agent-task-assessments",
    ],
)
def test_report_version_alone_preserves_completed_review(
    tmp_path: Path, report_name: str
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    path = tmp_path / f"artifacts/reports/{report_name}.json"
    report = json.loads(path.read_text())
    report["schema_version"] = 999
    path.write_text(json.dumps(report))

    assert (
        SynthesisWorkflow(tmp_path).status()["tasks"][definition["id"]]["status"]
        == "ready"
    )


@pytest.mark.parametrize(
    "report_name", ["task-acceptance", "task-smoke", "role-recoverability"]
)
@pytest.mark.parametrize(
    "field", ["task_sha256", "task_site_execution_sha256", "search_evidence_sha256"]
)
def test_audit_fingerprint_alone_does_not_reopen_review(
    tmp_path: Path, report_name: str, field: str
) -> None:
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    path = tmp_path / f"artifacts/reports/{report_name}.json"
    report = json.loads(path.read_text())
    report["tasks"][definition["id"]][field] = "different-record-metadata"
    path.write_text(json.dumps(report))
    original = path.read_bytes()

    workflow = SynthesisWorkflow(tmp_path)
    for _ in range(2):
        assert workflow.status()["tasks"][definition["id"]]["status"] == "ready"
    assert path.read_bytes() == original


def test_lifecycle_releases_after_task_quality_gates(tmp_path: Path) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition)
    _report(tmp_path, "task-confidence", {task_id: _high_confidence(digest)})
    workflow = SynthesisWorkflow(tmp_path)
    assert workflow.next_work_item(task_id) == {
        "kind": "build_release",
        "task_id": task_id,
        "status": "ready",
        "reason": "all task-quality gates are current",
    }


def test_model_reports_do_not_change_task_quality_status(tmp_path: Path) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition)
    _report(tmp_path, "task-confidence", {task_id: _high_confidence(digest)})
    _report(
        tmp_path,
        "task-qualification",
        {
            task_id: {
                "task_sha256": digest,
                "development_screen": {"status": "repair_required"},
            }
        },
    )

    row = SynthesisWorkflow(tmp_path).status()["tasks"][task_id]

    assert row["status"] == "ready"


def test_status_reads_submitted_human_admit_without_mutating_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition, "low")
    confidence_path = tmp_path / "artifacts/reports/task-confidence.json"
    _report(tmp_path, "task-confidence", {task_id: _low_confidence(digest)})
    original = confidence_path.read_bytes()
    submission_dir = tmp_path / "data/reviews/submissions" / task_id
    submission_dir.mkdir(parents=True)
    (submission_dir / "reviewer-a.json").write_text("{}")

    monkeypatch.setattr(
        "egocua.confidence_report.audit_construction_reviews",
        lambda **_kwargs: {"tasks": {task_id: {"status": "admit"}}},
    )

    row = SynthesisWorkflow(tmp_path).status()["tasks"][task_id]

    assert row["status"] == "ready"
    assert confidence_path.read_bytes() == original


def test_status_routes_submitted_human_reject_to_task_repair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition, "low")
    _report(tmp_path, "task-confidence", {task_id: _low_confidence(digest)})
    submission_dir = tmp_path / "data/reviews/submissions" / task_id
    submission_dir.mkdir(parents=True)
    (submission_dir / "reviewer-a.json").write_text("{}")
    monkeypatch.setattr(
        "egocua.confidence_report.audit_construction_reviews",
        lambda **_kwargs: {"tasks": {task_id: {"status": "reject"}}},
    )

    row = SynthesisWorkflow(tmp_path).status()["tasks"][task_id]

    assert row["status"] == "needs_repair"
    assert row["next_action"] == "repair_task"


@pytest.mark.parametrize("current_status", ["admit", "reject", "invalid", "deleted"])
def test_low_confidence_uses_the_current_human_decision(
    tmp_path: Path,
    monkeypatch,
    current_status: str,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition, "low")
    _report(tmp_path, "task-confidence", {task_id: _low_confidence(digest)})
    if current_status != "deleted":
        submission_dir = tmp_path / "data/reviews/submissions" / task_id
        submission_dir.mkdir(parents=True)
        (submission_dir / "reviewer-a.json").write_text("{}")
    monkeypatch.setattr(
        "egocua.confidence_report.audit_construction_reviews",
        lambda **_kwargs: {"tasks": {task_id: {"status": current_status}}},
    )

    row = SynthesisWorkflow(tmp_path).status([task_id])["tasks"][task_id]

    expected = {
        "admit": ("ready", "build_release"),
        "reject": ("needs_repair", "repair_task"),
        "invalid": ("needs_review", "human_review"),
        "deleted": ("needs_review", "human_review"),
    }
    assert (row["status"], row["next_action"]) == expected[current_status]


def test_human_submission_does_not_release_a_low_confidence_task(
    tmp_path: Path,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition, "low")
    cached = _low_confidence(digest)
    cached.update({"human_review_required": False, "route": "human_review_complete"})
    _report(tmp_path, "task-confidence", {task_id: cached})
    submission_dir = tmp_path / "data/reviews/submissions" / task_id
    submission_dir.mkdir(parents=True)
    (submission_dir / "reviewer-a.json").write_text(
        json.dumps({"decision": "admit", "note": ""})
    )

    workflow = SynthesisWorkflow(tmp_path)
    row = workflow.status([task_id])["tasks"][task_id]

    assert row["status"] == "needs_review"
    assert row["next_action"] == "human_review"
    with pytest.raises(ValueError, match="not release-ready"):
        workflow.release_evidence([task_id])


def test_synthesis_status_is_available_from_the_main_cli(
    tmp_path: Path,
    capsys,
) -> None:
    definition, digest = _workspace(tmp_path)
    task_id = str(definition["id"])
    _deterministic_reports(tmp_path, definition, digest)

    main(["synthesize", "status", "--root", str(tmp_path)])

    output = json.loads(capsys.readouterr().out)
    assert output["summary"]["tasks"] == 1
    assert output["tasks"][task_id]["status"] == "needs_review"


def test_next_human_item_does_not_rebuild_all_confidence(tmp_path, monkeypatch):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition, "low")
    _report(tmp_path, "task-confidence", {definition["id"]: _low_confidence(digest)})
    workflow = SynthesisWorkflow(tmp_path)
    monkeypatch.setattr(
        workflow,
        "_rebuild_confidence_report",
        lambda: pytest.fail("read-only routing rebuilt the full report"),
    )
    assert workflow.next_work_item(definition["id"])["kind"] == "human_review"


def test_unrelated_acceptance_rows_do_not_block_review_report(tmp_path):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    path = tmp_path / REPORTS["acceptance"].path
    acceptance = json.loads(path.read_text())
    acceptance["tasks"]["another-unadmitted-candidate"] = {"status": "fail"}
    path.write_text(json.dumps(acceptance))

    report = build_confidence_report(tmp_path)

    assert report["tasks"][definition["id"]]["level"] == "high"


def test_scoped_review_report_preserves_other_conclusions(tmp_path, monkeypatch):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    peer = {**definition, "id": "untouched-peer"}
    previous = _high_confidence(digest)
    _report(tmp_path, "task-confidence", {peer["id"]: previous})
    monkeypatch.setattr(
        "egocua.confidence_report.load_construction_tasks",
        lambda *args: [definition, peer],
    )
    original_packet = CurrentReviewMaterials.packet

    def packet_for_requested(self, task):
        assert task["id"] == definition["id"]
        return original_packet(self, task)

    monkeypatch.setattr(CurrentReviewMaterials, "packet", packet_for_requested)

    report = build_confidence_report(tmp_path, task_ids=[definition["id"]])

    assert report["tasks"][peer["id"]] == previous
    assert report["tasks"][definition["id"]]["level"] == "high"


def test_missing_review_record_is_not_a_whole_corpus_failure(tmp_path):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    path = tmp_path / REPORTS["assessment"].path
    assessments = json.loads(path.read_text())
    assessments["results"] = []
    path.write_text(json.dumps(assessments))

    report = build_confidence_report(tmp_path)

    row = report["tasks"][definition["id"]]
    assert row["route"] == "coding_agent_retry_pending"
    assert row["human_review_required"] is False


def test_known_trial_gap_is_visible_without_skipping_missing_audit(
    tmp_path, monkeypatch
):
    from egocua.development_gate import DevelopmentDecision

    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _assessment_report(tmp_path, definition)
    _report(tmp_path, "task-confidence", {definition["id"]: _high_confidence(digest)})
    (tmp_path / REPORTS["acceptance"].path).unlink()
    monkeypatch.setattr(
        "egocua.synthesis_workflow.DevelopmentEvidence.decision",
        lambda *_args: DevelopmentDecision(
            "run_development_trial",
            "Changed actual task inputs need one current trial.",
        ),
    )
    workflow = SynthesisWorkflow(tmp_path)
    result = workflow.status()
    row = result["tasks"][definition["id"]]
    assert row["next_action"] == "run_deterministic_audit"
    assert row["remaining_actions"] == [
        "run_deterministic_audit",
        "run_development_trial",
    ]
    assert result["summary"]["in_progress"] == 1
    assert result["summary"]["pending_actions"] == {
        "run_deterministic_audit": 1,
        "run_development_trial": 1,
    }
    with pytest.raises(SynthesisWorkflowError, match="prerequisites are incomplete"):
        workflow.require_development_inputs([definition["id"]])


def test_admission_rejects_incomplete_candidate_without_writes(tmp_path, monkeypatch):
    workflow = SynthesisWorkflow(tmp_path)
    monkeypatch.setattr(
        workflow,
        "status",
        lambda ids: {
            "tasks": {
                "new": {
                    "task_id": "new",
                    "status": "needs_review",
                    "reason": "missing valid trial",
                }
            }
        },
    )
    with pytest.raises(SynthesisWorkflowError, match="missing valid trial"):
        workflow.admit(["new"])
    assert list(tmp_path.iterdir()) == []


def test_admission_installs_only_ready_candidate(tmp_path, monkeypatch):
    import egocua.synthesis_workflow as module
    from egocua.construction import CurrentConstruction

    selection = {
        "schema_version": 3,
        "task_set": {"graft": ["old"], "refresh": [], "clarify": []},
        "source_manifests": ["old.json"],
    }
    tasks = [
        {"id": tid, "track": "graft", "source": {"dossier": f"{tid}.json"}}
        for tid in ["old", "new"]
    ]
    old_path = tmp_path / "data/tasks/graft/old.json"
    old_path.parent.mkdir(parents=True)
    old_path.write_text("preserved old task")
    workflow = SynthesisWorkflow(tmp_path)
    monkeypatch.setattr(
        workflow, "status", lambda ids: {"tasks": {"new": {"status": "ready"}}}
    )
    monkeypatch.setattr(module, "load_selection", lambda _: deepcopy(selection))
    monkeypatch.setattr(module, "load_construction_tasks", lambda root, ids: tasks)
    monkeypatch.setattr(
        CurrentConstruction, "source", lambda self, task: {"task_id": task["id"]}
    )
    validated = []
    monkeypatch.setattr(
        module,
        "validate_independent_recordings",
        lambda records, **kwargs: validated.extend(records),
    )

    result = workflow.admit(["new"])

    assert result == {"admitted": ["new"], "formal_tasks": 2}
    assert old_path.read_text() == "preserved old task"
    assert json.loads((old_path.parent / "new.json").read_text()) == tasks[1]
    current = json.loads((tmp_path / "data/selection.json").read_text())
    assert current["task_set"]["graft"] == ["new", "old"]
    assert current["source_manifests"] == ["old.json", "new.json"]
    assert len(validated) == 2


def test_ready_unselected_candidate_needs_admission_not_release(tmp_path, monkeypatch):
    from egocua.development_gate import DevelopmentDecision, DevelopmentEvidence

    definition, _ = _workspace(tmp_path)
    path = tmp_path / "data/selection.json"
    selection = json.loads(path.read_text())
    selection["task_set"]["graft"] = []
    path.write_text(json.dumps(selection))
    workflow = SynthesisWorkflow(tmp_path)
    monkeypatch.setattr(
        workflow,
        "_task_status",
        lambda task_id, evidence: workflow._row(
            task_id, "build_release", "all evidence complete"
        ),
    )
    monkeypatch.setattr(
        DevelopmentEvidence,
        "decision",
        lambda *args: DevelopmentDecision(None, "verified"),
    )

    row = workflow.status([definition["id"]])["tasks"][definition["id"]]

    assert row["status"] == "ready"
    assert row["next_action"] == "admit_task"
    assert row["remaining_actions"] == ["admit_task"]
    assert workflow.status()["summary"]["tasks"] == 0


def test_release_requires_formal_membership_before_reading_candidate(tmp_path):
    _workspace(tmp_path)
    with pytest.raises(SynthesisWorkflowError, match="requires formal membership"):
        SynthesisWorkflow(tmp_path).release_evidence(["unselected-candidate"])


def test_unrelated_candidate_audits_do_not_block_formal_release(tmp_path):
    definition, digest = _workspace(tmp_path)
    _deterministic_reports(tmp_path, definition, digest)
    _completed_review_reports(tmp_path, definition)
    _report(tmp_path, "task-confidence", {definition["id"]: _high_confidence(digest)})
    for name in ["acceptance", "smoke", "role", "confidence"]:
        path = tmp_path / REPORTS[name].path
        report = json.loads(path.read_text())
        report["tasks"]["unrelated-candidate"] = {"status": "fail"}
        path.write_text(json.dumps(report))

    evidence = SynthesisWorkflow(tmp_path).release_evidence([definition["id"]])

    assert evidence["acceptance"]["tasks"] == 1
    assert evidence["smoke"]["statuses"] == {"pass": 1}
    assert evidence["role_recoverability"]["statuses"] == {"pass": 1}
    assert (
        "unrelated-candidate" not in evidence["confidence"]["current_review_evidence"]
    )
