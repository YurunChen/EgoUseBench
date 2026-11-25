"""Route tasks from blind reconstruction and adversarial Coding-Agent review."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .confidence import (
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    REJECTED_CONFIDENCE,
    UNASSESSED_CONFIDENCE,
    high_confidence_sample,
    resolve_human_review,
    reusable_review_evidence,
    route_assessment,
    unassessed_confidence,
    unavailable_assessment_confidence,
)
from .environment import site_data_request
from .confidence_packets import (
    CurrentReviewMaterials,
    MATERIALS_HASH_POLICY,
    answer_boundary,
)
from .construct_validity import (
    ADMISSION_GATES,
    CONFIDENCE_POLICY_VERSION,
    EVIDENCE_DIMENSIONS,
    POLICY,
)
from .reviews import audit_construction_reviews
from .run_ledger import canonical_sha256, task_sha256
from .construction import load_construction_tasks
from .synthesis_reports import REPORTS
from .workspace import resolve_within

ACCEPTANCE_REPORT = REPORTS["acceptance"].path
OUTPUT_PATH = REPORTS["confidence"].path


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{context}: JSON root must be an object")
    return payload


def assessment_fingerprints(results: list[Any]) -> Counter[str]:
    bodies: dict[str, str] = {}
    task_counts: Counter[str] = Counter()
    for result in results:
        if not isinstance(result, dict):
            continue
        task_bodies = set()
        for row in [result, *result.get("history", [])]:
            assessments = row.get("assessments")
            if not isinstance(assessments, list) or len(assessments) != 2:
                continue
            content = [
                {
                    **assessment,
                    "verdict": {
                        key: value
                        for key, value in assessment["verdict"].items()
                        if key != "reviewer"
                    },
                }
                if isinstance(assessment, dict)
                and isinstance(assessment.get("verdict"), dict)
                else assessment
                for assessment in assessments
            ]
            body = canonical_sha256(content)
            bodies[canonical_sha256(assessments)] = body
            task_bodies.add(body)
        task_counts.update(task_bodies)
    return Counter(
        {fingerprint: task_counts[body] for fingerprint, body in bodies.items()}
    )


def reused_confidence(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    packet: Mapping[str, Any],
    fingerprints: Mapping[str, int],
) -> dict | None:
    evidence = reusable_review_evidence(task, result, packet)
    if evidence is None:
        return None
    task_hash = task_sha256(task)
    source = evidence["source"]
    if fingerprints.get(canonical_sha256(source["assessments"])) != 1:
        return None
    if not evidence["recheck_passes"]:
        return {
            **route_assessment(task_hash, source, expected_goal=task["goal"]),
            "agent_assessments_sha256": canonical_sha256(result["assessments"]),
            "source_assessments_sha256": canonical_sha256(source["assessments"]),
        }
    return {
        **unassessed_confidence(
            task_hash,
            reason="Retain unchanged blind-review evidence; recheck changed answer boundaries.",
        ),
        "agent_assessments_sha256": canonical_sha256(result["assessments"]),
        "source_assessments_sha256": canonical_sha256(source["assessments"]),
        "current_materials_sha256": packet["materials_sha256"],
        "recheck_passes": evidence["recheck_passes"],
        "retained_passes": [
            {
                "pass": "blind_reconstruction",
                "assessment_report": POLICY["assessment"]["report"],
                "task_id": task["id"],
                "materials_sha256": source["materials_sha256"],
                "assessment_sha256": canonical_sha256(source["assessments"][0]),
            }
        ],
    }


def task_confidence(
    task: Mapping[str, Any],
    *,
    acceptance: Mapping[str, Any] | None,
    assessment: Mapping[str, Any] | None,
    materials: CurrentReviewMaterials,
    fingerprints: Mapping[str, int],
) -> dict[str, Any]:
    """Route one task from its acceptance row and Coding-Agent assessment."""
    task_hash = task_sha256(task)
    if acceptance is None or not assessment:
        return unassessed_confidence(
            task_hash,
            reason="Task-specific acceptance or review evidence is missing.",
        )
    reviewable_source_pending = (
        acceptance.get("status") == "needs_source_review"
        and acceptance.get("integrity_pass") is True
        and acceptance.get("failed_checks") == []
    )
    if (
        acceptance.get("task_sha256") == task_hash
        and acceptance.get("status") != "candidate_ready"
        and not reviewable_source_pending
    ):
        return {
            "task_sha256": task_hash,
            "policy_version": CONFIDENCE_POLICY_VERSION,
            "level": REJECTED_CONFIDENCE,
            "agent_verdict": "repair",
            "agent_confidence": "high",
            "uncertainty_reasons": [],
            "human_review_required": False,
            "route": "repair",
        }
    packet = materials.packet(dict(task))
    materials_hash = packet["materials_sha256"]
    if (
        not isinstance(materials_hash, str)
        or assessment.get("materials_sha256") != materials_hash
        or assessment.get("answer_boundary") != answer_boundary(task)
        or len(assessment.get("assessments", [])) != 2
    ):
        scoped = reused_confidence(task, assessment, packet, fingerprints)
        return scoped or unavailable_assessment_confidence(task, assessment, packet)
    body = assessment.get("assessments")
    return route_assessment(
        task_hash,
        assessment,
        expected_goal=task["goal"],
        assessment_body_reused=(
            isinstance(body, list) and fingerprints[canonical_sha256(body)] > 1
        ),
    )


HUMAN_REVIEW_ROUTES = {
    "human_review_pending",
    "automatic_construction_admission",
    "evidence_collection_pending",
    "repair",
}


def apply_human_reviews(
    root: Path, rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Resolve submitted human reviews over the Coding-Agent routes."""
    resolved = {task_id: dict(row) for task_id, row in rows.items()}
    pending = sorted(
        task_id
        for task_id, row in resolved.items()
        if (
            row.get("route") in HUMAN_REVIEW_ROUTES
            or row.get("human_review_required") is True
        )
        and any((root / "data/reviews/submissions" / task_id).glob("*.json"))
    )
    if not pending:
        return resolved
    audit = audit_construction_reviews(
        root=root, include_all_tasks=True, task_ids=pending
    )
    for task_id, result in audit["tasks"].items():
        resolved[task_id] = resolve_human_review(resolved[task_id], result)
    return resolved


def build_confidence_report(
    root: str | Path = ".", *, task_ids: list[str] | None = None
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    with site_data_request(workspace):
        policy = POLICY
        acceptance_path = resolve_within(
            workspace, ACCEPTANCE_REPORT, context="task acceptance report"
        )
        acceptance = _read_object(acceptance_path, str(acceptance_path))
        acceptance_rows = acceptance.get("tasks")
        if not isinstance(acceptance_rows, dict):
            raise TypeError("task acceptance report has no task rows")
        assessment_path = resolve_within(
            workspace,
            policy["assessment"]["report"],
            context="Coding-Agent task assessment report",
        )
        assessment_report = _read_object(assessment_path, str(assessment_path))
        if (
            assessment_report.get("schema_version")
            != REPORTS["assessment"].schema_version
            or assessment_report.get("assessor") != policy["assessment"]["assessor"]
            or assessment_report.get("pass_names") != policy["assessment"]["pass_names"]
            or assessment_report.get("materials_policy")
            != "blind_release_only_then_expected_state"
            or not isinstance(assessment_report.get("agent"), str)
            or assessment_report.get("assessments_per_task") != 2
            or assessment_report.get("refresh_policy") != "explicit_task_ids"
            or assessment_report.get("freshness_policy") != MATERIALS_HASH_POLICY
        ):
            raise ValueError(
                "Coding-Agent task assessment report uses an unsupported contract"
            )
        results = assessment_report.get("results")
        if not isinstance(results, list):
            raise TypeError("Coding-Agent task assessment report has no result rows")
        result_by_id = {
            result.get("task_id"): result
            for result in results
            if isinstance(result, dict) and isinstance(result.get("task_id"), str)
        }
        fingerprints = assessment_fingerprints(results)
        requested = set(task_ids) if task_ids is not None else None
        tasks = load_construction_tasks(
            workspace, sorted(set(result_by_id) | (requested or set()))
        )
        current_ids = {task["id"] for task in tasks}
        if requested is not None and requested - current_ids:
            raise ValueError(f"unknown task IDs: {sorted(requested - current_ids)}")
        materials = CurrentReviewMaterials(workspace)
        rows: dict[str, dict[str, Any]] = {}
        report_path = workspace / REPORTS["confidence"].path
        if requested is not None and report_path.is_file():
            rows.update(_read_object(report_path, str(report_path))["tasks"])
        computed = {
            task["id"]: task_confidence(
                task,
                acceptance=acceptance_rows.get(task["id"]),
                assessment=result_by_id.get(task["id"]),
                materials=materials,
                fingerprints=fingerprints,
            )
            for task in tasks
            if requested is None or task["id"] in requested
        }
        rows.update(apply_human_reviews(workspace, computed))

        levels = Counter(row["level"] for row in rows.values())
        routes = Counter(row["route"] for row in rows.values())
        return {
            "schema_version": REPORTS["confidence"].schema_version,
            "policy": {
                "source": "egocua.construct_validity.POLICY",
                "sha256": canonical_sha256(policy),
                "version": CONFIDENCE_POLICY_VERSION,
                "admission_gates": list(ADMISSION_GATES),
                "evidence_dimensions": list(EVIDENCE_DIMENSIONS),
            },
            "acceptance_report": {
                "path": acceptance_path.relative_to(workspace).as_posix(),
                "sha256": hashlib.sha256(acceptance_path.read_bytes()).hexdigest(),
            },
            "coding_agent_assessment_report": {
                "path": assessment_path.relative_to(workspace).as_posix(),
                "sha256": hashlib.sha256(assessment_path.read_bytes()).hexdigest(),
                "agent": assessment_report["agent"],
                "prompt_version": assessment_report["prompt_version"],
                "assessor": assessment_report["assessor"],
                "pass_names": assessment_report["pass_names"],
                "materials_policy": assessment_report["materials_policy"],
            },
            "interpretation": (
                "The Skill-executing Coding Agent first reconstructs from blind release "
                "materials, then falsifies. Demonstrated defects return to repair; unresolved "
                "semantic uncertainty goes to human review. "
                "A separate high-confidence sample can audit review reliability."
            ),
            "quality_control": high_confidence_sample(
                {task["id"]: task for task in tasks}, rows
            ),
            "summary": {
                "tasks": len(rows),
                "levels": {
                    level: levels.get(level, 0)
                    for level in (
                        HIGH_CONFIDENCE,
                        LOW_CONFIDENCE,
                        REJECTED_CONFIDENCE,
                        UNASSESSED_CONFIDENCE,
                    )
                },
                "routes": dict(sorted(routes.items())),
                "human_review_queue": sum(
                    row["human_review_required"] for row in rows.values()
                ),
                "coding_agent_retry_queue": sum(
                    row["route"] == "coding_agent_retry_pending"
                    for row in rows.values()
                ),
            },
            "tasks": rows,
        }
