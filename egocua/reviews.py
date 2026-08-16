"""Audit two isolated construction reviews against the current workspace."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .review_workspace import REVIEW_SPEC, ReviewWorkspace, ReviewWorkspaceError
from .workspace import resolve_within


class ReviewAuditError(ValueError):
    """Raised when construction review data is incomplete or inconsistent."""


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewAuditError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewAuditError(f"{context}: JSON root must be an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_files(root: Path, task_id: str) -> list[Path]:
    directory = root / "data" / "reviews" / "submissions" / task_id
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _validate_submission(
    payload: Mapping[str, Any], *, path: Path, case: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = case["task"]["id"]
    required_keys = {
        "schema_version", "reviewer", "task_id", "decision", "note",
        "target_granularity", "unresolved", "reviewed_artifacts", "revision",
        "submitted_at_utc",
    }
    if set(payload) != required_keys or payload.get("schema_version") != 5:
        raise ReviewAuditError(f"{path}: invalid submission shape")
    reviewer = payload.get("reviewer")
    if not isinstance(reviewer, str) or not reviewer or path.stem != reviewer:
        raise ReviewAuditError(f"{path}: reviewer must match the submission filename")
    if payload.get("task_id") != task_id:
        raise ReviewAuditError(f"{path}: task_id must equal {task_id!r}")
    decision = payload.get("decision")
    note = payload.get("note")
    if (
        decision not in {"admit", "reject"}
        or not isinstance(note, str)
        or (decision == "reject" and not note.strip())
    ):
        raise ReviewAuditError(f"{path}: invalid overall review decision")
    unresolved = payload.get("unresolved")
    if not isinstance(unresolved, list) or any(
        not isinstance(item, str) or not item.strip() for item in unresolved
    ):
        raise ReviewAuditError(f"{path}: unresolved must contain non-empty strings")
    expected_unresolved = [] if decision == "admit" else [note.strip()]
    if unresolved != expected_unresolved:
        raise ReviewAuditError(f"{path}: unresolved does not follow the decision")
    if payload.get("target_granularity") != case["task"]["goal"]["target_granularity"]:
        raise ReviewAuditError(f"{path}: target granularity must come from the task")
    if payload.get("reviewed_artifacts") != case["reviewed_artifacts"]:
        raise ReviewAuditError(f"{path}: reviewed artifacts changed")
    if payload.get("revision") != case["revision"]:
        raise ReviewAuditError(f"{path}: review revision is stale")
    timestamp = payload.get("submitted_at_utc")
    try:
        datetime.strptime(str(timestamp), "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ReviewAuditError(f"{path}: invalid submission time") from exc
    return dict(payload)


def audit_construction_reviews(
    spec: str = REVIEW_SPEC, *, root: str | Path = ".",
) -> dict[str, Any]:
    workspace_root = Path(root).resolve()
    try:
        workspace = ReviewWorkspace.open(workspace_root, "audit", spec=spec)
    except ReviewWorkspaceError as exc:
        raise ReviewAuditError(str(exc)) from exc
    results: dict[str, dict[str, Any]] = {}
    required = workspace.spec["reviewer_policy"]["required_independent_reviews"]
    for task_id in workspace.ordered_ids:
        case = workspace.view(task_id)
        files = _review_files(workspace_root, task_id)
        if len(files) < required:
            results[task_id] = {
                "status": "pending",
                "reason": "missing_independent_reviews",
                "submitted_reviews": len(files),
                "required_reviews": required,
                "revision": case["revision"],
            }
            continue
        if len(files) > required:
            results[task_id] = {
                "status": "invalid",
                "errors": [f"expected exactly {required} submissions, found {len(files)}"],
            }
            continue
        submissions: list[dict[str, Any]] = []
        errors: list[str] = []
        for path in files:
            try:
                submissions.append(_validate_submission(
                    _load_object(path, str(path)), path=path, case=case
                ))
            except ReviewAuditError as exc:
                errors.append(str(exc))
        if errors:
            results[task_id] = {"status": "invalid", "errors": errors}
            continue
        reviewers = [submission["reviewer"] for submission in submissions]
        if len(set(reviewers)) != len(reviewers):
            results[task_id] = {
                "status": "invalid",
                "errors": ["reviewer IDs must be distinct"],
            }
            continue
        decisions = {submission["decision"] for submission in submissions}
        granularities = {
            submission["target_granularity"] for submission in submissions
        }
        if decisions != {"admit"} or len(granularities) != 1:
            results[task_id] = {
                "status": "reject",
                "reviewers": reviewers,
                "reasons": [
                    "one or more reviewers rejected the task"
                    if decisions != {"admit"}
                    else "reviewers disagree on target granularity"
                ],
            }
            continue
        results[task_id] = {
            "status": "admit",
            "reviewers": reviewers,
            "target_granularity": next(iter(granularities)),
            "submission_artifacts": {
                path.relative_to(workspace_root).as_posix(): _sha256(path)
                for path in files
            },
        }
    counts = Counter(result["status"] for result in results.values())
    spec_path = resolve_within(workspace_root, spec, context="construction review spec")
    return {
        "schema_version": 1,
        "spec": spec_path.relative_to(workspace_root).as_posix(),
        "spec_sha256": _sha256(spec_path),
        "counts": {
            status: counts.get(status, 0)
            for status in ("pending", "admit", "reject", "invalid")
        },
        "tasks": results,
        "all_admitted": bool(results) and all(
            result["status"] == "admit" for result in results.values()
        ),
    }


def construction_evidence_from_reviews(
    task: dict[str, Any], spec: str = REVIEW_SPEC, *, root: str | Path = ".",
) -> dict[str, Any]:
    workspace_root = Path(root).resolve()
    result = audit_construction_reviews(spec, root=workspace_root)
    admitted = result["tasks"].get(task["id"])
    if not isinstance(admitted, dict) or admitted.get("status") != "admit":
        raise ReviewAuditError(f"{task['id']}: construction reviews are not admitted")
    files = _review_files(workspace_root, task["id"])
    submissions = [_load_object(path, str(path)) for path in files]
    return {
        "audit": {
            "spec": result["spec"],
            "spec_sha256": result["spec_sha256"],
            "submission_artifacts": admitted["submission_artifacts"],
        },
        "reviews": [
            {
                "reviewer": submission["reviewer"],
                "decision": "admit",
                "source_evidence_supported": True,
                "intervention_isolated": True,
                "target_granularity": submission["target_granularity"],
                "acceptable_states": task["goal"]["acceptable_states"],
                "unresolved": [],
            }
            for submission in submissions
        ],
    }


def validate_construction_evidence(
    task: dict[str, Any], construction: dict[str, Any], *, root: str | Path,
) -> None:
    audit = construction.get("audit")
    if not isinstance(audit, dict) or not str(audit.get("spec", "")).strip():
        raise ReviewAuditError(f"{task['id']}: construction evidence needs a review spec")
    expected = construction_evidence_from_reviews(
        task, str(audit["spec"]), root=root
    )
    if construction != expected:
        raise ReviewAuditError(
            f"{task['id']}: construction evidence does not match current reviews"
        )
