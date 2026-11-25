"""Audit the blinded construction review against the current workspace."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .review_materials import (
    REVIEW_SPEC,
    ReviewMaterialError,
    ReviewMaterials,
    review_submission_matches_case,
)
from .selection import load_selection, task_ids as selected_task_ids
from .schema import validate_dataset
from .construction import load_construction_tasks
from .workspace import read_json_strict, resolve_within


class ReviewAuditError(ValueError):
    """Raised when construction review data is incomplete or inconsistent."""


class StaleReviewError(ReviewAuditError):
    """The recorded review no longer matches the current task materials."""


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = read_json_strict(path)
    except (OSError, ValueError) as exc:
        raise ReviewAuditError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewAuditError(f"{context}: JSON root must be an object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_files(root: Path, task_id: str) -> list[Path]:
    directory = root / "data" / "reviews" / "submissions" / task_id
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def validate_review_submission(
    payload: Mapping[str, Any],
    *,
    path: Path,
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one human submission using the current shared review materials."""
    task_id = case["task"]["id"]
    required_keys = {
        "schema_version",
        "reviewer",
        "task_id",
        "decision",
        "note",
        "target_granularity",
        "unresolved",
        "reviewed_artifacts",
        "revision",
        "submitted_at_utc",
    }
    if (
        set(payload) - {"purpose"} != required_keys
        or payload.get("schema_version") != 5
    ):
        raise ReviewAuditError(f"{path}: invalid submission shape")
    if "purpose" in payload and payload["purpose"] != "quality_control":
        raise ReviewAuditError(f"{path}: invalid review purpose")
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
    timestamp = payload.get("submitted_at_utc")
    try:
        datetime.strptime(str(timestamp), "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as exc:
        raise ReviewAuditError(f"{path}: invalid submission time") from exc
    reuse = next(
        (
            item
            for item in case["human_review_reuse"]
            if item["submission"]["path"] == path.as_posix()
            and item["submission"]["submitted_at_utc"] == payload["submitted_at_utc"]
        ),
        None,
    )
    if reuse is not None and reuse["outcome"] == "partial":
        raise StaleReviewError(
            f"{path}: prior conclusions retained; only affected scope remains"
        )
    if reuse is not None and (
        (reuse["outcome"] == "retained" and decision != "admit")
        or (reuse["outcome"] == "repair_verified" and decision != "reject")
    ):
        raise ReviewAuditError(
            f"{path}: reuse outcome contradicts the original decision"
        )
    if (
        reuse is not None
        and (
            (reuse["outcome"] == "retained" and decision == "admit")
            or (reuse["outcome"] == "repair_verified" and decision == "reject")
        )
        and payload.get("target_granularity")
        == case["task"]["goal"]["target_granularity"]
    ):
        return {**payload, "human_review_reuse": reuse}
    if not review_submission_matches_case(payload, case):
        if payload.get("reviewed_artifacts") == case["reviewed_artifacts"]:
            raise StaleReviewError(f"{path}: review revision is stale")
        raise StaleReviewError(f"{path}: reviewed task materials changed")
    if payload.get("target_granularity") != case["task"]["goal"]["target_granularity"]:
        raise ReviewAuditError(f"{path}: target granularity must come from the task")
    return dict(payload)


def audit_construction_reviews(
    spec: str = REVIEW_SPEC,
    *,
    root: str | Path = ".",
    include_all_tasks: bool = False,
    task_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    workspace_root = Path(root).resolve()
    requested_ids = None if task_ids is None else list(task_ids)
    if requested_ids is None and not include_all_tasks:
        selected_ids = selected_task_ids(load_selection(workspace_root))
        requested_ids = [
            task_id
            for task_id in selected_ids
            if _review_files(workspace_root, task_id)
        ]
    try:
        materials = ReviewMaterials(workspace_root, spec)
        selected_ids = selected_task_ids(load_selection(workspace_root))
        requested = set(selected_ids if requested_ids is None else requested_ids)
        tasks = load_construction_tasks(workspace_root, sorted(requested))
        validate_dataset(tasks, root=workspace_root, require_assets=False)
        missing = requested - {task["id"] for task in tasks}
        if missing:
            raise ReviewAuditError(
                f"selected executable tasks are missing: {sorted(missing)}"
            )
    except (ReviewMaterialError, OSError) as exc:
        raise ReviewAuditError(str(exc)) from exc
    results: dict[str, dict[str, Any]] = {}
    required = materials.spec["reviewer_policy"]["required_independent_reviews"]
    for task in tasks:
        task_id = task["id"]
        if task_id not in requested:
            continue
        try:
            case = materials.case(task)
        except ReviewMaterialError as exc:
            results[task_id] = {"status": "invalid", "errors": [str(exc)]}
            continue
        files = _review_files(workspace_root, task_id)
        if len(files) < required:
            results[task_id] = {
                "status": "pending",
                "reason": "missing_human_review",
                "submitted_reviews": len(files),
                "required_reviews": required,
                "revision": case["revision"],
            }
            continue
        if len(files) > required:
            results[task_id] = {
                "status": "invalid",
                "errors": [
                    f"expected exactly {required} submissions, found {len(files)}"
                ],
            }
            continue
        submissions: list[dict[str, Any]] = []
        errors: list[str] = []
        stale_reviews: list[str] = []
        for path in files:
            try:
                submissions.append(
                    validate_review_submission(
                        _load_object(path, str(path)),
                        path=path.relative_to(workspace_root),
                        case=case,
                    )
                )
            except StaleReviewError as exc:
                stale_reviews.append(str(exc))
            except ReviewAuditError as exc:
                errors.append(str(exc))
        if errors:
            results[task_id] = {"status": "invalid", "errors": errors}
            continue
        if stale_reviews:
            results[task_id] = {
                "status": "pending",
                "reason": "reviewed_materials_changed",
                "details": stale_reviews,
                "submitted_reviews": len(submissions),
                "required_reviews": required,
                "revision": case["revision"],
            }
            if case["human_review_reuse"]:
                results[task_id]["human_review_reuse"] = case["human_review_reuse"]
            continue
        reviewers = [submission["reviewer"] for submission in submissions]
        purpose = submissions[0].get("purpose", "admission")
        decisions = {
            "admit"
            if submission.get("human_review_reuse", {}).get("outcome")
            == "repair_verified"
            else submission["decision"]
            for submission in submissions
        }
        granularities = {submission["target_granularity"] for submission in submissions}
        if decisions != {"admit"} or len(granularities) != 1:
            results[task_id] = {
                "status": "reject",
                "purpose": purpose,
                "reviewers": reviewers,
                "reasons": [
                    "the reviewer requested a repair"
                    if decisions != {"admit"}
                    else "the submitted target granularity is invalid"
                ],
            }
            continue
        results[task_id] = {
            "status": "admit",
            "purpose": purpose,
            "reviewers": reviewers,
            "target_granularity": next(iter(granularities)),
            "submission_artifacts": {
                path.relative_to(workspace_root).as_posix(): _sha256(path)
                for path in files
            },
        }
        if case["human_review_reuse"]:
            results[task_id]["human_review_reuse"] = case["human_review_reuse"]
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
        "all_admitted": all(result["status"] == "admit" for result in results.values()),
    }


