"""Stateful blind-review session for one Coding-Agent assessment batch."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Any

from .confidence import (
    blind_state_matches_expected,
    reusable_review_evidence,
    verified_blind_evidence,
)
from .confidence_packets import (
    CurrentReviewMaterials,
    MATERIALS_HASH_POLICY,
    MATERIALS_POLICY,
    answer_boundary,
    validate_review_packet,
)
from .construct_validity import (
    ADMISSION_GATES,
    COUNTEREXAMPLE_GUIDANCE,
    COUNTEREXAMPLE_CHECKS,
    EVIDENCE_DIMENSIONS,
    PASS_NAMES,
    POLICY,
    ConstructValidityError,
    validate_assessment_verdict,
    validate_blind_resolutions,
)
from .schema import catalog_title_warnings, load_tasks
from .lineage import source_segment_references
from .sources import find_record
from .construction import load_construction_tasks
from .run_ledger import canonical_sha256, task_sha256
from .synthesis_reports import REPORTS
from .workspace import resolve_within, write_json_atomic

INDEX_PATH = "artifacts/reports/confidence-review-packets/index.json"
REPORT_PATH = POLICY["assessment"]["report"]
REPORT_SCHEMA_VERSION = REPORTS["assessment"].schema_version


class ConfidenceSessionError(ValueError):
    """Raised when a blind-review phase or material transition is invalid."""


def _review_transaction(update):
    """Serialize report updates while independent reviewers work outside the lock."""
    parameters = signature(update)

    @wraps(update)
    def transact(*args, **kwargs):
        arguments = parameters.bind(*args, **kwargs).arguments
        workspace = Path(arguments.get("root", ".")).resolve()
        reference = arguments.get("report_path", arguments.get("output", REPORT_PATH))
        path = resolve_within(
            workspace, reference, context="assessment report", must_exist=False
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                return update(*args, **kwargs)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    return transact


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfidenceSessionError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfidenceSessionError(f"{context}: JSON root must be an object")
    return value


def _index_rows(workspace: Path) -> dict[str, Any]:
    index_path = resolve_within(
        workspace,
        INDEX_PATH,
        context="confidence review packet index",
    )
    index = _read_object(index_path, str(index_path))
    rows = index.get("tasks")
    if (
        index.get("schema_version") != 1
        or index.get("materials_policy") != MATERIALS_POLICY
        or index.get("materials_hash_policy") != MATERIALS_HASH_POLICY
        or not isinstance(rows, dict)
        or not rows
    ):
        raise ConfidenceSessionError("confidence review packet index is unsupported")
    return rows


def _materials_sha256(task_id: str, packet_row: Mapping[str, Any]) -> str:
    value = packet_row.get("materials_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ConfidenceSessionError(
            f"{task_id}: packet index has no valid task-local materials hash"
        )
    return value


def _pending_result(
    task_id: str,
    reason: str,
    materials_sha256: str,
    boundary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "materials_sha256": materials_sha256,
        "status": "pending",
        "pending_reason": reason,
        "assessments": [],
        **(
            {"answer_boundary": deepcopy(dict(boundary))}
            if boundary is not None
            else {}
        ),
    }


def _assessment_history(previous: Mapping[str, Any]) -> list[dict[str, Any]]:
    history = previous.get("history", [])
    if not isinstance(history, list) or not all(
        isinstance(row, dict) for row in history
    ):
        raise ConfidenceSessionError(
            f"{previous['task_id']}: assessment history must be a list of snapshots"
        )
    snapshot = {
        key: previous[key]
        for key in (
            "task_sha256",
            "materials_sha256",
            "expected_states_sha256",
            "answer_boundary",
            "answer_boundary_registration",
            "status",
            "assessments",
            "pending_reason",
            "retained_passes",
            "material_verification",
            "resolution_reviews",
            "prior_review_evidence",
        )
        if key in previous
    }
    preserved = deepcopy(history)
    if snapshot not in preserved:
        preserved.append(deepcopy(snapshot))
    return preserved


def _refresh_results(
    index_rows: Mapping[str, Any],
    existing: Mapping[str, Any],
    invalidation_reasons: Mapping[str, str],
    task_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    existing_rows = existing.get("results")
    if not isinstance(existing_rows, list):
        raise ConfidenceSessionError(
            "existing confidence review has no result rows; use --reset"
        )
    by_id: dict[str, dict[str, Any]] = {}
    for result in existing_rows:
        if not isinstance(result, dict) or not isinstance(result.get("task_id"), str):
            raise ConfidenceSessionError(
                "existing confidence review has malformed rows; use --reset"
            )
        task_id = result["task_id"]
        if task_id in by_id:
            raise ConfidenceSessionError(
                "existing confidence review has duplicate task IDs; use --reset"
            )
        by_id[task_id] = result

    results: list[dict[str, Any]] = []
    for task_id, row in sorted(index_rows.items()):
        if not isinstance(row, Mapping):
            raise ConfidenceSessionError(f"{task_id}: packet index row is malformed")
        current_materials = _materials_sha256(task_id, row)
        previous = by_id.get(task_id)
        if previous is not None and task_ids is not None and task_id not in task_ids:
            results.append(deepcopy(previous))
            continue
        boundary = row.get("answer_boundary")
        previous_materials = (
            previous.get("materials_sha256") if previous is not None else None
        )
        materials_changed = (
            isinstance(previous_materials, str)
            and previous_materials != current_materials
        )
        if (
            previous is not None
            and task_id not in invalidation_reasons
            and isinstance(previous_materials, str)
            and not materials_changed
            and previous.get("status") in {"pending", "in_progress", "completed"}
            and isinstance(previous.get("assessments"), list)
        ):
            preserved = {
                key: value
                for key, value in previous.items()
                if key not in {"task_sha256", "review_packet"}
            }
            preserved["materials_sha256"] = current_materials
            if isinstance(boundary, Mapping):
                if "answer_boundary" not in previous and previous["assessments"]:
                    raise ConfidenceSessionError(
                        f"{task_id}: register the forward answer-boundary baseline first; "
                        "existing verdicts remain valid"
                    )
                if previous.get("answer_boundary") != boundary:
                    preserved["history"] = _assessment_history(previous)
                    first = previous["assessments"][:1]
                    preserved["assessments"] = first
                    preserved.pop("resolution_reviews", None)
                    preserved["status"] = "in_progress" if first else "pending"
                    preserved["pending_reason"] = "answer_boundary_changed"
                preserved["answer_boundary"] = deepcopy(dict(boundary))
            results.append(preserved)
        else:
            reason = "new_task_requires_assessment"
            if materials_changed:
                reason = "review_materials_changed"
            elif previous is not None and previous_materials is None:
                reason = "review_materials_missing"
            if previous is not None and str(
                previous.get("pending_reason", "")
            ).startswith("explicit_task_invalidation: "):
                reason = previous["pending_reason"]
            if task_id in invalidation_reasons:
                reason = f"explicit_task_invalidation: {invalidation_reasons[task_id]}"
            pending = _pending_result(
                task_id,
                reason,
                current_materials,
                boundary if isinstance(boundary, Mapping) else None,
            )
            if previous is not None:
                pending["history"] = _assessment_history(previous)
                if "prior_review_evidence" in previous:
                    pending["prior_review_evidence"] = deepcopy(
                        previous["prior_review_evidence"]
                    )
            results.append(pending)
    return results


@_review_transaction
def start_review(
    root: str | Path = ".",
    *,
    output: str = REPORT_PATH,
    reset: bool = False,
    refresh: bool = False,
    invalidation_reasons: Mapping[str, str] | None = None,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Create or refresh a batch while preserving unaffected task sessions."""

    workspace = Path(root).resolve()
    rows = _index_rows(workspace)
    selected = None if task_ids is None else set(task_ids)
    if selected is not None and not refresh:
        raise ConfidenceSessionError("scoped review requires --refresh")
    if selected is not None and selected - set(rows):
        raise ConfidenceSessionError(
            f"unknown task IDs: {sorted(selected - set(rows))}"
        )
    output_path = resolve_within(
        workspace,
        output,
        context="Coding-Agent assessment report",
        must_exist=False,
    )
    if reset and refresh:
        raise ConfidenceSessionError("--reset and --refresh are mutually exclusive")
    invalidated = dict(invalidation_reasons or {})
    if selected is not None and set(invalidated) - selected:
        raise ConfidenceSessionError(
            "task invalidations must stay within the refresh scope"
        )
    if any(
        not isinstance(task_id, str)
        or not task_id.strip()
        or not isinstance(reason, str)
        or not reason.strip()
        for task_id, reason in invalidated.items()
    ):
        raise ConfidenceSessionError(
            "every task invalidation needs a non-empty task ID and reason"
        )
    if invalidated and not refresh:
        raise ConfidenceSessionError("task invalidation requires --refresh")
    unknown = set(invalidated) - set(rows)
    if unknown:
        raise ConfidenceSessionError(
            f"unknown task IDs cannot be invalidated: {sorted(unknown)}"
        )
    if output_path.exists() and not reset and not refresh:
        raise FileExistsError(
            f"{output_path} already exists; use --refresh to preserve unchanged rows "
            "or --reset to start a new review"
        )

    retired_results: list[dict[str, Any]] = []
    if refresh and output_path.exists():
        existing = _read_object(output_path, str(output_path))
        retired_results = deepcopy(existing.get("retired_results", []))
        retired_results.extend(
            deepcopy(row) for row in existing["results"] if row["task_id"] not in rows
        )
        expected_contract = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "assessor": POLICY["assessment"]["assessor"],
            "pass_names": list(PASS_NAMES),
            "materials_policy": MATERIALS_POLICY,
            "assessments_per_task": len(PASS_NAMES),
            "refresh_policy": "explicit_task_ids",
            "freshness_policy": MATERIALS_HASH_POLICY,
        }
        mismatches = [
            key
            for key, value in expected_contract.items()
            if existing.get(key) != value
        ]
        if mismatches:
            raise ConfidenceSessionError(
                "unsupported assessment report structure; inspect the migration "
                f"before changing task evidence ({', '.join(mismatches)})"
            )
        results = _refresh_results(rows, existing, invalidated, selected)
    else:
        results = [
            _pending_result(
                task_id,
                "initial_assessment",
                _materials_sha256(task_id, rows[task_id]),
                rows[task_id].get("answer_boundary"),
            )
            for task_id in sorted(rows)
        ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "prompt_version": POLICY["assessment"]["prompt_version"],
        "assessor": POLICY["assessment"]["assessor"],
        "pass_names": list(PASS_NAMES),
        "materials_policy": MATERIALS_POLICY,
        "agent": "codex-skill-agent",
        "assessments_per_task": len(PASS_NAMES),
        "refresh_policy": "explicit_task_ids",
        "freshness_policy": MATERIALS_HASH_POLICY,
        "invalidations": [
            {"task_id": task_id, "reason": invalidated[task_id].strip()}
            for task_id in sorted(invalidated)
        ],
        "summary": _summary(results),
        "results": results,
        "retired_results": retired_results,
    }
    write_json_atomic(output_path, report)
    return report


def _row(report: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    rows = report.get("results")
    if not isinstance(rows, list):
        raise ConfidenceSessionError("assessment report has no result rows")
    matches = [
        row for row in rows if isinstance(row, dict) and row.get("task_id") == task_id
    ]
    if len(matches) != 1:
        raise ConfidenceSessionError(f"{task_id}: expected one active assessment row")
    return matches[0]


def _current_answer_boundary(workspace: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    task_id = row["task_id"]
    tasks = {task["id"]: task for task in load_construction_tasks(workspace, [task_id])}
    task = tasks.get(task_id)
    if task is None:
        raise ConfidenceSessionError(f"{task_id}: compiled task is missing")
    if "answer_boundary" not in row:
        raise ConfidenceSessionError(
            f"{task_id}: register the forward answer-boundary baseline first"
        )
    boundary = answer_boundary(task)
    if row["answer_boundary"] != boundary:
        raise ConfidenceSessionError(
            f"{task_id}: answer boundary changed; refresh this task before adversarial review"
        )
    return boundary


@_review_transaction
def register_answer_boundary_baselines(
    registrations: list[Mapping[str, Any]], *, root: str | Path = "."
) -> dict[str, Any]:
    """Start forward boundary tracking without claiming a retrospective review."""
    workspace = Path(root).resolve()
    path = resolve_within(workspace, REPORT_PATH, context="assessment report")
    report = _read_object(path, str(path))
    tasks = {task["id"]: task for task in load_tasks(workspace / "data/tasks")}
    current = CurrentReviewMaterials(workspace)
    registered_at = datetime.now(timezone.utc).isoformat()
    seen: set[str] = set()
    for registration in registrations:
        task_id = registration["task_id"]
        if task_id in seen:
            raise ConfidenceSessionError(f"{task_id}: duplicate baseline registration")
        seen.add(task_id)
        row = _row(report, task_id)
        task = tasks[task_id]
        if "answer_boundary" in row:
            raise ConfidenceSessionError(
                f"{task_id}: answer boundary is already registered"
            )
        if row.get("status") != "completed" or len(row.get("assessments", [])) != 2:
            raise ConfidenceSessionError(
                f"{task_id}: baseline requires sealed two-pass evidence"
            )
        if (
            registration["source_assessments"] != row["assessments"]
            or registration["source_materials_sha256"] != row.get("materials_sha256")
            or registration["answer_boundary"] != answer_boundary(task)
        ):
            raise ConfidenceSessionError(
                f"{task_id}: baseline reference or current boundary changed"
            )
        if not current.stored_packet_matches(task) or current.packet(task)[
            "materials_sha256"
        ] != row.get("materials_sha256"):
            raise ConfidenceSessionError(
                f"{task_id}: current public materials are not verified"
            )
        row["history"] = _assessment_history(row)
        source_index = next(
            index
            for index, source in enumerate(row["history"])
            if source.get("assessments") == row["assessments"]
            and source.get("materials_sha256") == row["materials_sha256"]
            and "answer_boundary" not in source
        )
        row["answer_boundary"] = deepcopy(registration["answer_boundary"])
        row["answer_boundary_registration"] = {
            "baseline_registered_at": registered_at,
            "purpose": "forward_change_tracking",
            "source_assessment": {
                "path": REPORT_PATH,
                "pointer": f"/results/{report['results'].index(row)}/history/{source_index}/assessments",
            },
        }
        row.pop("expected_states_sha256", None)
    write_json_atomic(path, report)
    return {"registered": len(seen), "baseline_registered_at": registered_at}


def prior_review_evidence(root: str | Path, task_id: str) -> list[dict[str, Any]]:
    """Read task-local prior findings before opening another blind review."""
    workspace = Path(root).resolve()
    report_path = resolve_within(
        workspace, REPORT_PATH, context="assessment report", must_exist=False
    )
    if not report_path.exists():
        return []
    report = _read_object(report_path, str(report_path))
    row = next((item for item in report["results"] if item["task_id"] == task_id), None)
    if row is None:
        return []
    if row["status"] == "in_progress":
        return []
    row_index = report["results"].index(row)
    evidence = (
        [
            {
                "path": REPORT_PATH,
                "pointer": f"/results/{row_index}",
                "body": deepcopy(row),
            }
        ]
        if row.get("assessments")
        else []
    )
    evidence.extend(
        [
            {
                "path": REPORT_PATH,
                "pointer": f"/results/{row_index}/history/{index}",
                "body": deepcopy(snapshot),
            }
            for index, snapshot in enumerate(row.get("history", []))
            if snapshot.get("assessments")
        ]
    )
    for reference in row.get("prior_review_evidence", []):
        path = resolve_within(
            workspace, reference["path"], context=f"{task_id} prior review"
        )
        body: Any = _read_object(path, str(path))
        pointer = reference["pointer"]
        if pointer and not pointer.startswith("/"):
            raise ConfidenceSessionError(
                f"{task_id}: invalid prior-review JSON pointer"
            )
        tokens = [
            part.replace("~1", "/").replace("~0", "~")
            for part in pointer.split("/")[1:]
        ]
        try:
            for token in tokens:
                body = body[int(token)] if isinstance(body, list) else body[token]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ConfidenceSessionError(
                f"{task_id}: prior review is missing at {reference['path']}#{pointer}"
            ) from exc
        pointer_task_id = tokens[-1] if tokens else None
        if (
            not isinstance(body, dict)
            or body.get("task_id", pointer_task_id) != task_id
        ):
            raise ConfidenceSessionError(
                f"{task_id}: prior review belongs to another task"
            )
        evidence.append({**deepcopy(reference), "body": deepcopy(body)})
    return evidence


def open_pass(
    task_id: str,
    *,
    root: str | Path = ".",
    report_path: str = REPORT_PATH,
) -> dict[str, Any]:
    """Reveal only the materials allowed in the task's current review phase."""

    workspace = Path(root).resolve()
    report_file = resolve_within(
        workspace,
        report_path,
        context="Coding-Agent assessment report",
    )
    report = _read_object(report_file, str(report_file))
    row = _row(report, task_id)
    assessments = row.get("assessments")
    if not isinstance(assessments, list):
        raise ConfidenceSessionError(f"{task_id}: assessments must be a list")
    if len(assessments) >= len(PASS_NAMES):
        raise ConfidenceSessionError(f"{task_id}: assessment is already complete")

    packet_row = _index_rows(workspace).get(task_id)
    if not isinstance(packet_row, Mapping) or not isinstance(
        packet_row.get("path"), str
    ):
        raise ConfidenceSessionError(f"{task_id}: review packet is missing")
    current_materials = _materials_sha256(task_id, packet_row)
    if row.get("materials_sha256") != current_materials:
        raise ConfidenceSessionError(
            f"{task_id}: review materials changed; refresh this task before review"
        )
    packet_path = resolve_within(
        workspace,
        packet_row["path"],
        context=f"{task_id} review packet",
    )
    packet = _read_object(packet_path, str(packet_path))
    validate_review_packet(packet)
    pass_name = PASS_NAMES[len(assessments)]
    materials: dict[str, Any] = {
        "task_id": task_id,
        "pass": pass_name,
        "review_packet": packet,
        "required_evidence_dimensions": list(EVIDENCE_DIMENSIONS),
        "derived_admission_gates": list(ADMISSION_GATES),
        "review_protocol": {
            "default_reviewers": 1,
            "independence": "not_the_constructor_and_no_constructor_context",
            "blind_restart": "use_a_new_context_that_has_not_seen_the_key",
            "evidence_references": "evidence.<dimension> or blind_reconstruction.evidence.<dimension>",
        },
    }
    if pass_name == PASS_NAMES[1]:
        materials["answer_boundary"] = _current_answer_boundary(workspace, row)
        materials["expected_goal"] = materials["answer_boundary"]["final_goal"]
        if "source_segments_dossier" in packet_row:
            dossier_path = resolve_within(
                workspace,
                packet_row["source_segments_dossier"],
                context=f"{task_id} segmented source",
            )
            dossier = _read_object(dossier_path, str(dossier_path))
            record = find_record(dossier, task_id, context=str(dossier_path))
            materials["source_video_segments"] = source_segment_references(
                record, dossier["dataset"], workspace
            )
            if materials["source_video_segments"] is None:
                raise ConfidenceSessionError(
                    f"{task_id}: registered segmented source is missing"
                )
        materials["sealed_blind_verdict"] = deepcopy(assessments[0].get("verdict"))
        materials["required_counterexample_checks"] = list(COUNTEREXAMPLE_CHECKS)
        materials["counterexample_guidance"] = dict(COUNTEREXAMPLE_GUIDANCE)
        materials["review_protocol"]["blind_resolutions"] = (
            "Resolve each sealed concern by concern path, exact original text, concrete "
            "evidence.<dimension>.<index> references in pass two, and reason. "
            "A correct complete blind final state remains required; high alone resolves nothing."
        )
        materials["advisories"] = _title_advisories(workspace, task_id, packet)
    return materials


def _title_advisories(
    workspace: Path, task_id: str, packet: Mapping[str, Any]
) -> list[str]:
    public_site = packet["materials"].get("public_site")
    if not isinstance(public_site, Mapping) or not isinstance(
        public_site.get("path"), str
    ):
        return []
    site_path = resolve_within(
        workspace, public_site["path"], context="public site snapshot"
    )
    site_data = _read_object(site_path, "public site snapshot")
    definition = next(
        task for task in load_construction_tasks(workspace, [task_id]) if task["id"] == task_id
    )
    return catalog_title_warnings(definition, site_data)


def _require_fresh_blind_context(
    row: Mapping[str, Any], verdict: Mapping[str, Any]
) -> None:
    reviewer = verdict["reviewer"]
    context_id = reviewer["context_id"]
    for snapshot in row.get("history", []):
        for assessment in snapshot.get("assessments", []):
            previous = assessment.get("verdict", {}).get("reviewer") or {}
            if (
                assessment.get("pass") == PASS_NAMES[1]
                and previous.get("context_id") == context_id
            ):
                raise ConfidenceSessionError(
                    "blind reconstruction needs a new context; this context has seen the key"
                )


@_review_transaction
def start_protocol_recheck(
    task_id: str, *, root: str | Path = ".", reason: str | None = None
) -> None:
    """Retain a proven unchanged blind pass and reopen answer-boundary review."""
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ConfidenceSessionError(
            "targeted second-pass recheck needs a concrete reason"
        )
    workspace = Path(root).resolve()
    path = resolve_within(workspace, REPORT_PATH, context="assessment report")
    report = _read_object(path, str(path))
    row = _row(report, task_id)
    index = _index_rows(workspace)[task_id]
    packet_path = resolve_within(workspace, index["path"], context="review packet")
    packet = _read_object(packet_path, str(packet_path))
    validate_review_packet(packet)
    task = next(
        task for task in load_construction_tasks(workspace, [task_id]) if task["id"] == task_id
    )
    if (
        index.get("task_sha256") != task_sha256(task)
        or index.get("materials_sha256") != packet["materials_sha256"]
    ):
        raise ConfidenceSessionError(
            f"{task_id}: refresh the current task's review packet first"
        )
    evidence = reusable_review_evidence(task, row, packet)
    if evidence is None or (not evidence["recheck_passes"] and reason is None):
        raise ConfidenceSessionError(
            f"{task_id}: unchanged blind-review evidence is not proven"
        )
    source = evidence["source"]
    passes = source["assessments"]
    row["history"] = _assessment_history(row)
    row.pop("resolution_reviews", None)
    row.update(
        materials_sha256=packet["materials_sha256"],
        answer_boundary=answer_boundary(task),
        assessments=deepcopy(passes[:1]),
        status="in_progress",
        pending_reason=evidence["pending_reason"],
        retained_passes=[
            {
                "pass": PASS_NAMES[0],
                "assessment_report": REPORT_PATH,
                "task_id": task_id,
                "materials_sha256": source["materials_sha256"],
                "assessment_sha256": canonical_sha256(passes[0]),
            }
        ],
    )
    if reason is not None:
        row["recheck_reason"] = reason
    report["summary"] = _summary(report["results"])
    write_json_atomic(path, report)


@_review_transaction
def import_blind_review(
    task_id: str,
    evidence: Mapping[str, str],
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Keep a pre-installation blind review with its actual original packet.

    Importing history does not approve the current task. The existing factual
    inheritance and second-pass interfaces decide whether it still applies.
    """
    workspace = Path(root).resolve()
    packet = _read_object(
        resolve_within(workspace, evidence["packet"], context="original blind packet"),
        "original blind packet",
    )
    verdict = _read_object(
        resolve_within(
            workspace, evidence["verdict"], context="original blind verdict"
        ),
        "original blind verdict",
    )
    validate_review_packet(packet)
    if packet["materials"]["task_id"] != task_id:
        raise ConfidenceSessionError("original blind packet belongs to another task")
    validate_assessment_verdict(verdict, pass_name=PASS_NAMES[0])
    path = resolve_within(workspace, REPORT_PATH, context="assessment report")
    report = _read_object(path, str(path))
    row = _row(report, task_id)
    source = {
        "task_id": task_id,
        "materials_sha256": packet["materials_sha256"],
        "status": "in_progress",
        "assessments": [
            {
                "index": 0,
                "pass": PASS_NAMES[0],
                "status": "completed",
                "verdict": deepcopy(verdict),
            }
        ],
        "source_reference": dict(evidence),
    }
    history = row.setdefault("history", [])
    if source not in history:
        history.append(source)
    write_json_atomic(path, report)
    return deepcopy(source)


@_review_transaction
def retain_verified_blind_review(
    task_id: str,
    verification: Mapping[str, Any],
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Record factual inheritance, preserve the blind original, and open only pass two."""
    workspace = Path(root).resolve()
    path = resolve_within(workspace, REPORT_PATH, context="assessment report")
    report = _read_object(path, str(path))
    row = _row(report, task_id)
    task = next(
        task for task in load_construction_tasks(workspace, [task_id]) if task["id"] == task_id
    )
    packet = CurrentReviewMaterials(workspace).packet(task)
    index = _index_rows(workspace)[task_id]
    stored_packet = _read_object(
        resolve_within(workspace, index["path"], context="review packet"),
        "review packet",
    )
    validate_review_packet(stored_packet)
    if (
        verification.get("current_materials_sha256") != packet["materials_sha256"]
        or index.get("materials_sha256") != packet["materials_sha256"]
        or stored_packet["materials_sha256"] != packet["materials_sha256"]
        or index.get("task_sha256") != task_sha256(task)
    ):
        raise ConfidenceSessionError(
            "refresh the current review packet and factual verification first"
        )
    inherited = deepcopy(row)
    for reference in prior_review_evidence(workspace, task_id):
        if reference["path"] == REPORT_PATH:
            continue
        body = reference["body"]
        source = (
            deepcopy(body)
            if "assessments" in body
            else {
                "status": "in_progress",
                "assessments": [
                    {
                        "index": 0,
                        "pass": PASS_NAMES[0],
                        "status": "completed",
                        "verdict": deepcopy(body),
                    }
                ],
            }
        )
        if "materials_sha256" in body:
            source["materials_sha256"] = body["materials_sha256"]
        source["source_reference"] = {
            key: deepcopy(value) for key, value in reference.items() if key != "body"
        }
        inherited.setdefault("history", []).append(source)
    try:
        source = verified_blind_evidence(task, inherited, verification)
    except ValueError as exc:
        raise ConfidenceSessionError(str(exc)) from exc
    blind = source["assessments"][0]
    blind_content = {
        key: value for key, value in blind["verdict"].items() if key != "reviewer"
    }
    for other in report["results"]:
        if other["task_id"] == task_id:
            continue
        for snapshot in [other, *other.get("history", [])]:
            for assessment in snapshot.get("assessments", []):
                if (
                    assessment.get("pass") == PASS_NAMES[0]
                    and {
                        key: value
                        for key, value in assessment.get("verdict", {}).items()
                        if key != "reviewer"
                    }
                    == blind_content
                ):
                    raise ConfidenceSessionError(
                        "blind review body was copied across tasks"
                    )
    row["history"] = _assessment_history(row)
    if source not in row["history"]:
        row["history"].append(deepcopy(source))
    row.pop("resolution_reviews", None)
    row.update(
        materials_sha256=packet["materials_sha256"],
        answer_boundary=answer_boundary(task),
        assessments=[deepcopy(blind)],
        status="in_progress",
        pending_reason="verified_material_inheritance",
        material_verification=deepcopy(dict(verification)),
        retained_passes=[
            {
                "pass": PASS_NAMES[0],
                "assessment_report": REPORT_PATH,
                "task_id": task_id,
                "materials_sha256": source.get("materials_sha256"),
                "assessment_sha256": canonical_sha256(blind),
            }
        ],
    )
    report["summary"] = _summary(report["results"])
    write_json_atomic(path, report)
    return deepcopy(row)


@_review_transaction
def record_assessment(
    task_id: str,
    pass_name: str,
    verdict: dict[str, Any],
    *,
    root: str | Path = ".",
    report_path: str = REPORT_PATH,
) -> dict[str, Any]:
    """Freeze one valid pass and advance exactly one review phase."""

    if (
        pass_name == PASS_NAMES[0]
        and "derived_final_state" in verdict
        and not blind_state_matches_expected(verdict["derived_final_state"], None)
    ):
        raise ConfidenceSessionError("blind answer must be a complete website state")
    workspace = Path(root).resolve()
    path = resolve_within(
        workspace,
        report_path,
        context="Coding-Agent assessment report",
    )
    report = _read_object(path, str(path))
    if report.get("pass_names") != list(PASS_NAMES):
        raise ConfidenceSessionError("assessment report has unsupported pass names")
    row = _row(report, task_id)
    packet_row = _index_rows(workspace).get(task_id)
    if not isinstance(packet_row, Mapping):
        raise ConfidenceSessionError(f"{task_id}: review packet is missing")
    current_materials = _materials_sha256(task_id, packet_row)
    if row.get("materials_sha256") != current_materials:
        raise ConfidenceSessionError(
            f"{task_id}: review materials changed; refresh this task before recording"
        )
    assessments = row.get("assessments")
    if not isinstance(assessments, list):
        raise ConfidenceSessionError(f"{task_id}: assessments must be a list")
    if len(assessments) >= len(PASS_NAMES):
        raise ConfidenceSessionError(f"{task_id}: assessment is already complete")
    expected = PASS_NAMES[len(assessments)]
    if pass_name != expected:
        raise ConfidenceSessionError(
            f"{task_id}: expected {expected!r} before {pass_name!r}"
        )
    if pass_name == PASS_NAMES[1]:
        _current_answer_boundary(workspace, row)
        if "blind_resolutions" in verdict:
            task = next(
                task
                for task in load_construction_tasks(workspace, [task_id])
                if task["id"] == task_id
            )
            if not CurrentReviewMaterials(workspace).stored_packet_matches(task):
                raise ConfidenceSessionError(
                    "review materials changed; refresh before resolving concerns"
                )
    try:
        validate_assessment_verdict(
            verdict,
            pass_name=pass_name,
            blind_verdict=assessments[0]["verdict"] if assessments else None,
        )
    except ConstructValidityError as exc:
        raise ConfidenceSessionError(str(exc)) from exc
    if verdict.get("reviewer") is None:
        raise ConfidenceSessionError(
            "new assessment requires the actual reviewer agent_id and context_id"
        )
    if pass_name == PASS_NAMES[0]:
        _require_fresh_blind_context(row, verdict)

    assessments.append(
        {
            "index": len(assessments),
            "pass": pass_name,
            "status": "completed",
            "verdict": verdict,
        }
    )
    row.pop("pending_reason", None)
    row["status"] = (
        "completed" if len(assessments) == len(PASS_NAMES) else "in_progress"
    )
    report["summary"] = _summary(report["results"])
    write_json_atomic(path, report)
    return row


@_review_transaction
def append_blind_resolutions(
    task_id: str,
    review: Mapping[str, Any],
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    """Append the second reviewer's current concern resolutions, preserving both sealed passes."""
    workspace = Path(root).resolve()
    path = resolve_within(workspace, REPORT_PATH, context="assessment report")
    report = _read_object(path, str(path))
    row = _row(report, task_id)
    assessments = row.get("assessments", [])
    if len(assessments) != 2 or row.get("status") != "completed":
        raise ConfidenceSessionError(
            "concern resolution requires two completed assessments"
        )
    task = next(
        task for task in load_construction_tasks(workspace, [task_id]) if task["id"] == task_id
    )
    current = CurrentReviewMaterials(workspace)
    if (
        not current.stored_packet_matches(task)
        or row.get("materials_sha256") != current.packet(task)["materials_sha256"]
    ):
        raise ConfidenceSessionError(
            "review materials changed; refresh before resolving concerns"
        )
    _current_answer_boundary(workspace, row)
    blind, second = (assessment["verdict"] for assessment in assessments)
    if set(review) != {"reviewer", "blind_resolutions"} or review[
        "reviewer"
    ] != second.get("reviewer"):
        raise ConfidenceSessionError(
            "resolution requires the actual second-pass reviewer"
        )
    try:
        for assessment in assessments:
            validate_assessment_verdict(
                assessment["verdict"], pass_name=assessment["pass"], blind_verdict=blind
            )
        validate_blind_resolutions(review["blind_resolutions"], blind, second)
    except ConstructValidityError as exc:
        raise ConfidenceSessionError(str(exc)) from exc
    row.setdefault("resolution_reviews", []).append(deepcopy(dict(review)))
    write_json_atomic(path, report)
    return deepcopy(row)


def _summary(rows: list[Any]) -> dict[str, int]:
    return {
        "tasks": len(rows),
        "completed": sum(
            isinstance(row, Mapping) and row.get("status") == "completed"
            for row in rows
        ),
        "in_progress": sum(
            isinstance(row, Mapping) and row.get("status") == "in_progress"
            for row in rows
        ),
        "pending": sum(
            isinstance(row, Mapping) and row.get("status") == "pending" for row in rows
        ),
    }
