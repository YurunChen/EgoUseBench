"""Build current development qualification evidence from real Agent trials."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .workspace import read_json_strict
from .development_gate import (
    DevelopmentEvidence,
    inherit_unchanged_review,
    record_mismatch,
    rescore_development_run,
)
from .environment import load_site_data, site_data_request
from .failure_attribution import attribute_run
from .qualification import (
    QUALIFICATION_POLICY_VERSION,
    failure_diagnostic,
    qualification_family,
    qualification_family_key,
    qualification_outcome,
)
from .run_ledger import (
    find_run_directories,
    read_run_identity,
    load_run_record,
    task_sha256,
)
from .construction import load_construction_tasks
from .sources import source_record

REPORT = "artifacts/reports/task-qualification.json"
TRIAGE = "data/reviews/task-qualification.json"
COMPLETED_STATUSES = {
    "executable_demonstrated_once",
    "qualified_after_triage",
}


def _should_replace_trial(
    previous_started: str,
    previous_screen: Mapping[str, Any],
    candidate_started: str,
    candidate_screen: Mapping[str, Any],
) -> bool:
    """Keep a completed audit stable; otherwise expose the newest unresolved run."""

    previous_complete = previous_screen.get("status") in COMPLETED_STATUSES
    candidate_complete = candidate_screen.get("status") in COMPLETED_STATUSES
    if previous_complete != candidate_complete:
        return candidate_complete
    if candidate_complete:
        return candidate_started < previous_started
    return candidate_started > previous_started


def _screen_records(
    root: Path,
    tasks: Mapping[str, dict[str, Any]],
    experiment_ids: tuple[str, ...] | None,
    triage_rows: Mapping[str, Any],
    legacy_admissions: frozenset[str] = frozenset(),
) -> tuple[
    dict[str, dict[str, Any]],
    Counter[str],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    selected: dict[
        str, tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], Path]
    ] = {}
    rejected: Counter[str] = Counter()
    development = DevelopmentEvidence(root)

    run_directories = find_run_directories(root)
    reviewed_directories: set[Path] = set()
    for review in triage_rows.values():
        if isinstance(review, Mapping) and isinstance(review.get("run_artifact"), str):
            reviewed_directories.add(root / review["run_artifact"])
    run_directories.update(reviewed_directories)
    for run_directory in sorted(run_directories):
        try:
            manifest = read_run_identity(run_directory)
        except (OSError, ValueError):
            rejected["unreadable_manifest"] += 1
            continue
        if (
            experiment_ids is not None
            and manifest["experiment_id"] not in experiment_ids
            and run_directory not in reviewed_directories
        ):
            continue
        if manifest.get("schema_version") != 4:
            rejected["legacy_schema"] += 1
            continue
        task_id = str(manifest.get("task_id") or "")
        task = tasks.get(task_id)
        if task is None:
            rejected["not_current_task"] += 1
            continue
        try:
            record = load_run_record(run_directory)
        except (OSError, ValueError, json.JSONDecodeError):
            rejected["invalid_artifact"] += 1
            continue
        meta = record.get("meta", {})
        mismatch = record_mismatch(task, record, root)
        if mismatch is not None:
            rejected[mismatch] += 1
            continue
        try:
            checked = rescore_development_run(task, record)
        except (TypeError, ValueError):
            rejected["checker_rejected"] += 1
            continue
        screen = {
            **qualification_outcome(checked),
            "run_artifact": run_directory.relative_to(root).as_posix(),
            "agent": checked.get("agent"),
            "model": checked.get("model"),
            "model_version": meta.get("model_version"),
            "condition": checked.get("condition", {}).get("name"),
        }
        review = triage_rows.get(task_id, {})
        if review.get("run_artifact") != screen["run_artifact"]:
            review = {}
        review = inherit_unchanged_review(task, record, review)
        decision = development.trajectory_decision(task, checked, review)
        if decision.action is None:
            screen.update(status="qualified_after_triage", next_action="retain")
        elif task_id in legacy_admissions and decision.action in {
            "review_development_trial",
            "verify_development_evidence",
        }:
            screen.update(status="executable_demonstrated_once", next_action="retain")
        else:
            status = {
                "repair_task": "repair_required",
                "run_development_trial": "invalid_run",
                "verify_development_evidence": "evidence_review_required",
                "review_development_trial": "trajectory_review_required"
                if checked.get("e2e_success")
                else "triage_required",
            }[decision.action]
            screen.update(
                status=status, next_action=decision.action, reason=decision.reason
            )
        if decision.action in {None, "repair_task", "run_development_trial"} and review:
            screen["coding_agent_triage"] = {
                "verdict": review.get("verdict"),
                "evidence": review.get("evidence"),
            }
        started = str(meta.get("started_at") or manifest.get("started_at_utc") or "")
        previous = selected.get(task_id)
        if previous is None or _should_replace_trial(
            previous[0],
            previous[1],
            started,
            screen,
        ):
            selected[task_id] = (
                started,
                screen,
                checked,
                record,
                run_directory,
            )
    screens = {task_id: screen for task_id, (_, screen, _, _, _) in selected.items()}
    records = {task_id: record for task_id, (_, _, _, record, _) in selected.items()}
    checked_records = {
        task_id: checked for task_id, (_, _, checked, _, _) in selected.items()
    }
    return screens, rejected, records, checked_records


def _failure_surface(screen: Mapping[str, Any], track: str) -> str | None:
    if screen.get("status") != "triage_required":
        return None
    if track == "clarify" and screen.get("asked") is False:
        return "clarification_or_grounding"
    if screen.get("premature_commitment") or screen.get("stale_commitment"):
        return "change_handling"
    if screen.get("false_update"):
        return "overreaction_to_update"
    return "grounding_search_or_web_execution"


def build_report(
    root: Path,
    *,
    experiment_ids: tuple[str, ...] | None = None,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    with site_data_request(root):
        task_list = load_construction_tasks(root, task_ids)
        requested = set(task_ids) if task_ids is not None else None
        tasks = {
            str(task["id"]): task for task in task_list
            if requested is None or task["id"] in requested
        }
        if requested is not None and requested - tasks.keys():
            raise ValueError(f"unknown qualification tasks: {sorted(requested - tasks.keys())}")
        triage_path = root / TRIAGE
        triage_rows: Mapping[str, Any] = {}
        triage_payload: dict[str, Any] = {}
        if triage_path.is_file():
            triage_payload = read_json_strict(triage_path)
            if isinstance(triage_payload.get("tasks"), dict):
                triage_rows = triage_payload["tasks"]
        development = DevelopmentEvidence(root)
        legacy_admissions = frozenset(
            task_id
            for task_id, task in tasks.items()
            if development.inherited_admission(task)
        )
        screens, rejected_runs, run_records, checked_records = _screen_records(
            root,
            tasks,
            experiment_ids,
            triage_rows,
            legacy_admissions,
        )
        families: dict[str, dict[str, Any]] = {}
        family_members: dict[str, list[str]] = {}
        for task_id, task in sorted(tasks.items()):
            family = qualification_family(
                task,
                source_record(task, root=root),
            )
            key = qualification_family_key(family)
            families[task_id] = {"key": key, **family}
            family_members.setdefault(key, []).append(task_id)
        rows: dict[str, dict[str, Any]] = {}

        for task_id, task in sorted(tasks.items()):
            screen = screens.get(
                task_id,
                {
                    "policy_version": QUALIFICATION_POLICY_VERSION,
                    "status": "not_run",
                    "next_action": "run_real_model_development_screen",
                    "e2e_success": None,
                },
            )
            if task_id in legacy_admissions and task_id not in screens:
                screen = {
                    **screen,
                    "status": "executable_demonstrated_once",
                    "next_action": "retain",
                    "evidence_basis": "retained_development_admission",
                }
            elif task_id not in screens and triage_rows.get(task_id):
                decision = development.decision(task)
                pending_status = {
                    "repair_task": "repair_required",
                    "verify_development_evidence": "evidence_review_required",
                }.get(decision.action)
                if pending_status is not None:
                    screen = {
                        **screen,
                        "status": pending_status,
                        "next_action": decision.action,
                        "reason": decision.reason,
                    }
            task_hash = task_sha256(task)
            rows[task_id] = {
                "task_sha256": task_hash,
                "track": task["track"],
                "site": task["site"],
                "qualification_family": families[task_id],
                "development_screen": {
                    **screen,
                    "required": task_id not in legacy_admissions,
                    "failure_surface": _failure_surface(screen, str(task["track"])),
                },
            }

        diagnostics: dict[str, dict[str, Any]] = {}
        attributions: list[dict[str, Any]] = []
        site_cache: dict[str, dict[str, Any]] = {}
        for task_id, screen in screens.items():
            task = tasks[task_id]
            record = run_records[task_id]
            checked = checked_records[task_id]
            triage = screen.get("coding_agent_triage")
            qualification = (
                str(triage.get("verdict")) if isinstance(triage, Mapping) else None
            )
            attribution = attribute_run(
                checked,
                record,
                qualification=qualification,
            )
            attribution["run_artifact"] = str(screen["run_artifact"])
            attributions.append(attribution)
            if screen.get("e2e_success") is True:
                continue
            site = str(task["site"])
            if site not in site_cache:
                site_cache[site] = load_site_data(root, site, require_assets=False)
            family = families[task_id]
            diagnostics[task_id] = failure_diagnostic(
                task,
                record,
                checked,
                family=family,
                related_task_ids=[
                    sibling
                    for sibling in family_members[family["key"]]
                    if sibling != task_id
                ],
                site_entities=site_cache[site]["entities"],
                run_artifact=str(screen["run_artifact"]),
            )

        prior: dict[str, Any] = {}
        if requested is not None and (root / REPORT).is_file():
            prior = read_json_strict(root / REPORT)
            if prior.get("schema_version") != 2:
                raise ValueError("unsupported existing qualification report")
            rows = {**prior["tasks"], **rows}
            diagnostics = {
                **{key: value for key, value in prior.get("diagnostics", {}).items()
                   if key not in requested},
                **diagnostics,
            }
            attributions = [
                value for value in prior.get("failure_attributions", [])
                if value["task_id"] not in requested
            ] + attributions
        statuses = Counter(row["development_screen"]["status"] for row in rows.values())
        selected_models = Counter(
            str(row["development_screen"]["model"])
            for row in rows.values()
            if isinstance(row["development_screen"].get("model"), str)
        )
        return {
            "schema_version": 2,
            "policy_version": QUALIFICATION_POLICY_VERSION,
            "experiments": sorted(
                set(prior.get("experiments", []))
                | (set(experiment_ids) if experiment_ids is not None else {
                    Path(screen["run_artifact"]).parts[2] for screen in screens.values()
                })
            ),
            "audit_scope": {
                "task_ids": sorted(requested) if requested is not None else None,
                "rejected_run_artifacts": "current_scan_only",
            },
            "used_for_task_repair": True,
            "valid_as_held_out_difficulty_evidence": False,
            "coding_agent_triage": TRIAGE,
            "interpretation": (
                "Reuse applicable development evidence. New or materially changed tasks "
                "need a reviewed execution or a named affected-scope review. "
                "Retaining task-validation evidence does not create a current model score. "
                "A model failure alone does not reject the task."
            ),
            "summary": {
                "tasks": len(rows),
                "statuses": dict(sorted(statuses.items())),
                "screened": sum(
                    isinstance(row["development_screen"].get("run_artifact"), str)
                    for row in rows.values()
                ),
                "pending": sum(
                    row["development_screen"]["status"] not in COMPLETED_STATUSES
                    for row in rows.values()
                ),
                "rejected_run_artifacts": dict(sorted(rejected_runs.items())),
                "diagnostic_packets": len(diagnostics),
                "qualification_families": len({
                    row["qualification_family"]["key"] for row in rows.values()
                }),
                "selected_models": dict(sorted(selected_models.items())),
            },
            "tasks": rows,
            "diagnostics": diagnostics,
            "failure_attributions": attributions,
        }
