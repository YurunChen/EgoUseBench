"""Aggregate PULSE attributions into the published failure report."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .failure_attribution import (
    PULSE_POLICY_VERSION,
    PULSE_STAGES,
    REVIEW_PACKET_FIELDS,
    apply_assessments,
    validate_attribution,
    verify_fault_references,
    verify_trajectory_review,
    historical_review_context,
    qualify_attribution_row,
    validate_judgment,
)
from .judge_validation import (
    reliability_report,
    select_reliability_sample,
    validate_judge_sample,
    judge_validation_result,
)


def _verify_derived_judge(row: Mapping[str, Any], root: Path | None) -> None:
    if root is None:
        raise ValueError("Judge evidence requires historical workspace")
    evidence = row["rule_evidence"]
    validate_judgment(
        historical_review_context(row, root),
        {
            "stage": row["candidate_stage"],
            "step": row["candidate_step"],
            "confidence": row["candidate_confidence"],
            "reason": evidence["reason"],
            "evidence": evidence["evidence_refs"],
            "uncertainty": evidence["uncertainty_reason"],
        },
    )


def _group_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_failures = [
        row for row in rows if row.get("run_verdict") == "valid_model_failure"
    ]
    known = [row for row in valid_failures if row.get("pulse_stage") in PULSE_STAGES]
    unknown = [
        row
        for row in valid_failures
        if row.get("attribution_status") == "evidence_insufficient"
    ]
    pending = [
        row
        for row in valid_failures
        if row.get("attribution_status")
        in {
            "coding_agent_pending",
            "rule_evaluation_pending",
        }
    ]
    successes = sum(row.get("run_verdict") == "success" for row in rows)
    summary: dict[str, Any] = {
        "runs": len(rows),
        "valid_runs": successes + len(valid_failures),
        "successes": successes,
        "invalid_runs": sum(row.get("run_verdict") == "invalid_run" for row in rows),
        "benchmark_faults": sum(
            row.get("run_verdict") == "benchmark_fault" for row in rows
        ),
        "unresolved_run_validity": sum(
            row.get("run_verdict") == "unknown" for row in rows
        ),
        "valid_model_failures": len(valid_failures),
        "unknown_failures": len(unknown),
        "unknown_reasons": dict(
            Counter(
                (row.get("rule_evidence") or {}).get("uncertainty_reason", "unrecorded")
                for row in unknown
            )
        ),
        "pending_attributions": len(pending),
        "resolved_attributions": len(known),
        "resolved_coverage": len(known) / len(valid_failures)
        if valid_failures
        else None,
        "attribution_coverage": (
            (len(known) + len(unknown)) / len(valid_failures)
            if valid_failures
            else None
        ),
        "stages": {},
    }
    for stage in PULSE_STAGES:
        count = sum(row.get("pulse_stage") == stage for row in known)
        summary["stages"][stage] = {
            "count": count,
            "share_of_failures": count / len(valid_failures)
            if valid_failures
            else None,
            "reached_stage": None,
            "conditional_failure_rate": None,
        }
    summary["conditional_rate_status"] = "not_measured_from_stage_labels"
    summary["unknown_share"] = (
        len(unknown) / len(valid_failures) if valid_failures else None
    )
    return summary


def _bootstrap_group(
    rows: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id") or row.get("run_id"))].append(row)
    task_ids = sorted(by_task)
    if not task_ids or samples <= 0:
        return {}
    rng = random.Random(seed)
    estimates: dict[tuple[str, str], list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled = [
            row
            for task_id in rng.choices(task_ids, k=len(task_ids))
            for row in by_task[task_id]
        ]
        summary = _group_summary(sampled)
        for stage in PULSE_STAGES:
            value = summary["stages"][stage]["share_of_failures"]
            if value is not None:
                estimates[(stage, "share_of_failures")].append(float(value))
    intervals: dict[str, Any] = {stage: {} for stage in PULSE_STAGES}
    for (stage, metric), values in estimates.items():
        values.sort()
        last = len(values) - 1
        intervals[stage][f"{metric}_95"] = [
            values[int(last * 0.025)],
            values[int(last * 0.975)],
        ]
    return intervals


def _matrix_label(row: Mapping[str, Any], dimension: str) -> str:
    if dimension == "model_setting":
        return f"{row['model']} / {row.get('setting', 'unknown')}"
    return str(row.get(dimension) or "unknown")


def _aggregate_attributions(
    rows: Sequence[Mapping[str, Any]], *, bootstrap_samples: int = 2000, seed: int = 0
) -> dict[str, Any]:
    """Build model, setting, task-type, and website summaries without mixing settings."""

    for row in rows:
        validate_attribution(row)
    matrices: dict[str, Any] = {}
    dimensions = ("model", "track", "site", "setting", "model_setting")
    for dimension_index, dimension in enumerate(dimensions):
        matrix: dict[str, Any] = {}
        groups = sorted({_matrix_label(row, dimension) for row in rows})
        for group_index, group in enumerate(groups):
            subset = [
                row for row in rows if _matrix_label(row, dimension) == group
            ]
            summary = _group_summary(subset)
            intervals = _bootstrap_group(
                subset,
                samples=bootstrap_samples,
                seed=seed + dimension_index * 1009 + group_index,
            )
            for stage in PULSE_STAGES:
                summary["stages"][stage].update(intervals.get(stage, {}))
            matrix[group] = summary
        matrices[dimension] = matrix
    overall = _group_summary(rows)
    return {
        "schema_version": 1,
        "policy_version": PULSE_POLICY_VERSION,
        "summary": overall,
        "matrices": matrices,
    }


def _apply_human_adjudications(
    rows: list[dict[str, Any]], adjudications: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_id = {row["run_id"]: row for row in rows}
    if set(adjudications) - by_id.keys():
        raise ValueError("human adjudication references an unknown run")
    result = []
    for row in rows:
        decision = adjudications.get(row["run_id"])
        if decision is None:
            result.append(row)
            continue
        if (row["run_verdict"] != "valid_model_failure"
                or row["attribution_status"] == "resolved"
                or decision.get("stage") not in PULSE_STAGES
                or not decision.get("reviewer_id")
                or not decision.get("reason")):
            raise ValueError("invalid human adjudication")
        updated = dict(row)
        updated.update(
            pulse_stage=decision["stage"], mechanism=None, first_decisive_step=None,
            candidate_stage=decision["stage"], candidate_mechanism=None,
            candidate_step=None, candidate_source="human",
            candidate_confidence="adjudicated", candidate_unknown_boundary=None,
            attribution_method="human_adjudication", confidence="adjudicated",
            attribution_status="resolved", review_required=False,
            rule_evidence={"policy_version": "human-adjudication-v1",
                           "reviewer_id": decision["reviewer_id"],
                           "reason": decision["reason"],
                           "original_evidence": row.get("rule_evidence")},
        )
        validate_attribution(updated)
        result.append(updated)
    return result


def build_failure_report(
    source: Mapping[str, Any],
    evidence: Mapping[str, Any] | None = None,
    *,
    root: str | Path | None = None,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Build the one paper-facing PULSE report."""

    workspace = Path(root).resolve() if root else None
    evidence = evidence or {}
    raw_rows = source.get("failure_attributions")
    if not isinstance(raw_rows, list):
        raise ValueError("source report has no failure_attributions")
    source_exclusions = source.get("source_exclusions", [])
    if not isinstance(source_exclusions, list):
        raise ValueError("source_exclusions must be a list")
    excluded_ids: set[str] = set()
    for excluded in source_exclusions:
        if not isinstance(excluded, Mapping) or any(
            not excluded.get(key)
            for key in ("run_id", "task_id", "run_artifact", "reasons")
        ):
            raise ValueError(
                "source exclusion requires run, task, artifact and reasons"
            )
        if excluded.get("status") != "not_rescorable" or not isinstance(
            excluded["reasons"], list
        ):
            raise ValueError(
                "source exclusion requires not_rescorable status and reason list"
            )
        if excluded["run_id"] in excluded_ids:
            raise ValueError("duplicate source exclusion run")
        excluded_ids.add(excluded["run_id"])
    counterfactuals = evidence.get("counterfactuals", {})
    assessments = evidence.get("assessments", [])
    ratings = evidence.get("ratings", [])
    if not isinstance(counterfactuals, Mapping):
        raise ValueError("counterfactuals must be an object keyed by run_id")
    if not isinstance(ratings, list):
        raise ValueError("ratings must be a list")
    if not isinstance(assessments, list):
        raise ValueError("assessments must be a list")

    rows: list[dict[str, Any]] = []
    known_runs: set[str] = set()
    reviewed_run_ids = {
        item["run_id"]
        for item in assessments
        if isinstance(item, Mapping) and item.get("run_id")
    }
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("failure attribution rows must be objects")
        row = dict(raw)
        if row.get("policy_version") in {"pulse-v4", "pulse-v5"}:
            raise ValueError(
                "historical attributions require final-state rescoring before v6 review"
            )
        validate_attribution(row)
        verify_fault_references(row, workspace)
        run_id = str(row["run_id"])
        if run_id in known_runs:
            raise ValueError(f"duplicate failure source run: {run_id}")
        if run_id in excluded_ids:
            raise ValueError("source run cannot be both scored and excluded")
        known_runs.add(run_id)
        row = qualify_attribution_row(
            row, evidence.get("run_qualifications", {}).get(run_id), workspace
        )
        if (
            row.get("attribution_method") == "trajectory_review"
            and run_id not in reviewed_run_ids
        ):
            if row["rule_evidence"]["policy_version"] == "trajectory-judge-v2":
                _verify_derived_judge(row, workspace)
            else:
                verify_trajectory_review(row, row["rule_evidence"], workspace)
        rows.append(row)
    rows = apply_assessments(
        rows,
        assessments,
        workspace,
    )
    automatic_rows = rows
    adjudications = evidence.get("adjudications", {})
    if not isinstance(adjudications, Mapping):
        raise ValueError("adjudications must be an object keyed by run_id")
    rows = _apply_human_adjudications(rows, adjudications)
    if counterfactuals:
        raise ValueError(
            "PULSE uses trajectory reviews; diagnostic reruns are separate"
        )

    sampled = select_reliability_sample(automatic_rows, seed=seed)
    sample_ids = {str(row["run_id"]) for row in sampled}
    selection = evidence.get("judge_validation")
    if selection is not None:
        validate_judge_sample(selection, known_runs)
        sample_ids.update(selection["calibration_run_ids"])
        sample_ids.update(selection["validation_run_ids"])
    sample_ids.update(str(rating["run_id"]) for rating in ratings)
    sampled = [row for row in rows if row["run_id"] in sample_ids]
    reliability = reliability_report(ratings, automatic_rows)
    sample_complete = bool(rows) and reliability["reviewed_cases"] == len(sampled)
    attribution_complete = (
        bool(rows)
        and not source_exclusions
        and not any(
            row["attribution_status"]
            in {"coding_agent_pending", "rule_evaluation_pending"}
            for row in rows
        )
    )
    reliability["sample_complete"] = sample_complete
    reliability["required_cases"] = len(sampled)
    reliability["acceptance"] = {
        "high_confidence_automatic": (
            sample_complete
            and attribution_complete
            and reliability["automatic_vs_human_agreement"] is not None
            and reliability["automatic_vs_human_agreement"] >= 0.85
        ),
    }
    judge_validation = judge_validation_result(evidence, rows, workspace)
    qualification_pending = [str(row["run_id"]) for row in rows if row["run_verdict"] == "unknown"]
    packet_ids = sample_ids | set(qualification_pending)
    packets = []
    for row in rows:
        if row["run_id"] not in packet_ids:
            continue
        packet = {key: row.get(key) for key in REVIEW_PACKET_FIELDS}
        packets.append(packet)
    return {
        "schema_version": 1,
        "scope": "model_evaluation_only",
        "policy_version": PULSE_POLICY_VERSION,
        "attributions": rows,
        "source_exclusions": deepcopy(source_exclusions),
        "source_summary": {
            "runs": len(rows) + len(source_exclusions),
            "scored_runs": len(rows),
            "not_rescorable": len(source_exclusions),
        },
        "workflow": {
            "processing_complete": attribution_complete,
            "source_blocked_run_ids": sorted(excluded_ids),
            "resolved_complete": bool(rows)
            and not source_exclusions
            and all(
                row["run_verdict"] != "unknown"
                and row["attribution_status"] in {"resolved", "excluded"}
                for row in rows
            ),
            "qualification_pending_run_ids": qualification_pending,
            "human_review_complete": sample_complete,
            "attribution_complete": attribution_complete,
            "coding_agent_pending_run_ids": [
                str(row["run_id"])
                for row in rows
                if row.get("attribution_status") == "coding_agent_pending"
            ],
            "rule_evaluation_pending_run_ids": [
                str(row["run_id"])
                for row in rows
                if row.get("attribution_status") == "rule_evaluation_pending"
            ],
            "evidence_insufficient_run_ids": [
                str(row["run_id"])
                for row in rows
                if row.get("attribution_status") == "evidence_insufficient"
            ],
            "resolved_run_ids": [
                str(row["run_id"])
                for row in rows
                if row.get("attribution_status") == "resolved"
            ],
        },
        "statistics": _aggregate_attributions(
            rows, bootstrap_samples=bootstrap_samples, seed=seed
        ),
        "human_review": {
            "required_run_ids": [
                str(row["run_id"]) for row in rows if row["review_required"]
            ],
            "reliability_sample_run_ids": [str(row["run_id"]) for row in sampled],
            "packets": packets,
        },
        "reliability": reliability,
        "judge_validation": judge_validation,
    }
