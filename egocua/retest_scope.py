"""Describe completion evidence without turning model success into task defects."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

COMPLETION_ANOMALY_POLICY_VERSION = "task-completion-observation-v2"
LEGACY_REPAIR_SCOPE_POLICY = "task-local-exact-binomial-v1"


def binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    """Return P(X >= successes) for X ~ Binomial(trials, probability)."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("successes and trials must satisfy 0 <= successes <= trials")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    return sum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )


def completion_anomaly(row: Mapping[str, Any]) -> dict[str, Any]:
    """Report observed outcomes; a success rate cannot authorize a task repair."""

    valid_runs = int(row.get("valid_runs", 0))
    expected_runs = int(row.get("expected_runs", 0))
    successes = int(row.get("successes", 0))
    if not 0 <= successes <= valid_runs <= expected_runs:
        raise ValueError("completion counts must satisfy 0 <= successes <= valid <= expected")
    complete = row.get("complete") is True and valid_runs == expected_runs and valid_runs > 0
    return {
        "status": "observed" if complete else "insufficient_evidence",
        "modify_task": False,
        "valid_runs": valid_runs,
        "successes": successes,
        "success_rate": successes / valid_runs if valid_runs else None,
    }


def build_completion_anomaly_plan(
    calibration: Mapping[str, Any],
    *,
    source_path: str,
) -> dict[str, Any]:
    """Keep the report entry point without deriving repair work from completion rates."""

    calibration_rows = calibration.get("tasks")
    if not isinstance(calibration_rows, Mapping):
        raise ValueError("difficulty calibration report has no task rows")
    rows: dict[str, dict[str, Any]] = {}
    for task_id, raw_row in sorted(calibration_rows.items()):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"{task_id}: invalid difficulty calibration row")
        decision = completion_anomaly(raw_row)
        rows[str(task_id)] = {
            "task_sha256": raw_row.get("task_sha256"),
            "track": raw_row.get("track"),
            "site": raw_row.get("site"),
            "subtype": raw_row.get("subtype"),
            **decision,
        }
    return {
        "schema_version": 1,
        "policy_version": COMPLETION_ANOMALY_POLICY_VERSION,
        "source": source_path,
        "policy": {
            "unit": "task",
            "purpose": "describe_observed_completion",
            "repair_requires": "reviewed_task_or_site_defect",
            "qualification_run_is_difficulty_evidence": False,
        },
        "summary": {
            "tasks": len(rows),
            "tasks_with_complete_evidence": sum(
                row["status"] != "insufficient_evidence" for row in rows.values()
            ),
            "repair_tasks": 0,
        },
        "repair_task_ids": [],
        "tasks": rows,
    }


def load_repair_task_ids(path: str | Path) -> frozenset[str]:
    """Read an explicitly supplied scope, including preserved historical plans."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("policy_version") not in {
            COMPLETION_ANOMALY_POLICY_VERSION, LEGACY_REPAIR_SCOPE_POLICY
        }
    ):
        raise ValueError("unsupported completion-anomaly repair plan")
    task_ids = payload.get("repair_task_ids")
    if not isinstance(task_ids, list) or any(
        not isinstance(task_id, str) or not task_id for task_id in task_ids
    ):
        raise ValueError("completion-anomaly repair plan has invalid task IDs")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("completion-anomaly repair plan contains duplicate task IDs")
    return frozenset(task_ids)
