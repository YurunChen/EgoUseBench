"""Development-only empirical calibration for task difficulty.

Calibration samples may expose an over-easy mechanism and therefore may influence
construction.  They are intentionally disqualified from held-out reporting.  This
module keeps that lifecycle distinction explicit while validating every run against
one registered plan.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .experiment import ExperimentPlan, protocol_mismatch as run_protocol_mismatch
from .checker import behavioral_failure_mode

from .adapters.browsergym import TRANSPORT

CALIBRATION_POLICY_VERSION = "development-difficulty-screening-v2"


def protocol_mismatch(
    record: Mapping[str, Any],
    plan: ExperimentPlan,
    schedule_index: int,
) -> str | None:
    """Return the first registered-protocol mismatch for one calibration run."""

    mismatch = run_protocol_mismatch(record, plan, schedule_index)
    if mismatch is not None:
        return mismatch
    if record["meta"].get("transport") != TRANSPORT:
        return "wrong_transport"
    if record["condition"] != plan.condition("online").as_dict():
        return "wrong_condition"
    return None


def failure_mode(checked: Mapping[str, Any]) -> str:
    """Map Checker facts to one stable, non-semantic behavioral failure bucket."""

    return behavioral_failure_mode(checked)


def route_signature(actions: Any) -> str:
    """Describe route diversity without retaining queries or hidden entity ids."""

    if not isinstance(actions, list):
        return ""
    action_types = [
        str(action.get("type"))
        for action in actions
        if isinstance(action, Mapping) and isinstance(action.get("type"), str)
    ]
    return ">".join(action_types)


def aggregate_calibration(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate task rows while keeping missing/invalid trials out of completion."""

    valid_runs = sum(int(row.get("valid_runs", 0)) for row in rows)
    successes = sum(int(row.get("successes", 0)) for row in rows)
    complete = [row for row in rows if row.get("complete") is True]
    pass_all = sum(row.get("pass_all_resets") is True for row in complete)
    pass_any = sum(row.get("pass_any_reset") is True for row in complete)
    modes = Counter(
        str(run["failure_mode"])
        for row in rows
        for run in row.get("replicates", {}).values()
        if isinstance(run, Mapping) and run.get("status") == "valid"
    )
    return {
        "tasks": len(rows),
        "complete_tasks": len(complete),
        "valid_runs": valid_runs,
        "expected_runs": sum(int(row.get("expected_runs", 0)) for row in rows),
        "successes": successes,
        "pass_at_1": round(successes / valid_runs, 6) if valid_runs else None,
        "pass_all_resets": round(pass_all / len(complete), 6) if complete else None,
        "pass_any_reset": round(pass_any / len(complete), 6) if complete else None,
        "failure_modes": dict(sorted(modes.items())),
    }
