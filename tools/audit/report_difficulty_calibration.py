"""Report registered development difficulty screening runs.

These runs may inform a later family-level redesign, so this report is never valid
held-out evidence and must never be consumed by task admission or final scoring.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from egocua.workspace import read_json_strict
from egocua.checker import rescore_run
from egocua.difficulty_calibration import (
    CALIBRATION_POLICY_VERSION,
    aggregate_calibration,
    failure_mode,
    protocol_mismatch,
    route_signature,
)
from egocua.experiment import ExperimentPlan, load_experiment_plan, scheduled_indexes
from egocua.failure_attribution import attribute_run
from egocua.qualification import trace_has_model_action, trace_has_provider_refusal
from egocua.run_ledger import (
    find_run_directories,
    read_run_identity,
    load_run_record,
    task_sha256,
)
from egocua.run_materials import material_mismatch
from egocua.schema import clarification_basis, load_tasks, task_subtype
from egocua.workspace import write_json_atomic

DEFAULT_PLAN = "configs/archive/difficulty-calibration-qwen3-5-flash.json"
DEFAULT_REPORT = "artifacts/reports/task-difficulty-calibration.json"


def _selected_records(
    root: Path,
    tasks: Mapping[str, dict[str, Any]],
    plan: ExperimentPlan,
) -> tuple[
    dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any], str]], Counter[str]
]:
    indexes = scheduled_indexes(tasks.values(), plan)
    candidates: dict[
        tuple[str, int], list[tuple[str, bool, dict[str, Any], dict[str, Any], str]]
    ] = {}
    rejected: Counter[str] = Counter()
    for run_directory in sorted(find_run_directories(root)):
        try:
            manifest = read_run_identity(run_directory)
            if manifest["experiment_id"] != plan.experiment_id:
                continue
            task_id = str(manifest.get("task_id") or "")
            replicate_id = manifest.get("replicate_id")
            task = tasks.get(task_id)
            if task is None or not isinstance(replicate_id, int):
                rejected["not_in_registered_matrix"] += 1
                continue
            key = (task_id, replicate_id)
            schedule_key = (task_id, "online", replicate_id)
            if schedule_key not in indexes:
                rejected["not_in_registered_matrix"] += 1
                continue
            record = load_run_record(run_directory)
            mismatch = protocol_mismatch(record, plan, indexes[schedule_key])
            mismatch = mismatch or material_mismatch(task, record, plan, root=root)
            if mismatch is not None:
                rejected[mismatch] += 1
                continue
            if trace_has_provider_refusal(record.get("trace")):
                rejected["provider_refusal"] += 1
                continue
            checked = _calibration_result(rescore_run(task, record), record)
            valid = bool(
                checked.get("run_valid") is True
                and checked.get("e2e_success") is not None
            )
            started = str(record.get("meta", {}).get("started_at") or "")
            artifact = run_directory.relative_to(root).as_posix()
            candidates.setdefault(key, []).append(
                (started, valid, record, checked, artifact)
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            rejected["invalid_artifact"] += 1

    selected: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any], str]] = {}
    for key, attempts in candidates.items():
        valid_attempts = [attempt for attempt in attempts if attempt[1]]
        chosen = (
            min(valid_attempts, key=lambda item: item[0])
            if valid_attempts
            else max(attempts, key=lambda item: item[0])
        )
        selected[key] = chosen[2], chosen[3], chosen[4]
    return selected, rejected


def _calibration_result(
    checked: dict[str, Any], record: Mapping[str, Any]
) -> dict[str, Any]:
    """Exclude runs where the model never acted; human and scripted runs are not calibrated."""
    if trace_has_model_action(record.get("trace")):
        return checked
    return {
        **checked,
        "run_valid": False,
        "e2e_success": None,
        "run_invalid_reasons": [*checked["run_invalid_reasons"], "missing_model_action"],
    }


def _task_rows(
    root: Path,
    tasks: Mapping[str, dict[str, Any]],
    plan: ExperimentPlan,
) -> tuple[dict[str, dict[str, Any]], Counter[str], list[dict[str, Any]]]:
    selected, rejected = _selected_records(root, tasks, plan)
    expected_replicates = tuple(plan.replicate_ids)
    difficulty_path = root / "artifacts/reports/task-difficulty.json"
    structural_rows = (
        read_json_strict(difficulty_path).get("tasks", {})
        if difficulty_path.is_file()
        else {}
    )
    rows: dict[str, dict[str, Any]] = {}
    attributions: list[dict[str, Any]] = []
    for task_id, task in sorted(tasks.items()):
        replicates: dict[str, dict[str, Any]] = {}
        routes: set[str] = set()
        steps: list[int] = []
        successes = 0
        valid_runs = 0
        for replicate_id in expected_replicates:
            key = (task_id, replicate_id)
            if key not in selected:
                replicates[str(replicate_id)] = {"status": "missing"}
                continue
            record, checked, artifact = selected[key]
            if (
                checked.get("run_valid") is not True
                or checked.get("e2e_success") is None
            ):
                attribution = attribute_run(checked, record)
                attribution["run_artifact"] = artifact
                attributions.append(attribution)
                replicates[str(replicate_id)] = {
                    "status": "invalid",
                    "run_artifact": artifact,
                    "failure_mode": "invalid_run",
                    "run_invalid_reasons": checked["run_invalid_reasons"],
                }
                continue
            success = checked.get("e2e_success") is True
            valid_runs += 1
            successes += success
            if isinstance(checked.get("steps"), int):
                steps.append(int(checked["steps"]))
            signature = route_signature(record.get("actions"))
            if signature:
                routes.add(signature)
            attribution = attribute_run(checked, record)
            attribution["run_artifact"] = artifact
            attributions.append(attribution)
            replicates[str(replicate_id)] = {
                "status": "valid",
                "success": success,
                "failure_mode": failure_mode(checked),
                "steps": checked.get("steps"),
                "run_artifact": artifact,
            }
        complete = valid_runs == len(expected_replicates)
        structural = structural_rows.get(task_id, {}).get("structural", {})
        rows[task_id] = {
            "task_sha256": task_sha256(task),
            "track": task["track"],
            "site": task["site"],
            "subtype": task_subtype(task),
            "clarification_basis": clarification_basis(task),
            "structural_band": structural.get("band", "unknown"),
            "expected_runs": len(expected_replicates),
            "valid_runs": valid_runs,
            "successes": successes,
            "complete": complete,
            "pass_at_1": round(successes / valid_runs, 6) if valid_runs else None,
            "pass_all_resets": successes == valid_runs if complete else None,
            "pass_any_reset": successes > 0 if complete else None,
            "mean_steps": round(sum(steps) / len(steps), 3) if steps else None,
            "unique_routes": len(routes),
            "replicates": replicates,
        }
    return rows, rejected, attributions


def _grouped(rows: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    keys = sorted({str(row[field]) for row in rows.values()})
    return {
        key: aggregate_calibration([row for row in rows.values() if str(row[field]) == key])
        for key in keys
    }


def build_report(root: Path, plan_path: Path) -> dict[str, Any]:
    plan = load_experiment_plan(plan_path)
    tasks = {str(task["id"]): task for task in load_tasks(root / "data/tasks")}
    rows, rejected, attributions = _task_rows(root, tasks, plan)
    summary = aggregate_calibration(list(rows.values()))
    summary.update(
        {
            "pending_runs": summary["expected_runs"] - summary["valid_runs"],
            "rejected_run_artifacts": dict(sorted(rejected.items())),
            "provisional": summary["complete_tasks"] != summary["tasks"],
        }
    )
    return {
        "schema_version": 1,
        "policy_version": CALIBRATION_POLICY_VERSION,
        "experiment_id": plan.experiment_id,
        "plan": plan_path.relative_to(root).as_posix(),
        "used_for_task_repair": True,
        "valid_as_held_out_difficulty_evidence": False,
        "interpretation": (
            "One-shot development difficulty screening under a later online release. "
            "It estimates collection-level completion and exposes concrete shortcuts, "
            "but cannot estimate stable per-task success probability or serve as held-out evidence."
        ),
        "registered_protocol": {
            "model": plan.model,
            "model_version": plan.model_version,
            "replicate_ids": list(plan.replicate_ids),
            "default_release_step": plan.default_release_step,
            "release_trigger": plan.release_trigger,
            "max_steps": plan.max_steps,
            "condition": plan.condition("online").as_dict(),
        },
        "summary": summary,
        "by_track": _grouped(rows, "track"),
        "by_site": _grouped(rows, "site"),
        "by_subtype": _grouped(rows, "subtype"),
        "by_clarification_basis": _grouped(
            {
                task_id: row
                for task_id, row in rows.items()
                if row["clarification_basis"] is not None
            },
            "clarification_basis",
        ),
        "by_structural_band": _grouped(rows, "structural_band"),
        "failure_attributions": attributions,
        "tasks": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--output", default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    report = build_report(root, root / args.plan)
    write_json_atomic(root / args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
