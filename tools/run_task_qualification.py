"""Run the next bounded BrowserGym development qualification batch.

This is development evidence.  It may expose tasks that need repair, but it must not
be reported as post-freeze benchmark difficulty.  Qualification runs at most ten
independent trials together.  One unresolved batch blocks the next batch until every
failure is diagnosed; an unseen mechanism family contributes at most one canary.
Each batch is frozen and executed like any other experiment, under the formal
online protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from egocua.construction import load_construction_tasks
from egocua.controlled_experiments import (
    export_study,
    load_validation_study,
    output_labels,
    read_study,
)
from egocua.execution import run_isolated
from egocua.experiment_output import experiment_folder_name
from egocua.qualification import (
    qualification_family,
    qualification_family_key,
)
from egocua.qualification_report import build_report
from egocua.retest_scope import load_repair_task_ids
from egocua.study_execution import EvaluationSettings, run_exported_evaluation
from egocua.synthesis_reports import REPORTS
from egocua.synthesis_workflow import SynthesisWorkflow
from egocua.workspace import write_json_atomic

ACCEPTANCE_REPORT = REPORTS["acceptance"].path
CONFIDENCE_REPORT = REPORTS["confidence"].path
REPORT = Path("artifacts/reports/task-qualification.json")
DEFAULT_CONCURRENCY = 10


BLOCKING_STATUSES = {
    "triage_required",
    "trajectory_review_required",
    "evidence_review_required",
    "repair_required",
}
RUNNABLE_STATUSES = {"not_run", "invalid_run"}


def _blocking_task_ids(
    report: dict[str, Any],
    task_ids: set[str] | frozenset[str] | None = None,
) -> list[str]:
    return [
        task_id
        for task_id, row in report["tasks"].items()
        if task_ids is None or task_id in task_ids
        if row["development_screen"].get("required") is not False
        if row["development_screen"]["status"] in BLOCKING_STATUSES
    ]


def _family_key(task: dict[str, Any], row: dict[str, Any]) -> str:
    family = row.get("qualification_family")
    if isinstance(family, dict) and isinstance(family.get("key"), str):
        return str(family["key"])
    if {"track", "site", "goal"} <= set(task):
        return qualification_family_key(qualification_family(task))
    return "unclassified"


def _select_tasks(
    tasks: list[dict[str, Any]],
    report: dict[str, Any],
    task_id: str | None,
    *,
    concurrency: int,
    requested_task_ids: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    rows = report["tasks"]
    if task_id is not None:
        matches = [task for task in tasks if str(task["id"]) == task_id]
        if not matches:
            raise ValueError(f"unknown task id: {task_id}")
        status = rows[task_id]["development_screen"]["status"]
        if status not in RUNNABLE_STATUSES:
            raise ValueError(f"{task_id}: status {status!r} is not runnable")
        return matches
    runnable = [
        task
        for task in tasks
        if requested_task_ids is None or str(task["id"]) in requested_task_ids
        if rows[str(task["id"])]["development_screen"]["status"] in RUNNABLE_STATUSES
        if requested_task_ids is not None
        or rows[str(task["id"])]["development_screen"].get("required") is not False
    ]
    if not runnable:
        return []

    screened_families = {
        str(row["qualification_family"]["key"])
        for row in rows.values()
        if row["development_screen"]["status"] not in RUNNABLE_STATUSES
        and isinstance(row.get("qualification_family"), dict)
        and isinstance(row["qualification_family"].get("key"), str)
    }
    unseen: list[dict[str, Any]] = []
    screened: list[dict[str, Any]] = []
    selected_unseen_families: set[str] = set()
    for task in runnable:
        key = _family_key(task, rows[str(task["id"])])
        if key in screened_families:
            screened.append(task)
        elif key not in selected_unseen_families:
            unseen.append(task)
            selected_unseen_families.add(key)
    return (unseen + screened)[:concurrency]


def _failed_runs(directory: Path) -> dict[str, str]:
    return {
        str(outcome["run"]["task_id"]): f"{outcome['error']['type']}: {outcome['error']['message']}"
        for outcome in read_study(directory).get("execution_results", [])
        if outcome.get("valid") is not True and "error" in outcome
    }


async def _run_batch(
    *,
    root: Path,
    tasks: list[dict[str, Any]],
    task_id: str | None,
    requested_task_ids: set[str] | frozenset[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    if concurrency < 1 or concurrency > DEFAULT_CONCURRENCY:
        raise ValueError(f"concurrency must be between 1 and {DEFAULT_CONCURRENCY}")
    known_ids = {str(task["id"]) for task in tasks}
    unknown = sorted((requested_task_ids or set()) - known_ids)
    if unknown:
        raise ValueError(f"unknown task ids in repair scope: {unknown}")
    active_scope = requested_task_ids
    if task_id is not None:
        active_scope = frozenset({task_id})

    previous = build_report(
        root, task_ids=sorted(active_scope) if active_scope else None
    )
    blocking = _blocking_task_ids(previous, active_scope)
    if blocking:
        return {
            "status": "blocked_by_unresolved_trial",
            "blocking_task_id": blocking[0],
            "blocking_count": len(blocking),
            "report": previous,
        }
    selected = _select_tasks(
        tasks,
        previous,
        task_id,
        concurrency=concurrency,
        requested_task_ids=requested_task_ids,
    )
    if not selected:
        return {"status": "complete", "report": previous}
    selected_ids = [str(task["id"]) for task in selected]
    SynthesisWorkflow(root).require_development_inputs(selected_ids)
    arms = load_validation_study(root, selected)
    directory = root / "experiments" / experiment_folder_name(output_labels(arms))
    export_study(arms, directory, root=root)
    try:
        await run_exported_evaluation(
            directory,
            EvaluationSettings(concurrency=len(selected), run_purpose="task_validation"),
            root,
        )
    except Exception:  # failed artifacts remain available for diagnosis
        if not _failed_runs(directory):
            raise
    errors = _failed_runs(directory)
    report = build_report(root, task_ids=selected_ids)
    trial_statuses = {
        selected_id: report["tasks"][selected_id]["development_screen"]["status"]
        for selected_id in selected_ids
    }
    return {
        "status": "batch_finished" if not errors else "batch_error",
        "experiment_directory": str(directory),
        "task_ids": selected_ids,
        "trial_statuses": trial_statuses,
        "errors": errors,
        "report": report,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="run one task, or repeat to limit a qualification batch to named tasks",
    )
    scope.add_argument(
        "--repair-plan",
        help="run only task IDs selected by a completion-anomaly repair plan",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        choices=range(1, DEFAULT_CONCURRENCY + 1),
        help="maximum simultaneous trials; one named task still runs alone",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    named_ids = args.task_id
    if len(named_ids) != len(set(named_ids)):
        parser.error("--task-id values must be unique")
    single_task_id = named_ids[0] if len(named_ids) == 1 else None
    requested_task_ids = None
    if len(named_ids) > 1:
        requested_task_ids = frozenset(named_ids)
    elif args.repair_plan:
        requested_task_ids = load_repair_task_ids(root / args.repair_plan)
    candidate_ids = named_ids or sorted(requested_task_ids or [])
    tasks = load_construction_tasks(root, candidate_ids)
    result = run_isolated(
        _run_batch(
            root=root,
            tasks=tasks,
            task_id=single_task_id,
            requested_task_ids=requested_task_ids,
            concurrency=args.concurrency,
        )
    )
    write_json_atomic(root / REPORT, result["report"])
    output = {key: value for key, value in result.items() if key != "report"}
    output["qualification_summary"] = result["report"]["summary"]
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if result["status"] == "batch_error":
        return 1
    if result["status"] == "blocked_by_unresolved_trial":
        return 2
    if any(
        status in BLOCKING_STATUSES
        for status in result.get("trial_statuses", {}).values()
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
