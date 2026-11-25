"""Execute a frozen controlled-experiment study once per unique input."""

from __future__ import annotations

import asyncio
import json
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checker import check_run
from .controlled_experiments import (
    DEFAULT_EVALUATION_CONCURRENCY,
    MAX_EVALUATION_CONCURRENCY,
    load_frozen_study,
    read_study,
)
from .execution import exception_details, request_rejection_kind, require_stream_input
from .run_ledger import load_run_record
from .run_materials import require_frozen_assets
from .study_summary import summarize_records, unresolved_request_failures
from .synthesis_workflow import TaskStatus
from .workspace import write_json_atomic


@dataclass(frozen=True)
class EvaluationSettings:
    canary_index: int = 0
    concurrency: int = DEFAULT_EVALUATION_CONCURRENCY
    run_purpose: str | None = None
    deferred_execution_indexes: tuple[int, ...] = ()
    execution_indexes: tuple[int, ...] | None = None


async def run_exported_evaluation(
    directory: Path, settings: EvaluationSettings, root: Path
) -> dict[str, Any]:
    """Run a canary then consume each unique execution once."""
    from .adapters.browsergym import (
        ModelCallStop,
        is_request_transport_error,
        run_task,
        service_slot,
        service_count,
    )

    if not (directory / "plan.json").is_file():
        raise ValueError(
            "historical studies are read-only; export a new experiment with --reuse-from before execution"
        )
    if not 1 <= settings.concurrency <= MAX_EVALUATION_CONCURRENCY:
        raise ValueError(
            f"evaluation concurrency must be between 1 and {MAX_EVALUATION_CONCURRENCY}"
        )
    definition = read_study(directory)
    if definition.get("superseded_by") or (definition.get("review_gate") or {}).get("superseded_by"):
        raise ValueError("study is superseded; resume its replacement and preserve this archive")
    registered_purpose = definition.get("run_purpose")
    if settings.run_purpose and registered_purpose not in (None, settings.run_purpose):
        raise ValueError("cannot override the frozen study's run purpose")
    run_purpose = settings.run_purpose or registered_purpose
    schedule = definition["execution_schedule"]
    if not 0 <= settings.canary_index < len(schedule):
        raise IndexError(f"unknown canary index: {settings.canary_index}")
    unknown_deferred = set(settings.deferred_execution_indexes) - {
        entry["execution_index"] for entry in schedule
    }
    if unknown_deferred:
        raise IndexError(
            f"unknown deferred execution indexes: {sorted(unknown_deferred)}"
        )
    arms, evidence, _gate = load_frozen_study(directory)
    by_name = {arm.name: arm for arm in arms}
    for arm in arms:
        require_frozen_assets(arm.tasks, arm.plan)
    service_stop = ModelCallStop()
    failed_requests: list[int] = []
    outcomes: list[dict[str, Any]] = definition.get("execution_results", [])
    completed: dict[int, tuple[str, dict[str, Any]]] = {}
    for outcome in outcomes:
        if outcome.get("valid") is True:
            record = load_run_record(Path(outcome["artifact_directory"]))
            if not check_run(record["task_snapshot"], record)["run_valid"]:
                raise ValueError(
                    "saved completed execution does not have a valid run archive"
                )
            completed[outcome["execution_index"]] = (
                outcome["run"]["experiment_id"],
                record,
            )
    records = list(completed.values())
    if records:
        summarize_records(
            arms, records, root=root, evidence=evidence, execution_results=outcomes
        )
    pending = [entry for entry in schedule if entry["execution_index"] not in completed]
    rejected_indexes = {
        row["execution_index"]
        for row in unresolved_request_failures(outcomes)
        if row["kind"] in ("moderation", "context_limit")
    }
    if settings.execution_indexes is None:
        pending = [
            entry for entry in pending
            if entry["execution_index"] not in rejected_indexes
        ]
    if settings.execution_indexes is not None:
        selected = set(settings.execution_indexes)
        unknown = selected - {entry["execution_index"] for entry in schedule}
        if unknown:
            raise IndexError(f"unknown selected execution indexes: {sorted(unknown)}")
        pending = [entry for entry in pending if entry["execution_index"] in selected]
    if any(
        by_name[entry["run"]["arm"]].plan.adapter != "browsergym" for entry in pending
    ):
        raise ValueError(
            "Frozen study uses a retired browser adapter; export a BrowserGym study before new execution"
        )
    deferred = [
        entry
        for entry in pending
        if entry["execution_index"] in settings.deferred_execution_indexes
    ]
    pending = [entry for entry in pending if entry not in deferred]
    task_statuses = (definition.get("review_gate") or {}).get(
        "formal_task_statuses", {}
    )
    blocked = [
        entry
        for entry in pending
        if task_statuses.get(entry["run"]["task_id"]) == TaskStatus.NEEDS_REPAIR
    ]
    pending = [entry for entry in pending if entry not in blocked]
    for entry in pending:
        run = entry["run"]
        plan = by_name[run["arm"]].plan
        require_stream_input(
            plan.condition(run["condition"]), plan.release_trigger, plan.max_steps,
            track=run["track"],
        )
    definition["execution_results"] = outcomes
    if run_purpose is not None:
        definition["run_purpose"] = run_purpose
    workers: list[asyncio.Task[None]] = []

    def save_progress() -> None:
        definition["execution_stop"] = (
            None
            if service_stop.error is None
            else {
                "type": type(service_stop.error).__name__,
                "message": str(service_stop.error),
            }
        )
        write_json_atomic(
            directory / "results.json",
            {
                key: definition[key]
                for key in ("execution_results", "execution_stop", "run_purpose")
                if key in definition
            },
        )

    async def execute(entry: dict[str, Any]) -> None:
        service_stop.require_running()
        run = entry["run"]
        arm = by_name[run["arm"]]
        task = next(task for task in arm.tasks if task["id"] == run["task_id"])
        plan = arm.plan
        outcome = {
            "execution_index": entry["execution_index"],
            "run": run,
            "artifact_directory": None,
        }
        outcomes.append(outcome)
        save_progress()
        print(
            json.dumps(
                {
                    "execution_started": entry["execution_index"],
                    "task_id": run["task_id"],
                    "condition": run["condition"],
                }
            ),
            flush=True,
        )
        try:
            record = await run_task(
                task,
                plan.condition(run["condition"]),
                **plan.run_parameters(task, run),
                asset_root=root,
                output_root=root / "experiments",
                experiment_directory=directory,
                service_stop=service_stop,
                run_purpose=run_purpose,
            )
            outcome["artifact_directory"] = record["meta"]["artifact_directory"]
            outcome["run_id"] = record["meta"]["run_id"]
            checked = check_run(task, record)
            outcome["valid"] = checked["run_valid"]
            outcome["success"] = checked["e2e_success"]
            if not checked["run_valid"]:
                raise RuntimeError(f"invalid model execution: {run['task_id']}")
            records.append((run["experiment_id"], record))
            print(
                json.dumps(
                    {
                        "execution_completed": entry["execution_index"],
                        "valid": True,
                        "success": checked["e2e_success"],
                        "completed": len(records),
                    }
                ),
                flush=True,
            )
        except BaseException as error:
            outcome["valid"] = False
            outcome["error"] = exception_details(error)
            if artifact := getattr(error, "run_artifact", None):
                outcome.update(artifact)
            rejection = request_rejection_kind(outcome["error"])
            if rejection or is_request_transport_error(error):
                failed_requests.append(entry["execution_index"])
                if rejection:
                    outcome["input_rejection"] = rejection
                else:
                    outcome["request_transport_error"] = True
                print(
                    json.dumps(
                        {
                            "execution_failed": entry["execution_index"],
                            "error": outcome["error"],
                        }
                    ),
                    flush=True,
                )
                return
            service_stop.stop(error)
            raise
        finally:
            save_progress()

    async def consume(queue: asyncio.Queue[dict[str, Any]]) -> None:
        while not queue.empty():
            service_stop.require_running()
            await execute(queue.get_nowait())

    loop = asyncio.get_running_loop()
    evaluation_task = asyncio.current_task()
    assert evaluation_task is not None

    def stop_on_signal(stop_signal: signal.Signals) -> None:
        error = InterruptedError(f"received {stop_signal.name}")
        service_stop.stop(error)
        print(json.dumps({"execution_interrupted": str(error)}), flush=True)
        evaluation_task.cancel(str(error))

    previous_handlers = {
        stop_signal: signal.getsignal(stop_signal)
        for stop_signal in (signal.SIGTERM, signal.SIGHUP)
    }
    for stop_signal in previous_handlers:
        loop.add_signal_handler(stop_signal, stop_on_signal, stop_signal)
    try:
        canary = next(
            (
                entry
                for entry in pending
                if entry["execution_index"] == settings.canary_index
            ),
            pending[0] if pending else None,
        )
        if settings.canary_index in completed:
            canary = None
        if canary is not None:
            await execute(canary)
        count = max(service_count(arm.plan.model) for arm in arms)
        queue_count = count if settings.concurrency >= count else 1
        queues: list[asyncio.Queue[dict[str, Any]]] = [
            asyncio.Queue() for _ in range(queue_count)
        ]
        for entry in pending:
            if canary is None or entry["execution_index"] != canary["execution_index"]:
                run = entry["run"]
                slot = service_slot(by_name[run["arm"]].plan.model, run["schedule_index"]) if queue_count > 1 else 0
                queues[slot].put_nowait(entry)
        workers = [
            asyncio.create_task(consume(queues[index % len(queues)]))
            for index in range(settings.concurrency)
        ]
        await asyncio.gather(*workers)
    except BaseException as error:
        service_stop.stop(error)
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
    finally:
        for stop_signal, handler in previous_handlers.items():
            loop.remove_signal_handler(stop_signal)
            signal.signal(stop_signal, handler)
        save_progress()
    report = summarize_records(
        arms, records, root=root, evidence=evidence, execution_results=outcomes
    )
    report["execution_stop"] = definition["execution_stop"]
    report["attempted_executions"] = len(outcomes)
    report["blocked_executions"] = blocked
    report["deferred_executions"] = deferred
    report["failed_request_execution_indexes"] = failed_requests
    report.update(
        {
            key: definition[key]
            for key in ("execution_results", "execution_stop", "run_purpose")
            if key in definition
        }
    )
    write_json_atomic(directory / "results.json", report)
    return {
        "planned_executions": len(schedule),
        "attempted_executions": len(outcomes),
        "completed_valid_executions": len(records),
        "remaining_executions": len(schedule) - len(records),
        "reused_valid_executions": len(completed),
        "blocked_executions": len(blocked),
        "deferred_executions": len(deferred),
        "failed_request_executions": len(failed_requests),
        "execution_stop": definition["execution_stop"],
    }
