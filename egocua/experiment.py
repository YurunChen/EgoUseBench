from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_ARMS = {
    "graft": ("a_only", "bundled", "injected"),
    "refresh": ("stale", "stream", "oracle_latest"),
    "clarify": ("oracle", "interactive", "forced_execute"),
}

PRIMARY_CONDITION = {
    "graft": "injected",
    "refresh": "stream",
    "clarify": "interactive",
}


class ExperimentPlanError(ValueError):
    """Raised when an experiment plan is not internally consistent."""


@dataclass(frozen=True)
class ExperimentPlan:
    experiment_id: str
    arms: dict[str, tuple[str, ...]]
    replicate_ids: tuple[int, ...]
    adapter: str
    model: str
    model_version: str
    max_steps: int
    scheduler_seed: int

    def arms_for(self, track: str) -> tuple[str, ...]:
        try:
            return self.arms[track]
        except KeyError as exc:
            raise ExperimentPlanError(f"experiment plan does not include track {track!r}") from exc

    def validate_arm(self, track: str, arm: str) -> None:
        if arm not in self.arms_for(track):
            raise ExperimentPlanError(f"arm {arm!r} is not enabled for track {track!r}")


def load_experiment_plan(path: str | Path) -> ExperimentPlan:
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict):
        raise ExperimentPlanError(f"{source}: experiment plan must be a JSON object")
    if payload.get("schema_version") != 2:
        raise ExperimentPlanError(f"{source}: experiment plan schema_version must be 2")

    experiment_id = str(payload.get("experiment_id", "")).strip()
    if not experiment_id:
        raise ExperimentPlanError(f"{source}: experiment_id must be non-empty")

    raw_arms = payload.get("arms")
    if not isinstance(raw_arms, dict) or not raw_arms:
        raise ExperimentPlanError(f"{source}: arms must map tracks to run arms")
    arms: dict[str, tuple[str, ...]] = {}
    for track, selected in raw_arms.items():
        if track not in SUPPORTED_ARMS:
            raise ExperimentPlanError(f"{source}: unsupported track {track!r}")
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)):
            raise ExperimentPlanError(f"{source}: {track} arms must be a non-empty unique list")
        unsupported = set(selected) - set(SUPPORTED_ARMS[track])
        if unsupported:
            raise ExperimentPlanError(f"{source}: unsupported {track} arms {sorted(unsupported)}")
        arms[track] = tuple(selected)

    raw_replicates = payload.get("replicate_ids")
    if (
        not isinstance(raw_replicates, list)
        or not raw_replicates
        or len(raw_replicates) != len(set(raw_replicates))
        or any(not isinstance(value, int) or value < 0 for value in raw_replicates)
    ):
        raise ExperimentPlanError(f"{source}: replicate_ids must be a non-empty unique list of non-negative integers")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ExperimentPlanError(f"{source}: runtime must be an object")
    adapter = str(runtime.get("adapter", "")).strip()
    model = str(runtime.get("model", "")).strip()
    model_version = str(runtime.get("model_version", "")).strip()
    max_steps = runtime.get("max_steps")
    if not adapter or not model or not model_version:
        raise ExperimentPlanError(f"{source}: runtime adapter, model, and model_version are required")
    if not isinstance(max_steps, int) or max_steps < 1:
        raise ExperimentPlanError(f"{source}: runtime.max_steps must be positive")
    scheduler_seed = payload.get("scheduler_seed")
    if not isinstance(scheduler_seed, int) or scheduler_seed < 0:
        raise ExperimentPlanError(f"{source}: scheduler_seed must be a non-negative integer")

    return ExperimentPlan(
        experiment_id,
        arms,
        tuple(raw_replicates),
        adapter,
        model,
        model_version,
        max_steps,
        scheduler_seed,
    )


def expand_runs(tasks: Iterable[dict[str, Any]], plan: ExperimentPlan) -> list[dict[str, Any]]:
    ordered_tasks = sorted(tasks, key=lambda task: task["id"])
    runs = [
        {
            "task_id": task["id"],
            "track": task["track"],
            "arm": arm,
            "replicate_id": replicate_id,
        }
        for task in ordered_tasks
        for arm in plan.arms_for(task["track"])
        for replicate_id in plan.replicate_ids
    ]
    random.Random(plan.scheduler_seed).shuffle(runs)
    return [{**run, "schedule_index": index} for index, run in enumerate(runs)]


def validate_run_records(
    tasks: Iterable[dict[str, Any]], records: Iterable[dict[str, Any]], plan: ExperimentPlan
) -> None:
    from collections import Counter

    schedule = expand_runs(tasks, plan)
    expected = {
        (run["task_id"], run["arm"], run["replicate_id"]): run["schedule_index"]
        for run in schedule
    }
    materialized_records = list(records)
    actual = [
        (record.get("task_id"), record.get("arm"), record.get("meta", {}).get("replicate_id"))
        for record in materialized_records
    ]
    duplicates = sorted(key for key, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if duplicates or missing or unexpected:
        raise ExperimentPlanError(
            "run matrix must contain each task/arm/replicate exactly once; "
            f"missing={missing[:5]}, duplicates={duplicates[:5]}, unexpected={unexpected[:5]}"
        )
    invalid_identity = []
    for record, key in zip(materialized_records, actual):
        meta = record.get("meta", {})
        observed = {
            "schedule_index": meta.get("schedule_index"),
            "agent": meta.get("agent"),
            "model": meta.get("model"),
            "model_version": meta.get("model_version"),
            "max_steps": meta.get("max_steps"),
        }
        expected_identity = {
            "schedule_index": expected[key],
            "agent": plan.adapter,
            "model": plan.model,
            "model_version": plan.model_version,
            "max_steps": plan.max_steps,
        }
        if observed != expected_identity:
            invalid_identity.append((key, observed, expected_identity))
    if invalid_identity:
        raise ExperimentPlanError(
            "run records disagree with the registered schedule/runtime; "
            f"mismatches={invalid_identity[:3]}"
        )
