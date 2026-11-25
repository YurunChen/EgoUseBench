"""Resolve registered frozen inputs for command and human-validation adapters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .experiment import (
    ExperimentPlan,
    ExperimentPlanError,
    load_experiment_plan,
    scheduled_indexes,
)
from .schema import load_tasks
from .timeline import COMPOSITE_TRACK


def _task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    for task in tasks:
        if task["id"] == task_id:
            return task
    raise ExperimentPlanError(f"unknown frozen task {task_id!r}")


@dataclass(frozen=True)
class FrozenRun:
    """One registered run of an exported experiment, read from its frozen inputs."""

    directory: Path
    plan: ExperimentPlan
    task: dict[str, Any]
    schedule_index: int
    replicate_id: int
    root: str

    @classmethod
    def resolve(cls, args: argparse.Namespace) -> "FrozenRun":
        directory = Path(args.plan).resolve().parent
        plan = load_experiment_plan(args.plan, args.arm)
        task = _task(
            load_tasks(
                directory / "tasks" / args.arm,
                experimental=COMPOSITE_TRACK in plan.track_conditions,
            ),
            args.task_id,
        )
        run_key = (task["id"], args.condition, args.replicate_id)
        schedule_index = scheduled_indexes([task], plan).get(run_key)
        if schedule_index is None:
            raise ExperimentPlanError(
                f"run is not registered in arm {args.arm!r}: {run_key}"
            )
        return cls(directory, plan, task, schedule_index, args.replicate_id, args.root)

    def adapter_settings(self) -> dict[str, Any]:
        """Archive inside the experiment, under the arm's registered protocol."""
        return {
            "replicate_id": self.replicate_id,
            "asset_root": self.root,
            "output_root": Path(self.root) / "experiments",
            "experiment_directory": self.directory,
            "experiment_id": self.plan.experiment_id,
            **self.plan.protocol_parameters(self.task),
        }

    def solvability_settings(self) -> dict[str, Any]:
        settings = self.adapter_settings()
        settings["source_max_steps"] = settings.pop("max_steps")
        settings.update(
            experiment_directory=None,
            experiment_id="human-solvability",
            source_study=str(self.directory),
        )
        return settings
