"""Registered frozen materials shared by human review and independent answering."""

from __future__ import annotations

import argparse
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .environment import entity_visual_path, task_public_site_data
from .execution import prepare_visual_task
from .frozen_run import FrozenRun
from .protocol import RunSession
from .timeline import Condition, complete_materials_condition
from .workspace import read_json_strict

SAMPLE = "data/experiments/human-reference.json"


@dataclass
class AnswerCase:
    task: dict[str, Any]
    session: RunSession
    public_site: dict[str, Any]
    asset_root: Path
    source: dict[str, Any]

    @classmethod
    def from_frozen(cls, run: FrozenRun, condition: Condition) -> AnswerCase:
        task = prepare_visual_task(run.task, condition, run.root)
        parameters = run.plan.protocol_parameters(task)
        snapshot = parameters["site_snapshot"]
        if snapshot is None:
            raise ValueError("人工答案核验需要冻结的网站快照")
        site = read_json_strict(snapshot)
        if site["site"] != task["site"]:
            raise ValueError("网站快照与任务不一致")
        session = RunSession(
            task,
            complete_materials_condition(condition),
            event_step=parameters["default_release_step"],
            reference_steps=parameters["reference_steps"],
        )
        return cls(
            task,
            session,
            task_public_site_data(task, site),
            Path(parameters.get("frozen_asset_root") or run.root),
            {
                "study": str(run.directory),
                "condition": condition.as_dict(),
                "site_snapshot": str(snapshot),
            },
        )

    def assets(self) -> list[str]:
        paths = []
        for entity in self.public_site["entities"]:
            image = entity_visual_path(self.task["site"], entity)
            if image:
                paths.append(image)
            paths.extend(
                preview["image_path"] for preview in entity.get("public_previews", [])
            )
        return list(dict.fromkeys(paths))


class HumanReferenceMaterials:
    """Resolve only the registered task, website and visual snapshots."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.sample = read_json_strict(self.root / SAMPLE)
        self.rows = self.sample["tasks"]
        ids = [row["task_id"] for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("人工核查清单存在重复任务")
        self._cases: dict[int, AnswerCase] = {}
        self._lock = threading.RLock()

    def case(self, item: int) -> AnswerCase:
        if type(item) is not int or not 1 <= item <= len(self.rows):
            raise ValueError("任务序号不在登记清单中")
        with self._lock:
            if item not in self._cases:
                row = self.rows[item - 1]
                run = FrozenRun.resolve(
                    argparse.Namespace(
                        plan=str(self.root / row["source_study"] / "plan.json"),
                        arm=row["arm"],
                        task_id=row["task_id"],
                        condition=row["condition"],
                        replicate_id=0,
                        root=str(self.root),
                    )
                )
                case = AnswerCase.from_frozen(run, run.plan.condition(row["condition"]))
                case.source["task_snapshot"] = str(
                    run.directory
                    / "tasks"
                    / row["arm"]
                    / case.task["track"]
                    / f"{case.task['id']}.json"
                )
                case.source["sample"] = row
                case.source["candidate_order_seed"] = (
                    self.sample["selection_seed"] + item
                )
                random.Random(case.source["candidate_order_seed"]).shuffle(
                    case.public_site["entities"]
                )
                self._cases[item] = case
            return self._cases[item]
