"""Human quality judgments over the single registered forty-task sample."""

from __future__ import annotations

import fcntl
import re
import threading
from datetime import UTC, datetime
from functools import cached_property
from pathlib import Path
from typing import Any

from .human_reference import HumanReferenceMaterials, SAMPLE
from .workspace import read_json_strict, write_json_atomic

DECISIONS = {"pass": "通过", "problem": "有问题", "uncertain": "无法判断"}
PROTOCOL = "human-task-quality-v1"


class SampleReviewWorkspace:
    def __init__(self, root: Path, reviewer: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", reviewer):
            raise ValueError("审核者编号只能使用字母、数字、下划线和短横线")
        self.root = root.resolve()
        self.reviewer = reviewer
        self.output = self.root / "data/reviews/human-reference" / reviewer
        self._lock = threading.RLock()

    @cached_property
    def materials(self) -> HumanReferenceMaterials:
        return HumanReferenceMaterials(self.root)

    def record(self, item: int) -> dict[str, Any] | None:
        case = self.materials.case(item)
        path = self.output / f"{item:02d}.json"
        if not path.exists():
            return None
        record = read_json_strict(path)
        if record["source"] != case.source or record["protocol"] != PROTOCOL:
            raise ValueError("已保存意见与本题材料不同，请先核对记录")
        return record

    def progress(self) -> list[dict[str, Any]]:
        return [
            {
                "item": item,
                "task_id": row["task_id"],
                "stratum": row["stratum"],
                "record": self.record(item),
            }
            for item, row in enumerate(self.materials.rows, 1)
        ]

    def submit(self, item: int, decision: str, note: str) -> None:
        if decision not in DECISIONS:
            raise ValueError("请选择通过、有问题或无法判断")
        note = note.strip()
        if decision != "pass" and not note:
            raise ValueError("请简短说明问题或无法判断的原因")
        if len(note) > 10000:
            raise ValueError("说明请控制在10000字以内")
        case = self.materials.case(item)
        with self._lock:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if self.record(item) is not None:
                    raise ValueError("本题意见已保存，不能覆盖；请继续下一题")
                write_json_atomic(
                    self.output / f"{item:02d}.json",
                    {
                        "protocol": PROTOCOL,
                        "reviewer": self.reviewer,
                        "sample": SAMPLE,
                        "item": item,
                        "task_id": case.task["id"],
                        "source": case.source,
                        "decision": decision,
                        "note": note,
                        "submitted_at": datetime.now(UTC).isoformat(),
                    },
                )
