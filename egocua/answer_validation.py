"""Independent human answer choices over the registered frozen task materials."""

from __future__ import annotations

import fcntl
import re
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .final_state import evaluate_final_state
from .human_reference import AnswerCase, HumanReferenceMaterials
from .workspace import read_json_strict, resolve_within, write_json_atomic

PROTOCOL = "human-answer-validation-v1"
JUDGMENTS = {"selected", "no_match", "insufficient"}


class AnswerValidationWorkspace:
    """Own frozen inputs, first submitted answers, and offline participant records."""

    def __init__(self, root: Path, participant: str, output: Path) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", participant):
            raise ValueError("参与者编号只能使用字母、数字、下划线和短横线")
        self.root = root.resolve()
        self.participant = participant
        self.output = output.resolve()
        self.materials = HumanReferenceMaterials(self.root)
        self.rows = self.materials.rows
        self._lock = threading.RLock()

    def case(self, item: int) -> AnswerCase:
        return self.materials.case(item)

    def record(self, item: int) -> dict[str, Any]:
        case = self.case(item)
        path = self.output / f"{item:02d}.json"
        if path.exists():
            record = read_json_strict(path)
            if record["source"] != case.source or record["protocol"] != PROTOCOL:
                raise ValueError("已保存记录的材料与当前清单不同，请由组织者核对")
            return record
        return {
            "protocol": PROTOCOL,
            "participant": self.participant,
            "item": item,
            "source": case.source,
            "started_at": time.time(),
            "submitted_at": None,
            "revision": 0,
            "questions": [],
            "judgment": "selected",
            "choices": {},
            "fields": {},
            "note": "",
            "matches_registered_answer": None,
        }

    def open(self, item: int) -> dict[str, Any]:
        with self._edit():
            record = self.record(item)
            if not (self.output / f"{item:02d}.json").exists():
                self._write(record)
            return record

    def public_record(self, item: int) -> dict[str, Any]:
        record = self.record(item)
        return {
            key: record[key]
            for key in (
                "participant",
                "item",
                "revision",
                "started_at",
                "submitted_at",
                "questions",
                "judgment",
                "choices",
                "fields",
                "note",
            )
        }

    def update(self, item: int, data: dict[str, Any]) -> dict[str, Any]:
        with self._edit():
            record = self.record(item)
            if record["submitted_at"] is not None:
                raise ValueError("首次提交已保存，不能覆盖")
            if (
                type(data.get("revision")) is not int
                or data["revision"] != record["revision"]
            ):
                raise ValueError("页面已变化，请刷新后继续；已有记录仍保留")
            action = data.get("action")
            case = self.case(item)
            if action == "ask":
                question = data.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise ValueError("请先填写问题")
                if record["questions"]:
                    raise ValueError("本题已经提问，回答已保留")
                case.session.questions_used = len(record["questions"])
                answer = case.session.ask_user(question.strip(), 1)
                record["questions"].append(
                    {"question": question.strip(), "answer_event": answer}
                )
            elif action in {"save", "submit"}:
                self._set_choices(record, data, case)
                if action == "submit":
                    if record["judgment"] == "selected" and not record["choices"]:
                        raise ValueError(
                            "请选择至少一个候选，或选择没有符合项／信息不足"
                        )
                    if record["judgment"] != "selected" and not record["note"].strip():
                        raise ValueError("请简短说明没有符合项或信息不足的原因")
                    record["submitted_at"] = time.time()
                    record["elapsed_seconds"] = round(
                        record["submitted_at"] - record["started_at"], 1
                    )
                    entities = case.public_site["entities"]
                    state = {
                        "selected_entity_ids": [
                            entities[int(option)]["entity_id"]
                            for option, quantity in record["choices"].items()
                            for _ in range(quantity)
                        ],
                        "fields": record["fields"],
                    }
                    record["selected_state"] = state
                    record["matches_registered_answer"] = (
                        evaluate_final_state(case.task["goal"], state)
                        if record["judgment"] == "selected"
                        else None
                    )
            else:
                raise ValueError("未知操作")
            record["revision"] += 1
            self._write(record)
            return self.public_record(item)

    @staticmethod
    def _set_choices(record: dict, data: dict, case: AnswerCase) -> None:
        judgment, choices = data.get("judgment"), data.get("choices")
        fields, note = data.get("fields"), data.get("note")
        if (
            not isinstance(judgment, str)
            or judgment not in JUDGMENTS
            or not isinstance(choices, dict)
        ):
            raise ValueError("请选择有效的判断和候选")
        allowed = {str(index) for index in range(len(case.public_site["entities"]))}
        if any(
            key not in allowed or type(value) is not int or not 1 <= value <= 10000
            for key, value in choices.items()
        ):
            raise ValueError("候选或数量不正确")
        if not isinstance(fields, dict) or any(
            key not in case.public_site["supported_fields"]
            or not isinstance(value, str)
            for key, value in fields.items()
        ):
            raise ValueError("任务字段不正确")
        if not isinstance(note, str):
            raise ValueError("说明应为文字")
        if judgment != "selected" and (choices or fields):
            raise ValueError("选择没有符合项或信息不足时，请清空已选结果")
        record.update(judgment=judgment, choices=choices, fields=fields, note=note)

    @contextmanager
    def _edit(self):
        with self._lock:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / ".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)

    def _write(self, record: dict[str, Any]) -> None:
        write_json_atomic(self.output / f"{record['item']:02d}.json", record)

    def progress(self) -> list[dict[str, Any]]:
        rows = []
        for index in range(1, len(self.rows) + 1):
            path = self.output / f"{index:02d}.json"
            record = read_json_strict(path) if path.exists() else None
            rows.append(
                {
                    "item": index,
                    "submitted": bool(record and record["submitted_at"]),
                    "started": record is not None,
                }
            )
        return rows

    def summary(self) -> dict[str, Any]:
        groups: dict[str, Counter] = {}
        for item, row in enumerate(self.rows, 1):
            path = self.output / f"{item:02d}.json"
            if not path.exists():
                continue
            record = read_json_strict(path)
            if record["submitted_at"] is None:
                continue
            group = "composite" if row["stratum"].startswith("composite:") else "single"
            counts = groups.setdefault(group, Counter())
            counts["submitted"] += 1
            counts[record["judgment"]] += 1
            if record["matches_registered_answer"] is not None:
                counts[
                    "matched" if record["matches_registered_answer"] else "different"
                ] += 1
        return {"participant": self.participant, "groups": groups}

    def candidate_asset(self, item: int, index: int) -> Path:
        paths = self.case(item).assets()
        if not 0 <= index < len(paths):
            raise ValueError("未找到候选图片")
        return resolve_within(self.root, paths[index], context="candidate image")
