from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .experiment import SUPPORTED_ARMS


class ProtocolError(RuntimeError):
    """Raised for an invalid interaction with a benchmark session."""


DYNAMIC_EVENT_STEP = 2

_QUESTION_MARKERS = (
    "?", "？", "which", "what", "do you", "did you", "is it", "was it",
    "can you", "could you", "would you", "please clarify", "哪", "什么", "是否",
    "吗", "请问", "你指", "你要", "你想",
)
_DISTINCTION_MARKERS = (
    "which", "mean", "refer", "intend", "want", "one", "item", "object",
    "product", "candidate", "option", "clarify", " or ", "哪", "指", "要",
    "想", "一个", "物品", "对象", "商品", "候选", "还是", "或者",
)
_SLOT_CUES = {
    "position": (
        "position", "side", "left", "right", "where", "location", "next to",
        "beside", "above", "below", "middle", "位置", "左", "右", "哪边", "哪侧",
        "旁边", "上面", "下面", "中间",
    ),
    "appearance": (
        "appearance", "look like", "looks like", "colour", "color", "shape", "size", "tall",
        "short", "large", "larger", "small", "smaller", "pattern", "material",
        "外观", "颜色", "形状", "大小", "大的", "小的", "高的", "矮的",
        "图案", "材质", "看起来",
    ),
    "brand": (
        "brand", "make", "manufacturer", "label", "logo", "品牌", "牌子",
        "厂商", "标志", "商标",
    ),
    "variant": (
        "variant", "version", "model", "type", "style", "design", "flavor",
        "flavour", "kind", "edition", "款", "型号", "版本", "口味", "种类",
        "类型", "风格", "造型", "版",
    ),
    "sequence": (
        "sequence", "order", "first", "second", "last", "earlier", "later",
        "before", "after", "initial", "final", "顺序", "第一", "第二", "最后",
        "先", "后", "之前", "之后", "最初", "最终", "先拿", "后拿",
    ),
}


def _contains(text: str, cue: str) -> bool:
    if cue.isascii() and cue.isalnum():
        return re.search(rf"\b{re.escape(cue)}\b", text) is not None
    return cue in text


def _contains_any(text: str, cues: tuple[str, ...]) -> bool:
    return any(_contains(text, cue) for cue in cues)


def question_matches_slot(slot: str, question: str) -> bool:
    """Accept a natural disambiguating question for the structured slot."""

    normalized = " ".join(str(question).casefold().split())
    cues = _SLOT_CUES.get(slot)
    if not normalized or cues is None:
        return False
    if not (
        _contains_any(normalized, _QUESTION_MARKERS)
        and _contains_any(normalized, _DISTINCTION_MARKERS)
    ):
        return False

    matching_slots = {
        candidate_slot
        for candidate_slot, candidate_cues in _SLOT_CUES.items()
        if _contains_any(normalized, candidate_cues)
    }
    return not matching_slots or slot in matching_slots


def clarification_question_is_valid(task: dict[str, Any], slot: str, question: str) -> bool:
    clarification = task["clarification"]
    return slot == clarification["valid_slot"] and question_matches_slot(slot, question)


def initial_visible_packets(task: dict[str, Any], arm: str) -> set[str]:
    if task["track"] == "refresh" and arm == "oracle_latest":
        return {"V0", "V1"}
    return {"V0"}


def _released_observation(packet: dict[str, Any]) -> dict[str, Any]:
    """Return only the fields an Agent Adapter needs to deliver an observation."""

    return {"frame_paths": list(packet["frame_paths"])}


@dataclass
class RunSession:
    task: dict[str, Any]
    arm: str
    event_step: int = DYNAMIC_EVENT_STEP
    events: list[dict[str, Any]] = field(default_factory=list)
    questions_used: int = 0
    _scheduled_event_sent: bool = False

    def __post_init__(self) -> None:
        if self.arm not in SUPPORTED_ARMS[self.task["track"]]:
            raise ProtocolError(f"arm {self.arm!r} is invalid for {self.task['id']}")

    def initial_payload(self) -> dict[str, Any]:
        track = self.task["track"]
        instructions = self.task["instructions"]
        visible = initial_visible_packets(self.task, self.arm)
        observations = [
            _released_observation(packet)
            for packet in self.task["visual_packets"]
            if packet["id"] in visible
        ]
        if track == "graft":
            instruction = instructions["bundled"] if self.arm == "bundled" else instructions["a"]
        elif track == "refresh":
            instruction = instructions["base"]
        else:
            instruction = instructions["oracle"] if self.arm == "oracle" else instructions["ambiguous"]
        payload = {
            "instruction": instruction,
            "observations": observations,
            "clarification_available": track == "clarify" and self.arm != "forced_execute",
        }
        return payload

    def has_pending_scheduled_event(self) -> bool:
        if self._scheduled_event_sent:
            return False
        return (self.task["track"], self.arm) in {("graft", "injected"), ("refresh", "stream")}

    def advance(self, step: int) -> list[dict[str, Any]]:
        if step < 1:
            raise ProtocolError("step must be positive")
        if self._scheduled_event_sent:
            return []
        if step < self.event_step:
            return []

        event: dict[str, Any] | None = None
        if self.task["track"] == "graft" and self.arm == "injected":
            event = {
                "type": "instruction_injection",
                "step": step,
                "payload": self.task["instructions"]["b"],
                "trigger_reason": "task_timeline",
            }
        elif self.task["track"] == "refresh" and self.arm == "stream":
            event = {
                "type": "visual_context_update",
                "step": step,
                "payload": _released_observation(self.task["visual_packets"][1]),
                "trigger_reason": "task_timeline",
            }
        if event is None:
            return []
        self._scheduled_event_sent = True
        self.events.append(event)
        return [event]

    def ask_user(self, slot: str, question: str, step: int) -> dict[str, Any]:
        if self.task["track"] != "clarify" or self.arm == "forced_execute":
            raise ProtocolError("clarification is not available in this run")
        if self.questions_used >= self.task["clarification"]["max_questions"]:
            return {
                "type": "clarification_answer",
                "step": step,
                "slot": slot,
                "question": question,
                "valid": False,
                "limit_reached": True,
                "payload": "The wearer is no longer available for questions. Proceed with your best judgment.",
            }
        self.questions_used += 1
        clarification = self.task["clarification"]
        question_relevant = question_matches_slot(slot, question)
        valid = slot == clarification["valid_slot"] and question_relevant
        answer = (
            clarification["canonical_answer"]
            if valid
            else "That question did not distinguish the candidates. Proceed with your best judgment."
        )
        event = {
            "type": "clarification_answer",
            "step": step,
            "slot": slot,
            "question": question,
            "valid": valid,
            "payload": answer,
        }
        self.events.append(event)
        return event
