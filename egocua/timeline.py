"""Evidence timeline: what the Agent receives, when, and when the referent is knowable.

This is the only module that knows how a task and an experiment Condition turn
a task file into releases of evidence. Everything
downstream (execution, sites, evaluator, adapters) sees only Releases with steps,
an optional clarification offer, and the sufficiency step t*.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .visual_sampling import sample_packet, validate_interval
from .frame_inputs import JPEG_QUALITY, packed_image_size, validate_jpeg_quality
from .messages import HISTORY_RETENTIONS, INITIAL_STEP, KEY_EVIDENCE, SOURCE_PREVIEW


class TimelineError(ValueError):
    """Raised when a task or condition cannot be compiled into a timeline."""


COMPUTER_USE = "computer_use"
OFFLINE_EGO_USE = "offline_ego_use"
HUMAN_SOLVABILITY = "human_solvability"
COMPOSITE_TRACK = "composite"
SCHEDULES: tuple[str, ...] = (
    "online",
    "upfront",
    COMPUTER_USE,
    OFFLINE_EGO_USE,
    HUMAN_SOLVABILITY,
)
SCOPES: tuple[str, ...] = ("full", "initial_only", "first_frame", "none")
ASKS: tuple[str, ...] = ("open", "allowed", "denied")
"""open: every track may ask once; allowed: only Clarify offers asking; denied: none."""
UNINFORMATIVE_REPLY = "Just what I asked for."
REPEATED_QUESTION_REPLY = "I've already answered."
CHANGE_CUE_TEXT = "Something just changed here."
"""Says only that the scene changed, never what changed or whether to act."""
ACTION_ALIGNED_REPLAY = "action_aligned"
COMPOSITE_PREVIEWS = ("graft", "refresh", "both")
COMPOSITE_EVENTS = "composite_events"


@dataclass(frozen=True)
class CompositeUpdate:
    """A scene change or user revision anchored in the source recording."""

    event_id: str
    mechanism: str
    onset_seconds: float
    request: str | None = None


@dataclass(frozen=True)
class CompositeInput:
    opening: str
    updates: tuple[CompositeUpdate, ...]
    clarification: bool
    explicit_events: bool

    @property
    def clarification_id(self) -> str | None:
        if not self.clarification:
            return None
        return "C1" if self.explicit_events else "C"

    @property
    def mechanisms(self) -> frozenset[str]:
        kinds = {update.mechanism for update in self.updates}
        if self.clarification:
            kinds.add("C")
        return frozenset(kinds)

    @property
    def event_ids(self) -> frozenset[str]:
        ids = {update.event_id for update in self.updates}
        if self.clarification_id is not None:
            ids.add(self.clarification_id)
        return frozenset(ids)


def composite_input(task: Mapping[str, Any]) -> CompositeInput:
    """Read explicit composite events or the one-update fields in old snapshots."""

    task_id = task["id"]
    instructions = task["instructions"]
    session = task["visual_session"]
    explicit = COMPOSITE_EVENTS in task
    scalar_fields = {"refresh_onset_seconds", "update_onset_seconds"} & session.keys()
    if explicit and scalar_fields:
        raise TimelineError(
            f"{task_id}: composite events cannot mix with scalar onsets"
        )
    has_legacy_graft = "update_onset_seconds" in scalar_fields
    required = {"base", "b"} if has_legacy_graft else {"base"}
    if instructions.keys() != required or any(
        not isinstance(value, str) or not value.strip()
        for value in instructions.values()
    ):
        raise TimelineError(
            f"{task_id}: composite requires base and only a scheduled follow-up "
            "in legacy instructions"
        )
    if explicit:
        updates = _explicit_composite_updates(task[COMPOSITE_EVENTS], str(task_id))
    else:
        rows = (
            ("R", "refresh_onset_seconds", None),
            ("G", "update_onset_seconds", instructions.get("b")),
        )
        updates = tuple(
            sorted(
                (
                    CompositeUpdate(
                        kind, kind, _composite_onset(task_id, session[field]), request
                    )
                    for kind, field, request in rows
                    if field in scalar_fields
                ),
                key=lambda update: update.onset_seconds,
            )
        )
    result = CompositeInput(
        instructions["base"], updates, "clarification" in task, explicit
    )
    if len(result.mechanisms) < 2:
        raise TimelineError(f"{task_id}: composite requires at least two mechanisms")
    return result


def _composite_onset(task_id: str, value: Any) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise TimelineError(
            f"{task_id}: composite onset_seconds must be finite and nonnegative"
        )
    return float(value)


def _explicit_composite_updates(raw: Any, task_id: str) -> tuple[CompositeUpdate, ...]:
    if not isinstance(raw, list) or not raw:
        raise TimelineError(f"{task_id}: composite_events must be a nonempty list")
    mechanisms = {"refresh": "R", "graft": "G"}
    counts = {"R": 0, "G": 0}
    updates: list[CompositeUpdate] = []
    for event in raw:
        if not isinstance(event, Mapping):
            raise TimelineError(f"{task_id}: each composite event must be an object")
        kind = event.get("kind")
        if not isinstance(kind, str) or kind not in mechanisms:
            raise TimelineError(
                f"{task_id}: composite event kind must be refresh or graft"
            )
        mechanism = mechanisms[kind]
        request = event.get("request")
        fields = {"kind", "onset_seconds"}
        if mechanism == "G":
            fields.add("request")
            if not isinstance(request, str) or not request.strip():
                raise TimelineError(
                    f"{task_id}: a Graft event requires a wearer request"
                )
        if event.keys() != fields:
            raise TimelineError(f"{task_id}: unexpected {kind} event fields")
        onset = _composite_onset(task_id, event["onset_seconds"])
        if updates and onset <= updates[-1].onset_seconds:
            raise TimelineError(
                f"{task_id}: composite onsets must be strictly increasing"
            )
        counts[mechanism] += 1
        updates.append(
            CompositeUpdate(
                f"{mechanism}{counts[mechanism]}", mechanism, onset, request
            )
        )
    return tuple(updates)


@dataclass(frozen=True)
class Condition:
    """One point in the experiment factor space (see timeline-refactor spec §9)."""

    name: str
    schedule: str
    scope: str
    ask: str
    release_at: float | None
    frame_interval_seconds: float | None = None
    frame_repeat_to_interval_seconds: float | None = None
    wearer_frames_per_image: int = 1
    wearer_jpeg_quality: int = JPEG_QUALITY
    strategy_hint: bool = False
    frame_delivery_steps: int | None = None
    batch_seconds: float | None = None
    visual_history: str | None = None
    change_cue: bool = False
    change_facts: bool = False
    follow_up_after_finish: bool = False
    replay_timing: str | None = None
    composite_upfront: str | None = None

    def __post_init__(self) -> None:
        try:
            validate_jpeg_quality(self.wearer_jpeg_quality)
        except ValueError as exc:
            raise TimelineError(str(exc)) from exc
        if self.wearer_frames_per_image not in (1, 4):
            raise TimelineError("wearer_frames_per_image must be 1 or 4")
        if not isinstance(self.name, str) or not self.name.strip():
            raise TimelineError("condition name must be non-empty")
        if self.schedule not in SCHEDULES:
            raise TimelineError(f"unknown schedule {self.schedule!r}")
        if self.scope not in SCOPES:
            raise TimelineError(f"unknown scope {self.scope!r}")
        if self.ask not in ASKS:
            raise TimelineError(f"unknown ask {self.ask!r}")
        if self.release_at is not None and not 0 < self.release_at < 1:
            raise TimelineError(
                "release_at must be a fraction strictly between 0 and 1"
            )
        if self.frame_interval_seconds is not None:
            try:
                validate_interval(self.frame_interval_seconds)
            except ValueError as exc:
                raise TimelineError(str(exc)) from exc
            if self.scope != "full" and self.frame_delivery_steps is None:
                raise TimelineError(
                    "frame_interval_seconds requires scope=full without replay"
                )
        if self.frame_repeat_to_interval_seconds is not None:
            validate_interval(self.frame_repeat_to_interval_seconds)
            if self.frame_interval_seconds is None or (
                self.frame_repeat_to_interval_seconds >= self.frame_interval_seconds
            ):
                raise TimelineError(
                    "frame repetition requires a shorter target interval"
                )
        if type(self.strategy_hint) is not bool:
            raise TimelineError("strategy_hint must be true or false")
        if self.frame_delivery_steps is not None:
            if (
                type(self.frame_delivery_steps) is not int
                or self.frame_delivery_steps < 1
            ):
                raise TimelineError("frame_delivery_steps must be a positive integer")
            if self.frame_interval_seconds is None:
                raise TimelineError(
                    "frame_delivery_steps requires frame_interval_seconds"
                )
            if self.schedule == "upfront" and (
                self.scope != "full" or self.batch_seconds is None
            ):
                raise TimelineError(
                    "upfront replay requires fixed batches and scope=full"
                )
            if self.scope == "first_frame" or (
                self.scope == "initial_only" and self.batch_seconds is None
            ):
                raise TimelineError("unsupported scope for this replay")
        if self.batch_seconds is not None:
            self._validate_batch_seconds()
        if self.replay_timing is not None and (
            self.replay_timing != ACTION_ALIGNED_REPLAY or self.batch_seconds is None
        ):
            raise TimelineError(
                "action_aligned is the timing mode for fixed-step replay"
            )
        if self.visual_history is not None:
            if self.visual_history not in HISTORY_RETENTIONS:
                raise TimelineError(f"unknown visual_history {self.visual_history!r}")
            if self.batch_seconds is None or self.scope not in {"full", "initial_only"}:
                raise TimelineError("visual_history requires fixed-step visual replay")
            if self.schedule == "upfront" and self.visual_history == "summary":
                raise TimelineError("upfront replay cannot summarize unseen batches")
        for flag in ("change_cue", "change_facts", "follow_up_after_finish"):
            value = getattr(self, flag)
            if type(value) is not bool:
                raise TimelineError(f"{flag} must be true or false")
            if value and (
                self.batch_seconds is None
                or self.schedule != "online"
                or self.scope != "full"
            ):
                raise TimelineError(
                    f"{flag} requires full online fixed-step visual replay"
                )
        if self.change_facts and not self.change_cue:
            raise TimelineError("change_facts requires change_cue")
        if self.composite_upfront is not None:
            if self.composite_upfront not in COMPOSITE_PREVIEWS:
                raise TimelineError(
                    f"unknown composite_upfront {self.composite_upfront!r}"
                )
            self.require_composite_replay()
        self._validate_interaction_setting()

    def require_composite_replay(self) -> None:
        """Composite evidence timing is independent of website actions."""
        if (
            self.schedule != "online"
            or self.scope != "full"
            or self.replay_timing != ACTION_ALIGNED_REPLAY
            or self.frame_delivery_steps != 1
            or self.release_at is not None
            or self.change_cue
            or self.follow_up_after_finish
        ):
            raise TimelineError(
                "composites require full online fixed-step visual replay"
            )

    def _validate_interaction_setting(self) -> None:
        if self.schedule == HUMAN_SOLVABILITY and (
            self.scope != "full"
            or self.ask != "open"
            or self.visual_history != "full"
            or self.release_at is not None
            or self.composite_upfront is not None
            or self.change_cue
            or self.change_facts
            or self.follow_up_after_finish
        ):
            raise TimelineError(
                "human solvability requires complete materials and open asking"
            )
        if self.schedule == COMPUTER_USE and (
            self.scope != "none"
            or self.ask != "denied"
            or self.frame_delivery_steps is not None
            or self.release_at is not None
            or self.strategy_hint
        ):
            raise TimelineError(
                "Computer Use requires text-only input and no wearer interaction"
            )
        if self.schedule == OFFLINE_EGO_USE and (
            self.scope != "full"
            or self.ask != "open"
            or self.replay_timing != ACTION_ALIGNED_REPLAY
            or self.visual_history != "full"
            or self.release_at is not None
            or self.strategy_hint
        ):
            raise TimelineError(
                "Offline Ego Use requires full reviewed replay and open asking"
            )

    def _validate_batch_seconds(self) -> None:
        if self.frame_delivery_steps is None:
            raise TimelineError("batch_seconds requires frame_delivery_steps")
        frames = self.batch_seconds / self.frame_spacing_seconds
        if self.batch_seconds <= 0 or abs(frames - round(frames)) > 1e-9:
            raise TimelineError(
                "batch_seconds must be a positive multiple of the frame spacing"
            )

    @property
    def replay_start_step(self) -> int:
        """Archived replay indexed its opening batch at zero; current replay starts at action one."""

        return INITIAL_STEP if self.replay_timing == ACTION_ALIGNED_REPLAY else 0

    @property
    def frame_spacing_seconds(self) -> float:
        """Session seconds between delivered frames, after optional repetition."""

        spacing = self.frame_repeat_to_interval_seconds or self.frame_interval_seconds
        if spacing is None:
            raise TimelineError("frame spacing requires frame_interval_seconds")
        return spacing

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "schedule": self.schedule,
            "scope": self.scope,
            "ask": self.ask,
            "release_at": self.release_at,
        }
        if self.frame_interval_seconds is not None:
            payload["frame_interval_seconds"] = self.frame_interval_seconds
        if self.frame_repeat_to_interval_seconds is not None:
            payload["frame_repeat_to_interval_seconds"] = (
                self.frame_repeat_to_interval_seconds
            )
        if self.wearer_frames_per_image != 1:
            payload["wearer_frames_per_image"] = self.wearer_frames_per_image
        if self.wearer_jpeg_quality != JPEG_QUALITY:
            payload["wearer_jpeg_quality"] = self.wearer_jpeg_quality
        if self.strategy_hint:
            payload["strategy_hint"] = True
        if self.frame_delivery_steps is not None:
            payload["frame_delivery_steps"] = self.frame_delivery_steps
        if self.batch_seconds is not None:
            payload["batch_seconds"] = self.batch_seconds
        if self.visual_history is not None:
            payload["visual_history"] = self.visual_history
        if self.change_cue:
            payload["change_cue"] = True
        if self.change_facts:
            payload["change_facts"] = True
        if self.follow_up_after_finish:
            payload["follow_up_after_finish"] = True
        if self.replay_timing is not None:
            payload["replay_timing"] = self.replay_timing
        if self.composite_upfront is not None:
            payload["composite_upfront"] = self.composite_upfront
        return payload

    @classmethod
    def from_dict(cls, value: Any) -> Condition:
        if not isinstance(value, Mapping):
            raise TimelineError("condition must be an object")
        payload = dict(value)
        required = {"name", "schedule", "scope", "ask", "release_at"}
        missing = required - set(payload)
        if missing:
            raise TimelineError(f"condition is missing fields {sorted(missing)}")
        unknown = (
            set(payload)
            - required
            - {
                "frame_interval_seconds",
                "frame_repeat_to_interval_seconds",
                "wearer_frames_per_image",
                "wearer_jpeg_quality",
                "strategy_hint",
                "frame_delivery_steps",
                "batch_seconds",
                "visual_history",
                "change_cue",
                "change_facts",
                "follow_up_after_finish",
                "replay_timing",
                "composite_upfront",
            }
        )
        if unknown:
            raise TimelineError(f"unknown condition fields {sorted(unknown)}")
        return cls(**payload)


STANDARD_CONDITIONS: dict[str, Condition] = {
    "online": Condition("online", "online", "full", "allowed", None),
    "upfront": Condition("upfront", "upfront", "full", "allowed", None),
    "initial_only": Condition(
        "initial_only", "online", "initial_only", "allowed", None
    ),
    "no_ask": Condition("no_ask", "online", "full", "denied", None),
    "first_frame": Condition("first_frame", "online", "first_frame", "allowed", None),
    "text_only": Condition("text_only", "online", "none", "allowed", None),
}
"""The six standard factor vectors (timeline-refactor spec §9).

Inside a plan a condition name is only a role label: controlled studies and the
formal benchmark reuse "online" for variants such as four-frame tiles. Identity is
the full factor dict, which records store and every reuse or scoring check
compares; nothing may compare conditions by name alone.
"""

# --- releases and offers -------------------------------------------------------

INSTRUCTION = "instruction"
OBSERVATION = "observation"
SCHEDULED_EVENT_TYPES = frozenset(
    {
        "instruction_injection",
        "visual_context_update",
    }
)


@dataclass(frozen=True)
class Release:
    ref: str
    step: int
    kind: str
    payload: Any
    # The glasses stream arrives on its own cadence; agent actions never hold it.
    streamed: bool = False
    # Trace-only: this streamed release carries the task's change of target.
    update: bool = False
    # Held back until the Agent finishes; only a standby delivers it.
    on_standby: bool = False
    composite_event_id: str | None = None

    def as_event(self) -> dict[str, Any]:
        """Render a scheduled release in the Run Artifact event format."""

        reason = "frame_cadence" if self.streamed else "task_timeline"
        types = {
            INSTRUCTION: "instruction_injection",
            OBSERVATION: "visual_context_update",
        }
        if self.kind not in types:
            raise TimelineError(f"release kind {self.kind!r} is not a scheduled event")
        event = {
            "type": types[self.kind],
            "step": self.step,
            "payload": self.payload,
            "trigger_reason": reason,
        }
        if self.update:
            event["update"] = True
        if self.composite_event_id is not None:
            event["composite_event_id"] = self.composite_event_id
        return event


def is_update_event(event: Mapping[str, Any]) -> bool:
    """A delivered change of target; ordinary stream batches are not updates."""

    if event.get("type") not in SCHEDULED_EVENT_TYPES:
        return False
    return event.get("trigger_reason") != "frame_cadence" or event.get("update") is True


def reveals_final_target(event: Mapping[str, Any]) -> bool:
    """A delivered update or a valid clarification answer."""

    return is_update_event(event) or (
        event.get("type") == "clarification_answer" and event.get("valid") is True
    )


@dataclass(frozen=True)
class ClarificationOffer:
    """One question the wearer answers; only Clarify holds a reviewed answer."""

    request: str
    canonical_answer: str | None = None
    candidate_ids: tuple[str, ...] = ()
    valid_slot: str | None = None
    max_questions: int = 1
    composite_event_id: str | None = None

    def answer(
        self,
        question: str,
        step: int,
        *,
        questions_used: int = 0,
    ) -> dict[str, Any]:
        """Answer one question in the Run Artifact event format."""

        event: dict[str, Any] = {
            "type": "clarification_answer",
            "step": step,
            "question": question,
        }
        if questions_used >= self.max_questions:
            return {
                **event,
                "valid": False,
                "limit_reached": True,
                "payload": REPEATED_QUESTION_REPLY,
            }
        if self.canonical_answer is None:
            return {**event, "valid": False, "payload": UNINFORMATIVE_REPLY}
        if self.composite_event_id is not None:
            event["composite_event_id"] = self.composite_event_id
        return {**event, "valid": True, "payload": self.canonical_answer}

    def recorded_answer_is_consistent(self, event: Mapping[str, Any]) -> bool:
        """The hidden answer was released exactly when the record says it was."""

        if self.canonical_answer is None:
            return event.get("valid") is not True
        if event.get("valid") is True and not event.get("limit_reached"):
            return (
                event.get("payload") == self.canonical_answer
                and event.get("composite_event_id") == self.composite_event_id
            )
        return self.canonical_answer not in str(event.get("payload", ""))


def _frames(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {"frame_paths": list(packet["frame_paths"])}


def _first_frame(packet: Mapping[str, Any]) -> dict[str, Any]:
    return {"frame_paths": list(packet["frame_paths"])[:1]}


def _packet(task: Mapping[str, Any], packet_id: str) -> Mapping[str, Any] | None:
    for packet in task.get("visual_packets", []):
        if packet.get("id") == packet_id:
            return packet
    return None


@dataclass(frozen=True)
class Timeline:
    """Compiled evidence schedule for one task under one Condition."""

    task_id: str
    track: str
    condition: Condition
    instruction: str
    initial_observations: tuple[dict[str, Any], ...]
    releases: tuple[Release, ...]
    offer: ClarificationOffer | None
    _sufficiency: int | None = field(default=None, repr=False)

    # -- compile ---------------------------------------------------------------

    @classmethod
    def compile(
        cls,
        task: Mapping[str, Any],
        condition: Condition,
        *,
        default_release_step: int,
        reference_steps: int | None,
    ) -> Timeline:
        if not isinstance(condition, Condition):
            raise TimelineError("timeline requires a Condition object")
        cond = condition
        track = str(task.get("track"))
        if track not in ("graft", "refresh", "clarify", COMPOSITE_TRACK):
            raise TimelineError(f"{task.get('id')}: unknown track {track!r}")
        if default_release_step < 1:
            raise TimelineError("default_release_step must be positive")
        if cond.schedule == HUMAN_SOLVABILITY:
            return _compile_human_solvability(task, cond)
        if track == COMPOSITE_TRACK:
            return _compile_composite(task, cond)
        if cond.composite_upfront is not None:
            raise TimelineError("composite_upfront applies only to composite tasks")
        if cond.schedule == COMPUTER_USE:
            return _compile_computer_use(task, cond)
        if cond.schedule == OFFLINE_EGO_USE:
            return _compile_offline_ego_use(task, cond)
        release_step = _release_step(cond, default_release_step, reference_steps)
        builder = {
            "graft": _compile_graft,
            "refresh": _compile_refresh,
            "clarify": _compile_clarify,
        }
        return builder[track](task, cond, release_step)

    # -- interface -------------------------------------------------------------

    def initial(self) -> dict[str, Any]:
        """What the Agent receives at step 0, in the Run Artifact payload format."""

        return {
            "instruction": self.instruction,
            "observations": [
                dict(observation) for observation in self.initial_observations
            ],
            "clarification_available": self.offer is not None,
        }

    def due(self, step: int) -> list[Release]:
        if step < 1:
            raise TimelineError("step must be positive")
        return [release for release in self.releases if release.step == step]

    def answer(
        self,
        question: str,
        step: int,
        *,
        questions_used: int = 0,
    ) -> dict[str, Any]:
        if self.offer is None:
            raise TimelineError("clarification is not available in this run")
        return self.offer.answer(question, step, questions_used=questions_used)

    def sufficiency_step(self, events: Iterable[Mapping[str, Any]] = ()) -> int | None:
        """t*: the earliest step at which the final referent is determinable.

        None means the referent cannot be determined under this condition (for
        example initial_only on a Refresh-replace task, or Clarify without a valid
        answer).
        """

        # Sparse packets may omit the decisive fact, even after a valid answer.
        # Their evidence visibility is reviewed separately from release timing.
        if self.condition.schedule not in {
            COMPUTER_USE,
            OFFLINE_EGO_USE,
            HUMAN_SOLVABILITY,
        } and (
            self.condition.scope in {"first_frame", "none"}
            or self.condition.frame_interval_seconds is not None
        ):
            return None
        if self._sufficiency is not None:
            return self._sufficiency
        if self.offer is None:
            return None
        for event in events:
            if (
                event.get("type") == "clarification_answer"
                and event.get("valid") is True
            ):
                return int(event.get("step", 0))
        return None


def _release_step(
    condition: Condition,
    default_release_step: int,
    reference_steps: int | None,
) -> int:
    if condition.release_at is not None:
        if reference_steps is None:
            raise TimelineError("release_at requires reference_steps")
        return max(1, round(condition.release_at * reference_steps))
    return default_release_step


def _compile_computer_use(task: Mapping[str, Any], cond: Condition) -> Timeline:
    request = task["instructions"].get(COMPUTER_USE)
    if not isinstance(request, str) or not request.strip():
        raise TimelineError(
            f"{task['id']}: missing self-contained Computer Use request"
        )
    return Timeline(str(task["id"]), str(task["track"]), cond, request, (), (), None, 0)


def _compile_human_solvability(task: Mapping[str, Any], cond: Condition) -> Timeline:
    if task["track"] != COMPOSITE_TRACK:
        return _compile_offline_ego_use(task, cond)
    batches = _session_batches(task, cond)
    definition = composite_input(task)
    start = batches[0]["timestamps_seconds"][0]
    instruction = f'At source t={start:g} s: "{definition.opening}"'
    for update in definition.updates:
        if update.request is not None:
            instruction += (
                f'\n\nAt source t={update.onset_seconds:g} s: "{update.request}"'
            )
    offer = (
        _clarify_offer(
            task, instruction, composite_event_id=definition.clarification_id
        )
        if definition.clarification
        else _open_offer(instruction, cond)
    )
    return Timeline(
        str(task["id"]), COMPOSITE_TRACK, cond, instruction, tuple(batches), (), offer
    )


def _compile_offline_ego_use(task: Mapping[str, Any], cond: Condition) -> Timeline:
    batches = _session_batches(task, cond)
    _key_batch_indices(task, batches)
    track = str(task["track"])
    instructions = task["instructions"]
    if track == "graft":
        start = batches[0]["timestamps_seconds"][0]
        onset = task["visual_session"].get("update_onset_seconds")
        if onset is None or _change_batch(task, batches) is None:
            raise TimelineError(
                f"{task['id']}: offline Graft requires a reviewed request onset"
            )
        instruction = (
            f'At source t={start:g} s: "{instructions["a"]}"\n\n'
            f'At source t={onset:g} s: "{instructions["b"]}"'
        )
    else:
        instruction = instructions["base" if track == "refresh" else "ambiguous"]
    offer = (
        _clarify_offer(task, instruction)
        if track == "clarify"
        else _open_offer(instruction, cond)
    )
    return Timeline(
        str(task["id"]),
        track,
        cond,
        instruction,
        tuple(batches),
        (),
        offer,
        None if track == "clarify" else 0,
    )


def _compile_graft(
    task: Mapping[str, Any], cond: Condition, release_step: int
) -> Timeline:
    instructions = task["instructions"]
    v0 = _packet(task, "V0")
    cadence = _cadence_delivery(task, cond)
    observations = (
        cadence[0] if cadence is not None else _initial_observations(v0, cond)
    )
    releases = cadence[1] if cadence is not None else ()
    sufficiency: int | None
    if cond.scope == "initial_only":
        instruction = str(instructions["a"])
        sufficiency = None
    elif cond.schedule == "upfront":
        instruction = "\n\n".join(str(instructions[key]) for key in ("a", "b"))
        sufficiency = 0
    else:
        instruction = str(instructions["a"])
        if cadence is None:
            follow_up = Release("B", release_step, INSTRUCTION, str(instructions["b"]))
            releases = (follow_up,)
        else:
            follow_up = next(release for release in releases if release.ref == "B")
        sufficiency = follow_up.step
    return Timeline(
        task_id=str(task["id"]),
        track="graft",
        condition=cond,
        instruction=instruction,
        initial_observations=observations,
        releases=releases,
        offer=_open_offer(instruction, cond),
        _sufficiency=sufficiency,
    )


def _compile_refresh(
    task: Mapping[str, Any], cond: Condition, release_step: int
) -> Timeline:
    instruction = str(task["instructions"]["base"])
    v0 = _packet(task, "V0")
    v1 = _packet(task, "V1")
    cadence = _cadence_delivery(task, cond)
    observations = list(
        cadence[0] if cadence is not None else _initial_observations(v0, cond)
    )
    releases = cadence[1] if cadence is not None else ()
    policy = task["refresh"]["policy"]
    sufficiency: int | None
    if cond.scope in {"initial_only", "none"}:
        sufficiency = 0 if policy == "retain" else None
    elif cond.schedule == "upfront":
        if cadence is None and v1 is not None:
            observations.extend(_initial_observations(v1, cond))
        sufficiency = 0
    else:
        if cadence is None and v1 is not None:
            later_packet = _initial_observations(v1, cond)[0]
            releases = (Release("V1", release_step, OBSERVATION, later_packet),)
        if policy == "retain":
            sufficiency = 0
        elif cond.replay_timing == ACTION_ALIGNED_REPLAY:
            sufficiency = next(
                (release.step for release in releases if release.update), None
            )
        else:
            sufficiency = release_step
    return Timeline(
        task_id=str(task["id"]),
        track="refresh",
        condition=cond,
        instruction=instruction,
        initial_observations=tuple(observations),
        releases=releases,
        offer=_open_offer(instruction, cond),
        _sufficiency=sufficiency,
    )


def _compile_clarify(
    task: Mapping[str, Any], cond: Condition, release_step: int
) -> Timeline:
    instructions = task["instructions"]
    v0 = _packet(task, "V0")
    cadence = _cadence_delivery(task, cond)
    observations = (
        cadence[0] if cadence is not None else _initial_observations(v0, cond)
    )
    oracle_text = cond.schedule == "upfront"
    instruction = str(
        instructions["oracle"] if oracle_text else instructions["ambiguous"]
    )
    offer = None
    if cond.ask in {"open", "allowed"}:
        offer = _clarify_offer(task, instruction)
    sufficiency = 0 if oracle_text else None
    return Timeline(
        task_id=str(task["id"]),
        track="clarify",
        condition=cond,
        instruction=instruction,
        initial_observations=observations,
        releases=() if cadence is None else cadence[1],
        offer=offer,
        _sufficiency=sufficiency,
    )


def _compile_composite(task: Mapping[str, Any], cond: Condition) -> Timeline:
    cond.require_composite_replay()
    task_id = str(task["id"])
    definition = composite_input(task)
    opening = definition.opening
    batches = _session_batches(task, cond)
    if task["visual_session"].get("skipped_seconds"):
        raise TimelineError(f"{task_id}: composite requires a continuous source window")
    _key_batch_indices(task, batches)
    scheduled = [
        (_batch_for_onset(task_id, batches, update.onset_seconds), update)
        for update in definition.updates
    ]
    if len({index for index, _ in scheduled}) != len(scheduled):
        raise TimelineError(
            f"{task_id}: composite updates must use distinct source batches"
        )
    if definition.clarification and cond.ask == "denied":
        raise TimelineError(f"{task_id}: this composite requires a clarification offer")

    preview = cond.composite_upfront
    if preview is not None and definition.mechanisms != {"R", "G"}:
        raise TimelineError("composite_upfront applies only to Refresh + Graft")
    observations = (batches[0],)
    refresh = {
        index: update.event_id if definition.explicit_events else None
        for index, update in scheduled
        if update.mechanism == "R"
    }
    if preview in {"refresh", "both"}:
        observations += tuple(
            {**batches[index], SOURCE_PREVIEW: True} for index in refresh
        )
    offer = (
        _clarify_offer(
            task,
            opening,
            composite_event_id=definition.clarification_id
            if definition.explicit_events
            else None,
        )
        if definition.clarification
        else _open_offer(opening, cond)
    )
    releases = _observation_releases(batches, cond, refresh)
    for index, update in scheduled:
        if update.mechanism != "G":
            continue
        if preview in {"graft", "both"}:
            opening += (
                f"\n\n[Later wearer request at source t={update.onset_seconds:g} s]"
                f"\n{update.request}"
            )
        else:
            event_id = update.event_id if definition.explicit_events else None
            releases += (
                Release(
                    event_id or "B",
                    cond.replay_start_step + index,
                    INSTRUCTION,
                    update.request,
                    streamed=True,
                    update=True,
                    composite_event_id=event_id,
                ),
            )
    releases = tuple(sorted(releases, key=lambda release: release.step))
    return Timeline(
        task_id, COMPOSITE_TRACK, cond, opening, observations, releases, offer
    )


def _clarify_offer(
    task: Mapping[str, Any], instruction: str, *, composite_event_id: str | None = None
) -> ClarificationOffer:
    clarification = task["clarification"]
    return ClarificationOffer(
        request=instruction,
        candidate_ids=tuple(str(item) for item in clarification["candidate_ids"]),
        valid_slot=str(clarification["valid_slot"]),
        canonical_answer=str(clarification["canonical_answer"]),
        composite_event_id=composite_event_id,
        max_questions=int(clarification.get("max_questions", 1)),
    )


def _open_offer(instruction: str, cond: Condition) -> ClarificationOffer | None:
    """Graft and Refresh may ask under open asking; the wearer has nothing to add."""

    return ClarificationOffer(request=instruction) if cond.ask == "open" else None


def _cadence_delivery(
    task: Mapping[str, Any], cond: Condition
) -> tuple[tuple[dict[str, Any], ...], tuple[Release, ...]] | None:
    """Deliver a prepared session. None is for archived inputs.

    Under online delivery, frames, requests and agent actions advance independently:
    whether the change lands before or after the first commitment is a run
    outcome. Refresh marks the frame batch containing the change; Graft's
    follow-up request arrives with that batch.
    """

    interval = cond.frame_delivery_steps
    if interval is None:
        return None
    track = task["track"]
    if cond.scope == "initial_only" and track == "clarify":
        raise TimelineError("initial_only replay applies only to Graft and Refresh")
    batches = _session_batches(task, cond)
    reviewed = _key_batch_indices(task, batches)
    if cond.visual_history is None:
        for index in reviewed:
            batches[index][KEY_EVIDENCE] = True
    change = _change_batch(task, batches)
    if cond.schedule == "upfront":
        end = max(reviewed, default=None) if track == "clarify" else change
        if end is None:
            raise TimelineError(
                f"{task.get('id')}: upfront replay requires reviewed evidence"
            )
        return tuple(batches[: end + 1]), ()
    if track == "refresh" and cond.scope == "initial_only":
        if change is None:
            raise TimelineError(
                "initial_only Refresh replay requires a reviewed change"
            )
        batches = batches[:change]
    marked_batch = change if track == "refresh" else None
    start_step = cond.replay_start_step
    marks = {} if marked_batch is None else {marked_batch: None}
    releases = _observation_releases(batches, cond, marks)
    if cond.change_cue:
        if track != "refresh" or change is None:
            raise TimelineError("change_cue applies only to a Refresh change")
        cue = CHANGE_CUE_TEXT
        if cond.change_facts:
            facts = task["visual_session"].get("cue_facts", {})
            if (
                not isinstance(facts.get("text"), str)
                or not facts["text"].strip()
                or facts.get("timestamps_seconds")
                != batches[change]["timestamps_seconds"]
            ):
                raise TimelineError(
                    f"{task['id']}: cue facts must describe the delivered change batch"
                )
            cue += " " + facts["text"]
        releases += (
            Release(
                "CUE",
                start_step + change * interval,
                INSTRUCTION,
                cue,
                streamed=True,
            ),
        )
    if cond.follow_up_after_finish and track != "graft":
        raise TimelineError("follow_up_after_finish applies only to Graft")
    if track == "graft":
        if change is None:
            raise TimelineError(
                f"{task.get('id')}: a streamed Graft follow-up request requires "
                "visual_session.update_onset_seconds"
            )
        if cond.scope == "initial_only":
            return (batches[0],), releases
        follow_up = Release(
            "B",
            start_step + change * interval,
            INSTRUCTION,
            str(task["instructions"]["b"]),
            streamed=True,
            update=True,
            on_standby=cond.follow_up_after_finish,
        )
        releases += (follow_up,)
    return (() if cond.scope == "none" else (batches[0],)), releases


def _observation_releases(
    batches: list[dict[str, Any]],
    cond: Condition,
    marked_batches: Mapping[int, str | None],
) -> tuple[Release, ...]:
    if cond.scope == "none":
        return ()
    interval = cond.frame_delivery_steps
    assert interval is not None
    return tuple(
        Release(
            f"S{index}",
            cond.replay_start_step + index * interval,
            OBSERVATION,
            batch,
            streamed=True,
            update=index in marked_batches,
            composite_event_id=marked_batches.get(index),
        )
        for index, batch in enumerate(batches[1:], start=1)
    )


def _tiled(observation: dict[str, Any], cond: Condition) -> dict[str, Any]:
    if cond.wearer_frames_per_image == 4:
        observation["wearer_frames_per_image"] = 4
        if cond.wearer_jpeg_quality != JPEG_QUALITY:
            observation["wearer_jpeg_quality"] = cond.wearer_jpeg_quality
    return observation


def _change_batch(
    task: Mapping[str, Any],
    batches: list[dict[str, Any]],
    onset_field: str = "update_onset_seconds",
) -> int | None:
    """Index of the first batch at or after the reviewed change onset."""

    onset = task["visual_session"].get(onset_field)
    if onset is None:
        return None
    return _batch_for_onset(str(task.get("id")), batches, onset)


def _batch_for_onset(task_id: str, batches: list[dict[str, Any]], onset: float) -> int:
    for index, batch in enumerate(batches):
        if batch["timestamps_seconds"][-1] >= onset:
            if index == 0:
                raise TimelineError(
                    f"{task_id}: update onset falls in the opening batch"
                )
            return index
    raise TimelineError(f"{task_id}: update onset falls after the session")


def _session_batches(task: Mapping[str, Any], cond: Condition) -> list[dict[str, Any]]:
    session = task.get("visual_session")
    pool = session.get("frequency_samples") if isinstance(session, Mapping) else None
    interval = cond.frame_interval_seconds
    if not isinstance(pool, Mapping) or interval is None:
        raise TimelineError(
            f"{task.get('id')}: frame_delivery_steps requires "
            "visual_session.frequency_samples"
        )
    sampled = sample_packet(
        {"frequency_samples": pool},
        interval,
        cond.frame_repeat_to_interval_seconds,
        session.get("skipped_seconds"),
    )
    paths = sampled["frame_paths"]
    times = sampled["timestamps_seconds"]
    packing = session.get("wearer_frames_per_image", cond.wearer_frames_per_image)
    try:
        packed_image_size(1, 1, packing)
    except ValueError as error:
        raise TimelineError(str(error)) from error
    group = cond.wearer_frames_per_image
    size = (
        group
        if cond.batch_seconds is None
        else round(cond.batch_seconds / cond.frame_spacing_seconds)
    )
    batches: list[dict[str, Any]] = []
    for start in range(0, len(paths), size):
        batch: dict[str, Any] = {
            "frame_paths": paths[start : start + size],
            "timestamps_seconds": times[start : start + size],
            "frame_interval_seconds": sampled["frame_interval_seconds"],
        }
        repeated = sampled.get("frame_repeat_to_interval_seconds")
        if repeated is not None:
            batch["frame_repeat_to_interval_seconds"] = repeated
        if session.get("wearer_max_image_edge") is not None:
            batch["wearer_max_image_edge"] = session["wearer_max_image_edge"]
        rendered = _tiled(batch, cond)
        if "wearer_frames_per_image" in session:
            rendered["wearer_frames_per_image"] = packing
        batches.append(rendered)
    if not batches:
        raise TimelineError(
            f"{task.get('id')}: continuous frame delivery produced no frames"
        )
    return batches


def _key_batch_indices(
    task: Mapping[str, Any], batches: list[dict[str, Any]]
) -> set[int]:
    """Batches holding the frame nearest each reviewed time; each must be streamed."""

    tolerance = batches[0]["frame_interval_seconds"]
    indices = set()
    for reviewed in task["visual_session"].get("reviewed_seconds", ()):
        distance, index = min(
            (abs(time - reviewed), index)
            for index, batch in enumerate(batches)
            for time in batch["timestamps_seconds"]
        )
        if distance > tolerance:
            raise TimelineError(
                f"{task.get('id')}: reviewed frame at {reviewed} s is not in the stream"
            )
        indices.add(index)
    return indices


def _initial_observations(
    v0: Mapping[str, Any] | None,
    cond: Condition,
) -> tuple[dict[str, Any], ...]:
    if v0 is None or cond.scope == "none":
        return ()
    if cond.frame_interval_seconds is not None:
        observation = sample_packet(
            v0, cond.frame_interval_seconds, cond.frame_repeat_to_interval_seconds
        )
    elif cond.scope == "first_frame":
        observation = _first_frame(v0)
    else:
        observation = _frames(v0)
    if cond.wearer_frames_per_image == 4:
        observation["wearer_frames_per_image"] = 4
        if cond.wearer_jpeg_quality != JPEG_QUALITY:
            observation["wearer_jpeg_quality"] = cond.wearer_jpeg_quality
        if "timestamps_seconds" not in observation and "timestamps_seconds" in v0:
            observation["timestamps_seconds"] = list(v0["timestamps_seconds"])[
                : len(observation["frame_paths"])
            ]
    return (observation,)


def complete_materials_condition(source: Condition) -> Condition:
    if source.scope != "full" or source.batch_seconds is None:
        raise ValueError("human solvability requires a frozen visual task")
    return replace(
        source,
        name=HUMAN_SOLVABILITY,
        schedule=HUMAN_SOLVABILITY,
        scope="full",
        ask="open",
        release_at=None,
        visual_history="full",
        replay_timing=ACTION_ALIGNED_REPLAY,
        strategy_hint=False,
        change_cue=False,
        change_facts=False,
        follow_up_after_finish=False,
        composite_upfront=None,
    )
