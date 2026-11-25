"""Run-time session over a compiled evidence Timeline.

`RunSession` is the stateful Interface the executor uses: it hands out the initial
payload, releases due evidence as the Agent advances, and answers one
clarification question. All track and condition knowledge lives in
`egocua.timeline`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable, Mapping
from typing import Any

from .timeline import INSTRUCTION, Condition, Timeline, TimelineError

__all__ = [
    "ProtocolError",
    "RunSession",
    "SELECTION_ACTIONS",
    "first_native_commitment_step",
    "RELEASE_DELIVERY_DESCRIPTIONS",
    "STAGE_RELEASE_TRIGGERS",
    "first_release_trigger",
    "validate_stage_budget",
]

SELECTION_ACTIONS = frozenset(
    {
        "select",
        "save",
        "select_entity",
        "deselect_entity",
        "add_to_cart",
        "remove_from_cart",
        "update_cart",
        "select_route",
        "select_destination",
    }
)
NATIVE_COMMITMENT_ACTIONS = SELECTION_ACTIONS | {"set_field"}
STAGE_RELEASE_TRIGGERS = frozenset(
    {"first_search", "first_detail", "first_selection", "two_selections"}
)
RELEASE_DELIVERY_DESCRIPTIONS = {
    "timeline_step": "At the scheduled Agent step.",
    "first_native_commitment": (
        "At the next Agent step after the first recorded website commitment "
        "(selection, deselection, cart, route, destination or field mutation), "
        "even if later undone or submitted; waiting alone does not release evidence. "
        "Release steps below are nominal."
    ),
    "first_search": "At the next Agent step after a non-empty search with results.",
    "first_detail": "At the next Agent step after successfully opening an entity detail.",
    "first_selection": "At the next Agent step after adding a new entity to website state.",
    "two_selections": "At the next Agent step after adding an entity produces at least two distinct website selections.",
}


def validate_stage_budget(
    trigger: str, budgets: tuple[int | None, int | None], total: int
) -> None:
    if budgets == (None, None):
        return
    if trigger not in STAGE_RELEASE_TRIGGERS:
        raise ValueError("stage budgets require a stage release trigger")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in budgets
    ):
        raise ValueError(
            "pre_release_max_steps and post_release_max_steps must both be positive"
        )
    if sum(value for value in budgets if value is not None) != total:
        raise ValueError("max_steps must equal the sum of pre/post release budgets")


def first_release_trigger(
    actions: Iterable[Mapping[str, Any]],
    trigger: str,
) -> tuple[int, Mapping[str, Any]] | None:
    """Find native action evidence without inspecting task answers."""

    for index, action in enumerate(actions):
        if trigger == "first_native_commitment":
            matched = action.get("type") in NATIVE_COMMITMENT_ACTIONS
        elif trigger == "first_search":
            results = action.get("results")
            matched = (
                action.get("type") == "search"
                and bool(str(action.get("query") or "").strip())
                and isinstance(results, int)
                and not isinstance(results, bool)
                and results > 0
            )
        elif trigger == "first_detail":
            # Native sites emit open_entity only after resolving an existing entity.
            matched = action.get("type") == "open_entity" and bool(
                action.get("entity_id")
            )
        elif trigger in {"first_selection", "two_selections"}:
            before, after = action.get("state_before"), action.get("state_after")
            before_ids = (
                before.get("selected_entity_ids")
                if isinstance(before, Mapping)
                else None
            )
            after_ids = (
                after.get("selected_entity_ids") if isinstance(after, Mapping) else None
            )
            matched = (
                action.get("type") in SELECTION_ACTIONS
                and isinstance(before_ids, list)
                and isinstance(after_ids, list)
                and bool(set(after_ids) - set(before_ids))
                and (trigger == "first_selection" or len(set(after_ids)) >= 2)
            )
        else:
            return None
        if matched:
            return index, action
    return None


def first_native_commitment_step(actions: Iterable[Mapping[str, Any]]) -> int | None:
    """Locate the first website mutation, even if it was later undone or submitted."""

    return next(
        (
            int(action.get("step", 0))
            for action in actions
            if action.get("type") in NATIVE_COMMITMENT_ACTIONS
        ),
        None,
    )


class ProtocolError(RuntimeError):
    """Raised for an invalid interaction with a benchmark session."""


@dataclass
class RunSession:
    """Stateful session over one compiled Timeline."""

    task: dict[str, Any]
    condition: Condition
    event_step: int
    reference_steps: int | None
    events: list[dict[str, Any]] = field(default_factory=list)
    questions_used: int = 0
    timeline: Timeline = field(init=False, repr=False)
    _delivered_release_refs: set[str] = field(default_factory=set, repr=False)
    _clock_offset: int = field(default=0, repr=False)
    """Source batches skipped while the Agent stood by; streams only."""
    _woken_refs: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        try:
            self.timeline = Timeline.compile(
                self.task,
                self.condition,
                default_release_step=self.event_step,
                reference_steps=self.reference_steps,
            )
        except TimelineError as exc:
            raise ProtocolError(str(exc)) from exc

    def initial_payload(self) -> dict[str, Any]:
        return self.timeline.initial()

    def advance(
        self,
        step: int,
        *,
        release_pending: bool = False,
        allow_scheduled_release: bool = True,
        trigger_reason: str | None = None,
    ) -> list[dict[str, Any]]:
        if step < 1:
            raise ProtocolError("step must be positive")
        source_step = step + self._clock_offset
        due = [
            release
            for release in self.timeline.releases
            if release.ref not in self._delivered_release_refs
            and (not release.on_standby or release.ref in self._woken_refs)
            and (
                release.step <= source_step
                if release.streamed
                else release_pending
                or (allow_scheduled_release and release.step <= step)
            )
        ]
        if not due:
            return []
        trigger_reason = trigger_reason or (
            "first_native_commitment" if release_pending else "task_timeline"
        )
        events = [
            {
                **release.as_event(),
                "step": step,
                **(
                    {"source_step": release.step}
                    if release.streamed
                    else {"trigger_reason": trigger_reason}
                ),
            }
            for release in due
        ]
        self._delivered_release_refs.update(release.ref for release in due)
        self.events.extend(events)
        return events

    def standby(self, step: int) -> dict[str, Any] | None:
        """After the Agent finishes, wait for the wearer's next streamed request.

        Only a request wakes the Agent; a scene change does not. Batches the
        wait passes over are skipped, never delivered later. A request held
        until the finish arrives at its own batch or, when that has already
        played, at the next one.
        """
        pending = [
            release
            for release in self.timeline.releases
            if release.streamed and release.ref not in self._delivered_release_refs
        ]
        wake = min(
            (r for r in pending if r.kind == INSTRUCTION and r.update),
            key=lambda release: release.step,
            default=None,
        )
        if wake is None:
            return None
        resume = max(wake.step, step + self._clock_offset + 1)
        skipped = [
            release.ref
            for release in pending
            if release.step < resume and release is not wake
        ]
        self._delivered_release_refs.update(skipped)
        self._woken_refs.add(wake.ref)
        self._clock_offset = resume - (step + 1)
        event = {
            "type": "standby",
            "step": step,
            "skipped_refs": skipped,
            "resume_source_step": resume,
        }
        self.events.append(event)
        return event

    def ask_user(self, question: str, step: int) -> dict[str, Any]:
        if self.timeline.offer is None:
            raise ProtocolError("clarification is not available in this run")
        event = self.timeline.answer(
            question, step, questions_used=self.questions_used
        )
        if not event.get("limit_reached"):
            self.questions_used += 1
            self.events.append(event)
        return event
