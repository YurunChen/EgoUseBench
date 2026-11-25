"""Read candidate judgments against the task's released evidence.

Within a fixed reference, evidence narrows the possible candidates. Refresh-replace
may establish a new reference when the later batch arrives; Refresh-retain may
preserve an initially clear reference. The final stable singleton determines t*.

Records may name every frame or only key events. Omitted frames retain the
previous judgment; both forms expand into the same in-memory curve.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .final_state import evaluate_final_state
from .schema import ValidationError, acceptable_states_of

# The row for evidence that arrives as words rather than as a frame — the user
# answering a question the agent asked, or volunteering an update unprompted.
UTTERANCE_ROW = "utterance"

# What the narrowing keyed on. Four values, because the distinction the benchmark
# rests on is exactly this: a referent fixed by the wearer's own vantage or by the
# order of their actions needs the video, and one fixed by how the scene looks does
# not — a sentence carries it just as well. `scene_appearance` covers both an
# object's own look and where it sits relative to other objects, since "the silver
# one" and "the one left of the Airheads bag" are equally writable; what makes
# position indexical is being relative to the wearer, not to the shelf.
NARROWING_CAUSES = frozenset(
    {
        "scene_appearance",
        "wearer_vantage",
        "action_order",
        "user_utterance",
    }
)
INDEXICAL_CAUSES = frozenset({"wearer_vantage", "action_order"})


class DecisionError(ValueError):
    """Raised when a survival curve is not a record of a decision."""


@dataclass(frozen=True)
class Observation:
    frame: str
    live: tuple[str, ...]
    narrowed_by: str | None
    why: str | None


@dataclass(frozen=True)
class DecisionState:
    """One complete commitment; alternatives remain separate states."""

    selected_entity_ids: tuple[str, ...]
    fields: Mapping[str, str] | None = None

    goal: Mapping[str, Any] | None = None

    def matches(self, state: Mapping[str, Any]) -> bool:
        if self.goal is not None:
            return evaluate_final_state(self.goal, state)
        return Counter(self.selected_entity_ids) == Counter(
            state["selected_entity_ids"]
        ) and (self.fields is None or self.fields == state["fields"])


@dataclass(frozen=True)
class DecisionCurve:
    """A frame-by-frame reading of which candidates remain possible.

    A candidate is a possible goal, with one or more complete alternative states.
    A literal entity list requires every listed entity; a Graft goal reference
    preserves the alternatives and field values owned by the compiled goal.
    """

    candidates: Mapping[str, tuple[DecisionState, ...]]
    observations: tuple[Observation, ...]
    established_by: str

    @property
    def settled_at(self) -> str | None:
        """First row of the final, stable single-reference interval."""

        if len(self.final_live) != 1:
            return None
        start = self.observations[-1].frame
        for observation in reversed(self.observations):
            if observation.live != self.final_live:
                break
            start = observation.frame
        return start

    @property
    def settled_by_frames(self) -> bool:
        return self.settled_at is not None and self.settled_at != UTTERANCE_ROW

    @property
    def final_live(self) -> tuple[str, ...]:
        return self.observations[-1].live

    @property
    def tempting_roles(self) -> tuple[str, ...]:
        """Candidates alive right up to the row that settled it, and then not.

        This is the failure surface. An agent that commits before the evidence
        settles picks one of these, which is the mistake the benchmark catches.
        """

        settled = self.settled_at
        alive_before: tuple[str, ...] = tuple(self.candidates)
        for observation in self.observations:
            if observation.frame == settled:
                break
            alive_before = observation.live
        return tuple(role for role in alive_before if role not in self.final_live)

    @property
    def narrowing_cause(self) -> str | None:
        """What settled it, read at the row where it was settled."""

        target = self.settled_at
        for observation in self.observations:
            if observation.frame == target:
                return observation.narrowed_by
        return None

    @property
    def tempting_entity_ids(self) -> tuple[str, ...]:
        """What committing to a losing candidate would put in the final state.

        Where candidates are goal states rather than single objects they overlap —
        "the gun" and "the gun and its sticks" share the gun — and the failure
        surface is only what the losing state adds, never what both agree on.
        """

        settled = set(self.entity_ids(self.final_live))
        return tuple(sorted(set(self.entity_ids(self.tempting_roles)) - settled))

    @property
    def is_indexical(self) -> bool:
        return self.narrowing_cause in INDEXICAL_CAUSES

    def entity_ids(self, roles: Iterable[str]) -> tuple[str, ...]:
        """Project referenced entities for diagnostics, never for state matching."""

        return tuple(
            entity_id
            for state in self.goal_states(roles)
            for entity_id in state.selected_entity_ids
        )

    def goal_states(self, roles: Iterable[str]) -> tuple[DecisionState, ...]:
        return tuple(state for role in roles for state in self.candidates[role])


def _candidate_states(
    value: Any,
    task: Mapping[str, Any] | None,
    context: str,
) -> tuple[DecisionState, ...]:
    if isinstance(value, dict):
        if (
            set(value) != {"goal_ref"}
            or not isinstance(value["goal_ref"], str)
            or value["goal_ref"] not in {"initial", "final"}
        ):
            raise DecisionError(
                f"{context}: goal_ref must be the sole key and name initial or final"
            )
        if task is None or task.get("track") != "graft":
            raise DecisionError(f"{context}: goal_ref requires a Graft task")
        graft = task.get("graft")
        goal = (
            task.get("goal")
            if value["goal_ref"] == "final"
            else (graft.get("initial_goal") if isinstance(graft, Mapping) else None)
        )
        if not isinstance(goal, dict):
            raise DecisionError(f"{context}: goal_ref has no owned goal")
        try:
            states = acceptable_states_of(
                goal, f"{context}/goal_ref/{value['goal_ref']}"
            )
        except ValidationError as exc:
            raise DecisionError(str(exc)) from exc
        return tuple(
            DecisionState(tuple(state["selected_entity_ids"]), dict(state["fields"]), {**goal, "acceptable_states": [state]})
            for state in states
        )
    ids = [value] if isinstance(value, str) else value
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(entity_id, str) or not entity_id for entity_id in ids)
    ):
        raise DecisionError(f"{context}: candidate must name at least one entity")
    return (DecisionState(tuple(ids)),)


def frame_ids(packets: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every released frame, in the order the packets deliver them."""

    return [
        f"{packet['id']}:{index}"
        for packet in packets
        for index in range(len(packet["frame_paths"]))
    ]


def _observation(entry: Any, context: str) -> Observation:
    if not isinstance(entry, dict):
        raise DecisionError(f"{context}: each observation must be an object")
    frame = entry.get("frame")
    live = entry.get("live")
    if not isinstance(frame, str) or not frame:
        raise DecisionError(f"{context}: observation needs a frame reference")
    if not isinstance(live, list) or any(
        not isinstance(role, str) or not role for role in live
    ):
        raise DecisionError(f"{context}/{frame}: live must list candidate roles")
    if len(set(live)) != len(live):
        raise DecisionError(f"{context}/{frame}: live repeats a candidate")
    narrowed_by = entry.get("narrowed_by")
    if narrowed_by is not None and narrowed_by not in NARROWING_CAUSES:
        raise DecisionError(
            f"{context}/{frame}: narrowed_by must be one of {sorted(NARROWING_CAUSES)}"
        )
    why = entry.get("why")
    if why is not None and (not isinstance(why, str) or not why.strip()):
        raise DecisionError(f"{context}/{frame}: why must be a non-empty explanation")
    return Observation(frame=frame, live=tuple(live), narrowed_by=narrowed_by, why=why)


def _require_one_row_per_frame(
    observations: Sequence[Observation], released: Sequence[str], context: str
) -> None:
    seen = [observation.frame for observation in observations]
    expected = list(released)
    if seen[-1:] == [UTTERANCE_ROW]:
        expected = expected + [UTTERANCE_ROW]
    if seen != expected:
        missing = [frame for frame in expected if frame not in seen]
        unknown = [frame for frame in seen if frame not in expected]
        raise DecisionError(
            f"{context}: the curve must read every released frame in order, optionally "
            f"followed by one {UTTERANCE_ROW!r} row; "
            f"missing={missing[:4]}, unknown={unknown[:4]}"
        )


def _expand_key_events(
    observations: Sequence[Observation], released: Sequence[str], context: str
) -> tuple[Observation, ...]:
    """Carry unchanged judgments forward without storing duplicate source rows."""
    expected = list(released) + [UTTERANCE_ROW]
    seen = [row.frame for row in observations]
    if (
        not released
        or seen[0] != released[0]
        or len(set(seen)) != len(seen)
        or any(frame not in expected for frame in seen)
        or seen != sorted(seen, key=expected.index)
    ):
        raise DecisionError(
            f"{context}: key events must start at the first released frame and "
            "use unique known frame references in release order"
        )
    by_frame = {row.frame: row for row in observations}
    current = observations[0]
    expanded = []
    for frame in released:
        if frame in by_frame:
            current = by_frame[frame]
            expanded.append(current)
        else:
            expanded.append(Observation(frame, current.live, None, None))
    if UTTERANCE_ROW in by_frame:
        expanded.append(by_frame[UTTERANCE_ROW])
    return tuple(expanded)


def _require_monotone_narrowing(
    observations: Sequence[Observation],
    candidates: Mapping[str, tuple[DecisionState, ...]],
    context: str,
    *,
    replacement_frame: str | None = None,
) -> None:
    previous: set[str] = set(candidates)
    for observation in observations:
        live = set(observation.live)
        unknown = live - set(candidates)
        if unknown:
            raise DecisionError(
                f"{context}/{observation.frame}: live names roles that are not "
                f"candidates: {sorted(unknown)}"
            )
        if not live:
            raise DecisionError(
                f"{context}/{observation.frame}: no candidate is left alive"
            )
        if observation.frame == replacement_frame and not live <= previous:
            previous = set(candidates)
        if not live <= previous:
            raise DecisionError(
                f"{context}/{observation.frame}: candidates came back to life: "
                f"{sorted(live - previous)}"
            )
        narrowed = live < previous
        if narrowed and not (observation.narrowed_by and observation.why):
            raise DecisionError(
                f"{context}/{observation.frame}: this frame ruled a candidate out, so it "
                f"needs narrowed_by and why"
            )
        if not narrowed and observation.narrowed_by:
            raise DecisionError(
                f"{context}/{observation.frame}: narrowed_by is set but nothing was ruled out"
            )
        previous = live


def parse(
    declared: Any,
    released_frames: Sequence[str],
    site_entity_ids: set[str],
    context: str,
    *,
    task: Mapping[str, Any] | None = None,
) -> DecisionCurve:
    """Read a declared curve, refusing anything that is not a record of a decision."""

    if not isinstance(declared, dict):
        raise DecisionError(f"{context}: decision must be an object")
    declared_candidates = declared.get("candidates")
    if not isinstance(declared_candidates, dict) or len(declared_candidates) < 2:
        raise DecisionError(f"{context}: decision.candidates needs at least two roles")
    candidates = {
        role: _candidate_states(value, task, f"{context}/{role}")
        for role, value in declared_candidates.items()
    }
    unknown_entities = {
        entity_id
        for states in candidates.values()
        for state in states
        for entity_id in state.selected_entity_ids
        if entity_id not in site_entity_ids
    }
    if unknown_entities:
        raise DecisionError(
            f"{context}: candidates point at entities the site does not have: "
            f"{sorted(unknown_entities)}"
        )
    established_by = declared.get("established_by")
    if not isinstance(established_by, str) or not established_by.strip():
        raise DecisionError(
            f"{context}: decision.established_by must name who read the frames"
        )

    rows = declared.get("observed")
    if not isinstance(rows, list) or not rows:
        raise DecisionError(f"{context}: decision.observed must read the frames")
    observations = tuple(_observation(entry, context) for entry in rows)
    recording = declared.get("recording", "per_frame")
    if recording == "key_events":
        observations = _expand_key_events(observations, released_frames, context)
    elif recording != "per_frame":
        raise DecisionError(f"{context}: unknown decision recording mode {recording!r}")
    _require_one_row_per_frame(observations, released_frames, context)
    replacing = (
        task is not None
        and task.get("track") == "refresh"
        and isinstance(task.get("refresh"), Mapping)
        and task["refresh"].get("policy") == "replace"
    )
    _require_monotone_narrowing(
        observations,
        candidates,
        context,
        replacement_frame="V1:0" if replacing else None,
    )

    retain = (
        task is not None
        and task.get("track") == "refresh"
        and isinstance(task.get("refresh"), Mapping)
        and task["refresh"].get("policy") == "retain"
    )
    changed_reference = (
        replacing
        and any(row.frame == "V1:0" for row in observations)
        and observations[0].live != observations[-1].live
    )
    if not retain and not changed_reference and not any(
        len(observation.live) > 1 for observation in observations
    ):
        raise DecisionError(
            f"{context}: no frame leaves more than one candidate alive, so there is "
            f"nothing to decide"
        )
    return DecisionCurve(
        candidates=candidates,
        observations=observations,
        established_by=established_by,
    )
