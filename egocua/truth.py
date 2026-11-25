"""Diagnostic Ground Truth compiled from a task: what the referent is, and when it changes.

The evaluator reads only this object plus the compiled Timeline; it never branches
on track names itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .schema import acceptable_states_of
from .final_state import goal_entity_ids, goal_signature
from .timeline import COMPOSITE_TRACK


@dataclass(frozen=True)
class Truth:
    task_id: str
    track: str
    initial_referent_ids: frozenset[str] | None
    tempting_referent_ids: frozenset[str]
    candidate_ids: frozenset[str]
    initial_acceptable_states: tuple[dict[str, Any], ...] | None
    final_acceptable_states: tuple[dict[str, Any], ...]
    later_release_changes_referent: bool
    later_release_should_preserve_referent: bool
    transition: str | None
    initial_goal: Mapping[str, Any] | None = None
    final_goal: Mapping[str, Any] | None = None

    @property
    def final_referent_ids(self) -> frozenset[str]:
        assert self.final_goal is not None
        return goal_entity_ids(self.final_goal)

    @property
    def invalidated_referent_ids(self) -> frozenset[str] | None:
        """Entities acceptable on initial evidence that the later release rules out."""

        if self.initial_referent_ids is None:
            return None
        return self.initial_referent_ids - self.final_referent_ids

    @property
    def required_after_update_ids(self) -> frozenset[str] | None:
        """Entities introduced by a goal update and required in the final state."""

        if self.transition == "relax":
            return frozenset()
        if self.initial_referent_ids is None:
            return None
        return self.final_referent_ids - self.initial_referent_ids

    @property
    def initial_goal_differs(self) -> bool | None:
        if self.initial_acceptable_states is None:
            return None
        assert self.initial_goal is not None and self.final_goal is not None
        return goal_signature(self.initial_goal) != goal_signature(self.final_goal)

    @classmethod
    def of(cls, task: Mapping[str, Any]) -> Truth:
        task_id = str(task["id"])
        track = str(task["track"])
        goal = task["goal"]
        final_states = tuple(acceptable_states_of(goal, f"{task_id}/goal"))
        declared: Mapping[str, Any] = (
            task["truth"] if isinstance(task.get("truth"), Mapping) else {}
        )
        tempting = frozenset(
            str(item) for item in declared.get("tempting_referent_ids") or []
        )
        if track == "graft":
            graft = task["graft"]
            initial_goal = graft["initial_goal"]
            initial_states = tuple(
                acceptable_states_of(initial_goal, f"{task_id}/graft/initial_goal")
            )
            initial_ids = goal_entity_ids(initial_goal)
            transition = str(graft["transition"])
            final_ids = goal_entity_ids(goal)
            invalidated = initial_ids - final_ids
            if not tempting:
                tempting = invalidated
            changes = transition in {"replace", "drop"} or bool(invalidated)
            return cls(
                task_id,
                track,
                initial_ids,
                tempting,
                frozenset(),
                initial_states,
                final_states,
                changes,
                False,
                transition,
                initial_goal=initial_goal,
                final_goal=goal,
            )
        if track == "refresh":
            refresh = task["refresh"]
            initial_states = (
                tuple(
                    acceptable_states_of(
                        {"acceptable_states": declared["initial_acceptable_states"]},
                        f"{task_id}/truth/initial_acceptable_states",
                    )
                )
                if "initial_acceptable_states" in declared
                else None
            )
            initial_ids = (
                frozenset(
                    entity_id
                    for state in initial_states
                    for entity_id in state["selected_entity_ids"]
                )
                if initial_states is not None
                else None
            )
            replace = refresh["policy"] == "replace"
            if replace and not tempting and initial_ids is not None:
                tempting = initial_ids - frozenset(
                    entity_id
                    for state in final_states
                    for entity_id in state["selected_entity_ids"]
                )
            return cls(
                task_id,
                track,
                initial_ids,
                tempting,
                frozenset(),
                initial_states,
                final_states,
                replace,
                not replace,
                None,
                initial_goal={"acceptable_states": list(initial_states)} if initial_states is not None else None,
                final_goal=goal,
            )
        if track == COMPOSITE_TRACK:
            return cls(
                task_id=task_id,
                track=track,
                initial_referent_ids=None,
                tempting_referent_ids=tempting,
                candidate_ids=frozenset(task.get("clarification", {}).get("candidate_ids", ())),
                initial_acceptable_states=None,
                final_acceptable_states=final_states,
                later_release_changes_referent=False,
                later_release_should_preserve_referent=False,
                transition=None,
                final_goal=goal,
            )
        clarification = task["clarification"]
        candidates = frozenset(str(item) for item in clarification["candidate_ids"])
        final_ids = goal_entity_ids(goal)
        if not tempting:
            tempting = candidates - final_ids
        return cls(
            task_id,
            track,
            candidates,
            tempting,
            candidates,
            final_states,
            final_states,
            False,
            False,
            None,
            initial_goal=goal,
            final_goal=goal,
        )
