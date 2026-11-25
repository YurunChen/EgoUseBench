"""Structural task difficulty without pretending that design is performance.

The synthesis pipeline knows the task's evidence and required state transition.  It
can therefore describe *where* difficulty was designed.  Only a real Agent run can
say whether the task was actually difficult for that Agent.  Keeping those two facts
separate prevents a long instruction or a hand-written ``hard`` label from becoming
fake evidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .decision import DecisionCurve
from .truth import Truth

STRUCTURAL_DIFFICULTY_VERSION = "structural-difficulty-v8"
COMPOSITIONAL_FAILURE_PREDICATES = 3

DIFFICULTY_BANDS = ("L1", "L2", "L3", "unclassified")


@dataclass(frozen=True)
class StructuralDifficulty:
    """Observable challenge axes derived from task truth, never hand-scored."""

    task_id: str
    band: str
    challenges: tuple[str, ...]
    burdens: tuple[str, ...]
    risks: tuple[str, ...]
    opportunities: tuple[str, ...]
    evidence: Mapping[str, Any]
    state_transition: Mapping[str, Any]
    web_execution: Mapping[str, Any]

    @property
    def selection_key(self) -> tuple[int, str]:
        """Keep risk ordering without preferring a level or counting burdens."""

        return (len(self.risks), self.task_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_version": STRUCTURAL_DIFFICULTY_VERSION,
            "band": self.band,
            "challenges": list(self.challenges),
            "burdens": list(self.burdens),
            "design_risks": list(self.risks),
            "improvement_opportunities": list(self.opportunities),
            "evidence": dict(self.evidence),
            "state_transition": dict(self.state_transition),
            "web_execution": dict(self.web_execution),
        }


def _intent_challenge(track: str) -> str:
    return {
        "clarify": "clarify_before_commitment",
        "graft": "integrate_late_instruction",
        "refresh": "respond_to_later_visual_evidence",
    }[track]


def reviewed_difficulty(construct: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Read a reviewed minimum-solution argument, not an automatic semantic proof."""

    evidence = construct.get("difficulty")
    if not isinstance(evidence, Mapping):
        return "unclassified", {"reason": "missing_difficulty_review"}
    level = evidence.get("level")
    if level not in DIFFICULTY_BANDS:
        raise ValueError(f"Unknown difficulty level: {level!r}")
    required_text = ("reviewed_by", "simplest_solution", "no_simpler_solution")
    if level == "unclassified" or any(
        not isinstance(evidence.get(key), str) or not evidence[key].strip()
        for key in required_text
    ):
        return "unclassified", {"reason": "incomplete_difficulty_review"}
    references = evidence.get("evidence_refs")
    conditions = evidence.get("necessary_conditions")
    dependencies = evidence.get("dependencies")
    if (
        not _nonempty_strings(references)
        or not isinstance(conditions, list)
        or not conditions
    ):
        return "unclassified", {"reason": "missing_necessity_evidence"}
    condition_fields = ("id", "requirement", "independent_counterexample")
    if any(
        not isinstance(row, Mapping)
        or not _nonempty_strings([row.get(key) for key in condition_fields])
        for row in conditions
    ):
        return "unclassified", {"reason": "incomplete_necessity_evidence"}
    condition_ids = {row["id"] for row in conditions}
    if len(condition_ids) != len(conditions):
        raise ValueError("Difficulty necessary-condition IDs must be unique")
    if not isinstance(dependencies, list):
        return "unclassified", {"reason": "dependencies_not_reviewed"}
    for dependency in dependencies:
        if (
            not isinstance(dependency, Mapping)
            or dependency.get("from") not in condition_ids
            or dependency.get("to") not in condition_ids
            or dependency["from"] == dependency["to"]
            or not _nonempty_strings([dependency.get("proof")])
        ):
            return "unclassified", {"reason": "incomplete_dependency_evidence"}
    fits_level = {
        "L1": len(conditions) == 1 and not dependencies,
        "L2": len(conditions) >= 2 and not dependencies,
        "L3": len(conditions) >= 2 and bool(dependencies),
    }
    if not fits_level[level]:
        return "unclassified", {"reason": "level_evidence_mismatch"}
    return str(level), {**dict(evidence), "method": "coding_agent_semantic_review"}


def _nonempty_strings(values: Any) -> bool:
    return (
        isinstance(values, list)
        and bool(values)
        and all(isinstance(value, str) and bool(value.strip()) for value in values)
    )


def _requires_temporal_grounding(source_record: Mapping[str, Any] | None) -> bool:
    if not source_record:
        return False
    source = source_record.get("source")
    if not isinstance(source, Mapping):
        return False
    events = source.get("observed_events")
    if not isinstance(events, list):
        return False
    temporal_roles = {"ordered_pair", "sequence", "temporal_order"}
    return any(
        isinstance(event, Mapping) and event.get("role") in temporal_roles
        for event in events
    )


def _selected_sets(states: tuple[dict[str, Any], ...]) -> tuple[frozenset[str], ...]:
    return tuple(frozenset(state["selected_entity_ids"]) for state in states)


def _native_edit_profile(
    truth: Truth,
) -> dict[str, int | None]:
    """Describe required website-state edits without guessing an Agent's path.

    Both initially acceptable states and final states may have alternatives.
    Clarify has no valid pre-clarification commitment; its single final save does
    not repair an earlier native state.
    """

    if (truth.final_goal or {}).get("unit_equivalences") or (truth.initial_goal or {}).get("unit_equivalences"):
        # Retail witnesses do not bound native edits across all mixed-package carts.
        return {
            **{f"{bound}_{metric}": None
               for bound in ("minimum", "maximum")
               for metric in ("total_edits", "additions", "removals", "retained")},
            "minimum_terminal_entities": None,
            "maximum_terminal_entities": None,
        }
    final_sets = _selected_sets(truth.final_acceptable_states)
    if truth.initial_acceptable_states is None:
        return {
            **{
                f"{bound}_{metric}": None
                for bound in ("minimum", "maximum")
                for metric in ("total_edits", "additions", "removals", "retained")
            },
            "minimum_terminal_entities": min(map(len, final_sets)),
            "maximum_terminal_entities": max(map(len, final_sets)),
        }
    if truth.track in {"graft", "refresh"}:
        initial_sets = _selected_sets(truth.initial_acceptable_states)
    else:
        initial_sets = (frozenset(),)
    transitions = [
        {
            "additions": len(final - initial),
            "removals": len(initial - final),
            "retained": len(initial & final),
        }
        for initial in initial_sets
        for final in final_sets
    ]
    totals = [row["additions"] + row["removals"] for row in transitions]
    terminal_sizes = [len(state) for state in final_sets]
    return {
        "minimum_total_edits": min(totals),
        "maximum_total_edits": max(totals),
        "minimum_additions": min(row["additions"] for row in transitions),
        "maximum_additions": max(row["additions"] for row in transitions),
        "minimum_removals": min(row["removals"] for row in transitions),
        "maximum_removals": max(row["removals"] for row in transitions),
        "minimum_retained": min(row["retained"] for row in transitions),
        "maximum_retained": max(row["retained"] for row in transitions),
        "minimum_terminal_entities": min(terminal_sizes),
        "maximum_terminal_entities": max(terminal_sizes),
    }


def _rank_only_plausibility(text: str, rejection: str = "") -> bool:
    """Flag review prose that cites result position but no shared property.

    This is only an improvement signal. It never rejects a task automatically;
    the Coding Agent must inspect the entity and either state the real semantic
    overlap or replace the neighbor.
    """

    lowered = text.casefold()
    lowered_rejection = rejection.casefold()
    position_terms = ("rank", "result", "candidate list", "appears in")
    semantic_terms = (
        "same ",
        "matches ",
        "matching ",
        "similar ",
        "brand",
        "category",
        "object",
        "method",
        "material",
        "color",
        "size",
        "location",
        "branch",
        "topic",
        "technique",
        "function",
        "tool",
        "product",
        "video",
        "channel",
    )
    cites_position = any(term in lowered for term in position_terms)
    rejects_own_semantics = any(
        phrase in lowered_rejection
        for phrase in ("no semantic connection", "unrelated to")
    )
    if not cites_position:
        return False
    if rejects_own_semantics:
        return True
    if any(term in lowered for term in semantic_terms):
        return False

    # A semantic rationale may name the actual overlap directly (for example
    # "Morton salt", "stainless steel saucepan", or "wash day routine")
    # without using one of the generic labels above.  Remove positional filler
    # and flag the row only when no substantive description remains.
    positional_noise = {
        "a",
        "an",
        "and",
        "appeared",
        "appears",
        "as",
        "at",
        "before",
        "candidate",
        "candidates",
        "change",
        "final",
        "first",
        "from",
        "high",
        "highly",
        "in",
        "initial",
        "is",
        "it",
        "list",
        "listed",
        "listing",
        "of",
        "one",
        "post",
        "rank",
        "ranked",
        "ranking",
        "ranks",
        "result",
        "results",
        "retained",
        "second",
        "set",
        "suggesting",
        "that",
        "the",
        "this",
        "top",
        "was",
        "which",
        "with",
    }
    substantive_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", lowered)
        if token not in positional_noise and not token.isdigit()
    ]
    return len(substantive_tokens) < 2


def structural_difficulty(
    task: Mapping[str, Any],
    curve: DecisionCurve,
    source_record: Mapping[str, Any] | None = None,
) -> StructuralDifficulty:
    """Compile the difficulty shape from facts the evaluator already understands.

    A challenge axis is present only when the task has an executable contract for
    it.  We do not infer difficulty from prose length, image count, catalog size, or
    a model's confidence.
    """

    truth = Truth.of(task)
    task_id = str(task["id"])
    track = str(task["track"])
    final_states = truth.final_acceptable_states
    edit_profile = _native_edit_profile(truth)
    minimum_edits = edit_profile["minimum_total_edits"]
    maximum_edits = edit_profile["maximum_total_edits"]
    final_entity_count = max(
        len(state["selected_entity_ids"]) for state in final_states
    )
    exact_field_count = max(len(state["fields"]) for state in final_states)
    public_scope = {
        str(entity_id)
        for entity_id in task.get("truth", {}).get("site_scope_entity_ids", [])
    }
    public_non_answers = public_scope - truth.final_referent_ids
    construct = source_record.get("construct", {}) if source_record else {}
    resolution = (
        construct.get("catalog_resolution", {})
        if isinstance(construct, Mapping)
        else {}
    )
    core_neighbors = resolution.get("core_neighbor_entity_ids", [])
    query_rounds = resolution.get("query_rounds", [])
    detail_fields = resolution.get("detail_decision_fields", [])
    rejection_rows = resolution.get("rejected", [])
    failure_predicates = {
        str(predicate)
        for row in rejection_rows
        if isinstance(row, Mapping) and row.get("entity_id") in core_neighbors
        for predicate in row.get("failure_predicates", [])
        if isinstance(predicate, str) and predicate.strip()
    }
    rank_only_core_rows = [
        row
        for row in rejection_rows
        if isinstance(row, Mapping)
        and row.get("entity_id") in set(core_neighbors)
        and _rank_only_plausibility(
            str(row.get("why_plausible") or ""),
            str(row.get("reason") or ""),
        )
    ]
    real_window = (
        isinstance(query_rounds, list)
        and bool(query_rounds)
        and all(
            isinstance(row, Mapping)
            and row.get("query")
            and row.get("snapshot_path")
            and row.get("result_entity_ids")
            for row in query_rounds
        )
    )
    complete_neighborhood = isinstance(core_neighbors, list) and bool(core_neighbors)
    compositional_neighborhood = (
        complete_neighborhood
        and len(failure_predicates) >= COMPOSITIONAL_FAILURE_PREDICATES
    )
    detail_discrimination = isinstance(detail_fields, list) and bool(detail_fields)

    challenges = [_intent_challenge(track)]
    temporal_grounding = _requires_temporal_grounding(source_record)
    if curve.is_indexical or temporal_grounding:
        challenges.append("temporal_or_wearer_relative_grounding")
    if track != "clarify" and truth.invalidated_referent_ids:
        challenges.append("repair_or_remove_earlier_commitment")
    if final_entity_count > 1 or exact_field_count > 0:
        challenges.append("compound_native_web_commitment")
        challenges.append("compound_terminal_state")
    if minimum_edits is not None and minimum_edits >= 2:
        challenges.append("guaranteed_native_state_repair")
    elif maximum_edits is not None and maximum_edits >= 2:
        challenges.append("contingent_native_state_repair")
    if public_non_answers:
        challenges.append("target_discrimination")
    if compositional_neighborhood:
        challenges.append("compositional_target_discrimination")
    if real_window:
        challenges.append("full_real_result_window")
    if detail_discrimination:
        challenges.append("detail_page_discrimination")

    # Burdens are independent families. Search-neighborhood size and detail-page
    # inspection are one website burden, not two votes toward a higher band.
    # Clarification is the Clarify track's change-handling mechanism, not a
    # second independent burden. Counting both would mechanically inflate every
    # Clarify task by one band.
    burdens = ["clarification_protocol" if track == "clarify" else "change_handling"]
    if (
        (track == "refresh" and curve.settled_by_frames)
        or curve.is_indexical
        or temporal_grounding
    ):
        burdens.append("temporal_visual_grounding")
    if compositional_neighborhood and detail_discrimination:
        burdens.append("website_evidence_integration")
    if (
        minimum_edits is not None
        and minimum_edits >= 2
        or final_entity_count > 1
        or exact_field_count > 0
    ):
        burdens.append("native_state_work")

    risks: list[str] = []
    if truth.initial_acceptable_states is None:
        risks.append("missing_initial_goal")
    if not complete_neighborhood:
        risks.append("limited_candidate_neighborhood")
    if not real_window:
        risks.append("single_query_collapse_risk")
    if not detail_discrimination:
        risks.append("card_only_decision_risk")
    has_temporal_native_work = (
        (
            track == "graft"
            and truth.initial_goal_differs
            and truth.transition != "relax"
        )
        or (
            track == "refresh"
            and bool(truth.initial_referent_ids)
            and (
                truth.later_release_changes_referent
                or truth.later_release_should_preserve_referent
            )
        )
        or (track == "clarify" and compositional_neighborhood and detail_discrimination)
    )
    if (
        not has_temporal_native_work
        and final_entity_count == 1
        and exact_field_count == 0
    ):
        risks.append("shallow_native_commitment")

    opportunities: list[str] = []
    if final_entity_count == 1 and exact_field_count == 0:
        opportunities.append("single_entity_terminal_state")
    if track != "clarify" and minimum_edits is not None and minimum_edits < 2:
        opportunities.append("no_guaranteed_native_state_repair")
    if track == "graft" and not curve.settled_by_frames:
        opportunities.append("utterance_only_change")
    if rank_only_core_rows:
        opportunities.append("rank_only_core_plausibility")

    band, difficulty_review = reviewed_difficulty(
        construct if isinstance(construct, Mapping) else {},
    )
    if truth.initial_acceptable_states is None:
        band = "unclassified"
        difficulty_review = {**difficulty_review, "reason": "missing_initial_goal"}

    return StructuralDifficulty(
        task_id=task_id,
        band=band,
        challenges=tuple(challenges),
        burdens=tuple(burdens),
        risks=tuple(risks),
        opportunities=tuple(opportunities),
        evidence={
            "difficulty_review": difficulty_review,
            "candidate_roles": len(curve.candidates),
            "decision_rows": len(curve.observations),
            "settled_by": "frames" if curve.settled_by_frames else "utterance",
            "narrowed_by": curve.narrowing_cause,
            "indexical": curve.is_indexical,
            "temporal_source_relation": temporal_grounding,
            "real_query_rounds": len(query_rounds)
            if isinstance(query_rounds, list)
            else 0,
            "core_neighbor_count": len(core_neighbors)
            if isinstance(core_neighbors, list)
            else 0,
            "independent_failure_predicates": len(failure_predicates),
            "rank_only_core_rows": len(rank_only_core_rows),
            "detail_decision_fields": list(detail_fields)
            if isinstance(detail_fields, list)
            else [],
        },
        state_transition={
            **{
                label: len(ids) if ids is not None else None
                for label, ids in (
                    ("initial_entities", truth.initial_referent_ids),
                    ("invalidated_entities", truth.invalidated_referent_ids),
                    ("required_after_update", truth.required_after_update_ids),
                )
            },
            "final_entities": len(truth.final_referent_ids),
            "preserve_under_later_evidence": truth.later_release_should_preserve_referent,
            **edit_profile,
        },
        web_execution={
            "final_entity_count": final_entity_count,
            "exact_field_count": exact_field_count,
            "acceptable_final_states": len(final_states),
            "public_scope_entity_count": len(public_scope),
            "public_non_answer_count": len(public_non_answers),
        },
    )
