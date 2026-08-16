"""Deterministic evaluation for the normalized cross-site final state."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping


def _valid_fields(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    )


def _valid_entity_ids(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item for item in value)
    )


def evaluate_final_state(goal: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Return whether a submitted state exactly matches one acceptable state.

    Entity order is irrelevant, but duplicates, extra entities, extra fields, missing
    fields, and unsubmitted states are rejected.
    """

    if state.get("submission_completed") is not True:
        return False
    selected = state.get("selected_entity_ids")
    fields = state.get("fields")
    if not _valid_entity_ids(selected) or not _valid_fields(fields):
        return False

    acceptable_states = goal.get("acceptable_states")
    if not isinstance(acceptable_states, list) or not acceptable_states:
        return False
    for acceptable in acceptable_states:
        if not isinstance(acceptable, dict):
            continue
        expected_ids = acceptable.get("selected_entity_ids")
        expected_fields = acceptable.get("fields")
        if not _valid_entity_ids(expected_ids) or not _valid_fields(expected_fields):
            continue
        if Counter(selected) == Counter(expected_ids) and fields == expected_fields:
            return True
    return False
