"""Exact website outcomes, including reviewed retail-package substitutions."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any, Mapping


EntityCounts = Counter[tuple[str, str]]


def _valid_fields(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    )


def _valid_entity_ids(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(item, str) and item for item in value
    )


def unit_equivalences_of(goal: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """Read explicit, disjoint positive package counts; never infer them from names."""
    groups = goal.get("unit_equivalences", {})
    if not isinstance(groups, dict):
        raise ValueError("unit_equivalences must map unit classes to package counts")
    seen: set[str] = set()
    for unit, packages in groups.items():
        if not isinstance(unit, str) or not unit or not isinstance(packages, dict) or not packages:
            raise ValueError("unit classes must have a name and at least one package")
        for entity_id, quantity in packages.items():
            if not isinstance(entity_id, str) or not entity_id or entity_id in seen:
                raise ValueError("a package must belong to exactly one unit class")
            if type(quantity) is not int or quantity <= 0:
                raise ValueError("package unit counts must be positive integers")
            seen.add(entity_id)
    return groups


def normalized_entity_counts(
    goal: Mapping[str, Any], entity_ids: list[str]
) -> EntityCounts:
    """Keep unlisted entities literal so extra products cannot disappear."""
    packages = {
        entity_id: (unit, quantity)
        for unit, members in unit_equivalences_of(goal).items()
        for entity_id, quantity in members.items()
    }
    counts: EntityCounts = Counter()
    for entity_id in entity_ids:
        if entity_id in packages:
            unit, quantity = packages[entity_id]
            counts[("unit", unit)] += quantity
        else:
            counts[("entity", entity_id)] += 1
    return counts


@lru_cache(maxsize=512)
def _reachable_totals(quantities: tuple[int, ...], total: int) -> tuple[bool, ...]:
    reachable = [True] + [False] * total
    for amount in range(1, total + 1):
        reachable[amount] = any(
            quantity <= amount and reachable[amount - quantity]
            for quantity in quantities
        )
    return tuple(reachable)


def _state_entity_ids(goal: Mapping[str, Any], state: Mapping[str, Any]) -> set[str]:
    counts = normalized_entity_counts(goal, state["selected_entity_ids"])
    entities = {name for (kind, name), count in counts.items() if kind == "entity" and count}
    for unit, packages in unit_equivalences_of(goal).items():
        total = counts[("unit", unit)]
        reachable = _reachable_totals(tuple(sorted(set(packages.values()))), total)
        entities.update(
            entity_id for entity_id, quantity in packages.items()
            if quantity <= total and reachable[total - quantity]
        )
    return entities


def goal_entity_ids(goal: Mapping[str, Any]) -> frozenset[str]:
    """All SKUs that can occur in an exact legal cart, not just its witnesses."""
    return frozenset(
        entity_id
        for state in goal["acceptable_states"]
        for entity_id in _state_entity_ids(goal, state)
    )


def goal_signature(goal: Mapping[str, Any]) -> frozenset[tuple]:
    """Compare outcomes and their reachable package choices without enumerating carts."""
    return frozenset(
        (
            tuple(sorted(normalized_entity_counts(goal, state["selected_entity_ids"]).items())),
            tuple(sorted(state["fields"].items())),
            tuple(sorted(
                (unit, entity_id, quantity)
                for unit, packages in unit_equivalences_of(goal).items()
                for entity_id, quantity in packages.items()
                if entity_id in _state_entity_ids(goal, state)
            )),
        )
        for state in goal["acceptable_states"]
    )


def evaluate_final_state(goal: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """Compare complete quantities and fields; packaging and order can differ."""
    selected = state.get("selected_entity_ids")
    fields = state.get("fields")
    if not _valid_entity_ids(selected) or not _valid_fields(fields):
        return False
    acceptable_states = goal.get("acceptable_states")
    if not isinstance(acceptable_states, list) or not acceptable_states:
        return False
    actual = normalized_entity_counts(goal, selected)
    for acceptable in acceptable_states:
        if not isinstance(acceptable, dict):
            continue
        expected_ids = acceptable.get("selected_entity_ids")
        expected_fields = acceptable.get("fields")
        if not _valid_entity_ids(expected_ids) or not _valid_fields(expected_fields):
            continue
        if actual == normalized_entity_counts(goal, expected_ids) and fields == expected_fields:
            return True
    return False
