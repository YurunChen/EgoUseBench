"""Released-text predicates shared by the design gates and closure repair.

A listing "satisfies" an accepted listing's predicate when its searchable text
contains every released word that the accepted listing's own text confirms.
"""

from __future__ import annotations

from .final_state import goal_entity_ids

import importlib
import re
from collections.abc import Mapping
from typing import Any

STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "to",
        "and",
        "of",
        "in",
        "on",
        "my",
        "i",
        "me",
        "it",
        "that",
        "this",
        "for",
        "with",
        "add",
        "cart",
        "one",
        "is",
        "at",
        "find",
        "what",
        "used",
        "which",
        "by",
        "from",
        "as",
        "be",
        "use",
        "put",
        "get",
        "please",
        "pick",
        "select",
        "save",
        "keep",
        "also",
        "too",
        "into",
        "up",
        "matching",
        "match",
        "same",
        "like",
        "now",
        "then",
        "later",
        "first",
        "last",
        "counter",
        "table",
        "shelf",
        "kitchen",
        "hand",
        "hands",
        "holding",
        "hold",
        "just",
        "video",
        "tutorial",
        "guide",
        "show",
        "shows",
        "how",
        "step",
        "steps",
        "listing",
        "map",
        "place",
        "visited",
        "report",
        "record",
        "help",
        "need",
        "want",
        "separate",
        "both",
        "long",
        "minute",
        "minutes",
    ]
)


MIN_PREDICATE_WORDS = 2
_FRACTIONS = {"¼": ".25", "½": ".5", "¾": ".75"}
_UNITS = {
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "cm": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "mm": "mm",
    "inch": "in",
    "inches": "in",
    "in": "in",
    "ounce": "oz",
    "ounces": "oz",
    "oz": "oz",
    "gram": "g",
    "grams": "g",
    "g": "g",
    "kg": "kg",
    "pound": "lb",
    "pounds": "lb",
    "lb": "lb",
    "lbs": "lb",
    "ml": "ml",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "quart": "qt",
    "quarts": "qt",
    "qt": "qt",
    "cup": "cup",
    "cups": "cup",
}


def _stem(word: str) -> str:
    return word[:-1] if word.endswith("s") and len(word) > 3 else word


def _measure(value: float, unit: str) -> str:
    """One token per physical size, so 8 inch and 20 cm meet as 20cm."""

    if unit == "in":
        value, unit = round(value * 2.54), "cm"
    elif unit == "mm":
        value, unit = round(value / 10), "cm"
    elif unit == "cm":
        value = round(value)
    return f"{value:g}{unit}"


def predicate_tokens(text: str) -> frozenset[str]:
    text = text.casefold()
    for fraction, decimal in _FRACTIONS.items():
        text = text.replace(fraction, decimal)
    tokens = {
        _measure(float(number), _UNITS[unit])
        for number, unit in re.findall(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]+)", text)
        if unit in _UNITS
    }
    tokens.update(
        _stem(word)
        for word in re.findall(r"[a-z]+", text)
        if word not in STOPWORDS and word not in _UNITS and len(word) > 2
    )
    return frozenset(tokens)


def released_text(task: Mapping[str, Any]) -> str:
    instructions = task.get("instructions", {})
    keys = {"graft": ("a", "b"), "refresh": ("base",), "clarify": ("ambiguous",)}[
        task["track"]
    ]
    parts = [str(instructions.get(key, "")) for key in keys]
    if task["track"] == "clarify":
        parts.append(str((task.get("clarification") or {}).get("canonical_answer", "")))
    return " ".join(parts)


def accepted_ids(task: Mapping[str, Any]) -> list[str]:
    return sorted(goal_entity_ids(task["goal"]))


_CATALOG_TOKENS: dict[int, tuple[Mapping[str, Any], dict[str, frozenset[str]]]] = {}


def catalog_tokens(
    site: str, site_data: Mapping[str, Any]
) -> dict[str, frozenset[str]]:
    cached = _CATALOG_TOKENS.get(id(site_data))
    if cached is not None and cached[0] is site_data:
        return cached[1]
    tokens = {
        str(entity.get("entity_id")): predicate_tokens(entity_text(site, entity))
        for entity in site_data.get("entities", [])
    }
    _CATALOG_TOKENS[id(site_data)] = (site_data, tokens)
    return tokens


def target_predicates(
    task: Mapping[str, Any], catalog: Mapping[str, frozenset[str]]
) -> dict[str, list[str]]:
    """Released words that each accepted listing's own text confirms."""

    words = predicate_tokens(released_text(task))
    return {
        entity_id: sorted(words & catalog.get(entity_id, frozenset()))
        for entity_id in accepted_ids(task)
    }


def entity_text(site: str, entity: Mapping[str, Any]) -> str:
    """The text a search on this site can actually match, per the site's own index."""

    if site == "egomart":
        from sites.egomart import pages

        return pages.searchable_text(dict(entity))
    from sites.content import _searchable_text

    config = _site_config(site)
    return _searchable_text(dict(entity), config)


def _site_config(site: str) -> Any:
    return importlib.import_module(f"sites.{site}").CONFIG


def decided_ids(task: Mapping[str, Any], resolution: Mapping[str, Any]) -> set[str]:
    rejected = (
        str(row.get("entity_id"))
        for row in resolution.get("rejected") or []
        if isinstance(row, Mapping)
    )
    tempting = (task.get("truth") or {}).get("tempting_referent_ids", [])
    candidates = (task.get("clarification") or {}).get("candidate_ids", [])
    return {
        *accepted_ids(task),
        *rejected,
        *(str(item) for item in tempting),
        *(str(item) for item in candidates),
    }


def undecided_window_listings(
    task: Mapping[str, Any],
    resolution: Mapping[str, Any],
    catalog: Mapping[str, frozenset[str]],
) -> dict[str, list[str]]:
    decided = decided_ids(task, resolution)
    window = [str(entity_id) for entity_id in resolution.get("window_entity_ids") or []]
    undecided: dict[str, list[str]] = {}
    for words in target_predicates(task, catalog).values():
        if len(words) < MIN_PREDICATE_WORDS:
            continue
        required = set(words)
        for other in window:
            if other not in decided and required <= catalog.get(other, frozenset()):
                undecided.setdefault(other, words)
    return dict(sorted(undecided.items()))
