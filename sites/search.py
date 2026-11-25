"""Shared deterministic relevance search for every frozen mock site."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeVar

T = TypeVar("T")
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(value: str) -> list[str]:
    """Normalize public text without task-specific aliases."""

    return TOKEN.findall(value.casefold())


def flatten_public_text(value: Any) -> str:
    """Flatten one declared public field into searchable text."""

    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Mapping):
        return " ".join(flatten_public_text(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(flatten_public_text(item) for item in value)
    return ""


def _variants(token: str) -> tuple[str, ...]:
    variants = [token]
    if len(token) > 3 and token.endswith("s"):
        variants.append(token[:-1])
    if len(token) > 4 and token.endswith("ies"):
        variants.append(token[:-3] + "y")
    return tuple(dict.fromkeys(variants))


def _term_frequency(tokens: Mapping[str, float], term: str) -> float:
    variants = _variants(term)
    exact = sum(tokens.get(variant, 0) for variant in variants)
    if exact:
        return float(exact)
    if len(term) < 4:
        return 0.0
    return 0.35 * sum(
        count
        for token, count in tokens.items()
        if token.startswith(term) or term.startswith(token)
    )


def rank_records(
    records: Iterable[T],
    query: str,
    fields: Callable[[T], Iterable[tuple[str, float]]],
) -> list[T]:
    """Rank partial matches with BM25 over shared public fields.

    The function never receives a task ID or answer role. Stable input order is
    the only tie-breaker, so the same catalog and query always produce the same
    result window.
    """

    query_tokens = tokenize(query)
    if not query_tokens:
        return list(records)
    prepared: list[tuple[int, T, dict[str, float], float, list[str]]] = []
    for index, record in enumerate(records):
        counts: dict[str, float] = defaultdict(float)
        weighted_length = 0.0
        normalized_fields: list[str] = []
        for value, weight in fields(record):
            field_tokens = tokenize(value)
            if not field_tokens:
                continue
            normalized_fields.append(" ".join(field_tokens))
            for token in field_tokens:
                counts[token] += weight
            weighted_length += len(field_tokens) * weight
        prepared.append(
            (index, record, counts, max(weighted_length, 1.0), normalized_fields)
        )
    if not prepared:
        return []
    average_length = sum(row[3] for row in prepared) / len(prepared)
    document_frequency = {
        term: sum(_term_frequency(row[2], term) > 0 for row in prepared)
        for term in dict.fromkeys(query_tokens)
    }
    phrase = " ".join(query_tokens)
    scored: list[tuple[float, int, T]] = []
    k1 = 1.5
    b = 0.72
    for index, record, counts, length, public_fields in prepared:
        score = 0.0
        matched_terms = 0
        for term in query_tokens:
            frequency = _term_frequency(counts, term)
            if frequency <= 0:
                continue
            matched_terms += 1
            frequency_docs = document_frequency[term]
            inverse_frequency = math.log(
                1 + (len(prepared) - frequency_docs + 0.5) / (frequency_docs + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * length / average_length)
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        if matched_terms == 0:
            continue
        score += 0.45 * matched_terms / len(query_tokens)
        if phrase and any(phrase in field for field in public_fields):
            score += 2.0
        scored.append((score, index, record))
    return [
        record for _, _, record in sorted(scored, key=lambda row: (-row[0], row[1]))
    ]
