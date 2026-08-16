"""Mine Ego2Web metadata for naturally occurring goal-divergence evidence.

Counts, per divergence source, how often the raw egocentric recordings contain
the events that EgoCUA-Bench tracks simulate:

- referent switches: one video, several instruction segments at different times
  (Refresh material);
- in-scene ambiguity: instructions that need a disambiguating attribute to pin
  down one object among visible candidates (Clarify material);
- bundled intents: one instruction already grounding several objects in the
  same scene (Graft material).

Outputs a JSON report plus a candidate list of same-video segment pairs for
manual visual verification.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

from egocua.workspace import resolve_within, write_json_atomic

TIME_PATTERN = re.compile(r"\d+(?::\d+){1,2}")

QUALIFIER_FAMILIES = {
    "color": re.compile(
        r"\b(white|black|red|blue|green|pink|yellow|purple|gray|grey|brown|orange|silver|gold|beige|navy)\b",
        re.IGNORECASE,
    ),
    "position": re.compile(
        r"\b(on the (table|left|right|shelf|counter|desk|floor)|next to|near the|behind the"
        r"|in front of|in the (middle|corner|center)|closest|top|bottom)\b",
        re.IGNORECASE,
    ),
    "sequence": re.compile(r"\b(first|second|third|last|finally|then picked|ends? up)\b", re.IGNORECASE),
    "variant": re.compile(r"\b(version|model|size|flavor|edition|brand|type)\b", re.IGNORECASE),
    "video_deixis": re.compile(
        r"\b(shown in the video|in the video|tried by|held by|worn by|the same|that's on|being (held|used))\b",
        re.IGNORECASE,
    ),
}
BUNDLE_PATTERN = re.compile(
    r"\b(add both|both items|add them (both )?to|and (two|three|a set of) [a-z]+"
    r"|(two|three|\d+) (cans|bottles|packs|boxes|sets|pairs) of)\b",
    re.IGNORECASE,
)
SEQUENTIAL_PICKUP_PATTERN = re.compile(
    r"\b(first|second|third|fourth|fifth|sixth|seventh|last)\b(?:\W+\w+){0,4}\W+"
    r"(pick(s|ed)? up|takes|taken|grab(s|bed)?|h(o|e)lds?)\b"
    r"|\b(pick(s|ed)? up|takes|grab(s|bed)?)\b(?:\W+\w+){0,4}\W+"
    r"\b(first|second|third|fourth|fifth|sixth|seventh|last)\b",
    re.IGNORECASE,
)
PHYSICAL_SELECTION_PATTERN = re.compile(
    r"\b(pick(?:s|ed)? up|grab(?:s|bed)?|take(?:s|n)? (?:from|off)|"
    r"touch(?:es|ed)?|tried by|physically picks|is holding)\b",
    re.IGNORECASE,
)
EXPLICIT_REPLACEMENT_PATTERN = re.compile(
    r"\b(put(?:s)? (?:it|the item|the product) back|return(?:s|ed)? (?:it|the item|the product) "
    r"(?:to the shelf|before)|switch(?:es|ed)? from .+ to|change(?:s|d)? (?:his|her|their|my) mind "
    r"(?:and|to)|instead (?:pick(?:s|ed)?|choose(?:s)?|chose|select(?:s|ed)?|take(?:s)?))\b",
    re.IGNORECASE,
)
CONSTRAINT_PATTERNS = {
    "rating": re.compile(r"\b(rating|rated|stars?)\b", re.IGNORECASE),
    "prime": re.compile(r"\bprime(?:-eligible)?\b", re.IGNORECASE),
    "price_cap": re.compile(
        r"\b(under|below|less than|no more than|budget)\b.{0,16}(\$|dollars?)",
        re.IGNORECASE,
    ),
    "availability": re.compile(
        r"\b(in stock|available|availability|shipping|delivery)\b",
        re.IGNORECASE,
    ),
}
RELATIONAL_PATTERN = re.compile(
    r"\b(compatible|compatibility|fits?|refill|replacement|accessor(?:y|ies)|"
    r"case for|filter for|spool|cartridge|adapter|battery for|works? with)\b",
    re.IGNORECASE,
)
PARALLEL_PATTERN = re.compile(
    r"\b(add both|add them (?:both )?to (?:the )?(?:cart|shopping cart)|"
    r"open two product pages|find two matching product pages|"
    r"product pages for \(a\).+\(b\)|both items)\b",
    re.IGNORECASE,
)
CLARIFY_POSITION_PATTERN = re.compile(
    r"\b(left|right|next to|beside|adjacent|closest|behind|in front of|"
    r"top left|top right|bottom left|bottom right)\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "the", "a", "an", "on", "in", "of", "and", "to", "with", "for", "same", "video", "shown",
    "find", "add", "cart", "price", "report", "compare", "search", "item", "items", "its",
    "that", "this", "their", "total", "online", "store", "any", "new", "final", "one", "two",
}


def _seconds(value: str) -> float:
    total = 0.0
    for part in value.split(":"):
        total = total * 60 + float(part)
    return total


def _ranges(clues: Any) -> list[tuple[float, float]]:
    texts = clues if isinstance(clues, list) else [clues]
    ranges = []
    for text in texts:
        stamps = [_seconds(stamp) for stamp in TIME_PATTERN.findall(str(text))]
        for start, end in zip(stamps[::2], stamps[1::2]):
            ranges.append((min(start, end), max(start, end)))
    return ranges


def _content_tokens(instruction: str) -> set[str]:
    tokens = re.findall(r"[a-z]+", instruction.casefold())
    return {token for token in tokens if len(token) > 2 and token not in STOPWORDS}


def _qualifier_hits(instruction: str) -> dict[str, bool]:
    return {family: bool(pattern.search(instruction)) for family, pattern in QUALIFIER_FAMILIES.items()}


def _row_candidate(row: dict[str, Any], signals: list[str]) -> dict[str, Any]:
    return {
        "annotation_id": row["id"],
        "media_id": row["vid"],
        "domain": row["domain"],
        "ranges": _ranges(row["visual_clues"]),
        "signals": signals,
        "instruction": row["instruction"],
    }


def _unique_video_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique = []
    for candidate in candidates:
        media_id = str(candidate["media_id"])
        if media_id in seen:
            continue
        seen.add(media_id)
        unique.append(candidate)
    return unique


def mine(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_video.setdefault(row["vid"], []).append(row)

    pair_candidates: list[dict[str, Any]] = []
    for vid, video_rows in by_video.items():
        for left, right in combinations(video_rows, 2):
            left_ranges, right_ranges = _ranges(left["visual_clues"]), _ranges(right["visual_clues"])
            if not left_ranges or not right_ranges:
                continue
            if left_ranges[0][0] > right_ranges[0][0]:
                left, right, left_ranges, right_ranges = right, left, right_ranges, left_ranges
            gap = right_ranges[0][0] - left_ranges[0][1]
            shared = _content_tokens(left["instruction"]) & _content_tokens(right["instruction"])
            pair_candidates.append({
                "vid": vid,
                "ids": [left["id"], right["id"]],
                "domains": [left["domain"], right["domain"]],
                "ranges": [left_ranges[0], right_ranges[0]],
                "gap_seconds": round(gap, 1),
                "disjoint": gap > 0,
                "shared_tokens": sorted(shared),
                "instructions": [left["instruction"], right["instruction"]],
            })

    qualifier_rows = [dict(_qualifier_hits(row["instruction"]), id=row["id"]) for row in rows]
    family_counts = {
        family: sum(1 for hits in qualifier_rows if hits[family]) for family in QUALIFIER_FAMILIES
    }
    disambiguating = [
        hits["id"] for hits in qualifier_rows
        if hits["video_deixis"] and (hits["color"] or hits["position"] or hits["variant"] or hits["sequence"])
    ]
    bundled = [row["id"] for row in rows if BUNDLE_PATTERN.search(row["instruction"])]
    sequential = [row["id"] for row in rows if SEQUENTIAL_PICKUP_PATTERN.search(row["instruction"])]
    sequential_videos = sorted({row["vid"] for row in rows if row["id"] in set(sequential)})
    physical_selection = [
        row["id"] for row in rows if PHYSICAL_SELECTION_PATTERN.search(row["instruction"])
    ]
    explicit_replacement = [
        row["id"] for row in rows if EXPLICIT_REPLACEMENT_PATTERN.search(row["instruction"])
    ]

    ecommerce_rows = [row for row in rows if row.get("domain") == "E-Commerce"]
    constraint_candidates = _unique_video_rows([
        _row_candidate(
            row,
            [name for name, pattern in CONSTRAINT_PATTERNS.items() if pattern.search(row["instruction"])],
        )
        for row in ecommerce_rows
        if any(pattern.search(row["instruction"]) for pattern in CONSTRAINT_PATTERNS.values())
        and QUALIFIER_FAMILIES["video_deixis"].search(row["instruction"])
    ])
    relational_candidates = _unique_video_rows([
        _row_candidate(row, ["compatibility_relation"])
        for row in ecommerce_rows
        if RELATIONAL_PATTERN.search(row["instruction"])
        and QUALIFIER_FAMILIES["video_deixis"].search(row["instruction"])
    ])
    parallel_bundled_candidates = _unique_video_rows([
        _row_candidate(row, ["two_final_targets_in_one_annotation"])
        for row in ecommerce_rows
        if PARALLEL_PATTERN.search(row["instruction"])
    ])
    parallel_pair_candidates = [
        pair for pair in pair_candidates
        if pair["disjoint"] and pair["domains"] == ["E-Commerce", "E-Commerce"]
    ]
    clarify_candidates = {
        family: _unique_video_rows([
            _row_candidate(row, [family])
            for row in ecommerce_rows
            if (
                CLARIFY_POSITION_PATTERN.search(row["instruction"])
                if family == "position"
                else QUALIFIER_FAMILIES[family].search(row["instruction"])
            )
            and QUALIFIER_FAMILIES["video_deixis"].search(row["instruction"])
        ])
        for family in ("variant", "sequence", "position")
    }

    rows_per_video = Counter(len(video_rows) for video_rows in by_video.values())
    disjoint_pairs = [pair for pair in pair_candidates if pair["disjoint"]]
    related_pairs = [pair for pair in disjoint_pairs if pair["shared_tokens"]]
    return {
        "rows": len(rows),
        "videos": len(by_video),
        "rows_per_video": dict(sorted(rows_per_video.items())),
        "refresh": {
            "multi_segment_videos": sum(1 for video_rows in by_video.values() if len(video_rows) >= 2),
            "segment_pairs": len(pair_candidates),
            "disjoint_pairs": len(disjoint_pairs),
            "disjoint_related_pairs": len(related_pairs),
            "median_gap_seconds": sorted(
                float(pair["gap_seconds"]) for pair in disjoint_pairs
            )[len(disjoint_pairs) // 2]
            if disjoint_pairs else None,
            "sequential_pickup_rows": len(sequential),
            "sequential_pickup_ids": sequential,
            "sequential_pickup_videos": len(sequential_videos),
            "physical_selection_rows": len(physical_selection),
            "physical_selection_ids": physical_selection,
            "explicit_replacement_rows": len(explicit_replacement),
            "explicit_replacement_ids": explicit_replacement,
        },
        "clarify": {
            "qualifier_family_counts": family_counts,
            "deixis_plus_qualifier_rows": len(disambiguating),
            "deixis_plus_qualifier_ids": disambiguating,
        },
        "graft": {"bundled_instruction_rows": len(bundled), "bundled_ids": bundled},
        "subtype_candidate_pool": {
            "status": "high_recall_metadata_candidates_not_video_screened_tasks",
            "admission_warning": (
                "Regex or same-video metadata membership never establishes construct validity. "
                "Every candidate still needs continuous-video review, exact released-observation "
                "review, independent source-cluster checks, and the registered construction gates."
            ),
            "graft": {
                "C": constraint_candidates,
                "P_bundled": parallel_bundled_candidates,
                "P_same_video_pairs": parallel_pair_candidates,
                "R": relational_candidates,
            },
            "clarify": clarify_candidates,
        },
        "pair_candidates": sorted(disjoint_pairs, key=lambda pair: -len(pair["shared_tokens"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="reference/Ego2Web/dataset/ego2web.json")
    parser.add_argument("--output", default="artifacts/reports/mining/divergence_mining.json")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    dataset_path = resolve_within(root, args.dataset, context="Ego2Web annotation dataset")
    output_path = resolve_within(
        root, args.output, context="divergence mining output", must_exist=False
    )
    rows = json.loads(dataset_path.read_text())
    report = mine(rows)
    write_json_atomic(output_path, report)
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"pair_candidates", "subtype_candidate_pool"}
    }
    pool = report["subtype_candidate_pool"]
    summary["subtype_candidate_counts"] = {
        "graft": {key: len(value) for key, value in pool["graft"].items()},
        "clarify": {key: len(value) for key, value in pool["clarify"].items()},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
