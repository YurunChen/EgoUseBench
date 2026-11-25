"""Block explicit answer fields and report advisory catalog fingerprints.

Each task is checked against its Agent-visible site projection. Explicit answer or
benchmark-role fields block that task; site and total counts sum those same facts.

Statistical contrasts inspect rendered metadata for referenced-vs-rest vocabulary,
value, presence and position patterns, and target-vs-neighbor presence differences.
They remain advisory: a heuristic gap can reflect natural source metadata and does
not establish an answer rule. A target's content differing from its neighbors is
also the discrimination the task is meant to require.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from egocua.environment import load_site_data, task_public_site_data
from egocua.run_ledger import task_sha256
from egocua.final_state import goal_entity_ids
from egocua.schema import referenced_entity_ids
from egocua.construction import load_construction_tasks
from egocua.synthesis_reports import REPORTS
from egocua.workspace import write_json_atomic
from sites.egomart.pages import brand_of

REPORT_PATH = REPORTS["role"].path

# Fields each site actually renders to the Agent. Kept next to the render code's
# contract: search cards, detail pages, and the search index read exactly these.
PUBLIC_FIELDS: dict[str, tuple[str, ...]] = {
    "egomart": ("name", "brand", "price", "attributes", "rating", "ratings_count"),
    "egomedia": (
        "title",
        "summary",
        "kind",
        "channel_title",
        "published_at",
        "duration_seconds",
        "view_count",
        "details",
    ),
    "egomaps": (
        "name",
        "summary",
        "kind",
        "address",
        "facts",
        "route_options",
        "rating",
        "ratings_count",
        "hours",
    ),
}
VISUAL_KEY = {
    "egomart": "image_path",
    "egomedia": "thumbnail_path",
    "egomaps": "photo_path",
}

# A token absent from the rest of the catalog only leaks role when it recurs across
# unrelated tasks; within one task it is just what that product is called. It must
# also span separate entities — one entity that two tasks share is not a pattern.
EXCLUSIVE_TOKEN_MIN_TASKS = 2
EXCLUSIVE_TOKEN_MIN_ENTITIES = 2
# Presence gaps above this threshold warrant inspection, not a leakage verdict.
FIELD_PRESENCE_MAX_GAP = 0.25
# A single value covering this much of the group while nearly absent outside it
# fingerprints the group; it must also span unrelated tasks.
VALUE_FINGERPRINT_MIN_SHARE = 0.30
VALUE_FINGERPRINT_MIN_TASKS = 2
VALUE_FINGERPRINT_MAX_BACKGROUND = 0.05
# Catalog position must not concentrate the group in any quarter of the catalog.
POSITION_MEDIAN_MARGIN = 0.25
FORBIDDEN_ROLE_FIELDS = frozenset(
    {
        "acceptable_states",
        "answer",
        "answer_role",
        "benchmark_role",
        "checker",
        "expected",
        "is_target",
        "task_id",
        "target_role",
    }
)


def _blocking_role_paths(value: Any, path: str = "$") -> list[str]:
    if isinstance(value, Mapping):
        findings: list[str] = []
        for key, item in value.items():
            field = str(key)
            child = f"{path}.{field}"
            if field.casefold() in FORBIDDEN_ROLE_FIELDS:
                findings.append(child)
            findings.extend(_blocking_role_paths(item, child))
        return findings
    if isinstance(value, list):
        return [
            finding
            for index, item in enumerate(value)
            for finding in _blocking_role_paths(item, f"{path}[{index}]")
        ]
    return []


def _text_of(entity: Mapping[str, Any], fields: Iterable[str]) -> str:
    parts: list[str] = []
    for field in fields:
        value = entity.get(field)
        if value is None:
            continue
        parts.append(
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )
    return " ".join(parts).casefold()


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z]{3,}", text)}


def _flat_values(
    entity: Mapping[str, Any], fields: Iterable[str]
) -> set[tuple[str, str]]:
    """Scalar (field, value) pairs, including one level into nested mappings."""

    pairs: set[tuple[str, str]] = set()
    for field in fields:
        value = entity.get(field)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            pairs.add((field, str(value)))
        elif isinstance(value, Mapping):
            for key, nested in value.items():
                if isinstance(nested, (str, int, float)) and not isinstance(
                    nested, bool
                ):
                    pairs.add((f"{field}.{key}", str(nested)))
    return pairs


def _cross_task_tokens(
    group: list[dict[str, Any]],
    rest: list[dict[str, Any]],
    fields: tuple[str, ...],
    tasks_of: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    """Tokens absent from the rest of the catalog that recur across unrelated tasks."""

    token_tasks: dict[str, set[str]] = {}
    token_entities: dict[str, set[str]] = {}
    for entity in group:
        entity_id = str(entity["entity_id"])
        owners = tasks_of.get(entity_id, set())
        for token in _tokens(_text_of(entity, fields)):
            token_tasks.setdefault(token, set()).update(owners)
            token_entities.setdefault(token, set()).add(entity_id)
    background: set[str] = set()
    for entity in rest:
        background |= _tokens(_text_of(entity, fields))
    return [
        {
            "token": token,
            "tasks": sorted(owners)[:6],
            "task_count": len(owners),
            "entity_count": len(token_entities[token]),
        }
        for token, owners in sorted(token_tasks.items(), key=lambda pair: -len(pair[1]))
        if len(owners) >= EXCLUSIVE_TOKEN_MIN_TASKS
        and len(token_entities[token]) >= EXCLUSIVE_TOKEN_MIN_ENTITIES
        and token not in background
    ][:10]


def _field_presence(
    group: list[dict[str, Any]],
    rest: list[dict[str, Any]],
    fields: tuple[str, ...],
    visual_key: str | None,
) -> list[dict[str, Any]]:
    checked = [*fields, *([visual_key] if visual_key else [])]
    findings = []
    for field in checked:
        in_group = sum(
            1 for entity in group if entity.get(field) not in (None, "", [], {})
        )
        in_rest = sum(
            1 for entity in rest if entity.get(field) not in (None, "", [], {})
        )
        group_rate = in_group / len(group) if group else 0.0
        rest_rate = in_rest / len(rest) if rest else 0.0
        if abs(group_rate - rest_rate) > FIELD_PRESENCE_MAX_GAP:
            findings.append(
                {
                    "field": field,
                    "present_in_group": round(group_rate, 3),
                    "present_in_rest": round(rest_rate, 3),
                }
            )
    return findings


def _value_fingerprints(
    group: list[dict[str, Any]],
    rest: list[dict[str, Any]],
    fields: tuple[str, ...],
    tasks_of: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    group_counts: Counter[tuple[str, str]] = Counter()
    value_tasks: dict[tuple[str, str], set[str]] = {}
    for entity in group:
        owners = tasks_of.get(str(entity["entity_id"]), set())
        for pair in _flat_values(entity, fields):
            group_counts[pair] += 1
            value_tasks.setdefault(pair, set()).update(owners)
    rest_counts: Counter[tuple[str, str]] = Counter()
    for entity in rest:
        rest_counts.update(_flat_values(entity, fields))
    findings = []
    for pair, count in group_counts.most_common():
        share = count / len(group) if group else 0.0
        background = rest_counts.get(pair, 0) / len(rest) if rest else 0.0
        if (
            share >= VALUE_FINGERPRINT_MIN_SHARE
            and background <= VALUE_FINGERPRINT_MAX_BACKGROUND
            and len(value_tasks.get(pair, set())) >= VALUE_FINGERPRINT_MIN_TASKS
        ):
            findings.append(
                {
                    "field": pair[0],
                    "value": pair[1][:80],
                    "share_in_group": round(share, 3),
                    "share_in_rest": round(background, 3),
                    "task_count": len(value_tasks[pair]),
                }
            )
    return findings[:10]


def _position(
    group_ids: set[str],
    ordered_ids: list[str],
) -> dict[str, Any] | None:
    index = {entity_id: position for position, entity_id in enumerate(ordered_ids)}
    positions = [index[entity_id] for entity_id in group_ids if entity_id in index]
    if not positions or len(ordered_ids) < 8:
        return None
    fraction = statistics.median(positions) / len(ordered_ids)
    if POSITION_MEDIAN_MARGIN <= fraction <= 1 - POSITION_MEDIAN_MARGIN:
        return None
    return {
        "median_position_fraction": round(fraction, 3),
        "acceptable_band": [POSITION_MEDIAN_MARGIN, 1 - POSITION_MEDIAN_MARGIN],
    }


def _contrast(
    label: str,
    group: list[dict[str, Any]],
    rest: list[dict[str, Any]],
    fields: tuple[str, ...],
    visual_key: str | None,
    ordered_ids: list[str] | None,
    tasks_of: Mapping[str, set[str]],
    *,
    content_channels: bool = True,
) -> dict[str, Any]:
    if not group or not rest:
        return {"contrast": label, "skipped": "one side is empty"}
    findings: dict[str, Any] = {
        "contrast": label,
        "group_size": len(group),
        "rest_size": len(rest),
        "exclusive_tokens": (
            _cross_task_tokens(group, rest, fields, tasks_of)
            if content_channels
            else []
        ),
        "field_presence": _field_presence(group, rest, fields, visual_key),
        "value_fingerprints": (
            _value_fingerprints(group, rest, fields, tasks_of)
            if content_channels
            else []
        ),
    }
    if ordered_ids is not None:
        position = _position(
            {str(entity["entity_id"]) for entity in group}, ordered_ids
        )
        if position:
            findings["position"] = position
    findings["advisory_findings"] = sum(
        len(findings[key]) if isinstance(findings.get(key), list) else 1
        for key in (
            "exclusive_tokens",
            "field_presence",
            "value_fingerprints",
            "position",
        )
        if findings.get(key)
    )
    return findings


def audit(root: Path, *, task_ids: list[str] | None = None) -> dict[str, Any]:
    tasks = load_construction_tasks(root, task_ids)
    if task_ids is not None:
        requested = set(task_ids)
        tasks = [task for task in tasks if task["id"] in requested]
    report: dict[str, Any] = {
        "schema_version": REPORTS["role"].schema_version,
        "sites": {},
        "tasks": {},
    }
    for site in ("egomart", "egomedia", "egomaps"):
        site_data = load_site_data(root, site)
        entities = site_data["entities"]
        if site == "egomart":
            entities = [{**entity, "brand": brand_of(entity)} for entity in entities]
        by_id = {str(entity["entity_id"]): entity for entity in entities}
        ordered_ids = [str(entity["entity_id"]) for entity in entities]
        fields = PUBLIC_FIELDS[site]
        visual_key = VISUAL_KEY[site]

        referenced: set[str] = set()
        targets: set[str] = set()
        neighbors: set[str] = set()
        tasks_of: dict[str, set[str]] = {}
        for task in tasks:
            if task["site"] != site:
                continue
            for entity_id in referenced_entity_ids(task):
                tasks_of.setdefault(entity_id, set()).add(task["id"])
            referenced.update(referenced_entity_ids(task))
            targets.update(goal_entity_ids(task["goal"]))
        neighbors = referenced - targets
        if not referenced:
            continue

        contrasts = [
            _contrast(
                "referenced_vs_rest",
                [by_id[i] for i in sorted(referenced) if i in by_id],
                [
                    entity
                    for entity in entities
                    if str(entity["entity_id"]) not in referenced
                ],
                fields,
                visual_key,
                ordered_ids,
                tasks_of,
            ),
            _contrast(
                "target_vs_neighbors",
                [by_id[i] for i in sorted(targets) if i in by_id],
                [by_id[i] for i in sorted(neighbors) if i in by_id],
                fields,
                visual_key,
                None,
                tasks_of,
                content_channels=False,
            ),
        ]
        advisory_fingerprints = sum(
            contrast.get("advisory_findings", 0) for contrast in contrasts
        )
        blocking_leaks = 0
        for task in (task for task in tasks if task["site"] == site):
            task_id = str(task["id"])
            public_site = task_public_site_data(task, site_data)
            blocking_paths = _blocking_role_paths(public_site)
            scoped_ids = set(referenced_entity_ids(task))
            target_ids = set(goal_entity_ids(task["goal"]))
            neighbor_ids = scoped_ids - target_ids
            task_contrast = _contrast(
                "target_vs_neighbors",
                [
                    by_id[entity_id]
                    for entity_id in sorted(target_ids)
                    if entity_id in by_id
                ],
                [
                    by_id[entity_id]
                    for entity_id in sorted(neighbor_ids)
                    if entity_id in by_id
                ],
                fields,
                visual_key,
                None,
                {entity_id: {task_id} for entity_id in scoped_ids},
                content_channels=False,
            )
            report["tasks"][task_id] = {
                "task_sha256": task_sha256(task),
                "status": "pass" if not blocking_paths else "fail",
                "blocking_leaks": len(blocking_paths),
                "blocking_paths": blocking_paths,
                "advisory_contrast": task_contrast,
            }
            blocking_leaks += len(blocking_paths)
        report["sites"][site] = {
            "catalog_size": len(entities),
            "referenced": len(referenced),
            "contrasts": contrasts,
            "blocking_leaks": blocking_leaks,
            "advisory_selection_fingerprints": advisory_fingerprints,
            "leaks": blocking_leaks,
        }
    report["total_leaks"] = sum(site["leaks"] for site in report["sites"].values())
    report["advisory_selection_fingerprints"] = sum(
        site["advisory_selection_fingerprints"] for site in report["sites"].values()
    )
    report["interpretation"] = (
        "Tasks block only explicit answer or benchmark-role fields in the Agent-visible "
        "site; site and total counts sum those same task findings. "
        "Statistical target/neighbor and site-level contrasts remain advisory because "
        "natural public metadata differences do not prove answer leakage."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=REPORT_PATH.as_posix())
    args = parser.parse_args()
    root = Path(args.root).resolve()
    report = audit(root)
    write_json_atomic(root / args.output, report)
    print(
        json.dumps(
            {
                "total_leaks": report["total_leaks"],
                "advisory_selection_fingerprints": report[
                    "advisory_selection_fingerprints"
                ],
                "by_site": {
                    site: data["leaks"] for site, data in report["sites"].items()
                },
                "output": args.output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
