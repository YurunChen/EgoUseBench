"""Dump the evidence needed to hand-author non-leaking task instructions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from egocua.final_state import goal_entity_ids
from egocua.sources import iter_records

WEB_FILTER_KEYS = {
    "price",
    "rating",
    "ratings_count",
    "view_count",
    "duration_minutes",
    "duration_seconds",
    "published_at",
}


def _screening_records(root: Path) -> dict[str, dict[str, Any]]:
    return {str(record["task_id"]): record for _, record in iter_records(root)}


def _entity_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path, key in (
        ("sites/egomart/products.json", "products"),
        ("sites/egomedia/videos.json", "videos"),
        ("sites/egomaps/places.json", "places"),
    ):
        for entity in json.loads((root / path).read_text())[key]:
            entity_id = (
                entity.get("item_id")
                or entity.get("video_id")
                or entity.get("place_id")
            )
            index[str(entity_id)] = entity
    return index


def _facets(entity: dict[str, Any]) -> dict[str, Any]:
    facets: dict[str, Any] = {}
    for key in ("attributes", "details", "facts", "route_options"):
        value = entity.get(key)
        if isinstance(value, dict):
            facets.update(value)
    for key in ("brand", "kind", "channel_title", "price", "rating"):
        if key in entity:
            facets[key] = entity[key]
    return facets


def _roles(task: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return (target entity IDs, rival entity IDs) for one task."""

    final = set(goal_entity_ids(task["goal"]))
    if task["track"] == "graft":
        initial = set(goal_entity_ids(task["graft"]["initial_goal"]))
        return final, initial - final
    if task["track"] == "clarify":
        return final, set(task["clarification"]["candidate_ids"]) - final
    return final, {str(task["refresh"]["initial_entity_id"])} - final


def build(root: Path) -> list[dict[str, Any]]:
    records = _screening_records(root)
    entities = _entity_index(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("data/tasks/*/*.json")):
        task = json.loads(path.read_text())
        record = records.get(task["id"], {})
        targets, rivals = _roles(task)
        target_facets = _facets(entities[sorted(targets)[0]]) if targets else {}
        rival_facets = _facets(entities[sorted(rivals)[0]]) if rivals else {}
        differing = {
            key: [target_facets.get(key), rival_facets.get(key)]
            for key in sorted(set(target_facets) | set(rival_facets))
            if str(target_facets.get(key)) != str(rival_facets.get(key))
        }
        row = {
            "id": task["id"],
            "track": task["track"],
            "site": task["site"],
            "subtype": record.get("subtype"),
            "referent_hint": (record.get("screening") or {}).get(
                "proposed_visible_referent"
            ),
            "frame_claim": (record.get("supported_claims") or [{}])[0].get("text"),
            "current": task["instructions"],
            "target_titles": [
                entities[i].get("name") or entities[i].get("title")
                for i in sorted(targets)
            ],
            "rival_titles": [
                entities[i].get("name") or entities[i].get("title")
                for i in sorted(rivals)
            ],
            "discriminators": {
                key: value
                for key, value in differing.items()
                if key not in WEB_FILTER_KEYS
            },
            "web_filter_only": {
                key: value for key, value in differing.items() if key in WEB_FILTER_KEYS
            },
        }
        if task["track"] == "clarify":
            row["valid_slot"] = task["clarification"]["valid_slot"]
            row["canonical_answer"] = task["clarification"]["canonical_answer"]
        if task["track"] == "refresh":
            row["policy"] = task["refresh"]["policy"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--out", default="artifacts/reports/instruction-evidence.json")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    rows = build(root)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
