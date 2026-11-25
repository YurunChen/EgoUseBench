"""Freeze deterministic search windows from the licensed local ABO catalog."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from egocua.workspace import write_json_atomic
from sites.search import flatten_public_text, rank_records
from tools.catalog.search_abo_products import (
    ABO_BASE,
    _image_paths,
    _listings,
    _preferred_name,
    _values,
)


OUTPUT = Path("data/sources/upstream_search_snapshots/abo")
QUERY_REVIEWS = Path("data/sources/search_query_reviews.json")
LICENSE = "Creative Commons Attribution 4.0 International"
ATTRIBUTION = (
    "Data and images: Amazon.com. Dataset and benchmark sets: Matthieu Guillaumin, "
    "Thomas Dideriksen, Kenan Deng, Himanshu Arora, Jasmine Collins, and Jitendra Malik."
)


def _records(metadata_root: Path, *, country: str | None) -> list[dict[str, Any]]:
    paths = _image_paths(metadata_root)
    records: list[dict[str, Any]] = []
    for record in _listings(metadata_root):
        if country and record.get("country") != country:
            continue
        asin = str(record.get("item_id") or "").strip()
        title = _preferred_name(record.get("item_name")).strip()
        image_path = paths.get(str(record.get("main_image_id") or ""))
        if not asin or not title or not image_path:
            continue
        domain = str(record.get("domain_name") or "amazon.com").strip()
        records.append(
            {
                "asin": asin,
                "title": title,
                "brand": _preferred_name(record.get("brand")).strip(),
                "product_types": _values(record.get("product_type")),
                "keywords": _values(record.get("item_keywords")),
                "color": _preferred_name(record.get("color")).strip(),
                "country": str(record.get("country") or "").strip(),
                "product_page_url": f"https://www.{domain}/dp/{asin}",
                "dataset_source_url": f"{ABO_BASE}/index.html",
                "image_url": f"{ABO_BASE}/images/original/{image_path}",
            }
        )
    return records


def _rank(
    records: list[dict[str, Any]], query: str, *, limit: int
) -> list[dict[str, Any]]:
    ranked = rank_records(
        records,
        query,
        lambda row: (
            (str(row["title"]), 3.0),
            (str(row["brand"]), 1.5),
            (flatten_public_text(row["product_types"]), 1.5),
            (str(row["color"]), 1.25),
            (flatten_public_text(row["keywords"]), 0.5),
        ),
    )
    return ranked[:limit]


def capture(
    root: Path,
    *,
    task_id: str,
    phase: str,
    query: str,
    country: str | None,
    limit: int,
    captured_at: str | None = None,
) -> dict[str, Any]:
    metadata_root = root / "reference/catalogs/amazon-berkeley-objects"
    records = _records(metadata_root, country=country)
    ranked = _rank(records, query, limit=limit)
    if not ranked:
        raise ValueError(f"ABO query returned no records: {query}")
    timestamp = captured_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {
        "schema_version": 1,
        "task_id": task_id,
        "phase": phase,
        "query": query,
        "captured_at": timestamp,
        "provider": "amazon_berkeley_objects",
        "retrieval": {
            "method": "deterministic_bm25_over_local_metadata",
            "country": country,
            "eligible_listing_count": len(records),
            "returned_count": len(ranked),
            "limit": limit,
            "complete_for_declared_window": len(ranked) <= limit,
        },
        "license": LICENSE,
        "attribution": ATTRIBUTION,
        "results": [
            {
                "rank": index,
                **{key: value for key, value in row.items() if key != "keywords"},
                "license": LICENSE,
                "attribution": ATTRIBUTION,
            }
            for index, row in enumerate(ranked, start=1)
        ],
    }
    destination = root / OUTPUT / f"{task_id}--{phase}.json"
    write_json_atomic(destination, payload)
    return payload


def _reviewed_query(root: Path, task_id: str, phase: str) -> str:
    reviews = json.loads((root / QUERY_REVIEWS).read_text())
    review = (reviews.get("tasks") or {}).get(task_id)
    if not isinstance(review, dict):
        raise ValueError(f"no frozen query review for {task_id}")
    key = {"before_change": "before_query", "after_change": "after_query"}[phase]
    query = str(review.get(key) or "").strip()
    if not query:
        raise ValueError(f"missing {key} for {task_id}")
    return query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-id", required=True)
    parser.add_argument(
        "--phase", choices=("before_change", "after_change"), required=True
    )
    parser.add_argument("--country", default="US")
    parser.add_argument("--limit", type=int, default=48)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    root = Path(args.root).resolve()
    query = _reviewed_query(root, args.task_id, args.phase)
    payload = capture(
        root,
        task_id=args.task_id,
        phase=args.phase,
        query=query,
        country=args.country or None,
        limit=args.limit,
    )
    print(
        json.dumps(
            {
                "task_id": payload["task_id"],
                "phase": payload["phase"],
                "query": payload["query"],
                "returned_count": len(payload["results"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
