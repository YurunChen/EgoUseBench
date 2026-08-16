from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any, Iterator


ABO_BASE = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com"


def _values(field: Any) -> list[str]:
    if isinstance(field, str):
        return [field]
    if not isinstance(field, list):
        return []
    values: list[str] = []
    for entry in field:
        if isinstance(entry, str):
            values.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("value"), str):
            values.append(entry["value"])
    return values


def _preferred_name(field: Any) -> str:
    if not isinstance(field, list):
        values = _values(field)
        return values[0] if values else ""
    for prefix in ("en_US", "en_GB", "en_"):
        for entry in field:
            if not isinstance(entry, dict) or not isinstance(entry.get("value"), str):
                continue
            language = str(entry.get("language_tag", ""))
            if language == prefix or (prefix.endswith("_") and language.startswith(prefix)):
                return entry["value"]
    values = _values(field)
    return values[0] if values else ""


def _listings(metadata_root: Path) -> Iterator[dict[str, Any]]:
    for path in sorted((metadata_root / "listings" / "metadata").glob("listings_*.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if isinstance(record, dict):
                    yield record


def _image_paths(metadata_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    path = metadata_root / "images" / "metadata" / "images.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            result[row["image_id"]] = row["path"]
    return result


def search(metadata_root: Path, query: str, *, country: str | None, limit: int) -> list[dict[str, Any]]:
    tokens = [token.casefold() for token in query.split() if token.strip()]
    if not tokens:
        raise ValueError("query must contain at least one token")

    matches: list[tuple[int, dict[str, Any]]] = []
    for record in _listings(metadata_root):
        if country and record.get("country") != country:
            continue
        name = _preferred_name(record.get("item_name"))
        brand = _preferred_name(record.get("brand"))
        product_types = _values(record.get("product_type"))
        keywords = _values(record.get("item_keywords"))
        searchable = " ".join([name, brand, *product_types, *keywords]).casefold()
        if not all(token in searchable for token in tokens):
            continue
        title = f"{brand} {name}".casefold()
        score = sum(3 for token in tokens if token in title)
        score += sum(1 for token in tokens if token in " ".join(product_types).casefold())
        matches.append((score, record))

    paths = _image_paths(metadata_root)
    output: list[dict[str, Any]] = []
    for _, record in sorted(matches, key=lambda value: (-value[0], _preferred_name(value[1].get("item_name")))):
        image_id = record.get("main_image_id")
        image_path = paths.get(str(image_id))
        if not image_path:
            continue
        domain = str(record.get("domain_name", ""))
        item_id = str(record.get("item_id", ""))
        output.append({
            "item_id": item_id,
            "name": _preferred_name(record.get("item_name")),
            "brand": _preferred_name(record.get("brand")),
            "product_types": _values(record.get("product_type")),
            "color": _preferred_name(record.get("color")),
            "country": record.get("country"),
            "product_page_url": f"https://www.{domain}/dp/{item_id}" if domain and item_id else None,
            "dataset_source_url": f"{ABO_BASE}/index.html",
            "image_url": f"{ABO_BASE}/images/original/{image_path}",
            "license": "CC BY 4.0; attribute Amazon.com and the ABO dataset creators",
        })
        if len(output) == limit:
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local Amazon Berkeley Objects metadata.")
    parser.add_argument("query", help="Whitespace-separated terms that must all occur in the listing metadata")
    parser.add_argument(
        "--metadata-root",
        default="reference/amazon-berkeley-objects",
        type=Path,
    )
    parser.add_argument("--country", default=None, help="Optional marketplace country code, such as US")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    try:
        result = search(args.metadata_root, args.query, country=args.country, limit=args.limit)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ABO search failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
