"""Deterministically randomize frozen catalog order so position cannot leak role.

Catalogs were written target-first, so the default "featured" ordering ranked every
task target near the top of its result page: an Agent could score well by taking the
first hit instead of comparing candidates. ADR-0005 requires catalog order to be
deterministically randomized and frozen; this tool enforces it.

Order is the sort over ``md5(salt + entity_id)``. That is reproducible, independent
of the previous order, and stable when entities are added later: existing entities
keep their relative order instead of being permuted by a fresh shuffle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

CATALOGS = (
    ("sites/egomart/products.json", "products", "item_id"),
    ("sites/egomedia/videos.json", "videos", "video_id"),
    ("sites/egomaps/places.json", "places", "place_id"),
)
ORDER_SALT = "egocua-catalog-order-v1"


def order_key(entity_id: str) -> str:
    return hashlib.md5(f"{ORDER_SALT}\0{entity_id}".encode()).hexdigest()


def sort_frozen(entities: list[dict[str, Any]], id_key: str) -> list[dict[str, Any]]:
    """Frozen order for a catalog's entities.

    Any tool that adds entities calls this before writing: order is an invariant of
    the catalog file, so appending without it leaves the file out of frozen order.
    """

    return sorted(entities, key=lambda entity: order_key(str(entity[id_key])))


def freeze(root: Path) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for relative, bucket, id_key in CATALOGS:
        path = root / relative
        payload = json.loads(path.read_text())
        entities = payload[bucket]
        before = [entity[id_key] for entity in entities]
        payload[bucket] = sort_frozen(entities, id_key)
        after = [entity[id_key] for entity in payload[bucket]]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        report.append(
            {
                "catalog": relative,
                "entities": len(entities),
                "reordered": before != after,
            }
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(freeze(Path(args.root).resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
