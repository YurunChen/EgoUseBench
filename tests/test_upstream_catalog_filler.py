from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from egocua.environment import load_site_data


ROOT = Path(__file__).resolve().parents[1]


def _photo_hashes(site: str, key: str) -> set[str]:
    data = json.loads(
        (ROOT / f"sites/{site}")
        .joinpath(
            {
                "egomart": "products.json",
                "egomedia": "videos.json",
                "egomaps": "places.json",
            }[site]
        )
        .read_text()
    )
    bucket = {
        "egomart": "products",
        "egomedia": "videos",
        "egomaps": "places",
    }[site]
    hashes: set[str] = set()
    for entity in data[bucket]:
        if key not in entity:
            continue
        path = ROOT / str(entity[key])
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    return hashes


def test_egomart_has_a_long_tail_of_stable_upstream_products() -> None:
    products = json.loads((ROOT / "sites/egomart/products.json").read_text())[
        "products"
    ]
    assert len(products) >= 900
    assert len({item["item_id"] for item in products}) == len(products)
    assert len({item["image_path"] for item in products}) == len(products)
    amazon = sum(
        1
        for item in products
        if "amazon."
        in str((item.get("image_provenance") or {}).get("source_url") or "")
        or "amazon." in str((item.get("text_provenance") or {}).get("source_url") or "")
        or "amazon-berkeley-objects"
        in str((item.get("image_provenance") or {}).get("image_url") or "")
    )
    assert amazon >= 400


def test_egomedia_has_a_long_tail_of_stable_youtube_videos() -> None:
    videos = json.loads((ROOT / "sites/egomedia/videos.json").read_text())["videos"]
    upstream = [
        video for video in videos if (video.get("source") or {}).get("page_url")
    ]
    youtube_ids = set()
    for video in upstream:
        match = re.search(
            r"(?:v=|/shorts/)([A-Za-z0-9_-]{11})",
            str((video.get("source") or {}).get("page_url") or ""),
        )
        if match:
            youtube_ids.add(match.group(1))
    assert len(upstream) >= 300
    assert len(youtube_ids) == len(upstream)
    assert len({video["thumbnail_path"] for video in upstream}) == len(upstream)


@pytest.mark.requires_local_data("sites/egomaps/static/photos")
def test_egomaps_has_task_irrelevant_places_with_distinct_photos() -> None:
    places = json.loads((ROOT / "sites/egomaps/places.json").read_text())["places"]
    hashes = _photo_hashes("egomaps", "photo_path")
    with_maps_url = sum(
        1
        for place in places
        if "google.com/maps" in str((place.get("source") or {}).get("page_url") or "")
    )
    assert len(places) >= 120
    assert len(hashes) >= 80
    assert with_maps_url >= 100
    data = load_site_data(ROOT, "egomaps")
    assert len(data["entities"]) == len(places)
