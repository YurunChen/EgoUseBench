#!/usr/bin/env python3
"""Replace mock catalog records with frozen Amazon / YouTube / Google Maps listings.

Task entity IDs and scored fields stay in place. Visible names, images, and
provenance are taken from the upstream site (or ABO for Amazon listings).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from egocua.schema import load_tasks, referenced_entity_ids
from tools.catalog.catalog_clients import http_get, yt_search_json


ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
USER_AGENT = "egoCUAbench-catalog-align/1.0 (research; local frozen mock)"


def _http_get(url: str, timeout: int = 30) -> bytes:
    return http_get(
        url, timeout=timeout, attempts=4, user_agent=USER_AGENT, backoff=0.4
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_image(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return _sha256(data)


def _task_entity_ids(site: str) -> set[str]:
    return {
        entity_id
        for task in load_tasks(ROOT / "data/tasks")
        if task.get("site") == site
        for entity_id in referenced_entity_ids(task)
    }


def _yt_search(query: str, *, used: set[str]) -> dict[str, Any] | None:
    queries = [query]
    words = [token for token in query.replace("—", " ").split() if token]
    if len(words) > 3:
        queries.append(" ".join(words[-4:]))
        queries.append(" ".join(words[:4]))
    for attempt in queries:
        for record in yt_search_json(attempt, limit=8, flat=False, timeout=90):
            video_id = str(record.get("id") or "")
            duration = record.get("duration")
            if (
                not video_id
                or video_id in used
                or not isinstance(duration, (int, float))
                or duration < 20
                or duration > 4 * 60 * 60
            ):
                continue
            upload = str(record.get("upload_date") or "")
            published = (
                f"{upload[0:4]}-{upload[4:6]}-{upload[6:8]}"
                if len(upload) == 8
                else NOW[:10]
            )
            views = record.get("view_count")
            return {
                "youtube_id": video_id,
                "title": str(record.get("title") or attempt),
                "channel_title": str(
                    record.get("uploader") or record.get("channel") or "YouTube"
                ),
                "duration_seconds": int(duration),
                "view_count": int(views)
                if isinstance(views, (int, float)) and views >= 0
                else 0,
                "published_at": published,
                "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
                "watch_url": f"https://www.youtube.com/watch?v={video_id}",
            }
    return None


def align_youtube(*, resume: bool) -> None:
    path = ROOT / "sites/egomedia/videos.json"
    payload = json.loads(path.read_text())
    cache_path = ROOT / ".scratch/upstream-catalog/youtube.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict[str, Any] = (
        json.loads(cache_path.read_text()) if resume and cache_path.exists() else {}
    )
    used = {
        str(item["youtube_id"])
        for item in cache.values()
        if isinstance(item, dict) and item.get("youtube_id")
    }
    updated = 0
    for video in payload["videos"]:
        video_id = video["video_id"]
        if video_id in cache and cache[video_id].get("youtube_id"):
            found = cache[video_id]
        else:
            query = str(video.get("title") or "tutorial")
            print(f"youtube search {video_id}: {query}", flush=True)
            found = _yt_search(query, used=used)
            if found is None:
                raise RuntimeError(f"no YouTube listing for {video_id}: {query}")
            cache[video_id] = found
            cache_path.write_text(json.dumps(cache, indent=2) + "\n")
            used.add(found["youtube_id"])
            time.sleep(0.4)
        image = _http_get(found["thumbnail_url"])
        relative = f"sites/egomedia/static/thumbnails/{video_id}.jpg"
        digest = _write_image(ROOT / relative, image)
        video["title"] = found["title"]
        video["channel_title"] = found["channel_title"]
        video["published_at"] = found["published_at"]
        video["duration_seconds"] = found["duration_seconds"]
        video["view_count"] = found["view_count"]
        video["thumbnail_path"] = relative
        video["source"] = {
            "record": "youtube.com listing",
            "page_url": found["watch_url"],
            "thumbnail_url": found["thumbnail_url"],
            "retrieved_at": NOW,
            "license_or_access_note": (
                "YouTube thumbnail and metadata retrieved for a local frozen mock; "
                "redistribution rights are not asserted."
            ),
            "thumbnail_sha256": digest,
        }
        updated += 1
    payload["content_origin"] = (
        "Frozen youtube.com listings. Task-scored facts remain benchmark-authored."
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"youtube updated {updated} videos")


def _nominatim(query: str) -> dict[str, Any] | None:
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }
    )
    time.sleep(1.1)
    rows = json.loads(_http_get(url).decode())
    if not rows:
        return None
    row = rows[0]
    address = row.get("display_name") or query
    return {
        "name": str(row.get("name") or query.split(",")[0].strip()),
        "address": address,
        "latitude": float(row["lat"]),
        "longitude": float(row["lon"]),
        "maps_url": (
            "https://www.google.com/maps/search/?api=1&query="
            + urllib.parse.quote_plus(query)
        ),
        "osm_url": f"https://www.openstreetmap.org/{row.get('osm_type', 'way')}/{row.get('osm_id', '')}",
    }


def align_maps() -> None:
    queries = {
        "l_6e86c83dabd9": ["Peace Water Winery, Carmel, Indiana"],
        "l_17eea1ef0ac0": [
            "Comfort Inn Indianapolis North - Carmel",
            "Hotel Carmel, Carmel, Indiana",
        ],
        "l_e4898a9a35fa": ["Renaissance Indianapolis North Hotel, Carmel, Indiana"],
        "l_cb2e2cee82c4": ["KAUST Souk, Discovery Square", "Discovery Square, Thuwal"],
        "l_6f56ee36a2c7": ["Discovery Square, Thuwal"],
        "l_ef8650dd668b": ["Tamimi, Thuwal"],
        "l_a4ba82d43c58": [
            "Callejón del Embudo, La Candelaria, Bogotá",
            "Callejon del Embudo Bogota",
        ],
        "l_6a57ddb8937c": [
            "Chorro de Quevedo, Bogota",
            "Plazoleta Chorro de Quevedo, Bogotá",
        ],
        "l_d88b4092851d": ["Plaza de Bolívar, Bogotá"],
    }
    path = ROOT / "sites/egomaps/places.json"
    payload = json.loads(path.read_text())
    for place in payload["places"]:
        place_id = place["place_id"]
        found = None
        query = queries[place_id][0]
        for query in queries[place_id]:
            print(f"maps lookup {place_id}: {query}", flush=True)
            found = _nominatim(query)
            if found is not None:
                break
        if found is None:
            raise RuntimeError(f"no Google Maps / OSM hit for {place_id}")
        place["address"] = found["address"]
        place["location"] = {
            "latitude": found["latitude"],
            "longitude": found["longitude"],
        }
        if place_id not in {
            "l_6e86c83dabd9",
            "l_cb2e2cee82c4",
            "l_6f56ee36a2c7",
            "l_a4ba82d43c58",
            "l_6a57ddb8937c",
            "l_ef8650dd668b",
        }:
            place["name"] = found["name"]
        source = dict(place.get("source") or {})
        source["page_url"] = found["maps_url"]
        source["retrieved_at"] = NOW
        source["access_note"] = (
            "Place name, address, and coordinates match the Google Maps query; "
            "photo remains a locally frozen asset."
        )
        source.pop("record", None)
        source.pop("photo_note", None)
        source.pop("method", None)
        place["source"] = source
    if payload["places"][0]["place_id"] == "l_6e86c83dabd9":
        hotel = next(
            item for item in payload["places"] if item["place_id"] == "l_17eea1ef0ac0"
        )
        payload["places"][0]["facts"]["nearest_place"] = hotel["name"]
        nearby = payload["places"][0].get("nearby") or []
        if nearby:
            nearby[0]["text"] = f"{hotel['name']}, walking distance"
    payload["content_origin"] = (
        "Frozen Google Maps places. Task-scored facts remain benchmark-authored."
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print("maps updated", len(payload["places"]), "places")


def _asin(url: str) -> str | None:
    match = re.search(r"/dp/([A-Z0-9]{10})", url)
    return match.group(1) if match else None


def _listing_url(host: str, item_id: str) -> str:
    domain = host if host.startswith("amazon.") else "amazon.com"
    if not domain.startswith("www."):
        domain = "www." + domain
    return f"https://{domain}/dp/{item_id}"


def align_amazon() -> None:
    from tools.catalog.search_abo_products import (
        ABO_BASE,
        _image_paths,
        _listings,
        _preferred_name,
        _values,
    )

    path = ROOT / "sites/egomart/products.json"
    payload = json.loads(path.read_text())
    needed = _task_entity_ids("egomart")
    used_asins: set[str] = set()
    replace: list[dict[str, Any]] = []
    for product in payload["products"]:
        url = (
            (product.get("text_provenance") or {}).get("source_url")
            or (product.get("image_provenance") or {}).get("source_url")
            or ""
        )
        host = urlparse(url).netloc.replace("www.", "")
        asin = _asin(url)
        is_amazon = "amazon." in host or "amazon-berkeley-objects" in host
        if is_amazon:
            if asin:
                used_asins.add(asin)
                page = f"https://www.amazon.com/dp/{asin}"
                if product.get("text_provenance"):
                    product["text_provenance"]["source_url"] = page
                    product["text_provenance"]["retrieved_at"] = NOW
                    product["text_provenance"]["access_note"] = (
                        "Amazon listing text frozen for the local mock. "
                        "Task-scored price and rating fields stay authored where present."
                    )
                if product.get("image_provenance"):
                    product["image_provenance"]["source_url"] = page
        else:
            replace.append(product)

    task_replace = [item for item in replace if item["item_id"] in needed]
    filler_replace = [item for item in replace if item["item_id"] not in needed]
    print(
        "amazon replace",
        len(replace),
        "task",
        len(task_replace),
        "filler",
        len(filler_replace),
        flush=True,
    )

    metadata_root = ROOT / "reference/catalogs/amazon-berkeley-objects"
    paths = _image_paths(metadata_root)
    queries = {
        item["item_id"]: [
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9]+", item.get("name") or "")
            if len(token) > 2
        ][:4]
        for item in task_replace
    }
    task_hits: dict[str, dict[str, Any]] = {}
    filler: list[dict[str, Any]] = []

    def as_listing(record: dict[str, Any], image_path: str) -> dict[str, Any]:
        item_id = str(record["item_id"])
        domain = str(record.get("domain_name") or "amazon.com")
        return {
            "item_id": item_id,
            "name": _preferred_name(record.get("item_name")),
            "brand": _preferred_name(record.get("brand")),
            "page_url": _listing_url(domain, item_id),
            "image_url": f"{ABO_BASE}/images/original/{image_path}",
        }

    for record in _listings(metadata_root):
        if record.get("country") != "US":
            continue
        item_id = str(record.get("item_id") or "")
        if not item_id or item_id in used_asins:
            continue
        image_path = paths.get(str(record.get("main_image_id")))
        if not image_path:
            continue
        name = _preferred_name(record.get("item_name"))
        if not name:
            continue
        searchable = " ".join(
            [
                name,
                _preferred_name(record.get("brand")),
                *_values(record.get("product_type")),
            ]
        ).casefold()
        assigned = False
        for product_id, tokens in queries.items():
            if product_id in task_hits or not tokens:
                continue
            if all(token in searchable for token in tokens):
                matched_listing = as_listing(record, image_path)
                task_hits[product_id] = matched_listing
                used_asins.add(item_id)
                assigned = True
                break
        if assigned:
            continue
        if len(filler) < len(filler_replace) + 8:
            filler.append(as_listing(record, image_path))
            used_asins.add(item_id)
        if len(task_hits) == len(queries) and len(filler) >= len(filler_replace):
            break

    filler_iter = iter(filler)
    skipped_task = 0
    for product in replace:
        if product["item_id"] in task_hits:
            listing = task_hits[product["item_id"]]
        elif product["item_id"] in needed:
            skipped_task += 1
            continue
        else:
            try:
                listing = next(filler_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"ran out of ABO US listings at {product['item_id']}"
                ) from exc
        print("amazon apply", product["item_id"], listing["name"][:70], flush=True)
        try:
            _apply_amazon_listing(product, listing)
        except Exception as exc:  # noqa: BLE001
            print("amazon skip", product["item_id"], exc, flush=True)
            if product["item_id"] in needed:
                continue
            try:
                listing = next(filler_iter)
            except StopIteration as exhausted:
                raise RuntimeError(
                    f"ran out of ABO US listings at {product['item_id']}"
                ) from exhausted
            _apply_amazon_listing(product, listing)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(
        "amazon catalog written",
        len(payload["products"]),
        "task hits",
        len(task_hits),
        "task extras kept",
        skipped_task,
    )


def _apply_amazon_listing(product: dict[str, Any], listing: dict[str, Any]) -> None:
    image = _http_get(listing["image_url"])
    digest = _sha256(image)
    relative = f"sites/egomart/static/products/{digest}.jpg"
    _write_image(ROOT / relative, image)
    product["name"] = listing["name"]
    if listing.get("brand"):
        product["brand"] = listing["brand"]
    product["image_path"] = relative
    product["image_provenance"] = {
        "source_url": listing["page_url"],
        "image_url": listing["image_url"],
        "retrieved_at": NOW,
        "license_or_access_note": (
            "CC BY 4.0; attribute Amazon.com and the Amazon Berkeley Objects dataset creators."
        ),
        "sha256": digest,
    }
    product["text_provenance"] = {
        "source_url": listing["page_url"],
        "retrieved_at": NOW,
        "access_note": (
            "Listing name, brand, and image originate from Amazon / ABO. "
            "Task-scored price, rating, size, and delivery fields stay frozen."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site", choices=("youtube", "maps", "amazon", "all"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.site in {"youtube", "all"}:
        align_youtube(resume=args.resume)
    if args.site in {"maps", "all"}:
        align_maps()
    if args.site in {"amazon", "all"}:
        align_amazon()


if __name__ == "__main__":
    main()
