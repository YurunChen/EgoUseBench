"""Add task-irrelevant upstream listings to EgoMart, EgoMedia, and EgoMaps.

Existing entity IDs, scored fields, and task-referenced records stay in place.
New rows are frozen local copies of Amazon/ABO, YouTube, and Wikipedia/OSM
places (Maps identity uses a Google Maps search URL, matching the current mock).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import random
import re
import subprocess
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from PIL import Image

from tools.catalog.catalog_clients import (
    http_get,
    wiki_geosearch,
    yt_search_json,
    yt_video_json,
)
from tools.catalog.freeze_catalog_order import sort_frozen

ROOT = Path(__file__).resolve().parents[2]
NOW = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
USER_AGENT = (
    "egoCUAbench-catalog-fill/1.0 (local research benchmark; "
    "https://en.wikipedia.org/wiki/User:EgoCUAbench)"
)
CACHE = ROOT / ".scratch/upstream-catalog"

MAP_ANCHORS = (
    (39.7684, -86.1581),  # Indianapolis
    (39.9783, -86.1280),  # Carmel, IN
    (4.5981, -74.0760),  # Bogotá
    (22.3095, 39.1044),  # Thuwal / KAUST
    (40.7580, -73.9855),  # Manhattan
    (37.7749, -122.4194),  # San Francisco
    (51.5074, -0.1278),  # London
    (48.8566, 2.3522),  # Paris
    (35.6812, 139.7671),  # Tokyo
    (41.8781, -87.6298),  # Chicago
    (47.6062, -122.3321),  # Seattle
    (30.2672, -97.7431),  # Austin
    (42.3601, -71.0589),  # Boston
    (39.7392, -104.9903),  # Denver
    (25.7617, -80.1918),  # Miami
    (52.5200, 13.4050),  # Berlin
    (41.9028, 12.4964),  # Rome
    (-33.8688, 151.2093),  # Sydney
    (43.6532, -79.3832),  # Toronto
    (19.4326, -99.1332),  # Mexico City
)

YOUTUBE_QUERIES = (
    "city walk tokyo 4k",
    "lofi hip hop mix",
    "sourdough baking at home",
    "oil painting time lapse landscape",
    "piano cover studio ghibli",
    "beginner pottery wheel",
    "national park hiking vlog",
    "street photography walk",
    "home espresso tutorial",
    "classical guitar recital",
    "aquarium coral reef 4k",
    "train journey countryside",
    "watercolor sketching urban",
    "birdwatching backyard",
    "library study with me",
    "origami crane tutorial",
    "jazz trio live session",
    "vegetable garden tour",
    "museum gallery walkthrough",
    "sailboat harbor documentary",
)


def _oid(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256(
        f"egocua-fill-v1\0{prefix}\0".encode() + "\0".join(parts).encode()
    ).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _http_get(url: str, timeout: int = 20) -> bytes:
    return http_get(url, timeout=timeout, user_agent=USER_AGENT)


def _jpeg_payload(data: bytes) -> bytes:
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()


def _write_jpeg(relative: str, data: bytes) -> str:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jpeg_payload(data)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _passes_mart_image(data: bytes) -> bool:
    with Image.open(io.BytesIO(data)) as image:
        width, height = image.size
        if max(width, height) < 1000:
            return False
        if not 0.75 <= width / height <= 1.33:
            return False
        rgba = image.convert("RGBA")
        corners: list[tuple[int, int, int, int]] = [
            cast(tuple[int, int, int, int], rgba.getpixel((0, 0))),
            cast(tuple[int, int, int, int], rgba.getpixel((width - 1, 0))),
            cast(tuple[int, int, int, int], rgba.getpixel((0, height - 1))),
            cast(tuple[int, int, int, int], rgba.getpixel((width - 1, height - 1))),
        ]
        return all(
            pixel[3] < 16 or min(pixel[0], pixel[1], pixel[2]) >= 240
            for pixel in corners
        )


def _abo_image_meta(metadata_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    path = metadata_root / "images" / "metadata" / "images.csv.gz"
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                width = int(row["width"])
                height = int(row["height"])
            except (KeyError, TypeError, ValueError):
                continue
            result[row["image_id"]] = {
                "path": row["path"],
                "width": width,
                "height": height,
            }
    return result


def fill_amazon(*, count: int) -> int:
    from tools.catalog.search_abo_products import (
        ABO_BASE,
        _listings,
        _preferred_name,
        _values,
    )

    path = ROOT / "sites/egomart/products.json"
    payload = json.loads(path.read_text())
    products: list[dict[str, Any]] = payload["products"]
    used_ids = {item["item_id"] for item in products}
    used_names = {
        " ".join(str(item.get("name") or "").casefold().split()) for item in products
    }
    used_asins = set()
    used_images = {str(item.get("image_path")) for item in products}
    for item in products:
        for blob in (
            item.get("text_provenance") or {},
            item.get("image_provenance") or {},
        ):
            url = str(blob.get("source_url") or "")
            match = re.search(r"/dp/([A-Z0-9]{10})", url)
            if match:
                used_asins.add(match.group(1))
    prices = [
        str(item["price"])
        for item in products
        if re.search(r"\d+(?:\.\d+)?", str(item.get("price") or ""))
    ]
    metadata_root = ROOT / "reference/catalogs/amazon-berkeley-objects"
    print("amazon loading ABO image index", flush=True)
    images = _abo_image_meta(metadata_root)
    candidates: list[dict[str, Any]] = []
    for record in _listings(metadata_root):
        if len(candidates) >= count + 80:
            break
        if record.get("country") != "US":
            continue
        asin = str(record.get("item_id") or "")
        if not asin or asin in used_asins:
            continue
        meta = images.get(str(record.get("main_image_id") or ""))
        name = _preferred_name(record.get("item_name"))
        brand = _preferred_name(record.get("brand"))
        if not meta or not name:
            continue
        width, height = int(meta["width"]), int(meta["height"])
        if max(width, height) < 1000 or not 0.75 <= width / height <= 1.33:
            continue
        folded = " ".join(name.casefold().split())
        if folded in used_names:
            continue
        item_id = _oid("p", asin)
        if item_id in used_ids:
            continue
        used_asins.add(asin)
        used_ids.add(item_id)
        used_names.add(folded)
        product_type = (_values(record.get("product_type")) or ["Product"])[0]
        color = _preferred_name(record.get("color"))
        attributes = {"product_type": product_type.replace("_", " ").title()}
        if color:
            attributes["listing_color"] = color
        candidates.append(
            {
                "asin": asin,
                "item_id": item_id,
                "name": name,
                "brand": brand or "Amazon",
                "attributes": attributes,
                "image_url": f"{ABO_BASE}/images/original/{meta['path']}",
                "price": prices[
                    int(hashlib.sha256(asin.encode()).hexdigest()[:8], 16) % len(prices)
                ],
            }
        )
    print(f"amazon candidates {len(candidates)}", flush=True)

    def download(candidate: dict[str, Any]) -> tuple[dict[str, Any], bytes, str] | None:
        try:
            raw = _http_get(candidate["image_url"])
        except RuntimeError:
            return None
        if not _passes_mart_image(raw):
            return None
        return candidate, raw, hashlib.sha256(raw).hexdigest()

    added = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download, candidate) for candidate in candidates]
        for future in as_completed(futures):
            if added >= count:
                continue
            got = future.result()
            if got is None:
                continue
            candidate, raw, digest = got
            relative = f"sites/egomart/static/products/{digest}.jpg"
            if relative in used_images:
                continue
            (ROOT / relative).parent.mkdir(parents=True, exist_ok=True)
            (ROOT / relative).write_bytes(raw)
            page = f"https://www.amazon.com/dp/{candidate['asin']}"
            products.append(
                {
                    "item_id": candidate["item_id"],
                    "name": candidate["name"],
                    "brand": candidate["brand"],
                    "price": candidate["price"],
                    "attributes": candidate["attributes"],
                    "image_path": relative,
                    "image_provenance": {
                        "source_url": page,
                        "image_url": candidate["image_url"],
                        "retrieved_at": NOW,
                        "license_or_access_note": (
                            "CC BY 4.0; attribute Amazon.com and the Amazon Berkeley Objects dataset creators."
                        ),
                        "sha256": digest,
                    },
                    "text_provenance": {
                        "source_url": page,
                        "retrieved_at": NOW,
                        "access_note": (
                            "Task-irrelevant Amazon/ABO listing frozen for the local mock. "
                            "Price is sampled from the existing catalog distribution."
                        ),
                    },
                }
            )
            used_images.add(relative)
            added += 1
            if added % 25 == 0:
                print(f"amazon added {added}/{count}", flush=True)
    payload["products"] = sort_frozen(products, "item_id")
    _dump(path, payload)
    print(f"amazon catalog {len(products)} (+{added})")
    return added


def _existing_youtube_ids(videos: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for video in videos:
        url = str((video.get("source") or {}).get("page_url") or "")
        match = re.search(r"(?:v=|/shorts/)([A-Za-z0-9_-]{11})", url)
        if match:
            found.add(match.group(1))
    return found


def _yt_search(query: str) -> list[dict[str, Any]]:
    return yt_search_json(query, limit=25, flat=True, timeout=120)


def _youtube_listing(
    record: dict[str, Any],
    *,
    youtube_id: str,
    video_id: str,
    thumbnail_path: str,
    thumbnail_sha256: str,
) -> dict[str, Any]:
    title = str(record.get("title") or "").strip()
    duration = record.get("duration")
    if (
        not title
        or not isinstance(duration, (int, float))
        or not 20 <= duration <= 4 * 60 * 60
    ):
        raise ValueError(f"{youtube_id}: unusable title or duration")
    upload = str(record.get("upload_date") or "")
    published = (
        f"{upload[0:4]}-{upload[4:6]}-{upload[6:8]}" if len(upload) == 8 else NOW[:10]
    )
    views = record.get("view_count")
    thumb_url = f"https://i.ytimg.com/vi/{youtube_id}/hqdefault.jpg"
    return {
        "video_id": video_id,
        "title": title,
        "kind": "video",
        "summary": title,
        "thumbnail_path": thumbnail_path,
        "channel_title": str(
            record.get("uploader") or record.get("channel") or "YouTube"
        ),
        "published_at": published,
        "duration_seconds": int(duration),
        "view_count": int(views)
        if isinstance(views, (int, float)) and views >= 0
        else 0,
        "details": {"format": "video"},
        "chapters": [],
        "facts": {},
        "source": {
            "record": "youtube.com listing",
            "page_url": f"https://www.youtube.com/watch?v={youtube_id}",
            "thumbnail_url": thumb_url,
            "retrieved_at": NOW,
            "license_or_access_note": (
                "YouTube thumbnail and metadata retrieved for a local frozen mock; "
                "redistribution rights are not asserted."
            ),
            "thumbnail_sha256": thumbnail_sha256,
            "access_note": "Task-grounding YouTube listing frozen for the local mock.",
        },
    }


def add_youtube_ids(*, youtube_ids: list[str]) -> int:
    """Deterministically freeze selected public YouTube listings into EgoMedia."""

    catalog_path = ROOT / "sites/egomedia/videos.json"
    payload = json.loads(catalog_path.read_text())
    videos: list[dict[str, Any]] = payload["videos"]
    existing = _existing_youtube_ids(videos)
    pending: list[tuple[str, bytes, dict[str, Any]]] = []
    reserved_ids = {video["video_id"] for video in videos}
    for youtube_id in dict.fromkeys(youtube_ids):
        if youtube_id in existing:
            print(f"youtube exists {youtube_id}", flush=True)
            continue
        print(f"youtube fetch {youtube_id}", flush=True)
        record = yt_video_json(youtube_id)
        video_id = _oid("m", youtube_id)
        if video_id in reserved_ids:
            raise RuntimeError(f"generated duplicate video id: {video_id}")
        thumb_url = f"https://i.ytimg.com/vi/{youtube_id}/hqdefault.jpg"
        raw = _http_get(thumb_url)
        relative = f"sites/egomedia/static/thumbnails/{video_id}.jpg"
        jpeg = _jpeg_payload(raw)
        digest = hashlib.sha256(jpeg).hexdigest()
        listing = _youtube_listing(
            record,
            youtube_id=youtube_id,
            video_id=video_id,
            thumbnail_path=relative,
            thumbnail_sha256=digest,
        )
        pending.append((relative, jpeg, listing))
        reserved_ids.add(video_id)
        existing.add(youtube_id)
    for relative, jpeg, listing in pending:
        asset_path = ROOT / relative
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(jpeg)
        videos.append(listing)
    payload["videos"] = sort_frozen(videos, "video_id")
    _dump(catalog_path, payload)
    print(f"youtube catalog {len(videos)} (+{len(pending)})")
    return len(pending)


def fill_youtube(*, count: int) -> int:
    path = ROOT / "sites/egomedia/videos.json"
    payload = json.loads(path.read_text())
    videos: list[dict[str, Any]] = payload["videos"]
    used_ids = {item["video_id"] for item in videos}
    used_yt = _existing_youtube_ids(videos)
    added = 0
    for query in YOUTUBE_QUERIES:
        if added >= count:
            break
        print(f"youtube search {query}", flush=True)
        try:
            rows = _yt_search(query)
        except (subprocess.TimeoutExpired, RuntimeError) as exc:
            print("youtube search failed", query, exc, flush=True)
            continue
        for record in rows:
            if added >= count:
                break
            youtube_id = str(record.get("id") or "")
            title = str(record.get("title") or "").strip()
            duration = record.get("duration")
            if (
                not youtube_id
                or youtube_id in used_yt
                or not title
                or not isinstance(duration, (int, float))
                or duration < 30
                or duration > 3 * 60 * 60
            ):
                continue
            video_id = _oid("m", youtube_id)
            if video_id in used_ids:
                continue
            thumb_url = f"https://i.ytimg.com/vi/{youtube_id}/hqdefault.jpg"
            try:
                raw = _http_get(thumb_url)
            except RuntimeError:
                continue
            relative = f"sites/egomedia/static/thumbnails/{video_id}.jpg"
            digest = _write_jpeg(relative, raw)
            upload = str(record.get("upload_date") or "")
            published = (
                f"{upload[0:4]}-{upload[4:6]}-{upload[6:8]}"
                if len(upload) == 8
                else NOW[:10]
            )
            views = record.get("view_count")
            channel = str(record.get("uploader") or record.get("channel") or "YouTube")
            videos.append(
                {
                    "video_id": video_id,
                    "title": title,
                    "kind": "video",
                    "summary": title,
                    "thumbnail_path": relative,
                    "channel_title": channel,
                    "published_at": published,
                    "duration_seconds": int(duration),
                    "view_count": int(views)
                    if isinstance(views, (int, float)) and views >= 0
                    else 0,
                    "details": {"format": "video"},
                    "chapters": [],
                    "facts": {},
                    "source": {
                        "record": "youtube.com listing",
                        "page_url": f"https://www.youtube.com/watch?v={youtube_id}",
                        "thumbnail_url": thumb_url,
                        "retrieved_at": NOW,
                        "license_or_access_note": (
                            "YouTube thumbnail and metadata retrieved for a local frozen mock; "
                            "redistribution rights are not asserted."
                        ),
                        "thumbnail_sha256": digest,
                        "access_note": "Task-irrelevant YouTube listing frozen for the local mock.",
                    },
                }
            )
            used_ids.add(video_id)
            used_yt.add(youtube_id)
            added += 1
        time.sleep(0.4)
        if added and added % 25 == 0:
            payload["videos"] = sort_frozen(videos, "video_id")
            _dump(path, payload)
    payload["videos"] = sort_frozen(videos, "video_id")
    _dump(path, payload)
    print(f"youtube catalog {len(videos)} (+{added})")
    return added


def _wiki_geosearch(lat: float, lon: float) -> list[dict[str, Any]]:
    return wiki_geosearch(lat, lon, user_agent=USER_AGENT)


def _place_kind(title: str, extract: str) -> str:
    text = f"{title} {extract}".casefold()
    mapping = (
        ("university", "university"),
        ("museum", "museum"),
        ("park", "park"),
        ("garden", "park"),
        ("hotel", "hotel"),
        ("stadium", "stadium"),
        ("church", "landmark"),
        ("cathedral", "landmark"),
        ("library", "landmark"),
        ("market", "shopping_center"),
        ("plaza", "plaza"),
        ("bridge", "landmark"),
        ("station", "landmark"),
        ("airport", "landmark"),
        ("restaurant", "restaurant"),
    )
    for token, kind in mapping:
        if token in text:
            return kind
    return "landmark"


def fill_maps(*, count: int) -> int:
    path = ROOT / "sites/egomaps/places.json"
    payload = json.loads(path.read_text())
    places: list[dict[str, Any]] = payload["places"]
    used_ids = {item["place_id"] for item in places}
    used_names = {
        " ".join(str(item.get("name") or "").casefold().split()) for item in places
    }
    used_photos = set()
    for item in places:
        photo = ROOT / str(item.get("photo_path") or "")
        if photo.is_file():
            used_photos.add(hashlib.sha256(photo.read_bytes()).hexdigest())
    added = 0
    rng = random.Random(19)
    for lat, lon in MAP_ANCHORS:
        if added >= count:
            break
        print(f"maps geosearch {lat},{lon}", flush=True)
        try:
            pages = _wiki_geosearch(lat, lon)
        except RuntimeError as exc:
            print("maps geosearch failed", lat, lon, exc, flush=True)
            continue
        rng.shuffle(pages)
        for page in pages:
            if added >= count:
                break
            title = str(page.get("title") or "").strip()
            extract = " ".join(str(page.get("extract") or "").split())
            thumb = page.get("thumbnail") or {}
            image_url = str(thumb.get("source") or "")
            coords = (page.get("coordinates") or [{}])[0]
            if not title or not image_url or not isinstance(coords, dict):
                continue
            latitude = coords.get("lat")
            longitude = coords.get("lon")
            folded = " ".join(title.casefold().split())
            if (
                folded in used_names
                or not isinstance(latitude, (int, float))
                or not isinstance(longitude, (int, float))
                or any(image_url.casefold().endswith(ext) for ext in (".svg", ".gif"))
            ):
                continue
            place_id = _oid("l", title)
            if place_id in used_ids:
                continue
            try:
                raw = _http_get(image_url, timeout=20)
                with Image.open(io.BytesIO(raw)) as image:
                    if min(image.size) < 240:
                        continue
            except Exception as exc:  # noqa: BLE001
                print("maps skip image", title, exc, flush=True)
                continue
            relative = f"sites/egomaps/static/photos/{place_id}.jpg"
            digest = _write_jpeg(relative, raw)
            if digest in used_photos:
                (ROOT / relative).unlink(missing_ok=True)
                continue
            walking = 4 + (int(digest[:2], 16) % 18)
            driving = 2 + (int(digest[2:4], 16) % 12)
            summary = (
                extract[:220] if len(extract) >= 24 else f"{title} is a mapped place."
            )
            places.append(
                {
                    "place_id": place_id,
                    "name": title,
                    "kind": _place_kind(title, extract),
                    "summary": summary,
                    "address": title,
                    "route_options": {
                        "walking": f"{walking} min",
                        "driving": f"{driving} min",
                    },
                    "facts": {},
                    "photo_path": relative,
                    "location": {
                        "latitude": float(latitude),
                        "longitude": float(longitude),
                    },
                    "source": {
                        "page_url": (
                            "https://www.google.com/maps/search/?api=1&query="
                            + urllib.parse.quote_plus(title)
                        ),
                        "image_url": image_url,
                        "wikipedia_url": str(page.get("fullurl") or ""),
                        "retrieved_at": NOW,
                        "sha256": digest,
                        "license_or_access_note": (
                            "Place photo from Wikimedia/Wikipedia; redistribution rights follow the "
                            "upstream file license. Maps URL is a Google Maps search query."
                        ),
                        "access_note": (
                            "Task-irrelevant frozen place. Name and coordinates from Wikipedia; "
                            "photo from Wikimedia."
                        ),
                    },
                }
            )
            used_ids.add(place_id)
            used_names.add(folded)
            used_photos.add(digest)
            added += 1
        time.sleep(0.3)
        if added:
            payload["places"] = sort_frozen(places, "place_id")
            _dump(path, payload)
    payload["places"] = sort_frozen(places, "place_id")
    _dump(path, payload)
    print(f"maps catalog {len(places)} (+{added})")
    return added


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("site", choices=("amazon", "youtube", "maps", "all"))
    parser.add_argument("--amazon-count", type=int, default=400)
    parser.add_argument("--youtube-count", type=int, default=250)
    parser.add_argument(
        "--youtube-id",
        action="append",
        default=[],
        help="freeze this exact public YouTube video id; repeat for multiple listings",
    )
    parser.add_argument("--maps-count", type=int, default=150)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    added = {}
    if args.site in {"amazon", "all"}:
        added["amazon"] = fill_amazon(count=args.amazon_count)
    if args.youtube_id:
        if args.site not in {"youtube", "all"}:
            parser.error("--youtube-id requires site youtube or all")
        added["youtube"] = add_youtube_ids(youtube_ids=args.youtube_id)
    elif args.site in {"youtube", "all"}:
        added["youtube"] = fill_youtube(count=args.youtube_count)
    if args.site in {"maps", "all"}:
        added["maps"] = fill_maps(count=args.maps_count)
    print(json.dumps(added, indent=2))


if __name__ == "__main__":
    main()
