"""Assign unique Wikimedia photos to task-referenced EgoMaps places.

Replaces benchmark-authored placeholder copies (e.g. Peace Water Winery JPEG)
with geosearch-sourced thumbnails near each entity's coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from egocua.schema import referenced_entity_ids, load_tasks
from egocua.selection import SELECTION_PATH
from tools.catalog.catalog_clients import http_get, wiki_geosearch
from egocua.selection import load_selection, task_ids as selected_task_ids

ROOT = Path(__file__).resolve().parents[1]
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
USER_AGENT = (
    "egoCUAbench-map-photo-assign/1.0 (local research benchmark; "
    "https://en.wikipedia.org/wiki/User:EgoCUAbench)"
)
WINERY_PLACE_ID = "l_6e86c83dabd9"
PLACEHOLDER_NOTE = (
    "Frozen map listing for local evaluation; scored facts are benchmark-authored."
)
NEIGHBOR_PLACE_IDS = (
    "l_3ee19952ae3f",  # Hillside Technical College
    "l_c33b7cdcd32a",  # Riverfront Arts House
    "l_ab2360c1f78d",  # Riverpark Arena Hotel
    "l_0c7ce9691745",  # Street Corner Diner
)
KIND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "university": ("university", "college", "campus", "school"),
    "cultural_venue": ("hall", "theater", "theatre", "cultural", "opera"),
    "stadium": ("stadium", "arena", "field"),
    "hotel": ("hotel", "inn", "lodging"),
    "restaurant": ("restaurant", "diner", "kitchen", "cafe"),
    "tennis_facility": ("tennis", "court", "club"),
    "park": ("park", "garden", "grounds"),
    "museum": ("museum", "gallery", "heritage"),
    "historic_district": ("historic", "old town", "quarter", "carriage"),
    "historic_center": ("market", "square", "center", "centre", "old town"),
    "shopping_center": ("market", "mall", "square", "shop"),
    "landmark": ("plaza", "monument", "landmark"),
}


def _http_get(url: str, timeout: int = 20) -> bytes:
    return http_get(url, timeout=timeout, user_agent=USER_AGENT)


def _write_jpeg(relative: str, data: bytes) -> str:
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as image:
        rgb = image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="JPEG", quality=90)
        payload = buffer.getvalue()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _wiki_geosearch(lat: float, lon: float) -> list[dict[str, Any]]:
    return wiki_geosearch(lat, lon, user_agent=USER_AGENT)


def _photo_digest(relative: str | None) -> str | None:
    if not relative:
        return None
    path = ROOT / relative
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_placeholder(place: dict[str, Any]) -> bool:
    source = place.get("source")
    if not isinstance(source, dict):
        return True
    if source.get("access_note") == PLACEHOLDER_NOTE and not source.get("image_url"):
        return True
    return False


def _kind_score(kind: str, title: str, extract: str) -> int:
    text = f"{title} {extract}".casefold()
    keywords = KIND_KEYWORDS.get(kind, KIND_KEYWORDS["landmark"])
    return sum(1 for token in keywords if token in text)


def _collect_target_ids(tasks: list[dict[str, Any]]) -> set[str]:
    target: set[str] = set(NEIGHBOR_PLACE_IDS)
    for task in tasks:
        if task.get("site") != "egomaps":
            continue
        target.update(referenced_entity_ids(task))
    return target


def _needs_reassignment(
    place: dict[str, Any],
    *,
    digest: str | None,
    digest_counts: dict[str, int],
) -> bool:
    place_id = str(place["place_id"])
    if place_id == WINERY_PLACE_ID:
        return False
    if digest is None:
        return True
    if digest_counts.get(digest, 0) > 1:
        return True
    return _is_placeholder(place)


def _pick_page(
    pages: list[dict[str, Any]],
    *,
    kind: str,
    used_photos: set[str],
    rng: random.Random,
) -> tuple[dict[str, Any], str, str] | None:
    scored: list[tuple[int, dict[str, Any]]] = []
    for page in pages:
        title = str(page.get("title") or "").strip()
        extract = " ".join(str(page.get("extract") or "").split())
        thumb = page.get("thumbnail") or {}
        image_url = str(thumb.get("source") or "")
        if not title or not image_url:
            continue
        if any(image_url.casefold().endswith(ext) for ext in (".svg", ".gif")):
            continue
        scored.append((_kind_score(kind, title, extract), page))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("title") or "")))
    candidates = [page for score, page in scored if score > 0] or [
        page for _, page in scored
    ]
    rng.shuffle(candidates)
    for page in candidates:
        title = str(page.get("title") or "").strip()
        image_url = str((page.get("thumbnail") or {}).get("source") or "")
        try:
            raw = _http_get(image_url, timeout=20)
            with Image.open(io.BytesIO(raw)) as image:
                if min(image.size) < 240:
                    continue
        except Exception:  # noqa: BLE001
            continue
        digest = hashlib.sha256(raw).hexdigest()
        if digest in used_photos:
            continue
        return page, image_url, digest
    return None


def assign_map_task_photos(
    *,
    tasks_path: str = "data/tasks",
    selection_path: str | None = SELECTION_PATH,
    dry_run: bool = False,
) -> dict[str, Any]:
    task_root = ROOT / tasks_path
    if selection_path:
        selection = load_selection(ROOT, selection_path)
        task_ids = set(selected_task_ids(selection))
        tasks = [task for task in load_tasks(task_root) if task["id"] in task_ids]
    else:
        tasks = load_tasks(task_root)

    places_path = ROOT / "sites/egomaps/places.json"
    payload = json.loads(places_path.read_text())
    places: list[dict[str, Any]] = payload["places"]
    by_id = {str(place["place_id"]): place for place in places}

    digest_counts: dict[str, int] = {}
    for place in places:
        digest = _photo_digest(str(place.get("photo_path") or ""))
        if digest:
            digest_counts[digest] = digest_counts.get(digest, 0) + 1

    used_photos = set(digest_counts)
    target_ids = _collect_target_ids(tasks)
    rng = random.Random(29)
    report: list[dict[str, Any]] = []
    updated = 0
    failed: list[str] = []

    for place_id in sorted(target_ids):
        target_place = by_id.get(place_id)
        if target_place is None:
            failed.append(place_id)
            report.append(
                {
                    "place_id": place_id,
                    "status": "missing",
                }
            )
            continue
        old_digest = _photo_digest(str(target_place.get("photo_path") or ""))
        if not _needs_reassignment(
            target_place, digest=old_digest, digest_counts=digest_counts
        ):
            report.append(
                {
                    "place_id": place_id,
                    "name": target_place.get("name"),
                    "status": "skipped",
                    "reason": "already_unique",
                    "sha256": old_digest,
                }
            )
            continue

        location = target_place.get("location") or {}
        lat = location.get("latitude", location.get("lat"))
        lon = location.get("longitude", location.get("lng", location.get("lon")))
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            failed.append(place_id)
            report.append(
                {
                    "place_id": place_id,
                    "name": target_place.get("name"),
                    "status": "failed",
                    "reason": "missing_coordinates",
                }
            )
            continue

        kind = str(target_place.get("kind") or "landmark")
        print(f"assign {place_id} {target_place.get('name')} @ {lat},{lon}", flush=True)
        try:
            pages = _wiki_geosearch(float(lat), float(lon))
        except RuntimeError as exc:
            failed.append(place_id)
            report.append(
                {
                    "place_id": place_id,
                    "name": target_place.get("name"),
                    "status": "failed",
                    "reason": f"geosearch: {exc}",
                }
            )
            continue

        picked = _pick_page(pages, kind=kind, used_photos=used_photos, rng=rng)
        if picked is None:
            failed.append(place_id)
            report.append(
                {
                    "place_id": place_id,
                    "name": target_place.get("name"),
                    "status": "failed",
                    "reason": "no_unique_thumbnail",
                }
            )
            continue

        page, image_url, new_digest = picked
        title = str(page.get("title") or "")
        page_url = str(page.get("canonicalurl") or page.get("fullurl") or "")
        relative = f"sites/egomaps/static/photos/{place_id}.jpg"

        if not dry_run:
            raw = _http_get(image_url, timeout=20)
            written = _write_jpeg(relative, raw)
            if written != new_digest:
                new_digest = written
            if old_digest and old_digest in digest_counts:
                digest_counts[old_digest] -= 1
                if digest_counts[old_digest] <= 0:
                    digest_counts.pop(old_digest, None)
                    used_photos.discard(old_digest)
            digest_counts[new_digest] = digest_counts.get(new_digest, 0) + 1
            used_photos.add(new_digest)
            target_place["photo_path"] = relative
            target_place["source"] = {
                "page_url": page_url
                or f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                "image_url": image_url,
                "retrieved_at": NOW,
                "license_or_access_note": (
                    "Wikimedia Commons thumbnail retrieved for a local frozen mock; "
                    "redistribution follows the upstream file license."
                ),
                "sha256": new_digest,
                "access_note": (
                    "Task-referenced map listing photo frozen for local evaluation; "
                    "scored facts remain benchmark-authored."
                ),
            }

        updated += 1
        report.append(
            {
                "place_id": place_id,
                "name": target_place.get("name"),
                "status": "assigned" if not dry_run else "would_assign",
                "wiki_title": title,
                "image_url": image_url,
                "photo_path": relative,
                "old_sha256": old_digest,
                "new_sha256": new_digest,
            }
        )

    if not dry_run and updated:
        payload["places"] = places
        places_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    summary = {
        "updated": updated,
        "failed": failed,
        "target_count": len(target_ids),
        "dry_run": dry_run,
        "entries": report,
    }
    report_path = ROOT / ".scratch/map-photo-assign/report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assign unique Wikimedia photos to CRS task-referenced EgoMaps places."
    )
    parser.add_argument("--tasks", default="data/tasks")
    parser.add_argument("--selection", default=SELECTION_PATH)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selection = None if args.all_tasks else args.selection
    print(
        json.dumps(
            assign_map_task_photos(
                tasks_path=args.tasks,
                selection_path=selection,
                dry_run=args.dry_run,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
