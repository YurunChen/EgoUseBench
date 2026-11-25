"""Materialize captured Amazon and YouTube result windows into shared Mock catalogs.

The capture files are the source of truth. Stable upstream IDs are deduplicated
site-wide, every public visual is downloaded from its own listing, and no task ID
or answer role enters the entity record or search ranking.

With --task-id, existing entity facts and images stay frozen. Their snapshot paths
are merged, and proposed metadata changes are reported for separate review.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from PIL import Image

from egocua.run_ledger import canonical_sha256
from egocua.workspace import write_json_atomic
from tools.catalog.freeze_catalog_order import sort_frozen
from tools.catalog.capture_amazon_product_details import (
    _product_title,
    validate_product_identity,
)


AMAZON_CAPTURES = Path("data/sources/upstream_search_snapshots/amazon")
ABO_CAPTURES = Path("data/sources/upstream_search_snapshots/abo")
AMAZON_DETAILS = Path("data/sources/upstream_detail_snapshots/amazon")
YOUTUBE_CAPTURES = Path("data/sources/upstream_search_snapshots/youtube")
YOUTUBE_DETAILS = Path("data/sources/upstream_detail_snapshots/youtube")
AMAZON_CATALOG = Path("sites/egomart/products.json")
YOUTUBE_CATALOG = Path("sites/egomedia/videos.json")
ENTITY_INDEX = Path("data/sources/upstream_entity_index.json")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0 Safari/537.36"
)


def _stable_id(prefix: str, namespace: str, upstream_id: str) -> str:
    digest = hashlib.sha256(f"{namespace}:{upstream_id}".encode()).hexdigest()
    return f"{prefix}_{digest[:12]}"


def _capture_paths(directory: Path, task_ids: set[str] | None) -> list[Path]:
    return [
        path
        for path in sorted(directory.glob("*.json"))
        if task_ids is None or path.stem.rsplit("--", 1)[0] in task_ids
    ]


def _capture_rows(
    root: Path,
    directory: Path,
    upstream_key: str,
    *,
    task_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in _capture_paths(root / directory, task_ids):
        payload = json.loads(path.read_text())
        if (
            task_ids is not None
            and payload.get("task_id") != path.stem.rsplit("--", 1)[0]
        ):
            raise ValueError(f"capture task ID disagrees with its filename: {path}")
        relative = path.relative_to(root).as_posix()
        for row in payload.get("results", []):
            upstream_id = str(row.get(upstream_key) or "").strip()
            if not upstream_id:
                continue
            merged = rows.setdefault(upstream_id, {})
            provider = str(payload.get("provider") or "").strip()
            if provider and merged.get("provider") in (None, ""):
                merged["provider"] = provider
            for key, value in row.items():
                if value not in (None, "", []) and merged.get(key) in (None, "", []):
                    merged[key] = value
            snapshots = merged.setdefault("snapshot_paths", [])
            if relative not in snapshots:
                snapshots.append(relative)
            captured = merged.setdefault("captured_at", [])
            timestamp = str(payload.get("captured_at") or "").strip()
            if timestamp and timestamp not in captured:
                captured.append(timestamp)
    return rows


def _amazon_capture_rows(
    root: Path,
    *,
    task_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    rows = _capture_rows(root, AMAZON_CAPTURES, "asin", task_ids=task_ids)
    licensed = _capture_rows(root, ABO_CAPTURES, "asin", task_ids=task_ids)
    for asin, row in licensed.items():
        if asin in rows:
            raise ValueError(f"ASIN appears in both live and ABO captures: {asin}")
        rows[asin] = row
    return rows


def _detail_rows(
    root: Path, directory: Path, upstream_key: str
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not (root / directory).is_dir():
        return rows
    for path in sorted((root / directory).glob("*.json")):
        payload = json.loads(path.read_text())
        upstream_id = str(payload.get(upstream_key) or "").strip()
        if upstream_id:
            rows[upstream_id] = {
                **payload,
                "snapshot_path": path.relative_to(root).as_posix(),
            }
    return rows


def _asin_from_product(product: dict[str, Any]) -> str | None:
    for provenance_key in ("text_provenance", "image_provenance"):
        provenance = product.get(provenance_key)
        if not isinstance(provenance, dict):
            continue
        match = re.search(
            r"/dp/([A-Z0-9]{10})", str(provenance.get("source_url") or "")
        )
        if match:
            return match.group(1)
    return None


def _youtube_id(video: dict[str, Any]) -> str | None:
    source = video.get("source")
    page_url = str(source.get("page_url") or "") if isinstance(source, dict) else ""
    parsed = urlparse(page_url)
    video_id = parse_qs(parsed.query).get("v", [""])[0]
    return video_id or None


def _review_count(value: Any) -> int | None:
    text = str(value or "").strip().replace("(", "").replace(")", "").replace(",", "")
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)([KMB]?)", text, re.IGNORECASE)
    if not match:
        return None
    multiplier = {"": 1, "K": 1_000, "M": 1_000_000, "B": 1_000_000_000}
    return int(float(match.group(1)) * multiplier[match.group(2).upper()])


def _rating(value: Any) -> float | None:
    match = re.search(r"[0-5](?:\.[0-9]+)?", str(value or ""))
    return float(match.group()) if match else None


def _fetch_image(url: str, *, attempts: int = 4) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Encoding": "identity",
                },
            )
            with urlopen(request, timeout=45) as response:
                body = response.read()
            if not body:
                raise ValueError("empty image")
            return body
        except Exception as exc:  # Network failures are retried, then surfaced.
            error = exc
            time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {error}")


def _jpeg(body: bytes, *, maximum_side: int) -> bytes:
    with Image.open(io.BytesIO(body)) as image:
        rgb = image.convert("RGB")
        rgb.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=84, optimize=True)
        return output.getvalue()


def _download_job(job: tuple[str, str, str, int]) -> tuple[str, str, bytes, str]:
    entity_id, relative, url, maximum_side = job
    body = _jpeg(_fetch_image(url), maximum_side=maximum_side)
    return entity_id, relative, body, hashlib.sha256(body).hexdigest()


def _download_images(
    root: Path,
    jobs: Iterable[tuple[str, str, str, int]],
    *,
    concurrency: int,
) -> dict[str, str]:
    materialized = list(jobs)
    digests: dict[str, str] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {pool.submit(_download_job, job): job[0] for job in materialized}
        for completed, future in enumerate(as_completed(pending), start=1):
            entity_id = pending[future]
            try:
                got_id, relative, body, digest = future.result()
            except Exception as exc:
                failures.append(f"{entity_id}: {exc}")
                continue
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(body)
            digests[got_id] = digest
            if completed % 250 == 0 or completed == len(materialized):
                print(f"images {completed}/{len(materialized)}", flush=True)
    if failures:
        raise RuntimeError(
            f"image materialization failed for {len(failures)} entities:\n"
            + "\n".join(failures[:40])
        )
    return digests


def _existing_digest(root: Path, relative: str) -> str | None:
    path = root / relative
    if not path.is_file() or path.stat().st_size == 0:
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _existing_entity_changes(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
    upstream_index: dict[str, str],
) -> list[dict[str, Any]]:
    changes = []
    for upstream_id, entity_id in sorted(upstream_index.items()):
        if entity_id not in before:
            continue
        original = before[entity_id]
        updated = after[entity_id]
        fields = sorted(
            key
            for key in original.keys() | updated.keys()
            if key not in original
            or key not in updated
            or original[key] != updated[key]
        )
        if fields:
            changes.append(
                {
                    "upstream_id": upstream_id,
                    "entity_id": entity_id,
                    "changed_fields": fields,
                }
            )
    return changes


def _merge_snapshot_paths(
    existing: dict[str, Any],
    proposed: dict[str, Any],
    provenance_key: str,
) -> None:
    provenance = existing.setdefault(provenance_key, {})
    previous_paths = provenance.get("snapshot_paths") or []
    if not isinstance(previous_paths, list):
        raise ValueError(f"{provenance_key}.snapshot_paths must be a list")
    provenance["snapshot_paths"] = list(
        dict.fromkeys(
            [
                *previous_paths,
                *proposed[provenance_key]["snapshot_paths"],
            ]
        )
    )


def _amazon_entities(
    root: Path,
    *,
    concurrency: int,
    task_ids: set[str] | None = None,
) -> tuple[dict[str, str], int, list[dict[str, Any]]]:
    captures = _amazon_capture_rows(root, task_ids=task_ids)
    if task_ids is not None and not captures:
        return {}, 0, []
    path = root / AMAZON_CATALOG
    payload = json.loads(path.read_text())
    products: list[dict[str, Any]] = payload["products"]
    by_asin = {
        asin: product
        for product in products
        if (asin := _asin_from_product(product)) is not None
    }
    occupied = {str(product["item_id"]): product for product in products}
    details = _detail_rows(root, AMAZON_DETAILS, "asin")
    original_entities: dict[str, dict[str, Any]] = {}
    proposed_entities: dict[str, dict[str, Any]] = {}
    index: dict[str, str] = {}
    jobs: list[tuple[str, str, str, int]] = []
    for asin, row in sorted(captures.items()):
        licensed_abo = row.get("provider") == "amazon_berkeley_objects"
        detail = {} if licensed_abo else (details.get(asin) or {})
        if detail:
            validate_product_identity(asin, detail)
        title = str(detail.get("title") or "").strip()
        if not _product_title(title):
            title = str(row.get("title") or "").strip()
        if not _product_title(title):
            raise ValueError(
                f"{asin}: no usable product title in detail or search capture"
            )
        source_url = str(
            row.get("product_page_url")
            or row.get("page_url")
            or f"https://www.amazon.ca/dp/{asin}"
        )
        marketplace = str(urlparse(source_url).hostname or "").removeprefix("www.")
        validate_product_identity(asin, {"source_url": source_url})
        product = by_asin.get(asin)
        if product is not None and task_ids is not None:
            original_entities[str(product["item_id"])] = copy.deepcopy(product)
            product = copy.deepcopy(product)
            proposed_entities[str(product["item_id"])] = product
        if product is None:
            entity_id = _stable_id("p", "amazon", asin)
            if entity_id in occupied:
                raise ValueError(f"stable product ID collision: {entity_id}")
            product = {"item_id": entity_id}
            products.append(product)
            occupied[entity_id] = product
        entity_id = str(product["item_id"])
        index[asin] = entity_id
        relative = f"sites/egomart/static/products/upstream/{entity_id}.jpg"
        # Keep the already-reviewed search-result image stable. Product-detail
        # capture enriches decision fields; it must not silently replace the
        # visual evidence that the task was built and reviewed against.
        image_url = str(row["image_url"])
        prior_image = product.get("image_provenance")
        prior_image = prior_image if isinstance(prior_image, dict) else {}
        existing_digest = _existing_digest(root, relative)
        prior_image_url = str(prior_image.get("image_url") or "")
        attributes = (
            {
                "listing_source": "Amazon Berkeley Objects",
                "listing_type": "Licensed catalog record",
                "Brand": str(row.get("brand") or ""),
                "Product type": list(row.get("product_types") or []),
                "Color": str(row.get("color") or ""),
                "Marketplace country": str(row.get("country") or ""),
            }
            if licensed_abo
            else {
                "listing_source": marketplace,
                "listing_type": "Sponsored" if row.get("sponsored") else "Organic",
            }
        )
        attributes = {
            key: value for key, value in attributes.items() if value not in ("", [])
        }
        detail_brand = str(detail.get("brand") or "").strip()
        if detail_brand:
            attributes["Brand"] = detail_brand
        product_details = detail.get("product_details")
        if isinstance(product_details, dict):
            attributes.update(
                {
                    str(key): value
                    for key, value in product_details.items()
                    if str(key).strip()
                    and value not in (None, "", [], {})
                    and str(key).strip(" \t\n\u200e\u200f")
                    not in {
                        "Customer Reviews",
                        "Best Sellers Rank",
                    }
                }
            )
        feature_bullets = detail.get("feature_bullets")
        if isinstance(feature_bullets, list):
            attributes.update(
                {
                    f"feature_{index}": value
                    for index, value in enumerate(feature_bullets[:8], start=1)
                    if str(value).strip()
                }
            )
        snapshot_paths = list(row["snapshot_paths"])
        detail_snapshot = str(detail.get("snapshot_path") or "")
        if detail_snapshot:
            snapshot_paths.append(detail_snapshot)
        detail_url = str(detail.get("page_url") or "")
        same_marketplace = (
            bool(detail_url)
            and urlparse(detail_url).hostname == urlparse(source_url).hostname
        )
        retrieved_at = str(
            detail.get("retrieved_at") or max(row.get("captured_at") or [""])
        )
        product.update(
            {
                "name": title,
                "price": str(
                    (detail.get("price") if same_marketplace else None)
                    or row.get("price")
                    or "Price unavailable"
                ),
                "attributes": attributes,
                "image_path": relative,
                "text_provenance": {
                    "source_url": source_url,
                    **({"detail_source_url": detail_url} if detail_url else {}),
                    "retrieved_at": retrieved_at,
                    "snapshot_paths": snapshot_paths,
                    "access_note": (
                        "Licensed Amazon Berkeley Objects listing metadata frozen for the local Mock; "
                        "CC BY 4.0 attribution is recorded in the capture."
                        if licensed_abo
                        else f"Public {marketplace} search-result and product-detail metadata frozen for the local Mock."
                    ),
                },
                "image_provenance": {
                    "source_url": source_url,
                    "image_url": image_url,
                    "retrieved_at": retrieved_at,
                    "license_or_access_note": (
                        "CC BY 4.0; attribute Amazon.com and the Amazon Berkeley Objects dataset creators."
                        if licensed_abo
                        else "Public listing image retrieved for local benchmark evaluation; redistribution rights are not asserted."
                    ),
                },
            }
        )
        # Amazon search cards do not expose a separate completeness signal for
        # brand. Verified brand remains searchable and visible in Product details.
        if licensed_abo and row.get("brand"):
            product["brand"] = str(row["brand"])
        else:
            product.pop("brand", None)
        rating = _rating(row.get("rating"))
        count = _review_count(row.get("review_count"))
        if rating is not None:
            product["rating"] = rating
        if count is not None:
            product["ratings_count"] = count
        if entity_id in original_entities:
            _merge_snapshot_paths(occupied[entity_id], product, "text_provenance")
            continue
        if existing_digest is None or (
            prior_image_url and prior_image_url != image_url
        ):
            jobs.append((entity_id, relative, image_url, 720))
        else:
            product["image_provenance"]["sha256"] = existing_digest
    digests = _download_images(root, jobs, concurrency=concurrency)
    for product in products:
        digest = digests.get(str(product["item_id"]))
        if digest is not None:
            product["image_provenance"]["sha256"] = digest
    payload["products"] = sort_frozen(products, "item_id")
    if task_ids is None:
        payload["content_origin"] = (
            "Shared frozen Amazon listings materialized from complete captured result windows."
        )
    write_json_atomic(path, payload)
    return (
        index,
        len(captures),
        _existing_entity_changes(original_entities, proposed_entities, index),
    )


def _youtube_entities(
    root: Path,
    *,
    concurrency: int,
    task_ids: set[str] | None = None,
) -> tuple[dict[str, str], int, list[dict[str, Any]]]:
    captures = _capture_rows(root, YOUTUBE_CAPTURES, "youtube_id", task_ids=task_ids)
    if task_ids is not None and not captures:
        return {}, 0, []
    path = root / YOUTUBE_CATALOG
    payload = json.loads(path.read_text())
    videos: list[dict[str, Any]] = payload["videos"]
    by_youtube_id = {
        youtube_id: video
        for video in videos
        if (youtube_id := _youtube_id(video)) is not None
    }
    occupied = {str(video["video_id"]): video for video in videos}
    details = _detail_rows(root, YOUTUBE_DETAILS, "id")
    original_entities: dict[str, dict[str, Any]] = {}
    proposed_entities: dict[str, dict[str, Any]] = {}
    index: dict[str, str] = {}
    jobs: list[tuple[str, str, str, int]] = []
    for youtube_id, row in sorted(captures.items()):
        detail = details.get(youtube_id) or {}
        video = by_youtube_id.get(youtube_id)
        if video is not None and task_ids is not None:
            original_entities[str(video["video_id"])] = copy.deepcopy(video)
            video = copy.deepcopy(video)
            proposed_entities[str(video["video_id"])] = video
        if video is None:
            entity_id = _stable_id("m", "youtube", youtube_id)
            if entity_id in occupied:
                raise ValueError(f"stable video ID collision: {entity_id}")
            video = {"video_id": entity_id}
            videos.append(video)
            occupied[entity_id] = video
        entity_id = str(video["video_id"])
        index[youtube_id] = entity_id
        relative = f"sites/egomedia/static/thumbnails/upstream/{entity_id}.jpg"
        thumbnail_url = str(row["thumbnail_url"])
        prior_source = video.get("source")
        prior_source = prior_source if isinstance(prior_source, dict) else {}
        existing_digest = _existing_digest(root, relative)
        provenance_thumbnail_url = str(
            prior_source.get("thumbnail_url") or thumbnail_url
        )
        metadata = {
            "title": str(detail.get("title") or row["title"]),
            "channel_title": str(
                detail.get("channel_title") or row.get("channel_title") or "YouTube"
            ),
            "duration_seconds": int(
                detail.get("duration_seconds")
                or row.get("duration_seconds")
                or (video.get("duration_seconds") if task_ids is not None else 0)
                or 0
            ),
            "view_count": int(
                detail.get("view_count")
                or row.get("view_count")
                or (video.get("view_count") if task_ids is not None else 0)
                or 0
            ),
            "published_at": str(
                detail.get("published_at") or "Not shown in captured results"
            ),
            "summary": str(detail.get("public_summary") or row["title"]),
            "chapters": list(detail.get("public_chapters") or []),
            "snapshot_paths": row["snapshot_paths"],
        }
        video.update(
            {
                "title": metadata["title"],
                "kind": "video",
                "summary": metadata["summary"],
                "thumbnail_path": relative,
                "channel_title": metadata["channel_title"],
                "published_at": metadata["published_at"],
                "duration_seconds": metadata["duration_seconds"],
                "view_count": metadata["view_count"],
                "details": dict(video.get("details") or {"format": "video"}),
                "chapters": metadata["chapters"],
                "facts": dict(video.get("facts") or {}),
                "source": {
                    "record": "youtube.com search result",
                    "page_url": f"https://www.youtube.com/watch?v={youtube_id}",
                    "thumbnail_url": provenance_thumbnail_url,
                    "retrieved_at": max(row.get("captured_at") or [""]),
                    "metadata_retrieved_at": str(
                        detail.get("retrieved_at")
                        or max(row.get("captured_at") or [""])
                    ),
                    "metadata_sha256": canonical_sha256(detail or metadata),
                    "metadata_snapshot_basis": (
                        "frozen_youtube_detail"
                        if detail
                        else "captured_complete_search_window"
                    ),
                    "snapshot_paths": [
                        *row["snapshot_paths"],
                        *(
                            [str(detail["snapshot_path"])]
                            if detail.get("snapshot_path")
                            else []
                        ),
                    ],
                    "license_or_access_note": (
                        "Public YouTube thumbnail and result metadata retrieved for local "
                        "benchmark evaluation; redistribution rights are not asserted."
                    ),
                },
            }
        )
        if entity_id in original_entities:
            _merge_snapshot_paths(occupied[entity_id], video, "source")
            continue
        if existing_digest is None:
            jobs.append((entity_id, relative, thumbnail_url, 640))
        else:
            video["source"]["thumbnail_sha256"] = existing_digest
    digests = _download_images(root, jobs, concurrency=concurrency)
    for video in videos:
        digest = digests.get(str(video["video_id"]))
        if digest is not None:
            video["source"]["thumbnail_sha256"] = digest
    payload["videos"] = sort_frozen(videos, "video_id")
    if task_ids is None:
        payload["content_origin"] = (
            "Shared frozen YouTube listings materialized from complete captured result windows."
        )
    write_json_atomic(path, payload)
    return (
        index,
        len(captures),
        _existing_entity_changes(original_entities, proposed_entities, index),
    )


def materialize(
    root: Path,
    *,
    site: str,
    concurrency: int = 12,
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if task_ids is not None:
        capture_directories = {
            "egomart": (AMAZON_CAPTURES, ABO_CAPTURES),
            "egomedia": YOUTUBE_CAPTURES,
        }
        found_ids = {
            path.stem.rsplit("--", 1)[0]
            for capture_site, directories in capture_directories.items()
            if site in {"all", capture_site}
            for directory in (
                directories if isinstance(directories, tuple) else (directories,)
            )
            for path in _capture_paths(root / directory, task_ids)
        }
        if missing_ids := task_ids - found_ids:
            raise ValueError(
                f"no captures for requested task IDs: {sorted(missing_ids)}"
            )
        output["requested_task_ids"] = sorted(task_ids)
        output["pending_existing_entity_updates"] = []
    index: dict[str, Any] = {"schema_version": 1, "sites": {}}
    if (root / ENTITY_INDEX).is_file():
        index = json.loads((root / ENTITY_INDEX).read_text())
    if site in {"all", "egomart"}:
        amazon, count, changed = _amazon_entities(
            root, concurrency=concurrency, task_ids=task_ids
        )
        if task_ids is None:
            index["sites"]["egomart"] = amazon
        else:
            if amazon:
                index["sites"].setdefault("egomart", {}).update(amazon)
            output["pending_existing_entity_updates"].extend(
                {"site": "egomart", **row} for row in changed
            )
        output["egomart_entities"] = count
    if site in {"all", "egomedia"}:
        youtube, count, changed = _youtube_entities(
            root, concurrency=concurrency, task_ids=task_ids
        )
        if task_ids is None:
            index["sites"]["egomedia"] = youtube
        else:
            if youtube:
                index["sites"].setdefault("egomedia", {}).update(youtube)
            output["pending_existing_entity_updates"].extend(
                {"site": "egomedia", **row} for row in changed
            )
        output["egomedia_entities"] = count
    write_json_atomic(root / ENTITY_INDEX, index)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--site", choices=("all", "egomart", "egomedia"), default="all")
    parser.add_argument("--concurrency", type=int, choices=range(1, 17), default=12)
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Materialize only this task's captures; repeat for multiple tasks.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                Path(args.root).resolve(),
                site=args.site,
                concurrency=args.concurrency,
                task_ids=set(args.task_id) if args.task_id else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
