"""Freeze public YouTube descriptions and chapters for EgoMedia listings.

Only upstream metadata is copied. The script never invents tutorial steps or
benchmark-specific facts. One persistent snapshot per YouTube ID is reused by
every task that references the entity.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from egocua.schema import load_tasks
from egocua.sources import task_detail_entity_ids
from egocua.workspace import write_json_atomic
from tools.catalog.catalog_clients import yt_video_json

CATALOG = Path("sites/egomedia/videos.json")
DETAIL_DIR = Path("data/sources/upstream_detail_snapshots/youtube")
CAPTION_BLOCK_SECONDS = 30
COMMENT_FIELDS = {
    "id": "id",
    "author": "author",
    "text": "text",
    "like_count": "like_count",
    "_time_text": "published",
    "is_pinned": "pinned",
    "author_is_uploader": "author_is_uploader",
}


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _youtube_id(entity: dict[str, Any]) -> str:
    source = entity.get("source")
    page_url = source.get("page_url") if isinstance(source, dict) else None
    if not isinstance(page_url, str):
        raise ValueError(f"{entity.get('video_id')}: canonical YouTube URL missing")
    parsed = urlparse(page_url)
    video_id = parse_qs(parsed.query).get("v", [""])[0]
    if parsed.netloc not in {"youtube.com", "www.youtube.com"} or not video_id:
        raise ValueError(f"{entity.get('video_id')}: canonical YouTube URL missing")
    return video_id


def _summary(description: Any, title: str, *, limit: int = 1200) -> str:
    normalized = " ".join(str(description or "").split()) or title
    if len(normalized) <= limit:
        return normalized
    clipped = normalized[: limit + 1].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return clipped + "…"


def _chapters(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = payload.get("chapters")
    if not isinstance(rows, list):
        return []
    chapters: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("title") or "").strip():
            continue
        start = row.get("start_time")
        end = row.get("end_time")
        if not isinstance(start, (int, float)) or isinstance(start, bool):
            continue
        start_text = _clock(int(start))
        end_text = _clock(int(end)) if isinstance(end, (int, float)) else None
        chapters.append(
            {
                "heading": str(row["title"]).strip(),
                "text": f"{start_text}–{end_text}"
                if end_text
                else f"Starts at {start_text}",
            }
        )
    return chapters


def _clock(seconds: int) -> str:
    hours, remainder = divmod(max(0, seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"
    )


def _public_comments(comments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            target: comment[source]
            for source, target in COMMENT_FIELDS.items()
            if source in comment
        }
        for comment in comments
    ]


def _projection(payload: dict[str, Any]) -> dict[str, Any]:
    upload = str(payload.get("upload_date") or "")
    published = (
        f"{upload[:4]}-{upload[4:6]}-{upload[6:8]}" if len(upload) == 8 else None
    )
    views = payload.get("view_count")
    duration = payload.get("duration")
    projected: dict[str, Any] = {
        "schema_version": 1,
        "provider": "youtube-video-detail-v1",
        "id": str(payload.get("id") or ""),
        "retrieved_at": _utc_now(),
        "title": str(payload.get("title") or "").strip(),
        "description": str(payload.get("description") or ""),
        "channel_title": str(
            payload.get("uploader") or payload.get("channel") or ""
        ).strip(),
        "published_at": published,
        "duration_seconds": int(duration)
        if isinstance(duration, (int, float))
        else None,
        "view_count": int(views)
        if isinstance(views, (int, float)) and views >= 0
        else None,
        "chapters": payload.get("chapters")
        if isinstance(payload.get("chapters"), list)
        else [],
    }
    projected["public_summary"] = _summary(
        projected["description"],
        str(projected["title"]),
    )
    projected["public_chapters"] = _chapters(projected)
    if "comments" in payload:
        projected["comments"] = payload["comments"]
        projected["public_comments"] = _public_comments(payload["comments"])
    if "comment_count" in payload:
        projected["comment_count"] = payload["comment_count"]
    return projected


def _normalize_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["schema_version"] = 1
    normalized["provider"] = "youtube-video-detail-v1"
    normalized["public_summary"] = _summary(
        normalized.get("description"),
        str(normalized.get("title") or ""),
    )
    normalized["public_chapters"] = _chapters(normalized) + _public_caption_chapters(
        normalized
    )
    if "comments" in normalized:
        normalized["public_comments"] = _public_comments(normalized["comments"])
    else:
        normalized.pop("public_comments", None)
    return normalized


def _caption_seconds(timestamp: str) -> float:
    if re.fullmatch(r"(?:\d+:)?\d+:\d{2}(?:\.\d+)?", timestamp) is None:
        raise ValueError(f"invalid public caption timestamp: {timestamp!r}")
    parts = timestamp.split(":")
    if float(parts[-1]) >= 60 or (len(parts) == 3 and int(parts[-2]) >= 60):
        raise ValueError(f"invalid public caption timestamp: {timestamp!r}")
    return sum(float(part) * 60**index for index, part in enumerate(reversed(parts)))


def _vtt_segments(raw: str) -> list[dict[str, str]]:
    if not raw.startswith("WEBVTT"):
        raise ValueError("public captions require an original WEBVTT response")
    segments = []
    for block in re.split(r"\r?\n\r?\n", raw):
        lines = block.splitlines()
        time_index = next(
            (index for index, line in enumerate(lines) if " --> " in line), None
        )
        if time_index is None:
            continue
        start, end = lines[time_index].split(" --> ", 1)
        end = end.split()[0]
        if _caption_seconds(end) < _caption_seconds(start):
            raise ValueError("public caption end precedes its start")
        words = html.unescape(re.sub(r"<[^>]+>", "", " ".join(lines[time_index + 1 :])))
        text = " ".join(words.split())
        if text:
            segments.append({"start": start, "end": end, "text": text})
    return segments


def _browser_caption_segments(raw: str) -> list[dict[str, str]]:
    return [
        {"start": match.group(1), "text": match.group(2)}
        for line in raw.splitlines()
        if (match := re.fullmatch(r"\[((?:\d+:)?\d+:\d{2})\] (.+)", line))
    ]


def _original_caption_segments(payload: dict[str, Any]) -> list[dict[str, str]]:
    captions = payload["public_captions"]
    if not isinstance(captions, dict) or captions.get("kind") not in {
        "automatic",
        "manual",
        "auto-generated",
        "unknown",
    }:
        raise ValueError("public captions require a captured platform caption kind")
    raw_fields = [key for key in ("raw_vtt", "raw_transcript") if key in captions]
    if len(raw_fields) != 1:
        raise ValueError("public captions require one original caption response")
    raw_key = raw_fields[0]
    raw = captions[raw_key]
    if not isinstance(raw, str) or hashlib.sha256(
        raw.encode()
    ).hexdigest() != captions.get(f"{raw_key}_sha256"):
        raise ValueError("public caption response hash mismatch")
    source = urlparse(str(captions.get("source_url") or ""))
    # Cite the owning video; an expiring caption response URL is optional evidence.
    if (
        source.scheme != "https"
        or source.hostname not in {"youtube.com", "www.youtube.com"}
        or source.path != "/watch"
        or parse_qs(source.query).get("v") != [payload.get("id")]
        or not captions.get("retrieved_at")
    ):
        raise ValueError(
            "public caption source does not match its canonical YouTube video"
        )
    segments = (
        _vtt_segments(raw) if raw_key == "raw_vtt" else _browser_caption_segments(raw)
    )
    if not segments:
        raise ValueError("original public caption response contains no timed text")
    return segments


def _public_caption_chapters(payload: dict[str, Any]) -> list[dict[str, str]]:
    if "public_captions" not in payload:
        return []
    segments = _original_caption_segments(payload)
    prefix = (
        "Auto-generated public transcript"
        if payload["public_captions"]["kind"] in {"automatic", "auto-generated"}
        else "Public transcript"
    )
    blocks: list[dict[str, str]] = []
    previous: list[str] = []
    previous_time = -1.0
    block_number = -1
    for segment in segments:
        start = _caption_seconds(segment["start"])
        if start < previous_time:
            raise ValueError("public captions are not in timestamp order")
        previous_time = start
        current = segment["text"].split()
        if payload["public_captions"]["kind"] not in {"automatic", "auto-generated"}:
            previous = []
        overlap = next(
            (
                count
                for count in range(min(len(previous), len(current)), 0, -1)
                if previous[-count:] == current[:count]
            ),
            0,
        )
        added, previous = current[overlap:], current
        if not added:
            continue
        current_block = int(start // CAPTION_BLOCK_SECONDS)
        if current_block != block_number:
            blocks.append(
                {"heading": f"{prefix} — {_clock(int(start))}", "text": " ".join(added)}
            )
            block_number = current_block
        else:
            blocks[-1]["text"] += " " + " ".join(added)
    return blocks


def _public_details(entity: dict[str, Any]) -> dict[str, Any]:
    """Keep neutral display metadata; procedure claims come from upstream text."""

    details = dict(entity.get("details") or {})
    details.pop("covered_steps", None)
    details["format"] = "video"
    return details


def _current_task_ids(root: Path) -> set[str]:
    return {
        entity_id
        for task in load_tasks(root / "data/tasks")
        if task["site"] == "egomedia"
        for entity_id in task_detail_entity_ids(task, root=root)
    }


def import_captions(root: Path, entity_id: str, caption_path: Path) -> dict[str, int]:
    """Attach one captured native VTT to its owning video and expose its full text."""
    catalog = json.loads((root / CATALOG).read_text())
    entity = next(row for row in catalog["videos"] if row["video_id"] == entity_id)
    youtube_id = _youtube_id(entity)
    if caption_path.name != f"egocua-public-{youtube_id}.en.vtt":
        raise ValueError("caption filename must identify the owning video and language")
    snapshot_path = root / DETAIL_DIR / f"{youtube_id}.json"
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot["id"] != youtube_id:
        raise ValueError("caption snapshot belongs to a different video")
    raw = caption_path.read_text()
    snapshot["public_captions"] = {
        "kind": "unknown",
        "language": "en",
        "source_url": entity["source"]["page_url"],
        "retrieved_at": _utc_now(),
        "raw_vtt": raw,
        "raw_vtt_sha256": hashlib.sha256(raw.encode()).hexdigest(),
    }
    normalized = _normalize_snapshot(snapshot)
    write_json_atomic(snapshot_path, normalized)
    return freeze(root, {entity_id}, fetch_missing=False)


def freeze(
    root: Path,
    entity_ids: set[str] | None,
    *,
    concurrency: int = 4,
    fetch_missing: bool = True,
) -> dict[str, int]:
    if not 1 <= concurrency <= 8:
        raise ValueError("concurrency must be between 1 and 8")
    if entity_ids is not None and not entity_ids:
        raise ValueError(
            "explicit entity_ids is empty; refusing to expand to all tasks"
        )
    catalog_path = root / CATALOG
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    requested = _current_task_ids(root) if entity_ids is None else entity_ids
    entities = {str(entity["video_id"]): entity for entity in catalog["videos"]}
    missing = sorted(requested - set(entities))
    if missing:
        raise ValueError(f"EgoMedia catalog is missing {missing}")

    youtube_ids = {
        entity_id: _youtube_id(entities[entity_id]) for entity_id in sorted(requested)
    }
    detail_dir = root / DETAIL_DIR
    detail_dir.mkdir(parents=True, exist_ok=True)
    details: dict[str, dict[str, Any]] = {}
    for youtube_id in sorted(set(youtube_ids.values())):
        detail_path = detail_dir / f"{youtube_id}.json"
        if detail_path.is_file():
            payload = _normalize_snapshot(json.loads(detail_path.read_text()))
            write_json_atomic(detail_path, payload)
            details[youtube_id] = payload
            continue

    reused = len(details)
    missing_before_fetch = sorted(set(youtube_ids.values()) - set(details))
    pending = missing_before_fetch if fetch_missing else []
    failed: list[str] = []
    fetched = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(yt_video_json, youtube_id): youtube_id for youtube_id in pending
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            youtube_id = futures[future]
            try:
                payload = _projection(future.result())
                if payload.get("id") != youtube_id or not payload.get("title"):
                    raise ValueError("unusable upstream YouTube metadata")
                write_json_atomic(detail_dir / f"{youtube_id}.json", payload)
                details[youtube_id] = payload
                fetched += 1
            except Exception as exc:  # Fetch failures are collected, then surfaced.
                failed.append(f"{youtube_id}: {type(exc).__name__}: {exc}")
            if completed % 25 == 0 or completed == len(pending):
                print(f"YouTube details {completed}/{len(pending)}", flush=True)
    for entity_id, youtube_id in youtube_ids.items():
        entity = entities[entity_id]
        detail_payload = details.get(youtube_id)
        if detail_payload is None:
            continue
        if detail_payload.get("id") != youtube_id or not detail_payload.get("title"):
            raise ValueError(f"{entity_id}: unusable upstream YouTube metadata")
        entity["title"] = detail_payload["title"]
        entity["summary"] = detail_payload["public_summary"]
        entity["description"] = str(detail_payload.get("description") or "")
        if detail_payload.get("channel_title"):
            entity["channel_title"] = detail_payload["channel_title"]
        for source_key, target_key in (
            ("published_at", "published_at"),
            ("duration_seconds", "duration_seconds"),
            ("view_count", "view_count"),
        ):
            if detail_payload.get(source_key) is not None:
                entity[target_key] = detail_payload[source_key]
        entity["chapters"] = detail_payload["public_chapters"]
        if "public_previews" in detail_payload:
            entity["public_previews"] = detail_payload["public_previews"]
        else:
            entity.pop("public_previews", None)
        if "public_comments" in detail_payload:
            entity["comments"] = detail_payload["public_comments"]
        else:
            entity.pop("comments", None)
        if "comment_count" in detail_payload:
            entity["comment_count"] = detail_payload["comment_count"]
        else:
            entity.pop("comment_count", None)
        entity["details"] = _public_details(entity)
        if "linked_manufacturer_instructions" in detail_payload:
            entity["details"]["linked_manufacturer_instructions"] = detail_payload[
                "linked_manufacturer_instructions"
            ]
        else:
            entity["details"].pop("linked_manufacturer_instructions", None)
        source = dict(entity["source"])
        source["metadata_retrieved_at"] = (
            detail_payload.get("retrieved_at") or _utc_now()
        )
        source["metadata_sha256"] = hashlib.sha256(
            json.dumps(
                detail_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        source["metadata_snapshot_basis"] = "frozen_youtube_detail"
        source["detail_snapshot_path"] = (DETAIL_DIR / f"{youtube_id}.json").as_posix()
        entity["source"] = source

    catalog["content_origin"] = (
        "Frozen public YouTube listings and thumbnails. Descriptions and chapters are "
        "shown only when a canonical upstream detail snapshot exists."
    )
    write_json_atomic(catalog_path, catalog)
    result = {
        "requested": len(requested),
        "reused": reused,
        "fetched": fetched,
        "updated": len(details),
        "missing": len(set(youtube_ids.values()) - set(details)),
    }
    if failed:
        raise RuntimeError(
            f"Applied {len(details)} available snapshots, then YouTube detail capture "
            f"failed for {len(failed)} videos:\n" + "\n".join(failed[:30])
        )
    return result


def backfill_existing(root: Path) -> dict[str, int]:
    """Register provenance for already-frozen upstream metadata without refetching it.

    Older catalog rows were captured from YouTube together with their thumbnails,
    but predate the explicit metadata digest fields.  The digest below covers the
    public fields already stored in the frozen row and never authors new content.
    """

    catalog_path = root / CATALOG
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    updated = 0
    for entity in catalog["videos"]:
        source = entity.get("source")
        if not isinstance(source, dict) or source.get("metadata_sha256"):
            continue
        projection = {
            "title": entity.get("title"),
            "summary": entity.get("summary"),
            "channel_title": entity.get("channel_title"),
            "published_at": entity.get("published_at"),
            "duration_seconds": entity.get("duration_seconds"),
            "view_count": entity.get("view_count"),
            "chapters": entity.get("chapters", []),
        }
        source["metadata_retrieved_at"] = source.get("retrieved_at") or _utc_now()
        source["metadata_sha256"] = hashlib.sha256(
            json.dumps(
                projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        source["metadata_snapshot_basis"] = "existing_frozen_youtube_listing"
        updated += 1
    write_json_atomic(catalog_path, catalog)
    return {"requested": len(catalog["videos"]), "fetched": 0, "updated": updated}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--entity-id", action="append")
    parser.add_argument("--caption-file", type=Path)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--apply-only",
        action="store_true",
        help="Apply persisted snapshots without requesting missing videos.",
    )
    parser.add_argument("--backfill-existing", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if args.caption_file:
        if not args.entity_id or len(args.entity_id) != 1 or args.backfill_existing:
            parser.error("--caption-file requires exactly one --entity-id")
        print(json.dumps(import_captions(root, args.entity_id[0], args.caption_file)))
        return 0
    result = (
        backfill_existing(root)
        if args.backfill_existing
        else freeze(
            root,
            set(args.entity_id) if args.entity_id else None,
            concurrency=args.concurrency,
            fetch_missing=not args.apply_only,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
