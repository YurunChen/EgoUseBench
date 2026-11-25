"""Remove catalog claims that cannot be traced to public upstream metadata.

Older synthesis code inferred tutorial steps from titles and summaries, then
rendered those in EgoMedia as if YouTube had supplied them.  This sanitizer is
deliberately conservative: it keeps timestamp-shaped chapters or exact chapter projections verified
against explicitly cited upstream captures and replaces task-authored detail fields with neutral
display metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from egocua.workspace import write_json_atomic

CATALOG = Path("sites/egomedia/videos.json")
TIMESTAMP = re.compile(
    r"^(?:\d+:)?\d{1,2}:\d{2}(?:–(?:\d+:)?\d{1,2}:\d{2})?$"
    r"|^Starts at (?:\d+:)?\d{1,2}:\d{2}$"
)


def _upstream_chapters(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(row, dict)
            and bool(str(row.get("heading") or "").strip())
            and TIMESTAMP.fullmatch(str(row.get("text") or "")) is not None
            for row in value
        )
    )


def _captured_chapters(entity: dict[str, Any], root: Path) -> bool:
    """Verify native chapter data against the entity's explicitly cited capture."""
    source = entity.get("source", {})
    if not source.get("page_url"):
        return False
    for relative in source.get("detail_snapshot_paths", []):
        path = (root / relative).resolve()
        path.relative_to(root.resolve())
        capture = json.loads(path.read_text())
        captured = [
            {"title": row["title"], "start_seconds": row["start_time"]}
            for row in capture.get("chapters", [])
        ]
        if capture.get("url") == source["page_url"] and captured == entity.get(
            "chapters"
        ):
            return True
    return False


def _projection(entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": entity.get("title"),
        "summary": entity.get("summary"),
        "channel_title": entity.get("channel_title"),
        "published_at": entity.get("published_at"),
        "duration_seconds": entity.get("duration_seconds"),
        "view_count": entity.get("view_count"),
        "chapters": entity.get("chapters", []),
    }


def _digest(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sanitize(root: Path) -> dict[str, int]:
    path = root / CATALOG
    payload = json.loads(path.read_text(encoding="utf-8"))
    removed_claims = 0
    removed_authored_chapters = 0
    refreshed_digests = 0

    for entity in payload["videos"]:
        details = entity.get("details")
        clean_details = dict(details) if isinstance(details, dict) else {}
        if clean_details.pop("covered_steps", None) is not None:
            removed_claims += 1
        clean_details["format"] = "video"
        entity["details"] = clean_details

        chapters = entity.get("chapters")
        if (
            chapters
            and not _upstream_chapters(chapters)
            and not _captured_chapters(entity, root)
        ):
            entity["chapters"] = []
            removed_authored_chapters += 1

        facts = entity.get("facts")
        if isinstance(facts, dict) and facts.pop("covered_steps", None) is not None:
            removed_claims += 1

        source = entity.get("source")
        if (
            isinstance(source, dict)
            and source.get("metadata_snapshot_basis")
            == "existing_frozen_youtube_listing"
        ):
            source["metadata_sha256"] = _digest(_projection(entity))
            source["metadata_snapshot_basis"] = (
                "sanitized_existing_frozen_youtube_listing"
            )
            refreshed_digests += 1

    payload["content_origin"] = (
        "Frozen public YouTube listings and thumbnails. Descriptions and "
        "timestamped chapters are retained only when present in upstream metadata; "
        "benchmark-authored procedure claims are not exposed."
    )
    payload["supported_fields"] = []
    write_json_atomic(path, payload)
    return {
        "entities": len(payload["videos"]),
        "removed_unsupported_claims": removed_claims,
        "removed_authored_chapters": removed_authored_chapters,
        "refreshed_existing_snapshot_digests": refreshed_digests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(sanitize(Path(args.root).resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
