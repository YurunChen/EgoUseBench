"""Mine high-recall HD-EPIC narration cues for natural Refresh events.

Keyword hits are discovery records, not task labels. Every hit still requires
continuous-video review and physical-object association checks before it can
enter a screening manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import re
from pathlib import Path
from typing import Any

from egocua.workspace import resolve_within, write_json_atomic


REFRESH_CUE_PATTERN = re.compile(
    r"(?:change(?:d)? (?:my )?mind|different selection|"
    r"(?:pick(?:ing)?(?: up)?|grab(?:bing)?|select(?:ing)?|use|using|"
    r"retrieve|retrieving) (?:a )?different [a-z]+|"
    r"use (?:this|it|the [^.;]{0,40}) instead|"
    r"instead of (?:the )?other|"
    r"return(?:ing)? [^.;]{0,80}(?:instead|different|another)|"
    r"put(?:ting)? [^.;]{0,80} back [^.;]{0,80}"
    r"(?:instead|different|another))",
    re.IGNORECASE,
)


def has_refresh_cue(narration: str) -> bool:
    """Return whether narration contains a high-recall replacement cue."""

    return REFRESH_CUE_PATTERN.search(narration) is not None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_hits(narrations_path: Path) -> list[dict[str, Any]]:
    """Read official narration cues without creating a report."""
    try:
        pandas = importlib.import_module("pandas")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "pandas is required only for this offline miner; run with "
            "`uv run --with pandas python -m tools.mine.mine_hdepic_refresh ...`"
        ) from error

    frame = pandas.read_pickle(narrations_path)
    required_columns = {
        "video_id",
        "unique_narration_id",
        "start_timestamp",
        "end_timestamp",
        "narration",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"HD-EPIC narration table is missing {sorted(missing)!r}")

    hits: list[dict[str, Any]] = []
    for record in frame.to_dict(orient="records"):
        narration = str(record["narration"])
        if not has_refresh_cue(narration):
            continue
        start, end = float(record["start_timestamp"]), float(record["end_timestamp"])
        if not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end:
            raise ValueError(
                f"Invalid narration interval for {record['unique_narration_id']!r}"
            )
        hits.append(
            {
                "record_id": str(record["unique_narration_id"]),
                "video_id": str(record["video_id"]),
                "start_seconds": start,
                "end_seconds": end,
                "narration": narration,
            }
        )

    hits.sort(
        key=lambda item: (item["video_id"], item["start_seconds"], item["record_id"])
    )
    return hits


def mine(
    narrations_path: Path,
    output_path: Path,
    *,
    source_name: str | None = None,
) -> dict[str, Any]:
    hits = find_hits(narrations_path)
    payload = {
        "schema_version": 1,
        "purpose": (
            "High-recall narration search only. A hit is not a Refresh task and "
            "must not define A/B ground truth without continuous video and "
            "physical-object association review."
        ),
        "source": {
            "path": source_name or narrations_path.as_posix(),
            "sha256": _sha256(narrations_path),
        },
        "query": {
            "regex": REFRESH_CUE_PATTERN.pattern,
            "case_sensitive": False,
        },
        "summary": {
            "hit_rows": len(hits),
            "unique_videos": len({item["video_id"] for item in hits}),
        },
        "hits": hits,
    }
    write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine high-recall HD-EPIC narration cues for Refresh screening."
    )
    parser.add_argument("--narrations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    workspace = Path(args.root).resolve()
    narrations = resolve_within(
        workspace, args.narrations, context="HD-EPIC narrations"
    )
    output = resolve_within(
        workspace,
        args.output,
        context="HD-EPIC mining output",
        must_exist=False,
    )
    result = mine(narrations, output, source_name=Path(args.narrations).as_posix())
    print(
        f"mined {result['summary']['hit_rows']} cue rows from "
        f"{result['summary']['unique_videos']} videos"
    )


if __name__ == "__main__":
    main()
