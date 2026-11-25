from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable
from itertools import pairwise
from pathlib import Path
from typing import Any

from egocua.workspace import resolve_within

MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png"}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ProvenanceError(ValueError):
    """Raised when extracted frames cannot be reproduced from their sidecars."""


def _record_path(record: dict[str, Any], context: str) -> str:
    frame_path = record.get("frame_path")
    if not isinstance(frame_path, str) or not frame_path.strip():
        raise ProvenanceError(f"{context}: frame_path must be a non-empty string")
    return frame_path


def _validate_record(record: dict[str, Any], context: str) -> None:
    for field in ("task_id", "packet_id", "media_id"):
        if not str(record.get(field, "")).strip():
            raise ProvenanceError(f"{context}: {field} must be non-empty")
    if not isinstance(record.get("frame_index"), int) or record["frame_index"] < 0:
        raise ProvenanceError(f"{context}: frame_index must be a non-negative integer")
    timestamp = record.get("timestamp_seconds")
    if not isinstance(timestamp, (int, float)) or timestamp < 0:
        raise ProvenanceError(f"{context}: timestamp_seconds must be non-negative")
    media_timestamp = record.get("media_timestamp_seconds")
    if not isinstance(media_timestamp, (int, float)) or media_timestamp < 0:
        raise ProvenanceError(
            f"{context}: media_timestamp_seconds must be non-negative"
        )
    digest = record.get("sha256")
    if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
        raise ProvenanceError(
            f"{context}: sha256 must be 64 lowercase hexadecimal characters"
        )


def _validate_media_mapping(payload: dict[str, Any], context: str) -> None:
    videos = payload.get("videos")
    frames = payload.get("frames", [])
    if not isinstance(videos, dict) or not videos:
        raise ProvenanceError(
            f"{context}: top-level videos must be a non-empty mapping"
        )

    for index, record in enumerate(frames):
        record_context = f"{context}/frames/{index}"
        media_id = str(record["media_id"])
        metadata = videos.get(media_id)
        if not isinstance(metadata, dict):
            raise ProvenanceError(f"{record_context}: unknown media_id {media_id!r}")
        duration = metadata.get("duration_seconds")
        source_start = metadata.get("source_start_seconds")
        if not isinstance(duration, (int, float)) or duration <= 0:
            raise ProvenanceError(f"{context}: {media_id} has invalid duration_seconds")
        if not isinstance(source_start, (int, float)) or source_start < 0:
            raise ProvenanceError(
                f"{context}: {media_id} has invalid source_start_seconds"
            )
        media_timestamp = float(record["media_timestamp_seconds"])
        source_timestamp = float(record["timestamp_seconds"])
        if media_timestamp >= float(duration):
            raise ProvenanceError(
                f"{record_context}: media timestamp is outside the media duration"
            )
        if not math.isclose(
            source_timestamp,
            float(source_start) + media_timestamp,
            abs_tol=1e-6,
        ):
            raise ProvenanceError(
                f"{record_context}: source/media timestamp offset does not match video metadata"
            )


def _validate_sampling_summaries(payload: dict[str, Any], context: str) -> None:
    summaries = payload.get("packet_sampling")
    frames = payload.get("frames", [])
    if not isinstance(summaries, list):
        raise ProvenanceError(f"{context}: top-level packet_sampling must be a list")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in frames:
        key = (str(record["task_id"]), str(record["packet_id"]))
        grouped.setdefault(key, []).append(record)

    declared: dict[tuple[str, str], dict[str, Any]] = {}
    for index, summary in enumerate(summaries):
        summary_context = f"{context}/packet_sampling/{index}"
        if not isinstance(summary, dict):
            raise ProvenanceError(f"{summary_context}: summary must be an object")
        key = (str(summary.get("task_id", "")), str(summary.get("packet_id", "")))
        if not all(key) or key in declared:
            raise ProvenanceError(
                f"{summary_context}: task_id/packet_id must be unique"
            )
        if summary.get("sampling_method") not in {
            "uniform_inclusive_endpoints",
            "event_aligned_timestamps",
        }:
            raise ProvenanceError(f"{summary_context}: unsupported sampling_method")
        declared[key] = summary

    if set(declared) != set(grouped):
        raise ProvenanceError(
            f"{context}: packet_sampling coverage must match frame packets exactly"
        )

    for key, records in grouped.items():
        summary = declared[key]
        ordered = sorted(records, key=lambda record: record["frame_index"])
        if [record["frame_index"] for record in ordered] != list(range(len(ordered))):
            raise ProvenanceError(f"{context}: {key} frame indexes must be contiguous")
        timestamps = [float(record["timestamp_seconds"]) for record in ordered]
        if any(current <= previous for previous, current in pairwise(timestamps)):
            raise ProvenanceError(
                f"{context}: {key} timestamps must be strictly increasing"
            )
        if summary.get("frame_count") != len(ordered):
            raise ProvenanceError(f"{context}: {key} frame_count does not match frames")
        for field, expected in (
            ("start_seconds", timestamps[0]),
            ("end_seconds", timestamps[-1]),
        ):
            value = summary.get(field)
            if not isinstance(value, (int, float)) or not math.isclose(
                float(value), expected, abs_tol=1e-6
            ):
                raise ProvenanceError(f"{context}: {key} {field} does not match frames")

        method = summary["sampling_method"]
        if len(timestamps) == 1:
            if (
                summary.get("inter_frame_seconds") is not None
                or summary.get("effective_hz") is not None
            ):
                raise ProvenanceError(
                    f"{context}: {key} one-frame sampling rates must be null"
                )
            continue
        interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        uniform = not any(
            not math.isclose(current - previous, interval, rel_tol=1e-4, abs_tol=1e-6)
            for previous, current in pairwise(timestamps)
        )
        if method == "event_aligned_timestamps":
            if uniform:
                raise ProvenanceError(
                    f"{context}: {key} event-aligned timestamps are actually uniform"
                )
            if "inter_frame_seconds" in summary or "effective_hz" in summary:
                raise ProvenanceError(
                    f"{context}: {key} event-aligned sampling must not declare one rate"
                )
            continue
        if not uniform:
            raise ProvenanceError(
                f"{context}: {key} timestamps are not uniformly spaced"
            )
        for field, expected in (
            ("inter_frame_seconds", round(interval, 6)),
            ("effective_hz", round(1 / interval, 6)),
        ):
            value = summary.get(field)
            if not isinstance(value, (int, float)) or not math.isclose(
                float(value), expected, abs_tol=1e-6
            ):
                raise ProvenanceError(f"{context}: {key} {field} does not match frames")


def validate_provenance(
    sidecars: Iterable[str | Path],
    *,
    root: str | Path = ".",
    coverage_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    materialized_sidecars = list(sidecars)
    if not materialized_sidecars:
        raise ProvenanceError("at least one provenance sidecar is required")

    seen_paths: dict[str, tuple[Any, ...]] = {}
    seen_slots: set[tuple[str, str, int]] = set()
    for sidecar_name in materialized_sidecars:
        try:
            sidecar_path = resolve_within(
                workspace, sidecar_name, context="provenance sidecar"
            )
        except (ValueError, FileNotFoundError) as exc:
            raise ProvenanceError(str(exc)) from exc
        try:
            payload = json.loads(sidecar_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            raise ProvenanceError(f"{sidecar_name}: cannot read JSON: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 2:
            raise ProvenanceError(
                f"{sidecar_name}: provenance schema_version must be 2"
            )
        frames = payload.get("frames")
        if not isinstance(frames, list):
            raise ProvenanceError(f"{sidecar_name}: top-level frames must be a list")

        for index, record in enumerate(frames):
            context = f"{sidecar_name}/frames/{index}"
            if not isinstance(record, dict):
                raise ProvenanceError(f"{context}: frame record must be an object")
            _validate_record(record, context)
            slot = (record["task_id"], record["packet_id"], record["frame_index"])
            if slot in seen_slots:
                raise ProvenanceError(f"{context}: duplicate frame slot {slot!r}")
            seen_slots.add(slot)
            frame_name = _record_path(record, context)
            source = tuple(
                record[field]
                for field in (
                    "media_id",
                    "timestamp_seconds",
                    "media_timestamp_seconds",
                    "sha256",
                )
            )
            if frame_name in seen_paths:
                if seen_paths[frame_name] != source:
                    raise ProvenanceError(
                        f"{context}: conflicting source for shared frame {frame_name!r}"
                    )
                continue
            seen_paths[frame_name] = source
            try:
                frame_path = resolve_within(workspace, frame_name, context=context)
            except (ValueError, FileNotFoundError) as exc:
                raise ProvenanceError(str(exc)) from exc
            actual_digest = hashlib.sha256(frame_path.read_bytes()).hexdigest()
            if actual_digest != record["sha256"]:
                raise ProvenanceError(
                    f"{context}: SHA-256 mismatch for {frame_name}; "
                    f"expected {record['sha256']}, got {actual_digest}"
                )
        _validate_media_mapping(payload, str(sidecar_name))
        _validate_sampling_summaries(payload, str(sidecar_name))

    coverage_files: set[str] | None = None
    if coverage_root is not None:
        try:
            coverage_directory = resolve_within(
                workspace,
                coverage_root,
                context="frame coverage root",
                must_exist=False,
            )
        except ValueError as exc:
            raise ProvenanceError(str(exc)) from exc
        if not coverage_directory.is_dir():
            raise ProvenanceError(
                f"frame coverage root is not a directory: {coverage_root}"
            )
        coverage_files = {
            path.relative_to(workspace).as_posix()
            for path in coverage_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
        }
        missing_from_sidecars = sorted(coverage_files - seen_paths.keys())
        outside_coverage = sorted(seen_paths.keys() - coverage_files)
        if missing_from_sidecars or outside_coverage:
            raise ProvenanceError(
                "frame coverage must match provenance records exactly; "
                f"untracked={missing_from_sidecars[:5]}, outside_coverage={outside_coverage[:5]}"
            )

    return {
        "status": "valid",
        "sidecars": len(materialized_sidecars),
        "frames": len(seen_paths),
        "coverage_files": len(coverage_files) if coverage_files is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate extracted frame metadata, hashes, and optional directory coverage."
    )
    parser.add_argument("sidecars", nargs="+")
    parser.add_argument("--root", default=".")
    parser.add_argument("--coverage-root")
    args = parser.parse_args()
    try:
        result = validate_provenance(
            args.sidecars, root=args.root, coverage_root=args.coverage_root
        )
    except ProvenanceError as exc:
        raise SystemExit(f"provenance validation error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
