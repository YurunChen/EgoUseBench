"""Extract reproducible, timestamped keyframes from source videos.

The input manifest is the source of truth for frame selection. The generated
sidecar records the source timestamp and SHA-256 digest of every frame so a
released task can be audited without treating uniformly sampled stills as
independent annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from itertools import pairwise
from pathlib import Path
from typing import Any

from egocua.workspace import resolve_within, write_json_atomic

_PACKET_LAYOUTS = {
    "graft": (("V0", 8),),
    "refresh": (("V0", 4), ("V1", 4)),
    "clarify": (("V0", 8),),
}
_REJECTION_CODES = {
    "behavior_change_not_required",
    "condition_changes_multiple_factors",
    "construct_not_isolated",
    "counterfactual_check_failed",
    "decisive_attribute_unreadable",
    "environment_mismatch",
    "evaluator_invalid",
    "event_not_visible",
    "event_origin_unsupported",
    "exact_entity_unlinkable",
    "ground_truth_not_deterministic",
    "human_unsolvable",
    "insufficient_candidate_ambiguity",
    "license_or_access_blocked",
    "predicate_exceeds_evidence",
    "redundant_decision_pattern",
    "temporal_relation_not_supported",
    "visual_not_necessary",
}


def _retains_packet_evidence(task: dict[str, Any]) -> bool:
    packets = task.get("packets")
    if not isinstance(packets, list) or not packets:
        return False
    frame_paths = [
        path
        for packet in packets if isinstance(packet, dict)
        for path in packet.get("frame_paths", [])
    ]
    return bool(frame_paths) and all(
        isinstance(path, str)
        and Path(path).suffix.lower() in {".jpg", ".jpeg", ".png"}
        for path in frame_paths
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError("evidence manifest must use schema_version 2")
    dataset = payload.get("dataset")
    if (
        not isinstance(dataset, dict)
        or not dataset.get("name")
        or not dataset.get("source_root")
    ):
        raise ValueError("evidence manifest needs dataset name and source_root")
    if not isinstance(payload.get("tasks"), list) or not payload["tasks"]:
        raise ValueError("evidence manifest needs at least one task")
    for task in payload["tasks"]:
        if task.get("status") == "rejected":
            code = task.get("rejection_code")
            if code not in _REJECTION_CODES:
                raise ValueError(
                    f"{task.get('task_id', 'task')}: invalid rejection_code {code!r}"
                )
            if not str(task.get("rejection_reason", "")).strip():
                raise ValueError(
                    f"{task.get('task_id', 'task')}: rejected task needs a reason"
                )
            if not _retains_packet_evidence(task):
                continue
        source = task.get("source")
        if not isinstance(source, dict) or not source.get("media_id"):
            raise ValueError("evidence task needs source.media_id")
    return payload


def _probe(video: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=width,height",
        "-of",
        "json",
        str(video),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return {
        "duration_seconds": float(payload["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
    }


def _extract_frame(video: Path, timestamp: float, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-ss",
        f"{timestamp:.3f}",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(destination),
    ]
    subprocess.run(command, check=True)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _media_start_seconds(source: dict[str, Any], task_id: str) -> float:
    raw_value = source.get("media_start_seconds", 0.0)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{task_id}: source.media_start_seconds must be numeric"
        ) from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(
            f"{task_id}: source.media_start_seconds must be finite and non-negative"
        )
    return value


def _validate_packet_sampling(task: dict[str, Any]) -> list[dict[str, Any]]:
    task_id = str(task.get("task_id", "")).strip()
    family = str(task.get("family", "")).strip()
    expected_layout = _PACKET_LAYOUTS.get(family)
    if expected_layout is None:
        raise ValueError(f"{task_id}: unsupported task family {family!r}")

    packets = task.get("packets")
    if not isinstance(packets, list):
        raise TypeError(f"{task_id}: packets must be a list")
    if any(not isinstance(packet, dict) for packet in packets):
        raise TypeError(f"{task_id}: every packet must be an object")
    expected_ids = [packet_id for packet_id, _ in expected_layout]
    actual_ids = [str(packet.get("id", "")).strip() for packet in packets]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{task_id}: {family} packets must be ordered as {expected_ids}, got {actual_ids}"
        )

    summaries: list[dict[str, Any]] = []
    for packet, (packet_id, expected_count) in zip(
        packets, expected_layout, strict=True
    ):
        raw_timestamps = packet.get("timestamps_seconds")
        frame_paths = packet.get("frame_paths")
        if (
            not isinstance(raw_timestamps, list)
            or len(raw_timestamps) != expected_count
        ):
            raise ValueError(
                f"{task_id}/{packet_id}: expected {expected_count} timestamps"
            )
        if not isinstance(frame_paths, list) or len(frame_paths) != expected_count:
            raise ValueError(
                f"{task_id}/{packet_id}: expected {expected_count} frame paths"
            )
        try:
            timestamps = [float(value) for value in raw_timestamps]
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{task_id}/{packet_id}: timestamps must be numeric"
            ) from error
        if not all(math.isfinite(value) for value in timestamps):
            raise ValueError(f"{task_id}/{packet_id}: timestamps must be finite")
        if any(current <= previous for previous, current in pairwise(timestamps)):
            raise ValueError(
                f"{task_id}/{packet_id}: timestamps must be strictly increasing"
            )

        interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
        deltas = [current - previous for previous, current in pairwise(timestamps)]
        uniform = all(
            math.isclose(delta, interval, rel_tol=1e-4, abs_tol=1e-6)
            for delta in deltas
        )
        summary = {
            "task_id": task_id,
            "packet_id": packet_id,
            "sampling_method": (
                "uniform_inclusive_endpoints" if uniform else "event_aligned_timestamps"
            ),
            "frame_count": len(timestamps),
            "start_seconds": timestamps[0],
            "end_seconds": timestamps[-1],
        }
        if uniform:
            summary.update({
                "inter_frame_seconds": round(interval, 6),
                "effective_hz": round(1 / interval, 6),
            })
        summaries.append(summary)

    if (
        family == "refresh"
        and summaries[0]["end_seconds"] >= summaries[1]["start_seconds"]
    ):
        raise ValueError(f"{task_id}: V1 must begin after V0 ends")
    return summaries


def extract(manifest_path: Path, root: Path, sidecar_path: Path) -> dict[str, Any]:
    manifest = _read_manifest(manifest_path)
    video_metadata: dict[str, dict[str, Any]] = {}
    frame_records: list[dict[str, Any]] = []
    packet_sampling: list[dict[str, Any]] = []

    seen_tasks: set[str] = set()
    for task in manifest["tasks"]:
        if task.get("status") == "rejected" and not _retains_packet_evidence(task):
            continue
        task_id = str(task.get("task_id", "")).strip()
        video_id = str(task["source"].get("media_id", "")).strip()
        media_start = _media_start_seconds(task["source"], task_id)
        if not task_id or task_id in seen_tasks:
            raise ValueError(f"task_id must be non-empty and unique: {task_id!r}")
        seen_tasks.add(task_id)
        packet_sampling.extend(_validate_packet_sampling(task))
        video = resolve_within(
            root,
            Path(manifest["dataset"]["source_root"]) / f"{video_id}.mp4",
            context=f"{task_id} source video",
        )
        probed = _probe(video)
        probed["source_start_seconds"] = media_start
        probed["source_end_seconds"] = media_start + probed["duration_seconds"]
        existing_metadata = video_metadata.setdefault(video_id, probed)
        if existing_metadata != probed:
            raise ValueError(
                f"{task_id}: media_id {video_id!r} has conflicting source offsets"
            )
        metadata = existing_metadata

        packet_ids: set[str] = set()
        for packet in task.get("packets", []):
            packet_id = str(packet.get("id", "")).strip()
            timestamps = packet.get("timestamps_seconds", [])
            declared_paths = packet.get("frame_paths", [])
            if not packet_id or packet_id in packet_ids:
                raise ValueError(f"{task_id}: packet ids must be non-empty and unique")
            packet_ids.add(packet_id)
            if not timestamps or len(timestamps) != len(set(timestamps)):
                raise ValueError(
                    f"{task_id}/{packet_id}: timestamps must be non-empty and unique"
                )
            if len(declared_paths) != len(timestamps):
                raise ValueError(
                    f"{task_id}/{packet_id}: frame/timestamp count mismatch"
                )

            for index, raw_timestamp in enumerate(timestamps):
                source_timestamp = float(raw_timestamp)
                media_timestamp = source_timestamp - media_start
                if media_timestamp < 0 or media_timestamp >= metadata["duration_seconds"]:
                    raise ValueError(
                        f"{task_id}/{packet_id}: source timestamp {source_timestamp} "
                        f"outside media interval [{media_start}, "
                        f"{metadata['source_end_seconds']})"
                    )
                relative_path = Path(declared_paths[index])
                destination = resolve_within(
                    root,
                    relative_path,
                    context=f"{task_id}/{packet_id} output frame",
                    must_exist=False,
                )
                _extract_frame(video, media_timestamp, destination)
                frame_records.append(
                    {
                        "task_id": task_id,
                        "packet_id": packet_id,
                        "frame_index": index,
                        "media_id": video_id,
                        "timestamp_seconds": source_timestamp,
                        "media_timestamp_seconds": media_timestamp,
                        "frame_path": relative_path.as_posix(),
                        "sha256": _digest(destination),
                    }
                )

    result = {
        "schema_version": 2,
        "manifest": manifest_path.relative_to(root).as_posix(),
        "source_dataset": manifest["dataset"]["name"],
        "videos": video_metadata,
        "packet_sampling": packet_sampling,
        "frames": frame_records,
    }
    write_json_atomic(sidecar_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--sidecar", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    manifest_path = resolve_within(root, args.manifest, context="evidence manifest")
    sidecar_path = resolve_within(
        root, args.sidecar, context="frame provenance output", must_exist=False
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    result = extract(manifest_path, root, sidecar_path)
    print(
        f"extracted {len(result['frames'])} timestamped frames from {len(result['videos'])} videos"
    )


if __name__ == "__main__":
    main()
