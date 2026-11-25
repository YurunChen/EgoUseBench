"""Bind each executable task to its own screened record and frame provenance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sources import SourceError, find_record, load_dossier
from .workspace import resolve_within


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_KEYS = {
    "dossier",
    "task_id",
    "frame_provenance",
}
TASK_SPEC_KEYS = {"task_spec", "task_spec_sha256"}
EGO4D_PARENT_MAPPING = "reference/sources/EgoSchema/repo/uid_to_ego4d.json"


class LineageError(ValueError):
    """Raised when a Task no longer matches its screened source evidence."""


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_continuous_media_video(
    root: str | Path,
    source_root: str,
    media_id: str,
    *,
    context: str,
) -> Path:
    """Resolve one continuous source video for a screened media_id."""

    video_root = resolve_within(
        root,
        source_root,
        context=f"{context} source video root",
        must_exist=False,
    )
    if not video_root.is_dir():
        raise LineageError(f"{context}: source video root is missing: {source_root}")
    suffixes = {".mp4", ".mov", ".mkv", ".avi"}
    candidates = sorted(
        path
        for path in video_root.glob(f"{media_id}*")
        if path.is_file() and path.suffix.casefold() in suffixes
    )
    if len(candidates) != 1:
        raise LineageError(
            f"{context}: media_id {media_id!r} resolves to {len(candidates)} videos"
        )
    return candidates[0]


@dataclass(frozen=True)
class SourceMediaSegment:
    """One real file mapped onto the original recording's elapsed timeline."""

    media_id: str
    original_video_id: str
    source_start_seconds: float
    source_end_seconds: float | None
    media_sha256: str | None

    def contains(self, timestamp: float) -> bool:
        return self.source_start_seconds <= timestamp and (
            self.source_end_seconds is None or timestamp < self.source_end_seconds
        )


def source_media_segments(
    source: Mapping[str, Any], *, context: str
) -> tuple[SourceMediaSegment, ...]:
    """Read explicit multi-file mapping, or retain the original single-file contract."""

    media_id = source.get("media_id")
    if not isinstance(media_id, str) or not media_id:
        raise LineageError(f"{context}: source media_id is missing")
    original_id = source.get("original_video_id", media_id)
    if "media_segments" not in source:
        media_start = source.get("media_start_seconds", 0)
        if isinstance(media_start, bool):
            raise LineageError(f"{context}: media_start_seconds must be numeric, not bool")
        return (
            SourceMediaSegment(
                media_id,
                original_id,
                float(media_start),
                None,
                source.get("media_sha256"),
            ),
        )
    rows = source["media_segments"]
    if not isinstance(rows, list) or len(rows) < 2:
        raise LineageError(f"{context}: media_segments needs at least two segments")
    if not isinstance(source.get("original_video_id"), str) or not original_id:
        raise LineageError(f"{context}: segmented source needs original_video_id")
    segments = []
    seen = set()
    for index, row in enumerate(rows):
        name = f"{context}/media_segments/{index}"
        if not isinstance(row, Mapping):
            raise LineageError(f"{name}: segment must be an object")
        identity = row.get("media_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise LineageError(f"{name}: media_id must be non-empty and unique")
        seen.add(identity)
        if row.get("original_video_id") != original_id:
            raise LineageError(f"{name}: segment belongs to a different recording")
        start, end = row.get("source_start_seconds"), row.get("source_end_seconds")
        if (
            type(start) not in (int, float)
            or type(end) not in (int, float)
            or not math.isfinite(start)
            or not math.isfinite(end)
            or not 0 <= start < end
        ):
            raise LineageError(f"{name}: invalid source interval")
        digest = _declared_digest(row.get("media_sha256"), name)
        segment = SourceMediaSegment(
            identity, original_id, float(start), float(end), digest
        )
        if segments and start < segments[-1].source_end_seconds:
            raise LineageError(f"{name}: source segments overlap or are out of order")
        segments.append(segment)
    if media_id != segments[0].media_id:
        raise LineageError(f"{context}: source media_id must name the first segment")
    if (
        "media_start_seconds" in source
        and source["media_start_seconds"] != segments[0].source_start_seconds
    ):
        raise LineageError(
            f"{context}: primary source offset disagrees with first segment"
        )
    if "media_sha256" in source and source["media_sha256"] != segments[0].media_sha256:
        raise LineageError(
            f"{context}: primary source digest disagrees with first segment"
        )
    return tuple(segments)


def media_segment_at(
    segments: tuple[SourceMediaSegment, ...], timestamp: float, *, context: str
) -> SourceMediaSegment:
    if type(timestamp) not in (int, float) or not math.isfinite(timestamp):
        raise LineageError(f"{context}: source timestamp must be finite")
    matches = [segment for segment in segments if segment.contains(timestamp)]
    if len(matches) != 1:
        raise LineageError(
            f"{context}: source timestamp {timestamp} has no unique media coverage"
        )
    return matches[0]


def media_segment_covering(
    segments: tuple[SourceMediaSegment, ...],
    interval: tuple[float, float],
    *,
    context: str,
) -> SourceMediaSegment:
    start, end = interval
    segment = media_segment_at(segments, start, context=context)
    if end < start or not segment.contains(end):
        raise LineageError(
            f"{context}: continuous interval is not covered by one real media segment; missing footage must be acquired"
        )
    return segment


def source_segment_references(
    record: Mapping[str, Any], dataset: Mapping[str, Any], root: Path
) -> dict[str, Any] | None:
    """Diagnostic source references; never a substitute for performer input frames."""

    source = record["source"]
    if "media_segments" not in source:
        return None
    context = str(record["task_id"])
    segments = source_media_segments(source, context=context)
    refs = []
    gaps = []
    for index, segment in enumerate(segments):
        video = resolve_continuous_media_video(
            root, dataset["source_root"], segment.media_id, context=context
        )
        refs.append(
            {
                "media_id": segment.media_id,
                "original_video_id": segment.original_video_id,
                "path": video.relative_to(root).as_posix(),
                "sha256": segment.media_sha256,
                "source_interval_seconds": [
                    segment.source_start_seconds,
                    segment.source_end_seconds,
                ],
                "playback_interval_seconds": [
                    0,
                    segment.source_end_seconds - segment.source_start_seconds,
                ],
            }
        )
        if (
            index
            and segments[index - 1].source_end_seconds < segment.source_start_seconds
        ):
            gaps.append(
                [segments[index - 1].source_end_seconds, segment.source_start_seconds]
            )
    return {
        "segments": refs,
        "unavailable_intervals_seconds": gaps,
        "inspection_requirement": "inspect_each_available_segment_in_order; intervening footage is unavailable",
    }


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LineageError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LineageError(f"{context}: JSON root must be an object")
    return payload


def _recording_identity(
    record: Mapping[str, Any],
    dataset: str,
    parent_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    source = record.get("source")
    context = str(record.get("task_id", "source record"))
    if not isinstance(source, Mapping):
        raise LineageError(f"{context}: source must be an object")
    if source.get("dataset", dataset) != dataset:
        raise LineageError(f"{context}: source and dossier datasets disagree")
    if "media_segments" in source:
        source_media_segments(source, context=context)
    media_id = source.get("media_id")
    recording_id = source.get("original_video_id") or media_id
    if not isinstance(media_id, str) or not media_id:
        raise LineageError(f"{context}: source media_id is missing")
    if not isinstance(recording_id, str) or not recording_id:
        raise LineageError(f"{context}: source recording ID is missing")
    identity: dict[str, Any] = {
        "task_id": context,
        "source_dataset": dataset,
        "media_id": media_id,
        "recording_dataset": dataset,
        "recording_id": recording_id,
        "identity_basis": "declared_source_identity",
    }
    if dataset not in {"Ego2Web", "EgoSchema"}:
        return identity
    parent = parent_mapping.get(media_id)
    if not isinstance(parent, Mapping):
        raise LineageError(f"{context}: no Ego4D parent mapping for {media_id!r}")
    parent_id = parent.get("video_uid")
    start, end = parent.get("starting_sec"), parent.get("ending_sec")
    if not isinstance(parent_id, str) or not parent_id:
        raise LineageError(f"{context}: Ego4D parent video_uid is missing")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        raise LineageError(f"{context}: invalid Ego4D parent clip interval")
    if (
        not all(
            not isinstance(value, bool) and math.isfinite(value)
            for value in (start, end)
        )
        or not 0 <= start < end
    ):
        raise LineageError(f"{context}: invalid Ego4D parent clip interval")
    return {
        **identity,
        "recording_dataset": "Ego4D",
        "recording_id": parent_id,
        "identity_basis": "canonical_parent_mapping",
        "clip_interval_seconds": [start, end],
    }


def task_recording_group(task: Mapping[str, Any], *, root: str | Path) -> str:
    """Derive the statistical unit from the original recording, not the task ID."""
    path, dossier = load_dossier(task, root=root)
    record = find_record(dossier, task["source"]["task_id"], context=str(path))
    dataset = dossier["dataset"]["name"]
    parent_mapping = (
        _load_object(Path(root) / EGO4D_PARENT_MAPPING, EGO4D_PARENT_MAPPING)
        if dataset in {"Ego2Web", "EgoSchema"}
        else {}
    )
    identity = _recording_identity(record, dataset, parent_mapping)
    return f"{identity['recording_dataset']}:{identity['recording_id']}"


def summarize_source_recordings(
    records: Iterable[tuple[Path, Mapping[str, Any]]],
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Count original recordings, retaining clip identity and auditable collisions.

    This reports shared parent recordings without deciding which task to retire.
    Other datasets retain their declared recording identity; this is not a claim
    that their parent lineage has been independently verified.
    """

    workspace = Path(root).resolve()
    datasets: dict[Path, str] = {}
    parent_mapping: dict[str, Any] = {}
    mapping_evidence: list[dict[str, str]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    clips: set[tuple[str, str]] = set()
    seen_tasks: set[str] = set()
    for manifest, record in records:
        manifest = Path(manifest)
        if manifest.is_absolute():
            manifest = manifest.relative_to(workspace)
        manifest_path = resolve_within(workspace, manifest, context="recording dossier")
        if manifest_path not in datasets:
            payload = _load_object(manifest_path, str(manifest))
            metadata = payload.get("dataset")
            dataset = metadata.get("name") if isinstance(metadata, Mapping) else None
            if not isinstance(dataset, str) or not dataset:
                raise LineageError(f"{manifest}: dataset.name is missing")
            datasets[manifest_path] = dataset
        dataset = datasets[manifest_path]
        if dataset in {"Ego2Web", "EgoSchema"} and not mapping_evidence:
            mapping_path = resolve_within(
                workspace,
                EGO4D_PARENT_MAPPING,
                context="Ego4D parent mapping",
                must_exist=False,
            )
            parent_mapping = _load_object(mapping_path, EGO4D_PARENT_MAPPING)
            mapping_evidence.append(
                {
                    "path": EGO4D_PARENT_MAPPING,
                    "sha256": file_sha256(mapping_path),
                }
            )
        identity = _recording_identity(record, dataset, parent_mapping)
        task_id = identity["task_id"]
        if task_id in seen_tasks:
            raise LineageError(f"duplicate task in recording summary: {task_id}")
        seen_tasks.add(task_id)
        if "media_segments" in record["source"]:
            clips.update(
                (dataset, segment.media_id)
                for segment in source_media_segments(record["source"], context=task_id)
            )
        else:
            clips.add((dataset, identity["media_id"]))
        key = (identity["recording_dataset"], identity["recording_id"])
        groups.setdefault(key, []).append(identity)
    collisions: list[dict[str, Any]] = [
        {
            "recording_dataset": dataset,
            "recording_id": recording_id,
            "tasks": sorted(members, key=lambda member: member["task_id"]),
        }
        for (dataset, recording_id), members in sorted(groups.items())
        if len(members) > 1
    ]
    return {
        "source_clusters": len(groups),
        "source_clips": len(clips),
        "source_recording_collisions": collisions,
        "source_recording_mappings": mapping_evidence,
    }


def validate_independent_recordings(
    records: Iterable[tuple[Path, Mapping[str, Any]]], *, root: str | Path
) -> dict[str, Any]:
    """Require formal members to use distinct original recordings."""
    summary = summarize_source_recordings(records, root=root)
    collisions = summary["source_recording_collisions"]
    if collisions:
        collision = collisions[0]
        members = [member["task_id"] for member in collision["tasks"]]
        raise LineageError(
            f"source cluster {collision['recording_dataset']}:{collision['recording_id']} "
            f"is shared by {', '.join(members)}"
        )
    return summary


def discover_frame_provenance(
    dossier_reference: str,
    *,
    root: str | Path,
) -> tuple[str, Path]:
    """Find the one sidecar that declares ownership of a screening dossier.

    Sidecars sit beside the manifests in a sibling directory, so a dossier's own
    folder is searched as well as `../provenance` — the two layouts differ only in
    where the file lives, never in which manifest it declares.
    """

    workspace = Path(root).resolve()
    dossier_path = resolve_within(
        workspace, dossier_reference, context="source dossier"
    )
    folders = {dossier_path.parent, dossier_path.parent.parent / "provenance"}
    matches: list[tuple[str, Path]] = []
    for sidecar_path in sorted(
        path
        for folder in folders
        if folder.is_dir()
        for path in folder.glob("*frame_provenance.json")
    ):
        payload = _load_object(sidecar_path, str(sidecar_path))
        if payload.get("manifest") == dossier_reference:
            matches.append(
                (sidecar_path.relative_to(workspace).as_posix(), sidecar_path)
            )
    if len(matches) != 1:
        raise LineageError(
            f"{dossier_reference}: expected exactly one declaring frame-provenance sidecar, "
            f"found {len(matches)}"
        )
    return matches[0]


def build_source_identity(
    record: dict[str, Any],
    *,
    dossier_reference: str,
    provenance_reference: str,
    root: str | Path,
) -> dict[str, str]:
    workspace = Path(root).resolve()
    resolve_within(workspace, dossier_reference, context="source dossier")
    resolve_within(workspace, provenance_reference, context="frame provenance")
    return {
        "dossier": dossier_reference,
        "task_id": str(record["task_id"]),
        "frame_provenance": provenance_reference,
    }


def _declared_digest(value: Any, context: str) -> str:
    digest = str(value or "")
    if not SHA256.fullmatch(digest):
        raise LineageError(f"{context}: expected a lowercase SHA-256 digest")
    return digest


def _matching_source_record(
    task: dict[str, Any],
    dossier: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    try:
        return find_record(dossier, str(task["id"]), context=context)
    except SourceError as exc:
        raise LineageError(str(exc)) from exc


def _validate_source_record(
    task: dict[str, Any],
    record: dict[str, Any],
) -> None:
    task_id = task["id"]
    expected = {
        "status": "video_screened",
        "family": task["track"],
    }
    mismatches = {
        key: (record.get(key), value)
        for key, value in expected.items()
        if record.get(key) != value
    }
    if mismatches:
        raise LineageError(f"{task_id}: source record identity mismatch: {mismatches}")
    source = record.get("source")
    if (
        not isinstance(source, dict)
        or not isinstance(source.get("media_id"), str)
        or not source["media_id"]
    ):
        raise LineageError(f"{task_id}: source media_id is missing")

    packets = record.get("packets")
    if not isinstance(packets, list):
        raise LineageError(f"{task_id}: source packets must be a list")
    source_packets = {
        packet.get("id"): packet for packet in packets if isinstance(packet, dict)
    }
    released_packets = {packet["id"]: packet for packet in task["visual_packets"]}
    if set(source_packets) != set(released_packets):
        raise LineageError(f"{task_id}: source and released packet IDs disagree")
    for packet_id, released in released_packets.items():
        if source_packets[packet_id].get("frame_paths") != released.get("frame_paths"):
            raise LineageError(
                f"{task_id}/{packet_id}: released frame paths disagree with source dossier"
            )


def _validate_frame_records(
    task: dict[str, Any],
    record: dict[str, Any],
    provenance: dict[str, Any],
    *,
    root: Path,
) -> None:
    task_id = task["id"]
    frames = provenance.get("frames")
    if not isinstance(frames, list):
        raise LineageError(f"{task_id}: provenance frames must be a list")
    task_frames = [
        frame
        for frame in frames
        if isinstance(frame, dict) and frame.get("task_id") == task_id
    ]
    expected_count = sum(
        len(packet["frame_paths"]) for packet in task["visual_packets"]
    )
    if len(task_frames) != expected_count:
        raise LineageError(
            f"{task_id}: provenance contains {len(task_frames)} task frames, "
            f"expected {expected_count}"
        )
    by_key = {
        (frame.get("packet_id"), frame.get("frame_index")): frame
        for frame in task_frames
    }
    if len(by_key) != len(task_frames):
        raise LineageError(f"{task_id}: duplicate frame provenance key")

    source_packets = {packet["id"]: packet for packet in record["packets"]}
    segmented = "media_segments" in record["source"]
    segments = source_media_segments(record["source"], context=task_id) if segmented else ()
    for packet in task["visual_packets"]:
        packet_id = packet["id"]
        source_timestamps = source_packets[packet_id].get("timestamps_seconds")
        if not isinstance(source_timestamps, list) or len(source_timestamps) != len(
            packet["frame_paths"]
        ):
            raise LineageError(
                f"{task_id}/{packet_id}: source timestamps are incomplete"
            )
        for index, (frame_path, timestamp) in enumerate(
            zip(packet["frame_paths"], source_timestamps, strict=True)
        ):
            provenance_record = by_key.get((packet_id, index))
            if provenance_record is None:
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: missing frame provenance"
                )
            segment = (
                media_segment_at(
                    segments, timestamp, context=f"{task_id}/{packet_id}/{index}"
                )
                if segmented
                else None
            )
            expected_media_id = (
                segment.media_id
                if segment is not None
                else record["source"]["media_id"]
            )
            if provenance_record.get("media_id") != expected_media_id:
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: source media_id disagrees with frame provenance"
                )
            if provenance_record.get("frame_path") != frame_path:
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: provenance path mismatch"
                )
            provenance_timestamp = provenance_record.get("timestamp_seconds")
            if (
                not isinstance(provenance_timestamp, (int, float))
                or not isinstance(timestamp, (int, float))
                or not math.isclose(
                    float(provenance_timestamp), float(timestamp), abs_tol=1e-6
                )
            ):
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: provenance timestamp mismatch"
                )
            if segment is not None:
                local_time = provenance_record.get("media_timestamp_seconds")
                metadata = provenance.get("videos", {}).get(segment.media_id, {})
                if (
                    type(local_time) not in (int, float)
                    or not math.isclose(
                        float(timestamp) - segment.source_start_seconds,
                        local_time,
                        abs_tol=1e-6,
                    )
                    or metadata.get("source_start_seconds")
                    != segment.source_start_seconds
                    or metadata.get("source_end_seconds") != segment.source_end_seconds
                    or metadata.get("original_video_id") != segment.original_video_id
                    or metadata.get("media_sha256") != segment.media_sha256
                    or type(metadata.get("duration_seconds")) not in (int, float)
                    or not math.isclose(
                        metadata["duration_seconds"],
                        segment.source_end_seconds - segment.source_start_seconds,
                        abs_tol=0.05,
                    )
                ):
                    raise LineageError(
                        f"{task_id}/{packet_id}/{index}: segment provenance offset or duration mismatch"
                    )
            resolved_frame = resolve_within(
                root,
                frame_path,
                context=f"{task_id}/{packet_id}/{index} released frame",
            )
            if file_sha256(resolved_frame) != provenance_record.get("sha256"):
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: released frame SHA-256 mismatch"
                )


def validate_source_lineage(task: dict[str, Any], *, root: str | Path) -> None:
    """Replay the complete source-to-Released-Observation identity check."""

    task_id = task["id"]
    source = task.get("source")
    source_keys = set(source) if isinstance(source, dict) else set()
    if (
        not isinstance(source, dict)
        or not SOURCE_KEYS <= source_keys
        or (source_keys & TASK_SPEC_KEYS and not TASK_SPEC_KEYS <= source_keys)
    ):
        raise LineageError(
            f"{task_id}: source must contain visual lineage and an optional paired task spec"
        )
    if source.get("task_id") != task_id:
        raise LineageError(f"{task_id}: source task_id mismatch")

    workspace = Path(root).resolve()
    dossier_reference = str(source["dossier"])
    dossier_path = resolve_within(
        workspace, dossier_reference, context=f"{task_id} dossier"
    )
    dossier = _load_object(dossier_path, f"{task_id} dossier")
    if dossier.get("schema_version") != 2:
        raise LineageError(f"{task_id}: source dossier schema_version must be 2")
    record = _matching_source_record(task, dossier, f"{task_id} dossier")
    _validate_source_record(task, record)

    provenance_reference = str(source["frame_provenance"])
    provenance_path = resolve_within(
        workspace, provenance_reference, context=f"{task_id} frame provenance"
    )
    provenance = _load_object(provenance_path, f"{task_id} frame provenance")
    if provenance.get("schema_version") != 2:
        raise LineageError(f"{task_id}: frame provenance schema_version must be 2")
    if provenance.get("manifest") != dossier_reference:
        raise LineageError(f"{task_id}: provenance does not declare the source dossier")
    _validate_frame_records(task, record, provenance, root=workspace)

    if TASK_SPEC_KEYS <= set(source):
        task_spec_reference = str(source["task_spec"])
        task_spec_path = resolve_within(
            workspace, task_spec_reference, context=f"{task_id} task spec"
        )
        task_spec_hash = _declared_digest(
            source.get("task_spec_sha256"), f"{task_id}/task_spec"
        )
        if file_sha256(task_spec_path) != task_spec_hash:
            raise LineageError(f"{task_id}: task spec SHA-256 mismatch")
        task_specs = _load_object(task_spec_path, f"{task_id} task spec").get("tasks")
        matches = [
            spec
            for spec in task_specs or []
            if isinstance(spec, dict) and spec.get("task_id") == task_id
        ]
        if len(matches) != 1 or matches[0].get("track") != task["track"]:
            raise LineageError(f"{task_id}: task spec identity mismatch")
