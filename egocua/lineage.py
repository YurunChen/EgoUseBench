"""Bind executable tasks to their screened source evidence and frame provenance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from .run_ledger import canonical_sha256
from .workspace import resolve_within


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_KEYS = {
    "dossier",
    "dossier_sha256",
    "task_id",
    "record_sha256",
    "frame_provenance",
    "frame_provenance_sha256",
}
TASK_SPEC_KEYS = {"task_spec", "task_spec_sha256"}


class LineageError(ValueError):
    """Raised when a Task no longer matches its screened source evidence."""


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LineageError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LineageError(f"{context}: JSON root must be an object")
    return payload


def discover_frame_provenance(
    dossier_reference: str, *, root: str | Path,
) -> tuple[str, Path]:
    """Find the one sidecar that declares ownership of a screening dossier."""

    workspace = Path(root).resolve()
    dossier_path = resolve_within(workspace, dossier_reference, context="source dossier")
    matches: list[tuple[str, Path]] = []
    for sidecar_path in sorted(dossier_path.parent.glob("*_frame_provenance.json")):
        payload = _load_object(sidecar_path, str(sidecar_path))
        if payload.get("manifest") == dossier_reference:
            matches.append((sidecar_path.relative_to(workspace).as_posix(), sidecar_path))
    if len(matches) != 1:
        raise LineageError(
            f"{dossier_reference}: expected exactly one declaring frame-provenance sidecar, "
            f"found {len(matches)}"
        )
    return matches[0]


def build_source_identity(
    record: dict[str, Any], *, dossier_reference: str,
    provenance_reference: str, root: str | Path,
) -> dict[str, str]:
    workspace = Path(root).resolve()
    dossier_path = resolve_within(workspace, dossier_reference, context="source dossier")
    provenance_path = resolve_within(
        workspace, provenance_reference, context="frame provenance"
    )
    return {
        "dossier": dossier_reference,
        "dossier_sha256": file_sha256(dossier_path),
        "task_id": str(record["task_id"]),
        "record_sha256": canonical_sha256(record),
        "frame_provenance": provenance_reference,
        "frame_provenance_sha256": file_sha256(provenance_path),
    }


def _declared_digest(value: Any, context: str) -> str:
    digest = str(value or "")
    if not SHA256.fullmatch(digest):
        raise LineageError(f"{context}: expected a lowercase SHA-256 digest")
    return digest


def _matching_source_record(
    task: dict[str, Any], dossier: dict[str, Any], context: str,
) -> dict[str, Any]:
    records = dossier.get("tasks")
    if not isinstance(records, list):
        raise LineageError(f"{context}: dossier tasks must be a list")
    matches = [
        record for record in records
        if isinstance(record, dict) and record.get("task_id") == task["id"]
    ]
    if len(matches) != 1:
        raise LineageError(
            f"{context}: expected one source record for {task['id']!r}, found {len(matches)}"
        )
    return matches[0]


def _validate_source_record(
    task: dict[str, Any], record: dict[str, Any], declared_hash: str,
) -> None:
    task_id = task["id"]
    if canonical_sha256(record) != declared_hash:
        raise LineageError(f"{task_id}: source record SHA-256 mismatch")
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
    task: dict[str, Any], record: dict[str, Any], provenance: dict[str, Any],
    *, root: Path,
) -> None:
    task_id = task["id"]
    frames = provenance.get("frames")
    if not isinstance(frames, list):
        raise LineageError(f"{task_id}: provenance frames must be a list")
    task_frames = [
        frame for frame in frames
        if isinstance(frame, dict) and frame.get("task_id") == task_id
    ]
    expected_count = sum(len(packet["frame_paths"]) for packet in task["visual_packets"])
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
    for packet in task["visual_packets"]:
        packet_id = packet["id"]
        source_timestamps = source_packets[packet_id].get("timestamps_seconds")
        if not isinstance(source_timestamps, list) or len(source_timestamps) != len(
            packet["frame_paths"]
        ):
            raise LineageError(f"{task_id}/{packet_id}: source timestamps are incomplete")
        for index, (frame_path, timestamp) in enumerate(
            zip(packet["frame_paths"], source_timestamps, strict=True)
        ):
            provenance_record = by_key.get((packet_id, index))
            if provenance_record is None:
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: missing frame provenance"
                )
            if provenance_record.get("frame_path") != frame_path:
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: provenance path mismatch"
                )
            provenance_timestamp = provenance_record.get("timestamp_seconds")
            if not isinstance(provenance_timestamp, (int, float)) or not isinstance(
                timestamp, (int, float)
            ) or not math.isclose(
                float(provenance_timestamp), float(timestamp), abs_tol=1e-6
            ):
                raise LineageError(
                    f"{task_id}/{packet_id}/{index}: provenance timestamp mismatch"
                )
            resolved_frame = resolve_within(
                root, frame_path, context=f"{task_id}/{packet_id}/{index} released frame"
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
    if not isinstance(source, dict) or source_keys not in (
        SOURCE_KEYS, SOURCE_KEYS | TASK_SPEC_KEYS
    ):
        raise LineageError(
            f"{task_id}: source must contain visual lineage and an optional paired task spec"
        )
    if source.get("task_id") != task_id:
        raise LineageError(f"{task_id}: source task_id mismatch")

    workspace = Path(root).resolve()
    dossier_reference = str(source["dossier"])
    dossier_path = resolve_within(workspace, dossier_reference, context=f"{task_id} dossier")
    dossier_hash = _declared_digest(source.get("dossier_sha256"), f"{task_id}/dossier")
    if file_sha256(dossier_path) != dossier_hash:
        raise LineageError(f"{task_id}: source dossier SHA-256 mismatch")
    dossier = _load_object(dossier_path, f"{task_id} dossier")
    if dossier.get("schema_version") != 2:
        raise LineageError(f"{task_id}: source dossier schema_version must be 2")
    record = _matching_source_record(task, dossier, f"{task_id} dossier")
    record_hash = _declared_digest(source.get("record_sha256"), f"{task_id}/record")
    _validate_source_record(task, record, record_hash)

    provenance_reference = str(source["frame_provenance"])
    provenance_path = resolve_within(
        workspace, provenance_reference, context=f"{task_id} frame provenance"
    )
    provenance_hash = _declared_digest(
        source.get("frame_provenance_sha256"), f"{task_id}/frame_provenance"
    )
    if file_sha256(provenance_path) != provenance_hash:
        raise LineageError(f"{task_id}: frame provenance SHA-256 mismatch")
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
            spec for spec in task_specs or []
            if isinstance(spec, dict) and spec.get("task_id") == task_id
        ]
        if len(matches) != 1 or matches[0].get("track") != task["track"]:
            raise LineageError(f"{task_id}: task spec identity mismatch")
