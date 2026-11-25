from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from egocua.checker import check_run
from egocua.answer_closure import (
    MIN_PREDICATE_WORDS,
    accepted_ids,
    catalog_tokens,
    target_predicates,
    undecided_window_listings,
)
from egocua.construction import (
    compile_task_from_site,
    opaque_item_id,
    resolve_authored_record,
)
from egocua.environment import (
    entity_visual_path,
    load_site_data,
)
from egocua.experiment import TASK_REVIEW_PLAN, ExperimentPlan, load_experiment_plan
from egocua.lineage import (
    LineageError, media_segment_at, media_segment_covering,
    resolve_continuous_media_video, source_media_segments, validate_source_lineage,
)
from egocua.neighborhood import search_evidence_sha256
from egocua.run_ledger import (
    task_sha256,
    task_site_execution_sha256,
)
from egocua.reviews import audit_construction_reviews
from egocua.final_state import (
    evaluate_final_state,
    goal_entity_ids,
    normalized_entity_counts,
)
from egocua.construction import load_construction_tasks
from egocua.schema import (
    goal_for_condition,
    referenced_entity_ids,
    semantic_visual_representative_ids,
    validate_task,
)
from egocua.selection import load_selection, task_ids as selected_task_ids
from egocua.sources import SourceError, find_record, frame_provenance_path, load_dossier
from egocua.synthesis_reports import REPORTS
from egocua.timeline import STANDARD_CONDITIONS, Condition, Timeline
from egocua.workspace import resolve_within, write_json_atomic
from tools.audit.report_necessity import NECESSITY_REPORT

TASKS = "data/tasks"
MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
STATUS_VALUES = {
    "candidate_ready",
    "needs_source_review",
    "fail",
}
LEAKAGE = re.compile(
    r"\b(?:V0|V1|visual[_ -]?packet|keyframe|annotation|ground truth|"
    r"selected_entity_ids|submission_completed|correct answer|wrong answer|"
    r"a_only|oracle_latest|forced_execute|instruction_injection|"
    r"visual_context_update)\b|\b[pml]_[0-9a-f]{12}\b",
    re.IGNORECASE,
)
REVIEW_WORDING_FLAGS = re.compile(
    r"\b(?:I am showing|shown in my video|this clip|my video|in this sequence|"
    r"later in the sequence|visible in the video)\b",
    re.IGNORECASE,
)
PROCEDURE_EQUIVALENCE_WORDING = re.compile(
    r"\bmake it\b[^.?!]*\bmatch(?:es|ing)?\b|\bfrom start to finish\b",
    re.IGNORECASE,
)
SITUATED_ACTIVITY = re.compile(
    r"\b(?:hold|holding|handle|handling|use|using|work|working|install|repair|"
    r"sew|cook|walk|move|pick(?:ing)? up|put(?:ting)? down|drop|wipe|wear)\w*\b",
    re.IGNORECASE,
)
ORDINARY_WEB_FILTER = re.compile(
    r"\b(?:rating|stars?|price|cheaper|lower-priced|reviews?|views?|"
    r"popular|shipping|delivery|discount)\b",
    re.IGNORECASE,
)
PHYSICAL_UPDATE = re.compile(
    r"\b(?:while I (?:keep|am still)|I(?:'m| am) still|holding|wearing|"
    r"installing|repairing|sewing|cooking|walking|moving|putting|picking|"
    r"working on|using (?:this|that|the))\b",
    re.IGNORECASE,
)
REQUIRED_CONSTRUCT_FIELDS = {
    "research_question",
    "divergence_source",
    "event_origin",
    "belief_before",
    "belief_after",
    "required_behavior_change",
    "no_update_failure",
    "overupdate_failure",
    "visual_dependency_claim",
    "task_blueprint",
}
CLARIFICATION_QUESTIONS = {
    "position": "Which side or position do you mean?",
    "appearance": "Which appearance do you mean?",
    "brand": "Which brand do you mean?",
    "variant": "Which variant do you mean?",
    "sequence": "Which one in the sequence do you mean?",
}


def _extract_frame_sha256(
    video: Path, timestamp: float, jpeg_qscale: int | None = 2
) -> str:
    """Reproduce a source frame using its recorded JPEG encoder setting."""
    if jpeg_qscale is not None and (
        type(jpeg_qscale) is not int or not 1 <= jpeg_qscale <= 31
    ):
        raise ValueError("jpeg_qscale must be null or an integer from 1 to 31")
    quality = [] if jpeg_qscale is None else ["-q:v", str(jpeg_qscale)]
    with tempfile.TemporaryDirectory(prefix="egocua-frame-") as directory:
        destination = Path(directory) / "frame.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                *quality,
                str(destination),
            ],
            check=True,
            capture_output=True,
        )
        return hashlib.sha256(destination.read_bytes()).hexdigest()


class MediaVerificationCache:
    """Cache immutable video checks without weakening their invalidation boundary."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._toolchain = self._toolchain_identity()
        self._entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                payload = _read_object(path, "media verification cache")
                if (
                    payload.get("schema_version") == self.SCHEMA_VERSION
                    and payload.get("toolchain") == self._toolchain
                    and isinstance(payload.get("entries"), dict)
                ):
                    self._entries = dict(payload["entries"])
            except ValueError:
                self._entries = {}

    @staticmethod
    def _toolchain_identity() -> dict[str, str]:
        identity: dict[str, str] = {}
        for command in ("ffmpeg", "ffprobe"):
            result = subprocess.run(
                [command, "-version"],
                check=True,
                capture_output=True,
                text=True,
            )
            identity[command] = result.stdout.splitlines()[0]
        return identity

    @staticmethod
    def _file_identity(path: Path) -> dict[str, int]:
        stat = path.stat()
        return {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }

    def _entry(self, video: Path) -> dict[str, Any] | None:
        key = str(video.resolve())
        entry = self._entries.get(key)
        if not isinstance(entry, dict):
            return None
        if entry.get("file_identity") != self._file_identity(video):
            return None
        return entry

    def inspect_video(self, video: Path) -> dict[str, Any]:
        with self._lock:
            cached = self._entry(video)
            if cached is not None:
                return copy.deepcopy(cached)

        probe = subprocess.run(
            [
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
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(probe.stdout)
        stream = payload["streams"][0]
        entry = {
            "file_identity": self._file_identity(video),
            "sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
            "duration_seconds": float(payload["format"]["duration"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "frames": {},
        }
        with self._lock:
            self._entries[str(video.resolve())] = entry
        return copy.deepcopy(entry)

    def frame_sha256(
        self, video: Path, timestamp: float, jpeg_qscale: int | None = 2
    ) -> str:
        key = f"{timestamp:.6f}/q{jpeg_qscale}"
        with self._lock:
            entry = self._entry(video)
            frames = entry.get("frames", {}) if entry is not None else {}
            cached = frames.get(key) if isinstance(frames, dict) else None
            if isinstance(cached, str):
                return cached

        self.inspect_video(video)
        digest = _extract_frame_sha256(video, timestamp, jpeg_qscale)
        with self._lock:
            entry = self._entries[str(video.resolve())]
            entry.setdefault("frames", {})[key] = digest
        return digest

    def save(self) -> None:
        write_json_atomic(
            self.path,
            {
                "schema_version": self.SCHEMA_VERSION,
                "toolchain": self._toolchain,
                "entries": self._entries,
            },
        )


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: JSON root must be an object")
    return payload


def _check(
    ok: bool, *, details: Mapping[str, Any] | None = None, errors: Iterable[str] = ()
) -> dict[str, Any]:
    messages = [str(error) for error in errors if str(error)]
    return {
        "status": "pass" if ok and not messages else "fail",
        **({"details": dict(details)} if details is not None else {}),
        **({"errors": messages} if messages else {}),
    }


def _source_material(
    task: dict[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    try:
        dossier_path, dossier = load_dossier(task, root=root)
        record = find_record(
            dossier, str(task["id"]), context=f"{task['id']} source record"
        )
        provenance_path = frame_provenance_path(task, root=root)
    except SourceError as exc:
        raise ValueError(str(exc)) from exc
    _read_object(provenance_path, f"{task['id']} frame provenance")
    return dossier, record, dossier_path, provenance_path


def _claim_errors(record: Mapping[str, Any]) -> list[str]:
    packets = record.get("packets", [])
    available = {
        f"{packet.get('id')}:{index}"
        for packet in packets
        if isinstance(packet, dict)
        for index, _ in enumerate(packet.get("frame_paths", []))
    }
    claims = record.get("supported_claims")
    if not isinstance(claims, list) or not claims:
        return ["source record has no supported_claims"]
    errors: list[str] = []
    for index, claim in enumerate(claims):
        references = claim.get("supported_by") if isinstance(claim, dict) else None
        if (
            not isinstance(claim, dict)
            or not str(claim.get("text", "")).strip()
            or not isinstance(references, list)
            or not references
        ):
            errors.append(f"supported_claims/{index} is incomplete")
            continue
        unknown = sorted(set(references) - available)
        if unknown:
            errors.append(
                f"supported_claims/{index} references unknown frames {unknown}"
            )
    return errors


def _released_frame_count(task: Mapping[str, Any]) -> int:
    return sum(
        len(packet.get("frame_paths", []))
        for packet in task.get("visual_packets", [])
        if isinstance(packet, Mapping)
    )


def _construct_check(
    task: Mapping[str, Any],
    source_record: Mapping[str, Any],
    *,
    is_primary: bool,
) -> dict[str, Any]:
    """Check construct prerequisites without pretending to replace human judgment."""

    errors: list[str] = []
    construct = source_record.get("construct")
    if not isinstance(construct, dict):
        return _check(False, errors=["source record has no construct contract"])
    missing = sorted(
        field
        for field in REQUIRED_CONSTRUCT_FIELDS
        if field not in construct or construct[field] in (None, "", [], {})
    )
    if missing:
        errors.append(f"construct contract is missing {missing}")

    raw_source = source_record.get("source", {})
    source = raw_source if isinstance(raw_source, dict) else {}
    evidence_text = " ".join(
        [
            *(str(value) for value in source.get("original_annotations", [])),
            *(
                str(claim.get("text", ""))
                for claim in source_record.get("supported_claims", [])
                if isinstance(claim, dict)
            ),
            *(str(value) for value in task.get("instructions", {}).values()),
            *(
                str(event.get("description", ""))
                for event in source.get("observed_events", [])
                if isinstance(event, dict)
            ),
        ]
    )
    risk_reasons: list[str] = []
    ongoing_activity_candidate = bool(SITUATED_ACTIVITY.search(evidence_text))
    if not ongoing_activity_candidate:
        risk_reasons.append(
            "no automatic evidence that the wearer is still acting in the physical scene"
        )

    if task["track"] == "graft":
        update = str(task.get("instructions", {}).get("b", ""))
        if ORDINARY_WEB_FILTER.search(update) and not PHYSICAL_UPDATE.search(update):
            risk_reasons.append(
                "the added instruction appears to be only an ordinary website filter"
            )
        if _released_frame_count(task) <= 1:
            risk_reasons.append(
                "a single released frame may make this equivalent to static photo grounding"
            )
    elif task["track"] == "refresh":
        packet_ids = [packet.get("id") for packet in task.get("visual_packets", [])]
        if packet_ids != ["V0", "V1"]:
            errors.append("Refresh must release one ordered V0/V1 pair")
        if construct.get("event_origin") != "observed":
            errors.append("Refresh visual change is not registered as observed")
    else:
        if len(task.get("clarification", {}).get("candidate_ids", [])) != 2:
            errors.append("Clarify must contain exactly two physical candidates")
        if (
            len(task.get("visual_packets", [])) == 1
            and task.get("clarification", {}).get("valid_slot") != "sequence"
        ):
            risk_reasons.append(
                "the ambiguity may be reproducible with one uploaded static photograph"
            )

    return _check(
        not errors,
        details={
            "pilot_scope": "primary_seed" if is_primary else "rescreening_candidate",
            "research_question": construct.get("research_question"),
            "visual_dependency_claim": construct.get("visual_dependency_claim"),
            "required_behavior_change": construct.get("required_behavior_change"),
            "ongoing_activity_candidate": ongoing_activity_candidate,
            "construct_risk_reasons": risk_reasons,
            "human_review_recommended": bool(risk_reasons),
            "admission_test": (
                "Verify scene grounding, the released update and all legal website states. "
                "Success with upfront information or one frame is diagnostic, not a defect."
            ),
        },
        errors=errors,
    )


def _video_check(
    task: dict[str, Any],
    dossier: Mapping[str, Any],
    record: Mapping[str, Any],
    provenance_path: Path,
    root: Path,
    *,
    verify_source_frames: bool,
    media_cache: MediaVerificationCache | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    dataset = dossier.get("dataset")
    source_root = dataset.get("source_root") if isinstance(dataset, dict) else None
    if not isinstance(source_root, str) or not source_root:
        return _check(False, errors=["source dossier has no dataset.source_root"])
    provenance = _read_object(provenance_path, f"{task['id']} frame provenance")
    task_frames = [
        frame
        for frame in provenance.get("frames", [])
        if isinstance(frame, dict) and frame.get("task_id") == task["id"]
    ]
    media_ids = sorted(
        {
            str(frame.get("media_id"))
            for frame in task_frames
            if str(frame.get("media_id", "")).strip()
        }
    )
    declared_source = record.get("source", {})
    try:
        segments = source_media_segments(declared_source, context=task["id"])
    except (LineageError, TypeError, ValueError) as exc:
        return _check(False, errors=[str(exc)])
    segmented = "media_segments" in declared_source
    by_media = {segment.media_id: segment for segment in segments}
    declared_media_ids = set(by_media)
    if not segmented:
        declared_media_ids.add(declared_source.get("original_video_id"))
    if segmented and "continuous_video_interval" in record.get("construct", {}).get(
        "evidence_requirements", []
    ):
        timestamps = [
            time
            for packet in record.get("packets", [])
            for time in packet.get("timestamps_seconds", [])
        ]
        try:
            media_segment_covering(
                segments, (min(timestamps), max(timestamps)), context=task["id"]
            )
        except (LineageError, ValueError) as exc:
            errors.append(str(exc))

    if not media_ids:
        errors.append("frame provenance has no media_id for this task")
    if not set(media_ids) <= declared_media_ids:
        errors.append("frame provenance media_id is not declared by the source record")

    resolved_videos: dict[str, str] = {}
    video_digests: dict[str, str] = {}
    video_metadata = provenance.get("videos", {})
    for media_id in sorted(by_media) if segmented else media_ids:
        try:
            video = resolve_continuous_media_video(
                root,
                source_root,
                media_id,
                context=task["id"],
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
            continue
        resolved_videos[media_id] = video.relative_to(root).as_posix()
        try:
            if media_cache is not None:
                inspected = media_cache.inspect_video(video)
                actual_duration = float(inspected["duration_seconds"])
                actual_width = int(inspected["width"])
                actual_height = int(inspected["height"])
                digest = str(inspected["sha256"])
            else:
                probe = subprocess.run(
                    [
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
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                probed = json.loads(probe.stdout)
                actual_duration = float(probed["format"]["duration"])
                stream = probed["streams"][0]
                actual_width = int(stream["width"])
                actual_height = int(stream["height"])
                digest = hashlib.sha256(video.read_bytes()).hexdigest()
        except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as exc:
            errors.append(f"media_id {media_id!r} cannot be decoded by ffprobe: {exc}")
            actual_duration = 0.0
            actual_width = actual_height = 0
            digest = ""
        video_digests[media_id] = digest
        segment = by_media.get(media_id)
        declared_digest = (
            segment.media_sha256
            if segment is not None
            else declared_source.get("media_sha256")
        )
        if declared_digest is not None and declared_digest != digest:
            errors.append(f"media_id {media_id!r} source video SHA-256 mismatch")

        metadata = (
            video_metadata.get(media_id) if isinstance(video_metadata, dict) else None
        )
        duration = (
            metadata.get("duration_seconds") if isinstance(metadata, dict) else None
        )
        source_start = (
            metadata.get("source_start_seconds") if isinstance(metadata, dict) else None
        )
        if not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"media_id {media_id!r} has no valid provenance duration")
            continue
        if not isinstance(source_start, (int, float)) or source_start < 0:
            errors.append(f"media_id {media_id!r} has no valid source_start_seconds")
        if media_id in resolved_videos and actual_duration > 0:
            if abs(actual_duration - float(duration)) > 0.05:
                errors.append(
                    f"media_id {media_id!r} real duration disagrees with provenance"
                )
            declared_width = (
                metadata.get("width") if isinstance(metadata, dict) else None
            )
            declared_height = (
                metadata.get("height") if isinstance(metadata, dict) else None
            )
            if actual_width != declared_width or actual_height != declared_height:
                errors.append(
                    f"media_id {media_id!r} real dimensions disagree with provenance"
                )
        if segmented and segment is not None:
            if (
                source_start != segment.source_start_seconds
                or metadata.get("source_end_seconds") != segment.source_end_seconds
            ):
                errors.append(
                    f"media_id {media_id!r} provenance source interval disagrees with segment"
                )
            if not math.isclose(
                float(duration),
                segment.source_end_seconds - segment.source_start_seconds,
                abs_tol=0.05,
            ):
                errors.append(
                    f"media_id {media_id!r} segment duration disagrees with provenance"
                )
            if actual_duration > 0 and not math.isclose(
                actual_duration,
                segment.source_end_seconds - segment.source_start_seconds,
                abs_tol=0.05,
            ):
                errors.append(
                    f"media_id {media_id!r} segment duration disagrees with real video"
                )
        for frame in task_frames:
            if frame.get("media_id") != media_id:
                continue
            timestamp = frame.get("media_timestamp_seconds")
            if segmented:
                source_time = frame.get("timestamp_seconds")
                try:
                    owner = media_segment_at(segments, source_time, context=task["id"])
                    if (
                        owner.media_id != media_id
                        or type(timestamp) not in (int, float)
                        or not math.isclose(
                            source_time,
                            owner.source_start_seconds + timestamp,
                            abs_tol=1e-6,
                        )
                    ):
                        errors.append(
                            f"frame {frame.get('packet_id')}:{frame.get('frame_index')} segment offset mismatch"
                        )
                except LineageError as exc:
                    errors.append(str(exc))
            if not isinstance(timestamp, (int, float)) or not 0 <= timestamp < duration:
                errors.append(
                    f"frame {frame.get('packet_id')}:{frame.get('frame_index')} "
                    f"falls outside media_id {media_id!r}"
                )

    jpeg_qscale = provenance.get("frame_extraction", {}).get("jpeg_qscale", 2)
    reextract_mismatches: list[str] = []
    if verify_source_frames and not errors:
        jobs = [
            (frame, Path(resolved_videos[str(frame["media_id"])]))
            for frame in task_frames
            if str(frame.get("media_id")) in resolved_videos
        ]

        def reextract(job: tuple[dict[str, Any], Path]) -> str | None:
            frame, relative_video = job
            video = root / relative_video
            key = f"{frame.get('packet_id')}:{frame.get('frame_index')}"
            try:
                if media_cache is not None:
                    digest = media_cache.frame_sha256(
                        video,
                        float(frame["media_timestamp_seconds"]),
                        jpeg_qscale,
                    )
                else:
                    digest = _extract_frame_sha256(
                        video, float(frame["media_timestamp_seconds"]), jpeg_qscale
                    )
            except (subprocess.CalledProcessError, KeyError, TypeError, ValueError):
                return f"{key} could not be re-extracted"
            if digest != frame.get("sha256"):
                return f"{key} re-extracted bytes disagree with provenance"
            return None

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as executor:
            reextract_mismatches = sorted(
                result for result in executor.map(reextract, jobs) if result is not None
            )
        errors.extend(reextract_mismatches)
    return _check(
        not errors,
        details={
            "media_ids": media_ids,
            "video_paths": resolved_videos,
            "video_sha256": video_digests,
            "source_frame_reextraction": (
                "verified" if verify_source_frames else "skipped"
            ),
            "reextract_mismatches": reextract_mismatches,
        },
        errors=errors,
    )


def _referenced_entity_ids(
    task: Mapping[str, Any],
    source_record: Mapping[str, Any] | None = None,
) -> set[str]:
    entity_ids = set(referenced_entity_ids(task))
    if task["track"] == "refresh":
        blueprint = (source_record or {}).get("construct", {}).get("task_blueprint", {})
        later_role = blueprint.get("later_distractor_role")
        if isinstance(later_role, str) and later_role:
            explicit_ids = blueprint.get("environment_role_ids", {})
            if isinstance(explicit_ids, dict) and later_role in explicit_ids:
                entity_ids.add(str(explicit_ids[later_role]))
            else:
                entity_ids.add(opaque_item_id(str(task["id"]), later_role))
    return entity_ids


def _goals(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        task["goal"],
        *([task["graft"]["initial_goal"]] if task["track"] == "graft" else []),
    ]


def _site_check(
    task: dict[str, Any],
    site_data: dict[str, Any],
    root: Path,
    source_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    entities = {str(entity["entity_id"]): entity for entity in site_data["entities"]}
    required_ids = _referenced_entity_ids(task, source_record)
    missing = sorted(required_ids - set(entities))
    if missing:
        errors.append(f"site is missing referenced entities {missing}")

    frame_hashes = {
        hashlib.sha256(
            resolve_within(
                root, frame, context=f"{task['id']} released frame"
            ).read_bytes()
        ).hexdigest()
        for packet in task["visual_packets"]
        for frame in packet["frame_paths"]
    }
    asset_paths: dict[str, str] = {}
    visual_digests: dict[str, str] = {}
    semantic_ids = set(
        semantic_visual_representative_ids(
            task,
            entities,
            root=root,
        )
    )
    for entity_id in sorted(required_ids - set(missing)):
        visual = entity_visual_path(task["site"], entities[entity_id])
        if not visual:
            errors.append(f"referenced entity {entity_id} has no website image")
            continue
        try:
            asset = resolve_within(
                root, visual, context=f"{task['id']} website image for {entity_id}"
            )
        except (ValueError, FileNotFoundError) as exc:
            errors.append(str(exc))
            continue
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        if digest in frame_hashes:
            errors.append(
                f"website image for {entity_id} duplicates a released source frame"
            )
        if entity_id in semantic_ids:
            previous_entity = visual_digests.get(digest)
            if previous_entity is not None:
                errors.append(
                    f"semantic entities {previous_entity} and {entity_id} "
                    "share the same website visual"
                )
            visual_digests[digest] = entity_id
        asset_paths[entity_id] = visual
        entity = entities[entity_id]
        if task["site"] == "egomedia":
            required = {
                "thumbnail_path",
                "channel_title",
                "published_at",
                "duration_seconds",
                "view_count",
            }
            absent = sorted(required - set(entity))
            if absent:
                errors.append(
                    f"referenced media entity {entity_id} lacks realistic metadata {absent}"
                )

    supported_fields = set(site_data.get("supported_fields", []))
    for goal_index, goal in enumerate(_goals(task)):
        for state_index, state in enumerate(goal["acceptable_states"]):
            selected = [
                entities[entity_id]
                for entity_id in state["selected_entity_ids"]
                if entity_id in entities
            ]
            for field, value in state["fields"].items():
                if field not in supported_fields:
                    errors.append(
                        f"goal {goal_index}/{state_index} uses unsupported field {field!r}"
                    )
                if not any(
                    isinstance(entity.get("facts"), dict)
                    and entity["facts"].get(field) == value
                    for entity in selected
                ):
                    errors.append(
                        f"goal {goal_index}/{state_index} answer {field!r}={value!r} "
                        "is not stated in a selected website entity"
                    )
    return _check(
        not errors,
        details={
            "referenced_entity_ids": sorted(required_ids),
            "website_images": asset_paths,
        },
        errors=errors,
    )


def _instruction_check(
    task: Mapping[str, Any],
    source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    instructions = task.get("instructions")
    if not isinstance(instructions, dict) or not instructions:
        return _check(False, errors=["instructions must be a non-empty object"])
    leakage: dict[str, list[str]] = {}
    token_counts: dict[str, int] = {}
    for name, value in instructions.items():
        if not isinstance(value, str) or not value.strip():
            errors.append(f"instruction {name!r} is empty")
            continue
        tokens = re.findall(r"\b\w+\b", value, re.UNICODE)
        token_counts[str(name)] = len(tokens)
        if len(tokens) < 3 or len(tokens) > 120:
            errors.append(
                f"instruction {name!r} has implausible basic length ({len(tokens)} tokens)"
            )
        matches = sorted({match.group(0) for match in LEAKAGE.finditer(value)})
        if matches:
            leakage[str(name)] = matches
            errors.append(f"instruction {name!r} leaks internal terms {matches}")
        if "[ANNOTATE" in value.upper() or "TODO" in value.upper():
            errors.append(f"instruction {name!r} contains an unfinished placeholder")
    construct = source_record.get("construct", {}) if source_record else {}
    evidence_type = (
        construct.get("evidence_type") if isinstance(construct, Mapping) else None
    )
    update = instructions.get("b")
    benchmark_equivalence_only = bool(
        task.get("track") == "graft"
        and evidence_type == "procedure_stage"
        and isinstance(update, str)
        and PROCEDURE_EQUIVALENCE_WORDING.search(update)
    )
    if benchmark_equivalence_only:
        errors.append(
            "instruction 'b' asks only for scene-equivalent procedure matching; "
            "state a concrete user problem, constraint, desired outcome, or risk instead"
        )
    return _check(
        not errors,
        details={
            "token_counts": token_counts,
            "leakage_matches": leakage,
            "wording_flags": {
                str(name): sorted(
                    {
                        match.group(0)
                        for match in REVIEW_WORDING_FLAGS.finditer(str(value))
                    }
                )
                for name, value in instructions.items()
                if REVIEW_WORDING_FLAGS.search(str(value))
            },
            "utility_gate": {
                "evidence_type": evidence_type,
                "benchmark_equivalence_only": benchmark_equivalence_only,
            },
            "semantic_naturalness_requires_semantic_review": True,
        },
        errors=errors,
    )


def _positive_record(
    task: dict[str, Any],
    condition: Condition,
    state: Mapping[str, Any],
    *,
    default_release_step: int,
    reference_steps: int | None,
    clarification_question: str | None = None,
) -> dict[str, Any]:
    timeline = Timeline.compile(
        task,
        condition,
        default_release_step=default_release_step,
        reference_steps=reference_steps,
    )
    actions = [release.as_event() for release in timeline.releases]
    step = max((release.step for release in timeline.releases), default=0) + 1
    if timeline.offer is not None and timeline.sufficiency_step() is None:
        slot = str(timeline.offer.valid_slot)
        actions.append(
            timeline.offer.answer(
                clarification_question or CLARIFICATION_QUESTIONS[slot],
                step=1,
            )
        )
        step = max(step, 2)
    actions.extend(
        {"type": "select_entity", "step": step, "entity_id": entity_id}
        for entity_id in state["selected_entity_ids"]
    )
    actions.append(
        {
            "type": "submit",
            "step": step + 1,
            "entity_ids": list(state["selected_entity_ids"]),
        }
    )
    return {
        "task_id": task["id"],
        "track": task["track"],
        "condition": condition.as_dict(),
        "meta": {
            "agent": "acceptance-audit",
            "model": "deterministic",
            "provider": "local",
            "model_version": "deterministic",
            "transport": "in-process",
            "replicate_id": 0,
            "schedule_index": None,
            "provider_seed": None,
            "randomness_control": "uncontrolled",
            "max_steps": max(step + 1, 1),
            "task_sha256": "acceptance-audit",
            "site": task["site"],
            "site_data_sha256": "acceptance-audit",
            "asset_sha256": {},
            "runtime": {},
            "operator_id": None,
            "default_release_step": default_release_step,
            "reference_steps": reference_steps,
            "started_at": 0.0,
            "duration_seconds": 0.0,
            "event_delivery_steps": [
                action["step"]
                for action in actions
                if action.get("type")
                in {
                    "instruction_injection",
                    "visual_context_update",
                }
            ],
            "run_id": None,
            "artifact_directory": None,
        },
        "run_status": {"valid": True, "reason": None},
        "initial_state": {
            "selected_entity_ids": [],
            "fields": {},
            "submission_completed": False,
        },
        "steps": step + 1,
        "actions": actions,
        "final_state": copy.deepcopy(dict(state)),
        "live_state": {
            "selected_entity_ids": list(state["selected_entity_ids"]),
            "fields": dict(state["fields"]),
        },
        "trace": [],
    }


def _positive_states(goal: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Replay witnesses and a concrete legal completion containing each package."""
    states = copy.deepcopy(goal["acceptable_states"])
    seen = {
        (tuple(sorted(s["selected_entity_ids"])), tuple(sorted(s["fields"].items())))
        for s in states
    }
    for witness in goal["acceptable_states"]:
        counts = normalized_entity_counts(goal, witness["selected_entity_ids"])
        for unit, packages in goal.get("unit_equivalences", {}).items():
            total = counts[("unit", unit)]
            paths: dict[int, list[str]] = {0: []}
            for amount in range(1, total + 1):
                for sku, quantity in packages.items():
                    if amount - quantity in paths:
                        paths[amount] = paths[amount - quantity] + [sku]
                        break
            base = [
                sku for sku in witness["selected_entity_ids"] if sku not in packages
            ]
            for sku, quantity in packages.items():
                if total - quantity not in paths:
                    continue
                selected = base + [sku] + paths[total - quantity]
                key = (
                    tuple(sorted(selected)),
                    tuple(sorted(witness["fields"].items())),
                )
                if key not in seen:
                    states.append(
                        {
                            "selected_entity_ids": selected,
                            "fields": dict(witness["fields"]),
                        }
                    )
                    seen.add(key)
    return states


def _checker_check(
    task: dict[str, Any],
    site_data: dict[str, Any],
    plan: ExperimentPlan,
    source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    positive_runs = 0
    negative_cases = 0
    canonical_question = None
    if task["track"] == "clarify":
        raw_question = (
            (source_record or {})
            .get("construct", {})
            .get("task_blueprint", {})
            .get("canonical_question")
        )
        if isinstance(raw_question, str) and raw_question.strip():
            canonical_question = raw_question.strip()
    all_site_ids = [str(entity["entity_id"]) for entity in site_data["entities"]]
    condition_names = plan.conditions_for(task["track"])
    for condition_name in condition_names:
        condition = plan.condition(condition_name)
        goal = goal_for_condition(task, condition)
        states = _positive_states(goal)
        records: list[dict[str, Any]] = []
        for state_index, state in enumerate(states):
            record = _positive_record(
                task,
                condition,
                state,
                default_release_step=plan.default_release_step,
                reference_steps=plan.reference_steps,
                clarification_question=canonical_question,
            )
            records.append(record)
            checked = check_run(task, record)
            if not (
                checked["run_valid"]
                and checked["evidence_status"] == "PASS"
                and checked["e2e_success"] is True
            ):
                errors.append(
                    f"production checker rejects valid {condition_name} "
                    f"alternative {state_index}"
                )
            positive_runs += 1

            unsubmitted = copy.deepcopy(record)
            unsubmitted["actions"] = [
                action for action in record["actions"] if action["type"] != "submit"
            ]
            unsubmitted["final_state"] = {
                "selected_entity_ids": [],
                "fields": {},
                "submission_completed": False,
            }
            unsubmitted_result = check_run(task, unsubmitted)
            if not (
                unsubmitted_result["run_valid"] and unsubmitted_result["e2e_success"]
            ):
                errors.append(
                    f"production checker rejects {condition_name}/without_extra_submit"
                )
            positive_runs += 1

        state = records[0]["live_state"]
        record = records[0]

        acceptable_ids = goal_entity_ids(goal)
        decoy = next(
            (
                entity_id
                for entity_id in all_site_ids
                if entity_id not in acceptable_ids
            ),
            None,
        )
        if decoy is None:
            errors.append(
                f"{condition_name} has no website decoy for negative checker replay"
            )
            continue
        mutations = {
            "wrong_entity": {**state, "selected_entity_ids": [decoy]},
            "target_plus_extra": {
                **state,
                "selected_entity_ids": [*state["selected_entity_ids"], decoy],
            },
            "extra_field": {
                **state,
                "fields": {**state["fields"], "unexpected_field": "unexpected"},
            },
        }
        if state["selected_entity_ids"]:
            mutations.update(
                {
                    "duplicate_entity": {
                        **state,
                        "selected_entity_ids": [
                            *state["selected_entity_ids"],
                            state["selected_entity_ids"][0],
                        ],
                    },
                    "missing_entity": {
                        **state,
                        "selected_entity_ids": state["selected_entity_ids"][1:],
                    },
                }
            )
        for name, live_state in mutations.items():
            if evaluate_final_state(goal, live_state):
                continue
            mutated = copy.deepcopy(record)
            mutated["live_state"] = live_state
            rejected = check_run(task, mutated)
            if not (
                rejected["evidence_status"] == "FAIL"
                and rejected["e2e_success"] is False
            ):
                errors.append(f"production checker accepts {condition_name}/{name}")
            negative_cases += 1

        timeline = Timeline.compile(
            task,
            condition,
            default_release_step=plan.default_release_step,
            reference_steps=plan.reference_steps,
        )
        scheduled_event_types = {
            release.as_event()["type"] for release in timeline.releases
        }
        if scheduled_event_types:
            missing_event = copy.deepcopy(record)
            missing_event["actions"] = [
                action
                for action in missing_event["actions"]
                if action["type"] not in scheduled_event_types
            ]
            if check_run(task, missing_event)["run_valid"] is not False:
                errors.append(
                    f"production checker accepts {condition_name}/missing_event"
                )
            negative_cases += 1

        if any(
            action.get("type") == "clarification_answer" for action in record["actions"]
        ):
            without_question = copy.deepcopy(record)
            without_question["actions"] = [
                action
                for action in record["actions"]
                if action.get("type") != "clarification_answer"
            ]
            no_question_result = check_run(task, without_question)
            if not (
                no_question_result["run_valid"] and no_question_result["e2e_success"]
            ):
                errors.append(
                    f"production checker rejects {condition_name}/correct_without_question"
                )
            positive_runs += 1
            wrong_target = next(
                candidate
                for candidate in task["clarification"]["candidate_ids"]
                if candidate not in state["selected_entity_ids"]
            )
            ignored_answer = copy.deepcopy(record)
            ignored_answer["actions"] = [
                ignored_answer["actions"][0],
                {"type": "select_entity", "step": 2, "entity_id": wrong_target},
                ignored_answer["actions"][-1],
            ]
            ignored_answer["live_state"]["selected_entity_ids"] = [wrong_target]
            ignored_result = check_run(task, ignored_answer)
            if ignored_result.get("e2e_success") is not False:
                errors.append("production checker accepts online/ignored_answer")
            negative_cases += 1
    return _check(
        not errors,
        details={
            "positive_paths": positive_runs,
            "adversarial_paths": negative_cases,
            "conditions": list(condition_names),
            "clarification_question_replayed": canonical_question,
        },
        errors=errors,
    )


def _current_review_state(
    result: Mapping[str, Any],
    *,
    required: int,
) -> dict[str, Any]:
    status = result.get("status")
    reviewers = sorted(set(str(item) for item in result.get("reviewers", [])))
    current_admits = reviewers if status == "admit" else []
    current_rejects = reviewers if status == "reject" else []
    stale_or_invalid = (
        [str(item) for item in result.get("errors", [])] if status == "invalid" else []
    )
    return {
        "required_independent_reviews": required,
        "current_admits": current_admits,
        "current_rejects": current_rejects,
        "stale_or_invalid": stale_or_invalid,
        "complete": status == "admit" and len(current_admits) >= required,
    }


def _advisory(
    ok: bool, *, details: Mapping[str, Any] | None = None, errors: Iterable[str] = ()
) -> dict[str, Any]:
    """Like _check, but a failing advisory check warns instead of failing integrity."""

    messages = [str(error) for error in errors if str(error)]
    return {
        "status": "pass" if ok and not messages else "warn",
        **({"details": dict(details)} if details is not None else {}),
        **({"warnings": messages} if messages else {}),
    }


def _text_pinning_check(
    task: Mapping[str, Any],
    site_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Flag lexical shortcuts for semantic review, without deciding visual necessity."""

    catalog = catalog_tokens(str(task["site"]), site_data)
    accepted = set(accepted_ids(task))
    targets: dict[str, dict[str, Any]] = {}
    for entity_id, words in target_predicates(task, catalog).items():
        if len(words) < MIN_PREDICATE_WORDS:
            targets[entity_id] = {"words": words, "hits": None, "pinned": False}
            continue
        required = set(words)
        hits = sorted(other for other, tokens in catalog.items() if required <= tokens)
        targets[entity_id] = {
            "words": words,
            "hits": len(hits),
            "pinned": set(hits) <= accepted,
        }
    pinned = sorted(entity_id for entity_id, row in targets.items() if row["pinned"])
    details = {"targets": targets, "pinned": pinned}
    return _advisory(
        not pinned,
        details=details,
        errors=[
            f"released words alone retrieve {eid} by {targets[eid]['words']}"
            for eid in pinned
        ],
    )


def _answer_closure_check(
    task: Mapping[str, Any],
    site_data: Mapping[str, Any],
    source_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Expose unresolved semantic evidence without treating it as an integrity defect."""

    construct = (source_record or {}).get("construct")
    resolution = (
        construct.get("catalog_resolution") if isinstance(construct, Mapping) else None
    )
    if not isinstance(resolution, Mapping):
        return _check(True, details={"window": None})
    catalog = catalog_tokens(str(task["site"]), site_data)
    undecided = undecided_window_listings(task, resolution, catalog)
    boundary = resolution.get("answer_boundary_review") or {}
    unresolved = boundary.get("unresolved_candidates") or []
    details = {
        "window": len(resolution.get("window_entity_ids") or []),
        "undecided": undecided,
        "unresolved_candidates": unresolved,
    }
    questions = []
    if undecided:
        questions.append(
            f"{len(undecided)} unreviewed window listings share an accepted listing's "
            "released words; inspect their actual applicability"
        )
    if unresolved:
        questions.append(
            f"{len(unresolved)} candidates await a semantic answer-boundary decision"
        )
    return _advisory(
        not undecided and not unresolved,
        details=details,
        errors=questions,
    )


SHORTCUT_SUCCESS_THRESHOLD = 2 / 3
TEMPTING_MINIMUM_RATE = 2 / 3


def _necessity_check(
    task: Mapping[str, Any],
    root: Path,
    plan: ExperimentPlan,
) -> dict[str, Any]:
    """Advisory: progressive-evidence necessity from registered probe conditions.

    Reads `artifacts/reports/necessity.json` (per-task outcomes of the probe
    conditions, produced from scored Run Artifacts). Pending until that report exists.
    """

    probe_names: list[str] = []
    for standard_name in ("text_only", "first_frame"):
        standard = STANDARD_CONDITIONS[standard_name]
        matches = [
            condition.name
            for condition in plan.conditions.values()
            if (condition.schedule, condition.scope, condition.ask)
            == (standard.schedule, standard.scope, standard.ask)
        ]
        if len(matches) != 1:
            return _advisory(
                False,
                errors=[
                    (
                        "experiment plan must define exactly one probe matching "
                        f"the standard {standard_name} factors"
                    )
                ],
            )
        probe_names.append(matches[0])
    report_names = (*probe_names, plan.reference_condition_name)
    report_path = root / NECESSITY_REPORT
    if not report_path.is_file():
        return _advisory(
            True, details={"status": "pending", "report": NECESSITY_REPORT}
        )
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return _advisory(False, errors=[f"{NECESSITY_REPORT} is not valid JSON"])
    row = (
        (report.get("tasks") or {}).get(str(task["id"]))
        if isinstance(report, dict)
        else None
    )
    if not isinstance(row, Mapping):
        return _advisory(
            True, details={"status": "pending", "report": NECESSITY_REPORT}
        )
    shortcut = [
        name
        for name in probe_names
        if isinstance(row.get(name), Mapping)
        and (row[name].get("e2e_success_rate") or 0) >= SHORTCUT_SUCCESS_THRESHOLD
    ]
    warnings = [f"shortcut condition {name} solves the task" for name in shortcut]
    initial_only_names = _initial_only_condition_names(plan)
    warnings.extend(_tempting_warnings(row, initial_only_names))
    return _advisory(
        not warnings,
        details={
            "status": "assessed",
            **{name: row.get(name) for name in (*report_names, *initial_only_names)},
        },
        errors=warnings,
    )


def _initial_only_condition_names(plan: ExperimentPlan) -> list[str]:
    standard = STANDARD_CONDITIONS["initial_only"]
    return [
        condition.name
        for condition in plan.conditions.values()
        if (condition.schedule, condition.scope) == (standard.schedule, standard.scope)
    ]


def _tempting_warnings(row: Mapping[str, Any], condition_names: list[str]) -> list[str]:
    """Warn when the registered tempting candidate fails to tempt the reference model."""

    warnings: list[str] = []
    for name in condition_names:
        probe = row.get(name)
        if not isinstance(probe, Mapping):
            continue
        tempting_rate = probe.get("tempting_pick_rate")
        if tempting_rate is not None and tempting_rate < TEMPTING_MINIMUM_RATE:
            warnings.append(
                f"tempting candidate does not tempt: {name} picked it in "
                f"{tempting_rate:.0%} of runs (need >= {TEMPTING_MINIMUM_RATE:.0%})"
            )
    return warnings


def _audit_task(
    task: dict[str, Any],
    root: Path,
    *,
    verify_source_frames: bool,
    is_primary: bool,
    plan: ExperimentPlan,
    review_result: Mapping[str, Any],
    required_reviews: int,
    media_cache: MediaVerificationCache | None = None,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    site_data: dict[str, Any] | None = None
    dossier: dict[str, Any] | None = None
    source_record: dict[str, Any] | None = None
    provenance_path: Path | None = None

    try:
        site_data = load_site_data(root, task["site"])
        validate_task(task, site_data, root=root, require_assets=True)
        checks["schema_and_frames"] = _check(
            True,
            details={
                "frame_count": _released_frame_count(task),
            },
        )
    except Exception as exc:  # one task must not hide the remaining audit rows
        checks["schema_and_frames"] = _check(False, errors=[str(exc)])

    lineage_errors: list[str] = []
    try:
        dossier, source_record, _, provenance_path = _source_material(task, root)
    except Exception as exc:
        lineage_errors.append(str(exc))
    try:
        validate_source_lineage(task, root=root)
    except Exception as exc:
        lineage_errors.append(str(exc))
    if source_record is not None:
        lineage_errors.extend(_claim_errors(source_record))
    checks["source_lineage"] = _check(
        not lineage_errors,
        details={
            "dossier": task.get("source", {}).get("dossier"),
            "frame_provenance": task.get("source", {}).get("frame_provenance"),
            "event_origin": (
                source_record.get("construct", {}).get("event_origin")
                if source_record is not None
                else None
            ),
        },
        errors=lineage_errors,
    )

    if source_record is not None:
        checks["glass_use_construct_prerequisites"] = _construct_check(
            task, source_record, is_primary=is_primary
        )
    else:
        checks["glass_use_construct_prerequisites"] = _check(
            False, errors=["source construct contract is unavailable"]
        )

    if (
        dossier is not None
        and source_record is not None
        and provenance_path is not None
    ):
        try:
            checks["continuous_video"] = _video_check(
                task,
                dossier,
                source_record,
                provenance_path,
                root,
                verify_source_frames=verify_source_frames,
                media_cache=media_cache,
            )
        except Exception as exc:
            checks["continuous_video"] = _check(False, errors=[str(exc)])
    else:
        checks["continuous_video"] = _check(
            False, errors=["source lineage is unavailable"]
        )

    if site_data is not None:
        try:
            checks["site_entities_images_and_answers"] = _site_check(
                task, site_data, root, source_record
            )
        except Exception as exc:
            checks["site_entities_images_and_answers"] = _check(
                False, errors=[str(exc)]
            )
        if source_record is not None:
            try:
                rebuilt = compile_task_from_site(
                    resolve_authored_record(source_record, root=root),
                    source_identity=task["source"],
                    site_data=site_data,
                    root=root,
                )
                checks["compiled_from_source_truth"] = _check(
                    rebuilt == task,
                    errors=(
                        []
                        if rebuilt == task
                        else [
                            "task JSON differs from the current source blueprint/site mapping"
                        ]
                    ),
                )
            except Exception as exc:
                checks["compiled_from_source_truth"] = _check(False, errors=[str(exc)])
        else:
            checks["compiled_from_source_truth"] = _check(
                False, errors=["source record is unavailable"]
            )
        try:
            checks["production_checker"] = _checker_check(
                task, site_data, plan, source_record
            )
        except Exception as exc:
            checks["production_checker"] = _check(False, errors=[str(exc)])
    else:
        for name in (
            "site_entities_images_and_answers",
            "compiled_from_source_truth",
            "production_checker",
        ):
            checks[name] = _check(False, errors=["site data is unavailable"])

    if site_data is not None:
        checks["text_pinning"] = _text_pinning_check(task, site_data)
        checks["answer_closure"] = _answer_closure_check(task, site_data, source_record)
    else:
        checks["text_pinning"] = _check(False, errors=["site data is unavailable"])
        checks["answer_closure"] = _check(False, errors=["site data is unavailable"])
    checks["necessity"] = _necessity_check(task, root, plan)

    checks["instruction_form_and_leakage"] = _instruction_check(task, source_record)
    failed_checks = sorted(
        name for name, result in checks.items() if result["status"] == "fail"
    )
    integrity_pass = not failed_checks
    review = _current_review_state(
        review_result,
        required=required_reviews,
    )
    pending_source_items = []
    if source_record is not None:
        pending_source_items = [
            *[str(item) for item in source_record.get("blocking_checks", [])],
            *[str(item) for item in source_record.get("unresolved", [])],
        ]

    if failed_checks:
        status = "fail"
    elif pending_source_items:
        status = "needs_source_review"
    else:
        status = "candidate_ready"
    if status not in STATUS_VALUES:
        raise AssertionError(f"unexpected acceptance status: {status}")
    site_scope_hash = None
    if site_data is not None:
        try:
            site_scope_hash = task_site_execution_sha256(task, site_data)
        except Exception:  # the corresponding site check already records the failure
            site_scope_hash = None
    search_scope_hash = None
    source_construct = (
        source_record.get("construct") if source_record is not None else None
    )
    if isinstance(source_construct, Mapping) and isinstance(
        source_construct.get("catalog_resolution"), Mapping
    ):
        try:
            search_scope_hash = search_evidence_sha256(
                task_id=str(task["id"]),
                site=str(task["site"]),
                resolution=source_construct["catalog_resolution"],
                root=root,
            )
        except (OSError, ValueError):
            search_scope_hash = None
    return {
        "task_id": task["id"],
        "task_sha256": task_sha256(task),
        "task_site_execution_sha256": site_scope_hash,
        "search_evidence_sha256": search_scope_hash,
        "track": task["track"],
        "site": task["site"],
        "pilot_scope": "primary_seed" if is_primary else "rescreening_candidate",
        "status": status,
        "integrity_pass": integrity_pass,
        "failed_checks": failed_checks,
        "pending_source_review_items": pending_source_items,
        "human_review": review,
        "checks": checks,
    }


def _batch_quality(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_goal: dict[str, list[str]] = {}
    by_instruction: dict[str, list[str]] = {}
    for task in tasks:
        goal_key = json.dumps(
            {"site": task["site"], "goal": task["goal"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        by_goal.setdefault(goal_key, []).append(str(task["id"]))
        for value in task.get("instructions", {}).values():
            if isinstance(value, str):
                normalized = " ".join(value.casefold().split())
                by_instruction.setdefault(normalized, []).append(str(task["id"]))
    duplicate_goals = [sorted(ids) for ids in by_goal.values() if len(ids) > 1]
    repeated_instruction_ids = [
        sorted(set(ids)) for ids in by_instruction.values() if len(set(ids)) > 1
    ]
    duplicate_instructions: list[tuple[str, ...]] = sorted(
        {tuple(ids) for ids in repeated_instruction_ids},
        key=lambda ids: (len(ids), ids),
    )
    web_filter_updates = [
        row["task_id"]
        for row in rows
        if any(
            "ordinary website filter" in reason
            for reason in row["checks"]["glass_use_construct_prerequisites"]
            .get("details", {})
            .get("construct_risk_reasons", [])
        )
    ]
    return {
        "duplicate_final_goal_groups": sorted(duplicate_goals),
        "duplicate_instruction_groups": [list(ids) for ids in duplicate_instructions],
        "graft_web_filter_only_risk": sorted(web_filter_updates),
        "interpretation": (
            "These are deterministic diversity warnings, not semantic rejection decisions. "
            "A human must still judge whether two tasks measure meaningfully different situated decisions."
        ),
    }


def build_acceptance_report(
    root: str | Path = ".",
    tasks_path: str | Path = TASKS,
    *,
    verify_source_frames: bool = True,
    task_ids: Iterable[str] | None = None,
    sites: Iterable[str] | None = None,
    previous_report: Mapping[str, Any] | None = None,
    media_cache: MediaVerificationCache | None = None,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    task_root = resolve_within(
        workspace, tasks_path, context="task acceptance input", must_exist=False
    )
    tasks = load_construction_tasks(workspace, list(task_ids or []), tasks_path=task_root)
    plan = load_experiment_plan(workspace / TASK_REVIEW_PLAN)
    selection = load_selection(workspace)
    seed_ids = set(selected_task_ids(selection))
    discovered_ids = {str(task["id"]) for task in tasks}
    if not seed_ids <= discovered_ids:
        missing = sorted(seed_ids - discovered_ids)
        raise ValueError(
            f"selected tasks are missing from the constructed pool: {missing}"
        )
    requested_ids = {str(task_id) for task_id in (task_ids or [])}
    requested_sites = {str(site) for site in (sites or [])}
    unknown_ids = sorted(requested_ids - discovered_ids)
    known_sites = {str(task["site"]) for task in tasks}
    unknown_sites = sorted(requested_sites - known_sites)
    if unknown_ids:
        raise ValueError(f"unknown task IDs: {unknown_ids}")
    if unknown_sites:
        raise ValueError(f"unknown sites: {unknown_sites}")

    incremental = bool(requested_ids or requested_sites)
    previous_rows: dict[str, dict[str, Any]] = {}
    if incremental:
        raw_previous = (previous_report or {}).get("tasks")
        if not isinstance(raw_previous, Mapping):
            raise ValueError(
                "incremental acceptance requires an existing complete report"
            )
        previous_rows = {
            str(task_id): dict(row)
            for task_id, row in raw_previous.items()
            if isinstance(row, Mapping)
        }

    audit_ids = (
        set(discovered_ids)
        if not incremental
        else requested_ids
        | {str(task["id"]) for task in tasks if str(task["site"]) in requested_sites}
    )
    review_spec = _read_object(
        workspace / "data/reviews/construction.json",
        "construction review spec",
    )
    required_reviews = review_spec["reviewer_policy"]["required_independent_reviews"]
    review_results = audit_construction_reviews(
        root=workspace,
        include_all_tasks=True,
        task_ids=audit_ids,
    )["tasks"]
    audited_rows = [
        _audit_task(
            task,
            workspace,
            verify_source_frames=verify_source_frames,
            is_primary=str(task["id"]) in seed_ids,
            plan=plan,
            review_result=review_results[str(task["id"])],
            required_reviews=required_reviews,
            media_cache=media_cache,
        )
        for task in tasks
        if str(task["id"]) in audit_ids
    ]
    rows_by_id = previous_rows if incremental else {}
    rows_by_id.update({row["task_id"]: row for row in audited_rows})
    missing_rows = sorted(discovered_ids - set(rows_by_id))
    if missing_rows:
        raise ValueError(
            f"incremental acceptance report is missing unchanged tasks: {missing_rows}"
        )
    rows = list(rows_by_id.values())
    statuses = Counter(row["status"] for row in rows)
    tracks = Counter(row["track"] for row in rows)
    sites = Counter(row["site"] for row in rows)
    return {
        "schema_version": REPORTS["acceptance"].schema_version,
        "scope": "all_constructed_tasks",
        "audit_scope": {
            "mode": "incremental" if incremental else "full",
            "requested_task_ids": sorted(requested_ids),
            "requested_sites": sorted(requested_sites),
            "audited_task_ids": [row["task_id"] for row in audited_rows],
            "preserved_task_count": len(rows) - len(audited_rows),
        },
        "admission_authority": False,
        "source_frame_reextraction": verify_source_frames,
        "interpretation": (
            "Integrity checks prove file, lineage, website-state and checker consistency only. "
            "Construct validity uses the released scene, updates and public website evidence; "
            "success with upfront information or one frame does not itself invalidate a task. "
            "The skill-executing multimodal coding agent "
            "reads these materials directly; deterministic warnings never determine "
            "confidence. candidate_ready means structurally ready for confidence routing, not "
            "admitted or release-ready."
        ),
        "summary": {
            "tasks": len(rows),
            "primary_seed_tasks": len(seed_ids),
            "rescreening_candidates": len(rows) - len(seed_ids),
            "integrity_checks_passed": sum(row["integrity_pass"] for row in rows),
            "construct_risk_flagged": sum(
                bool(
                    row["checks"]["glass_use_construct_prerequisites"]
                    .get("details", {})
                    .get("construct_risk_reasons", [])
                )
                for row in rows
            ),
            "advisory_warnings": {
                name: sum(
                    row["checks"].get(name, {}).get("status") == "warn" for row in rows
                )
                for name in ("text_pinning", "answer_closure", "necessity")
            },
            "design_gate_failures": {
                name: sum(
                    row["checks"].get(name, {}).get("status") == "fail" for row in rows
                )
                for name in ("text_pinning", "answer_closure")
            },
            "statuses": {
                status: statuses.get(status, 0) for status in sorted(STATUS_VALUES)
            },
            "tracks": dict(sorted(tracks.items())),
            "sites": dict(sorted(sites.items())),
        },
        "batch_quality": _batch_quality(tasks, rows),
        "tasks": {row["task_id"]: row for row in rows},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every constructed task against source video, released frames, frozen "
            "website truth, exact goals, and the production checker."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument("--output", default=REPORTS["acceptance"].path.as_posix())
    parser.add_argument(
        "--task-id",
        action="append",
        default=[],
        help="Audit and merge one affected task. Repeat for multiple tasks.",
    )
    parser.add_argument(
        "--site",
        action="append",
        default=[],
        help="Audit and merge every task on an affected site.",
    )
    parser.add_argument(
        "--media-cache",
        default="artifacts/cache/task-acceptance-media.json",
        help="Persistent cache for immutable video probe, hash, and frame checks.",
    )
    args = parser.parse_args()
    workspace = Path(args.root).resolve()
    output = resolve_within(
        workspace, args.output, context="task acceptance output", must_exist=False
    )
    previous_report = None
    if (args.task_id or args.site) and output.is_file():
        previous_report = _read_object(output, "previous task acceptance report")
    cache_path = resolve_within(
        workspace,
        args.media_cache,
        context="media verification cache",
        must_exist=False,
    )
    media_cache = MediaVerificationCache(cache_path)
    report = build_acceptance_report(
        workspace,
        args.tasks,
        task_ids=args.task_id,
        sites=args.site,
        previous_report=previous_report,
        media_cache=media_cache,
    )
    write_json_atomic(output, report)
    media_cache.save()
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["summary"]["statuses"]["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
