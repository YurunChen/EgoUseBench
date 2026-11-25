"""Build one reproducible, high-recall queue for the next screening pass.

The queue contains hypotheses, not benchmark tasks. The original version-2
signals exclude selected or version-2-screened media; Ego2Web also excludes
annotation events already rejected in source manifests. The independently
versioned narration signal excludes selected sources and previously rejected
events, and retains prior rejection references for other events in that source.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

from egocua.workspace import read_json_strict
from egocua.lineage import file_sha256
from egocua.workspace import write_json_atomic
from tools.mine.mine_divergence_events import mine as mine_ego2web
from tools.mine.mine_hdepic_decisions import mine as mine_hdepic
from tools.mine.mine_hdepic_refresh import find_hits as find_hdepic_refresh_hits

EGO2WEB_ANNOTATIONS = "reference/sources/Ego2Web/dataset/ego2web.json"
EGO2WEB_VIDEOS = "reference/sources/Ego2Web/dataset/ego2web_video"
HDEPIC_VIDEOS = "reference/sources/HD-EPIC/videos/HD-EPIC/Videos"
HDEPIC_NARRATIONS = "reference/sources/HD-EPIC/annotations/narrations-and-action-segments/HD_EPIC_Narrations.pkl"
HDEPIC_CLIP_ROOTS = (
    "reference/sources/HD-EPIC/clips-mining",
    "reference/sources/HD-EPIC/clips-screened",
)
HDEPIC_CLIP_NAME = re.compile(
    r"(?P<video>.+)__(?P<start>\d+(?:\.\d+)?)-(?P<end>\d+(?:\.\d+)?)"
)
NARRATED_CHOICE_SIGNAL = "refresh.narrated_choice_change"
CAPTAIN_ANNOTATIONS = (
    "reference/sources/CaptainCook4D/annotations/annotation_json/error_annotations.json"
)
CAPTAIN_VIDEOS = "reference/sources/CaptainCook4D/videos/360p"
DEFAULT_OUTPUT = "data/sources/audits/expansion_candidate_pool.json"
SCREENING_DECISIONS_GLOB = "expansion_screening_decisions_v*.json"


def _media_available(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _recording_is_listed(recording: str, media_ids: set[str]) -> bool:
    return any(
        media == recording or media.startswith(f"{recording}__") for media in media_ids
    )


def _selected_media(root: Path) -> set[str]:
    selection = read_json_strict(root / "data/selection.json")
    selected = {
        str(task_id)
        for task_ids in selection.get("task_set", {}).values()
        for task_id in task_ids
    }
    media: set[str] = set()
    for relative_path in selection.get("source_manifests", []):
        manifest = read_json_strict(root / str(relative_path))
        for task in manifest.get("tasks", []):
            if str(task.get("task_id")) not in selected:
                continue
            source = task.get("source") or {}
            media_id = source.get("original_video_id") or source.get("media_id")
            if media_id:
                media.add(str(media_id))
    return media


def _screened_expansion_media(root: Path) -> set[str]:
    screened: set[str] = set()
    audit_dir = root / "data/sources/audits"
    for path in sorted(audit_dir.glob(SCREENING_DECISIONS_GLOB)):
        decisions = read_json_strict(path)
        if decisions.get("candidate_pool_schema_version") != 2:
            continue
        screened.update(
            str(media_id)
            for dataset in decisions.get("datasets", {}).values()
            for media_id in dataset.get("media_ids", [])
        )
    return screened


def _ego2web_rejected_annotations(root: Path) -> set[tuple[str, str]]:
    rejected: set[tuple[str, str]] = set()
    for path in sorted((root / "data/sources/manifests").glob("*.json")):
        manifest = read_json_strict(path)
        if manifest.get("dataset", {}).get("name") != "Ego2Web":
            continue
        for task in manifest.get("tasks", []):
            if task.get("status") != "rejected":
                continue
            source = task["source"]
            media_id = str(source.get("original_video_id") or source["media_id"])
            rejected.update(
                (media_id, str(record_id)) for record_id in source.get("record_ids", [])
            )
    return rejected


def _ego2web_candidates(root: Path, screened: set[str]) -> list[dict[str, Any]]:
    rows = read_json_strict(root / EGO2WEB_ANNOTATIONS)
    report = mine_ego2web(rows)
    rejected = _ego2web_rejected_annotations(root)
    grouped: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any], signal: str) -> None:
        media_id = str(row.get("media_id") or row["vid"])
        annotation_ids = (
            [row["annotation_id"]] if "annotation_id" in row else row["ids"]
        )
        if media_id in screened or any(
            (media_id, str(record_id)) in rejected for record_id in annotation_ids
        ):
            return
        entry = grouped.setdefault(
            media_id,
            {
                "dataset": "Ego2Web",
                "media_id": media_id,
                "media_path": f"{EGO2WEB_VIDEOS}/{media_id}.mp4",
                "signals": [],
                "annotations": [],
                "stage": "metadata_mined",
            },
        )
        if signal not in entry["signals"]:
            entry["signals"].append(signal)
        if "annotation_id" in row:
            annotation = {
                "annotation_id": row["annotation_id"],
                "ranges": row["ranges"],
                "instruction": row["instruction"],
            }
        else:
            annotation = {
                "annotation_ids": row["ids"],
                "ranges": row["ranges"],
                "instructions": row["instructions"],
            }
        if annotation not in entry["annotations"]:
            entry["annotations"].append(annotation)

    pool = report["subtype_candidate_pool"]
    for row in pool["graft"]["refine"]:
        add(row, "graft.refine")
    for candidates in pool["graft"]["augment"].values():
        for row in candidates:
            add(row, "graft.augment")
    for subtype, candidates in pool["clarify"].items():
        for row in candidates:
            add(row, f"clarify.{subtype}")

    rows_by_id = {row["id"]: row for row in rows}
    for pair in report["pair_candidates"]:
        if not pair["shared_tokens"] or pair["vid"] in screened:
            continue
        if any(
            (str(pair["vid"]), str(record_id)) in rejected for record_id in pair["ids"]
        ):
            continue
        for annotation_id in pair["ids"]:
            row = rows_by_id[annotation_id]
            add(
                {
                    "media_id": row["vid"],
                    "annotation_id": row["id"],
                    "ranges": pair["ranges"],
                    "instruction": row["instruction"],
                },
                "refresh.same_video_related_pair",
            )

    candidates = sorted(grouped.values(), key=lambda row: row["media_id"])
    return [row for row in candidates if _media_available(root / row["media_path"])]


def _hdepic_candidates(root: Path, screened: set[str]) -> list[dict[str, Any]]:
    report = mine_hdepic(root)
    candidates = []
    for row in report["candidates"]:
        recording = row["recording"]
        if _recording_is_listed(recording, screened):
            continue
        participant = recording.split("-", 1)[0]
        media_path = f"{HDEPIC_VIDEOS}/{participant}/{recording}.mp4"
        candidates.append(
            {
                "dataset": "HD-EPIC",
                "media_id": recording,
                "media_path": media_path,
                "media_available": _media_available(root / media_path),
                "signals": ["clarify.sequence", "refresh.referent_switch"],
                "kind": row["kind"],
                "window_seconds": row["window_seconds"],
                "objects": row["objects"],
                "stage": "structural_annotation_mined",
            }
        )
    return candidates


def _hdepic_prior_rejections(root: Path) -> dict[str, list[dict[str, Any]]]:
    rejected: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((root / "data/sources/manifests").glob("*.json")):
        manifest = read_json_strict(path)
        if manifest.get("dataset", {}).get("name") != "HD-EPIC":
            continue
        for task in manifest.get("tasks", []):
            if task.get("status") != "rejected":
                continue
            source = task["source"]
            recording = (
                source.get("original_video_id") or source["media_id"].split("__")[0]
            )
            clip = HDEPIC_CLIP_NAME.fullmatch(source["media_id"])
            rejected.setdefault(recording, []).append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "task_id": task["task_id"],
                    "record_ids": source.get("record_ids", []),
                    "source_window_seconds": [float(clip["start"]), float(clip["end"])]
                    if clip
                    else None,
                    "rejection_reason": task.get("rejection_reason"),
                }
            )
    for path in sorted((root / "data/sources/audits").glob(SCREENING_DECISIONS_GLOB)):
        decisions = read_json_strict(path)
        dataset = decisions.get("datasets", {}).get("HD-EPIC", {})
        for media_id in dataset.get("rejected_media_ids", []):
            recording = str(media_id).split("__")[0]
            clip = HDEPIC_CLIP_NAME.fullmatch(str(media_id))
            rejected.setdefault(recording, []).append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": file_sha256(path),
                    "candidate_pool_schema_version": decisions.get(
                        "candidate_pool_schema_version"
                    ),
                    "source_window_seconds": [float(clip["start"]), float(clip["end"])]
                    if clip
                    else None,
                    "rejection_reason": dataset.get("rejection_reason"),
                }
            )
    return rejected


def _previously_rejected_event(
    hit: dict[str, Any], rejections: list[dict[str, Any]]
) -> bool:
    for rejection in rejections:
        if hit["record_id"] in rejection.get("record_ids", []):
            return True
        window = rejection.get("source_window_seconds")
        if (
            window
            and hit["start_seconds"] < window[1]
            and hit["end_seconds"] > window[0]
        ):
            return True
    return False


def _video_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(json.loads(result.stdout)["streams"][0]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid video duration for {path}: {duration}")
    return duration


def _hdepic_narrated_candidates(root: Path, selected: set[str]) -> list[dict[str, Any]]:
    hits = find_hdepic_refresh_hits(root / HDEPIC_NARRATIONS)
    prior_rejections = _hdepic_prior_rejections(root)
    clips: dict[str, list[tuple[Path, float, float]]] = {}
    for relative_root in HDEPIC_CLIP_ROOTS:
        for path in sorted((root / relative_root).glob("*.mp4")):
            if not _media_available(path):
                continue
            match = HDEPIC_CLIP_NAME.fullmatch(path.stem)
            if not match:
                raise ValueError(f"HD-EPIC clip must name its source interval: {path}")
            start, end = float(match["start"]), float(match["end"])
            if start >= end:
                raise ValueError(f"HD-EPIC clip has an invalid source interval: {path}")
            clips.setdefault(match["video"], []).append((path, start, end))

    candidates = []
    durations: dict[Path, float] = {}
    annotation_hash = file_sha256(root / HDEPIC_NARRATIONS)
    for hit in hits:
        recording = hit["video_id"]
        rejections = prior_rejections.get(recording, [])
        if _recording_is_listed(recording, selected) or _previously_rejected_event(
            hit, rejections
        ):
            continue
        start, end = hit["start_seconds"], hit["end_seconds"]
        for path, clip_start, clip_end in clips.get(recording, []):
            if not clip_start <= start < end <= clip_end:
                continue
            if path not in durations:
                durations[path] = _video_duration(path)
            if end > clip_start + durations[path]:
                continue
            candidates.append(
                {
                    "dataset": "HD-EPIC",
                    "media_id": recording,
                    "media_path": path.relative_to(root).as_posix(),
                    "media_available": True,
                    "media_sha256": file_sha256(path),
                    "media_duration_seconds": durations[path],
                    "source_start_seconds": clip_start,
                    "source_clip_window_seconds": [clip_start, clip_end],
                    "window_seconds": [start, end],
                    "media_window_seconds": [
                        round(start - clip_start, 6),
                        round(end - clip_start, 6),
                    ],
                    "signals": [NARRATED_CHOICE_SIGNAL],
                    "signal_schema_version": 1,
                    "annotation_id": hit["record_id"],
                    "narration": hit["narration"],
                    "annotation_source": {
                        "path": HDEPIC_NARRATIONS,
                        "sha256": annotation_hash,
                    },
                    "prior_rejections": rejections,
                    "stage": "narration_mined_local_clip",
                }
            )
            break
    return candidates


def _captain_candidates(root: Path, screened: set[str]) -> list[dict[str, Any]]:
    annotations = read_json_strict(root / CAPTAIN_ANNOTATIONS)
    candidates = []
    for recording in annotations:
        recording_id = str(recording["recording_id"])
        media_path = f"{CAPTAIN_VIDEOS}/{recording_id}_360p.mp4"
        if (
            recording_id in screened
            or not recording.get("is_error")
            or not _media_available(root / media_path)
        ):
            continue
        events = []
        signals = set()
        for step in recording["step_annotations"]:
            for error in step.get("errors", []):
                tag = str(error["tag"])
                if tag == "Preparation Error":
                    signals.add("refresh.object_or_tool_substitution")
                elif tag == "Measurement Error":
                    signals.add("refresh.constraint_change")
                elif tag == "Order Error":
                    signals.add("clarify.sequence")
                events.append(
                    {
                        "step_id": step["step_id"],
                        "step": step["description"],
                        "interval_seconds": [step["start_time"], step["end_time"]],
                        "error_type": tag,
                        "error": error["description"],
                    }
                )
        if not signals:
            continue
        candidates.append(
            {
                "dataset": "CaptainCook4D",
                "media_id": recording_id,
                "media_path": media_path,
                "activity_id": recording["activity_id"],
                "signals": sorted(signals),
                "events": events,
                "stage": "annotation_mined_local_video",
            }
        )
    return sorted(candidates, key=lambda row: row["media_id"])


def build(root: Path) -> dict[str, Any]:
    selected = _selected_media(root)
    screened = selected | _screened_expansion_media(root)
    datasets = {
        "Ego2Web": _ego2web_candidates(root, screened),
        "HD-EPIC": _hdepic_candidates(root, screened)
        + _hdepic_narrated_candidates(root, selected),
        "CaptainCook4D": _captain_candidates(root, screened),
    }
    return {
        "schema_version": 2,
        "candidate_pool_schema_version": 2,
        "status": "high_recall_hypotheses_not_benchmark_tasks",
        "admission_rule": (
            "A row becomes a constructed candidate only after continuous-video review, "
            "released-frame review, source-cluster separation, environment grounding, and "
            "the task-family construction gates all pass."
        ),
        "summary": {
            name: {
                "candidates": len(rows),
                "media": len({row["media_id"] for row in rows}),
                "media_available": len(
                    {
                        row["media_id"]
                        for row in rows
                        if row.get("media_available", True)
                    }
                ),
            }
            for name, rows in datasets.items()
        },
        "datasets": datasets,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the current summary without writing the pool",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    report = build(root)
    if not args.check:
        write_json_atomic(root / args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
