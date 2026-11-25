"""Review, record and check the session window each task streams under fixed-step replay.

``packets`` writes review materials for formal tasks without a reviewed window.
``apply`` writes accepted reviews into their source records and checks the schedule.
``schedule`` prints when each task's change reaches the agent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from egocua.experiment import load_experiment_plan
from egocua.lineage import LineageError
from egocua.sampling_media import (
    REFRESH_LEAD_IN_SECONDS,
    SourceMedia,
    prepare_session_tasks,
)
from egocua.schema import load_tasks
from egocua.selection import load_selection, task_ids
from egocua.sources import find_record, load_dossier
from egocua.messages import KEY_EVIDENCE
from egocua.timeline import OBSERVATION, Timeline, is_update_event
from egocua.workspace import read_json_strict, write_json_atomic

REVIEW_ROOT = Path("artifacts/reports/source-review-evidence/stream-windows")
DEFAULT_PLAN = "configs/benchmark.json"
INSTRUCTIONS = ".claude/skills/synthesize-benchmark-tasks/REFERENCE.md#stream-window"
TAIL_SECONDS = 64.0
"""Proposed footage after the last reviewed frame: eight 8-second steps."""
MEDIA_END_MARGIN_SECONDS = 0.5
THUMBNAIL_PIXELS = 256
SHEET_COLUMNS = SHEET_ROWS = 6
APPLIED_VERDICTS = {"accept", "revise"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("packets", "apply", "schedule"):
        command = commands.add_parser(name)
        command.add_argument("--root", default=".")
        command.add_argument("--task-id", action="append", default=[])
    commands.choices["apply"].add_argument("--reviews", default=str(REVIEW_ROOT))
    for name in ("apply", "schedule"):
        commands.choices[name].add_argument("--plan", default=DEFAULT_PLAN)
    args = parser.parse_args()
    root = Path(args.root)
    if args.command == "packets":
        for task in tasks_to_review(root, args.task_id):
            try:
                row = write_packet(root, task)
            except LineageError as error:
                row = {"task_id": task["id"], "source_error": str(error)}
            print(json.dumps(row, ensure_ascii=False))
    elif args.command == "apply":
        report = apply_reviews(root, Path(args.reviews), args.task_id, args.plan)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        for row in schedule(root, args.task_id, args.plan):
            print(json.dumps(row, ensure_ascii=False))


def _definitions(root: Path) -> dict[str, dict[str, Any]]:
    return {task["id"]: task for task in load_tasks(root / "data/tasks")}


def _source_record(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    _, dossier = load_dossier(task, root=root)
    return find_record(dossier, task["source"]["task_id"], context=task["id"])


def tasks_to_review(root: Path, requested: list[str]) -> list[dict[str, Any]]:
    """Explicit IDs, or every formal member whose window has no review record."""
    definitions = _definitions(root)
    if requested:
        return [definitions[task_id] for task_id in requested]
    return [
        definitions[task_id]
        for task_id in task_ids(load_selection(root))
        if "review_evidence" not in (_source_record(root, definitions[task_id]).get("stream") or {})
    ]


def proposal(task: dict[str, Any], source: SourceMedia, media_end: float) -> dict[str, Any]:
    """A starting window for the reviewer; the review decides the recorded values."""
    packets = {packet["id"]: packet for packet in task["visual_packets"]}
    opening = source.packet_timestamps(packets["V0"])
    existing = source.stream or {}
    brief: dict[str, Any] = {
        "task_id": task["id"],
        "track": task["track"],
        "instructions": task["instructions"],
        "v0_seconds": opening,
        "review_instructions": INSTRUCTIONS,
    }
    if task["track"] == "refresh":
        later = source.packet_timestamps(packets["V1"])
        start = min(opening[0], later[0] - REFRESH_LEAD_IN_SECONDS)
        brief.update(
            v1_seconds=later,
            refresh=task["refresh"],
            change_seconds=later[0],
            session_seconds=existing.get("session_seconds", [start, later[-1]]),
            skipped_seconds=source.refresh_wait(task, later[0]),
        )
        return brief
    end = min(media_end - MEDIA_END_MARGIN_SECONDS, opening[-1] + TAIL_SECONDS)
    brief["session_seconds"] = existing.get("session_seconds", [opening[0], round(end, 1)])
    if task["track"] == "graft":
        widest_gap = max(range(1, len(opening)), key=lambda i: opening[i] - opening[i - 1])
        brief["proposed_update_onset_seconds"] = existing.get(
            "update_onset_seconds", opening[widest_gap]
        )
    else:
        brief["clarification"] = task["clarification"]
    return brief


def write_packet(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    source = SourceMedia(task, root)
    opening = source.packet_timestamps(
        next(packet for packet in task["visual_packets"] if packet["id"] == "V0")
    )
    video, segment = source.covering_media([opening[0], opening[-1]])
    media_end = segment.source_start_seconds + _duration(video)
    if segment.source_end_seconds is not None:
        media_end = min(media_end, segment.source_end_seconds)
    brief = proposal(task, source, media_end)
    brief["media_seconds"] = [segment.source_start_seconds, media_end]
    folder = root / REVIEW_ROOT / task["id"]
    folder.mkdir(parents=True, exist_ok=True)
    brief["review_output"] = (REVIEW_ROOT / task["id"] / "review.json").as_posix()
    write_json_atomic(folder / "brief.json", brief)
    for packet in task["visual_packets"]:
        tiles = [
            _labelled(Image.open(root / path), f"{packet['id']}[{index}] t={time:g}s")
            for index, (path, time) in enumerate(
                zip(packet["frame_paths"], source.packet_timestamps(packet), strict=True)
            )
        ]
        _save_sheets(tiles, folder, packet["id"].lower())
    start = max(brief["session_seconds"][0], segment.source_start_seconds)
    end = brief["session_seconds"][1]
    if task["track"] != "refresh":
        end = min(end, media_end)
    cut = brief.get("skipped_seconds")
    spans = [(start, end)] if cut is None else [(start, cut[0]), (cut[1], end)]
    tiles = []
    for first, last in spans:
        span_video, span_segment = source.covering_media([first, last])
        tiles += _window_tiles(span_video, span_segment.source_start_seconds, first, last)
    _save_sheets(tiles, folder, "window")
    return {"task_id": task["id"], "folder": folder.as_posix(), "frames": len(tiles)}


def _duration(video: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _window_tiles(video: Path, media_start: float, start: float, end: float) -> list[Image.Image]:
    with tempfile.TemporaryDirectory(prefix="egocua-window-") as temporary:
        subprocess.run(
            [
                "ffmpeg", "-nostdin", "-v", "error",
                "-ss", f"{start - media_start:.3f}", "-t", f"{end - start:.3f}",
                "-i", str(video),
                "-vf", f"fps=1,scale={THUMBNAIL_PIXELS}:-1",
                f"{temporary}/%05d.jpg",
            ],
            check=True,
        )
        return [
            _labelled(Image.open(path), f"t={start + index:g}s")
            for index, path in enumerate(sorted(Path(temporary).glob("*.jpg")))
        ]


def _labelled(image: Image.Image, text: str) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((THUMBNAIL_PIXELS, THUMBNAIL_PIXELS))
    canvas = Image.new("RGB", (THUMBNAIL_PIXELS, image.height + 18), "black")
    canvas.paste(image, (0, 18))
    ImageDraw.Draw(canvas).text((4, 3), text, fill="white")
    return canvas


def _save_sheets(tiles: list[Image.Image], folder: Path, stem: str) -> None:
    per_sheet = SHEET_COLUMNS * SHEET_ROWS
    for first in range(0, len(tiles), per_sheet):
        page = tiles[first : first + per_sheet]
        height = max(tile.height for tile in page)
        rows = (len(page) - 1) // SHEET_COLUMNS + 1
        sheet = Image.new("RGB", (SHEET_COLUMNS * THUMBNAIL_PIXELS, rows * height))
        for index, tile in enumerate(page):
            sheet.paste(
                tile, ((index % SHEET_COLUMNS) * THUMBNAIL_PIXELS, (index // SHEET_COLUMNS) * height)
            )
        sheet.save(folder / f"{stem}_{first // per_sheet:02d}.jpg", quality=80)


def stream_block(track: str, review: dict[str, Any], evidence: str) -> dict[str, Any]:
    """The source-record fields a reviewed window owns."""
    block: dict[str, Any] = {"session_seconds": [float(value) for value in review["session_seconds"]]}
    if track == "graft":
        block["update_onset_seconds"] = float(review["update_onset_seconds"])
    block["review_evidence"] = evidence
    return block


def apply_reviews(
    root: Path, reviews: Path, requested: list[str], plan: str
) -> dict[str, Any]:
    """Write accepted windows, then keep only those whose schedule compiles."""
    definitions = _definitions(root)
    base = (root / reviews).resolve()
    folders = (
        [base / task_id for task_id in requested]
        if requested
        else sorted(path for path in base.iterdir() if path.is_dir())
    )
    dossiers: dict[Path, dict[str, Any]] = {}
    previous: dict[str, Any] = {}
    applied, skipped = [], {}
    for folder in folders:
        review_path = folder / "review.json"
        if not review_path.is_file():
            skipped[folder.name] = "no review"
            continue
        review = read_json_strict(review_path)
        if review["verdict"] not in APPLIED_VERDICTS:
            skipped[folder.name] = f"{review['verdict']}: {review.get('notes', '')}"
            continue
        task = definitions[folder.name]
        dossier_path, loaded = load_dossier(task, root=root)
        dossier = dossiers.setdefault(dossier_path, loaded)
        record = find_record(dossier, task["source"]["task_id"], context=task["id"])
        previous[task["id"]] = record.get("stream")
        evidence = review_path.relative_to(root.resolve()).as_posix()
        record["stream"] = stream_block(task["track"], review, evidence)
        applied.append(task["id"])
    for dossier_path, dossier in dossiers.items():
        write_json_atomic(dossier_path, dossier)
    failed = {}
    for task_id in applied:
        try:
            schedule(root, [task_id], plan)
        except ValueError as error:
            failed[task_id] = str(error)
    if failed:
        _restore(root, {task_id: previous[task_id] for task_id in failed}, definitions)
    return {
        "applied": [task_id for task_id in applied if task_id not in failed],
        "schedule_failed": failed,
        "skipped": skipped,
    }


def _restore(root: Path, streams: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> None:
    for task_id, stream in streams.items():
        task = definitions[task_id]
        dossier_path, dossier = load_dossier(task, root=root)
        record = find_record(dossier, task["source"]["task_id"], context=task_id)
        if stream is None:
            record.pop("stream", None)
        else:
            record["stream"] = stream
        write_json_atomic(dossier_path, dossier)


def schedule(root: Path, requested: list[str], plan: str) -> list[dict[str, Any]]:
    """Compile each task under the plan's fixed-step condition; invalid windows raise."""
    loaded = load_experiment_plan(root / plan)
    condition = next(
        loaded.condition(name)
        for name, value in loaded.conditions.items()
        if value.frame_delivery_steps is not None
    )
    definitions = _definitions(root)
    ids = requested or task_ids(load_selection(root))
    rows = []
    for task in prepare_session_tasks(
        [definitions[task_id] for task_id in ids],
        root,
        float(condition.frame_interval_seconds),
    ):
        timeline = Timeline.compile(task, condition, default_release_step=4, reference_steps=None)
        change = [release.step for release in timeline.releases if is_update_event(release.as_event())]
        rows.append(
            {
                "task_id": task["id"],
                "track": task["track"],
                "first_frame_seconds": timeline.initial_observations[0]["timestamps_seconds"][0],
                "skipped_seconds": task["visual_session"].get("skipped_seconds"),
                "change_source_step": change[0] if change else None,
                "key_source_steps": [
                    step
                    for step, batch in [
                        (condition.replay_start_step, timeline.initial_observations[0]),
                        *(
                            (release.step, release.payload)
                            for release in timeline.releases
                            if release.kind == OBSERVATION
                        ),
                    ]
                    if batch.get(KEY_EVIDENCE)
                ],
            }
        )
    return rows


if __name__ == "__main__":
    main()
