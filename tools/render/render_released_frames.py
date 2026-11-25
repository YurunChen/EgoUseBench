"""Montage a task's released frames into one labelled sheet.

Stage B asks what a candidate set looks like at each released frame, so what has to
be looked at is exactly the frames the agent receives — not the source video, which
holds seconds the task never delivers. Each tile carries its frame id, so a curve
row can name the frame it is about.

Usage:
    PYTHONPATH=. uv run python tools/render/render_released_frames.py --root . --task <id>
    PYTHONPATH=. uv run python tools/render/render_released_frames.py --root . --all
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from egocua.decision import frame_ids
from egocua.schema import load_tasks

OUT_DIR = "artifacts/screening/released-frames"
COLUMNS = 4
TILE_WIDTH = 320


class SheetError(RuntimeError):
    """Raised when ffmpeg cannot build the sheet."""


@dataclass(frozen=True)
class Sheet:
    task_id: str
    frames: list[tuple[str, Path]]

    @property
    def rows(self) -> int:
        return (len(self.frames) + COLUMNS - 1) // COLUMNS


def _run(command: list[str], context: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SheetError(f"{context}: {result.stderr.strip()[:200]}")


def render(sheet: Sheet, out_dir: Path) -> Path:
    staging = out_dir / f".{sheet.task_id}"
    staging.mkdir(parents=True, exist_ok=True)
    for index, (frame_id, source) in enumerate(sheet.frames):
        label = frame_id.replace(":", "\\:")
        _run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-vf",
                (
                    f"scale={TILE_WIDTH}:-1,"
                    f"drawtext=text='{label}':x=6:y=6:fontsize=20:fontcolor=yellow:"
                    f"box=1:boxcolor=black@0.65:boxborderw=4"
                ),
                str(staging / f"{index:02d}.jpg"),
            ],
            f"{sheet.task_id} {frame_id}",
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{sheet.task_id}.jpg"
    _run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(staging / "%02d.jpg"),
            "-filter_complex",
            f"tile={COLUMNS}x{sheet.rows}",
            "-frames:v",
            "1",
            str(target),
        ],
        f"{sheet.task_id}: tiling",
    )
    for tile in staging.glob("*.jpg"):
        tile.unlink()
    staging.rmdir()
    return target


def sheets_for(root: Path, tasks_root: Path, task_ids: set[str] | None) -> list[Sheet]:
    sheets = []
    if any((tasks_root / track).is_dir() for track in ("graft", "refresh", "clarify")):
        tasks = load_tasks(tasks_root)
    else:
        tasks = [
            json.loads(task_path.read_text())
            for task_path in sorted(tasks_root.glob("*.json"))
        ]
    for task in tasks:
        if task_ids is not None and task["id"] not in task_ids:
            continue
        paths = [
            root / path
            for packet in task["visual_packets"]
            for path in packet["frame_paths"]
        ]
        sheets.append(
            Sheet(
                task_id=str(task["id"]),
                frames=list(zip(frame_ids(task["visual_packets"]), paths, strict=True)),
            )
        )
    return sheets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--tasks-root",
        default="data/tasks",
        help="task directory to render, relative to --root unless absolute",
    )
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--task", action="append", help="task id, repeatable")
    parser.add_argument("--all", action="store_true", help="every live task")
    args = parser.parse_args(argv)
    if not args.all and not args.task:
        parser.error("pass --task <id> or --all")

    root = Path(args.root).resolve()
    tasks_root = Path(args.tasks_root)
    if not tasks_root.is_absolute():
        tasks_root = root / tasks_root
    out_dir = root / args.out
    sheets = sheets_for(root, tasks_root, None if args.all else set(args.task))
    for sheet in sheets:
        print(render(sheet, out_dir).relative_to(root).as_posix())
    if args.all:
        expected = {f"{sheet.task_id}.jpg" for sheet in sheets}
        for stale in out_dir.glob("*.jpg"):
            if stale.name not in expected:
                stale.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
