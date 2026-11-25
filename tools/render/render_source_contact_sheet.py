"""Render a contact sheet for one source video so a clip can be screened by eye.

Screening asks whether a clip carries a decision point at all — two same-type
candidates, an order or a vantage that separates them, and a cost for choosing
wrong. That question is answered by looking, and looking at twelve stills laid
out in a grid is far cheaper than stepping through a video.

Sheets are screening scratch, not released evidence: they are written under
artifacts/, never under data/observations/.

Usage:
    uv run python tools/render/render_source_contact_sheet.py --vid 7789eac2-... --window 130 150
    uv run python tools/render/render_source_contact_sheet.py --batch batch.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SOURCE_ROOT = "reference/sources/Ego2Web/dataset/ego2web_video"
DEFAULT_OUT_DIR = "artifacts/screening/contact-sheets"
DEFAULT_TILE_COUNT = 12
COLUMNS = 4
TILE_WIDTH = 320
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".mkv"})


class SheetError(RuntimeError):
    """Raised when a source video is missing or ffmpeg fails."""


@dataclass(frozen=True)
class SheetRequest:
    """One clip to look at: which video, and which seconds of it."""

    vid: str
    start: float
    end: float
    count: int = DEFAULT_TILE_COUNT
    times: tuple[float, ...] = ()
    label: str = ""

    @classmethod
    def from_dict(cls, entry: dict[str, Any]) -> "SheetRequest":
        times = tuple(float(value) for value in entry.get("times", ()))
        start = (
            float(entry["start"]) if "start" in entry else (times[0] if times else 0.0)
        )
        end = float(entry["end"]) if "end" in entry else (times[-1] if times else 0.0)
        return cls(
            vid=str(entry["vid"]),
            start=start,
            end=end,
            count=int(entry.get("count", DEFAULT_TILE_COUNT)),
            times=times,
            label=str(entry.get("label", "")),
        )

    @property
    def name(self) -> str:
        return self.label or self.vid

    def sample_times(self) -> list[float]:
        """Explicit times when given — a window that spans a gap must not be sampled
        uniformly across it, or every frame lands between the two things to compare."""

        if self.times:
            return list(self.times)
        if self.end <= self.start or self.count < 2:
            return [self.start]
        step = (self.end - self.start) / (self.count - 1)
        return [round(self.start + step * index, 3) for index in range(self.count)]


@dataclass(frozen=True)
class SheetPaths:
    """Where the source videos are read from and the sheets are written to."""

    root: Path
    source_root: str = DEFAULT_SOURCE_ROOT
    out_dir_name: str = DEFAULT_OUT_DIR

    @property
    def out_dir(self) -> Path:
        return self.root / self.out_dir_name

    def video(self, vid: str) -> Path:
        matches = [
            # Datasets nest by participant or shard, so search the whole tree.
            path
            for path in sorted((self.root / self.source_root).rglob(f"{vid}*"))
            if path.suffix.casefold() in VIDEO_SUFFIXES
        ]
        if len(matches) != 1:
            raise SheetError(f"{vid}: expected one source video, found {len(matches)}")
        return matches[0]


def _run_ffmpeg(command: list[str], context: str) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SheetError(f"{context}: {result.stderr.strip()[:200]}")


def _extract_tiles(request: SheetRequest, paths: SheetPaths) -> Path:
    """Write one stamped still per sample time; return the directory holding them."""

    video = paths.video(request.vid)
    tiles_dir = paths.out_dir / f".{request.name}-frames"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    for index, timestamp in enumerate(request.sample_times()):
        stamp = (
            f"drawtext=text='{timestamp:g}s':x=6:y=6:fontsize=18:fontcolor=yellow:"
            f"box=1:boxcolor=black@0.6:boxborderw=3"
        )
        _run_ffmpeg(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(timestamp),
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"scale={TILE_WIDTH}:-1,{stamp}",
                str(tiles_dir / f"{index:02d}.jpg"),
            ],
            f"{request.vid} @ {timestamp}s",
        )
    return tiles_dir


def _tile_into_sheet(tiles_dir: Path, sheet: Path, tile_count: int) -> None:
    rows = (tile_count + COLUMNS - 1) // COLUMNS
    _run_ffmpeg(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(tiles_dir / "%02d.jpg"),
            "-filter_complex",
            f"tile={COLUMNS}x{rows}",
            "-frames:v",
            "1",
            str(sheet),
        ],
        f"{sheet.stem}: tiling",
    )


def render(request: SheetRequest, paths: SheetPaths) -> Path:
    paths.out_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = _extract_tiles(request, paths)
    sheet = paths.out_dir / f"{request.name}.jpg"
    try:
        _tile_into_sheet(tiles_dir, sheet, len(request.sample_times()))
    finally:
        for tile in tiles_dir.glob("*.jpg"):
            tile.unlink()
        tiles_dir.rmdir()
    return sheet


def _requests_from_args(args: argparse.Namespace) -> list[SheetRequest]:
    if args.batch:
        return [
            SheetRequest.from_dict(entry)
            for entry in json.loads(Path(args.batch).read_text())
        ]
    start, end = args.window
    return [SheetRequest(vid=args.vid, start=start, end=end, count=args.count)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--out", default=DEFAULT_OUT_DIR)
    parser.add_argument("--vid", help="one source video id")
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("START", "END"),
        help="seconds to sample between",
    )
    parser.add_argument(
        "--batch", help="JSON list of {vid, start, end} objects to render together"
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_TILE_COUNT, help="stills per sheet"
    )
    args = parser.parse_args(argv)
    if not args.batch and not (args.vid and args.window):
        parser.error("pass --vid with --window, or --batch")

    paths = SheetPaths(
        root=Path(args.root).resolve(),
        source_root=args.source_root,
        out_dir_name=args.out,
    )
    for request in _requests_from_args(args):
        try:
            sheet = render(request, paths)
        except SheetError as exc:
            print(f"skip {exc}")
            continue
        print(sheet.relative_to(paths.root).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
