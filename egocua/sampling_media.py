"""Prepare reusable fixed-interval frames without changing canonical task files."""

from __future__ import annotations

import math
import subprocess
import tempfile
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image

from .frame_inputs import packed_image_size, validate_max_image_edge
from .lineage import (
    SourceMediaSegment,
    media_segment_covering,
    resolve_continuous_media_video,
    source_media_segments,
)
from .sources import find_record, load_dossier
from .timeline import COMPOSITE_TRACK
from .visual_sampling import is_skipped, sampling_times
from .workspace import resolve_within

NANOSECONDS_PER_SECOND = 1_000_000_000
REFRESH_LEAD_IN_SECONDS = 16
"""Source seconds a Refresh stream resumes before V1, about two steps."""
SamplingWindow = tuple[str, float, tuple[int, int], tuple[float, float]]


def prepare_sampling_tasks(
    tasks: list[dict[str, Any]],
    root: Path,
    intervals: tuple[float, ...],
) -> list[dict[str, Any]]:
    """Plan source extraction; no images or task files are written here."""
    prepared = deepcopy(tasks)
    for task in prepared:
        source = SourceMedia(task, root)
        for packet in task["visual_packets"]:
            timestamps = source.packet_timestamps(packet)
            packet["timestamps_seconds"] = list(timestamps)
            if intervals:
                packet["frequency_samples"] = source.pool(
                    [timestamps[0], timestamps[-1]],
                    intervals,
                    packet["frame_paths"],
                )
    return prepared


def prepare_session_tasks(
    tasks: list[dict[str, Any]], root: Path, *intervals: float
) -> list[dict[str, Any]]:
    """Plan one continuous stream per task from its reviewed source window.

    The manifest owns the window and, for Graft, when the follow-up request is
    made. Refresh's change is V1's first source frame and its default window
    runs from V0's first to V1's last frame; the wait between V0 and the
    change is cut so the change reaches an agent that is still working. Cut
    footage is never read, so each delivered span needs only its own file.
    The reviewed packet times mark the batches the agent's context keeps whole.
    """
    prepared = deepcopy(tasks)
    for task in prepared:
        source = SourceMedia(task, root)
        if task["track"] == COMPOSITE_TRACK:
            _prepare_composite_session(task, source, intervals)
            continue
        window, onset = source.stream_window(task)
        cut = source.refresh_wait(task, onset) if task["track"] == "refresh" else None
        session: dict[str, Any] = {
            "frequency_samples": source.pool(
                window,
                intervals,
                [path for packet in task["visual_packets"] for path in packet["frame_paths"]],
                cut,
            ),
            "reviewed_seconds": sorted(
                {
                    time
                    for packet in task["visual_packets"]
                    for time in source.packet_timestamps(packet)
                }
            ),
        }
        if onset is not None:
            session["update_onset_seconds"] = onset
        if cut is not None:
            session["skipped_seconds"] = cut
        max_edge = validate_max_image_edge(
            (source.stream or {}).get("wearer_max_image_edge")
        )
        if max_edge is not None:
            session["wearer_max_image_edge"] = max_edge
        packing = (source.stream or {}).get("wearer_frames_per_image")
        if packing is not None:
            packed_image_size(1, 1, packing)
            session["wearer_frames_per_image"] = packing
        task["visual_session"] = session
    return prepared


def _prepare_composite_session(
    task: dict[str, Any], source: SourceMedia, intervals: tuple[float, ...]
) -> None:
    if intervals != (1.0,):
        raise ValueError("composite material uses the fixed 1 fps replay")
    window = task["source"]["window_seconds"]
    if (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in window)
        or not 0 <= window[0] < window[1] <= window[0] + 160
    ):
        raise ValueError(f"{task['id']}: invalid composite source window")
    pool = source.pool(window, intervals, source.packets["V0"]["frame_paths"])
    if len(pool["frame_paths"]) < 8:
        raise ValueError(f"{task['id']}: composite opening requires eight frames")
    task["visual_session"] = {**task.get("visual_session", {}), "frequency_samples": pool}
    task["visual_packets"] = [{"id": "V0", "frame_paths": pool["frame_paths"][:8]}]


class SourceMedia:
    """One task's continuous source video and its registered packet timeline."""

    def __init__(self, task: dict[str, Any], root: Path) -> None:
        _, dossier = load_dossier(task, root=root)
        record = find_record(dossier, task["source"]["task_id"], context=task["id"])
        self.task_id = task["id"]
        self.root = root
        self.segments = source_media_segments(record["source"], context=task["id"])
        segment_spec = task["source"].get("recording_segment")
        if task.get("track") == COMPOSITE_TRACK and segment_spec is not None:
            self._use_reviewed_segment(segment_spec)
        else:
            self.videos = {
                segment.media_id: resolve_continuous_media_video(
                    root,
                    dossier["dataset"]["source_root"],
                    segment.media_id,
                    context=task["id"],
                )
                for segment in self.segments
            }
        for segment in self.segments:
            if (
                not math.isfinite(segment.source_start_seconds)
                or segment.source_start_seconds < 0
            ):
                raise ValueError(f"{task['id']}: invalid source media offset")
        self.packets = {packet["id"]: packet for packet in record["packets"]}
        self.stream = record.get("stream")

    def _use_reviewed_segment(self, spec: dict[str, Any]) -> None:
        """Select an explicitly reviewed file from the registered original recording."""
        video = resolve_within(self.root, spec["path"], context=f"{self.task_id} video")
        start, end = spec["source_seconds"]
        original = self.segments[0].original_video_id
        if (
            video.stem != Path(original).name
            and not video.stem.startswith(f"{Path(original).name}__")
        ):
            raise ValueError(f"{self.task_id}: reviewed segment belongs to a different recording")
        if (
            any(type(value) not in (float, int) or not math.isfinite(value) for value in (start, end))
            or not 0 <= start < end
        ):
            raise ValueError(f"{self.task_id}: invalid reviewed segment interval")
        self.segments = (SourceMediaSegment(video.stem, original, start, end, None),)
        self.videos = {video.stem: video}

    def stream_window(self, task: dict[str, Any]) -> tuple[list[float], float | None]:
        """Reviewed session window and change onset, in source seconds."""
        packets = {packet["id"]: packet for packet in task["visual_packets"]}
        opening = self.packet_timestamps(packets["V0"])
        stream = self.stream or {}
        if task["track"] == "refresh":
            later = self.packet_timestamps(packets["V1"])
            if later[0] <= opening[-1]:
                raise ValueError(f"{self.task_id}: V1 must follow V0 in the source video")
            if "update_onset_seconds" in stream:
                raise ValueError(f"{self.task_id}: Refresh changes at V1's first frame")
            window = stream.get("session_seconds", [opening[0], later[-1]])
            onset: float | None = later[0]
            last_reviewed = later[-1]
        else:
            if "session_seconds" not in stream:
                raise ValueError(
                    f"{self.task_id}: frames after the opening need a reviewed "
                    "stream window in the source manifest"
                )
            window = stream["session_seconds"]
            onset = stream.get("update_onset_seconds")
            if (onset is None) != (task["track"] == "clarify"):
                raise ValueError(
                    f"{self.task_id}: only Graft records when the follow-up is made"
                )
            last_reviewed = opening[-1]
        start, end = window
        if not start <= opening[0] or end < last_reviewed:
            raise ValueError(f"{self.task_id}: stream window must cover reviewed frames")
        if onset is not None and not start < onset <= end:
            raise ValueError(f"{self.task_id}: update onset falls outside the stream")
        return [float(start), float(end)], onset

    def refresh_wait(self, task: dict[str, Any], onset: float) -> list[float] | None:
        """Open source interval between V0's last frame and the lead-in to V1."""
        v0 = next(packet for packet in task["visual_packets"] if packet["id"] == "V0")
        opening_end = self.packet_timestamps(v0)[-1]
        resume = onset - REFRESH_LEAD_IN_SECONDS
        return [opening_end, resume] if resume > opening_end else None

    def covering_media(self, window: list[float]) -> tuple[Path, SourceMediaSegment]:
        """The one local video file that holds the whole source window."""
        segment = media_segment_covering(
            self.segments, (window[0], window[-1]), context=self.task_id
        )
        return self.videos[segment.media_id], segment

    def packet_timestamps(self, packet: dict[str, Any]) -> list[float]:
        source_packet = self.packets[packet["id"]]
        if source_packet["frame_paths"] != packet["frame_paths"]:
            raise ValueError(
                f"{self.task_id}: source packet does not match task frames"
            )
        timestamps = source_packet["timestamps_seconds"]
        if len(timestamps) != len(packet["frame_paths"]) or any(
            right <= left for left, right in pairwise(timestamps)
        ):
            raise ValueError(f"{self.task_id}: invalid source timestamps")
        return list(timestamps)

    def pool(
        self,
        window: list[float],
        intervals: tuple[float, ...],
        registered_frames: list[str],
        skipped: list[float] | None = None,
    ) -> dict[str, Any]:
        """Frames of each delivered span; footage inside a cut is never read."""
        times = sorted(
            {
                time
                for interval in intervals
                for time in sampling_times(window, interval)
                if not is_skipped(time, skipped)
            }
        )
        if not times:
            raise ValueError(f"{self.task_id}: sampling outside source media")
        spans = [window] if skipped is None else [[window[0], skipped[0]], [skipped[1], window[1]]]
        sizes = set()
        for path in registered_frames:
            with Image.open(
                resolve_within(self.root, path, context="registered frame")
            ) as image:
                sizes.add(image.size)
        if len(sizes) != 1:
            raise ValueError(
                f"{self.task_id}: inconsistent registered image dimensions"
            )
        width, height = sizes.pop()
        sources, frame_paths = [], []
        for span in spans:
            video, segment = self.covering_media(span)
            relative_video = video.relative_to(self.root.resolve())
            cache = (
                Path("data/experiments/assets/sampling")
                / relative_video.with_suffix("")
                / f"{width}x{height}"
            )
            sources.append(
                {
                    "source_video": relative_video.as_posix(),
                    "media_start_seconds": segment.source_start_seconds,
                    "window_seconds": span,
                }
            )
            frame_paths += [
                (cache / f"{time:.9f}.jpg").as_posix()
                for time in times
                if span[0] <= time <= span[1]
            ]
        return {
            "sources": sources,
            "image_size": [width, height],
            "encoder": "ffmpeg-fps-jpeg-q2-v2",
            "window_seconds": window,
            "timestamps_seconds": times,
            "frame_paths": frame_paths,
        }


def materialize_sampling_frames(tasks: list[dict[str, Any]], root: Path) -> None:
    """Extract each unique frame once. Missing sources or failed decoding stop export."""
    for window, frames in _sampling_batches(tasks).items():
        _extract_window(window, frames, root)


def _sampling_batches(
    tasks: list[dict[str, Any]],
) -> dict[SamplingWindow, dict[float, str]]:
    batches: dict[SamplingWindow, dict[float, str]] = {}
    for task in tasks:
        pools = [packet.get("frequency_samples") for packet in task["visual_packets"]]
        pools.append(task.get("visual_session", {}).get("frequency_samples"))
        for pool in pools:
            if pool is None:
                continue
            frames = dict(zip(pool["timestamps_seconds"], pool["frame_paths"], strict=True))
            for source in pool["sources"]:
                start, end = source["window_seconds"]
                key = (
                    source["source_video"],
                    source["media_start_seconds"],
                    tuple(pool["image_size"]),
                    (start, end),
                )
                batches.setdefault(key, {}).update(
                    (time, path) for time, path in frames.items() if start <= time <= end
                )
    return batches


def _extract_window(
    window: SamplingWindow, frames: dict[float, str], root: Path
) -> None:
    source, offset, (width, height), _ = window
    video = resolve_within(root, source, context="sampling video")
    destinations = {
        time: resolve_within(root, path, context="sampling frame", must_exist=False)
        for time, path in frames.items()
    }
    if all(path.is_file() for path in destinations.values()):
        return
    times = sorted(frames)
    ticks = [round(time * NANOSECONDS_PER_SECOND) for time in times]
    spacing = (
        math.gcd(*(right - left for left, right in pairwise(ticks)))
        if len(ticks) > 1
        else NANOSECONDS_PER_SECOND
    )
    step = spacing / NANOSECONDS_PER_SECOND
    count = (ticks[-1] - ticks[0]) // spacing + 1
    with tempfile.TemporaryDirectory(prefix="egocua-sampling-") as temporary:
        output = Path(temporary) / "%09d.jpg"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{times[0] - offset:.9f}",
                "-i",
                str(video),
                "-frames:v",
                str(count),
                "-vf",
                f"fps=1/{step:.9f}:start_time=0:round=up,scale={width}:{height}",
                "-q:v",
                "2",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        for tick, time in zip(ticks, times, strict=True):
            destination = destinations[time]
            if not destination.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                index = (tick - ticks[0]) // spacing + 1
                (Path(temporary) / f"{index:09d}.jpg").replace(destination)
