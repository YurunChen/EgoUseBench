"""Source-time sampling; evidence batches keep their existing release schedule."""

from __future__ import annotations

import math
from collections.abc import Mapping
from itertools import pairwise
from typing import Any


def validate_interval(value: float) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("frame interval must be a finite positive number of seconds")


def sampling_times(window: list[float], interval: float) -> list[float]:
    """Anchor at the packet start; never append an off-grid endpoint."""
    validate_interval(interval)
    if (
        len(window) != 2
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for value in window
        )
        or window[1] < window[0]
    ):
        raise ValueError(
            "sampling window must contain ordered nonnegative source seconds"
        )
    start, end = window
    count = math.floor((end - start) / interval + 1e-9) + 1
    return [round(start + index * interval, 9) for index in range(count)]


def is_skipped(time: float, skipped: list[float] | None) -> bool:
    """A cut drops its interior; both delivered edge frames stay."""
    return skipped is not None and skipped[0] < time < skipped[1]


def sample_packet(
    packet: Mapping[str, Any],
    interval: float,
    repeat_to_interval: float | None = None,
    skipped: list[float] | None = None,
) -> dict[str, Any]:
    """Select exact grid frames from a prepared source-video pool, not keyframes."""
    pool = packet.get("frequency_samples")
    if not isinstance(pool, Mapping):
        raise ValueError(
            "fixed-interval sampling requires prepared source-video frames"
        )
    expected = sampling_times(pool["window_seconds"], interval)
    timestamps, paths = pool["timestamps_seconds"], pool["frame_paths"]
    if (
        len(timestamps) != len(paths)
        or not timestamps
        or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in timestamps
        )
        or any(right <= left for left, right in pairwise(timestamps))
    ):
        raise ValueError("sampling pool needs aligned, strictly increasing timestamps")
    by_time = dict(zip(timestamps, paths, strict=True))
    if any(
        timestamp not in by_time
        for timestamp in expected
        if not is_skipped(timestamp, skipped)
    ):
        raise ValueError(
            "sampling pool is missing requested source timestamps; re-prepare frames"
        )
    selected = expected
    if repeat_to_interval is not None:
        count = len(sampling_times(pool["window_seconds"], repeat_to_interval))
        selected = [expected[index * len(expected) // count] for index in range(count)]
    delivered = [time for time in selected if not is_skipped(time, skipped)]
    sampled: dict[str, Any] = {
        "frame_paths": [by_time[timestamp] for timestamp in delivered],
        "timestamps_seconds": delivered,
        "frame_interval_seconds": interval,
    }
    if repeat_to_interval is not None:
        sampled["frame_repeat_to_interval_seconds"] = repeat_to_interval
    return sampled


def packet_asset_paths(packet: Mapping[str, Any]) -> list[str]:
    """All frozen visual assets, including the source-time sampling pool."""
    return [
        *packet.get("frame_paths", []),
        *packet.get("frequency_samples", {}).get("frame_paths", []),
    ]
