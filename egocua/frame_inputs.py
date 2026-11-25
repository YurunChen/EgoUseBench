"""Render wearer frames with fixed packing and optional task-specific size limits."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw

from .workspace import resolve_within

LABEL_HEIGHT = 24
JPEG_QUALITY = 95
POSITIONS = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass(frozen=True)
class ObservationImage:
    media_type: str
    data: str
    label: str
    layout: list[dict[str, Any]]

    @property
    def encoded_bytes(self) -> bytes:
        return base64.b64decode(self.data)


def packed_image_size(
    width: int, height: int, frames_per_image: int
) -> tuple[int, int]:
    if type(frames_per_image) is not int or frames_per_image not in (1, 4):
        raise ValueError("wearer_frames_per_image must be 1 or 4")
    return (
        (width, height)
        if frames_per_image == 1
        else (2 * width, 2 * (height + LABEL_HEIGHT))
    )


def validate_jpeg_quality(value: Any) -> int:
    if type(value) is not int or not 1 <= value <= JPEG_QUALITY:
        raise ValueError("wearer_jpeg_quality must be an integer from 1 to 95")
    return value


def validate_max_image_edge(value: Any) -> int | None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("wearer_max_image_edge must be a positive integer or None")
    return value


def _fit_image(
    image: ObservationImage, max_edge: int | None, quality: int
) -> ObservationImage:
    if max_edge is None:
        return image
    with Image.open(BytesIO(image.encoded_bytes)) as source:
        old_width, old_height = source.size
        if max(source.size) <= max_edge:
            return image
        resized = source.convert("RGB")
        resized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    width, height = resized.size
    layout = []
    for frame in image.layout:
        bounds = frame.get("bounds", [0, 0, old_width, old_height])
        layout.append({
            **frame,
            "original_size": frame.get("original_size", [old_width, old_height]),
            "bounds": [
                round(bounds[0] * width / old_width),
                round(bounds[1] * height / old_height),
                round(bounds[2] * width / old_width),
                round(bounds[3] * height / old_height),
            ],
        })
    encoded = BytesIO()
    resized.save(encoded, format="JPEG", quality=quality)
    return ObservationImage(
        "image/jpeg", base64.b64encode(encoded.getvalue()).decode(), image.label, layout
    )


def _frame_label(index: int, timestamp: float | None) -> str:
    label = f"Frame {index + 1}"
    return label if timestamp is None else f"{label} | source t={timestamp:g} s"


def _grid(
    frames: list[dict[str, Any]], root: str | Path, quality: int
) -> ObservationImage:
    images: list[Image.Image] = []
    for frame in frames:
        path = resolve_within(root, frame["frame_path"], context="wearer frame image")
        with Image.open(path) as source_image:
            images.append(source_image.convert("RGB"))
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", packed_image_size(width, height, 4), (0, 0, 0))
    layout = []
    labels = []
    for slot, (frame, image) in enumerate(zip(frames, images, strict=True)):
        x = (slot % 2) * width
        y = (slot // 2) * (height + LABEL_HEIGHT)
        label = _frame_label(frame["frame_index"], frame["timestamp_seconds"])
        label_strip = Image.new("RGB", (width, LABEL_HEIGHT), (0, 0, 0))
        ImageDraw.Draw(label_strip).text((4, 4), label, fill="white")
        canvas.paste(label_strip, (x, y))
        canvas.paste(image, (x, y + LABEL_HEIGHT))
        layout.append(
            {
                **frame,
                "position": POSITIONS[slot],
                "bounds": [
                    x,
                    y + LABEL_HEIGHT,
                    x + image.width,
                    y + LABEL_HEIGHT + image.height,
                ],
                "original_size": [image.width, image.height],
            }
        )
        labels.append(f"{POSITIONS[slot]}: {label}")
    encoded = BytesIO()
    canvas.save(encoded, format="JPEG", quality=quality)
    return ObservationImage(
        "image/jpeg",
        base64.b64encode(encoded.getvalue()).decode(),
        "; ".join(labels),
        layout,
    )


def observation_images(
    observation: Mapping[str, Any],
    asset_root: str | Path,
    frame_indices: range | None = None,
) -> list[ObservationImage]:
    """Render every selected frame using the registered packing and size limit."""
    paths = observation["frame_paths"]
    packing = observation.get("wearer_frames_per_image", 1)
    packed_image_size(1, 1, packing)
    quality = validate_jpeg_quality(
        observation.get("wearer_jpeg_quality", JPEG_QUALITY)
    )
    max_edge = validate_max_image_edge(observation.get("wearer_max_image_edge"))
    timestamps = observation.get("timestamps_seconds", [None] * len(paths))
    if len(timestamps) != len(paths):
        raise ValueError("released frame timestamps must align with frame_paths")
    frames = [
        {"frame_path": path, "frame_index": index, "timestamp_seconds": timestamp}
        for index, (path, timestamp) in enumerate(zip(paths, timestamps, strict=True))
    ]
    if frame_indices is not None and not frame_indices:
        return []
    if packing == 4:
        starts = (
            range(0, len(frames), 4)
            if frame_indices is None
            else range(frame_indices.start // 4 * 4, frame_indices.stop, 4)
        )
        results = [
            _grid(frames[start : start + 4], asset_root, quality) for start in starts
        ]
    else:
        selected_frames = (
            frames if frame_indices is None else [frames[index] for index in frame_indices]
        )
        results = []
        for frame in selected_frames:
            path = resolve_within(
                asset_root, frame["frame_path"], context="wearer frame image"
            )
            results.append(
                ObservationImage(
                    mimetypes.guess_type(path.name)[0] or "image/jpeg",
                    base64.b64encode(path.read_bytes()).decode(),
                    _frame_label(frame["frame_index"], frame["timestamp_seconds"]),
                    [{**frame, "position": "single"}],
                )
            )
    return [_fit_image(image, max_edge, quality) for image in results]
