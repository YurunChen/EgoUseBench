from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from egocua.environment import entity_visual_path, load_site_data
from egocua.schema import referenced_entity_ids
from egocua.workspace import resolve_within


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _load_task(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise ValueError(f"{path}: invalid task")
    return payload


def render_site_audit(
    task_path: str | Path,
    output_path: str | Path,
    *,
    root: str | Path = ".",
    observation_only: bool = False,
) -> dict[str, Any]:
    """Render released observations and the referenced frozen site entities."""
    workspace = Path(root).resolve()
    task_file = resolve_within(workspace, task_path, context="task")
    output_file = resolve_within(
        workspace,
        output_path,
        context="site audit output",
        must_exist=False,
    )
    task = _load_task(task_file)
    entities: list[dict[str, Any]] = []
    if not observation_only:
        site_data = load_site_data(workspace, task["site"])
        entity_by_id = {entity["entity_id"]: entity for entity in site_data["entities"]}
        entities = [
            entity_by_id[entity_id] for entity_id in referenced_entity_ids(task)
        ]

    columns = 4
    cell_width = 360
    cell_height = 410
    header_height = 84
    frames = [
        (packet["id"], index, frame)
        for packet in task["visual_packets"]
        for index, frame in enumerate(packet["frame_paths"])
    ]
    observation_rows = (len(frames) + columns - 1) // columns if observation_only else 1
    observation_height = observation_rows * 250
    canvas = Image.new(
        "RGB",
        (
            columns * cell_width,
            header_height
            + observation_height
            + ((len(entities) + columns - 1) // columns) * cell_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    body_font = _font(17)
    small_font = _font(14)
    task_id = task["id"]
    audit_name = "Released frames" if observation_only else f"{task['site']} task audit"
    draw.text((24, 18), f"{audit_name} - {task_id}", fill="#111827", font=title_font)
    draw.text(
        (24, 53),
        (
            "Exact released frame packets in temporal order."
            if observation_only
            else "Released observations and task-referenced entities in the frozen site."
        ),
        fill="#475569",
        font=small_font,
    )

    frame_width = (
        cell_width if observation_only else (columns * cell_width - 48) // len(frames)
    )
    for frame_number, (packet_id, index, frame) in enumerate(frames):
        frame_file = resolve_within(
            workspace, frame, context=f"{task_id} Released Observation frame"
        )
        with Image.open(frame_file) as source:
            frame_image = source.convert("RGB")
        frame_image.thumbnail((frame_width - 24, 185), Image.Resampling.LANCZOS)
        if observation_only:
            row, column = divmod(frame_number, columns)
            left = column * cell_width
            top = header_height + row * 250 + 10
        else:
            left = 24 + frame_number * frame_width
            top = header_height + 10
        canvas.paste(frame_image, (left + (frame_width - frame_image.width) // 2, top))
        draw.text(
            (left + 18, top + 202),
            f"{packet_id}:{index}",
            fill="#334155",
            font=small_font,
        )

    for index, entity in enumerate(entities):
        row, column = divmod(index, columns)
        left = column * cell_width
        top = header_height + observation_height + row * cell_height
        color = "#475569"
        draw.rectangle(
            (left + 8, top + 8, left + cell_width - 8, top + cell_height - 8),
            outline=color,
            width=4,
        )

        image_path = entity_visual_path(task["site"], entity)
        if isinstance(image_path, str) and image_path:
            image_file = resolve_within(
                workspace,
                image_path,
                context=f"{task_id} entity image",
            )
            with Image.open(image_file) as source:
                entity_image = source.convert("RGB")
            entity_image.thumbnail((300, 260), Image.Resampling.LANCZOS)
            canvas.paste(
                entity_image,
                (left + (cell_width - entity_image.width) // 2, top + 22),
            )

        text_top = top + 292
        draw.text(
            (left + 18, text_top),
            f"{index + 1:02d}  {entity['entity_id']}",
            fill=color,
            font=body_font,
        )
        title = entity.get("title") or entity.get("name") or ""
        for offset, line in enumerate(textwrap.wrap(str(title), width=42)[:3]):
            draw.text(
                (left + 18, text_top + 28 + offset * 20),
                line,
                fill="#111827",
                font=small_font,
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_file, format="JPEG", quality=92, optimize=True)
    return {
        "task_id": task_id,
        "entities": len(entities),
        "released_frames": len(frames),
        "observation_only": observation_only,
        "output": output_file.as_posix(),
        "sha256": hashlib.sha256(output_file.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one task's observations and referenced frozen site entities."
    )
    parser.add_argument("task")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--observation-only", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            render_site_audit(
                args.task,
                args.output,
                root=args.root,
                observation_only=args.observation_only,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
