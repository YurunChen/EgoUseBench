"""Export the project site's samples and task statistics.

Samples are archived Online runs on EgoMart whose recordings may be shown
publicly. For each run the script cuts the source footage behind the frames
the agent received (sped up so one step lasts the same on every page, and held
as a still where no new frame arrived), and copies the page screenshots it saw
and the messages and actions in run.json into
website/static/samples/<sample>/, then writes website/replays.local.json together with
counts of the formal task set. It never edits run archives or task files.

The mock store renders a third-party wordmark in its header, so page
screenshots get the EgoMart name painted over it and are cropped above the
footer.

Usage:
    uv run python tools/render/render_website_data.py --root .
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from egocua.sampling_media import SourceMedia
from website.replay import validate_replay_bundle


RESULTS = {
    "gpt-6.1-terra": "20261116-190219-186569_变量对照_当前任务最终复评_gpt-6.1-terra",
    "gemini-3.8-flash": "20261116-190709-954660_正式评测_当前任务最终复评_gemini-3.8-flash",
    "doubao-seed-2-1-pro-260915": (
        "20261116-190912-137466_正式评测_当前任务最终复评_doubao-seed-2-1-pro-260915"
    ),
    "deepseek-flash": "20261116-191126-187777_变量对照_当前任务最终复评_deepseek-flash",
    "kimi-k2.6": "20261116-191623-993294_变量对照_当前任务最终复评_kimi-k2.6",
    "qwen3.5-27b": "20261116-192601-052272_变量对照_当前任务最终复评_qwen3.5-27b",
}
MODEL_NAMES = {
    "gpt-6.1-terra": "GPT-6.1 Terra",
    "gemini-3.8-flash": "Gemini 3.8 Flash",
    "doubao-seed-2-1-pro-260915": "Doubao Seed 2.1 Pro",
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "kimi-k2.6": "Kimi K2.6",
    "qwen3.5-27b": "qwen3.5-27b",
}
ONLINE_CONDITIONS = {"online::online", "formal::online"}
SAMPLES = (
    ("candidate-mart-graft-p-22", ("gpt-6.1-terra", "qwen3.5-27b")),
    ("candidate-mart-graft-replace-2026101601", ("gpt-6.1-terra", "deepseek-flash")),
    ("candidate-mart-refresh-replace-2026110501", ("gpt-6.1-terra", "kimi-k2.6")),
    ("candidate-mart-refresh-retain-2026110522", ("gpt-6.1-terra", "kimi-k2.6")),
    ("candidate-mart-clarify-position-2026102901", (
        "doubao-seed-2-1-pro-260915", "gpt-6.1-terra",
    )),
)
PUBLIC_SOURCES = {
    "CaptainCook4D": "Apache 2.0",
    "HD-EPIC": "CC BY 4.0",
    "EgoOops": "CC BY-SA 4.0",
    "HoloAssist": "CDLA Permissive 2.0",
}
FRAMES_PER_STEP = 8
PLAYBACK_SPEED = 2.5
CLIP_FPS = 25
CLIP_HEIGHT = 540
PAGE_CROP = (0, 0, 1512, 800)
HEADER_FILL = (19, 25, 33)
WORDMARK_BOX = (10, 6, 128, 56)
PLACEHOLDER_BOX = (168, 16, 330, 46)
FONT = "/System/Library/Fonts/HelveticaNeue.ttc"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def task_files(root: Path) -> dict[str, Path]:
    return {path.stem: path for path in (root / "data/tasks").glob("*/*.json")}


def transition(task: dict) -> str:
    if task["track"] == "graft":
        return task["graft"]["transition"]
    if task["track"] == "refresh":
        return task["refresh"]["policy"]
    return "resolve"


def source_dataset(root: Path, task: dict) -> str:
    return read_json(root / task["source"]["frame_provenance"])["source_dataset"]


def task_stats(root: Path, files: dict[str, Path]) -> dict:
    members = read_json(root / "data/selection.json")["task_set"]
    tracks, sites, sources = Counter(), Counter(), Counter()
    for track, ids in members.items():
        for task_id in ids:
            task = read_json(files[task_id])
            tracks[(track, transition(task))] += 1
            sites[task["site"]] += 1
            sources[source_dataset(root, task)] += 1
    return {
        "tasks": sum(len(ids) for ids in members.values()),
        "tracks": [
            {
                "track": track,
                "total": len(ids),
                "transitions": [
                    {"name": name, "count": count}
                    for (owner, name), count in tracks.most_common()
                    if owner == track
                ],
            }
            for track, ids in members.items()
        ],
        "sites": [{"name": name, "count": count} for name, count in sites.most_common()],
        "sources": [
            {"name": name, "count": count, "public": name in PUBLIC_SOURCES}
            for name, count in sources.most_common()
        ],
    }


def online_run(root: Path, model: str, task_id: str) -> tuple[dict, Path]:
    results = read_json(root / "experiments" / RESULTS[model] / "results.json")
    record = next(
        run for run in results["runs"]
        if run["task_id"] == task_id and run["condition"]["name"] in ONLINE_CONDITIONS
    )
    run_dir = next((root / "experiments").glob(f"*/runs/{record['run_id']}"))
    return record, run_dir


def step_frames(run: dict) -> list[list[str]]:
    """New frames delivered before each step, padded to a fixed batch size."""
    delivered: dict[int, list[str]] = {}
    for action in run["actions"]:
        if action["type"] == "visual_context_update":
            delivered[action["step"]] = action["payload"]["frame_paths"]
    later = [path for paths in delivered.values() for path in paths]
    opening = []
    for image in run["trace"]["wearer_image_layouts"]:
        if "before step 1:" not in image["observation_label"]:
            continue
        for frame in image["layout"]:
            if frame["frame_path"] not in later and frame["frame_path"] not in opening:
                opening.append(frame["frame_path"])
    delivered[1] = opening[-FRAMES_PER_STEP:]
    frames, last = [], delivered[1]
    for step in range(1, run["steps"] + 1):
        batch = delivered.get(step) or last[-1:]
        last = batch
        hold = [batch[-1]] * max(0, FRAMES_PER_STEP - len(batch))
        frames.append((batch + hold)[:FRAMES_PER_STEP])
    return frames


def describe(output: dict, result: dict) -> dict:
    if not output:
        return {"verb": "no valid action", "text": ""}
    if "done" in output:
        verb = "standby" if result.get("standby") else "done"
        return {"verb": verb, "text": output["done"]["text"]}
    code = output.get("browser_action", {}).get("code", "")
    fill = re.match(r"fill\(bid='\w+', value='(.*)'\)", code)
    if fill:
        return {"verb": "type", "text": fill.group(1)}
    if code.startswith("click"):
        return {"verb": "click", "text": ""}
    if code.startswith("goto"):
        return {"verb": "go to", "text": ""}
    ask = output.get("ask_user") or {}
    if ask:
        return {"verb": "ask", "text": ask.get("question", "")}
    return {"verb": code.split("(")[0] or "wait", "text": ""}


def step_records(run: dict, names: dict[str, str]) -> list[dict]:
    events: dict[int, list[dict]] = {}
    cart_after: dict[int, list[str]] = {}
    for action in run["actions"]:
        step = action.get("step")
        kind = action["type"]
        if kind == "search":
            events.setdefault(step, []).append({"kind": "search", "text": action["query"]})
        elif kind == "open_entity":
            events.setdefault(step, []).append(
                {"kind": "open", "text": names.get(action["entity_id"], action["entity_id"])}
            )
        elif kind in ("select_entity", "deselect_entity"):
            cart_after[step] = action["state_after"]["selected_entity_ids"]
            events.setdefault(step, []).append({
                "kind": "add" if kind == "select_entity" else "remove",
                "text": names.get(action["entity_id"], action["entity_id"]),
            })
    messages: dict[int, list[dict]] = {}
    for action in run["actions"]:
        if action["type"] == "instruction_injection":
            messages.setdefault(action["step"], []).append(
                {"who": "wearer", "text": action["payload"]}
            )
        elif action["type"] == "clarification_answer":
            messages.setdefault(action["step"], []).extend([
                {"who": "agent", "text": action["question"]},
                {"who": "wearer", "text": action["payload"]},
            ])
    first = re.search(
        r'The wearer said before step 1: "(.*?)"\n',
        run["trace"]["history"][0]["state_message"],
    )
    messages.setdefault(1, []).insert(0, {"who": "wearer", "text": first.group(1)})

    records, cart = [], []
    for entry in run["trace"]["history"]:
        step = entry["metadata"]["step_number"]
        output = ((entry.get("model_output") or {}).get("action") or [{}])[0]
        cart = cart_after.get(step, cart)
        records.append({
            "step": step,
            "path": "/" + entry["state"]["url"].split("/", 3)[-1],
            "action": describe(output, entry["result"][0] if entry["result"] else {}),
            "events": events.get(step, []),
            "messages": messages.get(step, []),
            "cart_after": len(cart),
        })
    return records


def source_second(path: str) -> float | None:
    try:
        return float(Path(path).stem)
    except ValueError:
        return None


def clip_segments(frames: list[list[str]]) -> list[dict]:
    """Delivered seconds as source footage; a repeated frame holds as a still."""
    segments: list[dict] = []
    previous = None
    for path in (path for batch in frames for path in batch):
        second = source_second(path)
        last = segments[-1] if segments else None
        if path == previous or second is None:
            if last and last["kind"] == "hold" and last["path"] == path:
                last["seconds"] += 1
            else:
                segments.append({"kind": "hold", "path": path, "seconds": 1})
        elif last and last["kind"] == "live" and second == last["start"] + last["seconds"]:
            last["seconds"] += 1
        else:
            segments.append({"kind": "live", "path": path, "start": second, "seconds": 1})
        previous = path
    return segments


def render_clip(root: Path, task: dict, frames: list[list[str]], out: Path) -> None:
    source = SourceMedia(task, root)
    encode = [
        "-an", "-c:v", "libx264", "-crf", "28", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-r", str(CLIP_FPS), "-video_track_timescale", "12800",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for index, segment in enumerate(clip_segments(frames)):
            count = round(segment["seconds"] * CLIP_FPS / PLAYBACK_SPEED)
            part = Path(tmp) / f"{index:03d}.mp4"
            if segment["kind"] == "live":
                window = [segment["start"], segment["start"] + segment["seconds"]]
                video, media = source.covering_media(window)
                inputs = ["-ss", f"{window[0] - media.source_start_seconds:.3f}", "-i", str(video)]
                speed = f"setpts=(PTS-STARTPTS)/{PLAYBACK_SPEED},"
            else:
                inputs = ["-loop", "1", "-i", str(root / segment["path"])]
                speed = ""
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", *inputs,
                 "-vf", f"{speed}fps={CLIP_FPS},scale=-2:{CLIP_HEIGHT},setsar=1",
                 "-frames:v", str(count), *encode, str(part)],
                check=True,
            )
            parts.append(part)
        listing = Path(tmp) / "parts.txt"
        listing.write_text("".join(f"file '{part}'\n" for part in parts), encoding="utf-8")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
             "-i", str(listing), "-c", "copy", "-movflags", "+faststart",
             str(out / "wearer.mp4")],
            check=True,
        )


def render_stills(root: Path, frames: list[list[str]], out: Path) -> None:
    for step, batch in enumerate(frames, start=1):
        image = Image.open(root / batch[len(batch) // 2]).convert("RGB")
        image.thumbnail((240, 135))
        image.save(out / f"view-{step}.jpg", quality=80)
    poster = Image.open(root / frames[0][0]).convert("RGB")
    poster.thumbnail((960, 540))
    poster.save(out / "poster.jpg", quality=84)


def rebrand(page: Image.Image, query_filled: bool) -> Image.Image:
    page = page.crop(PAGE_CROP)
    draw = ImageDraw.Draw(page)
    draw.rectangle(WORDMARK_BOX, fill=HEADER_FILL)
    draw.text((20, 18), "egomart", font=ImageFont.truetype(FONT, 26), fill="white")
    if not query_filled:
        draw.rectangle(PLACEHOLDER_BOX, fill="white")
        draw.text(
            (178, 20), "Search EgoMart",
            font=ImageFont.truetype(FONT, 18), fill=(110, 110, 110),
        )
    return page


def render_pages(run_dir: Path, records: list[dict], out: Path) -> None:
    previous_verb = ""
    for record in records:
        query_filled = record["path"].startswith("/search") or previous_verb == "type"
        previous_verb = record["action"]["verb"]
        page = Image.open(
            run_dir / "images/website" / f"step-{record['step']}.png"
        ).convert("RGB")
        rebrand(page, query_filled).resize((1134, 600), Image.LANCZOS).save(
            out / f"page-{record['step']}.jpg", quality=82
        )


def export_run(root, task, model, names, samples_dir) -> dict:
    record, run_dir = online_run(root, model, task["id"])
    run = read_json(run_dir / "run.json")
    sample_id = f"{task['id'].removeprefix('candidate-mart-')}--{model.split('-')[0]}"
    out = samples_dir / sample_id
    out.mkdir(parents=True, exist_ok=True)
    frames = step_frames(run)
    records = step_records(run, names)
    render_clip(root, task, frames, out)
    render_stills(root, frames, out)
    render_pages(run_dir, records, out)
    final_ids = run.get("live_state", {}).get("selected_entity_ids", [])
    return {
        "id": sample_id,
        "model": MODEL_NAMES[model],
        "success": record["e2e_success"],
        "asked": bool(record.get("asked")),
        "run": record["run_id"],
        "final_cart": [names.get(item, item) for item in final_ids],
        "steps": records,
    }


def build_replay_data(root: Path, samples_dir: Path) -> dict:
    files = task_files(root)
    catalog = read_json(root / "sites/egomart/products.json")["products"]
    names = {item["item_id"]: item["name"] for item in catalog}
    samples = []
    for task_id, models in SAMPLES:
        task = read_json(files[task_id])
        dataset = source_dataset(root, task)
        if dataset not in PUBLIC_SOURCES:
            raise ValueError(f"{task_id} uses {dataset}, which may not be shown publicly")
        samples.append({
            "task_id": task_id,
            "track": task["track"],
            "transition": transition(task),
            "site": task["site"],
            "source": dataset,
            "license": PUBLIC_SOURCES[dataset],
            "runs": [export_run(root, task, model, names, samples_dir) for model in models],
        })
    return {
        "seconds_per_step": FRAMES_PER_STEP / PLAYBACK_SPEED,
        "stats": task_stats(root, files),
        "samples": samples,
    }


def export_replays(root: Path) -> None:
    website_dir = root / "website"
    published_samples = website_dir / "static/samples"
    published_index = website_dir / "replays.local.json"
    backup = website_dir / ".replay-previous"
    if backup.exists():
        raise FileExistsError(f"Recover the previous replay bundle from {backup} before exporting")
    with tempfile.TemporaryDirectory(dir=website_dir, prefix=".replay-") as folder:
        staging = Path(folder)
        samples = staging / "samples"
        samples.mkdir()
        data = build_replay_data(root, samples)
        validate_replay_bundle(data["samples"], samples)
        index = staging / published_index.name
        index.write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
        )
        if published_samples.exists():
            published_samples.rename(backup)
        try:
            samples.rename(published_samples)
            index.replace(published_index)
        except OSError:
            if published_samples.exists():
                published_samples.rename(samples)
            if backup.exists():
                backup.rename(published_samples)
            raise
        if backup.exists():
            shutil.rmtree(backup)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    export_replays(args.root.resolve())


if __name__ == "__main__":
    main()
