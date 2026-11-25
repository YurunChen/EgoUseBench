"""Capture real YouTube result windows for every EgoMedia task query.

The capture is discovery evidence. It preserves upstream ordering and stable video
IDs without changing a task, checker, or Mock ranking. A later materialization
step may freeze selected rows into the shared EgoMedia catalog.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from egocua.schema import load_tasks
from egocua.workspace import write_json_atomic
from tools.catalog.catalog_clients import yt_search_json

OUTPUT_DIR = Path("data/sources/upstream_search_snapshots/youtube")
RESULT_LIMIT = 40
MAX_CONCURRENCY = 4


def _clean(row: Mapping[str, Any], rank: int) -> dict[str, Any] | None:
    video_id = str(row.get("id") or "").strip()
    title = str(row.get("title") or "").strip()
    if len(video_id) != 11 or not title:
        return None
    duration = row.get("duration")
    return {
        "rank": rank,
        "youtube_id": video_id,
        "title": title,
        "channel_title": str(row.get("channel") or row.get("uploader") or "YouTube"),
        "duration_seconds": int(duration)
        if isinstance(duration, (int, float))
        else None,
        "view_count": int(row["view_count"])
        if isinstance(row.get("view_count"), (int, float))
        else None,
        "page_url": f"https://www.youtube.com/watch?v={video_id}",
        "thumbnail_url": f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg",
    }


def _capture(task_id: str, stage: str, query: str) -> tuple[str, str, dict[str, Any]]:
    rows = yt_search_json(query, limit=RESULT_LIMIT, flat=True, timeout=180)
    results = [
        cleaned
        for rank, row in enumerate(rows, start=1)
        if (cleaned := _clean(row, rank)) is not None
    ]
    if not results:
        raise RuntimeError(
            f"{task_id}/{stage}: upstream returned only {len(results)} usable results"
        )
    return (
        task_id,
        stage,
        {
            "schema_version": 1,
            "task_id": task_id,
            "site": "egomedia",
            "stage": stage,
            "query": query,
            "provider": "youtube-search-v1",
            "collection_status": "complete",
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "result_count": len(results),
            "collection_anomalies": (
                ["fewer_than_20_results"] if len(results) < 20 else []
            ),
            "results": results,
        },
    )


def capture(
    root: Path, *, concurrency: int, task_id: str | None = None
) -> dict[str, int]:
    tasks = {
        str(task["id"]): task
        for task in load_tasks(root / "data/tasks")
        if task["site"] == "egomedia"
    }
    reviews = json.loads((root / "data/sources/search_query_reviews.json").read_text())[
        "tasks"
    ]
    if task_id is not None:
        if task_id not in tasks and task_id not in reviews:
            raise ValueError(f"unknown EgoMedia task: {task_id}")
        tasks = {task_id: tasks.get(task_id, {"id": task_id, "site": "egomedia"})}
    jobs: list[tuple[str, str, str]] = []
    for task_id in sorted(tasks):
        row = reviews.get(task_id) or {}
        for stage, field in (
            ("before_change", "before_query"),
            ("after_change", "after_query"),
        ):
            query = str(row.get(field) or "").strip()
            if (
                stage == "after_change"
                and query == str(row.get("before_query") or "").strip()
            ):
                continue
            if not query:
                raise ValueError(f"{task_id}: missing reviewed {field}")
            existing_path = root / OUTPUT_DIR / f"{task_id}--{stage}.json"
            if existing_path.is_file():
                existing = json.loads(existing_path.read_text())
                if (
                    existing.get("provider") == "youtube-search-v1"
                    and existing.get("collection_status") == "complete"
                    and existing.get("query") == query
                    and bool(existing.get("results"))
                ):
                    continue
            jobs.append((task_id, stage, query))

    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {
            pool.submit(_capture, task_id, stage, query): (task_id, stage)
            for task_id, stage, query in jobs
        }
        for future in as_completed(pending):
            task_id, stage, payload = future.result()
            write_json_atomic(output / f"{task_id}--{stage}.json", payload)
            completed += 1
            print(f"captured {completed}/{len(jobs)} {task_id} {stage}", flush=True)
    return {"tasks": len(tasks), "captures": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--task-id")
    args = parser.parse_args()
    print(
        json.dumps(
            capture(
                Path(args.root).resolve(),
                concurrency=args.concurrency,
                task_id=args.task_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
