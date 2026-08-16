from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from egocua.schema import load_tasks
from egocua.workspace import write_json_atomic


DOMAIN_WORKFLOWS = {
    "Knowledge Lookup": "knowledge",
    "Media Retrieval": "media",
    "Local / Maps": "maps",
    "Others": "community",
}


def _signals(instruction: str) -> list[str]:
    text = instruction.casefold()
    patterns = {
        "structured_report": r"\b(report|provide|identify|retrieve|state|tell me)\b",
        "open_resource": r"\b(open|navigate|stop on)\b",
        "compare": r"\b(compare|difference|lowest|highest|top[- ]rated|top[- ]listed)\b",
        "multi_resource": r"\b(both|each|two|cross-check)\b",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def _source_record(task: dict[str, Any], root: Path) -> dict[str, Any]:
    dossier_path = root / task["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())
    task_id = task["source"].get("task_id", task["id"])
    try:
        return next(record for record in dossier["tasks"] if record["task_id"] == task_id)
    except StopIteration as exc:
        raise ValueError(f"{task['id']}: source dossier has no matching record") from exc


def mine(
    tasks_path: str | Path,
    ego2web_path: str | Path,
    *,
    root: str | Path = ".",
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    tasks = load_tasks(tasks_path)
    ego2web_rows = json.loads(Path(ego2web_path).read_text())
    rows_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in ego2web_rows:
        rows_by_video.setdefault(str(row["vid"]), []).append(row)

    candidates: list[dict[str, Any]] = []
    for task in tasks:
        source = _source_record(task, workspace)["source"]
        media_id = str(source["media_id"])
        for row in rows_by_video.get(media_id, []):
            workflow = DOMAIN_WORKFLOWS.get(str(row.get("domain")))
            if workflow is None:
                continue
            candidates.append({
                "candidate_id": f"cross-{task['id']}-ego2web-{row['id']}",
                "status": "mined",
                "source_task_id": task["id"],
                "track": task["track"],
                "source": {
                    "dataset": "Ego2Web",
                    "record_id": row["id"],
                    "media_id": media_id,
                    "domain": row["domain"],
                    "original_annotation": row["instruction"],
                    "visual_clues": row.get("visual_clues"),
                },
                "released_observation": task["visual_packets"],
                "workflow_candidate": workflow,
                "lexical_signals": _signals(str(row["instruction"])),
                "blocking_checks": [
                    "continuous_video_review_for_cross_domain_instruction",
                    "frozen_resource_and_distractor_linkage",
                    "deterministic_final_state_design",
                ],
            })

    by_workflow = Counter(candidate["workflow_candidate"] for candidate in candidates)
    by_track = Counter(candidate["track"] for candidate in candidates)
    return {
        "schema_version": 1,
        "status": "mined",
        "source_dataset": "Ego2Web",
        "selection_rule": (
            "Non-e-commerce Ego2Web annotations whose source video is already used by a "
            "current EgoCUA-Bench task. Keyword signals are retrieval hints, not task labels."
        ),
        "summary": {
            "candidates": len(candidates),
            "source_tasks": len({candidate["source_task_id"] for candidate in candidates}),
            "by_workflow": dict(sorted(by_workflow.items())),
            "by_track": dict(sorted(by_track.items())),
        },
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="data/tasks")
    parser.add_argument("--ego2web", default="reference/Ego2Web/dataset/ego2web.json")
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output", default="data/sources/cross_domain_extension_candidate_queue.json"
    )
    args = parser.parse_args()
    payload = mine(args.tasks, args.ego2web, root=args.root)
    write_json_atomic(args.output, payload)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
