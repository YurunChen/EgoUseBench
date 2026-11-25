"""Build artifacts/reports/necessity.json from scored probe runs.

Input: one or more score reports produced by ``egocua score`` (their ``runs``
lists carry per-run ``condition``, ``e2e_success``, and ``final_in_tempting``).
Output: per task × condition success and tempting-pick rates, in the shape the
acceptance audit's ``necessity`` advisory check reads.

The v3 synthesis contract (.scratch/task-synthesis-v3/spec.md §2) uses this
report two ways: shortcut probes (``text_only`` / ``first_frame``) succeeding
mark a task as solvable without progressive evidence, and the ``initial_only``
tempting-pick rate verifies that the registered tempting candidate actually
tempts the reference model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from egocua.workspace import write_json_atomic

NECESSITY_REPORT = "artifacts/reports/necessity.json"


def _rate(values: list[Any]) -> float | None:
    known = [bool(value) for value in values if value is not None]
    return mean(known) if known else None


def build_necessity_report(score_reports: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for report in score_reports:
        for row in report.get("runs", []):
            condition = row.get("condition") or {}
            name = condition.get("name")
            task_id = row.get("task_id")
            if not isinstance(name, str) or not isinstance(task_id, str):
                continue
            grouped[(task_id, name)].append(row)
    tasks: dict[str, dict[str, Any]] = defaultdict(dict)
    for (task_id, name), rows in sorted(grouped.items()):
        tasks[task_id][name] = {
            "runs": len(rows),
            "e2e_success_rate": _rate([row.get("e2e_success") for row in rows]),
            "tempting_pick_rate": _rate([row.get("final_in_tempting") for row in rows]),
        }
    return {
        "schema_version": 1,
        "interpretation": (
            "Per-task probe outcomes. A shortcut probe (text_only / first_frame) with "
            "e2e_success_rate >= 2/3 flags the task as solvable without progressive "
            "evidence; an initial_only tempting_pick_rate < 2/3 flags a tempting "
            "candidate that does not actually tempt the reference model."
        ),
        "tasks": dict(tasks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate scored probe runs into the necessity report."
    )
    parser.add_argument(
        "scores", nargs="+", help="score report JSON files from `egocua score`"
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=NECESSITY_REPORT)
    args = parser.parse_args()
    reports = [json.loads(Path(path).read_text()) for path in args.scores]
    payload = build_necessity_report(reports)
    output = Path(args.root).resolve() / args.output
    write_json_atomic(output, payload)
    print(
        json.dumps(
            {
                "tasks": len(payload["tasks"]),
                "conditions": sorted(
                    {name for rows in payload["tasks"].values() for name in rows}
                ),
                "output": str(output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
