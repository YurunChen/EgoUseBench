from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from egocua.reviews import (
    ReviewAuditError,
    audit_construction_reviews,
    construction_evidence_from_reviews,
)
from egocua.schema import load_tasks
from egocua.workspace import resolve_within, write_json_atomic


def promote_reviews(
    spec: str, tasks_path: str, output: str, *, root: str = ".",
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    try:
        task_root = resolve_within(
            workspace, tasks_path, context="candidate tasks", must_exist=False
        )
        destination = resolve_within(
            workspace, output, context="construction evidence output", must_exist=False
        )
    except ValueError as exc:
        raise ReviewAuditError(str(exc)) from exc
    tasks = {task["id"]: task for task in load_tasks(task_root)}
    evidence: dict[str, dict[str, Any]] = {}
    audit = audit_construction_reviews(spec, root=workspace)
    for task_id, result in audit["tasks"].items():
        if result["status"] != "admit":
            continue
        if task_id not in tasks:
            raise ReviewAuditError(f"{task_id}: admitted task is absent from {tasks_path}")
        evidence[task_id] = {
            "status": "constructed",
            "construction": construction_evidence_from_reviews(
                tasks[task_id], spec, root=workspace
            ),
        }
    write_json_atomic(destination, {"schema_version": 1, "tasks": evidence})
    return {"promoted": len(evidence), "output": destination.as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive construction validation evidence from admitted review submissions."
    )
    parser.add_argument("--spec", default="data/reviews/construction.json")
    parser.add_argument(
        "--tasks", default="data/tasks"
    )
    parser.add_argument(
        "--output", default="artifacts/reports/construction-evidence.json"
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    try:
        result = promote_reviews(
            args.spec, args.tasks, args.output, root=args.root
        )
    except ReviewAuditError as exc:
        raise SystemExit(f"promotion error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
