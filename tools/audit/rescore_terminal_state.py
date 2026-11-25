"""Rescore an archived experiment without altering its runs or old results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egocua.checker import EVALUATOR_VERSION, check_run
from egocua.run_ledger import load_run_record
from egocua.workspace import resolve_within, write_json_atomic


def rescore_experiment(summary: dict, root: Path) -> dict:
    tasks = {}
    for task_id, item in summary["tasks"].items():
        reference = Path(item["artifact"])
        if reference.is_absolute():
            reference = reference.resolve().relative_to(root)
        directory = resolve_within(
            root, reference, context="source run", must_exist=False
        )
        record = load_run_record(directory)
        if (
            record["task_id"] != task_id
            or record["meta"]["run_id"] != item["checked"]["run_id"]
        ):
            raise ValueError(f"{task_id}: experiment and run disagree")
        row = {
            "run_id": record["meta"]["run_id"],
            "artifact": reference.as_posix(),
            "status": "not_rescorable",
            "reasons": [],
        }
        if "live_state" not in record:
            row["reasons"].append("missing_terminal_live_state")
        definition = record.get("task_snapshot")
        if definition is None:
            row["reasons"].append("missing_matching_historical_task")
        elif definition["id"] != task_id:
            raise ValueError(
                f"{task_id}: historical task snapshot belongs to a different task"
            )
        if definition is not None and not row["reasons"]:
            checked = check_run(definition, record)
            row.update(status="rescored", checked=checked)
        tasks[task_id] = row
    scored = [row["checked"] for row in tasks.values() if row["status"] == "rescored"]
    valid = [row for row in scored if row["e2e_success"] is not None]
    successes = sum(row["e2e_success"] is True for row in valid)
    return {
        "schema_version": 1,
        "evaluator_version": EVALUATOR_VERSION,
        "summary": {
            "tasks": len(tasks),
            "rescored": len(scored),
            "not_rescorable": len(tasks) - len(scored),
            "valid_scored": len(valid),
            "successes": successes,
            "success_rate": successes / len(valid) if valid else None,
        },
        "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    source = resolve_within(root, args.summary, context="source summary")
    output = resolve_within(
        root, args.output, context="rescore output", must_exist=False
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {output}")
    if root / "artifacts/reports" not in output.parents:
        raise ValueError("rescore output must be in artifacts/reports")
    result = rescore_experiment(json.loads(source.read_text()), root)
    result["source_summary"] = source.relative_to(root).as_posix()
    write_json_atomic(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
