"""Derived experiment scores; raw run snapshots remain the evidence source."""

from pathlib import Path
from typing import Any
from .workspace import write_json_atomic


def score_summary(checked: dict[str, Any]) -> dict[str, Any]:
    """Keep measured results and evaluator identity, not a second answer catalogue."""
    return {
        **{
            key: value
            for key, value in checked.items()
            if key
            not in {"evaluator", "task_sha256", "asset_sha256", "site_data_sha256"}
        },
        "evaluator": {
            key: checked["evaluator"][key] for key in ("version", "predicate")
        },
    }


def write_experiment_results(directory: Path) -> None:
    """Generate scores once from actual run snapshots; never launch a model."""
    from .checker import check_run
    from .run_ledger import load_run_record

    rows = []
    for path in sorted((directory / "runs").glob("*/run.json")):
        record = load_run_record(path.parent)
        checked = check_run(record["task_snapshot"], record)
        rows.append(
            {
                "run_id": record["meta"]["run_id"],
                "task_id": record["task_id"],
                "condition": record["condition"],
                "replicate_id": record["meta"]["replicate_id"],
                "artifact_directory": str(path.parent),
                "model": record["meta"]["model"],
                **{
                    key: checked.get(key)
                    for key in (
                        "run_valid",
                        "run_invalid_reasons",
                        "e2e_success",
                        "steps",
                    )
                },
                "evaluator": score_summary(checked)["evaluator"],
            }
        )
    valid = [row for row in rows if row["run_valid"]]
    success = sum(row["e2e_success"] is True for row in valid)
    write_json_atomic(
        directory / "results.json",
        {
            "runs": rows,
            "summary": {
                "runs": len(rows),
                "valid": len(valid),
                "invalid": len(rows) - len(valid),
                "success": success,
                "success_rate": success / len(valid) if valid else None,
            },
        },
    )
