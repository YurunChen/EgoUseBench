"""Read-only views of one canonical run; scores use its historical task snapshot."""

from pathlib import Path
from typing import Any

from .run_ledger import inspect_run_artifact, load_run_record
from .workspace import read_json_strict

RUN_VIEWS = ("summary", "input", "trace", "actions", "result", "score")


def read_run_view(
    path: str | Path, view: str, results: str | Path | None = None
) -> dict[str, Any] | list[Any]:
    if results is not None and view != "score":
        raise ValueError("--results is only used with --view score")
    if view not in RUN_VIEWS:
        raise ValueError(f"unknown run view: {view}")
    if view == "summary":
        return inspect_run_artifact(path)
    record = load_run_record(path)
    if view in {"trace", "actions"}:
        return record[view]
    if view == "result":
        return {
            "run_status": record["run_status"],
            "initial_state": record["initial_state"],
            "live_state": record.get("live_state"),
            "termination_reason": record["meta"].get("termination_reason"),
        }
    if view == "score":
        result = read_json_strict(
            Path(results)
            if results is not None
            else Path(path).resolve().parent.parent / "results.json"
        )
        matches = [
            row for row in result["runs"] if row["run_id"] == record["meta"]["run_id"]
        ]
        if not matches:
            raise ValueError("run has no saved score in this experiment's results.json")
        return matches
    trace = record["trace"]
    if not isinstance(trace, dict):
        raise ValueError("run has no structured model-input trace")
    return {
        "model": record["meta"]["model"],
        "initial_context": record.get("initial_context"),
        "agent_settings": trace["agent_settings"],
        "wearer_image_layouts": trace["wearer_image_layouts"],
        "requests": trace["llm_requests"],
        "observations": [
            {
                "step": entry["metadata"]["step_number"],
                "text": entry["state_message"],
                "website": entry["state"],
            }
            for entry in trace["history"]
        ],
    }
