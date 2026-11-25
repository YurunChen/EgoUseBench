"""Load recorded study results and their frozen scoring goals for attribution."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .checker import check_run, rescore_run
from .controlled_experiments import load_frozen_study, read_study
from .failure_attribution import attribute_run
from .run_ledger import load_run_record
from .workspace import read_json_strict, resolve_within


def study_source(
    studies: list[str], evidence: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reuse unique recorded runs; reject changed scores or conflicting references."""
    rows: dict[str, dict[str, Any]] = {}
    source_studies: dict[str, list[str]] = {}
    coverage = []
    for reference in studies:
        path = Path(reference)
        if path.is_absolute():
            path = path.resolve().relative_to(root)
        directory = resolve_within(root, path, context="attribution study", must_exist=False)
        definition = read_study(directory)
        results = read_json_strict(directory / "results.json")
        current = (directory / "plan.json").is_file()
        tasks = {}
        if current:
            arms, _, _ = load_frozen_study(directory)
            tasks = {(arm.name, task["id"]): task for arm in arms for task in arm.tasks}
        executions = {}
        for execution in definition.get("execution_results", []):
            run_id = execution.get("run_id")
            if not run_id:
                if execution.get("valid") is True:
                    raise ValueError("valid study execution is missing run_id")
                continue
            if run_id in executions:
                raise ValueError(f"study contains duplicate execution: {run_id}")
            executions[run_id] = execution
        counts: dict[str, dict[str, int]] = {}
        seen: set[str] = set()
        for stored in results["runs"]:
            run_id = stored["run_id"]
            if run_id in seen:
                raise ValueError(f"study contains duplicate result: {run_id}")
            seen.add(run_id)
            execution = executions.get(run_id)
            if not execution or not execution.get("artifact_directory"):
                raise ValueError(f"scored study execution is missing artifact_directory: {run_id}")
            artifact = Path(execution["artifact_directory"])
            if artifact.is_absolute():
                artifact = artifact.resolve().relative_to(root)
            if run_id in rows and rows[run_id]["run_artifact"] != artifact.as_posix():
                raise ValueError(f"study run has conflicting artifact paths: {run_id}")
            run_directory = resolve_within(root, artifact, context="study run", must_exist=False)
            record = load_run_record(run_directory)
            if (record["meta"]["run_id"] != run_id
                    or record["task_id"] != stored["task_id"]
                    or record["track"] != stored["track"]
                    or record["meta"]["model"] != stored["model"]):
                raise ValueError(f"study result does not match archived run identity: {run_id}")
            checked = stored
            if current:
                run = execution["run"]
                if run["task_id"] != stored["task_id"]:
                    raise ValueError(f"study execution task mismatch: {run_id}")
                scoring = execution.get("scoring", {}).get("task_source")
                task = tasks[(run["arm"], run["task_id"])]
                checked = (rescore_run(task, record) if scoring == "current_frozen_task"
                           else check_run(record["task_snapshot"], record))
                if any(checked[key] != stored[key] for key in ("run_valid", "e2e_success")):
                    raise ValueError(f"archived scoring differs from recorded result: {run_id}")
            row = attribute_run(
                checked, record, qualification=evidence.get("run_qualifications", {}).get(run_id)
            )
            if current:
                row["setting"] = execution["run"]["condition"]
            row["run_artifact"] = artifact.as_posix()
            if run_id in rows:
                if rows[run_id] != row:
                    raise ValueError(f"study run has conflicting recorded grades: {run_id}")
            rows[run_id] = row
            sources = source_studies.setdefault(run_id, [])
            if path.as_posix() not in sources:
                sources.append(path.as_posix())
            count = counts.setdefault(row.get("setting", "unknown"), {"runs": 0, "failures": 0})
            count["runs"] += 1
            count["failures"] += row["run_verdict"] == "valid_model_failure"
        coverage.append({"study": path.as_posix(), "settings": counts})
    known = set(rows)
    scoped = {
        key: [deepcopy(item) for item in evidence.get(key, []) if item["run_id"] in known]
        for key in ("assessments", "ratings")
    }
    scoped["run_qualifications"] = {
        key: deepcopy(value) for key, value in evidence.get("run_qualifications", {}).items()
        if key in known
    }
    scoped["adjudications"] = {
        key: deepcopy(value) for key, value in evidence.get("adjudications", {}).items()
        if key in known
    }
    if "judge_validation" in evidence:
        scoped["judge_validation"] = deepcopy(evidence["judge_validation"])
    return {"failure_attributions": list(rows.values()), "source_studies": source_studies,
            "study_coverage": coverage}, scoped
