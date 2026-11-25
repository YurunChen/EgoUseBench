"""Build the current reviewable task package from compiled candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from egocua.workspace import read_json_strict
from egocua.schema import clarification_basis, task_transition
from egocua.selection import load_selection, task_ids
from egocua.synthesis_reports import REPORTS
from egocua.synthesis_workflow import SynthesisWorkflow
from egocua.workspace import write_json_atomic
from egocua.confidence_report import build_confidence_report
from tools.audit.report_task_difficulty import build_report as build_difficulty_report

FINAL_ROOT = Path("data/final-tasks")
TASKS_ROOT = Path("data/tasks")
CONFIDENCE_REPORT = REPORTS["confidence"].path
DIFFICULTY_REPORT = Path("artifacts/reports/task-difficulty.json")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _compiled_tasks(root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    tasks: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted((root / TASKS_ROOT).glob("*/*.json")):
        task = read_json_strict(path)
        task_id = str(task["id"])
        if task_id in tasks:
            raise ValueError(f"duplicate compiled task: {task_id}")
        tasks[task_id] = (task, path)
    return tasks


def build_release_plan(
    selection: dict[str, Any],
    task_index: list[dict[str, Any]],
) -> dict[str, Any]:
    declared_ids = set(task_ids(selection))
    actual_ids = {str(task["task_id"]) for task in task_index}
    if declared_ids != actual_ids:
        raise ValueError("release plan does not match the compiled tasks")
    by_track = Counter(str(task["track"]) for task in task_index)
    by_subtype = Counter(f"{task['track']}/{task['subtype']}" for task in task_index)
    clarification_bases = Counter(
        str(task["clarification_basis"])
        for task in task_index
        if task.get("clarification_basis") is not None
    )
    bands = Counter(task["difficulty"]["structural"]["band"] for task in task_index)
    burdens = Counter(
        burden
        for task in task_index
        for burden in task["difficulty"]["structural"].get("burdens", [])
    )
    risks = Counter(
        risk
        for task in task_index
        for risk in task["difficulty"]["structural"].get("design_risks", [])
    )
    return {
        "schema_version": 1,
        "policy": "all_task_set_members_passed_lifecycle",
        "summary": {
            "release_tasks": len(task_index),
            "open_slots": 0,
            "reserve_tasks": 0,
            "by_track": dict(sorted(by_track.items())),
            "by_subtype": dict(sorted(by_subtype.items())),
            "clarification_bases": dict(sorted(clarification_bases.items())),
            "structural_bands": dict(sorted(bands.items())),
            "burdens": dict(sorted(burdens.items())),
            "design_risks": dict(sorted(risks.items())),
        },
        "task_ids": sorted(actual_ids),
    }


def build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    selection = load_selection(root)
    selected_ids = task_ids(selection)
    workflow = SynthesisWorkflow(root)
    compiled = _compiled_tasks(root)
    missing = sorted(set(selected_ids) - compiled.keys())
    if missing:
        raise ValueError(f"missing compiled definitions for formal tasks: {', '.join(missing)}")
    compiled = {task_id: compiled[task_id] for task_id in selected_ids}

    confidence = build_confidence_report(root)
    write_json_atomic(root / CONFIDENCE_REPORT, confidence)
    reports = workflow.release_evidence(selected_ids)
    difficulty = build_difficulty_report(root)
    write_json_atomic(root / DIFFICULTY_REPORT, difficulty)
    reports["difficulty"] = {
        "path": DIFFICULTY_REPORT.as_posix(),
        "sha256": _sha256(root / DIFFICULTY_REPORT),
        **difficulty["summary"],
    }

    final_root = root / FINAL_ROOT
    staging = root / f"{FINAL_ROOT.as_posix()}.building"
    if staging.exists():
        shutil.rmtree(staging)

    task_index = []
    by_track: Counter[str] = Counter()
    by_site: Counter[str] = Counter()
    by_subtype: Counter[str] = Counter()
    clarification_bases: Counter[str] = Counter()
    for task_id, (task, source_path) in sorted(compiled.items()):
        track = str(task["track"])
        site = str(task["site"])
        subtype = task_transition(task)
        destination = staging / "tasks" / track / f"{task_id}.json"
        _write_json(destination, task)
        task_index.append(
            {
                "task_id": task_id,
                "track": track,
                "site": site,
                "subtype": subtype,
                "clarification_basis": clarification_basis(task),
                "path": destination.relative_to(staging).as_posix(),
                "sha256": _sha256(destination),
                "source_path": source_path.relative_to(root).as_posix(),
                "confidence": confidence["tasks"][task_id],
                "difficulty": difficulty["tasks"][task_id],
            }
        )
        by_track[track] += 1
        by_site[site] += 1
        by_subtype[f"{track}/{subtype}"] += 1
        basis = clarification_basis(task)
        if basis is not None:
            clarification_bases[basis] += 1

    release_plan = build_release_plan(selection, task_index)
    _write_json(staging / "release-plan.json", release_plan)

    summary: dict[str, Any] = {
        "tasks": len(task_index),
        "by_track": dict(sorted(by_track.items())),
        "by_site": dict(sorted(by_site.items())),
        "by_subtype": dict(sorted(by_subtype.items())),
        "clarification_bases": dict(sorted(clarification_bases.items())),
        "release": release_plan["summary"],
        "confidence": confidence["summary"],
        "difficulty": difficulty["summary"],
    }
    manifest = {
        "schema_version": 1,
        "snapshot_kind": "frozen_release",
        "admission_authority": "SynthesisWorkflow.status",
        "summary": summary,
        "reports": reports,
        "release_plan": {
            "path": "release-plan.json",
            "sha256": _sha256(staging / "release-plan.json"),
        },
        "tasks": task_index,
    }
    _write_json(staging / "manifest.json", manifest)

    if final_root.exists():
        shutil.rmtree(final_root)
    staging.rename(final_root)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(json.dumps(build(Path(args.root)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
