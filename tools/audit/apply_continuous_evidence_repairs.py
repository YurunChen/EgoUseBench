"""Turn a confirmed continuous-video review failure into source evidence metadata.

The Coding Agent decides which tasks need motion or order evidence.  This tool
persists that task-local decision in the screened source record. Evidence
requirements are read from that owner; executable tasks are not rewritten.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from egocua.confidence_packets import CONTINUOUS_VIDEO_REQUIREMENT
from egocua.lineage import file_sha256, resolve_continuous_media_video
from egocua.schema import load_tasks
from egocua.synthesis_reports import REPORTS
from egocua.workspace import read_json_strict, resolve_within, write_json_atomic

REPORT_PATH = REPORTS["assessment"].path


def continuous_repair_ids(report: Mapping[str, Any]) -> list[str]:
    """Return two-pass, high-confidence repairs caused by missing video motion."""

    rows = report.get("results")
    if not isinstance(rows, list):
        raise ValueError("Coding-Agent report has no results")
    task_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("status") != "completed":
            continue
        assessments = row.get("assessments")
        if not isinstance(assessments, list) or len(assessments) != 2:
            continue
        verdicts = [
            assessment.get("verdict")
            for assessment in assessments
            if isinstance(assessment, Mapping)
        ]
        if len(verdicts) != 2 or any(
            not isinstance(verdict, Mapping)
            or verdict.get("decision") != "repair"
            or verdict.get("confidence") != "high"
            for verdict in verdicts
        ):
            continue
        second_verdict = verdicts[1]
        if not isinstance(second_verdict, Mapping):
            continue
        checks = second_verdict.get("counterexample_checks")
        if isinstance(checks, Mapping) and checks.get("continuous_interval") == "fail":
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError("continuous-video repair has no task_id")
            task_ids.append(task_id)
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("continuous-video repair task IDs are not unique")
    return sorted(task_ids)


def apply(
    root: Path,
    *,
    report_path: str | Path = REPORT_PATH,
    expected_count: int | None = None,
) -> dict[str, int]:
    report = read_json_strict(
        resolve_within(root, report_path, context="review report")
    )
    task_ids = continuous_repair_ids(report)
    if expected_count is not None and len(task_ids) != expected_count:
        raise ValueError(
            f"expected {expected_count} continuous-video repairs, found {len(task_ids)}"
        )

    tasks = {str(task["id"]): task for task in load_tasks(root / "data/tasks")}
    unknown = set(task_ids) - set(tasks)
    if unknown:
        raise ValueError(f"review report references unknown tasks: {sorted(unknown)}")

    manifests: dict[Path, dict[str, Any]] = {}
    record_paths: dict[str, Path] = {}
    added_video_hashes = 0
    for task_id in task_ids:
        task = tasks[task_id]
        dossier_ref = task.get("source", {}).get("dossier")
        if not isinstance(dossier_ref, str):
            raise ValueError(f"{task_id}: source dossier is missing")
        dossier_path = resolve_within(root, dossier_ref, context=f"{task_id} dossier")
        dossier = manifests.setdefault(dossier_path, read_json_strict(dossier_path))
        dataset = dossier.get("dataset")
        source_root = (
            dataset.get("source_root") if isinstance(dataset, Mapping) else None
        )
        if not isinstance(source_root, str):
            raise ValueError(f"{task_id}: dataset source_root is missing")
        matches = [
            record
            for record in dossier.get("tasks", [])
            if isinstance(record, dict) and record.get("task_id") == task_id
        ]
        if len(matches) != 1:
            raise ValueError(f"{task_id}: expected exactly one source record")
        record = matches[0]
        construct = record.get("construct")
        if not isinstance(construct, dict):
            raise ValueError(f"{task_id}: source construct is missing")
        requirements = construct.setdefault("evidence_requirements", [])
        if not isinstance(requirements, list) or any(
            not isinstance(requirement, str) for requirement in requirements
        ):
            raise ValueError(f"{task_id}: invalid evidence_requirements")
        if CONTINUOUS_VIDEO_REQUIREMENT not in requirements:
            requirements.append(CONTINUOUS_VIDEO_REQUIREMENT)
        source = record.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("media_id"), str):
            raise ValueError(f"{task_id}: source media_id is missing")
        if not isinstance(source.get("media_sha256"), str):
            video = resolve_continuous_media_video(
                root,
                source_root,
                source["media_id"],
                context=task_id,
            )
            source["media_sha256"] = file_sha256(video)
            added_video_hashes += 1
        record_paths[task_id] = dossier_path

    for path, manifest in manifests.items():
        write_json_atomic(path, manifest)

    return {
        "repairs": len(task_ids),
        "source_manifests": len(set(record_paths.values())),
        "added_video_hashes": added_video_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--report", default=REPORT_PATH.as_posix())
    parser.add_argument("--expect", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            apply(
                Path(args.root).resolve(),
                report_path=args.report,
                expected_count=args.expect,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
