"""Build the structural difficulty coverage profile for current tasks.

This report describes where a task was designed to require independent reasoning.
It does not contain development-model results and is not an empirical difficulty
measurement.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from egocua import decision
from egocua.difficulty import STRUCTURAL_DIFFICULTY_VERSION, structural_difficulty
from egocua.environment import load_site_data
from egocua.run_ledger import task_sha256
from egocua.schema import clarification_basis, load_tasks, task_transition
from egocua.sources import source_record
from egocua.workspace import write_json_atomic

REPORT = "artifacts/reports/task-difficulty.json"


def build_report(root: Path) -> dict[str, Any]:
    tasks = load_tasks(root / "data/tasks")
    site_ids: dict[str, set[str]] = {}
    rows: dict[str, dict[str, Any]] = {}

    for task in sorted(tasks, key=lambda item: str(item["id"])):
        task_id = str(task["id"])
        site = str(task["site"])
        if site not in site_ids:
            site_ids[site] = {
                str(entity["entity_id"])
                for entity in load_site_data(root, site)["entities"]
            }
        record = source_record(task, root=root)
        construct = record.get("construct") or {}
        curve = decision.parse(
            construct["decision"],
            decision.frame_ids(task["visual_packets"]),
            site_ids[site],
            task_id,
            task=task,
        )
        structural = structural_difficulty(task, curve, record)
        rows[task_id] = {
            "task_sha256": task_sha256(task),
            "track": task["track"],
            "subtype": task_transition(task),
            "clarification_basis": clarification_basis(task),
            "site": site,
            "structural": structural.as_dict(),
        }

    bands = Counter(row["structural"]["band"] for row in rows.values())
    challenges = Counter(
        challenge
        for row in rows.values()
        for challenge in row["structural"]["challenges"]
    )
    burdens = Counter(
        burden for row in rows.values() for burden in row["structural"]["burdens"]
    )
    risks = Counter(
        risk for row in rows.values() for risk in row["structural"]["design_risks"]
    )
    opportunities = Counter(
        opportunity
        for row in rows.values()
        for opportunity in row["structural"]["improvement_opportunities"]
    )
    tracks = Counter(str(row["track"]) for row in rows.values())
    subtypes = Counter(f"{row['track']}/{row['subtype']}" for row in rows.values())
    clarification_bases = Counter(
        str(row["clarification_basis"])
        for row in rows.values()
        if row["clarification_basis"] is not None
    )
    sites = Counter(str(row["site"]) for row in rows.values())
    return {
        "schema_version": 2,
        "structural_policy_version": STRUCTURAL_DIFFICULTY_VERSION,
        "interpretation": (
            "L1/L2/L3 describe a reviewed minimum legal solution: single decision, "
            "independent conditions, or dependent objects/states. Missing semantic "
            "evidence stays unclassified. Challenge counts and model outcomes do not "
            "assign a level or prioritize a task."
        ),
        "summary": {
            "tasks": len(rows),
            "types": {
                "tracks": dict(sorted(tracks.items())),
                "subtypes": dict(sorted(subtypes.items())),
                "clarification_bases": dict(sorted(clarification_bases.items())),
                "sites": dict(sorted(sites.items())),
            },
            "structural_bands": dict(sorted(bands.items())),
            "challenge_axes": dict(sorted(challenges.items())),
            "burden_groups": dict(sorted(burdens.items())),
            "design_risks": dict(sorted(risks.items())),
            "tasks_with_design_risk": sum(
                bool(row["structural"]["design_risks"]) for row in rows.values()
            ),
            "improvement_opportunities": dict(sorted(opportunities.items())),
            "tasks_with_improvement_opportunity": sum(
                bool(row["structural"]["improvement_opportunities"])
                for row in rows.values()
            ),
        },
        "tasks": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=REPORT)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    report = build_report(root)
    write_json_atomic(root / args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
