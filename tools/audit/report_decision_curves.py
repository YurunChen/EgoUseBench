"""What the corpus's decision curves say, and which tasks still have none.

This is a report, not a gate. The judgements — is the tempting candidate really
tempting at that frame, does the instruction give the answer away, is the wearer's
order really what settles it — are made by whoever synthesises the task while
looking at the frames, and adjudicated by one blinded reviewer on the review
site. Neither of those is something a number can stand in for, so nothing here
counts, scores, or thresholds.

What it does do is state the corpus back: how many tasks were established by
reading frames rather than by declaring four fields, where each curve settles, and
what settles it.

Usage:
    PYTHONPATH=. uv run python tools/audit/report_decision_curves.py --root .
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from egocua import decision
from egocua.environment import load_site_data
from egocua.schema import load_tasks
from egocua.sources import source_record
from egocua.workspace import write_json_atomic

REPORT = "artifacts/reports/decision-curves.json"


def _curve_summary(
    task: Mapping[str, Any], construct: Mapping[str, Any], entity_ids: set[str]
) -> dict[str, Any]:
    curve = decision.parse(
        construct["decision"],
        decision.frame_ids(task["visual_packets"]),
        entity_ids,
        str(task["id"]),
        task=task,
    )
    return {
        "established_by": curve.established_by,
        "candidates": len(curve.candidates),
        "settled_at": curve.settled_at,
        "settled_by": "frames" if curve.settled_by_frames else "utterance",
        "narrowed_by": curve.narrowing_cause,
        "indexical": curve.is_indexical,
        "tempting": list(curve.tempting_entity_ids),
    }


def build_report(root: Path) -> dict[str, Any]:
    tasks = load_tasks(root / "data/tasks")
    sites: dict[str, set[str]] = {}
    rows: dict[str, dict[str, Any]] = {}
    for task in tasks:
        site = str(task["site"])
        if site not in sites:
            sites[site] = {
                str(entity["entity_id"])
                for entity in load_site_data(root, site)["entities"]
            }
        construct = source_record(task, root=root).get("construct") or {}
        rows[str(task["id"])] = {
            "track": task["track"],
            "site": site,
            **_curve_summary(task, construct, sites[site]),
        }

    established = list(rows.values())
    return {
        "schema_version": 1,
        "interpretation": (
            "Every live task carries a curve — a frame-by-frame reading of which "
            "candidates were still possible — because nothing else compiles. What this "
            "shows is where each one settles and what settled it, which is the corpus "
            "saying what kind of difficulty it actually has."
        ),
        "summary": {
            "tasks": len(rows),
            "settled_by": dict(Counter(row["settled_by"] for row in established)),
            "narrowed_by": dict(Counter(row["narrowed_by"] for row in established)),
            "indexical": sum(1 for row in established if row["indexical"]),
        },
        "tasks": dict(sorted(rows.items())),
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
