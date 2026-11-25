"""Validate evidence-grounded trajectory reviews and build the PULSE report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from egocua.failure_attribution import (
    FAILURE_EVIDENCE_PATH,
    FAILURE_REPORT_PATH,
    FAILURE_SOURCE_PATH,
    attribute_run,
)
from egocua.attribution_sources import study_source
from egocua.failure_report import build_failure_report
from egocua.run_ledger import load_run_record
from egocua.workspace import read_json_strict, resolve_within, write_json_atomic

DEFAULT_INPUT = FAILURE_SOURCE_PATH.as_posix()
DEFAULT_EVIDENCE = FAILURE_EVIDENCE_PATH.as_posix()
DEFAULT_OUTPUT = FAILURE_REPORT_PATH.as_posix()


def _benchmark_source(summary: dict, evidence: dict, root: Path) -> tuple[dict, dict]:
    rows = []
    exclusions = []
    qualifications = evidence.get("run_qualifications", {})
    for task_id, item in summary["tasks"].items():
        reference = Path(item["artifact"])
        if reference.is_absolute():
            reference = reference.resolve().relative_to(root)
        directory = resolve_within(
            root, reference, context="benchmark run", must_exist=False
        )
        record = load_run_record(directory)
        if item.get("status") == "not_rescorable":
            if (
                task_id != record["task_id"]
                or item.get("run_id") != record["meta"]["run_id"]
            ):
                raise ValueError("benchmark summary does not match run artifact")
            reasons = item.get("reasons")
            if (
                not isinstance(reasons, list)
                or not reasons
                or any(not isinstance(reason, str) or not reason for reason in reasons)
            ):
                raise ValueError(
                    "unscorable benchmark source requires explicit reasons"
                )
            exclusions.append(
                {
                    "run_id": item["run_id"],
                    "task_id": task_id,
                    "run_artifact": directory.relative_to(root).as_posix(),
                    "status": "not_rescorable",
                    "reasons": reasons,
                }
            )
            continue
        if "checked" not in item:
            raise ValueError(
                "benchmark source requires checked results or not_rescorable status"
            )
        checked = item["checked"]
        if (
            task_id != record["task_id"]
            or checked["run_id"] != record["meta"]["run_id"]
        ):
            raise ValueError("benchmark summary does not match run artifact")
        row = attribute_run(
            checked, record, qualification=qualifications.get(checked["run_id"])
        )
        row["run_artifact"] = directory.relative_to(root).as_posix()
        rows.append(row)
    run_ids = {row["run_id"] for row in rows}
    scoped_evidence: dict[str, Any] = {
        key: [entry for entry in evidence.get(key, []) if entry["run_id"] in run_ids]
        for key in ("assessments", "ratings")
    }
    scoped_evidence["run_qualifications"] = {
        run_id: review for run_id, review in qualifications.items() if run_id in run_ids
    }
    scoped_evidence["adjudications"] = {
        run_id: decision for run_id, decision in evidence.get("adjudications", {}).items()
        if run_id in run_ids
    }
    return {
        "failure_attributions": rows,
        "source_exclusions": exclusions,
    }, scoped_evidence



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input", default=DEFAULT_INPUT)
    inputs.add_argument(
        "--study",
        action="append",
        help="Read recorded results from a study directory; repeat for multiple studies",
    )
    inputs.add_argument(
        "--benchmark-summary", help="Use one completed benchmark experiment"
    )
    inputs.add_argument("--judge-config", help="Use the frozen study scope registered for this Judge")
    parser.add_argument("--evidence", default=DEFAULT_EVIDENCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    source_path = root / args.input
    evidence_path = root / args.evidence
    evidence: dict[str, Any] = (
        read_json_strict(evidence_path) if evidence_path.is_file() else {}
    )
    studies = args.study
    if args.judge_config:
        studies = read_json_strict(root / args.judge_config)["studies"]
        if (not isinstance(studies, list) or not studies
                or any(not isinstance(value, str) or not value for value in studies)
                or len(set(studies)) != len(studies)):
            raise ValueError("Judge studies must be distinct nonempty study references")
    if studies:
        source, evidence = study_source(studies, evidence, root)
    elif args.benchmark_summary:
        source, evidence = _benchmark_source(
            read_json_strict(root / args.benchmark_summary),
            evidence,
            root,
        )
    else:
        source = read_json_strict(source_path)
    report = build_failure_report(
        source,
        evidence,
        root=root,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if studies:
        report["source_studies"] = source["source_studies"]
        report["study_coverage"] = source["study_coverage"]
    if args.benchmark_summary:
        report["source_benchmark"] = args.benchmark_summary
        report["run_qualifications"] = evidence["run_qualifications"]
    write_json_atomic(root / args.output, report)
    print(
        json.dumps(
            {
                "runs": len(report["attributions"]),
                "source_not_rescorable": len(report["source_exclusions"]),
                "valid_model_failures": report["statistics"]["summary"][
                    "valid_model_failures"
                ],
                "unknown_failures": report["statistics"]["summary"]["unknown_failures"],
                "processing_complete": report["workflow"]["processing_complete"],
                "resolved_complete": report["workflow"]["resolved_complete"],
                "resolved_attributions": report["statistics"]["summary"][
                    "resolved_attributions"
                ],
                "human_review_complete": report["workflow"]["human_review_complete"],
                "coding_agent_pending": len(
                    report["workflow"]["coding_agent_pending_run_ids"]
                ),
                "rule_evaluation_pending": len(
                    report["workflow"]["rule_evaluation_pending_run_ids"]
                ),
                "human_review_required": len(
                    report["human_review"]["required_run_ids"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
