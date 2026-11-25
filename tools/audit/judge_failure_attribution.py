"""Prepare or run independent evidence-grounded failure Judges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egocua.attribution_batch import (
    BatchOptions,
    prepare_judge_batch,
    prepare_judge_validation,
    run_judge_batch,
    judge_transport,
)
from egocua.failure_attribution import FAILURE_REPORT_PATH
from egocua.workspace import read_json_strict, write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--report", type=Path, default=FAILURE_REPORT_PATH)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/failure-judge.json")
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--affected-run-id", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument("--prepare", action="store_true")
    preparation.add_argument("--prepare-validation", action="store_true")
    parser.add_argument("--validation-phase", choices=("calibration", "validation"))
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--new-validation", action="store_true")
    args = parser.parse_args(argv)
    if args.new_validation and not args.prepare_validation:
        parser.error("--new-validation requires --prepare-validation")
    root = args.root.resolve()
    options = BatchOptions(
        root=root,
        batch_id=args.batch_id,
        report=args.report,
        run_ids=frozenset(args.run_id),
        affected_run_ids=frozenset(args.affected_run_id),
        concurrency=args.concurrency,
        resume_failed=args.resume_failed,
        validation_phase=args.validation_phase,
        new_validation=args.new_validation,
    )
    if args.prepare_validation:
        selection = prepare_judge_validation(options)
        write_json_atomic(options.directory / "validation-preparation.json", selection)
        summary = {
            "calibration": len(selection["calibration_run_ids"]),
            "validation": len(selection["validation_run_ids"]),
        }
    elif args.prepare:
        manifest = prepare_judge_batch(options)
        write_json_atomic(options.directory / "preparation.json", manifest)
        summary = {
            "candidates": len(manifest["candidates"]),
            "skipped": len(manifest["skipped"]),
        }
    else:
        config = read_json_strict(root / args.config)
        batch = run_judge_batch(
            options,
            config,
            lambda: judge_transport(config, root / args.env),
        )
        results = list(batch["results"].values())
        summary = {
            "completed": sum(item["status"] == "completed" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
            "reused": len(batch.get("reused", [])),
            "interrupted": batch.get("interrupted", False),
            "stop_reason": batch.get("stop_reason"),
            "skipped": len(batch["skipped"]),
        }
    print(
        json.dumps({"batch_id": args.batch_id, **summary}, ensure_ascii=False, indent=2)
    )
    return 130 if summary.get("interrupted") else 1 if summary.get("failed", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
