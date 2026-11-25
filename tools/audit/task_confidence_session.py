"""Operate the task-local Coding-Agent blind-review session."""

from __future__ import annotations

import argparse
import json

from egocua.confidence_session import (
    ConfidenceSessionError,
    open_pass,
    record_assessment,
    start_review,
)
from egocua.synthesis_workflow import SynthesisWorkflow


def _invalidations(
    values: list[str], parser: argparse.ArgumentParser
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        task_id, separator, reason = value.partition("=")
        if not separator or not task_id.strip() or not reason.strip():
            parser.error("--invalidate-task must use TASK_ID=REASON")
        result[task_id.strip()] = reason.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start")
    mode = start.add_mutually_exclusive_group()
    mode.add_argument("--refresh", action="store_true")
    mode.add_argument("--reset", action="store_true")
    scope = start.add_mutually_exclusive_group()
    scope.add_argument("--task-id", action="append", default=[])
    scope.add_argument("--all-tasks", action="store_true")
    start.add_argument("--invalidate-task", action="append", default=[])

    opened = commands.add_parser("open")
    opened.add_argument("task_id")

    record = commands.add_parser("record")
    record.add_argument("task_id")
    record.add_argument("pass_name")
    record.add_argument("verdict_json")

    args = parser.parse_args()
    try:
        if args.command == "start":
            if args.refresh and not (args.task_id or args.all_tasks):
                parser.error("--refresh requires --task-id or --all-tasks")
            if not args.refresh and (args.task_id or args.all_tasks):
                parser.error("review scope requires --refresh")
            report = start_review(
                args.root,
                reset=args.reset,
                refresh=args.refresh,
                invalidation_reasons=_invalidations(args.invalidate_task, parser),
                task_ids=args.task_id if args.task_id else None,
            )
            result: object = report["summary"]
        elif args.command == "open":
            result = open_pass(args.task_id, root=args.root)
        else:
            verdict = json.loads(args.verdict_json)
            row = record_assessment(
                args.task_id,
                args.pass_name,
                verdict,
                root=args.root,
            )
            result = {
                "task_id": row["task_id"],
                "assessment_status": row["status"],
                "passes": len(row["assessments"]),
                "lifecycle": SynthesisWorkflow(args.root).status([args.task_id])["tasks"][
                    args.task_id
                ],
            }
    except ConfidenceSessionError as exc:
        raise SystemExit(f"confidence session error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
