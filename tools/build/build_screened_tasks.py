"""CLI for compiling the review queue into executable tasks."""

from __future__ import annotations

import argparse
import json

from egocua.construction import build_selected, check_selected, write_selected

build = build_selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build every source-screened task that has a frozen site mapping."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--task-id",
        action="append",
        help="recompile only these selected tasks; the full build remains the release check",
    )
    args = parser.parse_args()
    if args.check:
        report = check_selected(args.root)
    elif args.task_id:
        report = write_selected(args.root, task_ids=args.task_id)
    else:
        report = write_selected(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
