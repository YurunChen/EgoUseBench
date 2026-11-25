"""Describe observed completion without imposing a success-rate ceiling or repairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egocua.retest_scope import build_completion_anomaly_plan
from egocua.workspace import write_json_atomic

DEFAULT_INPUT = "artifacts/reports/task-difficulty-calibration.json"
DEFAULT_OUTPUT = "artifacts/reports/completion-anomaly-repair-plan.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    plan = build_completion_anomaly_plan(
        json.loads((root / args.input).read_text(encoding="utf-8")),
        source_path=args.input,
    )
    write_json_atomic(root / args.output, plan)
    print(json.dumps(plan["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
