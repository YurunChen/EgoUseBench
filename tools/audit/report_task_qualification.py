"""Build current development qualification evidence from real Agent trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egocua.qualification_report import REPORT, build_report
from egocua.workspace import write_json_atomic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=REPORT)
    parser.add_argument("--experiment-id", action="append")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    experiment_ids = tuple(args.experiment_id) if args.experiment_id else None
    report = build_report(root, experiment_ids=experiment_ids)
    write_json_atomic(root / args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
