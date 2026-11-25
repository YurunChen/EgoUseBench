"""Build the task confidence report through the shared review service."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egocua.confidence_report import OUTPUT_PATH, build_confidence_report
from egocua.workspace import resolve_within, write_json_atomic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH.as_posix())
    args = parser.parse_args()
    workspace = Path(args.root).resolve()
    report = build_confidence_report(workspace)
    output = resolve_within(
        workspace,
        args.output,
        context="task confidence output",
        must_exist=False,
    )
    write_json_atomic(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
