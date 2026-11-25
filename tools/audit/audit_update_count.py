"""Read-only audit of registered request-count pairs; never launches a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egocua.update_count import audit_case, load_cases


def main() -> None:
    parser = argparse.ArgumentParser(description="只读核查更新次数对照，不调用模型或写文件")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--task-id", action="append")
    args = parser.parse_args()
    rows = [audit_case(args.root.resolve(), case) for case in load_cases(args.root, args.task_id)]
    included = [row for row in rows if row["included"]]
    failed = sum(bool(row["errors"]) for row in included)
    print(json.dumps({
        "passed": len(included) - failed,
        "failed": failed,
        "excluded": len(rows) - len(included),
        "tasks": rows,
    }, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
