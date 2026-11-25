"""Materialize answer-free Coding-Agent confidence review packets."""

from __future__ import annotations

import argparse
import json

from egocua.confidence_packets import OUTPUT_DIR, build_packets
from egocua.experiment import TASK_REVIEW_PLAN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--plan", default=TASK_REVIEW_PLAN)
    args = parser.parse_args()
    report = build_packets(args.root, output_dir=args.output_dir, plan_path=args.plan)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
