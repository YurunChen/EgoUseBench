from __future__ import annotations

import argparse
import json

from egocua.reviews import ReviewAuditError, audit_construction_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit independent construction reviews.")
    parser.add_argument("--spec", default="data/reviews/construction.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--require-all-admitted", action="store_true")
    args = parser.parse_args()
    try:
        result = audit_construction_reviews(args.spec, root=args.root)
    except ReviewAuditError as exc:
        raise SystemExit(f"review audit error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_all_admitted and not result["all_admitted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
