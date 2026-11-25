"""Build the portable forty-task review package from registered frozen inputs."""

import argparse
from datetime import datetime
from pathlib import Path

from egocua.portable_review import export_review_package


def main():
    parser = argparse.ArgumentParser(description="导出40题离线质量核查包")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or args.root / "artifacts/human-review-packages" / (
        "EgoUseBench-40题离线审核-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    print(export_review_package(args.root, destination), flush=True)


if __name__ == "__main__":
    main()
