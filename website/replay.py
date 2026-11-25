"""Validate the assets referenced by the project site's replay index."""

from pathlib import Path
from typing import Any

def validate_replay_bundle(samples: list[dict[str, Any]], root: Path) -> None:
    for sample in samples:
        for run in sample["runs"]:
            assets = ["wearer.mp4", "poster.jpg"]
            for step in run["steps"]:
                assets.extend(f"{kind}-{step['step']}.jpg" for kind in ("page", "view"))
            for asset in assets:
                path = root / run["id"] / asset
                if not path.is_file():
                    raise FileNotFoundError(f"Incomplete website replay bundle: {path}")
