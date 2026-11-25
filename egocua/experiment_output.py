"""Experiment folders and local timestamps; no model calls or mutable task state."""

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
from zoneinfo import ZoneInfo

from .workspace import write_json_atomic

LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


def local_timestamp() -> str:
    return datetime.now(LOCAL_TIMEZONE).isoformat(timespec="milliseconds")


def experiment_folder_name(plan: dict[str, Any]) -> str:
    category, variable, model = (str(plan[key]) for key in ("category", "variable", "model"))
    if not model or "\\" in model or any(
        not label or "/" in label or "\\" in label for label in (category, variable)
    ):
        raise ValueError("experiment labels must be nonempty folder-name segments")
    if "/" in model:
        model = quote(model, safe="")
    stamp = datetime.now(LOCAL_TIMEZONE).strftime("%Y%m%d-%H%M%S-%f")
    return f"{stamp}_{category}_{variable}_{model}"


def create_experiment_directory(root: str | Path, plan: dict[str, Any]) -> Path:
    directory = Path(root).resolve() / experiment_folder_name(plan)
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(
        directory / "plan.json", {"created_at": local_timestamp(), **plan}
    )
    return directory
