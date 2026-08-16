"""Canonical workspace paths and containment-safe asset resolution."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any


def resolve_within(
    root: str | Path, relative_path: str | Path, *, context: str, must_exist: bool = True
) -> Path:
    """Resolve one relative path and reject paths that escape ``root``."""

    root_path = Path(root).resolve()
    candidate = Path(relative_path)
    if not str(candidate).strip() or candidate.is_absolute():
        raise ValueError(f"{context}: path must be relative to the workspace: {relative_path}")
    resolved = (root_path / candidate).resolve()
    if resolved == root_path or root_path not in resolved.parents:
        raise ValueError(f"{context}: path escapes the workspace: {relative_path}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def write_text_atomic(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text)
    os.replace(temporary, destination)


def write_bytes_atomic(path: str | Path, content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, destination)


def write_json_atomic(path: str | Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
