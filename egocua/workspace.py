"""Canonical workspace paths and containment-safe asset resolution."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from sites._paths import resolve_within

__all__ = [
    "read_json_strict",
    "resolve_within",
    "write_bytes_atomic",
    "write_json_atomic",
    "write_text_atomic",
]


def read_json_strict(path: str | Path) -> Any:
    """Read JSON without silently accepting duplicate object keys."""

    source = Path(path)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{source}: duplicate JSON key {key!r}")
            value[key] = item
        return value

    return json.loads(source.read_text(), object_pairs_hook=unique_object)


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
