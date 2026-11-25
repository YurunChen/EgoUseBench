"""Containment-safe asset resolution shared by the sites and the harness."""

from __future__ import annotations

from pathlib import Path


def resolve_within(
    root: str | Path,
    relative_path: str | Path,
    *,
    context: str,
    must_exist: bool = True,
) -> Path:
    """Resolve one relative path and reject paths that escape ``root``."""

    root_path = Path(root).resolve()
    candidate = Path(relative_path)
    if not str(candidate).strip() or candidate.is_absolute():
        raise ValueError(
            f"{context}: path must be relative to the workspace: {relative_path}"
        )
    resolved = (root_path / candidate).resolve()
    if resolved == root_path or root_path not in resolved.parents:
        raise ValueError(f"{context}: path escapes the workspace: {relative_path}")
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved
