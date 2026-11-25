from __future__ import annotations

from pathlib import Path

import pytest

from egocua.workspace import read_json_strict


def test_strict_json_reader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"task": 1, "task": 2}')

    with pytest.raises(ValueError, match="duplicate JSON key 'task'"):
        read_json_strict(path)


def test_owned_json_sources_have_unique_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        *root.glob("configs/*.json"),
        *root.glob("data/**/*.json"),
        *root.glob("sites/*.json"),
    ]

    for path in paths:
        read_json_strict(path)
