"""Keep the code-only checkout testable without copying local research inputs."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def pytest_collection_modifyitems(items):
    for item in items:
        for requirement in item.iter_markers(name="requires_local_data"):
            missing = [path for path in requirement.args if not (ROOT / path).exists()]
            if missing:
                item.add_marker(pytest.mark.skip(reason=f"Local inputs absent: {', '.join(missing)}"))
