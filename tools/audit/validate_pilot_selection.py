"""CLI for the task-selection invariants owned by egocua.selection."""

from __future__ import annotations

from egocua.selection import (  # noqa: F401
    SelectionError,
    task_ids,
    validate_selection,
)
from egocua.selection import main


if __name__ == "__main__":
    main()
