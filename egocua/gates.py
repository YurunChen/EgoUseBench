from __future__ import annotations

from pathlib import Path
from typing import Any

from .environment import load_site_data
from .lineage import LineageError, validate_source_lineage
from .schema import ValidationError, validate_task
from .sources import source_record
from .synthesis_workflow import SynthesisWorkflow, TaskStatus
from .workspace import resolve_within


def validate_task_for_model_run(
    task: dict[str, Any],
    *,
    root: str | Path,
) -> None:
    """Check executable task inputs without requiring task-publication reviews."""
    site_data = load_site_data(root, task["site"], require_assets=True)
    validate_task(task, site_data, root=root, require_assets=True)
    try:
        source_record(task, root=root)
        resolve_within(
            root,
            task["source"]["frame_provenance"],
            context=f"{task['id']} frame provenance",
        )
        if "task_spec" in task["source"]:
            resolve_within(
                root, task["source"]["task_spec"], context=f"{task['id']} task spec"
            )
    except (ValueError, FileNotFoundError) as exc:
        raise ValidationError(str(exc)) from exc


def validate_task_for_human_run(task: dict[str, Any], *, root: str | Path) -> None:
    """Allow exact-harness human collection only for tasks the workflow marks ready."""

    site_data = load_site_data(root, task["site"], require_assets=True)
    validate_task(task, site_data, root=root, require_assets=True)
    try:
        validate_source_lineage(task, root=root)
    except LineageError as exc:
        raise ValidationError(str(exc)) from exc
    row = SynthesisWorkflow(root).status([task["id"]])["tasks"][task["id"]]
    if row["status"] != TaskStatus.READY:
        raise ValidationError(
            f"{task['id']}: human run requires a ready task, "
            f"current status is {row['status']} ({row['next_action']})"
        )
