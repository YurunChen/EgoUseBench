"""Execution-material comparisons shared by studies and development review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from .environment import entity_visual_path, load_site_data, task_public_site_data
from .experiment import ExperimentPlan
from .run_ledger import SITE_RUNTIME_CONTRACTS, task_local_content
from .timeline import Condition, Timeline
from .update_count import VARIANT_SUFFIX
from .workspace import read_json_strict, resolve_within


def clarified_instruction_snapshot(
    task: dict[str, Any], review: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Resolve one explicitly reviewed wording edit, never infer equivalent text."""
    clarification = review.get("instruction_clarification")
    if clarification is None:
        return None
    if not clarification.get("reviewer") or not clarification.get("reason"):
        raise ValueError("instruction clarification requires its reviewer and reason")
    before = clarification["before_task"]
    if (
        task["instructions"] != clarification["after_instructions"]
        or before["instructions"] == task["instructions"]
        or task_local_content({**before, "instructions": task["instructions"]})
        != task_local_content(task)
    ):
        return None
    return before


def material_mismatch(
    task: dict[str, Any], record: dict[str, Any], plan: ExperimentPlan, *, root: Path
) -> str | None:
    """Compare a run's saved inputs with a task before reusing its outcome."""
    if record.get("task_id") != task["id"]:
        return "wrong_task"
    snapshot = record.get("task_snapshot")
    if not isinstance(snapshot, dict):
        return "missing_task_snapshot"
    if record["meta"]["runtime"].get(
        "initial_site_state"
    ) != plan.initial_site_states.get(str(task["id"])):
        return "changed_initial_site_state"
    condition = Condition.from_dict(record["condition"])
    registered = replace(plan, conditions={condition.name: condition})
    old_inputs, old_paths = path_independent_inputs(
        execution_inputs(snapshot, registered, condition.name)
    )
    new_inputs, new_paths = path_independent_inputs(
        execution_inputs(task, registered, condition.name)
    )
    if old_inputs != new_inputs:
        return "changed_execution_inputs"
    site_problem = site_evidence_mismatch(record, root)
    if site_problem == "missing_current_site_evidence":
        return site_problem
    runtime = record["meta"]["runtime"]
    current_root = root
    if task["id"].endswith(VARIANT_SUFFIX):
        frozen_site_path = plan.site_snapshot_for(task)
        if frozen_site_path is None or plan.frozen_asset_root is None:
            raise ValueError("update-count validation requires the original frozen website and images")
        current_site = read_json_strict(Path(frozen_site_path))
        current_root = Path(plan.frozen_asset_root)
    else:
        current_site = load_site_data(root, str(task["site"]))
    path = Path(runtime["site_snapshot"])
    old_site = read_json_strict(path if path.is_absolute() else root / path)
    if task_public_site_data(snapshot, old_site) != task_public_site_data(
        task, current_site
    ):
        return "changed_public_site"
    visual_problem = visual_mismatch(record, (old_paths, new_paths), root, current_root=current_root)
    if visual_problem:
        return visual_problem
    if site_problem:
        return site_problem
    current_materials = public_site_materials(task, current_site)
    if public_site_materials(
        snapshot, old_site
    ) != current_materials or not site_images_match(current_materials["images"], current_root):
        return "changed_public_site"
    return None


def site_evidence_mismatch(record: dict[str, Any], root: Path) -> str | None:
    """Require recorded website evidence for reuse, without rejudging the old run."""
    runtime = record["meta"]["runtime"]
    snapshot = runtime.get("site_snapshot")
    if not isinstance(snapshot, str) or not snapshot:
        return "missing_current_site_evidence"
    path = Path(snapshot)
    if not (path if path.is_absolute() else root / path).is_file():
        return "missing_current_site_evidence"
    contract = runtime.get("site_contract")
    if not contract:
        return "missing_site_runtime_contract"
    if contract != SITE_RUNTIME_CONTRACTS[record["meta"]["site"]]:
        return "unverified_site_runtime_contract"
    return None


_BASE_CONDITION_INPUTS = (
    "schedule",
    "scope",
    "ask",
    "release_at",
    "frame_interval_seconds",
    "frame_repeat_to_interval_seconds",
    "wearer_frames_per_image",
    "wearer_jpeg_quality",
)
"""Always projected, defaults included; later fields appear only when as_dict emits them."""


def _timeline_inputs(timeline: Timeline) -> dict[str, Any]:
    """Project exactly what reaches the model, so new defaulted fields keep identities."""
    condition = {name: getattr(timeline.condition, name) for name in _BASE_CONDITION_INPUTS}
    condition.update(
        (name, value)
        for name, value in timeline.condition.as_dict().items()
        if name != "name" and name not in condition
    )
    offer = timeline.offer
    return {
        "task_id": timeline.task_id,
        "track": timeline.track,
        "condition": condition,
        "instruction": timeline.instruction,
        "initial_observations": list(timeline.initial_observations),
        "releases": [
            {
                key: value
                for key, value in asdict(release).items()
                if key not in {"streamed", "update"} or value
            }
            for release in timeline.releases
        ],
        "offer": None
        if offer is None
        else {
            "valid_slot": offer.valid_slot,
            "canonical_answer": offer.canonical_answer,
            "max_questions": offer.max_questions,
        },
    }


def _canonical_frame_timestamps(value: Any) -> Any:
    """Source seconds compare by value, not JSON integer/float notation."""
    if isinstance(value, dict):
        return {
            key: [float(second) for second in child]
            if key == "timestamps_seconds"
            else _canonical_frame_timestamps(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_canonical_frame_timestamps(child) for child in value]
    return value


def execution_inputs(task: dict[str, Any], plan: ExperimentPlan, condition: str) -> str:
    timeline = _canonical_frame_timestamps(_timeline_inputs(
        Timeline.compile(
            task,
            plan.condition(condition),
            default_release_step=plan.default_release_step,
            reference_steps=plan.reference_steps,
        )
    ))
    runtime = {
        "adapter": plan.adapter,
        "model": plan.model,
        "model_version": plan.model_version,
        "max_steps": plan.max_steps,
        "release_trigger": plan.release_trigger,
        "pre_release_max_steps": plan.pre_release_max_steps,
        "post_release_max_steps": plan.post_release_max_steps,
    }
    initial_state = plan.initial_site_states.get(str(task["id"]))
    if initial_state is not None:
        runtime["initial_site_state"] = initial_state
    task_inputs = {key: task[key] for key in ("id", "track", "site")}
    scope = task.get("truth", {}).get("site_scope_entity_ids")
    if scope:
        task_inputs["truth"] = {"site_scope_entity_ids": sorted(scope)}
    return json.dumps(
        {
            "task": task_inputs,
            "timeline": timeline,
            "runtime": runtime,
            "default_release_step": plan.default_release_step,
            "reference_steps": plan.reference_steps,
            "scheduler_seed": plan.scheduler_seed,
        },
        sort_keys=True,
    )


def same_public_site(
    tasks: tuple[dict[str, Any], dict[str, Any]],
    snapshots: tuple[str | None, str | None],
) -> bool:
    if any(path is None for path in snapshots):
        return False
    site_data = []
    for value in snapshots:
        path = Path(str(value))
        if not path.is_file():
            return False
        stat = path.stat()
        site_data.append(_frozen_site_data(path, stat.st_mtime_ns, stat.st_size))
    old, current = (
        public_site_materials(task, data)
        for task, data in zip(tasks, site_data, strict=True)
    )
    return old == current


def public_site_materials(
    task: dict[str, Any], site_data: dict[str, Any]
) -> dict[str, Any]:
    """Compare public content and its existing image evidence, not audit notes."""
    public = task_public_site_data(task, site_data)
    visible_ids = {entity["entity_id"] for entity in public["entities"]}
    images = {}
    for entity in site_data["entities"]:
        if entity["entity_id"] not in visible_ids:
            continue
        path = entity_visual_path(task["site"], entity)
        if path:
            provenance = entity.get("image_provenance", {})
            source = entity.get("source", {})
            images[path] = (
                provenance.get("sha256")
                if task["site"] == "egomart"
                else source.get("thumbnail_sha256") or source.get("sha256")
            )
        for preview in entity.get("public_previews", []):
            images[preview["image_path"]] = preview["provenance"].get("sha256")
    missing = [path for path, digest in images.items() if not digest]
    if missing:
        raise ValueError(
            f"{task['id']}: missing historical site image evidence: {missing}"
        )
    return {"data": public, "images": images}


def site_images_match(images: dict[str, str], root: Path) -> bool:
    """Check current pixels against recorded source evidence at material admission."""
    return all(
        hashlib.sha256(
            resolve_within(root, path, context="public site image").read_bytes()
        ).hexdigest()
        == expected
        for path, expected in images.items()
    )


@lru_cache(maxsize=6)
def _frozen_site_data(path: Path, modified_ns: int, size: int) -> dict[str, Any]:
    """Avoid reparsing shared snapshots for each task; timestamps only expire cache."""
    return read_json_strict(path)


def path_independent_inputs(inputs: str) -> tuple[str, list[str]]:
    payload = json.loads(inputs)
    paths: list[str] = []

    def remove_paths(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "frame_paths":
                    paths.extend(child)
                    value[key] = len(child)
                else:
                    remove_paths(child)
        elif isinstance(value, list):
            for child in value:
                remove_paths(child)

    remove_paths(payload)
    return json.dumps(payload, sort_keys=True), paths


def visual_mismatch(
    record: dict[str, Any], paths: tuple[list[str], list[str]], root: Path,
    *, current_root: Path | None = None,
) -> str | None:
    """Compare current frames with evidence saved when the run consumed them."""
    old, current = paths
    if len(old) != len(current):
        return "changed_visual_inputs"
    if not old:
        return None
    if record.get("condition", {}).get("frame_delivery_steps") is not None:
        delivered = {
            path
            for batch in record["meta"].get("released_visual_inputs", [])
            for path in batch["frame_paths"]
        }
        if not delivered or not delivered.issubset(old):
            return "missing_visual_evidence"
        pairs = [
            (left, right)
            for left, right in zip(old, current, strict=True)
            if left in delivered
        ]
    else:
        pairs = list(zip(old, current, strict=True))
    archive = record["meta"]["runtime"].get("visual_asset_root")
    if not isinstance(archive, str) or not archive:
        return "missing_visual_evidence"
    archive_root = Path(archive)
    if not archive_root.is_absolute():
        archive_root = root / archive_root
    if not archive_root.is_dir():
        return "missing_visual_evidence"
    for left, right in pairs:
        try:
            current_bytes = resolve_within(
                current_root or root, right, context="current frame"
            ).read_bytes()
            previous = resolve_within(
                archive_root, left, context="archived frame"
            ).read_bytes()
        except FileNotFoundError:
            return "missing_visual_evidence"
        if previous != current_bytes:
            return "changed_visual_inputs"
    return None



def wearer_input_paths(tasks: list[dict[str, Any]], plan: ExperimentPlan) -> set[str]:
    """Enumerate the compiled opening and every streamed batch, not only V0."""
    return {
        path
        for task in tasks
        for condition in plan.conditions_for(task["track"])
        for path in path_independent_inputs(execution_inputs(task, plan, condition))[1]
    }


def require_frozen_assets(tasks: list[dict[str, Any]], plan: ExperimentPlan) -> None:
    """Reject missing frozen inputs before starting any paid execution."""
    if plan.frozen_asset_root is None:
        return
    root = Path(plan.frozen_asset_root)
    paths = wearer_input_paths(tasks, plan)
    snapshots = {}
    images = {}
    for task in tasks:
        snapshot = plan.site_snapshot_for(task)
        if snapshot is None:
            raise ValueError(f"{task['id']}: missing frozen site snapshot")
        if snapshot not in snapshots:
            snapshots[snapshot] = read_json_strict(Path(snapshot))
        for path, digest in public_site_materials(task, snapshots[snapshot])["images"].items():
            if images.setdefault(path, digest) != digest:
                raise ValueError(f"conflicting frozen website image: {path}")
    for path in sorted(paths | set(images)):
        resolve_within(root, path, context="frozen study asset")
    if not site_images_match(images, root):
        raise ValueError("frozen website image differs from registered source evidence")
