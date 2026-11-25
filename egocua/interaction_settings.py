"""Authoring materials and protocol choices for the matched RQ2 settings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re

from .timeline import ACTION_ALIGNED_REPLAY, COMPUTER_USE, OFFLINE_EGO_USE, Condition
from .workspace import read_json_strict

INTERACTION_SETTINGS = "interaction_settings"
SETTING_ORDER = (COMPUTER_USE, OFFLINE_EGO_USE, "online")
MATERIALS = "data/experiments/interaction-settings.json"
ENTITY_REFERENCE = re.compile(r"\b[pml]_[a-f0-9]{12}\b", re.IGNORECASE)


def load_computer_requests(root: Path, task_ids: list[str]) -> dict[str, str]:
    """Require a distinct authored request for every selected original task."""
    payload = read_json_strict(root / MATERIALS)
    if not isinstance(payload, dict) or set(payload) != {"schema_version", "computer_use"}:
        raise ValueError(f"{MATERIALS}: expected schema_version and computer_use")
    requests = payload["computer_use"]
    if payload["schema_version"] != 1 or not isinstance(requests, dict):
        raise ValueError(f"{MATERIALS}: unsupported material format")
    missing = set(task_ids) - requests.keys()
    if missing:
        raise ValueError(f"{MATERIALS}: missing requests for {sorted(missing)}")
    selected = {}
    for task_id in task_ids:
        request = requests[task_id]
        if not isinstance(request, str) or not request.strip() or ENTITY_REFERENCE.search(request):
            raise ValueError(f"{MATERIALS}: invalid Computer Use request for {task_id}")
        selected[task_id] = request.strip()
    return selected


def setting_conditions(online: Condition) -> tuple[Condition, Condition, Condition]:
    """Keep the formal online condition identical and derive two isolated inputs."""
    if (
        online.name != "online" or online.schedule != "online"
        or online.scope != "full" or online.ask != "open"
        or online.replay_timing != ACTION_ALIGNED_REPLAY
        or online.release_at is not None or online.change_cue
        or online.follow_up_after_finish or online.strategy_hint
    ):
        raise ValueError("interaction settings require the unmodified formal online replay")
    return (
        Condition(COMPUTER_USE, COMPUTER_USE, "none", "denied", None),
        replace(online, name=OFFLINE_EGO_USE, schedule=OFFLINE_EGO_USE, visual_history="full"),
        online,
    )
