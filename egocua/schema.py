from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .environment import (
    EnvironmentError,
    SITES,
    load_site_data,
    site_entity_ids,
    validate_referenced_site_assets,
)
from .workspace import resolve_within

TRACKS = {"graft", "refresh", "clarify"}
CLARIFICATION_SLOTS = {"position", "appearance", "brand", "variant", "sequence"}
CLARIFICATION_PERMISSION_PHRASES = ("one of the", "either of the", "any of the")
TARGET_GRANULARITIES = {"category", "brand", "variant", "exact_instance"}
PILOT_DISTRIBUTION = {
    "graft": {"C": 55, "P": 9, "R": 10},
    "refresh": {"retain": 7, "replace": 10},
    "clarify": {"variant": 6, "sequence": 4, "position": 4},
}
REMOVED_TASK_FIELDS = {
    "subtype", "conditions", "expectations", "catalog", "environment", "review",
    "review_status", "human_validation", "model_validation", "evaluation_contract",
    "annotation_notes",
}


class ValidationError(ValueError):
    """Raised when a task does not satisfy the benchmark contract."""


def load_tasks(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_dir():
        raise ValidationError(
            f"{source}: tasks must be a directory containing graft/, refresh/, and clarify/"
        )
    tasks: list[dict[str, Any]] = []
    for track in sorted(TRACKS):
        track_dir = source / track
        if not track_dir.is_dir():
            continue
        for task_path in sorted(track_dir.glob("*.json")):
            task = json.loads(task_path.read_text())
            if not isinstance(task, dict):
                raise ValidationError(f"{task_path}: task file must contain one JSON object")
            if task.get("track") != track:
                raise ValidationError(
                    f"{task_path}: directory class {track!r} disagrees with "
                    f"task.track {task.get('track')!r}"
                )
            tasks.append(task)
    stray = [child for child in source.iterdir() if child.is_file() and child.suffix == ".json"]
    if stray:
        raise ValidationError(
            f"{source}: task JSON files must live under graft/, refresh/, or clarify/"
        )
    if not tasks:
        raise ValidationError(f"{source}: no task JSON files found under track directories")
    return tasks


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"{context}: missing {key!r}")
    return mapping[key]


def task_subtype(task: dict[str, Any]) -> str:
    """Derive the analysis subtype from the track-specific source of truth."""

    track = task["track"]
    if track == "graft":
        return str(task["graft"]["integration_type"])
    if track == "refresh":
        return str(task["refresh"]["policy"])
    slot = str(task["clarification"]["valid_slot"])
    return slot if slot in {"position", "sequence"} else "variant"


def acceptable_states_of(goal: dict[str, Any], context: str) -> list[dict[str, Any]]:
    states = goal.get("acceptable_states")
    if not isinstance(states, list) or not states:
        raise ValidationError(f"{context}: acceptable_states must be a non-empty list")
    for index, state in enumerate(states):
        state_context = f"{context}/acceptable_states/{index}"
        if not isinstance(state, dict) or set(state) != {
            "selected_entity_ids", "fields", "submission_completed"
        }:
            raise ValidationError(
                f"{state_context}: state must contain selected_entity_ids, fields, and "
                "submission_completed"
            )
        entity_ids = state["selected_entity_ids"]
        if not isinstance(entity_ids, list) or not entity_ids or any(
            not isinstance(entity_id, str) or not entity_id for entity_id in entity_ids
        ):
            raise ValidationError(f"{state_context}: selected_entity_ids must be non-empty strings")
        fields = state["fields"]
        if not isinstance(fields, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in fields.items()
        ):
            raise ValidationError(f"{state_context}: fields must map strings to strings")
        if state["submission_completed"] is not True:
            raise ValidationError(f"{state_context}: acceptable state must be submitted")
    return states


def goal_for_arm(task: dict[str, Any], arm: str) -> dict[str, Any]:
    if task["track"] == "graft" and arm == "a_only":
        return task["graft"]["initial_goal"]
    return task["goal"]


def selected_entity_sets_of(goal: dict[str, Any], context: str) -> list[list[str]]:
    return [
        list(state["selected_entity_ids"])
        for state in acceptable_states_of(goal, context)
    ]


def single_entity_target(task: dict[str, Any], arm: str) -> str:
    states = acceptable_states_of(goal_for_arm(task, arm), f"{task['id']}/{arm}")
    if len(states) != 1 or len(states[0]["selected_entity_ids"]) != 1:
        raise ValidationError(f"{task['id']}/{arm}: arm requires one unique entity target")
    return str(states[0]["selected_entity_ids"][0])


def _canonical(goal: dict[str, Any], context: str) -> frozenset[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]]:
    return frozenset(
        (
            tuple(sorted(state["selected_entity_ids"])),
            tuple(sorted(state["fields"].items())),
        )
        for state in acceptable_states_of(goal, context)
    )


def _validate_goal(goal: dict[str, Any], entity_ids: set[str], context: str) -> None:
    if goal.get("target_granularity") not in TARGET_GRANULARITIES:
        raise ValidationError(
            f"{context}: target_granularity must be one of {sorted(TARGET_GRANULARITIES)}"
        )
    for state in acceptable_states_of(goal, context):
        if not set(state["selected_entity_ids"]) <= entity_ids:
            raise ValidationError(f"{context}: acceptable state IDs must exist in the task site")


def _multiset_contains(container: list[str], required: list[str]) -> bool:
    return not (Counter(required) - Counter(container))


def _validate_graft_relation(task: dict[str, Any], initial_goal: dict[str, Any]) -> None:
    task_id = task["id"]
    integration_type = task["graft"]["integration_type"]
    initial_states = acceptable_states_of(initial_goal, f"{task_id}/graft/initial_goal")
    final_states = acceptable_states_of(task["goal"], f"{task_id}/goal")

    if integration_type == "C":
        refines = all(
            any(
                Counter(final["selected_entity_ids"]) == Counter(initial["selected_entity_ids"])
                and initial["fields"].items() <= final["fields"].items()
                for initial in initial_states
            )
            for final in final_states
        )
        became_stricter = (
            len(final_states) < len(initial_states)
            or any(bool(final["fields"]) for final in final_states)
        )
        if not refines or not became_stricter:
            raise ValidationError(
                f"{task_id}: C requires the update to narrow alternatives or add exact fields"
            )
        return

    if not all(
        any(
            len(final["selected_entity_ids"]) > len(initial["selected_entity_ids"])
            and _multiset_contains(
                final["selected_entity_ids"], initial["selected_entity_ids"]
            )
            for initial in initial_states
        )
        for final in final_states
    ):
        raise ValidationError(
            f"{task_id}: {integration_type} requires every updated goal to retain an "
            "initial entity and add an entity"
        )


def _validate_packets(task: dict[str, Any], root: Path | None) -> None:
    packets = _require(task, "visual_packets", task["id"])
    if not isinstance(packets, list):
        raise ValidationError(f"{task['id']}: visual_packets must be a list")
    actual_ids = {packet.get("id") for packet in packets if isinstance(packet, dict)}
    expected_ids = {"V0", "V1"} if task["track"] == "refresh" else {"V0"}
    if actual_ids != expected_ids or len(packets) != len(expected_ids):
        raise ValidationError(
            f"{task['id']}: visual packet ids {sorted(str(value) for value in actual_ids)} "
            f"invalid for {task['track']}"
        )
    for packet in packets:
        if "target_state" in packet:
            raise ValidationError(
                f"{task['id']}/{packet.get('id')}: target_state belongs in hidden ground truth"
            )
        frame_paths = packet.get("frame_paths", [])
        expected_count = 4 if task["track"] == "refresh" else 8
        if len(frame_paths) != expected_count:
            raise ValidationError(
                f"{task['id']}/{packet['id']}: expected {expected_count} frames, "
                f"got {len(frame_paths)}"
            )
        if root:
            for frame_path in frame_paths:
                try:
                    resolve_within(root, frame_path, context=f"{task['id']} frame")
                except (ValueError, FileNotFoundError) as exc:
                    raise ValidationError(str(exc)) from exc


def _validate_annotation_complete(task: dict[str, Any]) -> None:
    values = [task["instructions"], task.get("clarification", {}).get("canonical_answer", "")]
    if "[ANNOTATE" in json.dumps(values, ensure_ascii=False).upper():
        raise ValidationError(f"{task['id']}: task contains annotation placeholders")


def _validate_released_language(task: dict[str, Any]) -> None:
    released = json.dumps(task["instructions"], ensure_ascii=False).casefold()
    internal_phrases = ("frozen average rating", "frozen price")
    found = [phrase for phrase in internal_phrases if phrase in released]
    if found:
        raise ValidationError(
            f"{task['id']}: released instruction exposes benchmark-internal wording {found}"
        )


def _referenced_site_entity_ids(task: dict[str, Any]) -> set[str]:
    goals = [task["goal"]]
    if task["track"] == "graft":
        goals.append(task["graft"]["initial_goal"])
    entity_ids = {
        str(entity_id)
        for goal in goals
        for state in acceptable_states_of(goal, f"{task['id']}/site-assets")
        for entity_id in state["selected_entity_ids"]
    }
    if task["track"] == "refresh":
        entity_ids.add(str(task["refresh"]["initial_entity_id"]))
    elif task["track"] == "clarify":
        entity_ids.update(str(value) for value in task["clarification"]["candidate_ids"])
    return entity_ids


def _validate_site_assets(
    task: dict[str, Any], site_data: dict[str, Any], root: str | Path | None,
) -> None:
    if root is None:
        raise ValidationError(f"{task['id']}: asset validation requires the workspace root")
    referenced_ids = _referenced_site_entity_ids(task)
    entities_by_id = {
        str(entity["entity_id"]): entity
        for entity in site_data["entities"]
    }
    referenced_entities = [entities_by_id[entity_id] for entity_id in sorted(referenced_ids)]
    released_frames = (
        frame_path
        for packet in task["visual_packets"]
        for frame_path in packet["frame_paths"]
    )
    try:
        validate_referenced_site_assets(
            str(task["site"]),
            referenced_entities,
            root=root,
            released_frame_paths=released_frames,
        )
    except EnvironmentError as exc:
        raise ValidationError(f"{task['id']}: {exc}") from exc


def validate_task(
    task: dict[str, Any], site_data: dict[str, Any], *, root: str | Path | None = None,
    require_assets: bool = False,
) -> None:
    task_id = _require(task, "id", "task")
    if task.get("schema_version") != 2:
        raise ValidationError(f"{task_id}: task schema_version must be 2")
    redundant = sorted(REMOVED_TASK_FIELDS & set(task))
    if redundant:
        raise ValidationError(f"{task_id}: task contains removed fields {redundant}")
    track = _require(task, "track", task_id)
    if track not in TRACKS:
        raise ValidationError(f"{task_id}: unknown track {track!r}")
    track_field = track if track != "clarify" else "clarification"
    allowed_task_fields = {
        "schema_version", "id", "track", "site", "source", "instructions",
        "visual_packets", "goal", track_field,
    }
    unknown_task_fields = sorted(set(task) - allowed_task_fields)
    if unknown_task_fields:
        raise ValidationError(f"{task_id}: unsupported task fields {unknown_task_fields}")
    site = _require(task, "site", task_id)
    if site not in SITES or site_data.get("site") != site:
        raise ValidationError(f"{task_id}: invalid or mismatched site {site!r}")
    _require(task, "source", task_id)
    instructions = _require(task, "instructions", task_id)
    _validate_released_language(task)
    goal = _require(task, "goal", task_id)
    if require_assets:
        _validate_annotation_complete(task)
    _validate_packets(task, Path(root) if root and require_assets else None)
    entity_ids = site_entity_ids(site_data)
    _validate_goal(goal, entity_ids, f"{task_id}/goal")

    if track == "graft":
        graft = _require(task, "graft", task_id)
        if set(graft) != {"integration_type", "initial_goal"}:
            raise ValidationError(
                f"{task_id}: graft state is derived from initial_goal and goal only"
            )
        if graft.get("integration_type") not in {"C", "P", "R"}:
            raise ValidationError(f"{task_id}: invalid integration_type")
        initial_goal = _require(graft, "initial_goal", f"{task_id}/graft")
        _validate_goal(initial_goal, entity_ids, f"{task_id}/graft/initial_goal")
        if initial_goal["target_granularity"] != goal["target_granularity"]:
            raise ValidationError(f"{task_id}: initial and updated goals need one granularity")
        for key in ("a", "b", "bundled"):
            _require(instructions, key, task_id)
        _validate_graft_relation(task, initial_goal)
    elif track == "refresh":
        refresh = _require(task, "refresh", task_id)
        if set(refresh) != {"policy", "initial_entity_id"}:
            raise ValidationError(
                f"{task_id}: refresh must contain policy and initial_entity_id only"
            )
        if refresh.get("policy") not in {"retain", "replace"}:
            raise ValidationError(f"{task_id}: invalid refresh policy")
        if "initial_target_id" in refresh or "final_target_id" in refresh:
            raise ValidationError(f"{task_id}: refresh must use initial_entity_id only")
        initial_entity_id = refresh.get("initial_entity_id")
        if initial_entity_id not in entity_ids:
            raise ValidationError(f"{task_id}: initial_entity_id must exist in the task site")
        _require(instructions, "base", task_id)
        final_entity_id = single_entity_target(task, "stream")
        if (initial_entity_id != final_entity_id) != (refresh["policy"] == "replace"):
            raise ValidationError(f"{task_id}: refresh policy disagrees with its final target")
    else:
        clarification = _require(task, "clarification", task_id)
        if set(clarification) != {
            "candidate_ids", "valid_slot", "canonical_answer", "max_questions"
        }:
            raise ValidationError(f"{task_id}: clarification contains unsupported fields")
        candidates = clarification.get("candidate_ids", [])
        if len(candidates) != 2 or len(set(candidates)) != 2:
            raise ValidationError(f"{task_id}: clarification requires exactly 2 candidates")
        if not set(candidates) <= entity_ids:
            raise ValidationError(f"{task_id}: clarification candidates must exist in the task site")
        if clarification.get("valid_slot") not in CLARIFICATION_SLOTS:
            raise ValidationError(f"{task_id}: invalid clarification slot")
        if not str(clarification.get("canonical_answer", "")).strip():
            raise ValidationError(f"{task_id}: canonical_answer must be non-empty")
        if clarification.get("max_questions") != 1:
            raise ValidationError(f"{task_id}: pilot requires max_questions=1")
        for key in ("ambiguous", "oracle"):
            _require(instructions, key, task_id)
        ambiguous = str(instructions["ambiguous"]).lower()
        if any(phrase in ambiguous for phrase in CLARIFICATION_PERMISSION_PHRASES):
            raise ValidationError(
                f"{task_id}: ambiguous instruction must underspecify one intended target"
            )
        if single_entity_target(task, "oracle") not in candidates:
            raise ValidationError(f"{task_id}: expected target must be a clarification candidate")

    if require_assets:
        _validate_site_assets(task, site_data, root)


def validate_dataset(
    tasks: Iterable[dict[str, Any]], *, root: str | Path = ".",
    site_data_by_site: dict[str, dict[str, Any]] | None = None,
    require_assets: bool = False,
) -> dict[str, int]:
    materialized = list(tasks)
    loaded = dict(site_data_by_site or {})
    counts = {track: 0 for track in sorted(TRACKS)}
    seen: set[str] = set()
    for task in materialized:
        site = str(task.get("site", ""))
        if site not in loaded:
            try:
                loaded[site] = load_site_data(
                    root,
                    site,
                    require_assets=require_assets,
                    released_frame_paths=(
                        frame
                        for current in materialized if current.get("site") == site
                        for packet in current.get("visual_packets", [])
                        for frame in packet.get("frame_paths", [])
                    ),
                )
            except EnvironmentError as exc:
                raise ValidationError(str(exc)) from exc
        validate_task(task, loaded[site], root=root, require_assets=require_assets)
        if task["id"] in seen:
            raise ValidationError(f"duplicate task id: {task['id']}")
        seen.add(task["id"])
        counts[task["track"]] += 1
    _validate_unique_task_content(materialized, root=root, require_assets=require_assets)
    return counts


def _validate_unique_task_content(
    tasks: list[dict[str, Any]], *, root: str | Path, require_assets: bool
) -> None:
    instruction_owners: dict[str, tuple[str, str]] = {}
    frame_path_owners: dict[str, tuple[str, str]] = {}
    frame_hash_owners: dict[str, tuple[str, str]] = {}
    for task in tasks:
        task_id = str(task["id"])
        for role, value in task.get("instructions", {}).items():
            if not isinstance(value, str):
                continue
            normalized = " ".join(value.casefold().split())
            if normalized in instruction_owners:
                previous = instruction_owners[normalized]
                raise ValidationError(
                    f"duplicate instruction text for {previous[0]}/{previous[1]} "
                    f"and {task_id}/{role}"
                )
            instruction_owners[normalized] = (task_id, str(role))
        for packet in task.get("visual_packets", []):
            packet_id = str(packet.get("id", ""))
            for raw_path in packet.get("frame_paths", []):
                frame_path = str(raw_path)
                owner = (task_id, packet_id)
                if frame_path in frame_path_owners:
                    previous = frame_path_owners[frame_path]
                    raise ValidationError(
                        f"duplicate released frame for {previous[0]}/{previous[1]} "
                        f"and {task_id}/{packet_id}: {frame_path}"
                    )
                frame_path_owners[frame_path] = owner
                if require_assets:
                    resolved = resolve_within(root, frame_path, context=f"{task_id}/{packet_id} frame")
                    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                    if digest in frame_hash_owners:
                        previous = frame_hash_owners[digest]
                        raise ValidationError(
                            f"duplicate released frame bytes for {previous[0]}/{previous[1]} "
                            f"and {task_id}/{packet_id}"
                        )
                    frame_hash_owners[digest] = owner


def validate_pilot_distribution(tasks: Iterable[dict[str, Any]]) -> None:
    materialized = list(tasks)
    actual = {
        track: {
            subtype: sum(
                1 for task in materialized
                if task["track"] == track and task_subtype(task) == subtype
            )
            for subtype in expected
        }
        for track, expected in PILOT_DISTRIBUTION.items()
    }
    if actual != PILOT_DISTRIBUTION:
        raise ValidationError(
            f"pilot distribution mismatch: expected {PILOT_DISTRIBUTION}, got {actual}"
        )
