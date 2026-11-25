from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .environment import (
    SITES,
    EnvironmentError,
    entity_visual_sha256,
    load_site_data,
    site_entity_ids,
    validate_entity_visual_distinctness,
    validate_referenced_site_assets,
)
from .timeline import COMPOSITE_EVENTS, COMPOSITE_TRACK, Condition, TimelineError, composite_input
from .final_state import goal_entity_ids, goal_signature, normalized_entity_counts, unit_equivalences_of
from .visual_sampling import packet_asset_paths
from .frame_inputs import packed_image_size, validate_max_image_edge
from .workspace import read_json_strict, resolve_within

TRACK_ORDER = ("graft", "refresh", "clarify")
TRACKS = set(TRACK_ORDER)
EXPERIMENT_TRACKS = (*TRACK_ORDER, COMPOSITE_TRACK)
CLARIFICATION_SLOTS = {"position", "appearance", "brand", "variant", "sequence"}
SITE_ID_TOKEN = {
    "egomart": "mart",
    "egomedia": "media",
    "egomaps": "maps",
}
TASK_ID_PATTERN = re.compile(
    r"^(seed|candidate)-(mart|media|maps)-"
    r"(?:graft-(?:[cpr]|refine|relax|augment|replace|drop)|refresh-(?:replace|retain)|"
    r"clarify-(?:position|appearance|brand|variant|sequence))"
    r"-\d{2,}$"
)
CATALOG_LEAK_RUN = 4
CATALOG_LEAK_STOPWORDS = frozenset(
    """a an and are as at be by for from i in into is it its me my of on one or
    that the then this to up with you your""".split()
)
TARGET_GRANULARITIES = {"category", "brand", "variant", "exact_instance"}
GRAFT_TRANSITIONS = {"refine", "relax", "augment", "replace", "drop"}
REFRESH_TRANSITIONS = {"replace", "retain"}
CLARIFY_TRANSITION = "resolve"
CLARIFICATION_BASES = {"position", "sequence", "visible_attribute"}
DEIXIS_LEVELS = {1, 2, 3}
REMOVED_TASK_FIELDS = {
    "subtype",
    "conditions",
    "expectations",
    "catalog",
    "environment",
    "review",
    "review_status",
    "human_validation",
    "model_validation",
    "evaluation_contract",
    "annotation_notes",
}


class ValidationError(ValueError):
    """Raised when a task does not satisfy the benchmark contract."""


def load_tasks(path: str | Path, *, experimental: bool = False) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_dir():
        raise ValidationError(
            f"{source}: tasks must be a directory containing graft/, refresh/, and clarify/"
        )
    tasks: list[dict[str, Any]] = []
    for track in sorted(EXPERIMENT_TRACKS if experimental else TRACKS):
        track_dir = source / track
        if not track_dir.is_dir():
            continue
        for task_path in sorted(track_dir.glob("*.json")):
            task = read_json_strict(task_path)
            if not isinstance(task, dict):
                raise ValidationError(
                    f"{task_path}: task file must contain one JSON object"
                )
            if task.get("track") != track:
                raise ValidationError(
                    f"{task_path}: directory class {track!r} disagrees with "
                    f"task.track {task.get('track')!r}"
                )
            tasks.append(task)
    stray = [
        child
        for child in source.iterdir()
        if child.is_file() and child.suffix == ".json"
    ]
    if stray:
        raise ValidationError(
            f"{source}: task JSON files must live under graft/, refresh/, or clarify/"
        )
    if not tasks:
        raise ValidationError(
            f"{source}: no task JSON files found under track directories"
        )
    return tasks


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"{context}: missing {key!r}")
    return mapping[key]


def task_transition(task: Mapping[str, Any]) -> str:
    """Return the one semantic change class shared by tasks and reports."""

    track = task["track"]
    if track == "graft":
        return str(task["graft"]["transition"])
    if track == "refresh":
        return str(task["refresh"]["policy"])
    if track == COMPOSITE_TRACK:
        return COMPOSITE_TRACK
    return CLARIFY_TRANSITION


def clarification_basis(task: Mapping[str, Any]) -> str | None:
    """Group the evidence used to resolve a Clarify task."""

    if task.get("track") != "clarify":
        return None
    slot = str(task["clarification"]["valid_slot"])
    return slot if slot in {"position", "sequence"} else "visible_attribute"


def task_subtype(task: dict[str, Any]) -> str:
    """Compatibility name for the primary semantic transition."""

    return task_transition(task)


def validate_executable_task_id(task: dict[str, Any]) -> None:
    """Require canonical ids for production tasks; allow test-* fixture ids."""

    task_id = str(task.get("id") or "")
    if not task_id or task_id.startswith("test-"):
        return
    expected = canonical_task_id(task)
    if task_id != expected:
        raise ValidationError(
            f"{task_id}: executable id must be {expected!r} "
            f"(origin-site-track-subtype-serial)"
        )
    if TASK_ID_PATTERN.fullmatch(task_id) is None:
        raise ValidationError(
            f"{task_id}: executable id does not match TASK_ID_PATTERN"
        )


def canonical_task_id(task: dict[str, Any]) -> str:
    """Keep existing IDs stable; use the semantic transition for new IDs."""

    current = str(task.get("id") or "")
    if TASK_ID_PATTERN.fullmatch(current):
        return current
    if current.startswith("seed-"):
        origin = "seed"
    else:
        origin = "candidate"
    try:
        site = SITE_ID_TOKEN[str(task["site"])]
    except KeyError as exc:
        raise ValidationError(f"{current or '<missing>'}: invalid site") from exc
    track = str(task["track"])
    if track == "graft":
        subtype = task_transition(task)
    elif track == "refresh":
        subtype = str(task["refresh"]["policy"])
    else:
        subtype = str(task["clarification"]["valid_slot"])
    serial = re.search(r"(\d{2,})$", current)
    if serial is None:
        raise ValidationError(
            f"{current or '<missing>'}: task id must end in a serial of at least two digits"
        )
    return f"{origin}-{site}-{track}-{subtype}-{serial.group(1)}"


def acceptable_states_of(goal: dict[str, Any], context: str) -> list[dict[str, Any]]:
    states = goal.get("acceptable_states")
    if not isinstance(states, list) or not states:
        raise ValidationError(f"{context}: acceptable_states must be a non-empty list")
    for index, state in enumerate(states):
        state_context = f"{context}/acceptable_states/{index}"
        if (
            not isinstance(state, dict)
            or not {"selected_entity_ids", "fields"} <= set(state)
            or set(state) - {"selected_entity_ids", "fields", "submission_completed"}
        ):
            raise ValidationError(
                f"{state_context}: state must contain selected_entity_ids, fields, and "
                "optional historical submission_completed"
            )
        entity_ids = state["selected_entity_ids"]
        if not isinstance(entity_ids, list) or any(
            not isinstance(entity_id, str) or not entity_id for entity_id in entity_ids
        ):
            raise ValidationError(
                f"{state_context}: selected_entity_ids must be non-empty strings"
            )
        fields = state["fields"]
        if not isinstance(fields, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in fields.items()
        ):
            raise ValidationError(
                f"{state_context}: fields must map strings to strings"
            )
        if "submission_completed" in state and not isinstance(
            state["submission_completed"], bool
        ):
            raise ValidationError(
                f"{state_context}: historical submission flag must be boolean"
            )
    canonical = {
        (
            tuple(sorted(state["selected_entity_ids"])),
            tuple(sorted(state["fields"].items())),
        )
        for state in states
    }
    if len(canonical) != len(states):
        raise ValidationError(
            f"{context}: acceptable_states contains duplicate alternatives"
        )
    return states


def goal_for_condition(
    task: dict[str, Any],
    condition: Condition,
) -> dict[str, Any]:
    """Return the scoring goal derived from the active experiment condition."""

    return task["goal"]


def selected_entity_sets_of(goal: dict[str, Any], context: str) -> list[list[str]]:
    return [
        list(state["selected_entity_ids"])
        for state in acceptable_states_of(goal, context)
    ]


def single_entity_target(task: dict[str, Any], condition: Condition) -> str:
    return _single_entity_target(
        goal_for_condition(task, condition), f"{task['id']}/{condition.name}"
    )


def _single_entity_target(goal: dict[str, Any], context: str) -> str:
    states = acceptable_states_of(goal, context)
    if len(states) != 1 or len(states[0]["selected_entity_ids"]) != 1 or len(goal_entity_ids(goal)) != 1:
        raise ValidationError(f"{context}: goal requires one unique entity target")
    return str(states[0]["selected_entity_ids"][0])


def canonical_goal(goal: dict[str, Any], context: str) -> frozenset[tuple]:
    acceptable_states_of(goal, context)
    return goal_signature(goal)


def _validate_goal(goal: dict[str, Any], entity_ids: set[str], context: str) -> None:
    granularity = goal.get("target_granularity")
    if granularity not in TARGET_GRANULARITIES:
        raise ValidationError(
            f"{context}: target_granularity must be one of {sorted(TARGET_GRANULARITIES)}"
        )
    states = acceptable_states_of(goal, context)
    try:
        units = unit_equivalences_of(goal)
    except ValueError as exc:
        raise ValidationError(f"{context}: {exc}") from exc
    mapped_ids = {entity_id for packages in units.values() for entity_id in packages}
    if not mapped_ids <= entity_ids:
        raise ValidationError(f"{context}: unit package IDs must exist in the task site")
    used_units = {
        name for state in states
        for (kind, name) in normalized_entity_counts(goal, state["selected_entity_ids"])
        if kind == "unit"
    }
    if set(units) != used_units:
        raise ValidationError(f"{context}: each unit class needs an executable witness")
    if granularity == "exact_instance" and units:
        raise ValidationError(f"{context}: exact_instance cannot declare package substitutions")
    if granularity == "exact_instance" and len(states) != 1:
        raise ValidationError(
            f"{context}: exact_instance requires one exact acceptable state"
        )
    field_sets = {tuple(sorted(state["fields"].items())) for state in states}
    if len(field_sets) != 1:
        raise ValidationError(
            f"{context}: equivalent acceptable states must keep the same fields"
        )
    for state in states:
        if not set(state["selected_entity_ids"]) <= entity_ids:
            raise ValidationError(
                f"{context}: acceptable state IDs must exist in the task site"
            )


def _multiset_contains(container: list[str], required: list[str]) -> bool:
    return not (Counter(required) - Counter(container))


def _withdraws_a_commitment(
    initial_states: list[dict[str, Any]], final_states: list[dict[str, Any]]
) -> bool:
    return all(
        any(
            _multiset_contains(
                initial["selected_entity_ids"], final["selected_entity_ids"]
            )
            and len(final["selected_entity_ids"]) < len(initial["selected_entity_ids"])
            for initial in initial_states
        )
        for final in final_states
    )


def _augments_a_commitment(
    initial_states: list[dict[str, Any]], final_states: list[dict[str, Any]]
) -> bool:
    return all(
        any(
            len(final["selected_entity_ids"]) > len(initial["selected_entity_ids"])
            and _multiset_contains(
                final["selected_entity_ids"], initial["selected_entity_ids"]
            )
            for initial in initial_states
        )
        for final in final_states
    )


def _keeps_action_shape(
    initial_states: list[dict[str, Any]], final_states: list[dict[str, Any]]
) -> bool:
    initial_sizes = {len(state["selected_entity_ids"]) for state in initial_states}
    return all(
        len(state["selected_entity_ids"]) in initial_sizes for state in final_states
    )


def _validate_graft_relation(
    task: dict[str, Any], initial_goal: dict[str, Any]
) -> None:
    task_id = task["id"]
    transition = task["graft"]["transition"]
    initial_states = acceptable_states_of(initial_goal, f"{task_id}/graft/initial_goal")
    final_states = acceptable_states_of(task["goal"], f"{task_id}/goal")
    if canonical_goal(task["goal"], f"{task_id}/goal") == canonical_goal(
        initial_goal, f"{task_id}/graft/initial_goal"
    ):
        raise ValidationError(f"{task_id}: Graft transition must change the final goal")

    if transition == "relax":
        initial = canonical_goal(initial_goal, f"{task_id}/graft/initial_goal")
        final = canonical_goal(task["goal"], f"{task_id}/goal")
        if not initial < final or not _keeps_action_shape(initial_states, final_states):
            raise ValidationError(
                f"{task_id}: relax must preserve every initial state and expand "
                "the acceptable states without adding commitments"
            )
        return

    if transition == "augment":
        if not _augments_a_commitment(initial_states, final_states):
            raise ValidationError(
                f"{task_id}: augment must preserve a complete initial state and add a goal"
            )
        return

    if transition == "drop":
        if not _withdraws_a_commitment(initial_states, final_states):
            raise ValidationError(
                f"{task_id}: drop must remove a previously required goal without adding one"
            )
        return

    if transition == "replace":
        # A revised quantity can keep a SKU and change its cart quantity.
        return

    if not _keeps_action_shape(initial_states, final_states):
        raise ValidationError(
            f"{task_id}: refine must narrow or specialize the existing goal without "
            "changing the number of required commitments"
        )


def _validate_truth_block(task: dict[str, Any], entity_ids: set[str]) -> None:
    if "truth" not in task:
        return
    task_id = task["id"]
    truth = task["truth"]
    allowed = {
        "tempting_referent_ids",
        "site_scope_entity_ids",
        "initial_acceptable_states",
    }
    if not isinstance(truth, dict) or not set(truth) <= allowed:
        raise ValidationError(f"{task_id}: truth may only declare {sorted(allowed)}")
    tempting = truth.get("tempting_referent_ids", [])
    if not isinstance(tempting, list) or any(
        not isinstance(item, str) or item not in entity_ids for item in tempting
    ):
        raise ValidationError(
            f"{task_id}: tempting_referent_ids must be site entity IDs"
        )
    final_ids = set(goal_entity_ids(task["goal"]))
    initial_ids: set[str] = set()
    if "initial_acceptable_states" in truth:
        if task["track"] != "refresh":
            raise ValidationError(
                f"{task_id}: initial_acceptable_states requires Refresh"
            )
        initial_goal = {
            "target_granularity": task["goal"]["target_granularity"],
            "acceptable_states": truth["initial_acceptable_states"],
        }
        _validate_goal(
            initial_goal, entity_ids, f"{task_id}/truth/initial_acceptable_states"
        )
        initial_states = acceptable_states_of(initial_goal, f"{task_id}/truth")
        if any(len(state["selected_entity_ids"]) != 1 for state in initial_states):
            raise ValidationError(
                f"{task_id}: Refresh initial states must each select one entity"
            )
        initial_ids = {str(state["selected_entity_ids"][0]) for state in initial_states}
        if task["refresh"]["initial_entity_id"] not in initial_ids:
            raise ValidationError(
                f"{task_id}: initial representative is absent from initial states"
            )
        retained = task["refresh"]["policy"] == "retain"
        if (retained and initial_ids != final_ids) or (
            not retained and initial_ids == final_ids
        ):
            raise ValidationError(
                f"{task_id}: Refresh policy disagrees with its initial states"
            )
    if set(tempting) & final_ids:
        raise ValidationError(
            f"{task_id}: tempting_referent_ids cannot be final targets"
        )
    scope = truth.get("site_scope_entity_ids", [])
    if (
        not isinstance(scope, list)
        or len(scope) != len(set(scope))
        or any(not isinstance(item, str) or item not in entity_ids for item in scope)
    ):
        raise ValidationError(
            f"{task_id}: site_scope_entity_ids must be unique site entity IDs"
        )
    if scope:
        semantic_ids = set(final_ids) | set(tempting) | initial_ids
        if task["track"] == "graft":
            semantic_ids.update(goal_entity_ids(task["graft"]["initial_goal"]))
        elif task["track"] == "refresh":
            semantic_ids.add(str(task["refresh"]["initial_entity_id"]))
        elif "clarification" in task:
            semantic_ids.update(
                str(item) for item in task["clarification"]["candidate_ids"]
            )
        if not semantic_ids <= set(scope):
            raise ValidationError(
                f"{task_id}: site_scope_entity_ids must include every task-referenced entity"
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
            for frame_path in packet_asset_paths(packet):
                try:
                    resolve_within(root, frame_path, context=f"{task['id']} frame")
                except (ValueError, FileNotFoundError) as exc:
                    raise ValidationError(str(exc)) from exc
    _validate_visual_session(task, root)


def _validate_visual_session(task: dict[str, Any], root: Path | None) -> None:
    if "visual_session" not in task:
        return
    session = task["visual_session"]
    if isinstance(session, dict):
        try:
            validate_max_image_edge(session.get("wearer_max_image_edge"))
            if "wearer_frames_per_image" in session:
                packed_image_size(1, 1, session["wearer_frames_per_image"])
        except ValueError as error:
            raise ValidationError(f"{task['id']}: {error}") from error
    pool = session.get("frequency_samples") if isinstance(session, dict) else None
    timestamps = pool.get("timestamps_seconds") if isinstance(pool, dict) else None
    paths = pool.get("frame_paths") if isinstance(pool, dict) else None
    window = pool.get("window_seconds") if isinstance(pool, dict) else None
    if (
        not isinstance(pool, dict)
        or not isinstance(timestamps, list)
        or not isinstance(paths, list)
        or not timestamps
        or len(timestamps) != len(paths)
        or not isinstance(window, list)
        or len(window) != 2
    ):
        raise ValidationError(
            f"{task['id']}: visual_session.frequency_samples needs a window and "
            "aligned timestamps and frame paths"
        )
    if root:
        for frame_path in paths:
            try:
                resolve_within(root, frame_path, context=f"{task['id']} session frame")
            except (ValueError, FileNotFoundError) as exc:
                raise ValidationError(str(exc)) from exc


def _validate_annotation_complete(task: dict[str, Any]) -> None:
    values = [
        task["instructions"],
        task.get(COMPOSITE_EVENTS, []),
        task.get("clarification", {}).get("canonical_answer", ""),
    ]
    if "[ANNOTATE" in json.dumps(values, ensure_ascii=False).upper():
        raise ValidationError(f"{task['id']}: task contains annotation placeholders")


def _released_instruction_text(task: Mapping[str, Any]) -> str:
    instructions = json.dumps(task["instructions"], ensure_ascii=False)
    if COMPOSITE_EVENTS in task:
        instructions += json.dumps(task[COMPOSITE_EVENTS], ensure_ascii=False)
    return instructions


def _validate_released_language(task: dict[str, Any]) -> None:
    released = _released_instruction_text(task).casefold()
    internal_phrases = ("frozen average rating", "frozen price")
    found = [phrase for phrase in internal_phrases if phrase in released]
    if found:
        raise ValidationError(
            f"{task['id']}: released instruction exposes benchmark-internal wording {found}"
        )


def _catalog_tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", str(text).casefold())
    return [word for word in words if word not in CATALOG_LEAK_STOPWORDS]


def _token_runs(tokens: list[str]) -> set[tuple[str, ...]]:
    size = CATALOG_LEAK_RUN
    return {
        tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1)
    }


def _quotes_catalog_title(released: list[str], title: list[str]) -> bool:
    """Distinguish copied listing text from ordinary category language.

    Three-word phrases such as ``black frying pan`` and short titles such as
    ``Green Soup`` are legitimate task language. A listing therefore needs a
    four-token verbatim run to trigger a review warning.
    """

    if not title:
        return False
    if len(title) <= 3:
        return False
    return bool(_token_runs(title) & _token_runs(released))


def catalog_title_warnings(
    task: dict[str, Any], site_data: dict[str, Any]
) -> list[str]:
    """Flag copied listing phrases for semantic review, not automatic rejection."""
    released = _catalog_tokens(_released_instruction_text(task))
    referenced_ids = set(semantic_entity_ids(task))
    return [
        f"{task['id']}: released instruction quotes the catalog entry for "
        f"{entity['entity_id']}; review whether the phrase reveals the answer"
        for entity in site_data["entities"]
        if str(entity["entity_id"]) in referenced_ids
        and _quotes_catalog_title(
            released, _catalog_tokens(entity.get("title") or entity.get("name") or "")
        )
    ]


def semantic_entity_ids(task: Mapping[str, Any]) -> list[str]:
    """Entity IDs that define the task answer, transition, or physical ambiguity.

    Order: final goal entities, then Graft initial-goal entities, then the Refresh
    initial entity or the Clarify candidates, then any tempting referents used by
    the negative evaluator.
    """

    groups: list[list[str]] = [sorted(goal_entity_ids(task["goal"]))]
    if task["track"] == "graft":
        groups.append(sorted(goal_entity_ids(task["graft"]["initial_goal"])))
    elif task["track"] == "refresh":
        groups.append([str(task["refresh"]["initial_entity_id"])])
        groups.extend(
            [str(entity_id) for entity_id in state["selected_entity_ids"]]
            for state in task.get("truth", {}).get("initial_acceptable_states", [])
        )
    elif "clarification" in task:
        groups.append([str(value) for value in task["clarification"]["candidate_ids"]])
    groups.append(
        [str(value) for value in task.get("truth", {}).get("tempting_referent_ids", [])]
    )
    return list(dict.fromkeys(entity_id for group in groups for entity_id in group))


def semantic_visual_representative_ids(
    task: Mapping[str, Any],
    entities_by_id: Mapping[str, dict[str, Any]],
    *,
    root: str | Path,
) -> list[str]:
    """Collapse same-image entities only when interchangeable in every goal phase."""

    goal_phases = [task["goal"]]
    if task["track"] == "graft":
        goal_phases.append(task["graft"]["initial_goal"])
    elif task["track"] == "refresh" and "initial_acceptable_states" in task.get("truth", {}):
        goal_phases.append({"acceptable_states": task["truth"]["initial_acceptable_states"]})
    seen_visual_contexts: set[tuple] = set()
    representatives: list[str] = []
    for entity_id in semantic_entity_ids(task):
        entity = entities_by_id.get(entity_id)
        contexts = _entity_goal_contexts(entity_id, goal_phases)
        if any(contexts) and entity is not None:
            digest = entity_visual_sha256(
                str(task["site"]),
                entity,
                root=root,
            )
            if digest is not None:
                key = (digest, contexts)
                if key in seen_visual_contexts:
                    continue
                seen_visual_contexts.add(key)
        representatives.append(entity_id)
    return representatives


def _entity_goal_contexts(
    entity_id: str, goal_phases: list[dict[str, Any]],
) -> tuple[frozenset[str], ...]:
    """Keep visually identical packages distinct when their unit contribution differs."""
    contexts = []
    for goal in goal_phases:
        phase = set()
        contribution = normalized_entity_counts(goal, [entity_id])
        for state in goal["acceptable_states"]:
            witness = {**goal, "acceptable_states": [state]}
            if entity_id not in goal_entity_ids(witness):
                continue
            remaining = normalized_entity_counts(goal, state["selected_entity_ids"]) - contribution
            phase.add(repr((sorted(remaining.items()), sorted(state["fields"].items()))))
        contexts.append(frozenset(phase))
    return tuple(contexts)


def referenced_entity_ids(task: Mapping[str, Any]) -> list[str]:
    """Semantic entity IDs plus every public result in the task search window."""

    return list(
        dict.fromkeys(
            [
                *semantic_entity_ids(task),
                *(
                    str(value)
                    for value in task.get("truth", {}).get("site_scope_entity_ids", [])
                ),
            ]
        )
    )


def _referenced_site_entity_ids(task: dict[str, Any]) -> set[str]:
    return set(referenced_entity_ids(task))


def _validate_site_assets(
    task: dict[str, Any],
    site_data: dict[str, Any],
    root: str | Path | None,
) -> None:
    if root is None:
        raise ValidationError(
            f"{task['id']}: asset validation requires the workspace root"
        )
    referenced_ids = _referenced_site_entity_ids(task)
    entities_by_id = {
        str(entity["entity_id"]): entity for entity in site_data["entities"]
    }
    referenced_entities = [
        entities_by_id[entity_id] for entity_id in sorted(referenced_ids)
    ]
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
    task: dict[str, Any],
    site_data: dict[str, Any],
    *,
    root: str | Path | None = None,
    require_assets: bool = False,
) -> None:
    task_id = _require(task, "id", "task")
    if task.get("schema_version") != 2:
        raise ValidationError(f"{task_id}: task schema_version must be 2")
    redundant = sorted(REMOVED_TASK_FIELDS & set(task))
    if redundant:
        raise ValidationError(f"{task_id}: task contains removed fields {redundant}")
    track = _require(task, "track", task_id)
    if track not in EXPERIMENT_TRACKS:
        raise ValidationError(f"{task_id}: unknown track {track!r}")
    track_field = track if track != "clarify" else "clarification"
    allowed_task_fields = {
        "schema_version",
        "id",
        "track",
        "site",
        "source",
        "instructions",
        "visual_packets",
        "goal",
        track_field,
        "truth",
        "visual_session",
    }
    if track == COMPOSITE_TRACK:
        allowed_task_fields = allowed_task_fields - {COMPOSITE_TRACK} | {
            "clarification", COMPOSITE_EVENTS
        }
    unknown_task_fields = sorted(set(task) - allowed_task_fields)
    if unknown_task_fields:
        raise ValidationError(
            f"{task_id}: unsupported task fields {unknown_task_fields}"
        )
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
    _validate_truth_block(task, entity_ids)

    if track == "graft":
        graft = _require(task, "graft", task_id)
        if set(graft) != {"transition", "initial_goal"}:
            raise ValidationError(
                f"{task_id}: graft must contain transition and initial_goal only"
            )
        if graft.get("transition") not in GRAFT_TRANSITIONS:
            raise ValidationError(f"{task_id}: invalid Graft transition")
        initial_goal = _require(graft, "initial_goal", f"{task_id}/graft")
        _validate_goal(initial_goal, entity_ids, f"{task_id}/graft/initial_goal")
        if initial_goal["target_granularity"] != goal["target_granularity"]:
            raise ValidationError(
                f"{task_id}: initial and updated goals need one granularity"
            )
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
            raise ValidationError(
                f"{task_id}: initial_entity_id must exist in the task site"
            )
        _require(instructions, "base", task_id)
        final_states = acceptable_states_of(goal, f"{task_id}/goal")
        if any(len(state["selected_entity_ids"]) != 1 for state in final_states):
            raise ValidationError(
                f"{task_id}: refresh acceptable states must each select one entity"
            )
    elif track == "clarify" or "clarification" in task:
        clarification = _require(task, "clarification", task_id)
        required_clarification = {
            "candidate_ids",
            "valid_slot",
            "canonical_answer",
            "max_questions",
        }
        if not required_clarification <= set(clarification) or not set(
            clarification
        ) <= required_clarification | {"deixis_level"}:
            raise ValidationError(
                f"{task_id}: clarification contains unsupported fields"
            )
        if (
            "deixis_level" in clarification
            and clarification["deixis_level"] not in DEIXIS_LEVELS
        ):
            raise ValidationError(f"{task_id}: deixis_level must be 1, 2, or 3")
        candidates = clarification.get("candidate_ids", [])
        if len(candidates) != 2 or len(set(candidates)) != 2:
            raise ValidationError(
                f"{task_id}: clarification requires exactly 2 candidates"
            )
        if not set(candidates) <= entity_ids:
            raise ValidationError(
                f"{task_id}: clarification candidates must exist in the task site"
            )
        if clarification.get("valid_slot") not in CLARIFICATION_SLOTS:
            raise ValidationError(f"{task_id}: invalid clarification slot")
        if not str(clarification.get("canonical_answer", "")).strip():
            raise ValidationError(f"{task_id}: canonical_answer must be non-empty")
        if clarification.get("max_questions") != 1:
            raise ValidationError(f"{task_id}: pilot requires max_questions=1")
        final_ids = set(goal_entity_ids(goal))
        if track == "clarify":
            for key in ("ambiguous", "oracle"):
                _require(instructions, key, task_id)
            final_states = acceptable_states_of(goal, f"{task_id}/goal")
            if any(len(state["selected_entity_ids"]) != 1 for state in final_states):
                raise ValidationError(
                    f"{task_id}: clarification acceptable states must each select one entity"
                )
            final_ids = {str(state["selected_entity_ids"][0]) for state in final_states}
        if track == "clarify" and len(final_ids & set(candidates)) != 1:
            raise ValidationError(
                f"{task_id}: exactly one canonical answer must be a clarification candidate"
            )

    if track == COMPOSITE_TRACK:
        _validate_composite_input(task)

    if require_assets:
        _validate_site_assets(task, site_data, root)
        assert root is not None  # _validate_site_assets rejects a missing root
        entities_by_id = {
            str(entity["entity_id"]): entity for entity in site_data["entities"]
        }
        # Real result windows may legitimately contain variants that reuse a
        # title or product image. Visual distinctness is required only for the
        # entities that define the answer, transition, or physical ambiguity.
        representative_ids = semantic_visual_representative_ids(
            task,
            entities_by_id,
            root=root,
        )
        referenced_entities = [
            entities_by_id[entity_id]
            for entity_id in representative_ids
            if entity_id in entities_by_id
        ]
        try:
            validate_entity_visual_distinctness(
                str(task["site"]),
                referenced_entities,
                root=root,
                context=task_id,
            )
        except EnvironmentError as exc:
            raise ValidationError(str(exc)) from exc


def _validate_composite_input(task: dict[str, Any]) -> None:
    task_id = task["id"]
    session = _require(task, "visual_session", task_id)
    try:
        composite_input(task)
    except TimelineError as error:
        raise ValidationError(str(error)) from error
    pool = session["frequency_samples"]
    if (
        session.get("skipped_seconds") is not None
        or len(pool.get("sources", [])) != 1
        or task["visual_packets"][0]["frame_paths"] != pool["frame_paths"][:8]
    ):
        raise ValidationError(f"{task_id}: composite requires one continuous source and its real opening")


def validate_dataset(
    tasks: Iterable[dict[str, Any]],
    *,
    root: str | Path = ".",
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
                        for current in materialized
                        if current.get("site") == site
                        for packet in current.get("visual_packets", [])
                        for frame in packet.get("frame_paths", [])
                    ),
                )
            except EnvironmentError as exc:
                raise ValidationError(str(exc)) from exc
        validate_executable_task_id(task)
        validate_task(task, loaded[site], root=root, require_assets=require_assets)
        if task["id"] in seen:
            raise ValidationError(f"duplicate task id: {task['id']}")
        seen.add(task["id"])
        counts[task["track"]] += 1
    _validate_unique_task_content(
        materialized, root=root, require_assets=require_assets
    )
    if require_assets:
        _validate_referenced_map_photo_independence(materialized, root=root)
    return counts


def _validate_referenced_map_photo_independence(
    tasks: list[dict[str, Any]],
    *,
    root: str | Path,
) -> None:
    """Require CRS-referenced EgoMaps entities to use pairwise-distinct photos."""

    map_tasks = [task for task in tasks if task.get("site") == "egomaps"]
    if not map_tasks:
        return
    entities_by_id = {
        str(entity["entity_id"]): entity
        for entity in load_site_data(root, "egomaps")["entities"]
    }
    digest_owners: dict[str, str] = {}
    for task in map_tasks:
        for entity_id in sorted(_referenced_site_entity_ids(task)):
            entity = entities_by_id.get(entity_id)
            if entity is None:
                continue
            digest = entity_visual_sha256("egomaps", entity, root=root)
            if digest is None:
                continue
            previous = digest_owners.get(digest)
            if previous is not None and previous != entity_id:
                raise ValidationError(
                    f"referenced map entities {previous} and {entity_id} "
                    f"share photo bytes {digest[:12]}"
                )
            digest_owners[digest] = entity_id


def _validate_unique_task_content(
    tasks: list[dict[str, Any]], *, root: str | Path, require_assets: bool
) -> None:
    frame_path_owners: dict[str, tuple[str, str]] = {}
    frame_hash_owners: dict[str, tuple[str, str]] = {}
    for task in tasks:
        task_id = str(task["id"])
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
                    resolved = resolve_within(
                        root, frame_path, context=f"{task_id}/{packet_id} frame"
                    )
                    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                    # Same pixels may seed another commitment surface on a different site.
                    # Reject only when the duplicate stays on the same site.
                    site = str(task.get("site") or "")
                    hash_key = f"{site}:{digest}"
                    if hash_key in frame_hash_owners:
                        previous = frame_hash_owners[hash_key]
                        raise ValidationError(
                            f"duplicate released frame bytes for {previous[0]}/{previous[1]} "
                            f"and {task_id}/{packet_id}"
                        )
                    frame_hash_owners[hash_key] = owner
