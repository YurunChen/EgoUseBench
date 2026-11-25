"""Compile reviewed construction roles into executable tasks and site products."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections import Counter
from collections.abc import Mapping
from itertools import product
from functools import cached_property
from pathlib import Path
from typing import Any

from . import decision
from .decision import DecisionCurve, DecisionError
from .environment import load_site_data
from .lineage import (
    LineageError,
    build_source_identity,
    discover_frame_provenance,
    summarize_source_recordings,
    validate_independent_recordings,
)
from .neighborhood import NeighborhoodError, validate_search_neighborhood
from .run_ledger import task_local_content
from .schema import (
    CLARIFICATION_SLOTS,
    GRAFT_TRANSITIONS,
    TARGET_GRANULARITIES,
    TRACK_ORDER,
    load_tasks,
    validate_dataset,
    validate_task,
)
from .selection import SelectionError, load_selection, task_ids as selected_task_ids
from .sources import SourceError, screened_records, load_dossier, find_record
from .truth import Truth
from .final_state import evaluate_final_state, goal_entity_ids, unit_equivalences_of
from .workspace import write_json_atomic


class ConstructionError(ValueError):
    """Raised when a construction spec cannot be compiled without guessing."""


CLARIFICATION_SLOTS_BY_SUBTYPE = {
    "position": {"position"},
    "sequence": {"sequence"},
    "variant": {"appearance", "brand", "variant"},
}
EVIDENCE_TYPES = {
    "procedure_stage",
    "spec_reveal",
    "object_swap",
    "in_view_deixis",
    "record_identity",
    "place_approach",
}
COMMITMENT_SURFACES = {"cart", "media_save", "document_fields", "place_identity"}


def _acceptable_states(
    options: list[list[str]],
    fields: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "selected_entity_ids": entity_ids,
            "fields": dict(fields or {}),
        }
        for entity_ids in options
    ]


def _compile_unit_equivalences(blueprint: Mapping[str, Any], goal: dict[str, Any]) -> None:
    groups = unit_equivalences_of(blueprint)
    witness_ids = {
        entity_id for state in goal["acceptable_states"]
        for entity_id in state["selected_entity_ids"]
    }
    used = {unit: dict(packages) for unit, packages in groups.items() if witness_ids & packages.keys()}
    if used:
        goal["unit_equivalences"] = used


def compile_quantity_task(
    task: Mapping[str, Any], blueprint: Mapping[str, Any], level: str,
) -> dict[str, Any]:
    """Compile reviewed Scope quantity material without altering its visual lineage."""
    compiled = deepcopy(dict(task))
    compiled["instructions"] = deepcopy(blueprint["instructions"][level])
    compiled["graft"]["initial_goal"] = deepcopy(blueprint["initial_goal"])
    compiled["goal"] = deepcopy(blueprint["final_goals"][level])
    for goal in [compiled["graft"]["initial_goal"], compiled["goal"]]:
        _compile_unit_equivalences(blueprint, goal)
    return compiled


def _goal_fields(blueprint: Mapping[str, Any], key: str) -> dict[str, str]:
    value = blueprint.get(key, {})
    if not isinstance(value, dict) or any(
        not isinstance(field, str)
        or not field
        or not isinstance(answer, str)
        or not answer
        for field, answer in value.items()
    ):
        raise ConstructionError(
            f"{key} must map non-empty strings to non-empty strings"
        )
    return dict(value)


def _nonempty_strings(value: Any, keys: set[str]) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == keys
        and all(isinstance(value[key], str) and value[key].strip() for key in keys)
    )


def _decision_curve(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    site_data: Mapping[str, Any],
    task_id: str,
) -> DecisionCurve:
    """Read the survival curve, and check that it ends where the goal says it does."""

    declared = (record.get("construct") or {})["decision"]
    entity_ids = {str(entity["entity_id"]) for entity in site_data["entities"]}
    try:
        curve = decision.parse(
            declared,
            decision.frame_ids(task["visual_packets"]),
            entity_ids,
            task_id,
            task=task,
        )
    except DecisionError as exc:
        raise ConstructionError(str(exc)) from exc

    if curve.settled_at is None:
        raise ConstructionError(
            f"{task_id}: the curve never settles on one candidate, so the task has no "
            f"answer; add the {decision.UTTERANCE_ROW!r} row if what the user says resolves it"
        )
    if task["track"] == "clarify":
        if curve.settled_by_frames:
            raise ConstructionError(
                f"{task_id}: the frames settle the referent at {curve.settled_at}, so "
                f"nothing is left to ask; this is not a Clarify task"
            )
        candidates = set(task["clarification"]["candidate_ids"])
        alive_at_last_frame = {
            entity_id
            for observation in curve.observations
            if observation.frame != decision.UTTERANCE_ROW
            for entity_id in curve.entity_ids(observation.live)
        } & set(curve.entity_ids(curve.candidates))
        if candidates != alive_at_last_frame:
            raise ConstructionError(
                f"{task_id}: the frames leave {sorted(alive_at_last_frame)} in play, but "
                f"clarification offers {sorted(candidates)}"
            )
    elif task["track"] == "refresh" and not curve.settled_by_frames:
        raise ConstructionError(
            f"{task_id}: only what the user says settles this referent, so the later "
            f"packet is not doing the work a Refresh task claims it does"
        )
    if not all(
        evaluate_final_state(task["goal"], {
            "selected_entity_ids": list(candidate.selected_entity_ids),
            "fields": dict(candidate.fields) if candidate.fields is not None else state["fields"],
        })
        for candidate in curve.goal_states(curve.final_live)
        for state in task["goal"]["acceptable_states"][:1]
    ):
        raise ConstructionError(
            f"{task_id}: the curve settles on {curve.goal_states(curve.final_live)}, which "
            "the goal does not accept (expected a complete acceptable goal)"
        )
    return curve


def _validate_blueprint(record: dict[str, Any]) -> dict[str, Any]:
    task_id = str(record.get("task_id", ""))
    track = record.get("family")
    subtype = record.get("subtype")
    construct = record.get("construct")
    if not isinstance(construct, dict):
        raise ConstructionError(f"{task_id}: construct contract is missing")
    blueprint = construct.get("task_blueprint")
    if not isinstance(blueprint, dict):
        raise ConstructionError(f"{task_id}: task_blueprint is missing")
    if blueprint.get("target_granularity") not in TARGET_GRANULARITIES:
        raise ConstructionError(f"{task_id}: target_granularity is invalid")

    if "decision" not in construct:
        raise ConstructionError(
            f"{task_id}: no construct.decision. A task claims that its evidence supports "
            f"more than one candidate until it does not, and that claim is the curve; "
            f"without it there is nothing to compile."
        )
    stale = [
        name
        for name in (
            "release_anchor_seconds",
            "tempting_role",
            "deixis_level",
            "neighbor_role_ids",
            "integration_type",
            "update_kind",
        )
        if name in construct
    ]
    if stale:
        raise ConstructionError(
            f"{task_id}: {stale} are read off construct.decision and the goal pair; remove them"
        )

    evidence_type = construct.get("evidence_type")
    commitment_surface = construct.get("commitment_surface")
    requires_axes = evidence_type is not None or commitment_surface is not None
    if requires_axes:
        if evidence_type not in EVIDENCE_TYPES:
            raise ConstructionError(
                f"{task_id}: construct.evidence_type must be one of {sorted(EVIDENCE_TYPES)}"
            )
        if commitment_surface not in COMMITMENT_SURFACES:
            raise ConstructionError(
                f"{task_id}: construct.commitment_surface must be one of "
                f"{sorted(COMMITMENT_SURFACES)}"
            )

    if track == "graft":
        if subtype not in GRAFT_TRANSITIONS:
            raise ConstructionError(
                f"{task_id}: Graft subtype must be one of {sorted(GRAFT_TRANSITIONS)}"
            )
        if "b_essential_witness" in blueprint:
            raise ConstructionError(
                f"{task_id}: Graft blueprint contains obsolete field 'b_essential_witness'"
            )
        if not _nonempty_strings(blueprint.get("instructions"), {"a", "b", "bundled"}):
            raise ConstructionError(
                f"{task_id}: Graft instructions must define a, b, and bundled"
            )
        _role_sets(blueprint.get("initial_goal_roles"), f"{task_id}/initial_goal_roles")
        _role_sets(blueprint.get("final_goal_roles"), f"{task_id}/final_goal_roles")
    elif track == "refresh":
        if (
            not isinstance(blueprint.get("instruction"), str)
            or not blueprint["instruction"].strip()
        ):
            raise ConstructionError(f"{task_id}: Refresh instruction must be non-empty")
        for key in ("initial_target_role", "final_target_role"):
            if not isinstance(blueprint.get(key), str) or not blueprint[key].strip():
                raise ConstructionError(f"{task_id}: Refresh {key} must be non-empty")
    elif track == "clarify":
        obsolete_fields = {
            "instructions",
            "clarification",
            "source_candidate_roles",
            "hidden_target_assignment",
            "final_goal_roles",
            "catalog_target_roles",
        } & set(blueprint)
        if obsolete_fields:
            raise ConstructionError(
                f"{task_id}: Clarify blueprint contains obsolete fields {sorted(obsolete_fields)}"
            )
        if not _nonempty_strings(
            {
                "ambiguous": blueprint.get("ambiguous_instruction"),
                "oracle": blueprint.get("oracle_instruction"),
            },
            {"ambiguous", "oracle"},
        ):
            raise ConstructionError(
                f"{task_id}: Clarify instructions must be non-empty"
            )
        candidates = blueprint.get("candidate_roles")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 2
            or any(not isinstance(role, str) or not role for role in candidates)
            or len(set(candidates)) != 2
        ):
            raise ConstructionError(
                f"{task_id}: Clarify requires exactly two candidate roles"
            )
        if blueprint.get("hidden_target_role") not in candidates:
            raise ConstructionError(
                f"{task_id}: hidden_target_role must be one of the candidates"
            )
        slot = blueprint.get("valid_slot")
        if (
            slot not in CLARIFICATION_SLOTS
            or subtype not in CLARIFICATION_SLOTS_BY_SUBTYPE
            or slot not in CLARIFICATION_SLOTS_BY_SUBTYPE[subtype]
        ):
            raise ConstructionError(
                f"{task_id}: Clarify valid_slot is invalid for subtype {subtype!r}"
            )
        for key in ("canonical_question", "canonical_answer"):
            if not isinstance(blueprint.get(key), str) or not blueprint[key].strip():
                raise ConstructionError(f"{task_id}: Clarify {key} must be non-empty")
    else:
        raise ConstructionError(f"{task_id}: invalid task family")
    return blueprint


def opaque_item_id(task_id: str, role: str) -> str:
    digest = hashlib.sha256(f"egocua-v1\0{task_id}\0{role}".encode()).hexdigest()[:12]
    return f"p_{digest}"


def _role_sets(value: Any, context: str) -> list[list[str]]:
    if (
        not isinstance(value, list)
        or not value
        or not all(
            isinstance(option, list)
            and option
            and all(isinstance(role, str) and role for role in option)
            for option in value
        )
    ):
        raise ConstructionError(
            f"{context}: must be a non-empty list of non-empty role lists"
        )
    return value


def _map_sets(
    value: Any,
    role_ids: Mapping[str, str],
    context: str,
    equivalent_role_ids: Mapping[str, list[str]] | None = None,
) -> list[list[str]]:
    options = _role_sets(value, context)
    try:
        expanded: list[list[str]] = []
        seen: set[frozenset[str]] = set()
        equivalents = equivalent_role_ids or {}
        for option in options:
            choices = [[role_ids[role], *equivalents.get(role, [])] for role in option]
            distinct_assignments = 0
            for state in product(*choices):
                selected = frozenset(state)
                if len(selected) != len(option):
                    continue
                distinct_assignments += 1
                if selected not in seen:
                    seen.add(selected)
                    expanded.append(list(state))
            if not distinct_assignments:
                raise ConstructionError(
                    f"{context}: roles {option!r} have no distinct-entity assignment"
                )
        return expanded
    except KeyError as exc:
        raise ConstructionError(
            f"{context}: product spec has no role {exc.args[0]!r}"
        ) from exc


def _blueprint_roles(record: dict[str, Any]) -> set[str]:
    blueprint = _validate_blueprint(record)
    track = record["family"]
    roles: set[str] = set()
    if track == "graft":
        for key in ("initial_goal_roles", "final_goal_roles"):
            for option in _role_sets(blueprint.get(key), f"{record['task_id']}/{key}"):
                roles.update(option)
    elif track == "refresh":
        roles.update(
            {
                str(blueprint["initial_target_role"]),
                str(blueprint["final_target_role"]),
            }
        )
        later_distractor = blueprint.get("later_distractor_role")
        if isinstance(later_distractor, str) and later_distractor:
            roles.add(later_distractor)
    else:
        roles.update(str(role) for role in blueprint["candidate_roles"])
    construct = record.get("construct", {})
    roles.update(str(role) for role in construct.get("neighbor_role_ids", []))
    tempting = construct.get("tempting_role")
    if isinstance(tempting, str) and tempting:
        roles.add(tempting)
    return roles


def _compile_task_from_roles(
    record: dict[str, Any],
    role_ids: Mapping[str, str],
    *,
    source_identity: dict[str, str],
    site_data: dict[str, Any],
    root: str | Path,
    equivalent_role_ids: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    task_id = str(record["task_id"])
    track = str(record["family"])
    subtype = record.get("subtype")
    blueprint = _validate_blueprint(record)
    granularity = blueprint.get("target_granularity")
    packets = [
        {"id": packet.get("id"), "frame_paths": packet.get("frame_paths")}
        for packet in record.get("packets", [])
    ]
    task: dict[str, Any] = {
        "schema_version": 2,
        "id": task_id,
        "track": track,
        "site": str(site_data["site"]),
        "source": source_identity,
        "instructions": {},
        "visual_packets": packets,
    }

    if track == "graft":
        task["goal"] = {
            "target_granularity": granularity,
            "acceptable_states": _acceptable_states(
                _map_sets(
                    blueprint.get("final_goal_roles"),
                    role_ids,
                    f"{task_id}/final_goal_roles",
                    equivalent_role_ids,
                ),
                _goal_fields(blueprint, "final_goal_fields"),
            ),
        }
        task["instructions"] = blueprint.get("instructions")
        task["graft"] = {
            "transition": subtype,
            "initial_goal": {
                "target_granularity": granularity,
                "acceptable_states": _acceptable_states(
                    _map_sets(
                        blueprint.get("initial_goal_roles"),
                        role_ids,
                        f"{task_id}/initial_goal_roles",
                        equivalent_role_ids,
                    ),
                    _goal_fields(blueprint, "initial_goal_fields"),
                ),
            },
        }
    elif track == "refresh":
        initial_role = str(blueprint["initial_target_role"])
        final_role = str(blueprint["final_target_role"])
        if initial_role not in role_ids:
            raise ConstructionError(
                f"{task_id}: initial_target_role is absent from the product spec"
            )
        if final_role not in role_ids:
            raise ConstructionError(
                f"{task_id}: final_target_role is absent from the product spec"
            )
        task["goal"] = {
            "target_granularity": granularity,
            "acceptable_states": _acceptable_states(
                _map_sets(
                    [[final_role]],
                    role_ids,
                    f"{task_id}/final_target_role",
                    equivalent_role_ids,
                ),
                _goal_fields(blueprint, "goal_fields"),
            ),
        }
        task["instructions"] = {"base": blueprint.get("instruction")}
        task["refresh"] = {
            "policy": subtype,
            "initial_entity_id": role_ids[initial_role],
        }
    else:
        candidate_roles = [str(role) for role in blueprint["candidate_roles"]]
        hidden_role = str(blueprint["hidden_target_role"])
        missing_roles = sorted(set(candidate_roles) - set(role_ids))
        if missing_roles:
            raise ConstructionError(
                f"{task_id}: product spec has no candidate roles {missing_roles}"
            )
        task["goal"] = {
            "target_granularity": granularity,
            "acceptable_states": _acceptable_states(
                _map_sets(
                    [[hidden_role]],
                    role_ids,
                    f"{task_id}/hidden_target_role",
                    equivalent_role_ids,
                ),
                _goal_fields(blueprint, "goal_fields"),
            ),
        }
        task["instructions"] = {
            "ambiguous": blueprint.get("ambiguous_instruction"),
            "oracle": blueprint.get("oracle_instruction"),
        }
        task["clarification"] = {
            "candidate_ids": [role_ids[role] for role in candidate_roles],
            "valid_slot": blueprint.get("valid_slot"),
            "canonical_answer": blueprint.get("canonical_answer"),
            "max_questions": 1,
            "deixis_level": record["construct"].get("deixis_level"),
        }

    neighbors: list[str] = []
    catalog_resolution = record.get("construct", {}).get("catalog_resolution")
    if not isinstance(catalog_resolution, Mapping):
        raise ConstructionError(f"{task_id}: real catalog_resolution is required")
    raw_window_ids = catalog_resolution.get("window_entity_ids")
    preliminary_window_ids = (
        {entity_id for entity_id in raw_window_ids if isinstance(entity_id, str)}
        if isinstance(raw_window_ids, list)
        else set()
    )
    _compile_unit_equivalences(blueprint, task["goal"])
    if track == "graft":
        _compile_unit_equivalences(blueprint, task["graft"]["initial_goal"])
    # Only a complete acceptable state exposed by this task's real result
    # window is an answer.  A global-catalog alternative with one missing item
    # must not exempt its remaining item from the distractor rings.
    exposed_final_states = [
        state
        for state in task["goal"]["acceptable_states"]
        if set(state["selected_entity_ids"]) <= preliminary_window_ids
    ]
    exposed_goal = {**task["goal"], "acceptable_states": exposed_final_states}
    if "unit_equivalences" in exposed_goal:
        exposed_goal["unit_equivalences"] = {
            unit: {entity_id: quantity for entity_id, quantity in packages.items()
                   if entity_id in preliminary_window_ids}
            for unit, packages in exposed_goal["unit_equivalences"].items()
        }
        exposed_goal["unit_equivalences"] = {unit: packages for unit, packages in exposed_goal["unit_equivalences"].items() if packages}
    final_ids = set(goal_entity_ids(exposed_goal))
    core_ids = {
        str(entity_id)
        for entity_id in catalog_resolution.get("core_neighbor_entity_ids", [])
        if isinstance(entity_id, str)
    }
    rejected_ids = {
        str(row.get("entity_id"))
        for row in catalog_resolution.get("rejected", [])
        if isinstance(row, Mapping) and isinstance(row.get("entity_id"), str)
    }
    answer_conflicts = final_ids & (core_ids | rejected_ids)
    if answer_conflicts:
        raise ConstructionError(
            f"{task_id}: accepted answers are also classified as rejected neighbors: "
            f"{sorted(answer_conflicts)}"
        )
    try:
        neighbors = validate_search_neighborhood(
            task_id=task_id,
            site=str(task["site"]),
            resolution=catalog_resolution,
            available_entities={
                str(entity["entity_id"]): entity for entity in site_data["entities"]
            },
            final_ids=final_ids,
            root=Path(root).resolve(),
        )
    except NeighborhoodError as exc:
        raise ConstructionError(str(exc)) from exc
    visible_ids = set(neighbors)

    def restrict_goal(goal: dict[str, Any], context: str) -> None:
        states = [
            state
            for state in goal["acceptable_states"]
            if set(state["selected_entity_ids"]) <= visible_ids
        ]
        if not states:
            raise ConstructionError(
                f"{task_id}: {context} has no complete state in the real search window"
            )
        goal["acceptable_states"] = states
        if "unit_equivalences" in goal:
            goal["unit_equivalences"] = {
                unit: {entity_id: quantity for entity_id, quantity in packages.items()
                       if entity_id in visible_ids}
                for unit, packages in goal["unit_equivalences"].items()
            }

    # A category blueprint may enumerate interchangeable global-catalog variants.
    # The compiled truth keeps only complete states exposed by this task's two
    # frozen real-listing queries, so an invisible variant cannot become a hidden
    # checker-only answer.
    restrict_goal(task["goal"], "final goal")
    if track == "graft":
        restrict_goal(task["graft"]["initial_goal"], "initial goal")
    initial_goal = None
    if track == "refresh":
        initial_goal = {
            "acceptable_states": _acceptable_states(
                _map_sets(
                    [[str(blueprint["initial_target_role"])]],
                    role_ids,
                    f"{task_id}/initial_target_role",
                    equivalent_role_ids,
                ),
                _goal_fields(blueprint, "initial_goal_fields"),
            ),
        }
        restrict_goal(initial_goal, "initial goal")
    curve = _decision_curve(record, task, site_data, task_id)
    task["truth"] = {
        "tempting_referent_ids": [
            entity_id
            for entity_id in curve.tempting_entity_ids
            if entity_id in visible_ids
        ],
        "site_scope_entity_ids": neighbors,
    }
    if initial_goal is not None:
        task["truth"]["initial_acceptable_states"] = initial_goal["acceptable_states"]
    if track == "clarify":
        task["clarification"]["deixis_level"] = 3 if curve.is_indexical else 1
    compiled_truth = Truth.of(task)
    if not (
        compiled_truth.tempting_referent_ids
        or (track == "graft" and compiled_truth.initial_goal_differs)
    ):
        raise ConstructionError(
            f"{task_id}: the curve creates no observable difference between the "
            f"initial and final goal, so there is no failure surface"
        )

    validate_task(task, site_data, root=root, require_assets=True)
    return task


def compile_task_from_site(
    record: dict[str, Any],
    *,
    source_identity: dict[str, str],
    site_data: dict[str, Any],
    root: str | Path,
) -> dict[str, Any]:
    """Compile a screened blueprint against an already frozen site."""

    task_id = str(record.get("task_id", ""))
    if not task_id or record.get("status") != "video_screened":
        raise ConstructionError(
            "only identified video_screened records can be compiled"
        )
    entity_ids = {
        str(entity.get("entity_id"))
        for entity in site_data.get("entities", [])
        if isinstance(entity, dict) and entity.get("entity_id")
    }
    roles = _blueprint_roles(record)
    blueprint = record.get("construct", {}).get("task_blueprint", {})
    explicit_role_ids = blueprint.get("environment_role_ids", {})
    if not isinstance(explicit_role_ids, dict):
        raise ConstructionError(f"{task_id}: environment_role_ids must be an object")
    unknown_roles = sorted(set(explicit_role_ids) - set(roles))
    if unknown_roles:
        raise ConstructionError(
            f"{task_id}: environment_role_ids declares unknown roles {unknown_roles}"
        )
    role_ids = {
        role: str(explicit_role_ids.get(role) or opaque_item_id(task_id, role))
        for role in roles
    }
    raw_equivalents = blueprint.get("equivalent_role_ids", {})
    if not isinstance(raw_equivalents, dict):
        raise ConstructionError(f"{task_id}: equivalent_role_ids must be an object")
    if raw_equivalents and blueprint.get("target_granularity") == "exact_instance":
        raise ConstructionError(
            f"{task_id}: exact_instance goals cannot declare equivalent_role_ids"
        )
    unknown_equivalent_roles = sorted(set(raw_equivalents) - roles)
    if unknown_equivalent_roles:
        raise ConstructionError(
            f"{task_id}: equivalent_role_ids declares unknown roles "
            f"{unknown_equivalent_roles}"
        )
    equivalent_role_ids: dict[str, list[str]] = {}
    for role, values in raw_equivalents.items():
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ConstructionError(
                f"{task_id}: equivalent_role_ids[{role!r}] must be a non-empty "
                "list of entity IDs"
            )
        deduplicated = list(dict.fromkeys(values))
        if role_ids[role] in deduplicated:
            raise ConstructionError(
                f"{task_id}: equivalent_role_ids[{role!r}] repeats the canonical entity"
            )
        equivalent_role_ids[role] = deduplicated
    referenced_ids = set(role_ids.values()) | {
        entity_id for values in equivalent_role_ids.values() for entity_id in values
    }
    missing = sorted(referenced_ids - entity_ids)
    if missing:
        raise ConstructionError(
            f"{task_id}: frozen site is missing referenced entities {missing}"
        )
    return _compile_task_from_roles(
        record,
        role_ids,
        source_identity=source_identity,
        site_data=site_data,
        root=root,
        equivalent_role_ids=equivalent_role_ids,
    )


AUTHORED_INSTRUCTIONS = "data/instructions/authored.json"
_AUTHORED_FIELDS = {
    "graft": {"a", "b", "bundled"},
    "refresh": {"base"},
    "clarify": {"ambiguous", "oracle", "canonical_answer"},
}


def _authored_track(blueprint: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    track = record.get("family") or record.get("track")
    if track in _AUTHORED_FIELDS:
        return str(track)
    if "ambiguous_instruction" in blueprint:
        return "clarify"
    if "instruction" in blueprint:
        return "refresh"
    return "graft"


def _resolve_authored_record(
    record: dict[str, Any], authored: Mapping[str, str]
) -> None:
    blueprint = record.get("construct", {}).get("task_blueprint")
    if not isinstance(blueprint, dict):
        raise ConstructionError("authored instructions have no task blueprint")
    _resolve_authored_blueprint(
        _authored_track(blueprint, record),
        blueprint,
        authored,
    )


def _resolve_authored_blueprint(
    track: str,
    blueprint: dict[str, Any],
    authored: Mapping[str, str],
) -> None:
    """Resolve the single authored wording source into an in-memory blueprint."""

    expected = _AUTHORED_FIELDS[track]
    missing = sorted(expected - set(authored))
    if missing:
        raise ConstructionError(f"authored entry is missing {missing}")
    unknown = sorted(set(authored) - expected)
    if unknown:
        raise ConstructionError(f"authored entry has unsupported keys {unknown}")
    if track == "graft":
        blueprint["instructions"] = {
            key: authored[key] for key in ("a", "b", "bundled")
        }
    elif track == "refresh":
        blueprint["instruction"] = authored["base"]
    else:
        blueprint["ambiguous_instruction"] = authored["ambiguous"]
        blueprint["oracle_instruction"] = authored["oracle"]
        blueprint["canonical_answer"] = authored["canonical_answer"]


def _authored_overlay(root: Path) -> dict[str, dict[str, str]]:
    path = root / AUTHORED_INSTRUCTIONS
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text())
    instructions = payload.get("instructions") if isinstance(payload, dict) else None
    if not isinstance(instructions, dict):
        raise ConstructionError(
            f"{AUTHORED_INSTRUCTIONS}: instructions must be an object"
        )
    return instructions


def _apply_authored_overlay(
    record: dict[str, Any], overlay: Mapping[str, dict[str, str]]
) -> None:
    task_id = str(record.get("task_id", ""))
    authored = overlay.get(task_id)
    if authored is None:
        raise ConstructionError(
            f"{task_id}: selected task is missing its authored instruction"
        )
    try:
        _resolve_authored_record(record, authored)
    except ConstructionError as exc:
        raise ConstructionError(f"{task_id}: {exc}") from exc


def resolve_authored_record(
    record: Mapping[str, Any],
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Return a compile-ready record without changing the source-manifest record."""

    resolved = deepcopy(dict(record))
    _apply_authored_overlay(resolved, _authored_overlay(Path(root).resolve()))
    return resolved


def _screened_records(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    try:
        return screened_records(root)
    except SourceError as exc:
        raise ConstructionError(str(exc)) from exc


def load_construction_tasks(
    root: str | Path,
    task_ids: list[str] | None = None,
    *,
    tasks_path: str | Path = "data/tasks",
) -> list[dict[str, Any]]:
    """Read compiled members and explicitly requested, source-backed candidates.

    Candidate compilation is read-only and does not grant formal membership.
    Review and trial callers consume the same executable definition.
    """
    workspace = Path(root).resolve()
    tasks = load_tasks(workspace / tasks_path)
    requested = set(task_ids or [])
    missing = requested - {task["id"] for task in tasks}
    if not missing:
        return tasks
    records = {
        record["task_id"]: (path, record)
        for path, record in _screened_records(workspace)
    }
    unknown = missing - set(records)
    if unknown:
        raise ConstructionError(f"unknown construction candidates: {sorted(unknown)}")
    authored = _authored_overlay(workspace)
    sites: dict[str, dict[str, Any]] = {}
    for task_id in sorted(missing):
        path, record = records[task_id]
        reference = path.relative_to(workspace).as_posix()
        provenance, _ = discover_frame_provenance(reference, root=workspace)
        resolved = deepcopy(record)
        _apply_authored_overlay(resolved, authored)
        site = str(resolved["construct"]["task_blueprint"]["site"])
        if site not in sites:
            sites[site] = load_site_data(workspace, site)
        tasks.append(
            compile_task_from_site(
                resolved,
                source_identity=build_source_identity(
                    record,
                    dossier_reference=reference,
                    provenance_reference=provenance,
                    root=workspace,
                ),
                site_data=sites[site],
                root=workspace,
            )
        )
    return tasks


class CurrentConstruction:
    """Compare current owners with executable tasks using one request's materials."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._dossiers: dict[str, dict[str, Any]] = {}
        self._sites: dict[str, dict[str, Any]] = {}

    @cached_property
    def _authored(self) -> dict[str, dict[str, str]]:
        return _authored_overlay(self.root)

    def source(self, task: Mapping[str, Any]) -> dict[str, Any]:
        reference = str(task["source"]["dossier"])
        if reference not in self._dossiers:
            self._dossiers[reference] = load_dossier(task, root=self.root)[1]
        source_id = str(task["source"].get("task_id") or task["id"])
        return find_record(
            self._dossiers[reference], source_id, context=f"{task['id']} source record"
        )

    def site(self, site: str) -> dict[str, Any]:
        if site not in self._sites:
            self._sites[site] = load_site_data(self.root, site)
        return self._sites[site]

    def matches(self, task: dict[str, Any]) -> bool:
        record = deepcopy(self.source(task))
        _apply_authored_overlay(record, self._authored)
        rebuilt = compile_task_from_site(
            record,
            source_identity=task["source"],
            site_data=self.site(str(task["site"])),
            root=self.root,
        )
        return task_local_content(rebuilt) == task_local_content(task)


def build_selected(
    root: str | Path = ".",
    *,
    task_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Compile the review queue against frozen sites and authored instructions."""

    workspace = Path(root).resolve()
    try:
        selection = load_selection(workspace)
        selected_ids = selected_task_ids(selection)
    except SelectionError as exc:
        raise ConstructionError(str(exc)) from exc
    if task_ids is not None:
        unknown = set(task_ids) - set(selected_ids)
        if unknown:
            raise ConstructionError(
                f"task IDs are not selected members: {sorted(unknown)}"
            )
    allowed_manifests = set(selection.get("source_manifests", []))
    screened = _screened_records(workspace)
    records_by_id = {
        str(record["task_id"]): (manifest, record) for manifest, record in screened
    }
    missing = sorted(set(selected_ids) - set(records_by_id))
    if missing:
        raise ConstructionError(f"selected source records are missing: {missing}")
    try:
        source_summary = validate_independent_recordings(
            [records_by_id[task_id] for task_id in selected_ids], root=workspace
        )
    except LineageError as exc:
        raise ConstructionError(str(exc)) from exc
    if task_ids is not None:
        requested = set(task_ids)
        selected_ids = [task_id for task_id in selected_ids if task_id in requested]
    site_cache: dict[str, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []
    authored = _authored_overlay(workspace)

    for task_id in selected_ids:
        manifest, source_record = records_by_id[task_id]
        record = deepcopy(source_record)
        _apply_authored_overlay(record, authored)
        blueprint = record["construct"]["task_blueprint"]
        dossier_reference = manifest.relative_to(workspace).as_posix()
        if dossier_reference not in allowed_manifests:
            raise ConstructionError(
                f"{task_id}: selected source manifest is not registered by selection.json"
            )
        provenance_reference, _ = discover_frame_provenance(
            dossier_reference, root=workspace
        )
        source_identity = build_source_identity(
            source_record,
            dossier_reference=dossier_reference,
            provenance_reference=provenance_reference,
            root=workspace,
        )
        site = str(blueprint["site"])
        if site not in site_cache:
            site_cache[site] = load_site_data(workspace, site)
        try:
            task = compile_task_from_site(
                record,
                source_identity=source_identity,
                site_data=site_cache[site],
                root=workspace,
            )
        except ConstructionError as exc:
            raise ConstructionError(
                f"{task_id}: selected task cannot be constructed: {exc}"
            ) from exc
        tasks.append(task)

    tasks.sort(key=lambda task: (TRACK_ORDER.index(task["track"]), task["id"]))
    validate_dataset(tasks, root=workspace, require_assets=True)
    report = {
        "schema_version": 1,
        "screened_records": len(screened),
        "selected_records": len(selected_ids),
        "constructed_tasks": len(tasks),
        "rescreening_candidates": len(screened) - len(selected_ids),
        **(
            source_summary
            if task_ids is None
            else summarize_source_recordings(
                [records_by_id[task_id] for task_id in selected_ids], root=workspace
            )
        ),
        "by_track": dict(sorted(Counter(task["track"] for task in tasks).items())),
        "by_site": dict(sorted(Counter(task["site"] for task in tasks).items())),
    }
    return tasks, selection, report


def write_selected(
    root: str | Path = ".", *, task_ids: list[str] | None = None
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    tasks, _, report = (
        build_selected(workspace)
        if task_ids is None
        else build_selected(workspace, task_ids=task_ids)
    )
    task_root = workspace / "data/tasks"
    expected: set[Path] = set()
    for task in tasks:
        path = task_root / task["track"] / f"{task['id']}.json"
        write_json_atomic(path, task)
        expected.add(path)
    updated_ids = {task["id"] for task in tasks}
    for stale in task_root.glob("*/*.json"):
        if stale not in expected and (task_ids is None or stale.stem in updated_ids):
            stale.unlink()
    if task_ids is None:
        write_json_atomic(workspace / "artifacts/reports/task-build.json", report)
    return report


def check_selected(root: str | Path = ".") -> dict[str, Any]:
    workspace = Path(root).resolve()
    tasks, _, report = build_selected(workspace)
    current_by_id = {
        task["id"]: task_local_content(task)
        for task in load_tasks(workspace / "data/tasks")
    }
    expected_by_id = {task["id"]: task_local_content(task) for task in tasks}
    if expected_by_id != current_by_id:
        raise ConstructionError(
            "data/tasks is stale; run tools.build.build_screened_tasks"
        )
    return report
