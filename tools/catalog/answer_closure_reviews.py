"""Decide open window listings: the `answer_closure` gate's, or a new search round's.

`packets` writes one decision packet per task. Without `--stage-suffix` it lists the
listings the closure gate leaves open; with it, every listing first returned by the
task's query rounds whose stage ends with that suffix (see `append_search_round`).
A Coding Agent reads the frames and public listings and writes
`REVIEWS_DIR/<task>.json`. `apply` turns those decisions into source truth: an
accepted listing joins its role's `equivalent_role_ids` (a role that is not part of
the final state stays a classified final non-answer); a rejected one gets a
`catalog_resolution.rejected` row. A round-scoped review must decide every listing
in its scope. `apply` recompiles the touched tasks; rerun the acceptance audit.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from egocua.answer_closure import (
    accepted_ids,
    catalog_tokens,
    target_predicates,
    undecided_window_listings,
)
from egocua.environment import entity_visual_path, load_site_data
from egocua.schema import load_tasks
from egocua.selection import load_selection, task_ids
from egocua.sources import iter_manifests
from egocua.workspace import read_json_strict, write_json_atomic
from tools.catalog.apply_core_neighbor_reviews import RANKING_JUSTIFICATION
from tools.catalog.propose_core_neighbors import PREDICATES

REVIEWS_DIR = Path("data/sources/answer_closure_reviews")
PACKETS_DIR = Path("artifacts/closure-packets")
PUBLIC_FIELD_LIMIT = 700
HIDDEN_FIELDS = {
    "entity_id",
    "item_id",
    "text_provenance",
    "image_provenance",
    "source",
}
RING_FIELDS = ("core_neighbor_entity_ids", "related_entity_ids", "ambient_entity_ids")
REVIEW_PREDICATES = PREDICATES | {"insufficient_public_evidence"}


class ClosureReviewError(ValueError):
    """Raised when a closure decision cannot become source truth."""


def _records(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Path], dict]:
    records: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    manifests: dict[Path, dict[str, Any]] = {}
    for path, payload in iter_manifests(root):
        manifests[path] = payload
        for record in payload.get("tasks", []):
            if isinstance(record, dict) and record.get("task_id"):
                records[str(record["task_id"])] = record
                paths[str(record["task_id"])] = path
    return records, paths, manifests


def _role_of(blueprint: Mapping[str, Any]) -> dict[str, str]:
    roles = {
        str(eid): str(role) for role, eid in blueprint["environment_role_ids"].items()
    }
    for role, alternatives in (blueprint.get("equivalent_role_ids") or {}).items():
        roles.update({str(eid): str(role) for eid in alternatives})
    return roles


def _public_view(site: str, entity: Mapping[str, Any]) -> dict[str, Any]:
    view: dict[str, Any] = {
        key: str(value)[:PUBLIC_FIELD_LIMIT]
        for key, value in entity.items()
        if key not in HIDDEN_FIELDS and not key.endswith("_path")
    }
    view["image"] = entity_visual_path(site, dict(entity))
    return view


def _ring_of(resolution: Mapping[str, Any], entity_id: str) -> str:
    for field in RING_FIELDS:
        if entity_id in (resolution.get(field) or []):
            return field.split("_", 1)[0]
    return "window"


def round_listings(resolution: Mapping[str, Any], stage_suffix: str) -> list[str]:
    """Listings first returned by the query rounds whose stage ends with the suffix."""

    earlier: set[str] = set()
    added: list[str] = []
    for row in resolution.get("query_rounds") or []:
        results = [str(entity_id) for entity_id in row.get("result_entity_ids") or []]
        if not str(row.get("stage", "")).endswith(stage_suffix):
            earlier.update(results)
            continue
        added.extend(e for e in results if e not in added)
    return [entity_id for entity_id in added if entity_id not in earlier]


def _open_listings(
    task: Mapping[str, Any],
    resolution: Mapping[str, Any],
    catalog: Mapping[str, frozenset[str]],
    stage_suffix: str | None,
) -> dict[str, list[str]]:
    if stage_suffix is None:
        return undecided_window_listings(task, resolution, catalog)
    final = set(accepted_ids(task))
    return {
        entity_id: []
        for entity_id in round_listings(resolution, stage_suffix)
        if entity_id not in final
    }


def build_packet(
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    site_data: Mapping[str, Any],
    stage_suffix: str | None = None,
) -> dict[str, Any] | None:
    """Return the decision packet, or None when nothing in its scope is open."""

    site = str(task["site"])
    resolution = record["construct"]["catalog_resolution"]
    catalog = catalog_tokens(site, site_data)
    undecided = _open_listings(task, resolution, catalog, stage_suffix)
    if not undecided:
        return None
    entities = {str(entity["entity_id"]): entity for entity in site_data["entities"]}
    roles = _role_of(record["construct"]["task_blueprint"])
    predicates = target_predicates(task, catalog)
    return {
        "task_id": task["id"],
        "track": task["track"],
        "site": site,
        "instructions": task["instructions"],
        "clarification_answer": (task.get("clarification") or {}).get(
            "canonical_answer"
        ),
        "frames": {
            packet["id"]: packet["frame_paths"] for packet in task["visual_packets"]
        },
        "accepted": [
            {
                "entity_id": entity_id,
                "role": roles.get(entity_id),
                "confirmed_released_words": predicates[entity_id],
                **_public_view(site, entities[entity_id]),
            }
            for entity_id in accepted_ids(task)
        ],
        "undecided": [
            {
                "entity_id": entity_id,
                "matched_words": words,
                "candidate_roles": sorted(
                    {
                        str(roles.get(accepted))
                        for accepted, confirmed in predicates.items()
                        if confirmed and set(confirmed) <= catalog[entity_id]
                    }
                ),
                "ring": _ring_of(resolution, entity_id),
                **_public_view(site, entities[entity_id]),
            }
            for entity_id, words in undecided.items()
        ],
        "existing_rejections": [
            {"entity_id": row["entity_id"], "reason": row["reason"]}
            for row in resolution.get("rejected") or []
        ],
        "failure_predicates": sorted(REVIEW_PREDICATES),
        "stage_suffix": stage_suffix,
        "review_path": (REVIEWS_DIR / f"{task['id']}.json").as_posix(),
    }


def write_packets(
    root: Path, wanted: set[str] | None, stage_suffix: str | None = None
) -> dict[str, int]:
    tasks = {str(task["id"]): task for task in load_tasks(root / "data/tasks")}
    members = [
        tid for tid in task_ids(load_selection(root)) if wanted is None or tid in wanted
    ]
    records, _, _ = _records(root)
    sites: dict[str, dict[str, Any]] = {}
    written: dict[str, int] = {}
    for task_id in members:
        task = tasks[task_id]
        resolution = (records[task_id].get("construct") or {}).get("catalog_resolution")
        if not isinstance(resolution, Mapping):
            continue
        site = str(task["site"])
        if site not in sites:
            sites[site] = load_site_data(root, site)
        site_data = sites[site]
        packet = build_packet(task, records[task_id], site_data, stage_suffix)
        if packet is None:
            continue
        write_json_atomic(root / PACKETS_DIR / f"{task_id}.json", packet)
        written[task_id] = len(packet["undecided"])
    return written


def _validated_decision(
    entity_id: str, decision: Mapping[str, Any], roles: set[str]
) -> dict[str, Any]:
    reason = str(decision.get("reason") or "").strip()
    if not reason or RANKING_JUSTIFICATION.search(reason):
        raise ClosureReviewError(
            f"{entity_id}: a released, non-ranking reason is required"
        )
    if decision.get("decision") == "accept":
        if "roles" in decision:
            exact = decision["roles"]
            if not isinstance(exact, list) or not exact or not set(exact) <= roles:
                raise ClosureReviewError(
                    f"{entity_id}: accept roles must be a non-empty subset of {sorted(roles)}"
                )
            return {"decision": "accept", "roles": list(exact), "reason": reason}
        if decision.get("role") not in roles:
            raise ClosureReviewError(
                f"{entity_id}: accept needs one of the roles {sorted(roles)}"
            )
        return {"decision": "accept", "role": decision["role"], "reason": reason}
    predicates = decision.get("failure_predicates")
    if (
        decision.get("decision") != "reject"
        or not isinstance(predicates, list)
        or not predicates
        or not set(predicates) <= REVIEW_PREDICATES
    ):
        raise ClosureReviewError(
            f"{entity_id}: reject needs failure_predicates from {sorted(REVIEW_PREDICATES)}"
        )
    return {
        "decision": "reject",
        "reason": reason,
        "failure_predicates": list(predicates),
        "withdraw": decision.get("withdraw") is True,
    }


def _accepted_roles(blueprint: Mapping[str, Any], entity_id: str) -> set[str]:
    return {
        role
        for role, alternatives in blueprint.get("equivalent_role_ids", {}).items()
        if entity_id in alternatives
    } | {
        role
        for role, primary in blueprint["environment_role_ids"].items()
        if primary == entity_id
    }


def _withdraw(blueprint: dict[str, Any], entity_id: str, keep: set[str]) -> None:
    """Remove an earlier accept from every role outside `keep`."""

    primaries = {
        role
        for role, primary in blueprint["environment_role_ids"].items()
        if primary == entity_id and role not in keep
    }
    if primaries:
        raise ClosureReviewError(
            f"{entity_id} is the primary listing of {sorted(primaries)}; reassign the role first"
        )
    equivalents = blueprint.get("equivalent_role_ids", {})
    for role, alternatives in equivalents.items():
        if role not in keep and entity_id in alternatives:
            alternatives.remove(entity_id)
    for role in [role for role, alternatives in equivalents.items() if not alternatives]:
        del equivalents[role]


def _non_final_roles(blueprint: Mapping[str, Any]) -> set[str]:
    """Roles answered only before the later change (Graft initial, Refresh initial)."""

    if "final_goal_roles" in blueprint:
        final = {role for state in blueprint["final_goal_roles"] for role in state}
    elif "final_target_role" in blueprint:
        final = {blueprint["final_target_role"]}
    else:
        return set()
    return set(blueprint["environment_role_ids"]) - final


def _accept(
    resolution: dict[str, Any], blueprint: dict[str, Any], entity_id: str, role: str
) -> None:
    alternatives = blueprint.setdefault("equivalent_role_ids", {}).setdefault(role, [])
    if (
        entity_id not in alternatives
        and entity_id != blueprint["environment_role_ids"][role]
    ):
        alternatives.append(entity_id)
    previous = [
        row
        for row in resolution.get("rejected", [])
        if row.get("entity_id") == entity_id
    ]
    if previous:
        resolution.setdefault("rejection_history", []).extend(previous)
    if role in _non_final_roles(blueprint):
        if _ring_of(resolution, entity_id) == "window":
            resolution.setdefault("related_entity_ids", []).append(entity_id)
        _reject(
            resolution,
            entity_id,
            {
                "reason": f"Satisfies the earlier request as {role}; the later change "
                "replaces that role in the final state.",
                "why_plausible": f"It is an accepted {role} answer for the earlier request.",
                "failure_predicates": ["later_change_mismatch"],
            },
            [],
        )
        return
    for field in RING_FIELDS:
        resolution[field] = [
            eid for eid in resolution.get(field, []) if eid != entity_id
        ]
    resolution["rejected"] = [
        row
        for row in resolution.get("rejected", [])
        if row.get("entity_id") != entity_id
    ]


def _reject(
    resolution: dict[str, Any],
    entity_id: str,
    decision: Mapping[str, Any],
    words: list[str],
) -> None:
    rows = [
        row
        for row in resolution.get("rejected", [])
        if row.get("entity_id") != entity_id
    ]
    why_plausible = decision.get("why_plausible") or (
        "Its listing text carries every released word an accepted "
        f"listing confirms: {', '.join(words)}."
        if words
        else "A real search for this task returned it."
    )
    rows.append(
        {
            "entity_id": entity_id,
            "ring": _ring_of(resolution, entity_id),
            "reason": decision["reason"],
            "why_plausible": why_plausible,
            "failure_predicates": decision["failure_predicates"],
            "semantic_review": "coding_agent_accepted",
            "review_reference": REVIEWS_DIR.as_posix(),
        }
    )
    resolution["rejected"] = rows


def _apply_decision(
    resolution: dict[str, Any],
    blueprint: dict[str, Any],
    entity_id: str,
    decision: Mapping[str, Any],
    words: list[str],
) -> None:
    """`roles` sets the exact accepted roles; a reject of an accepted listing must withdraw it."""

    if decision["decision"] == "accept" and "roles" in decision:
        exact = set(decision["roles"])
        _withdraw(blueprint, entity_id, exact)
        non_final = _non_final_roles(blueprint)
        for role in sorted(exact, key=lambda role: role not in non_final):
            _accept(resolution, blueprint, entity_id, role)
        return
    if decision["decision"] == "accept":
        _accept(resolution, blueprint, entity_id, decision["role"])
        return
    accepted = _accepted_roles(blueprint, entity_id)
    if accepted:
        if not decision["withdraw"]:
            raise ClosureReviewError(
                f"{entity_id} is accepted for {sorted(accepted)}; set withdraw to reject it"
            )
        _withdraw(blueprint, entity_id, set())
        if _ring_of(resolution, entity_id) == "window":
            resolution.setdefault("related_entity_ids", []).append(entity_id)
    _reject(resolution, entity_id, decision, words)


SLOT_INDEX = {"item_1": 0, "item_2": 1}


def retained_count(task: Mapping[str, Any]) -> int:
    """How many initial items each final state still contains: 1 local, 0 global."""

    if task["goal"].get("unit_equivalences") or task["graft"]["initial_goal"].get("unit_equivalences"):
        raise ClosureReviewError("quantity tasks must update the reviewed source unit mapping, not append a SKU slot")
    initial = [
        state["selected_entity_ids"]
        for state in task["graft"]["initial_goal"]["acceptable_states"]
    ]
    overlaps = {
        max(len(set(state["selected_entity_ids"]) & set(before)) for before in initial)
        for state in task["goal"]["acceptable_states"]
    }
    if overlaps != {0} and overlaps != {1}:
        raise ClosureReviewError("scope states do not share one retained count")
    return overlaps.pop()


def add_scope_equivalent(task: dict[str, Any], entity_id: str, slot: str) -> int:
    """Add one listing to a scope slot, keeping the local or global contract."""

    index = SLOT_INDEX[slot]
    kept = retained_count(task)
    initial = [
        state["selected_entity_ids"]
        for state in task["graft"]["initial_goal"]["acceptable_states"]
    ]
    states = task["goal"]["acceptable_states"]
    if any(state["selected_entity_ids"][index] == entity_id for state in states):
        return 0
    seen = {tuple(state["selected_entity_ids"]) for state in states}
    added = []
    for state in states:
        ids = list(state["selected_entity_ids"])
        ids[index] = entity_id
        key = tuple(ids)
        if key in seen or len(set(ids)) != 2:
            continue
        overlap = max(len(set(ids) & set(before)) for before in initial)
        if overlap != kept:
            continue
        seen.add(key)
        added.append({"selected_entity_ids": ids, "fields": {}})
    if not added:
        raise ClosureReviewError(
            f"{entity_id} cannot join {slot} without breaking the scope contract"
        )
    states.extend(added)
    return len(added)


def apply_reviews(root: Path, wanted: set[str] | None) -> dict[str, dict[str, int]]:
    records, paths, manifests = _records(root)
    tasks = {str(task["id"]): task for task in load_tasks(root / "data/tasks")}
    sites: dict[str, dict[str, Any]] = {}
    applied: dict[str, dict[str, int]] = {}
    for review_path in sorted((root / REVIEWS_DIR).glob("*.json")):
        task_id = review_path.stem
        if wanted is not None and task_id not in wanted:
            continue
        if task_id not in tasks:
            continue
        review = read_json_strict(review_path)
        if not str(review.get("reviewer") or "").strip():
            raise ClosureReviewError(f"{task_id}: reviewer is required")
        task, record = tasks[task_id], records[task_id]
        site = str(task["site"])
        site_data = sites.setdefault(site, load_site_data(root, site))
        resolution = record["construct"]["catalog_resolution"]
        blueprint = record["construct"]["task_blueprint"]
        window = set(resolution["window_entity_ids"])
        catalog = catalog_tokens(site, site_data)
        matched = undecided_window_listings(task, resolution, catalog)
        stage_suffix = review.get("stage_suffix")
        if stage_suffix is not None:
            missing = set(
                _open_listings(task, resolution, catalog, str(stage_suffix))
            ) - set(review["listings"])
            if missing:
                raise ClosureReviewError(
                    f"{task_id}: round-scoped review leaves {sorted(missing)} undecided"
                )
        counts = {"accept": 0, "reject": 0}
        for entity_id, raw in review["listings"].items():
            if entity_id not in window:
                raise ClosureReviewError(
                    f"{task_id}: {entity_id} is not in the frozen window"
                )
            decision = _validated_decision(
                entity_id, raw, set(blueprint["environment_role_ids"])
            )
            _apply_decision(
                resolution, blueprint, entity_id, decision, matched.get(entity_id, [])
            )
            counts[decision["decision"]] += 1
        applied[task_id] = counts
    for path in {paths[task_id] for task_id in applied}:
        write_json_atomic(path, manifests[path])
    if applied:
        from egocua.construction import write_selected

        write_selected(root, task_ids=sorted(applied))
    return applied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("packets", "apply"))
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-id", action="append")
    parser.add_argument(
        "--stage-suffix",
        help="packets: list listings first returned by rounds whose stage ends with this",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    wanted = set(args.task_id) if args.task_id else None
    result = (
        write_packets(root, wanted, args.stage_suffix)
        if args.command == "packets"
        else apply_reviews(root, wanted)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
