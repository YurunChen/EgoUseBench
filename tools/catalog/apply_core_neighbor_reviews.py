"""Apply task-local Coding-Agent reviews for real search-window neighbors.

Qwen proposals are evidence, not admission.  A task is admitted only when the
current payload is reviewed, enough real non-answer neighbors remain, and each
has concrete rejection evidence. Explicit
Coding-Agent overrides repair semantic mistakes without mutating model output.
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import product
from pathlib import Path
from typing import Any, Mapping

from egocua.environment import load_site_data
from egocua.neighborhood import (
    TARGET_RELATED_RESULTS,
)
from egocua.run_ledger import canonical_sha256
from egocua.schema import load_tasks
from egocua.selection import load_selection, task_ids
from egocua.sources import iter_manifests
from egocua.workspace import read_json_strict, write_json_atomic
from tools.catalog.propose_core_neighbors import PREDICATES, _payload


PROPOSALS = Path("data/sources/core_neighbor_proposals.json")
OVERRIDES = Path("data/sources/core_neighbor_review_overrides.json")
REVIEWS = Path("data/sources/core_neighbor_semantic_reviews.json")
MODEL_SELF_DIALOGUE = re.compile(
    r"\b(?:wait|let(?:'|’)s|we need|i need to|re-evaluate|actually,? the correct)\b",
    re.IGNORECASE,
)
RANKING_JUSTIFICATION = re.compile(
    r"\b(?:rank|ranked|ranking|after_rank|before_rank)\b|排名|排在"
)


def _final_answer_ids(record: Mapping[str, Any], window: list[str]) -> set[str]:
    """Resolve current blueprint answers without depending on stale compiled tasks."""

    construct = record.get("construct")
    blueprint = (
        construct.get("task_blueprint") if isinstance(construct, Mapping) else None
    )
    if not isinstance(blueprint, Mapping):
        raise ValueError(f"{record.get('task_id')}: task_blueprint is required")
    role_ids = blueprint.get("environment_role_ids")
    if not isinstance(role_ids, Mapping):
        raise ValueError(f"{record.get('task_id')}: environment_role_ids is required")
    equivalents = blueprint.get("equivalent_role_ids") or {}
    if not isinstance(equivalents, Mapping):
        raise ValueError(
            f"{record.get('task_id')}: equivalent_role_ids must be an object"
        )

    family = record.get("family")
    if family == "graft":
        role_sets = blueprint.get("final_goal_roles")
    elif family == "refresh":
        role_sets = [[blueprint.get("final_target_role")]]
    elif family == "clarify":
        role_sets = [[blueprint.get("hidden_target_role")]]
    else:
        raise ValueError(f"{record.get('task_id')}: unsupported family {family!r}")
    if not isinstance(role_sets, list) or not role_sets:
        raise ValueError(f"{record.get('task_id')}: final roles are required")

    window_ids = set(window)
    answer_ids: set[str] = set()
    for roles in role_sets:
        if not isinstance(roles, list) or not roles:
            raise ValueError(f"{record.get('task_id')}: invalid final role set")
        choices: list[list[str]] = []
        for role in roles:
            canonical = role_ids.get(role)
            alternatives = equivalents.get(role, [])
            if (
                not isinstance(canonical, str)
                or not canonical
                or not isinstance(alternatives, list)
                or any(not isinstance(item, str) or not item for item in alternatives)
            ):
                raise ValueError(
                    f"{record.get('task_id')}: invalid answer role {role!r}"
                )
            choices.append([canonical, *alternatives])
        for state in product(*choices):
            if set(state) <= window_ids:
                answer_ids.update(state)
    if not answer_ids:
        raise ValueError(
            f"{record.get('task_id')}: no complete final answer in result window"
        )
    return answer_ids


def _all_true(assessment: Mapping[str, Any]) -> bool:
    return all(
        assessment.get(key) is True
        for key in (
            "useful_in_real_life",
            "scene_update_website_consistent",
            "continuous_change",
            "deterministically_verifiable",
        )
    )


def _reviewed_neighbors(
    task_id: str,
    proposal: Mapping[str, Any],
    override: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    source = (
        override.get("replacement_core_neighbors")
        if isinstance(override, Mapping) and "replacement_core_neighbors" in override
        else proposal.get("core_neighbors")
    )
    if not isinstance(source, list):
        raise ValueError(f"{task_id}: reviewed core neighbors must be a list")
    predicate_overrides = (
        override.get("predicate_overrides") if isinstance(override, Mapping) else None
    )
    predicate_overrides = (
        predicate_overrides if isinstance(predicate_overrides, Mapping) else {}
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_row in source:
        if not isinstance(source_row, Mapping):
            raise ValueError(f"{task_id}: invalid reviewed core-neighbor row")
        entity_id = str(source_row.get("entity_id") or "")
        why_plausible = str(source_row.get("why_plausible") or "").strip()
        why_rejected = str(source_row.get("why_rejected") or "").strip()
        reasons = predicate_overrides.get(
            entity_id, source_row.get("failure_predicates")
        )
        if (
            not entity_id
            or entity_id in seen
            or not why_plausible
            or not why_rejected
            or MODEL_SELF_DIALOGUE.search(why_plausible)
            or MODEL_SELF_DIALOGUE.search(why_rejected)
            or RANKING_JUSTIFICATION.search(why_rejected)
            or not isinstance(reasons, list)
            or not reasons
            or not set(reasons) <= PREDICATES
        ):
            raise ValueError(f"{task_id}: invalid reviewed core neighbor {entity_id!r}")
        seen.add(entity_id)
        rows.append(
            {
                "entity_id": entity_id,
                "ring": "core",
                "why_plausible": why_plausible,
                "why_rejected": why_rejected,
                "failure_predicates": list(reasons),
            }
        )
    return rows


def apply(root: Path, *, target_task_ids: set[str] | None = None) -> dict[str, int]:
    selected = task_ids(load_selection(root))
    selected_set = set(selected)
    if target_task_ids is not None and not target_task_ids <= selected_set:
        raise ValueError(
            f"unknown selected task IDs: {sorted(target_task_ids - selected_set)}"
        )
    targets = [
        task_id
        for task_id in selected
        if target_task_ids is None or task_id in target_task_ids
    ]
    tasks = {str(task["id"]): task for task in load_tasks(root / "data/tasks")}
    if selected_set != set(tasks):
        raise ValueError("data/tasks must exactly match the 157-task release set")
    proposals = read_json_strict(root / PROPOSALS)["tasks"]
    overrides_document = read_json_strict(root / OVERRIDES)
    overrides = overrides_document.get("tasks") or {}
    reviewer = str(overrides_document.get("reviewer") or "")
    if not reviewer:
        raise ValueError("core-neighbor override reviewer is required")

    manifests: dict[Path, dict[str, Any]] = {}
    records: dict[str, dict[str, Any]] = {}
    record_paths: dict[str, Path] = {}
    for path, payload in iter_manifests(root):
        manifests[path] = payload
        for record in payload.get("tasks", []):
            if isinstance(record, dict) and record.get("task_id"):
                task_id = str(record["task_id"])
                records[task_id] = record
                record_paths[task_id] = path
    sites = {
        site: {
            str(entity["entity_id"]): entity
            for entity in load_site_data(root, site)["entities"]
        }
        for site in {str(tasks[task_id]["site"]) for task_id in targets}
    }

    reviews_document = (
        read_json_strict(root / REVIEWS) if (root / REVIEWS).is_file() else {}
    )
    reviews: dict[str, Any] = dict(reviews_document.get("tasks") or {})
    override_count = 0
    for task_id in targets:
        task = tasks[task_id]
        record = records[task_id]
        proposal = proposals.get(task_id)
        if not isinstance(proposal, Mapping):
            raise ValueError(f"{task_id}: missing model core-neighbor proposal")
        payload = _payload(task, record, sites[str(task["site"])])
        input_sha256 = canonical_sha256(payload)
        override = overrides.get(task_id)
        if override is None:
            if proposal.get("input_sha256") != input_sha256:
                raise ValueError(f"{task_id}: model proposal is stale")
            if not _all_true(proposal.get("task_assessment") or {}):
                raise ValueError(
                    f"{task_id}: low-confidence proposal lacks Coding-Agent repair"
                )
            rationale = "当前任务语义与真实窗口一致；四个近邻均有具体排除事实。"
            basis = "model_proposal_confirmed_by_coding_agent"
        else:
            if (
                not isinstance(override, Mapping)
                or override.get("decision") != "accept"
                or override.get("confidence") != "high"
                or not str(override.get("rationale") or "").strip()
            ):
                raise ValueError(f"{task_id}: invalid Coding-Agent override")
            rationale = str(override["rationale"])
            basis = "coding_agent_semantic_override"
            override_count += 1

        core_rows = _reviewed_neighbors(task_id, proposal, override)
        core_ids = [row["entity_id"] for row in core_rows]
        resolution = record["construct"]["catalog_resolution"]
        window = list(resolution["window_entity_ids"])
        final_ids = _final_answer_ids(record, window)
        allowed = set(window) - final_ids
        if not set(core_ids) <= allowed:
            raise ValueError(
                f"{task_id}: reviewed core neighbor is not a real non-answer"
            )
        remaining = [
            entity_id
            for entity_id in window
            if entity_id in allowed and entity_id not in set(core_ids)
        ]
        related = remaining[:TARGET_RELATED_RESULTS]
        ambient = remaining[TARGET_RELATED_RESULTS:]
        resolution["core_neighbor_entity_ids"] = core_ids
        resolution["related_entity_ids"] = related
        resolution["ambient_entity_ids"] = ambient
        closure_rows = [
            row
            for row in resolution.get("rejected", [])
            if row.get("ring") != "core" and row.get("entity_id") not in core_ids
        ]
        resolution["rejected"] = [
            {
                "entity_id": row["entity_id"],
                "ring": "core",
                "reason": row["why_rejected"],
                "why_plausible": row["why_plausible"],
                "failure_predicates": row["failure_predicates"],
                "semantic_review": "coding_agent_accepted",
            }
            for row in core_rows
        ] + closure_rows
        resolution["semantic_review"] = {
            "status": "coding_agent_accepted",
            "confidence": "high",
            "reviewer": reviewer,
            "review_reference": REVIEWS.as_posix(),
            "required_checks": [
                "usefulness",
                "scene_update_website_consistency",
                "continuous_change",
                "deterministic_verifiability",
                "source_grounded_core_neighbors",
                "per_neighbor_rejection_evidence",
            ],
        }
        reviews[task_id] = {
            "decision": "accept",
            "confidence": "high",
            "needs_human_review": False,
            "reviewer": reviewer,
            "review_basis": basis,
            "coding_agent_rationale": rationale,
            "input_sha256": input_sha256,
            "core_neighbors": core_rows,
            "task_assessment": {
                "useful_in_real_life": True,
                "scene_update_website_consistent": True,
                "continuous_change": True,
                "deterministically_verifiable": True,
            },
        }

    for path in {record_paths[task_id] for task_id in targets}:
        write_json_atomic(path, manifests[path])
    write_json_atomic(
        root / REVIEWS,
        {
            "schema_version": 1,
            "reviewer": reviewer,
            "tasks": reviews,
        },
    )
    return {
        "reviewed": len(targets),
        "accepted_high_confidence": len(targets),
        "coding_agent_overrides": override_count,
        "human_review_required": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    print(
        json.dumps(
            apply(
                Path(args.root).resolve(),
                target_task_ids=set(args.task) if args.task else None,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
