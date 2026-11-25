"""Admit Coding-Agent-reviewed upstream landings and update task role bindings.

Model proposals are suggestions. A proposal with any failed semantic dimension is
blocked unless the Skill-executing Coding Agent records an explicit evidence-based
override. An already reviewed role binding can only change through such an override,
so rerunning an old proposal cannot roll back a later task repair. Reviews and role
bindings are task-local; no catalog hash invalidates unrelated tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from egocua.schema import load_tasks
from egocua.selection import load_selection, task_ids as selected_task_ids
from egocua.sources import iter_manifests
from egocua.workspace import read_json_strict, write_json_atomic


PROPOSALS = {
    "egomart": Path("data/sources/amazon_landing_proposals.json"),
    "egomedia": Path("data/sources/youtube_landing_proposals.json"),
}
CAPTURES = {
    "egomart": (Path("data/sources/upstream_search_snapshots/amazon"), "asin"),
    "egomedia": (Path("data/sources/upstream_search_snapshots/youtube"), "youtube_id"),
}
OVERRIDES = Path("data/sources/landing_review_overrides.json")
ENTITY_INDEX = Path("data/sources/upstream_entity_index.json")
OUTPUT = Path("data/sources/landing_semantic_reviews.json")
ASSESSMENT_KEYS = (
    "useful",
    "scene_update_website_consistent",
    "continuous_change",
    "verifiable_from_public_metadata",
)


def _captured_ids(root: Path, site: str, task_id: str) -> set[str]:
    directory, key = CAPTURES[site]
    ids: set[str] = set()
    for path in (root / directory).glob(f"{task_id}--*.json"):
        payload = read_json_strict(path)
        ids.update(str(row[key]) for row in payload.get("results", []) if row.get(key))
    return ids


def _replace_strings(value: Any, replacements: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {
            key: _replace_strings(item, replacements) for key, item in value.items()
        }
    return value


def _replace_resolution_binding(
    resolution: Any,
    old_entity_id: str,
    new_entity_id: str,
) -> Any:
    """Keep a frozen search window only when it already proves the new landing."""

    if not isinstance(resolution, dict):
        return resolution
    window = resolution.get("window_entity_ids")
    if old_entity_id == new_entity_id or (
        isinstance(window, list) and new_entity_id in window
    ):
        return resolution
    raise ValueError(
        "reviewed landing is outside the frozen result window; rebuild this task's "
        "search neighborhood instead of rewriting captured evidence"
    )


def _reviewed_landings(
    proposal: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, str]:
    proposal_landings = {
        str(row["role"]): str(row.get("asin") or row.get("youtube_id"))
        for row in proposal.get("role_landings", [])
    }
    override_landings = {
        str(role): str(upstream_id)
        for role, upstream_id in (override.get("role_landings") or {}).items()
    }
    # A semantic repair may rename or remove roles. In that case the old
    # proposal is historical evidence, not live truth, and must not leak
    # stale role names back into the repaired blueprint.
    if override.get("replace_role_landings") is True:
        return override_landings
    return {**proposal_landings, **override_landings}


def _require_explicit_repair_override(
    *,
    task_id: str,
    role: str,
    current: str,
    proposed: str,
    prior_review_status: str,
    explicit_role_overrides: set[str],
) -> None:
    if (
        prior_review_status == "coding_agent_accepted"
        and current != proposed
        and role not in explicit_role_overrides
    ):
        raise ValueError(
            f"{task_id}: old landing proposal would overwrite repaired role "
            f"{role!r}; record an explicit role_landings override"
        )


def review(
    root: Path,
    *,
    apply: bool,
    requested_task_ids: set[str] | None = None,
) -> dict[str, int]:
    built = {
        str(task["id"]): task
        for task in load_tasks(root / "data/tasks")
        if task["site"] in PROPOSALS
    }
    records = {
        str(record["task_id"]): record
        for _, payload in iter_manifests(root)
        for record in payload.get("tasks", [])
        if isinstance(record, Mapping) and record.get("task_id")
    }
    tasks: dict[str, dict[str, Any]] = {}
    selected = selected_task_ids(load_selection(root))
    selected_set = set(selected)
    if requested_task_ids is not None and not requested_task_ids <= selected_set:
        raise ValueError(
            f"unknown selected task IDs: {sorted(requested_task_ids - selected_set)}"
        )
    for task_id in selected:
        if requested_task_ids is not None and task_id not in requested_task_ids:
            continue
        record = records.get(task_id)
        if record is None:
            continue
        blueprint = (record.get("construct") or {}).get("task_blueprint") or {}
        site = str(
            blueprint.get("site") or built.get(task_id, {}).get("site") or "egomart"
        )
        if site not in PROPOSALS:
            continue
        tasks[task_id] = built.get(task_id) or {
            "id": task_id,
            "track": str(record.get("family") or ""),
            "site": site,
        }
    proposal_rows: dict[str, dict[str, Any]] = {}
    for site, relative in PROPOSALS.items():
        payload = read_json_strict(root / relative)
        for task_id, row in payload["tasks"].items():
            proposal_rows[str(task_id)] = {"site": site, **row}
    overrides = read_json_strict(root / OVERRIDES).get("tasks", {})
    missing = sorted(set(tasks) - set(proposal_rows))
    if missing:
        raise ValueError(f"tasks lack landing proposals: {missing}")

    reviews: dict[str, dict[str, Any]] = {}
    if requested_task_ids is not None and (root / OUTPUT).is_file():
        existing = read_json_strict(root / OUTPUT)
        reviews.update(existing.get("tasks") or {})
    unresolved: list[str] = []
    for task_id in sorted(tasks):
        proposal = proposal_rows[task_id]
        site = str(proposal["site"])
        override = overrides.get(task_id) or {}
        assessment = proposal.get("task_assessment") or {}
        model_pass = all(assessment.get(key) is True for key in ASSESSMENT_KEYS)
        if not model_pass and override.get("decision") != "accept":
            unresolved.append(
                f"{task_id}: {assessment.get('issue') or 'model semantic failure'}"
            )
            continue
        landings = _reviewed_landings(proposal, override)
        captured = _captured_ids(root, site, task_id)
        if not landings or any(
            upstream_id not in captured for upstream_id in landings.values()
        ):
            unresolved.append(
                f"{task_id}: reviewed role landing is outside captured windows"
            )
            continue
        confidence = str(override.get("confidence") or "high")
        if confidence not in {"high", "low"}:
            unresolved.append(f"{task_id}: confidence must be high or low")
            continue
        reviews[task_id] = {
            "decision": "accept",
            "confidence": confidence,
            "needs_human_review": confidence == "low",
            "reviewer": "skill-executing-coding-agent",
            "model_proposal_passed": model_pass,
            "coding_agent_rationale": str(
                override.get("rationale")
                or "模型建议与视频职责、用户变化和公开网站元数据一致。"
            ),
            "role_landings": landings,
            "explicit_role_overrides": sorted(
                str(role) for role in (override.get("role_landings") or {})
            ),
        }
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} task landings remain unresolved:\n"
            + "\n".join(unresolved)
        )

    output = {
        "schema_version": 1,
        "review_policy": (
            "Model proposes; the Skill-executing Coding Agent admits or repairs. "
            "Only low-confidence accepted tasks require one human reviewer."
        ),
        "tasks": reviews,
    }
    write_json_atomic(root / OUTPUT, output)
    scoped_reviews = [reviews[task_id] for task_id in tasks]
    if not apply:
        return {
            "reviewed": len(scoped_reviews),
            "high_confidence": sum(
                row["confidence"] == "high" for row in scoped_reviews
            ),
            "low_confidence": sum(row["confidence"] == "low" for row in scoped_reviews),
            "applied": 0,
        }

    index = read_json_strict(root / ENTITY_INDEX)["sites"]
    touched: dict[Path, dict[str, Any]] = {}
    applied = 0
    for path, payload in iter_manifests(root):
        changed = False
        for record in payload.get("tasks", []):
            if not isinstance(record, dict):
                raise ValueError(f"{path}: task record must be an object")
            task_id = str(record.get("task_id") or "")
            task_review = reviews.get(task_id) if task_id in tasks else None
            if task_review is None:
                continue
            construct = record.get("construct")
            if not isinstance(construct, dict):
                raise ValueError(f"{task_id}: construct must be an object")
            blueprint = construct.get("task_blueprint")
            if not isinstance(blueprint, dict):
                raise ValueError(f"{task_id}: task_blueprint must be an object")
            role_ids = blueprint.get("environment_role_ids")
            if not isinstance(role_ids, dict):
                raise ValueError(f"{task_id}: environment_role_ids must be an object")
            prior_review = construct.get("landing_semantic_review") or {}
            explicit_role_overrides = set(task_review["explicit_role_overrides"])
            replacements: dict[str, str] = {}
            for role, upstream_id in task_review["role_landings"].items():
                if role not in role_ids:
                    raise ValueError(f"{task_id}: reviewed unknown role {role}")
                entity_id = str(index[tasks[task_id]["site"]][upstream_id])
                _require_explicit_repair_override(
                    task_id=task_id,
                    role=role,
                    current=str(role_ids[role]),
                    proposed=entity_id,
                    prior_review_status=str(prior_review.get("status") or ""),
                    explicit_role_overrides=explicit_role_overrides,
                )
                replacements[str(role_ids[role])] = entity_id
                role_ids[role] = entity_id
            construct["decision"] = _replace_strings(
                construct.get("decision", {}),
                replacements,
            )
            # Keep the last valid resolution as a bootstrap for construction.
            # The upstream-window builder replaces it task-locally after the new
            # semantic bindings compile; deleting it creates a circular dependency
            # between task construction and result-window generation.
            if construct.get("catalog_resolution"):
                resolution = construct["catalog_resolution"]
                for old_entity_id, new_entity_id in replacements.items():
                    resolution = _replace_resolution_binding(
                        resolution,
                        old_entity_id,
                        new_entity_id,
                    )
                construct["catalog_resolution"] = resolution
            construct["landing_semantic_review"] = {
                "status": "coding_agent_accepted",
                "confidence": task_review["confidence"],
                "review_reference": OUTPUT.as_posix(),
            }
            changed = True
            applied += 1
        if changed:
            touched[path] = payload
    for path, payload in touched.items():
        write_json_atomic(path, payload)
    return {
        "reviewed": len(scoped_reviews),
        "high_confidence": sum(row["confidence"] == "high" for row in scoped_reviews),
        "low_confidence": sum(row["confidence"] == "low" for row in scoped_reviews),
        "applied": applied,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--task", action="append")
    args = parser.parse_args()
    print(
        json.dumps(
            review(
                Path(args.root).resolve(),
                apply=args.apply,
                requested_task_ids=set(args.task) if args.task else None,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
