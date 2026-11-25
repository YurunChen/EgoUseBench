"""Propose real Amazon entities for EgoMart semantic roles.

This is a constrained model proposal, never an admission decision. Every ASIN is
validated against the captured first-page window and must later be accepted or
repaired by the Skill-executing Coding Agent.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


from egocua.schema import load_tasks
from tools.catalog.landing_proposals import LandingProposals, role_sets, source_records
from egocua.construction import resolve_authored_record
from egocua.selection import load_selection, task_ids as selected_task_ids


OUTPUT = Path("data/sources/amazon_landing_proposals.json")
CAPTURES = Path("data/sources/upstream_search_snapshots/amazon")


def _payload(
    root: Path,
    task: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(task["id"])
    record = resolve_authored_record(record, root=root)
    blueprint = record["construct"]["task_blueprint"]
    initial, final = role_sets(task, blueprint)
    active_roles = initial | final
    if task["track"] == "refresh" and blueprint.get("later_distractor_role"):
        active_roles.add(str(blueprint["later_distractor_role"]))

    before = json.loads(
        (root / CAPTURES / f"{task_id}--before_change.json").read_text()
    )
    after = json.loads((root / CAPTURES / f"{task_id}--after_change.json").read_text())
    before_ids = {str(row["asin"]) for row in before["results"]}
    after_ids = {str(row["asin"]) for row in after["results"]}
    initial_only = initial - final
    final_only = final - initial
    role_stage_overrides = {
        str(role): str(stage)
        for role, stage in blueprint.get("catalog_role_stages", {}).items()
    }

    requirements: list[dict[str, Any]] = []
    allowed_by_role: dict[str, set[str]] = {}
    for role in sorted(active_roles):
        if role in role_stage_overrides:
            stage = role_stage_overrides[role]
        elif task["track"] == "clarify":
            stage = "after_change" if role in final else "before_change"
        elif role in initial and role in final:
            if task["track"] == "refresh":
                stage = "before_change"
            elif final_only:
                stage = "before_change"
            elif initial_only:
                stage = "after_change"
            else:
                stage = "both"
        elif role in initial:
            stage = "before_change"
        elif role in final:
            stage = "after_change"
        else:
            stage = "after_change"
        if stage not in {"before_change", "after_change", "both"}:
            raise ValueError(f"{task_id}/{role}: invalid catalog stage {stage!r}")
        allowed = (
            before_ids & after_ids
            if stage == "both"
            else before_ids
            if stage == "before_change"
            else after_ids
        )
        if not allowed:
            raise ValueError(f"{task_id}/{role}: no real candidates for {stage}")
        requirements.append(
            {
                "role": role,
                "required_stage": stage,
                "is_initial": role in initial,
                "is_final": role in final,
            }
        )
        allowed_by_role[role] = allowed

    def rows(capture: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                key: row.get(key)
                for key in (
                    "rank",
                    "asin",
                    "title",
                    "price",
                    "rating",
                    "review_count",
                    "sponsored",
                    "image_url",
                    "page_url",
                )
            }
            for row in capture["results"]
        ]

    before_rows = rows(before)
    after_rows = rows(after)
    index = {str(row["asin"]): row for row in [*before_rows, *after_rows]}
    if task["track"] == "graft":
        instructions = blueprint["instructions"]
    elif task["track"] == "refresh":
        instructions = {"base": blueprint["instruction"]}
    else:
        instructions = {
            "ambiguous": blueprint["ambiguous_instruction"],
            "oracle": blueprint["oracle_instruction"],
        }
    return {
        "proposal_policy_version": "scene-user-website-separation-v2",
        "task_id": task_id,
        "track": task["track"],
        "instructions": instructions,
        "source_supported_claims": record.get("supported_claims", []),
        "change_contract": {
            key: record["construct"].get(key)
            for key in (
                "belief_before",
                "belief_after",
                "required_behavior_change",
                "visual_dependency_claim",
            )
        },
        "role_requirements": requirements,
        "allowed_candidates_by_role": {
            role: [index[asin] for asin in sorted(allowed)]
            for role, allowed in allowed_by_role.items()
        },
        "before_query": before["query"],
        "after_query": after["query"],
        "before_results": before_rows,
        "after_results": after_rows,
    }


def _prompt(payload: Mapping[str, Any], feedback: str) -> str:
    return (
        """Map one multimodal shopping task's semantic roles to captured real
Amazon results. Return strict JSON only:
{"task_assessment":{"useful":true,"scene_update_website_consistent":true,
"continuous_change":true,"verifiable_from_public_metadata":true,"issue":""},
"role_landings":[{"role":"...","asin":"...","reason":"..."}]}

Rules:
- Return exactly one row for every role_requirements role.
- Copy ASINs only from that role's allowed_candidates_by_role.
- Keep evidence responsibilities separate. The released scene proves physical
  sequence, left/right position, and what the wearer handled; the Amazon listing
  does not need to repeat those episode facts. Do not fail a task merely because a
  product title cannot say that it was "used first" or stood "on the left".
- The public title/details must prove the catalog identity and every website-only
  constraint after the object has been recognized from the scene: product type,
  brand/model when required, size, compatibility, pack count, material, or variant.
  A generic listing is not enough for a scene-readable brand or exact model.
- Preserve add/replace/retain semantics. Search rank alone is never evidence.
- Sponsored rows may be distractors but cannot be a standard answer.
- Use distinct products for distinct roles.
- Set an assessment false when the scene, change, or public metadata is
  insufficient. Still return the best schema-valid candidates so the Coding Agent
  can repair the task.

Validator feedback:
"""
        + feedback
        + "\n\nTask:\n"
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse(text: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object")
    row = json.loads(match.group(0))
    if set(row) != {"task_assessment", "role_landings"}:
        raise ValueError("unexpected schema")
    assessment = row["task_assessment"]
    assessment_keys = {
        "useful",
        "scene_update_website_consistent",
        "continuous_change",
        "verifiable_from_public_metadata",
        "issue",
    }
    if not isinstance(assessment, dict) or set(assessment) != assessment_keys:
        raise ValueError("invalid assessment")
    for key in assessment_keys - {"issue"}:
        if not isinstance(assessment[key], bool):
            raise ValueError(f"{key} must be boolean")
    if not isinstance(assessment["issue"], str):
        raise ValueError("issue must be a string")

    requirements = {
        str(item["role"]): str(item["required_stage"])
        for item in payload["role_requirements"]
    }
    before = {str(item["asin"]): item for item in payload["before_results"]}
    after = {str(item["asin"]): item for item in payload["after_results"]}
    landings = row["role_landings"]
    if not isinstance(landings, list) or {
        str(item.get("role")) for item in landings if isinstance(item, Mapping)
    } != set(requirements):
        raise ValueError("role_landings must cover every active role exactly")
    seen_roles: set[str] = set()
    seen_asins: set[str] = set()
    for item in landings:
        if not isinstance(item, dict) or set(item) != {"role", "asin", "reason"}:
            raise ValueError("invalid landing row")
        role = str(item["role"])
        asin = str(item["asin"])
        if role in seen_roles or asin in seen_asins:
            raise ValueError("roles and ASINs must be unique")
        seen_roles.add(role)
        seen_asins.add(asin)
        stage = requirements[role]
        if stage in {"before_change", "both"} and asin not in before:
            raise ValueError(f"{role} is outside the before window")
        if stage in {"after_change", "both"} and asin not in after:
            raise ValueError(f"{role} is outside the after window")
        source = after.get(asin) or before.get(asin) or {}
        if source.get("sponsored"):
            raise ValueError(f"{role} selects a sponsored result")
        if not str(item["reason"]).strip():
            raise ValueError(f"{role} lacks a reason")
    if (
        not all(assessment[key] for key in assessment_keys - {"issue"})
        and not assessment["issue"].strip()
    ):
        raise ValueError("failed assessment requires an issue")
    return row


def build(
    root: Path,
    requested_task_ids: set[str] | None,
    concurrency: int,
) -> dict[str, int]:
    records = source_records(root)
    built = {str(task["id"]): task for task in load_tasks(root / "data/tasks")}
    selected = set(selected_task_ids(load_selection(root)))
    requested = requested_task_ids if requested_task_ids is not None else selected
    tasks: list[dict[str, Any]] = []
    for task_id in sorted(requested):
        record = records.get(task_id)
        if record is None:
            continue
        blueprint = (record.get("construct") or {}).get("task_blueprint") or {}
        if str(blueprint.get("site") or "egomart") != "egomart":
            continue
        task = built.get(task_id)
        if task is None:
            track = str(record.get("family") or "")
            if track not in {"graft", "refresh", "clarify"}:
                raise ValueError(f"{task_id}: cannot infer task track")
            task = {"id": task_id, "track": track, "site": "egomart"}
            if track == "graft":
                task["graft"] = {"transition": record.get("subtype")}
        tasks.append(task)
    payloads = [_payload(root, task, records[str(task["id"])]) for task in tasks]
    return LandingProposals(OUTPUT, _prompt, _parse).build(root, payloads, concurrency)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                Path(args.root).resolve(),
                set(args.task_id) if args.task_id else None,
                args.concurrency,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
