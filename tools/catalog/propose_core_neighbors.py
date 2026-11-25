"""Propose semantically close real-listing neighbors for every task window.

Qwen proposes supported candidates and explicit rejection predicates. It cannot admit
the task: the proposal remains pending until the Coding Agent reviews the scene,
change contract, website metadata, and every selected neighbor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import requests
from dotenv import load_dotenv

from egocua.environment import load_site_data
from egocua.run_ledger import canonical_sha256
from egocua.schema import load_tasks
from egocua.sources import iter_manifests
from egocua.truth import Truth
from egocua.workspace import write_json_atomic

OUTPUT = Path("data/sources/core_neighbor_proposals.json")
MODEL = "qwen3.5-flash"
PREDICATES = {
    "old_goal_only",
    "scene_evidence_mismatch",
    "later_change_mismatch",
    "website_detail_mismatch",
    "wrong_method_same_object",
    "wrong_object_same_method",
    "wrong_variant_or_branch",
}


def _title(entity: Mapping[str, Any]) -> str:
    return str(entity.get("name") or entity.get("title") or "")


def _public_summary(entity: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        key: entity.get(key)
        for key in (
            "brand",
            "kind",
            "price",
            "attributes",
            "channel_title",
            "duration_seconds",
            "summary",
            "details",
            "chapters",
            "address",
            "facts",
            "location",
        )
        if entity.get(key) not in (None, "", [], {})
    }
    # Keep the model prompt bounded while preserving the website facts used for
    # detail-page discrimination.
    for key in ("summary",):
        value = fields.get(key)
        if isinstance(value, str):
            fields[key] = value[:500]
    chapters = fields.get("chapters")
    if isinstance(chapters, list):
        fields["chapters"] = chapters[:8]
    return fields


def _records(root: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for _, payload in iter_manifests(root):
        for row in payload.get("tasks", []):
            if isinstance(row, Mapping) and row.get("task_id"):
                result[str(row["task_id"])] = row
    return result


def _graft_role_entity_ids(
    record: Mapping[str, Any],
    field: str,
) -> set[str]:
    """Resolve current Graft semantics from the manifest, not stale compiled tasks."""

    blueprint = record["construct"]["task_blueprint"]
    role_ids = blueprint["environment_role_ids"]
    equivalents = blueprint.get("equivalent_role_ids") or {}
    role_sets = blueprint[field]
    result: set[str] = set()
    for roles in role_sets:
        for role in roles:
            result.add(str(role_ids[role]))
            result.update(str(value) for value in equivalents.get(role, []))
    return result


def _payload(
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    resolution = record["construct"]["catalog_resolution"]
    rounds = resolution["query_rounds"]
    before = list(rounds[0]["result_entity_ids"])
    after = list(rounds[1]["result_entity_ids"])
    before_set = set(before)
    after_set = set(after)
    if record.get("family") == "graft" and not task["goal"].get("unit_equivalences"):
        initial_ids = _graft_role_entity_ids(record, "initial_goal_roles")
        final_answer_ids = _graft_role_entity_ids(record, "final_goal_roles")
        tempting_ids = initial_ids - final_answer_ids
        current_instructions = record["construct"]["task_blueprint"]["instructions"]
    else:
        truth = Truth.of(task)
        initial_ids = set(truth.initial_referent_ids)
        final_answer_ids = set(truth.final_referent_ids)
        tempting_ids = set(truth.tempting_referent_ids)
        current_instructions = task["instructions"]

    def entity_row(entity_id: str) -> dict[str, Any]:
        entity = by_id[entity_id]
        return {
            "entity_id": entity_id,
            "title": _title(entity),
            "public_detail": _public_summary(entity),
        }

    def semantic_row(entity_id: str) -> dict[str, Any]:
        entity = by_id[entity_id]
        return {
            "title": _title(entity),
            "public_detail": _public_summary(entity),
        }

    candidates = []
    for entity_id in resolution["window_entity_ids"]:
        if entity_id in final_answer_ids:
            continue
        row = entity_row(entity_id)
        row.update(
            {
                "before_rank": before.index(entity_id) + 1
                if entity_id in before_set
                else None,
                "after_rank": after.index(entity_id) + 1
                if entity_id in after_set
                else None,
            }
        )
        candidates.append(row)
    construct = record["construct"]
    return {
        "task_id": task["id"],
        "site": task["site"],
        "track": task["track"],
        "instructions": current_instructions,
        "source_supported_claims": record.get("supported_claims", []),
        "change_contract": {
            key: construct.get(key)
            for key in (
                "belief_before",
                "belief_after",
                "required_behavior_change",
                "no_update_failure",
                "overupdate_failure",
                "visual_dependency_claim",
            )
        },
        "queries": [{"stage": row["stage"], "query": row["query"]} for row in rounds],
        "semantic_entities": {
            "initial": [semantic_row(value) for value in sorted(initial_ids)],
            "final": [semantic_row(value) for value in sorted(final_answer_ids)],
            "tempting": [semantic_row(value) for value in sorted(tempting_ids)],
        },
        "candidate_rows": candidates,
    }


def _prompt(payload: Mapping[str, Any], feedback: str = "") -> str:
    return (
        """You propose close, real website distractors for one multimodal browser task.
Return strict JSON with exactly two keys:
{"task_assessment": {...}, "core_neighbors": [...]}

task_assessment has:
- useful_in_real_life: boolean
- scene_update_website_consistent: boolean
- continuous_change: boolean
- deterministically_verifiable: boolean
- issue: one concise sentence, empty only when all four are true

Interpret continuous_change by track: Graft uses the later utterance, Refresh uses
the later released frames, and Clarify uses the user's clarification reply.  The
change is valid when that later evidence narrows, replaces, retains, or adds to the
earlier live candidate set; it does not require motion inside the website.

core_neighbors contains only defensible distinct rows, with no fixed count. Each row has:
- entity_id: copied from candidate_rows
- why_plausible: why a capable agent could reasonably consider it
- why_rejected: the concrete scene, update, or website-detail fact that rules it out
- failure_predicates: one or two values from this closed list:
  old_goal_only, scene_evidence_mismatch, later_change_mismatch,
  website_detail_mismatch, wrong_method_same_object,
  wrong_object_same_method, wrong_variant_or_branch

Rules:
- Select genuine near-neighbors, never merely the first available rows.
- A core neighbor should match at least two of scene, user change, and website category,
  then fail on a concrete third fact.
- Use only predicates supported by each row's concrete facts. Repeated predicates are valid.
- Do not select a semantic entity or invent any fact not present in the payload.
- Prefer candidates that require opening details to reject.
- If no defensible close neighbors exist, return an empty list and explain the evidence gap; never pad the list.
- The model only proposes. Do not claim human or Coding-Agent approval.

Previous-attempt validation feedback (empty on the first attempt):
"""
        + feedback
        + """

Task payload:
"""
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse(
    text: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("model returned no JSON object")
    row = json.loads(match.group(0))
    if set(row) != {"task_assessment", "core_neighbors"}:
        raise ValueError("unexpected proposal schema")
    assessment = row["task_assessment"]
    required_assessment = {
        "useful_in_real_life",
        "scene_update_website_consistent",
        "continuous_change",
        "deterministically_verifiable",
        "issue",
    }
    if not isinstance(assessment, dict) or set(assessment) != required_assessment:
        raise ValueError("invalid task_assessment")
    for key in required_assessment - {"issue"}:
        if not isinstance(assessment[key], bool):
            raise ValueError(f"{key} must be boolean")
    if not isinstance(assessment["issue"], str):
        raise ValueError("issue must be a string")
    neighbors = row["core_neighbors"]
    if not isinstance(neighbors, list):
        raise ValueError("core neighbors must be a list")
    allowed_ids = {str(item["entity_id"]) for item in payload["candidate_rows"]}
    seen: set[str] = set()
    for neighbor in neighbors:
        if not isinstance(neighbor, dict) or set(neighbor) != {
            "entity_id",
            "why_plausible",
            "why_rejected",
            "failure_predicates",
        }:
            raise ValueError("invalid core-neighbor schema")
        entity_id = str(neighbor["entity_id"]).strip()
        neighbor["entity_id"] = entity_id
        if entity_id not in allowed_ids or entity_id in seen:
            raise ValueError("core neighbor is outside the real non-answer window")
        seen.add(entity_id)
        if (
            not str(neighbor["why_plausible"]).strip()
            or not str(neighbor["why_rejected"]).strip()
        ):
            raise ValueError("core neighbor lacks a concrete explanation")
        reasons = neighbor["failure_predicates"]
        if (
            not isinstance(reasons, list)
            or not 1 <= len(reasons) <= 2
            or not set(reasons) <= PREDICATES
        ):
            raise ValueError("invalid failure predicates")
    return row


def _propose(
    payload: dict[str, Any],
    endpoint: str,
    api_key: str,
) -> tuple[str, dict[str, Any]]:
    feedback = ""
    last_error = "model produced no valid proposal"
    for attempt in range(1, 4):
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": _prompt(payload, feedback)},
                ],
                "temperature": 0 if attempt == 1 else 0.2,
                "max_tokens": 1800,
                "enable_thinking": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        try:
            proposal = _parse(
                str(response.json()["choices"][0]["message"]["content"]),
                payload,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            feedback = (
                f"Your previous JSON failed the deterministic validator: {last_error}. "
                "Use only entity_id values copied exactly from candidate_rows, never from "
                "semantic_entities, and explain each rejection with supported facts."
            )
            continue
        return str(payload["task_id"]), {
            **proposal,
            "model": MODEL,
            "attempts": attempt,
            "input_sha256": canonical_sha256(payload),
            "status": "proposed_pending_coding_agent_review",
        }
    raise ValueError(last_error)


def build(
    root: Path,
    task_ids: set[str] | None,
    concurrency: int,
    *,
    force: bool,
) -> dict[str, int]:
    if not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be between 1 and 4")
    load_dotenv(root / ".env", override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"

    tasks = load_tasks(root / "data/tasks")
    if task_ids is not None:
        tasks = [task for task in tasks if str(task["id"]) in task_ids]
    records = _records(root)
    sites = {
        site: {
            str(entity["entity_id"]): entity
            for entity in load_site_data(root, site)["entities"]
        }
        for site in {str(task["site"]) for task in tasks}
    }
    payloads = [
        _payload(task, records[str(task["id"])], sites[str(task["site"])])
        for task in tasks
    ]
    output_path = root / OUTPUT
    document: dict[str, Any] = (
        json.loads(output_path.read_text())
        if output_path.is_file()
        else {"schema_version": 1, "model": MODEL, "tasks": {}}
    )
    pending = [
        payload
        for payload in payloads
        if force
        or document["tasks"].get(str(payload["task_id"]), {}).get("input_sha256")
        != canonical_sha256(payload)
    ]
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_propose, payload, endpoint, api_key): str(
                payload["task_id"]
            )
            for payload in pending
        }
        for future in as_completed(futures):
            try:
                task_id, row = future.result()
            except Exception as exc:  # resumable: failures are absent from the output
                failed += 1
                print(
                    f"failed {futures[future]} {type(exc).__name__}: {exc}", flush=True
                )
                continue
            document["tasks"][task_id] = row
            write_json_atomic(output_path, document)
            completed += 1
            print(f"proposed {completed}/{len(pending)} {task_id}", flush=True)
    return {
        "requested": len(payloads),
        "reused": len(payloads) - len(pending),
        "proposed": completed,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                Path(args.root).resolve(),
                set(args.task_id) if args.task_id else None,
                args.concurrency,
                force=args.force,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
