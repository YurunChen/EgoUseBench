from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from egocua.adapters.openai_requests import OpenAIRequestsTransport
from egocua.construction import opaque_item_id
from egocua.environment import entity_visual_path, load_site_data
from egocua.schema import acceptable_states_of
from egocua.workspace import resolve_within
from tools.validate_pilot_selection import validate_selection


PROMPT_VERSION = "pilot-input-target-and-glass-use-preaudit-v8"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: top level must be an object")
    return payload


def _selected_ids(selection: dict[str, Any]) -> list[str]:
    return [
        task_id
        for task_ids in selection["primary"].values()
        for task_id in task_ids
    ]


def _tasks(selection: dict[str, Any], workspace: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for manifest_name in selection["source_manifests"]:
        manifest_path = resolve_within(
            workspace, manifest_name, context="model preaudit source manifest"
        )
        for task in _read_json(manifest_path)["tasks"]:
            index[task["task_id"]] = task
    return index


def _compiled_tasks(workspace: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    index: dict[str, tuple[dict[str, Any], Path]] = {}
    for path in sorted((workspace / "data" / "tasks").glob("*/*.json")):
        task = _read_json(path)
        index[task["id"]] = (task, path)
    return index


def _source_task_for_compiled(
    compiled_task: dict[str, Any], workspace: Path,
) -> tuple[dict[str, Any], str]:
    manifest_name = str(compiled_task["source"]["dossier"])
    manifest_path = resolve_within(
        workspace, manifest_name, context="model preaudit source manifest"
    )
    matches = [
        record for record in _read_json(manifest_path).get("tasks", [])
        if isinstance(record, dict) and record.get("task_id") == compiled_task["id"]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{compiled_task['id']}: expected one source record in {manifest_name}"
        )
    return matches[0], manifest_name


def _manifest_digests(
    selection: dict[str, Any], workspace: Path
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for manifest_name in selection["source_manifests"]:
        manifest_path = resolve_within(
            workspace, manifest_name, context="model preaudit source manifest"
        )
        records.append(
            {
                "path": manifest_name,
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
        )
    return records


def _rubric(family: str) -> str:
    if family == "graft":
        return (
            "First check that B arrives while the same physical activity remains underway and "
            "changes a pending web commitment; an ordinary later rating, price, or sorting filter "
            "is not sufficient Glass-Use motivation. "
            "Check that the eight frames ground every physical referent needed by instruction A; "
            "that A and B define a real change in final web state; that B preserves, constrains, "
            "or relates to A as claimed; and that no unseen exact model identity is asserted. "
            "Catalog-only rating, compatibility, format, and final-state facts may remain downstream "
            "dependencies and must not be treated as visible facts."
        )
    if family == "refresh":
        return (
            "First check that V1 is later evidence from the same continuous physical event, so a "
            "single task-start image cannot preserve the measured retain/revise decision. "
            "Treat V0 and V1 as ordered packets. Check that V0 grounds the initial target, V1 grounds "
            "the later target or distractor, the retain/replace wording makes the correct final referent "
            "deterministic, and the visual change would require a different online decision."
        )
    return (
        "First check that the ambiguity exists at a candidate-commitment point in the wearer's "
        "ongoing physical activity; flag a generic two-object ambiguity that is unchanged after "
        "uploading one static photograph. "
        "Check that exactly two candidates satisfy the ambiguous instruction, neither is a natural "
        "default, both are visually grounded as distinct site products, the legal slot and "
        "canonical answer select exactly one, and no third candidate remains plausible."
    )


def _proposed_entities(
    source_task: dict[str, Any], task: dict[str, Any], workspace: Path
) -> list[dict[str, Any]]:
    entities = {
        entity["entity_id"]: entity
        for entity in load_site_data(workspace, task["site"])["entities"]
    }
    roles: dict[str, set[str]] = {}

    def add(item_id: str, role: str) -> None:
        roles.setdefault(item_id, set()).add(role)

    for state in acceptable_states_of(task["goal"], f"{task['id']}/goal"):
        for entity_id in state["selected_entity_ids"]:
            add(entity_id, "final_acceptable")
    if task["track"] == "graft":
        for state in acceptable_states_of(
            task["graft"]["initial_goal"], f"{task['id']}/initial_goal"
        ):
            for entity_id in state["selected_entity_ids"]:
                add(entity_id, "initial_acceptable")
    elif task["track"] == "refresh":
        add(task["refresh"]["initial_entity_id"], "initial_target")
        blueprint = source_task["construct"]["task_blueprint"]
        for field, role_name in (
            ("initial_target_role", "initial_observed"),
            ("final_target_role", "final_observed"),
            ("later_distractor_role", "later_distractor"),
        ):
            role = blueprint.get(field)
            item_id = opaque_item_id(task["id"], role) if role else None
            if item_id is not None and item_id in entities:
                add(item_id, role_name)
    else:
        for item_id in task["clarification"]["candidate_ids"]:
            add(item_id, "visual_candidate")

    evidence: list[dict[str, Any]] = []
    for item_id in sorted(roles):
        entity = entities[item_id]
        image_path = entity_visual_path(task["site"], entity)
        record = {
            "entity_id": item_id,
            "roles": sorted(roles[item_id]),
            "title": entity.get("title") or entity.get("name") or item_id,
            "attributes": entity.get("attributes", {}),
            "public_metadata": {
                key: value for key, value in entity.items()
                if key not in {
                    "entity_id", "source", "facts", "image_path",
                    "thumbnail_path", "photo_path",
                }
            },
        }
        if isinstance(image_path, str) and image_path:
            resolved = resolve_within(workspace, image_path, context="proposed site entity image")
            record.update({
                "image_path": image_path,
                "image_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
            })
        evidence.append(record)
    return evidence


def _content(
    task: dict[str, Any], compiled_task: dict[str, Any],
    proposed_entities: list[dict[str, Any]], workspace: Path,
) -> list[dict[str, Any]]:
    construct = task["construct"]
    summary = {
        "task_id": task["task_id"],
        "family": task["family"],
        "subtype": task["subtype"],
        "supported_claims": task.get("supported_claims", []),
        "research_question": construct.get("research_question"),
        "required_behavior_change": construct.get("required_behavior_change"),
        "visual_dependency_claim": construct.get("visual_dependency_claim"),
        "task_blueprint": construct.get("task_blueprint"),
        "released_instructions": compiled_task["instructions"],
        "goal": compiled_task["goal"],
        "track_contract": compiled_task.get(compiled_task["track"]),
    }
    prompt = f"""You are independently preauditing the exact ordered images that an egocentric web-agent benchmark would deliver. This is an advisory image-input audit, not human approval and not an end-to-end web run.

Task construction:
{json.dumps(summary, ensure_ascii=False, indent=2)}

Family-specific rubric:
{_rubric(task['family'])}

This preaudit does not test whether the frozen site's buttons, cart, save state, or submit action work; those are replayed separately through the production runtime and checker. Do not set `pass=false`, raise risk, or add an issue merely because those runtime effects cannot be proven from images. Put genuinely relevant website facts in `site_product_dependencies`, and reserve `issues` for a concrete inconsistency, unsupported visual claim, or construct-validity problem visible in the supplied evidence.

Judge only from the supplied images, task text, and frozen site listings. Do not use filenames, annotations, target labels, or outside knowledge as visual evidence. Product roles describe the benchmark contract: `final_acceptable` is the intended target, while a Clarify product labeled only `visual_candidate` is the deliberate alternative, not another proposed target. These labels identify what claim to verify but do not prove that the claim is visually correct. Exact SKU identity is required only for exact_instance granularity; category, brand, and variant tasks need only the corresponding visible match. For Clarify, decide whether a third product is plausible only after applying every released instruction constraint, including format, quantity, and deliverable type; broad same-category similarity alone is insufficient. Website-only rating, price, format, and compatibility facts are valid downstream constraints and are not reasons to fail when the physical referent and proposed site products are otherwise grounded. Reject unsupported temporal order, candidate-count ambiguity, a proposed product that does not visually match the released referent at the declared granularity, or a construction where the intervention would not change the evaluated web action.

Before those consistency checks, apply the Benchmark's construct test. The task must measure when an Agent may commit during a user-initiated, short, hands-free egocentric session while the user is still manipulating, moving through, or inspecting the physical scene. Ask whether one task-start photograph plus one complete instruction would preserve the same decision, timing, and web commitment. If yes, identify that as a core-benchmark problem. A sentence saying that the user's hands are busy is not evidence unless the ordered source images support the activity. Do not treat internal consistency or model solvability as proof of Glass-Use necessity.

Return JSON only with these keys:
- pass: boolean
- risk: "low" | "medium" | "high"
- image_evidence_sufficient: boolean
- behavior_change_required: boolean
- site_product_mapping_plausible: boolean
- observed_roles: array of short descriptions
- issues: array of concrete problems grounded in the supplied inputs
- site_product_dependencies: array of facts that require the later frozen site
"""
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for packet in task["packets"]:
        content.append(
            {
                "type": "text",
                "text": f"Ordered packet {packet['id']} ({len(packet['frame_paths'])} frames):",
            }
        )
        for frame_path in packet["frame_paths"]:
            path = resolve_within(workspace, frame_path, context="model preaudit frame")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "media_type": "image/jpeg",
                        "data": base64.b64encode(path.read_bytes()).decode(),
                    },
                }
            )
    for entity in proposed_entities:
        content.append(
            {
                "type": "text",
                "text": (
                    "Proposed site entity to verify: "
                    f"entity_id={entity['entity_id']} roles={entity['roles']} "
                    f"title={entity['title']} "
                    f"public_metadata={json.dumps(entity['public_metadata'], ensure_ascii=False)}"
                ),
            }
        )
        if "image_path" in entity:
            image_path = resolve_within(
                workspace, entity["image_path"], context="proposed site entity image"
            )
            content.append({
                "type": "image",
                "source": {
                    "media_type": mimetypes.guess_type(image_path.name)[0] or "image/jpeg",
                    "data": base64.b64encode(image_path.read_bytes()).decode(),
                },
            })
    audit_path = resolve_within(
        workspace,
        f"artifacts/reports/site-audits/{task['task_id']}.jpg",
        context="joint task/site audit",
    )
    content.extend([
        {
            "type": "text",
            "text": (
                "Joint audit of the exact released observations and the task-referenced listings "
                f"inside the fixed {compiled_task['site']} site. Verify the proposed targets and plausible "
                "alternatives rather than trusting labels:"
            ),
        },
        {
            "type": "image",
            "source": {
                "media_type": "image/jpeg",
                "data": base64.b64encode(audit_path.read_bytes()).decode(),
            },
        },
    ])
    return content


def _parse_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines)
    payload = json.loads(stripped)
    if not isinstance(payload, dict) or not isinstance(payload.get("pass"), bool):
        raise ValueError("model response must be a JSON object with boolean pass")
    return payload


def _summary(results: list[dict[str, Any]]) -> dict[str, int | str]:
    completed = sum(result.get("status") == "completed" for result in results)
    passed = sum(result.get("verdict", {}).get("pass") is True for result in results)
    failed = sum(result.get("verdict", {}).get("pass") is False for result in results)
    errors = sum(result.get("status") == "error" for result in results)
    return {
        "status": "complete" if errors == 0 else "partial",
        "requested": len(results),
        "completed": completed,
        "passed": passed,
        "failed": failed,
        "errors": errors,
    }


def _audit_one(
    task: dict[str, Any],
    compiled_task: dict[str, Any],
    compiled_path: Path,
    workspace: Path,
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> dict[str, Any]:
    transport = OpenAIRequestsTransport(
        api_key=api_key, base_url=base_url, timeout_seconds=240
    )
    try:
        proposed_entities = _proposed_entities(task, compiled_task, workspace)
        response = transport(
            {
                "model": model,
                "system": (
                    "Be conservative and evidence-bound. Passing means the task is suitable for "
                    "continued construction, not that it is release-ready."
                ),
                "messages": [{
                    "role": "user",
                    "content": _content(
                        task, compiled_task, proposed_entities, workspace
                    ),
                }],
                "max_tokens": 1400,
            }
        )
        text = response["content"][0]["text"]
        verdict = _parse_response(text)
        return {
            "task_id": task["task_id"],
            "family": task["family"],
            "subtype": task["subtype"],
            "status": "completed",
            "compiled_task": {
                "path": compiled_path.relative_to(workspace).as_posix(),
                "sha256": hashlib.sha256(compiled_path.read_bytes()).hexdigest(),
            },
            "site_audit": {
                "path": f"artifacts/reports/site-audits/{task['task_id']}.jpg",
                "sha256": hashlib.sha256(
                    (workspace / "artifacts" / "reports" / "site-audits"
                     / f"{task['task_id']}.jpg").read_bytes()
                ).hexdigest(),
            },
            "proposed_entities": proposed_entities,
            "verdict": verdict,
        }
    except Exception as exc:  # keep a full batch auditable when one request fails
        return {
            "task_id": task["task_id"],
            "family": task["family"],
            "subtype": task["subtype"],
            "status": "error",
            "error": str(exc).replace(api_key, "[REDACTED]")[:2000],
        }


def run_preaudit(
    selection_path: str | Path,
    output_path: str | Path,
    *,
    root: str | Path = ".",
    env_path: str | Path = ".env",
    model: str = "gpt-5.4-mini",
    workers: int = 3,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    validate_selection(selection_path, root=workspace)
    selection_file = resolve_within(
        workspace, selection_path, context="model preaudit selection"
    )
    selection = _read_json(selection_file)
    index = _tasks(selection, workspace)
    compiled_index = _compiled_tasks(workspace)
    task_ids = _selected_ids(selection)
    missing_compiled = sorted(set(task_ids) - set(compiled_index))
    if missing_compiled:
        raise ValueError(f"selected tasks are not compiled: {missing_compiled[:5]}")
    credentials = OpenAIRequestsTransport.from_env(env_path)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _audit_one,
                index[task_id],
                compiled_index[task_id][0],
                compiled_index[task_id][1],
                workspace,
                api_key=credentials.api_key,
                base_url=credentials.base_url,
                model=model,
            ): task_id
            for task_id in task_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            verdict = result.get("verdict", {})
            print(
                f"[{completed}/{len(futures)}] {result['task_id']}: "
                f"{result['status']} pass={verdict.get('pass')} risk={verdict.get('risk')}",
                flush=True,
            )

    results.sort(key=lambda record: task_ids.index(record["task_id"]))
    payload = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "selection": {
            "path": str(selection_path),
            "sha256": hashlib.sha256(selection_file.read_bytes()).hexdigest(),
            "source_manifests": _manifest_digests(selection, workspace),
        },
        "interpretation": (
            "Advisory model preaudit only. It may identify candidates for human review but cannot "
            "promote, approve, or define benchmark ground truth."
        ),
        "summary": _summary(results),
        "results": results,
    }
    destination = resolve_within(
        workspace, output_path, context="model preaudit output", must_exist=False
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def run_preaudit_all(
    output_path: str | Path,
    *,
    root: str | Path = ".",
    env_path: str | Path = ".env",
    model: str = "gpt-5.4-mini",
    workers: int = 3,
) -> dict[str, Any]:
    """Advisory multimodal audit for every constructed task, without promoting it."""

    workspace = Path(root).resolve()
    compiled_index = _compiled_tasks(workspace)
    task_ids = sorted(compiled_index)
    source_index: dict[str, dict[str, Any]] = {}
    manifest_names: set[str] = set()
    for task_id in task_ids:
        source_task, manifest_name = _source_task_for_compiled(
            compiled_index[task_id][0], workspace
        )
        source_index[task_id] = source_task
        manifest_names.add(manifest_name)
    credentials = OpenAIRequestsTransport.from_env(env_path)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _audit_one,
                source_index[task_id],
                compiled_index[task_id][0],
                compiled_index[task_id][1],
                workspace,
                api_key=credentials.api_key,
                base_url=credentials.base_url,
                model=model,
            ): task_id
            for task_id in task_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            verdict = result.get("verdict", {})
            print(
                f"[{completed}/{len(futures)}] {result['task_id']}: "
                f"{result['status']} pass={verdict.get('pass')} risk={verdict.get('risk')}",
                flush=True,
            )

    results.sort(key=lambda record: task_ids.index(record["task_id"]))
    payload = {
        "schema_version": 1,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "model": model,
        "scope": "all_constructed_tasks",
        "source_manifests": [
            {
                "path": manifest_name,
                "sha256": hashlib.sha256(
                    resolve_within(
                        workspace,
                        manifest_name,
                        context="model preaudit source manifest",
                    ).read_bytes()
                ).hexdigest(),
            }
            for manifest_name in sorted(manifest_names)
        ],
        "interpretation": (
            "Advisory model preaudit only. It may identify candidates for human review but cannot "
            "promote, approve, or define benchmark ground truth."
        ),
        "summary": _summary(results),
        "results": results,
    }
    destination = resolve_within(
        workspace, output_path, context="model preaudit output", must_exist=False
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an advisory multimodal preaudit over selected or all tasks."
    )
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--selection")
    scope.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.all_tasks:
        payload = run_preaudit_all(
            args.output,
            root=args.root,
            env_path=args.env,
            model=args.model,
            workers=args.workers,
        )
    else:
        payload = run_preaudit(
            args.selection,
            args.output,
            root=args.root,
            env_path=args.env,
            model=args.model,
            workers=args.workers,
        )
    print(json.dumps(payload["summary"], indent=2))
    if payload["summary"]["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
