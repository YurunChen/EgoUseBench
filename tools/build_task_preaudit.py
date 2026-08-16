from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from egocua.environment import entity_visual_path, load_site_data
from egocua.schema import load_tasks
from egocua.workspace import resolve_within, write_json_atomic


INTERNAL_TASK_LANGUAGE = re.compile(
    r"\bsubmit\b|\bfrozen\b|\bV0\b|\bV1\b|"
    r"catalog\s+(?:compatibility\s+|fit\s+)?metadata",
    re.IGNORECASE,
)


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _selected_ids(selection: dict[str, Any]) -> list[str]:
    primary = selection.get("primary")
    if not isinstance(primary, dict):
        raise ValueError("selection.primary must be an object")
    return [task_id for task_ids in primary.values() for task_id in task_ids]


def _task_entity_ids(task: dict[str, Any]) -> set[str]:
    ids = {
        entity_id
        for state in task.get("goal", {}).get("acceptable_states", [])
        for entity_id in state.get("selected_entity_ids", [])
        if isinstance(entity_id, str)
    }
    graft = task.get("graft")
    if isinstance(graft, dict):
        ids.update(
            entity_id
            for state in graft.get("initial_goal", {}).get("acceptable_states", [])
            for entity_id in state.get("selected_entity_ids", [])
            if isinstance(entity_id, str)
        )
    refresh = task.get("refresh")
    if isinstance(refresh, dict) and isinstance(refresh.get("initial_entity_id"), str):
        ids.add(refresh["initial_entity_id"])
    clarification = task.get("clarification")
    if isinstance(clarification, dict):
        ids.update(
            entity_id for entity_id in clarification.get("candidate_ids", [])
            if isinstance(entity_id, str)
        )
    return ids


def _site_realism_ready(
    workspace: Path,
    site: str,
    entities: list[dict[str, Any]],
    required_ids: set[str],
) -> bool:
    relevant = [entity for entity in entities if entity.get("entity_id") in required_ids]
    if len(relevant) != len(required_ids):
        return False
    def asset_exists(entity: dict[str, Any]) -> bool:
        value = entity_visual_path(site, entity)
        return isinstance(value, str) and bool(value) and resolve_within(
            workspace, value, context=f"{site} realism asset"
        ).is_file()
    if site == "egomart":
        return all(asset_exists(entity) for entity in relevant)
    if site == "egomedia":
        required = {
            "thumbnail_path", "channel_title", "published_at",
            "duration_seconds", "view_count",
        }
        return all(
            required <= set(entity) and asset_exists(entity)
            for entity in relevant
        )
    if site == "egomaps":
        return all(
            entity.get("location") and asset_exists(entity)
            for entity in relevant
        )
    return False


def _instruction_is_natural(task: dict[str, Any]) -> bool:
    instructions = task.get("instructions", {})
    return isinstance(instructions, dict) and bool(instructions) and all(
        isinstance(value, str)
        and bool(value.strip())
        and INTERNAL_TASK_LANGUAGE.search(value) is None
        for value in instructions.values()
    )


def _claim_references_are_valid(source: dict[str, Any]) -> bool:
    available = {
        f"{packet.get('id')}:{index}"
        for packet in source.get("packets", [])
        for index, _ in enumerate(packet.get("frame_paths", []))
    }
    claims = source.get("supported_claims", [])
    return bool(claims) and all(
        isinstance(claim, dict)
        and bool(str(claim.get("text", "")).strip())
        and isinstance(claim.get("supported_by"), list)
        and bool(claim["supported_by"])
        and set(claim["supported_by"]) <= available
        for claim in claims
    )


def _origin_is_valid(task: dict[str, Any], source: dict[str, Any]) -> bool:
    origin = source.get("construct", {}).get("event_origin")
    if origin not in {"observed", "annotation_derived", "counterfactual_recombination"}:
        return False
    return task["track"] != "refresh" or origin == "observed"


def build_preaudit(root: str | Path = ".") -> dict[str, Any]:
    workspace = Path(root).resolve()
    selection = _read_object(workspace / "data/selection.json")
    selected_ids = _selected_ids(selection)
    compiled = {task["id"]: task for task in load_tasks(workspace / "data/tasks")}
    source_records: dict[str, tuple[dict[str, Any], Path]] = {}
    for reference in selection["source_manifests"]:
        manifest_path = resolve_within(
            workspace, reference, context="preaudit source manifest"
        )
        for record in _read_object(manifest_path).get("tasks", []):
            task_id = str(record.get("task_id", ""))
            if task_id in selected_ids:
                if task_id in source_records:
                    raise ValueError(f"duplicate selected source record: {task_id}")
                source_records[task_id] = (record, manifest_path)

    audit_index_path = workspace / "artifacts/reports/site-audits/index.json"
    audit_index = _read_object(audit_index_path).get("tasks", {})
    site_data = {
        site: load_site_data(workspace, site)["entities"]
        for site in {task["site"] for task in compiled.values()}
    }
    rows: list[dict[str, Any]] = []
    for task_id in selected_ids:
        task = compiled.get(task_id)
        source_entry = source_records.get(task_id)
        site_audit = audit_index.get(task_id)
        if task is None or source_entry is None or not isinstance(site_audit, dict):
            raise ValueError(f"{task_id}: compiled task, source record or site audit missing")
        source, manifest_path = source_entry
        frame_paths = [
            frame_path
            for packet in source.get("packets", [])
            for frame_path in packet.get("frame_paths", [])
        ]
        unresolved = [str(value) for value in source.get("unresolved", [])]
        blocking_checks = [str(value) for value in source.get("blocking_checks", [])]
        checks = {
            "source_video_screened": source.get("status") == "video_screened",
            "source_cluster_present": bool(
                source.get("source", {}).get("media_id")
                or source.get("source", {}).get("original_video_id")
            ),
            "released_frames_present": bool(frame_paths)
            and all(resolve_within(workspace, path, context="preaudit frame").is_file()
                    for path in frame_paths),
            "supported_claim_recorded": bool(source.get("supported_claims")),
            "supported_claim_frames_resolve": _claim_references_are_valid(source),
            "scenario_origin_valid": _origin_is_valid(task, source),
            "natural_user_instruction": _instruction_is_natural(task),
            "behavior_change_recorded": bool(
                source.get("construct", {}).get("required_behavior_change")
            ),
            "deterministic_goal_present": bool(
                task.get("goal", {}).get("acceptable_states")
            ),
            "joint_site_audit_current": (
                site_audit.get("task_sha256")
                == hashlib.sha256(
                    (workspace / f"data/tasks/{task['track']}/{task_id}.json").read_bytes()
                ).hexdigest()
            ),
        }
        structural_pass = all(checks.values())
        realism_ready = _site_realism_ready(
            workspace,
            task["site"],
            site_data[task["site"]],
            _task_entity_ids(task),
        )
        if not realism_ready:
            decision = "candidate_blocked_site_realism"
        elif unresolved or blocking_checks:
            decision = "primary_preaudit_pass_with_followups"
        else:
            decision = "primary_preaudit_pass"
        rows.append({
            "task_id": task_id,
            "track": task["track"],
            "site": task["site"],
            "source_cluster": (
                source.get("source", {}).get("media_id")
                or source.get("source", {}).get("original_video_id")
            ),
            "source_manifest": manifest_path.relative_to(workspace).as_posix(),
            "site_audit": site_audit["audit_path"],
            "checks": checks,
            "structural_pass": structural_pass,
            "site_realism_ready": realism_ready,
            "unresolved": unresolved,
            "blocking_checks": blocking_checks,
            "independent_human_review_pending": True,
            "decision": decision if structural_pass else "invalid",
        })

    decisions = Counter(row["decision"] for row in rows)
    tracks = Counter(row["track"] for row in rows)
    sites = Counter(row["site"] for row in rows)
    return {
        "schema_version": 1,
        "review_scope": "primary_agent_advisory_preaudit",
        "admission_authority": False,
        "interpretation": (
            "A primary_preaudit_pass means the current source, released frames, realistic website "
            "mapping and deterministic goal are internally consistent. A site-realism blocker keeps "
            "a structurally executable task in the candidate pool until its web data and images are "
            "source-bound. No verdict replaces two independent human reviews."
        ),
        "summary": {
            "tasks": len(rows),
            "decisions": dict(sorted(decisions.items())),
            "tracks": dict(sorted(tracks.items())),
            "sites": dict(sorted(sites.items())),
        },
        "tasks": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a task-by-task advisory preaudit from current source truth."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/reports/task-preaudit.json")
    args = parser.parse_args()
    workspace = Path(args.root).resolve()
    output = resolve_within(workspace, args.output, context="preaudit output", must_exist=False)
    report = build_preaudit(workspace)
    write_json_atomic(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
