"""Propose source-faithful YouTube landings from captured real result windows.

The model maps existing semantic roles to real search results. It cannot update
manifests or admit a task; the Coding Agent must review every proposal first.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping


from egocua.schema import load_tasks
from tools.catalog.landing_proposals import LandingProposals, role_sets, source_records

OUTPUT = Path("data/sources/youtube_landing_proposals.json")
CAPTURES = Path("data/sources/upstream_search_snapshots/youtube")


def _payload(
    root: Path,
    task: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    task_id = str(task["id"])
    blueprint = record["construct"]["task_blueprint"]
    role_ids = {
        str(role): str(entity_id)
        for role, entity_id in blueprint["environment_role_ids"].items()
    }
    initial, final = role_sets(task, blueprint)
    active_roles = initial | final
    if task["track"] == "refresh" and blueprint.get("later_distractor_role"):
        active_roles.add(str(blueprint["later_distractor_role"]))
    role_ids = {
        role: entity_id for role, entity_id in role_ids.items() if role in active_roles
    }
    initial_only = initial - final
    final_only = final - initial
    before = json.loads(
        (root / CAPTURES / f"{task_id}--before_change.json").read_text()
    )
    after = json.loads((root / CAPTURES / f"{task_id}--after_change.json").read_text())
    before_ids = {str(row["youtube_id"]) for row in before["results"]}
    after_ids = {str(row["youtube_id"]) for row in after["results"]}

    role_requirements: list[dict[str, Any]] = []
    allowed_ids_by_role: dict[str, set[str]] = {}
    for role, old_entity_id in role_ids.items():
        if role in initial and role in final:
            if task["track"] == "clarify":
                stage = "both"
            elif task["track"] == "refresh":
                # Retain tasks intentionally keep the pre-update commitment. The
                # later frames justify retaining it; they need not force a second
                # query or make mutable upstream ranking reproduce the same row.
                stage = "before_change"
            elif not initial_only and not final_only:
                stage = "both"
            elif final_only:
                stage = "before_change"
            else:
                stage = "after_change"
        elif role in initial:
            stage = "before_change"
        elif role in final:
            stage = "after_change"
        else:
            # Refresh distractors are introduced by the later visual packet.
            stage = "after_change" if task["track"] == "refresh" else "before_change"
        allowed = (
            before_ids & after_ids
            if stage == "both"
            else before_ids
            if stage == "before_change"
            else after_ids
        )
        if not allowed:
            raise ValueError(
                f"{task_id}/{role}: no candidates for required stage {stage}"
            )
        role_requirements.append(
            {
                "role": role,
                "old_entity_id": old_entity_id,
                "required_stage": stage,
                "is_initial": role in initial,
                "is_final": role in final,
            }
        )
        allowed_ids_by_role[role] = allowed

    def rows(capture: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                key: row.get(key)
                for key in (
                    "rank",
                    "youtube_id",
                    "title",
                    "channel_title",
                    "duration_seconds",
                    "view_count",
                )
            }
            for row in capture["results"]
        ]

    before_rows = rows(before)
    after_rows = rows(after)
    row_index = {str(row["youtube_id"]): row for row in [*before_rows, *after_rows]}
    return {
        "task_id": task_id,
        "track": task["track"],
        "instructions": task["instructions"],
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
        "role_requirements": role_requirements,
        "allowed_candidates_by_role": {
            role: [row_index[video_id] for video_id in sorted(allowed)]
            for role, allowed in allowed_ids_by_role.items()
        },
        "before_query": before["query"],
        "after_query": after["query"],
        "before_results": before_rows,
        "after_results": after_rows,
    }


def _prompt(payload: Mapping[str, Any], feedback: str) -> str:
    return (
        """Map the semantic roles of one multimodal browser task to real YouTube
search results. Return strict JSON only:
{"task_assessment":{"useful":true,"scene_update_website_consistent":true,
"continuous_change":true,"verifiable_from_public_metadata":true,"issue":""},
"role_landings":[{"role":"...","youtube_id":"...","reason":"..."}]}

Rules:
- Return exactly one role_landings row for every role_requirements role.
- Copy each youtube_id only from that role's allowed_candidates_by_role list.
- Every chosen video must satisfy the role at the exact granularity supported by
  its public title/channel/duration. Do not infer hidden steps from a generic title.
- Preserve the causal change: before roles support A; final roles satisfy S plus B;
  removed or distracting roles must remain plausible before the update.
- Choose distinct videos for distinct roles.
- Prefer a natural top result with several same-topic competitors. Ranking alone is
  never evidence of correctness.
- Set any assessment field false when the source, update, or public metadata cannot
  support the requested role. Explain the concrete problem in issue. The Coding
  Agent will repair or replace the task. Even when assessment is false, return the
  best available distinct candidate for every role so the schema stays complete.

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
    before = {str(item["youtube_id"]) for item in payload["before_results"]}
    after = {str(item["youtube_id"]) for item in payload["after_results"]}
    landings = row["role_landings"]
    if not isinstance(landings, list) or {
        str(item.get("role")) for item in landings if isinstance(item, Mapping)
    } != set(requirements):
        raise ValueError("role_landings must cover every role exactly")
    seen_roles: set[str] = set()
    seen_videos: set[str] = set()
    for item in landings:
        if not isinstance(item, dict) or set(item) != {"role", "youtube_id", "reason"}:
            raise ValueError("invalid landing row")
        role = str(item["role"])
        video_id = str(item["youtube_id"])
        if role in seen_roles or video_id in seen_videos:
            raise ValueError("roles and selected videos must be unique")
        seen_roles.add(role)
        seen_videos.add(video_id)
        stage = requirements[role]
        if stage in {"before_change", "both"} and video_id not in before:
            raise ValueError(f"{role} is outside the before result window")
        if stage in {"after_change", "both"} and video_id not in after:
            raise ValueError(f"{role} is outside the after result window")
        if not str(item["reason"]).strip():
            raise ValueError(f"{role} lacks a reason")
    if (
        not all(assessment[key] for key in assessment_keys - {"issue"})
        and not assessment["issue"].strip()
    ):
        raise ValueError("failed assessment requires an issue")
    return row


def build(root: Path, task_ids: set[str] | None, concurrency: int) -> dict[str, int]:
    records = source_records(root)
    tasks = [
        task
        for task in load_tasks(root / "data/tasks")
        if task["site"] == "egomedia"
        and (task_ids is None or str(task["id"]) in task_ids)
    ]
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
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
