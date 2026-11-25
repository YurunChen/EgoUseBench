"""Read and bind construction-review evidence independently of task status or UI."""

from __future__ import annotations

from .final_state import goal_entity_ids

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .construct_validity import CONFIDENCE_POLICY_VERSION
from .lineage import source_segment_references
from .synthesis_reports import REPORTS
from .construction import opaque_item_id
from .environment import entity_visual_path, load_site_data, public_site_data
from .experiment import TASK_REVIEW_PLAN, ExperimentPlan, load_experiment_plan
from .run_ledger import canonical_sha256, task_sha256
from .schema import TRACKS
from .sources import SourceError, find_record, load_dossier
from .timeline import Condition, Timeline
from .workspace import read_json_strict, resolve_within

REVIEW_SPEC = "data/reviews/construction.json"

if TYPE_CHECKING:
    from .confidence_packets import CurrentReviewMaterials


class ReviewMaterialError(ValueError):
    """Raised when review evidence cannot be read or verified."""


def _load_object(path: Path, context: str) -> dict[str, Any]:
    payload = read_json_strict(path)
    if not isinstance(payload, dict):
        raise ReviewMaterialError(f"{context}: JSON root must be an object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_submission_matches_case(
    payload: Mapping[str, Any],
    case: Mapping[str, Any],
) -> bool:
    """Require the complete semantic and asset binding of the current review."""

    submitted_artifacts = payload.get("reviewed_artifacts")
    if not isinstance(submitted_artifacts, Mapping):
        return False
    return (
        dict(submitted_artifacts) == dict(case["reviewed_artifacts"])
        and payload.get("revision") == case["revision"]
    )


def _check_definitions(spec: Mapping[str, Any], track: str) -> list[dict[str, str]]:
    groups = spec.get("checks")
    if not isinstance(groups, dict):
        raise ReviewMaterialError("construction review spec needs checks")
    checks = [*groups.get("common", []), *groups.get(track, [])]
    if not checks or any(
        not isinstance(check, dict)
        or set(check) != {"id", "question"}
        or not str(check["id"]).strip()
        or not str(check["question"]).strip()
        for check in checks
    ):
        raise ReviewMaterialError(f"construction review checks are invalid for {track}")
    ids = [str(check["id"]) for check in checks]
    if len(ids) != len(set(ids)):
        raise ReviewMaterialError(
            f"construction review checks are duplicated for {track}"
        )
    return [
        {"id": str(check["id"]), "question": str(check["question"])} for check in checks
    ]


def referenced_entities(
    task: Mapping[str, Any],
    source_record: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    final_ids = sorted(goal_entity_ids(task["goal"]))
    roles: dict[str, list[str]] = {entity_id: ["最终目标"] for entity_id in final_ids}
    if task["track"] == "graft":
        initial_ids = sorted(goal_entity_ids(task["graft"]["initial_goal"]))
        for entity_id in initial_ids:
            roles.setdefault(entity_id, []).append("最初目标")
    elif task["track"] == "refresh":
        roles.setdefault(task["refresh"]["initial_entity_id"], []).append("V0目标")
        blueprint = (source_record or {}).get("construct", {}).get("task_blueprint", {})
        later_role = blueprint.get("later_distractor_role")
        if isinstance(later_role, str) and later_role:
            explicit_ids = blueprint.get("environment_role_ids", {})
            later_id = (
                str(explicit_ids.get(later_role))
                if isinstance(explicit_ids, dict) and explicit_ids.get(later_role)
                else opaque_item_id(str(task["id"]), later_role)
            )
            roles.setdefault(later_id, []).append("后来出现的对象")
    else:
        for entity_id in task["clarification"]["candidate_ids"]:
            roles.setdefault(entity_id, []).append("澄清候选")
    return roles


def _experiment_view(
    task: dict[str, Any],
    condition: Condition,
    plan: ExperimentPlan,
) -> dict[str, Any]:
    """What the Agent receives under the plan's reference condition."""

    timeline = Timeline.compile(
        task,
        condition,
        default_release_step=plan.default_release_step,
        reference_steps=plan.reference_steps,
    )
    initial = timeline.initial()
    releases = [
        {"ref": release.ref, "step": release.step, "kind": release.kind}
        for release in timeline.releases
    ]
    first = timeline.releases[0] if timeline.releases else None
    visible = ["V0"] if initial["observations"] else []
    return {
        "condition": condition.name,
        "instruction": initial["instruction"],
        "visible_packets": visible,
        "clarification_available": initial["clarification_available"],
        "scheduled_event": {**first.as_event()} if first else None,
        "timeline_releases": releases,
        "sufficiency_step": timeline.sufficiency_step(),
    }


def validate_review_spec(spec: Mapping[str, Any]) -> None:
    if spec.get("schema_version") != 1:
        raise ReviewMaterialError("construction review spec schema_version must be 1")
    policy = spec.get("reviewer_policy")
    expected = {
        "required_independent_reviews": 1,
        "applies_to": "coding_agent_low_confidence_only",
        "reviewer_must_not_see_coding_agent_output": True,
        "admit_outcome": "admit",
        "reject_outcome": "repair",
    }
    if policy != expected:
        raise ReviewMaterialError("construction review policy is invalid")
    confidence_policy = spec.get("confidence_policy")
    expected_confidence = {
        "version": CONFIDENCE_POLICY_VERSION,
        "method": "blind_reconstruction_then_adversarial_falsification",
        "high_confidence_route": "automatic_construction_admission",
        "low_confidence_route": "human_review_pending",
        "high_confidence_failure_route": "repair",
        "assessment_error_route": "retry_coding_agent",
        "numeric_probability_required": False,
    }
    if confidence_policy != expected_confidence:
        raise ReviewMaterialError("construction confidence policy is invalid")
    for track in TRACKS:
        _check_definitions(spec, track)


class ReviewMaterials:
    """Own the task-local evidence and revision shared by audits and review pages."""

    def __init__(self, root: str | Path, spec: str = REVIEW_SPEC) -> None:
        self.root = Path(root).resolve()
        self.spec_path = resolve_within(
            self.root, spec, context="construction review spec"
        )
        self.spec = _load_object(self.spec_path, "construction review spec")
        validate_review_spec(self.spec)
        self.plan = load_experiment_plan(self.root / TASK_REVIEW_PLAN)
        self._assessment_report: dict[str, Any] | None = None
        self._current_materials: CurrentReviewMaterials | None = None

    def case(self, task: dict[str, Any]) -> dict[str, Any]:
        record, dataset, video = self._source_material(task)
        artifacts = self._artifact_hashes(task, record, video)
        checks = _check_definitions(self.spec, task["track"])
        experiment = _experiment_view(
            task, self.plan.reference_condition_for(task["track"]), self.plan
        )
        return {
            "task": task,
            "source_record": record,
            "source_dataset": dataset,
            "source_video": video,
            "checks": checks,
            "reviewer_policy": self.spec["reviewer_policy"],
            "experiment": experiment,
            "reviewed_artifacts": artifacts,
            "revision": canonical_sha256(
                {
                    "policy": self.spec["reviewer_policy"],
                    "checks": checks,
                    "artifacts": artifacts,
                    "experiment": experiment,
                }
            ),
            "human_review_reuse": self._human_review_reuse(task),
        }

    def _human_review_reuse(self, task: dict[str, Any]) -> list[dict[str, Any]]:
        from .confidence import reusable_review_evidence, route_assessment
        from .confidence_packets import CurrentReviewMaterials

        if self._assessment_report is None:
            path = self.root / REPORTS["assessment"].path
            self._assessment_report = (
                _load_object(path, "Coding-Agent assessment report")
                if path.is_file()
                else {}
            )
        report = self._assessment_report
        if report.get("schema_version") != REPORTS["assessment"].schema_version:
            return []
        record = next(
            (
                row
                for row in report.get("results", [])
                if row.get("task_id") == task["id"]
            ),
            {},
        )
        if not any(
            assessment.get("verdict", {}).get("human_review_reuse")
            for source in [record, *record.get("history", [])]
            for assessment in source.get("assessments", [])
        ):
            return []
        if self._current_materials is None:
            self._current_materials = CurrentReviewMaterials(self.root)
        retained = reusable_review_evidence(
            task, record, self._current_materials.packet(task)
        )
        if retained is None or retained["recheck_passes"]:
            return []
        second = retained["source"]["assessments"][1]["verdict"]
        current = route_assessment(
            task_sha256(task),
            retained["source"],
            expected_goal=task["goal"],
        )
        reuse = [
            {**item, "reviewer": second["reviewer"]}
            for item in second.get("human_review_reuse", [])
        ]
        directory = self.root / "data/reviews/submissions" / str(task["id"])
        matching = []
        for item in reuse:
            submission = resolve_within(
                self.root,
                item["submission"]["path"],
                context="retained human submission",
            )
            if submission.parent != directory or submission.suffix != ".json":
                raise ReviewMaterialError(
                    "retained human review must belong to this task"
                )
            original = _load_object(submission, "retained human submission")
            if (
                original.get("submitted_at_utc")
                == item["submission"]["submitted_at_utc"]
            ):
                if (
                    item["outcome"] != "partial"
                    and current["route"] != "automatic_construction_admission"
                ):
                    raise ReviewMaterialError(
                        "unresolved current Coding-Agent concerns cannot inherit a "
                        "whole human decision; use partial reuse or resolve the "
                        "sealed concerns first"
                    )
                matching.append(item)
        return matching

    def _artifact_hashes(
        self,
        task: Mapping[str, Any],
        source_record: Mapping[str, Any],
        source_video: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        site_data = public_site_data(
            task["site"], load_site_data(self.root, task["site"])
        )
        entity_by_id = {entity["entity_id"]: entity for entity in site_data["entities"]}
        entity_visuals = [
            visual_path
            for entity_id in referenced_entities(task, source_record)
            if entity_id in entity_by_id
            if (
                visual_path := entity_visual_path(task["site"], entity_by_id[entity_id])
            )
            is not None
            if resolve_within(
                self.root,
                visual_path,
                context=f"{task['id']} review website image",
                must_exist=False,
            ).is_file()
        ]
        semantic_hashes = {
            "@review-contract": canonical_sha256(self.spec),
            f"@task/{task['id']}": task_sha256(task),
            **{
                f"@entity/{task['site']}/{entity_id}": canonical_sha256(
                    entity_by_id[entity_id]
                )
                for entity_id in referenced_entities(task, source_record)
                if entity_id in entity_by_id
            },
        }
        video_artifacts = {}
        if source_video:
            videos = (
                source_video["segments"]
                if "segments" in source_video
                else [source_video]
            )
            for video in videos:
                video_path = str(video["path"])
                media_sha256 = (
                    video.get("sha256")
                    if "segments" in source_video
                    else source_record.get("source", {}).get("media_sha256")
                )
                video_artifacts[video_path] = (
                    media_sha256
                    if isinstance(media_sha256, str) and media_sha256
                    else _file_sha256(
                        resolve_within(
                            self.root, video_path, context=f"{task['id']} source video"
                        )
                    )
                )
        references = [
            *[
                frame
                for packet in task["visual_packets"]
                for frame in packet["frame_paths"]
            ],
            *entity_visuals,
        ]
        return {
            **semantic_hashes,
            **video_artifacts,
            **{
                reference: _file_sha256(
                    resolve_within(
                        self.root, reference, context=f"{task['id']} review artifact"
                    )
                )
                for reference in dict.fromkeys(references)
            },
        }

    def _source_material(
        self,
        task: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        try:
            _, dossier = load_dossier(task, root=self.root)
            record = find_record(
                dossier, str(task["id"]), context=f"{task['id']} source record"
            )
        except (SourceError, ValueError, FileNotFoundError) as exc:
            raise ReviewMaterialError(f"{task['id']}: {exc}") from exc
        records = [record]
        dataset = dossier.get("dataset", {})
        if not isinstance(dataset, dict):
            dataset = {}
        source = records[0].get("source", {})
        source_video: dict[str, Any] | None = None
        if "media_segments" in source:
            return (
                records[0],
                dataset,
                source_segment_references(records[0], dataset, self.root),
            )
        source_root = dataset.get("source_root")
        media_id = source.get("media_id") if isinstance(source, dict) else None
        if isinstance(source_root, str) and isinstance(media_id, str):
            base = resolve_within(
                self.root,
                source_root,
                context="source video directory",
                must_exist=False,
            )
            if base.is_dir():
                candidates = [base / f"{media_id}.mp4", base / f"{media_id}.MP4"]
                candidates.extend(sorted(base.glob(f"{media_id}*")))
                match = next((path for path in candidates if path.is_file()), None)
                if match is not None:
                    timestamps = [
                        float(timestamp)
                        for packet in records[0].get("packets", [])
                        for timestamp in packet.get("timestamps_seconds", [])
                        if isinstance(timestamp, (int, float))
                    ]
                    media_start = float(source.get("media_start_seconds", 0) or 0)
                    local_start = max(0.0, min(timestamps) - media_start - 1.0)
                    local_end = max(local_start, max(timestamps) - media_start + 1.0)
                    source_video = {
                        "path": match.relative_to(self.root).as_posix(),
                        "start_seconds": round(local_start, 3),
                        "end_seconds": round(local_end, 3),
                    }
        return records[0], dataset, source_video
