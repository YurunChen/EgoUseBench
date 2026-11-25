"""One lifecycle interface for benchmark-task synthesis."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .confidence import (
    CONFIDENCE_LEVELS,
    confidence_review_required,
)
from .confidence_packets import CurrentReviewMaterials
from .confidence_report import (
    apply_human_reviews,
    assessment_fingerprints,
    task_confidence,
)
from .confidence_session import (
    open_pass,
    prior_review_evidence,
    record_assessment,
    start_protocol_recheck,
    start_review,
)
from .construction import CurrentConstruction, load_construction_tasks, write_selected
from .development_gate import DevelopmentDecision, DevelopmentEvidence
from .environment import load_site_data, site_data_request
from .experiment import ExperimentPlan
from .run_ledger import task_local_content
from .sampling_media import prepare_session_tasks
from .schema import validate_task
from .selection import SELECTION_PATH, load_selection, task_ids as selected_task_ids
from .lineage import validate_independent_recordings
from .synthesis_reports import REPORTS
from .timeline import COMPOSITE_TRACK, Timeline, TimelineError, composite_input
from .update_count import VARIANT_SUFFIX, audit_case, compile_variant, load_cases
from .workspace import read_json_strict, write_json_atomic


class TaskStatus(StrEnum):
    NEEDS_REPAIR = "needs_repair"
    NEEDS_REVIEW = "needs_review"
    READY = "ready"


TASK_STATUS_LABELS = {
    TaskStatus.NEEDS_REPAIR: "待完善",
    TaskStatus.NEEDS_REVIEW: "处理中",
    TaskStatus.READY: "已合格",
}


@dataclass(frozen=True)
class _TaskAction:
    status: TaskStatus
    priority: int


_TASK_ACTIONS = {
    "compile_task": _TaskAction(TaskStatus.NEEDS_REVIEW, 1),
    "run_deterministic_audit": _TaskAction(TaskStatus.NEEDS_REVIEW, 1),
    "repair_task": _TaskAction(TaskStatus.NEEDS_REPAIR, 0),
    "review_protocol_counterexamples": _TaskAction(TaskStatus.NEEDS_REVIEW, 2),
    "coding_agent_assess": _TaskAction(TaskStatus.NEEDS_REVIEW, 3),
    "resolve_source_review": _TaskAction(TaskStatus.NEEDS_REVIEW, 4),
    "human_review": _TaskAction(TaskStatus.NEEDS_REVIEW, 5),
    "verify_development_evidence": _TaskAction(TaskStatus.NEEDS_REVIEW, 6),
    "run_development_trial": _TaskAction(TaskStatus.NEEDS_REVIEW, 6),
    "review_development_trial": _TaskAction(TaskStatus.NEEDS_REVIEW, 7),
    "review_stream_window": _TaskAction(TaskStatus.NEEDS_REVIEW, 7),
    "admit_task": _TaskAction(TaskStatus.READY, 8),
    "build_release": _TaskAction(TaskStatus.READY, 8),
    "ready_for_composite_study": _TaskAction(TaskStatus.READY, 8),
}


@dataclass(frozen=True)
class _TaskEvidence:
    audits: Mapping[str, Any]
    confidence: Mapping[str, Any]
    source: Mapping[str, Any]
    development: DevelopmentDecision


REVIEWABLE_SOURCE_CHECKS = {
    "independent_blind_review_pending",
    "independent_construct_review_pending",
    "task_level_semantic_review_pending",
    "source_redistribution_review_pending",
    "source_redistribution_permission_pending",
}


STREAM_WINDOW_ACTION = "review_stream_window"
"""The formal protocol is fixed-step replay, which streams a reviewed session window."""

COMPOSITE_DRAFTS = Path("data/experiments/composite")
COMPOSITE_REVIEWS = Path("data/reviews/composite")
_COMPOSITE_ID = re.compile(r"(RG|CG|CR|CRG)-[0-9]{2,}(?:-update-count)?")


class SynthesisWorkflowError(ValueError):
    """Raised when synthesis lifecycle artifacts cannot be interpreted safely."""


def _has_reviewed_stream_window(source: Mapping[str, Any]) -> bool:
    stream = source.get("stream") or {}
    return "session_seconds" in stream and "review_evidence" in stream


def _read_optional_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = read_json_strict(path)
    if not isinstance(value, dict):
        raise SynthesisWorkflowError(f"{path}: expected one JSON object")
    return value


def load_composite_drafts(
    root: Path, task_ids: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    directory = root / COMPOSITE_DRAFTS
    requested = (
        sorted(path.stem for path in directory.glob("*.json"))
        if task_ids is None
        else list(task_ids)
    )
    if not requested or len(requested) != len(set(requested)):
        raise SynthesisWorkflowError("composite study requires non-empty, distinct task IDs")
    tasks = []
    for task_id in requested:
        if not _COMPOSITE_ID.fullmatch(task_id):
            raise SynthesisWorkflowError(f"not a composite task ID: {task_id!r}")
        path = directory / f"{task_id}.json"
        if task_id.endswith(VARIANT_SUFFIX):
            (case,) = load_cases(root, [task_id.removesuffix(VARIANT_SUFFIX)])
            if not case["included"]:
                raise SynthesisWorkflowError(f"{task_id}: excluded: {case['exclusion_reason']}")
            original = read_json_strict(root / case["source_task_path"])
            draft = compile_variant(original, case)
        else:
            if not path.is_file():
                raise SynthesisWorkflowError(f"unregistered composite task: {task_id}")
            draft = _read_optional_object(path)
        if draft.get("id") != task_id or draft.get("track") != COMPOSITE_TRACK:
            raise SynthesisWorkflowError(f"{path}: incorrect composite identity")
        if not all(
            isinstance(draft.get(key), dict)
            for key in ("source", "instructions", "visual_session", "goal", "truth")
        ):
            raise SynthesisWorkflowError(f"{path}: incomplete composite definition")
        if "visual_packets" in draft or "frequency_samples" in draft["visual_session"]:
            raise SynthesisWorkflowError(f"{path}: store the source window, not generated frames")
        if not draft["truth"].get("site_scope_entity_ids"):
            raise SynthesisWorkflowError(f"{path}: composite task needs a reviewed website scope")
        try:
            definition = composite_input(draft)
        except TimelineError as error:
            raise SynthesisWorkflowError(str(error)) from error
        if definition.mechanisms != set(task_id.split("-")[0]):
            raise SynthesisWorkflowError(f"{path}: mechanisms do not match the registered group")
        tasks.append(draft)
    return tasks


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _composite_review_decision(
    draft: dict[str, Any], review: Mapping[str, Any]
) -> DevelopmentDecision:
    reviewed_task = review.get("reviewed_task")
    if (
        not isinstance(reviewed_task, dict)
        or task_local_content(reviewed_task) != task_local_content(draft)
    ):
        return DevelopmentDecision(
            "coding_agent_assess", "the current composite definition has no matching semantic review"
        )
    evidence = review.get("evidence")
    counterexamples = evidence.get("counterexamples") if isinstance(evidence, dict) else None
    definition = composite_input(draft)
    if (
        not _nonempty_text(review.get("reviewer"))
        or not isinstance(counterexamples, dict)
        or set(counterexamples) != definition.event_ids
        or not all(map(_nonempty_text, counterexamples.values()))
        or not all(
            _nonempty_text(evidence.get(key))
            for key in ("naturalness", "answers", "operations")
        )
    ):
        unit = "event" if definition.explicit_events else "mechanism"
        return DevelopmentDecision(
            "coding_agent_assess",
            f"review must cover naturalness, final answers, operations and each removed {unit}",
        )
    if review.get("verdict") == "task_or_site_defect":
        return DevelopmentDecision("repair_task", "semantic review records a task or website defect")
    if review.get("verdict") != "accepted":
        return DevelopmentDecision("coding_agent_assess", "the semantic review has no accepted verdict")
    return DevelopmentDecision(None, "current composite semantic review is complete")


def prepare_composite_candidate(
    root: Path, draft: dict[str, Any], plan: ExperimentPlan
) -> dict[str, Any]:
    condition = plan.condition("online")
    if condition.frame_interval_seconds is None:
        raise SynthesisWorkflowError("composite study requires fixed-step video sampling")
    (task,) = prepare_session_tasks([draft], root, float(condition.frame_interval_seconds))
    validate_task(task, load_site_data(root, str(task["site"])))
    timeline = Timeline.compile(
        task, condition, default_release_step=plan.default_release_step,
        reference_steps=plan.reference_steps,
    )
    late = [
        release.ref
        for release in timeline.releases
        if release.update and release.step > plan.max_steps
    ]
    if late:
        raise SynthesisWorkflowError(f"{task['id']}: updates outside the active budget: {late}")
    return task


def _audit_rows(report: Mapping[str, Any]) -> Mapping[str, Any]:
    if not report:
        return {}
    value = report.get("tasks")
    if not isinstance(value, Mapping):
        raise SynthesisWorkflowError("audit report tasks must be an object")
    if any(not isinstance(row, Mapping) for row in value.values()):
        raise SynthesisWorkflowError("audit task evidence must be an object")
    return value


def _assessment_rows(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if not report:
        return {}
    value = report.get("results")
    if not isinstance(value, list):
        raise SynthesisWorkflowError("assessment report results must be a list")
    if any(
        not isinstance(row, Mapping) or not isinstance(row.get("task_id"), str)
        for row in value
    ):
        raise SynthesisWorkflowError("assessment report rows must identify a task")
    rows = {str(row["task_id"]): row for row in value}
    if len(rows) != len(value):
        raise SynthesisWorkflowError("assessment report contains duplicate task IDs")
    return rows


def construction_review_complete(
    task_id: str, workflow_status: Mapping[str, Any]
) -> bool:
    """Construction review precedes development testing, not final release."""
    return workflow_status["tasks"][task_id]["next_action"] in {
        "verify_development_evidence",
        "run_development_trial",
        "review_development_trial",
        "admit_task",
        "build_release",
        "ready_for_composite_study",
    }



def status_summary(lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize rows derived by the shared lifecycle, including partial refreshes."""
    statuses = Counter(row["status"] for row in lifecycle.values())
    return {
        "tasks": len(lifecycle),
        "status_counts": {status.value: statuses[status] for status in TaskStatus},
        "ready": statuses[TaskStatus.READY],
        "human_review": sum(
            row["next_action"] == "human_review" for row in lifecycle.values()
        ),
        "in_progress": sum(
            row["next_action"] not in {"human_review", "build_release", "ready_for_composite_study"}
            for row in lifecycle.values()
        ),
        "remaining": len(lifecycle) - statuses[TaskStatus.READY],
        "next_actions": dict(
            sorted(Counter(row["next_action"] for row in lifecycle.values()).items())
        ),
        "pending_actions": dict(
            sorted(
                Counter(
                    action
                    for row in lifecycle.values()
                    for action in row["remaining_actions"]
                ).items()
            )
        ),
    }


class SynthesisWorkflow:
    """Expose synthesis state, the next work item, and semantic decisions."""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    def status(
        self,
        task_ids: Iterable[str] | None = None,
        *,
        live_human_reviews: bool = True,
        composite: bool = False,
        update_count: bool = False,
    ) -> dict[str, Any]:
        requested_ids = (
            None if task_ids is None else [str(task_id) for task_id in task_ids]
        )
        if update_count:
            if composite:
                raise SynthesisWorkflowError("choose composites or update-count variants")
            requested_ids = requested_ids or [case["id"] for case in load_cases(self.root) if case["included"]]
            if not requested_ids:
                raise SynthesisWorkflowError("no included update-count variants")
            if any(not task_id.endswith(VARIANT_SUFFIX) for task_id in requested_ids):
                raise SynthesisWorkflowError("update-count status requires variant task IDs")
        if composite or any(
            _COMPOSITE_ID.fullmatch(task_id) for task_id in requested_ids or []
        ):
            return self._composite_status(requested_ids)
        with site_data_request(self.root):
            tasks = {
                str(task["id"]): task
                for task in load_construction_tasks(self.root, requested_ids)
            }
            members = set(selected_task_ids(load_selection(self.root)))
            selected = members if requested_ids is None else set(requested_ids)
            unknown = members - set(tasks)
            if unknown:
                raise SynthesisWorkflowError(
                    f"task_set members are missing: {sorted(unknown)}"
                )

            reports = {
                name: _read_optional_object(self.root / REPORTS[name].path)
                for name in ("acceptance", "smoke", "role", "assessment")
            }
            report_rows = {
                name: _audit_rows(reports[name])
                for name in ("acceptance", "smoke", "role")
            }
            construction = CurrentConstruction(self.root)
            compiled = {
                task_id: construction.matches(tasks[task_id])
                for task_id in sorted(selected)
            }
            assessment_rows = _assessment_rows(reports["assessment"])
            materials = CurrentReviewMaterials(self.root)
            fingerprints = assessment_fingerprints(list(assessment_rows.values()))
            agent_rows = {
                task_id: task_confidence(
                    tasks[task_id],
                    acceptance=report_rows["acceptance"].get(task_id),
                    assessment=assessment_rows.get(task_id),
                    materials=materials,
                    fingerprints=fingerprints,
                )
                for task_id in selected
                if compiled[task_id]
            }
            confidence_rows = (
                apply_human_reviews(self.root, agent_rows)
                if live_human_reviews
                else agent_rows
            )
            development = DevelopmentEvidence(self.root)
            lifecycle = {}
            for task_id in sorted(selected):
                if not compiled[task_id]:
                    lifecycle[task_id] = self._row(
                        task_id,
                        "compile_task",
                        "current authored instructions or source blueprint differ from the executable task",
                    )
                    continue
                task = tasks[task_id]
                source = construction.source(task)
                evidence = _TaskEvidence(
                    audits={
                        "acceptance": report_rows["acceptance"].get(task_id),
                        "smoke": report_rows["smoke"].get(task_id),
                        "role_recoverability": report_rows["role"].get(task_id),
                    },
                    confidence=confidence_rows.get(task_id),
                    source=source,
                    development=development.decision(task),
                )
                row = self._task_status(task_id, evidence)
                if task_id not in members and row["status"] == TaskStatus.READY:
                    row = self._row(
                        task_id,
                        "admit_task",
                        "candidate evidence is complete; install it in the formal task set before release",
                    )
                action = evidence.development.action
                if action is not None and action not in row["remaining_actions"]:
                    row["remaining_actions"].append(action)
                if not _has_reviewed_stream_window(source):
                    if row["status"] == TaskStatus.READY:
                        row = {
                            **self._row(
                                task_id,
                                STREAM_WINDOW_ACTION,
                                "fixed-step replay needs a reviewed session window",
                            ),
                            "remaining_actions": [
                                STREAM_WINDOW_ACTION,
                                *row["remaining_actions"],
                            ],
                        }
                    else:
                        row["remaining_actions"].append(STREAM_WINDOW_ACTION)
                lifecycle[task_id] = row
            return {
                "schema_version": 2,
                "summary": status_summary(lifecycle),
                "tasks": lifecycle,
                "review_evidence": {
                    task_id: confidence_rows[task_id]
                    for task_id in sorted(selected)
                    if task_id in confidence_rows
                },
            }

    def _composite_review(self, draft: dict[str, Any]) -> dict[str, Any]:
        return _read_optional_object(self.root / COMPOSITE_REVIEWS / f"{draft['id']}.json")

    def composite_trial_inputs(
        self, task_ids: Iterable[str] | None, plan: ExperimentPlan
    ) -> list[dict[str, Any]]:
        """Prepare reviewed experimental tasks for one online development study."""
        with site_data_request(self.root):
            tasks = []
            for draft in load_composite_drafts(self.root, task_ids):
                semantic = _composite_review_decision(draft, self._composite_review(draft))
                if semantic.action is not None:
                    raise SynthesisWorkflowError(f"{draft['id']}: {semantic.reason}")
                tasks.append(prepare_composite_candidate(self.root, draft, plan))
            return tasks

    def _composite_status(self, task_ids: list[str] | None) -> dict[str, Any]:
        lifecycle = {}
        reviews = {}
        update_count_audits = {}
        development = DevelopmentEvidence(self.root)
        with site_data_request(self.root):
            for draft in load_composite_drafts(self.root, task_ids):
                task_id = draft["id"]
                review = self._composite_review(draft)
                if review:
                    reviews[task_id] = review
                if task_id.endswith(VARIANT_SUFFIX):
                    (case,) = load_cases(self.root, [task_id.removesuffix(VARIANT_SUFFIX)])
                    audit = audit_case(self.root, case)
                    update_count_audits[case["pair_id"]] = audit
                    if audit["errors"]:
                        lifecycle[task_id] = self._row(task_id, "repair_task", "; ".join(audit["errors"]))
                        continue
                semantic = _composite_review_decision(draft, review)
                if semantic.action is not None:
                    lifecycle[task_id] = self._row(task_id, semantic.action, semantic.reason)
                    continue
                try:
                    task = prepare_composite_candidate(self.root, draft, development.trial_plan(draft))
                except (OSError, ValueError) as error:
                    lifecycle[task_id] = self._row(task_id, "repair_task", str(error))
                    continue
                trial = development.decision(task)
                lifecycle[task_id] = self._row(
                    task_id, trial.action or "ready_for_composite_study", trial.reason
                )
        return {
            "schema_version": 2,
            "summary": status_summary(lifecycle),
            "tasks": lifecycle,
            "review_evidence": reviews,
            **({"update_count_audits": update_count_audits} if update_count_audits else {}),
        }

    def admit(self, task_ids: Iterable[str]) -> dict[str, Any]:
        """Install ready candidates without publishing a release or changing old tasks."""
        requested = sorted(set(task_ids))
        if not requested:
            raise SynthesisWorkflowError("admission requires at least one task")
        if any(_COMPOSITE_ID.fullmatch(task_id) for task_id in requested):
            raise SynthesisWorkflowError("composite tasks are experimental, not formal members")
        report = self.status(requested)
        blocked = [
            row for row in report["tasks"].values() if row["status"] != TaskStatus.READY
        ]
        if blocked:
            raise SynthesisWorkflowError(
                "candidates are not ready: "
                + "; ".join(f"{row['task_id']}: {row['reason']}" for row in blocked)
            )
        selection = load_selection(self.root)
        members = set(selected_task_ids(selection))
        added = sorted(set(requested) - members)
        if not added:
            return {"admitted": [], "formal_tasks": len(members)}
        tasks = {
            task["id"]: task for task in load_construction_tasks(self.root, requested)
        }
        construction = CurrentConstruction(self.root)
        validate_independent_recordings(
            [
                (
                    self.root / tasks[tid]["source"]["dossier"],
                    construction.source(tasks[tid]),
                )
                for tid in sorted(members | set(added))
            ],
            root=self.root,
        )
        for task_id in added:
            task = tasks[task_id]
            reference = task["source"]["dossier"]
            if reference not in selection["source_manifests"]:
                selection["source_manifests"].append(reference)
            selection["task_set"][task["track"]].append(task_id)
            selection["task_set"][task["track"]].sort()
            write_json_atomic(
                self.root / "data/tasks" / task["track"] / f"{task_id}.json", task
            )
        write_json_atomic(self.root / SELECTION_PATH, selection)
        return {"admitted": added, "formal_tasks": len(members) + len(added)}

    def require_development_inputs(self, task_ids: Iterable[str]) -> None:
        """Require current task evidence before a development trial can start."""

        for task_id, row in self.status(task_ids)["tasks"].items():
            if row["next_action"] not in {
                "run_development_trial", "build_release", "ready_for_composite_study",
            }:
                raise SynthesisWorkflowError(
                    f"{task_id}: development trial prerequisites are incomplete: "
                    f"{row['next_action']}: {row['reason']}"
                )

    def next_work_item(self, task_id: str | None = None) -> dict[str, Any]:
        report = self.status([task_id] if task_id else None)
        row = min(
            report["tasks"].values(),
            key=lambda item: (
                _TASK_ACTIONS[item["next_action"]].priority,
                item["task_id"],
            ),
        )
        if _COMPOSITE_ID.fullmatch(row["task_id"]):
            return {
                "kind": row["next_action"],
                "task_id": row["task_id"],
                "status": row["status"],
                "reason": row["reason"],
            }
        if row["next_action"] == "compile_task":
            write_selected(self.root, task_ids=[row["task_id"]])
            return {
                **self.next_work_item(row["task_id"]),
                "compiled_task": row["task_id"],
            }
        if row.get("recheck_passes"):
            from .confidence_packets import build_packets

            build_packets(self.root, task_ids=[row["task_id"]])
            start_protocol_recheck(row["task_id"], root=self.root)
            self._rebuild_confidence_report(row["task_id"])
            opened = open_pass(row["task_id"], root=self.root)
            return {
                "kind": "protocol_counterexample_review",
                "task_id": row["task_id"],
                "status": row["status"],
                "reason": row["reason"],
                "recheck_passes": row["recheck_passes"],
                "retained_passes": row["retained_passes"],
                "pass": opened["pass"],
                "work_item": opened,
                "instruction": "Review the changed answer boundaries and final-state scoring. "
                "Do not repeat the retained visual blind reconstruction or mark the task released.",
            }
        if row["next_action"] == "coding_agent_assess":
            prior = prior_review_evidence(self.root, row["task_id"])
            if prior:
                return {
                    "kind": "reuse_review_evidence",
                    "task_id": row["task_id"],
                    "status": row["status"],
                    "reason": "已有具体审核材料；先沿用已检查事实，只补剩余问题。",
                    "work_item": {"prior_review_evidence": prior},
                    "instruction": "核对旧事实与当前任务的实际差异，继续补查列出的缺项。"
                    "这些材料可能含旧答案，不能把本次续查写成新盲审。"
                    "完成后通过现有审核接口登记实际结论；引用本身不代表审核通过。",
                }
            self._prepare_confidence_materials(row["task_id"])
            refreshed = self.status([row["task_id"]])["tasks"][row["task_id"]]
            if refreshed["next_action"] != "coding_agent_assess":
                return self.next_work_item(row["task_id"])
            opened = open_pass(row["task_id"], root=self.root)
            return {
                "kind": "coding_agent_assessment",
                "task_id": row["task_id"],
                "status": refreshed["status"],
                "pass": opened["pass"],
                "work_item": opened,
            }
        return {
            "kind": row["next_action"],
            "task_id": row["task_id"],
            "status": row["status"],
            "reason": row["reason"],
        }

    def release_evidence(self, task_ids: Iterable[str]) -> dict[str, Any]:
        """Return the one verified evidence view used by release packaging."""

        selected = tuple(sorted({str(task_id) for task_id in task_ids}))
        outside = set(selected) - set(selected_task_ids(load_selection(self.root)))
        if outside:
            raise SynthesisWorkflowError(
                f"release requires formal membership: {sorted(outside)}"
            )
        lifecycle = self.status(selected)
        blocked = {
            task_id: row["status"]
            for task_id, row in lifecycle["tasks"].items()
            if row["status"] != TaskStatus.READY
        }
        if blocked:
            raise SynthesisWorkflowError(f"task set is not release-ready: {blocked}")
        selected_set = set(selected)
        reports = {
            name: _read_optional_object(self.root / REPORTS[name].path)
            for name in ("acceptance", "smoke", "role", "confidence")
        }
        for name, report in reports.items():
            rows = _audit_rows(report)
            missing = selected_set - set(rows)
            if missing:
                raise SynthesisWorkflowError(
                    f"{name} report is missing requested release tasks: {sorted(missing)}"
                )

        confidence_rows = lifecycle["review_evidence"]
        for task_id, row in confidence_rows.items():
            if confidence_review_required(row) or row["route"] not in {
                "automatic_construction_admission",
                "human_review_complete",
            }:
                raise SynthesisWorkflowError(
                    f"{task_id}: current review evidence does not permit release"
                )

        def descriptor(name: str) -> dict[str, Any]:
            relative = REPORTS[name].path
            path = self.root / relative
            return {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        def audit_descriptor(name: str) -> dict[str, Any]:
            rows = _audit_rows(reports[name])
            return {
                **descriptor(name),
                "tasks": len(selected),
                "statuses": dict(
                    Counter(rows[task_id]["status"] for task_id in selected)
                ),
            }

        levels = Counter(row["level"] for row in confidence_rows.values())
        return {
            "acceptance": audit_descriptor("acceptance"),
            "confidence": {
                **descriptor("confidence"),
                "tasks": len(confidence_rows),
                "levels": {level: levels[level] for level in sorted(CONFIDENCE_LEVELS)},
                "routes": dict(
                    Counter(row["route"] for row in confidence_rows.values())
                ),
                "human_review_queue": sum(
                    row["human_review_required"] for row in confidence_rows.values()
                ),
                "coding_agent_retry_queue": sum(
                    row["route"] == "coding_agent_retry_pending"
                    for row in confidence_rows.values()
                ),
                "current_review_evidence": confidence_rows,
            },
            "role_recoverability": audit_descriptor("role"),
            "smoke": audit_descriptor("smoke"),
        }

    def decide(self, task_id: str, verdict: Mapping[str, Any]) -> dict[str, Any]:
        if _COMPOSITE_ID.fullmatch(task_id):
            (draft,) = load_composite_drafts(self.root, [task_id])
            semantic = _composite_review_decision(draft, verdict)
            if semantic.action == "coding_agent_assess":
                raise SynthesisWorkflowError(f"{task_id}: {semantic.reason}")
            write_json_atomic(self.root / COMPOSITE_REVIEWS / f"{task_id}.json", verdict)
            return {
                "task_id": task_id,
                "recorded_pass": "composite_semantic_review",
                "assessment_status": "completed",
                "lifecycle": self.status([task_id])["tasks"][task_id],
            }
        self._prepare_confidence_materials(task_id)
        opened = open_pass(task_id, root=self.root)
        result = record_assessment(
            task_id,
            str(opened["pass"]),
            dict(verdict),
            root=self.root,
        )
        if result["status"] == "completed":
            self._rebuild_confidence_report(task_id)
        return {
            "task_id": task_id,
            "recorded_pass": opened["pass"],
            "assessment_status": result["status"],
            "lifecycle": self.status([task_id])["tasks"][task_id],
        }

    def _task_status(
        self,
        task_id: str,
        evidence: _TaskEvidence,
    ) -> dict[str, Any]:
        confidence = evidence.confidence
        source = evidence.source
        source_checks = {str(item) for item in source.get("blocking_checks", [])}
        source_repairs = source_checks - REVIEWABLE_SOURCE_CHECKS
        if source_repairs:
            return self._row(
                task_id,
                "repair_task",
                f"source evidence requires repair: {', '.join(sorted(source_repairs))}",
            )
        if confidence.get("route") == "repair":
            confidence_review_required(confidence)
            return self._row(
                task_id,
                "repair_task",
                "Coding-Agent or human review found a current semantic defect",
            )
        audit_status = self._audit_status(task_id, evidence.audits)
        if audit_status is not None:
            return audit_status
        source_pending = bool(source_checks) or bool(source.get("unresolved"))
        if confidence.get("recheck_passes") == ["adversarial_falsification"]:
            return {
                **self._row(
                    task_id,
                    "review_protocol_counterexamples",
                    "原视觉盲解证据已保留；变更后的答案边界仍需对抗复核。",
                ),
                "recheck_passes": confidence["recheck_passes"],
                "retained_passes": confidence["retained_passes"],
            }
        if confidence.get("level") == "unassessed":
            return self._row(
                task_id,
                "coding_agent_assess",
                "; ".join(confidence["assessment_errors"]),
            )
        if confidence.get("live_human_review_status") == "invalid":
            return self._row(
                task_id,
                "human_review",
                "human review submission is invalid or stale; correct the current submission",
            )
        try:
            confidence_review_required(confidence)
        except (TypeError, ValueError) as exc:
            raise SynthesisWorkflowError(
                f"{task_id}: invalid confidence route: {exc}"
            ) from exc
        route = confidence.get("route")
        if route == "evidence_collection_pending":
            return self._row(
                task_id,
                "resolve_source_review",
                "; ".join(
                    item["reason"]
                    for item in confidence["follow_up"]
                    if item["action"] == "collect_evidence"
                ),
            )
        if route == "coding_agent_retry_pending":
            return self._row(
                task_id,
                "coding_agent_assess",
                "; ".join(confidence.get("assessment_errors", []))
                or "Coding-Agent assessment is incomplete",
            )
        if (
            confidence.get("human_review_required") is True
            or route == "human_review_pending"
        ):
            reuse = confidence.get("human_review_reuse", [])
            reason = (
                "旧人工结论已保留；仅受影响的问题待判："
                + "; ".join(
                    conclusion
                    for item in reuse
                    for conclusion in item["affected_conclusions"]
                )
                if reuse
                else "a semantic decision is awaiting human review"
            )
            return self._row(
                task_id,
                "human_review",
                reason,
            )
        if route not in {"automatic_construction_admission", "human_review_complete"}:
            raise SynthesisWorkflowError(
                f"{task_id}: unsupported confidence route: {route!r}"
            )
        if source_pending:
            return self._row(
                task_id,
                "resolve_source_review",
                "source evidence awaits its recorded review; semantic review is retained",
            )
        development = evidence.development
        if development.action:
            return self._row(task_id, development.action, development.reason)
        return self._row(
            task_id,
            "build_release",
            "all task-quality gates are current",
        )

    def _audit_status(
        self, task_id: str, audits: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        missing = []
        failures = []
        for name, row in audits.items():
            if row is None:
                missing.append(name)
                continue
            expected = "candidate_ready" if name == "acceptance" else "pass"
            if row.get("status") == expected:
                continue
            if (
                name == "acceptance"
                and row.get("status") == "needs_source_review"
                and row.get("integrity_pass") is True
                and row.get("failed_checks") == []
            ):
                continue
            failures.append(name)
        if failures:
            return self._row(
                task_id,
                "repair_task",
                f"deterministic evidence failed: {', '.join(sorted(failures))}",
            )
        if missing:
            return self._row(
                task_id,
                "run_deterministic_audit",
                f"deterministic evidence is missing: {', '.join(sorted(missing))}",
            )
        return None

    @staticmethod
    def _row(
        task_id: str,
        next_action: str,
        reason: str,
    ) -> dict[str, Any]:
        status = _TASK_ACTIONS[next_action].status
        return {
            "task_id": task_id,
            "status": status.value,
            "status_label": "需人审"
            if next_action == "human_review"
            else TASK_STATUS_LABELS[status],
            "next_action": next_action,
            "remaining_actions": []
            if next_action in {"build_release", "ready_for_composite_study"}
            else [next_action],
            "reason": reason,
        }

    def _prepare_confidence_materials(self, task_id: str) -> None:
        if not self._packet_index_is_current(task_id):
            from .confidence_packets import build_packets

            build_packets(self.root, task_ids=[task_id])
        start_review(self.root, refresh=True, task_ids=[task_id])
        self._rebuild_confidence_report(task_id)

    def _packet_index_is_current(self, task_id: str) -> bool:
        tasks = {
            str(task["id"]): task
            for task in load_construction_tasks(self.root, [task_id])
        }
        task = tasks.get(task_id)
        return task is not None and CurrentReviewMaterials(
            self.root
        ).stored_packet_matches(task)

    def _rebuild_confidence_report(self, task_id: str) -> None:
        from .confidence_report import build_confidence_report

        report = build_confidence_report(self.root, task_ids=[task_id])
        write_json_atomic(self.root / REPORTS["confidence"].path, report)
