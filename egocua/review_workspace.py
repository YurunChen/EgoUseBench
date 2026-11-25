"""One local interface for reviewing benchmark construction validity."""

from __future__ import annotations

from .final_state import goal_entity_ids

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sites.egomart.pages import rating_of, visible_attributes

from . import review_materials
from .reviews import (
    ReviewAuditError,
    StaleReviewError as StaleSubmissionError,
    validate_review_submission,
)
from .environment import entity_visual_path, load_site_data
from .experiment import TASK_REVIEW_PLAN, ExperimentPlan, load_experiment_plan
from .lineage import validate_source_lineage
from .schema import load_tasks, task_transition, validate_dataset, validate_task
from .selection import SELECTION_PATH, SelectionError, task_ids
from .synthesis_workflow import SynthesisWorkflow, TaskStatus
from .synthesis_reports import REPORTS
from .workspace import (
    read_json_strict,
    resolve_within,
    write_bytes_atomic,
    write_json_atomic,
)

SELECTION = SELECTION_PATH
TASKS = "data/tasks"
SUBMISSIONS = "data/reviews/submissions"
REVIEWER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
GRANULARITIES = {"category", "brand", "variant", "exact_instance"}
EVENT_ORIGINS = {
    "redesign_same_source": (
        "沿用原视频重新设计",
        "对象和活动来自原视频，任务需求按该场景重新编写。",
    ),
    "observed": (
        "视频中真实发生",
        "对象或场景变化直接来自连续视频。",
    ),
    "annotation_derived": (
        "由原始标注改写",
        "真实画面和区分信息来自原始数据，只把部分信息留到澄清时提供。",
    ),
    "counterfactual_recombination": (
        "基于真实画面编写",
        "画面和对象来自真实视频，但用户中途补充的话不是视频中原本记录的对话。",
    ),
}
SITE_LABELS = {
    "egomart": "购物",
    "egomedia": "媒体",
    "egomaps": "地图",
}
TRACK_LABELS = {
    "graft": "中途补充",
    "refresh": "画面更新",
    "clarify": "需要澄清",
}
TRANSITION_LABELS = {
    "refine": "细化原目标",
    "relax": "放宽原约束",
    "augment": "增加目标",
    "replace": "替换目标",
    "drop": "撤回目标",
    "retain": "保留目标",
    "resolve": "消除指代歧义",
}
ROLE_LABELS = {
    "最终目标": "最终目标",
    "最初目标": "最初可接受",
    "V0目标": "开始时的目标",
    "后来出现的对象": "后来出现的对象",
    "澄清候选": "澄清候选",
}
SLOT_LABELS = {
    "position": "位置",
    "appearance": "外观",
    "brand": "品牌",
    "variant": "款式",
    "sequence": "先后顺序",
}


class ReviewWorkspaceError(ValueError):
    """Raised when a review cannot be shown or submitted safely."""


class StaleReviewError(ReviewWorkspaceError):
    """Raised when reviewed material changed after the page was opened."""


class AlreadySubmittedError(ReviewWorkspaceError):
    """Raised when one reviewer tries to replace a current submission."""


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = read_json_strict(path)
    except (OSError, ValueError) as exc:
        raise ReviewWorkspaceError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewWorkspaceError(f"{context}: JSON root must be an object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _review_queue(selection: Mapping[str, Any]) -> list[str]:
    try:
        return task_ids(selection)
    except SelectionError as exc:
        raise ReviewWorkspaceError(str(exc)) from exc


def _targeted_review_queue(
    lifecycle: Mapping[str, Any],
    selected_ids: list[str],
) -> list[str]:
    return [
        task_id
        for task_id in selected_ids
        if lifecycle["tasks"][task_id]["next_action"] == "human_review"
    ]


def _task_focus(task: Mapping[str, Any]) -> str:
    if task["track"] == "graft":
        return "确认最初要求可以独立完成，后来补充的要求确实改变最终目标，并保留仍需完成的内容。"
    if task["track"] == "refresh":
        policy = task["refresh"]["policy"]
        action = "保持最初目标" if policy == "retain" else "更新为后来出现的目标"
        return f"确认连续画面中的变化真实可辨，并且按照指令应该{action}。"
    return "确认初始画面中确实存在两个合理候选，而且一次自然澄清可以唯一确定目标。"


def _design_profile(task: Mapping[str, Any]) -> dict[str, Any]:
    final_states = task["goal"]["acceptable_states"]
    final_entity_count = min(
        len(state["selected_entity_ids"]) for state in final_states
    )
    required_field_count = max(len(state["fields"]) for state in final_states)
    if task["track"] == "graft":
        initial_options = len(task["graft"]["initial_goal"]["acceptable_states"])
        change = {
            "refine": "细化原目标：原目标仍有效，但条件更具体",
            "relax": "放宽原约束：保留原有合法选择，并允许更多选择",
            "augment": "增加目标：保留原目标，并新增一个目标",
            "replace": "替换目标：撤销旧目标，改为新目标",
            "drop": "撤回目标：删除不再需要的目标",
        }[task["graft"]["transition"]]
    elif task["track"] == "refresh":
        initial_options = 1
        change = (
            "新画面出现后仍保持原目标"
            if task["refresh"]["policy"] == "retain"
            else "新画面出现后更新目标"
        )
    else:
        initial_options = len(task["clarification"]["candidate_ids"])
        change = "确认前有两个合理候选，回答后只剩一个"
    return {
        "initial_options": initial_options,
        "final_entity_count": final_entity_count,
        "required_field_count": required_field_count,
        "change": change,
    }


def _scenario_evidence(source_record: Mapping[str, Any]) -> dict[str, Any]:
    construct = source_record.get("construct", {})
    source = source_record.get("source", {})
    origin = str(construct.get("event_origin", ""))
    label, explanation = EVENT_ORIGINS.get(
        origin, ("来源未说明", "当前来源记录没有说明任务变化如何产生。")
    )
    events = source.get("observed_events", [])
    if not isinstance(events, list):
        events = []
    claims = source_record.get("supported_claims", [])
    if not isinstance(claims, list):
        claims = []
    return {
        "origin": origin,
        "label": label,
        "explanation": explanation,
        "observed_events": [event for event in events if isinstance(event, dict)],
        "supported_claims": [claim for claim in claims if isinstance(claim, dict)],
    }


def _motivation_view(
    task: Mapping[str, Any], source_record: Mapping[str, Any]
) -> dict[str, str]:
    construct = source_record.get("construct", {})
    research_question = str(construct.get("research_question", "")).strip()
    visual_dependency = str(construct.get("visual_dependency_claim", "")).strip()
    required_change = str(construct.get("required_behavior_change", "")).strip()
    if task["track"] == "graft":
        track_test = (
            "补充要求必须在同一现实操作仍在进行时到达，并改变尚未完成的网页选择；"
            "普通评分、价格或排序条件本身不构成眼镜难点。"
        )
        task_risk = (
            "这道题只在开始提供现实画面，后续变化来自用户补充。请确认补充与现场活动有关，"
            "而且确实改变了网页选择。"
        )
    elif task["track"] == "refresh":
        track_test = (
            "后来画面必须来自同一连续事件，并使 Agent 决定保持原目标或撤销后重选。"
        )
        task_risk = ""
    else:
        track_test = "歧义应来自现实场景中的多个合理指代，用户回答应能明确所指对象。"
        task_risk = ""
    return {
        "project_question": ("现实画面和后续信息是否共同决定需要完成的网页操作？"),
        "static_replacement_test": (
            "保留用户指令、后续信息和网站内容，只去掉现实画面，是否仍能直接确定答案？"
            "同时核对后续信息是否使模型需要重新判断网页选择。"
        ),
        "research_question": research_question or "来源记录没有写明研究问题。",
        "visual_dependency": visual_dependency
        or "来源记录没有写明画面为什么不可替代。",
        "required_change": required_change
        or "来源记录没有写明新证据到达后必须改变什么行为。",
        "track_test": track_test,
        "task_risk": task_risk,
    }


def _flatten_visible_value(label: str, value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        rows: list[dict[str, str]] = []
        for key, child in value.items():
            rows.extend(
                _flatten_visible_value(f"{label} · {str(key).replace('_', ' ')}", child)
            )
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value, start=1):
            if isinstance(child, dict):
                child_label = str(child.get("heading") or child.get("label") or index)
                child_value = child.get("text", child)
                rows.extend(
                    _flatten_visible_value(f"{label} · {child_label}", child_value)
                )
            else:
                rows.extend(_flatten_visible_value(f"{label} · {index}", child))
        return rows
    return [{"label": label, "value": str(value)}]


def _visible_entity_fields(
    site: str, entity: Mapping[str, Any]
) -> list[dict[str, str]]:
    if site == "egomart":
        rating, count = rating_of(dict(entity))
        rows = [
            {"label": "品牌", "value": str(entity.get("brand") or "未注明")},
            {"label": "价格", "value": str(entity.get("price") or "未提供")},
        ]
        if rating is not None and count is not None:
            rows.append(
                {"label": "平均评分", "value": f"{rating:g} / 5（{count:,} 条评分）"}
            )
        for key, value in visible_attributes(dict(entity)).items():
            rows.extend(_flatten_visible_value(str(key).replace("_", " "), value))
        return rows

    ignored = {
        "entity_id",
        "title",
        "name",
        "summary",
        "description",
        "source",
        "facts",
        "image_path",
        "thumbnail_path",
        "photo_path",
        "image_provenance",
        "text_provenance",
    }
    rows = []
    for key, value in entity.items():
        if key not in ignored and not key.endswith("_provenance"):
            rows.extend(_flatten_visible_value(str(key).replace("_", " "), value))
    if site == "egomaps":
        facts = entity.get("facts")
        if isinstance(facts, dict):
            for key, value in facts.items():
                rows.extend(_flatten_visible_value(str(key).replace("_", " "), value))
    return rows


def _goal_final_entity_ids(task: Mapping[str, Any]) -> set[str]:
    return set(goal_entity_ids(task["goal"]))


def _instruction_preview(task: Mapping[str, Any]) -> str:
    instructions = task.get("instructions") or {}
    if task["track"] == "graft":
        text = str(instructions.get("a") or "")
    elif task["track"] == "refresh":
        text = str(instructions.get("base") or "")
    else:
        text = str(instructions.get("ambiguous") or "")
    text = " ".join(text.split())
    if len(text) > 88:
        return text[:87] + "…"
    return text


def _review_briefing(task: Mapping[str, Any]) -> dict[str, Any]:
    instructions = task.get("instructions") or {}
    if task["track"] == "graft":
        lines = [
            {"label": "最初要求", "text": str(instructions.get("a") or "")},
            {
                "label": "中途补充",
                "text": str(instructions.get("b") or ""),
                "note": "按证据时间线在登记步释放。",
            },
        ]
    elif task["track"] == "refresh":
        policy = task["refresh"]["policy"]
        lines = [
            {"label": "用户要求", "text": str(instructions.get("base") or "")},
            {
                "label": "目标规则",
                "text": (
                    "后来画面出现后，仍保持最初目标。"
                    if policy == "retain"
                    else "后来画面出现后，改为后来出现的目标。"
                ),
            },
        ]
    else:
        clarification = task["clarification"]
        lines = [
            {"label": "最初要求", "text": str(instructions.get("ambiguous") or "")},
            {
                "label": "澄清后的完整要求",
                "text": str(instructions.get("oracle") or ""),
            },
            {
                "label": "用户的标准回答",
                "text": str(clarification.get("canonical_answer") or ""),
                "note": SLOT_LABELS.get(
                    str(clarification.get("valid_slot")), "一次澄清"
                ),
            },
        ]
    packets = []
    for packet in task["visual_packets"]:
        packet_id = str(packet["id"])
        packets.append(
            {
                "id": packet_id,
                "label": "后来画面" if packet_id == "V1" else "开始可见",
                "frame_paths": list(packet["frame_paths"]),
            }
        )
    return {
        "instruction_lines": lines,
        "packets": packets,
        "final_ids": sorted(_goal_final_entity_ids(task)),
    }


ACCEPTANCE_REPORT = REPORTS["acceptance"].path
ADVISORY_CHECKS = {
    "text_pinning": "文本锁定",
    "necessity": "渐进证据必要性",
}


def _acceptance_advisories(root: Path, task_id: str) -> list[str]:
    """Advisory findings from the last acceptance audit, shown as review warnings."""

    path = root / ACCEPTANCE_REPORT
    if not path.is_file():
        return []
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    row = (report.get("tasks") or {}).get(task_id) if isinstance(report, dict) else None
    if not isinstance(row, Mapping):
        return []
    warnings: list[str] = []
    for name, label in ADVISORY_CHECKS.items():
        check = (row.get("checks") or {}).get(name)
        if isinstance(check, Mapping) and check.get("status") == "warn":
            detail = "；".join(str(item) for item in check.get("warnings", []))
            warnings.append(f"审计提示（{label}）：{detail or '见 acceptance 报告'}")
    return warnings


class ReviewWorkspace:
    """Deep module that turns repository truth into an isolated review workflow."""

    def __init__(
        self,
        root: Path,
        reviewer_id: str,
        spec_path: Path,
        spec: dict[str, Any],
        tasks: dict[str, dict[str, Any]],
        ordered_ids: list[str],
        selected_ids: list[str],
        scope: str,
        plan: ExperimentPlan,
    ) -> None:
        self.root = root
        self.reviewer_id = reviewer_id
        self.spec_path = spec_path
        self.spec = spec
        self.tasks = tasks
        self.ordered_ids = ordered_ids
        self.selected_ids = selected_ids
        self.scope = scope
        self.plan = plan
        self.materials = review_materials.ReviewMaterials(
            root, spec_path.relative_to(root).as_posix()
        )

    @classmethod
    def open(
        cls,
        root: str | Path,
        reviewer_id: str,
        *,
        spec: str = review_materials.REVIEW_SPEC,
        include_all_tasks: bool = False,
        task_ids: Iterable[str] | None = None,
    ) -> ReviewWorkspace:
        workspace = Path(root).resolve()
        if not REVIEWER_ID.fullmatch(reviewer_id):
            raise ReviewWorkspaceError(
                "reviewer id must use letters, numbers, underscores, or hyphens"
            )
        spec_path = resolve_within(workspace, spec, context="construction review spec")
        payload = _load_object(spec_path, "construction review spec")
        cls._validate_spec(payload)
        plan_path = resolve_within(
            workspace, TASK_REVIEW_PLAN, context="experiment plan"
        )
        plan = load_experiment_plan(plan_path)
        selection_path = resolve_within(workspace, SELECTION, context="task selection")
        selection = _load_object(selection_path, "task selection")
        selected_candidates = _review_queue(selection)
        task_root = resolve_within(
            workspace, TASKS, context="task directory", must_exist=False
        )
        if not task_root.is_dir():
            raise ReviewWorkspaceError("task directory is missing")
        loaded = load_tasks(task_root)
        # The review queue must remain inspectable when one candidate has missing
        # assets. Each case validates its own release material and fails closed.
        validate_dataset(loaded, root=workspace, require_assets=False)
        tasks = {task["id"]: task for task in loaded}
        if not set(selected_candidates) <= set(tasks):
            missing = sorted(set(selected_candidates) - set(tasks))
            raise ReviewWorkspaceError(
                f"selected executable tasks are missing: {missing}"
            )
        requested_ids = None if task_ids is None else list(dict.fromkeys(task_ids))
        if requested_ids is not None:
            unknown = set(requested_ids) - set(selected_candidates)
            if unknown:
                raise ReviewWorkspaceError(
                    f"unknown explicit review tasks: {sorted(unknown)}"
                )
            ordered_ids = [task["id"] for task in loaded if task["id"] in requested_ids]
            review_tasks = {task_id: tasks[task_id] for task_id in ordered_ids}
            scope = "explicit"
        elif include_all_tasks:
            ordered_ids = selected_candidates
            review_tasks = {task_id: tasks[task_id] for task_id in ordered_ids}
            scope = "all"
        else:
            ordered_ids = _targeted_review_queue(
                SynthesisWorkflow(workspace).status(selected_candidates),
                selected_candidates,
            )
            review_tasks = {task_id: tasks[task_id] for task_id in ordered_ids}
            scope = "targeted_confidence_review"
        return cls(
            workspace,
            reviewer_id,
            spec_path,
            payload,
            review_tasks,
            ordered_ids,
            selected_candidates,
            scope,
            plan,
        )

    @staticmethod
    def _validate_spec(spec: Mapping[str, Any]) -> None:
        try:
            review_materials.validate_review_spec(spec)
        except review_materials.ReviewMaterialError as exc:
            raise ReviewWorkspaceError(str(exc)) from exc

    def view(
        self,
        task_id: str | None = None,
        *,
        track: str | None = None,
        site: str | None = None,
        status: str | None = None,
        lifecycle: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Show current materials, optionally sharing this request's status result."""

        self._ensure_review_active()
        if task_id is not None and task_id not in self.tasks:
            raise ReviewWorkspaceError(f"unknown review task: {task_id}")
        if lifecycle is None:
            lifecycle = SynthesisWorkflow(self.root).status(
                self.ordered_ids if task_id is None else [task_id]
            )
        if task_id is None:
            return self._progress(lifecycle, track=track, site=site, status=status)
        case = self.review_materials(task_id)
        lifecycle = lifecycle["tasks"][task_id]
        source_review_pending = lifecycle["next_action"] == "resolve_source_review"
        if (
            lifecycle["status"] != TaskStatus.READY
            and lifecycle["next_action"]
            not in {
                "human_review",
                "verify_development_evidence",
                "run_development_trial",
                "review_development_trial",
                "review_stream_window",
            }
            and not source_review_pending
        ):
            message = (
                "当前任务尚不符合人工质量审核条件，请先完成生命周期要求的审核或修复。"
            )
            case["quality_review_blockers"].append(message)
            case["review_warnings"].append(message)
        return case

    def pending(self, lifecycle: Mapping[str, Any]) -> ReviewWorkspace:
        """Select the human queue from an all-task workspace and current status."""

        ordered_ids = _targeted_review_queue(lifecycle, self.selected_ids)
        return ReviewWorkspace(
            self.root,
            self.reviewer_id,
            self.spec_path,
            self.spec,
            {task_id: self.tasks[task_id] for task_id in ordered_ids},
            ordered_ids,
            self.selected_ids,
            "targeted_confidence_review",
            self.plan,
        )

    def review_materials(self, task_id: str) -> dict[str, Any]:
        """Read the reviewed materials without deriving task lifecycle state."""

        try:
            task = self.tasks[task_id]
        except KeyError as exc:
            raise ReviewWorkspaceError(f"unknown review task: {task_id}") from exc
        task = _load_object(
            self.root / TASKS / task["track"] / f"{task_id}.json",
            f"review task {task_id}",
        )
        if task.get("id") != task_id:
            raise StaleReviewError("任务清单已经变化，请重新查看")
        try:
            return self._case(task)
        except review_materials.ReviewMaterialError as exc:
            raise ReviewWorkspaceError(str(exc)) from exc

    def _ensure_review_active(self) -> None:
        if _load_object(self.spec_path, "construction review spec") != self.spec:
            raise StaleReviewError("审核规范已经变化，请重启人工审核网站")
        selection = _load_object(self.root / SELECTION, "task selection")
        if _review_queue(selection) != self.selected_ids or any(
            not (self.root / TASKS / task["track"] / f"{task_id}.json").is_file()
            for task_id, task in self.tasks.items()
        ):
            raise StaleReviewError("任务清单已经变化，请重启人工审核网站")

    def submit(
        self,
        task_id: str,
        response: Mapping[str, Any],
        *,
        expected_revision: str,
        lifecycle: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        case = self.view(task_id, lifecycle=lifecycle)
        if case["revision"] != expected_revision:
            raise StaleReviewError("审核材料已经变化，请重新查看后提交")
        destination = self._submission_path(task_id)
        if destination.exists() and case["own_status"] == "submitted":
            raise AlreadySubmittedError("这名审核者已经提交过该任务")
        submission = self._validated_response(case, response)
        if destination.exists():
            previous = destination.read_bytes()
            digest = hashlib.sha256(previous).hexdigest()
            history = (
                destination.parent / "history" / self.reviewer_id / f"{digest}.json"
            )
            write_bytes_atomic(history, previous)
        write_json_atomic(destination, submission)
        return {
            "task_id": task_id,
            "reviewer": self.reviewer_id,
            "decision": submission["decision"],
            "submission": destination.relative_to(self.root).as_posix(),
            "sha256": _file_sha256(destination),
        }

    def _progress(
        self,
        lifecycle: Mapping[str, Any],
        *,
        track: str | None,
        site: str | None,
        status: str | None,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        submission_counts = {"todo": 0, "submitted": 0, "stale": 0, "invalid": 0}
        lifecycle = lifecycle["tasks"]
        for task_id in self.ordered_ids:
            task = self.tasks[task_id]
            own_status = self._own_status(task)
            submission_counts[own_status] += 1
            if track and task["track"] != track:
                continue
            if site and task["site"] != site:
                continue
            task_status = lifecycle[task_id]
            if status and task_status["status"] != status:
                continue
            rows.append(
                {
                    "id": task_id,
                    "track": task["track"],
                    "site": task["site"],
                    "transition": task_transition(task),
                    "transition_label": TRANSITION_LABELS[task_transition(task)],
                    "status": task_status["status"],
                    "status_label": task_status["status_label"],
                    "review_submission_status": own_status,
                    "preview": _instruction_preview(task),
                    "focus": _task_focus(task),
                }
            )
        return {
            "reviewer": self.reviewer_id,
            "scope": self.scope,
            "total": len(self.ordered_ids),
            "submission_counts": submission_counts,
            "tasks": rows,
        }

    def _case(self, task: dict[str, Any]) -> dict[str, Any]:
        evidence = self.materials.case(task)
        source_record = evidence["source_record"]
        source_dataset = evidence["source_dataset"]
        source_video = evidence["source_video"]
        site_data = load_site_data(self.root, task["site"], require_assets=False)
        by_id = {entity["entity_id"]: entity for entity in site_data["entities"]}
        roles = review_materials.referenced_entities(task, source_record)
        missing_entity_ids = sorted(set(roles) - set(by_id))
        final_ids = _goal_final_entity_ids(task)
        entities = []
        for entity_id, role in roles.items():
            if entity_id not in by_id:
                continue
            entity = by_id[entity_id]
            visual_path = entity_visual_path(task["site"], entity)
            visual_exists = bool(
                visual_path
                and resolve_within(
                    self.root,
                    visual_path,
                    context=f"{task['id']} website image",
                    must_exist=False,
                ).is_file()
            )
            entities.append(
                {
                    "roles": role,
                    **entity,
                    "is_final": entity_id in final_ids,
                    "visual_path": visual_path if visual_exists else None,
                    "visible_fields": _visible_entity_fields(task["site"], entity),
                }
            )
        review_warnings = []
        for reuse in evidence["human_review_reuse"]:
            review_warnings.append(
                "旧人工意见已保留；仅新增问题待判："
                + "；".join(reuse["affected_conclusions"])
                if reuse["outcome"] == "partial"
                else "旧人工意见已保留，当前具名复核已确认完整结论。"
            )
        release_blockers = []
        scenario_evidence = _scenario_evidence(source_record)
        motivation = _motivation_view(task, source_record)
        try:
            validate_task(task, site_data, root=self.root, require_assets=True)
        except (ValueError, FileNotFoundError) as exc:
            message = f"任务发布材料不完整：{exc}"
            review_warnings.append(message)
            release_blockers.append(message)
        try:
            validate_source_lineage(task, root=self.root)
        except ValueError as exc:
            message = f"任务与来源证据的版本不一致：{exc}"
            review_warnings.append(message)
            release_blockers.append(message)
        pending_source_items = [
            (
                "；".join(
                    str(item[key])
                    for key in ("question", "decisive_reason")
                    if item.get(key)
                )
                if isinstance(item, Mapping)
                else str(item)
            )
            for field in ("blocking_checks", "unresolved")
            for item in source_record.get(field, [])
            if str(item).strip()
        ]
        if missing_entity_ids:
            message = "来源画面中的任务相关对象没有对应的网站实体：" + "、".join(
                missing_entity_ids
            )
            review_warnings.append(message)
            release_blockers.append(message)
        if not scenario_evidence["supported_claims"]:
            message = "来源记录没有逐帧支持的关键事实，这道题目前不能保留。"
            review_warnings.append(message)
            release_blockers.append(message)
        if task["track"] == "refresh" and scenario_evidence["origin"] != "observed":
            message = "Refresh 必须来自连续视频中真实发生的变化，当前来源不符合要求。"
            review_warnings.append(message)
            release_blockers.append(message)
        if len(entities) < 2:
            review_warnings.append(
                "当前任务只关联了一个网页候选，无法直接核对画面中的干扰对象是否也有对应网页实体。"
                "请打开完整网站检查；如果仍找不到可验证的干扰项，应将网站目标与干扰项判为不通过。"
            )
        media_fields = {
            "thumbnail_path",
            "channel_title",
            "published_at",
            "duration_seconds",
            "view_count",
        }
        if task["site"] == "egomedia" and any(
            not media_fields <= set(entity) for entity in entities
        ):
            message = (
                "媒体网站还没有绑定独立缩略图、频道、发布时间和观看量，当前页面不足以模拟真实视频搜索。"
                "这道题只能作为候选，不能直接通过正式发布审核。"
            )
            review_warnings.append(message)
            release_blockers.append(message)
        if task["site"] == "egomaps" and any(
            not entity.get("location") or not entity.get("address")
            for entity in entities
        ):
            message = (
                "地图网站还没有完整的地址、坐标和地图路线视图。"
                "这道题只能作为候选，不能直接通过正式发布审核。"
            )
            review_warnings.append(message)
            release_blockers.append(message)
        quality_review_blockers = list(release_blockers)
        if pending_source_items:
            message = (
                f"来源记录仍有 {len(pending_source_items)} 项未解决；"
                "质量结论不会解除这些发布阻塞。"
            )
            review_warnings.append(message)
            release_blockers.append(message)
        review_warnings.extend(_acceptance_advisories(self.root, task["id"]))
        artifacts = evidence["reviewed_artifacts"]
        checks = evidence["checks"]
        experiment = evidence["experiment"]
        briefing = _review_briefing(task)
        # New fields per §8.3
        tempting_ids = task.get("truth", {}).get("tempting_referent_ids", [])
        deixis_level = task.get("clarification", {}).get("deixis_level")
        transition = task_transition(task)
        index = self.ordered_ids.index(task["id"])
        revision = evidence["revision"]
        case = {
            "task": task,
            "focus": _task_focus(task),
            "design_profile": _design_profile(task),
            "scenario_evidence": scenario_evidence,
            "motivation": motivation,
            "pending_source_items": pending_source_items,
            "checks": checks,
            "reviewer_policy": self.spec["reviewer_policy"],
            "experiment": experiment,
            "briefing": briefing,
            "neighbors": {
                "previous": self.ordered_ids[index - 1] if index else None,
                "next": (
                    self.ordered_ids[index + 1]
                    if index + 1 < len(self.ordered_ids)
                    else None
                ),
            },
            "role_labels": ROLE_LABELS,
            "site_label": SITE_LABELS.get(str(task["site"]), task["site"]),
            "track_label": TRACK_LABELS.get(str(task["track"]), task["track"]),
            "transition_label": TRANSITION_LABELS[transition],
            "entities": entities,
            "review_warnings": review_warnings,
            "release_blockers": release_blockers,
            "quality_review_blockers": quality_review_blockers,
            "site_entity_count": len(site_data["entities"]),
            "source_record": source_record,
            "source_dataset": source_dataset,
            "source_video": source_video,
            "tempting_referent_ids": tempting_ids,
            "deixis_level": deixis_level,
            "transition": transition,
            "reviewed_artifacts": artifacts,
            "revision": revision,
            "human_review_reuse": evidence["human_review_reuse"],
        }
        case["own_status"] = self._own_status(task, case=case)
        return case

    def _submission_path(self, task_id: str) -> Path:
        return self.root / SUBMISSIONS / task_id / f"{self.reviewer_id}.json"

    def _own_status(
        self,
        task: Mapping[str, Any],
        *,
        case: Mapping[str, Any] | None = None,
    ) -> str:
        path = self._submission_path(str(task["id"]))
        if not path.is_file():
            return "todo"
        try:
            payload = _load_object(path, str(path))
        except ReviewWorkspaceError:
            return "invalid"
        if case is None:
            return self._case(dict(task))["own_status"]
        try:
            validate_review_submission(
                payload, path=path.relative_to(self.root), case=case
            )
        except StaleSubmissionError:
            return "stale"
        except ReviewAuditError:
            return "invalid"
        return "submitted"

    def _validated_response(
        self,
        case: Mapping[str, Any],
        response: Mapping[str, Any],
    ) -> dict[str, Any]:
        task = case["task"]
        purpose = response.get("purpose")
        if purpose is not None and purpose != "quality_control":
            raise ReviewWorkspaceError("无效的审核用途")
        decision = response.get("decision")
        note = str(response.get("note", "")).strip()
        if decision not in {"admit", "reject"}:
            raise ReviewWorkspaceError("请选择保留或指出问题")
        if decision == "admit" and case["quality_review_blockers"]:
            raise ReviewWorkspaceError(
                "这道题仍有未解决的验收问题，目前不能保留："
                + str(case["quality_review_blockers"][0])
            )
        if decision == "reject" and not note:
            raise ReviewWorkspaceError("发现问题时请简要说明")
        granularity = task["goal"]["target_granularity"]
        if granularity not in GRANULARITIES:
            raise ReviewWorkspaceError("任务中的目标粒度无效")
        unresolved = [note] if decision == "reject" else []
        return {
            "schema_version": 5,
            **({"purpose": purpose} if purpose is not None else {}),
            "reviewer": self.reviewer_id,
            "task_id": task["id"],
            "decision": decision,
            "note": note,
            "target_granularity": granularity,
            "unresolved": unresolved,
            "reviewed_artifacts": case["reviewed_artifacts"],
            "revision": case["revision"],
            "submitted_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
