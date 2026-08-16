"""One local interface for reviewing benchmark construction validity."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .construction import opaque_item_id
from .environment import entity_visual_path, load_site_data
from .experiment import PRIMARY_CONDITION, SUPPORTED_ARMS
from .lineage import validate_source_lineage
from .protocol import RunSession, initial_visible_packets
from .run_ledger import canonical_sha256
from .schema import load_tasks, validate_dataset, validate_task
from .workspace import resolve_within, write_json_atomic
from sites.egomart.pages import rating_of, visible_attributes


REVIEW_SPEC = "data/reviews/construction.json"
SELECTION = "data/selection.json"
TASKS = "data/tasks"
SUBMISSIONS = "data/reviews/submissions"
REVIEWER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
GRANULARITIES = {"category", "brand", "variant", "exact_instance"}
SITE_DATA_FILES = {
    "egomart": "sites/egomart/products.json",
    "egodocs": "sites/egodocs/documents.json",
    "egomedia": "sites/egomedia/videos.json",
    "egomaps": "sites/egomaps/places.json",
}
EVENT_ORIGINS = {
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


class ReviewWorkspaceError(ValueError):
    """Raised when a review cannot be shown or submitted safely."""


class StaleReviewError(ReviewWorkspaceError):
    """Raised when reviewed material changed after the page was opened."""


class AlreadySubmittedError(ReviewWorkspaceError):
    """Raised when one reviewer tries to replace a final submission."""


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewWorkspaceError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReviewWorkspaceError(f"{context}: JSON root must be an object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_task_ids(primary: Any) -> list[str]:
    result: list[str] = []

    def visit(value: Any) -> None:
        if isinstance(value, str):
            result.append(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(primary)
    if not result or len(result) != len(set(result)):
        raise ReviewWorkspaceError("selection primary tasks must be non-empty and unique")
    return result


def _check_definitions(spec: Mapping[str, Any], track: str) -> list[dict[str, str]]:
    groups = spec.get("checks")
    if not isinstance(groups, dict):
        raise ReviewWorkspaceError("construction review spec needs checks")
    checks = [*groups.get("common", []), *groups.get(track, [])]
    if not checks or any(
        not isinstance(check, dict)
        or set(check) != {"id", "question"}
        or not str(check["id"]).strip()
        or not str(check["question"]).strip()
        for check in checks
    ):
        raise ReviewWorkspaceError(f"construction review checks are invalid for {track}")
    ids = [str(check["id"]) for check in checks]
    if len(ids) != len(set(ids)):
        raise ReviewWorkspaceError(f"construction review checks are duplicated for {track}")
    return [{"id": str(check["id"]), "question": str(check["question"])} for check in checks]


def _referenced_entities(
    task: Mapping[str, Any], source_record: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    final_ids = list(dict.fromkeys(
        entity_id
        for state in task["goal"]["acceptable_states"]
        for entity_id in state["selected_entity_ids"]
    ))
    roles: dict[str, list[str]] = {entity_id: ["最终目标"] for entity_id in final_ids}
    if task["track"] == "graft":
        initial_ids = list(dict.fromkeys(
            entity_id
            for state in task["graft"]["initial_goal"]["acceptable_states"]
            for entity_id in state["selected_entity_ids"]
        ))
        for entity_id in initial_ids:
            roles.setdefault(entity_id, []).append("最初目标")
    elif task["track"] == "refresh":
        roles.setdefault(task["refresh"]["initial_entity_id"], []).append("V0目标")
        blueprint = (source_record or {}).get("construct", {}).get("task_blueprint", {})
        later_role = blueprint.get("later_distractor_role")
        if isinstance(later_role, str) and later_role:
            explicit_ids = blueprint.get("environment_role_ids", {})
            later_id = str(explicit_ids.get(later_role)) if isinstance(
                explicit_ids, dict
            ) and explicit_ids.get(later_role) else opaque_item_id(
                str(task["id"]), later_role
            )
            roles.setdefault(later_id, []).append("后来出现的对象")
    else:
        for entity_id in task["clarification"]["candidate_ids"]:
            roles.setdefault(entity_id, []).append("澄清候选")
    return roles


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
            "C": "补充后缩小原有候选",
            "P": "补充后增加一个并行目标",
            "R": "补充后增加一个与原目标相关的目标",
        }[task["graft"]["integration_type"]]
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
    minimum_key_actions = (
        1  # search
        + final_entity_count  # open the required records
        + final_entity_count  # select the required records
        + required_field_count
        + 1  # submit
        + (1 if task["track"] == "clarify" else 0)
    )
    return {
        "initial_options": initial_options,
        "final_entity_count": final_entity_count,
        "required_field_count": required_field_count,
        "minimum_key_actions": minimum_key_actions,
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


def _motivation_view(task: Mapping[str, Any], source_record: Mapping[str, Any]) -> dict[str, str]:
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
            "这道题只有任务开始时的一组现实画面。请重点判断：如果把眼镜画面换成"
            "普通上传照片，并保留同样的中途补充，是否仍然测到完全相同的能力；"
            "如果是，这道题不进入核心集合。"
        )
    elif task["track"] == "refresh":
        track_test = (
            "后来画面必须来自同一连续事件，并使 Agent 决定保持原目标或撤销后重选。"
        )
        task_risk = ""
    else:
        track_test = (
            "歧义必须存在于用户当前所处的现实场景，并且 Agent 在选择网页目标前必须先确认。"
        )
        task_risk = ""
    return {
        "project_question": (
            "这道题是否必须放在一个短时、免手持的眼镜会话中，才能测到 Agent 何时可以对网页目标作出承诺？"
        ),
        "static_replacement_test": (
            "把任务改成开始时的一张照片和一次完整指令。如果仍然测到完全相同的决定、时机和网页动作，这道题不进入核心 Benchmark。"
        ),
        "research_question": research_question or "来源记录没有写明研究问题。",
        "visual_dependency": visual_dependency or "来源记录没有写明画面为什么不可替代。",
        "required_change": required_change or "来源记录没有写明新证据到达后必须改变什么行为。",
        "track_test": track_test,
        "task_risk": task_risk,
    }


def _flatten_visible_value(label: str, value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        rows: list[dict[str, str]] = []
        for key, child in value.items():
            rows.extend(_flatten_visible_value(
                f"{label} · {str(key).replace('_', ' ')}", child
            ))
        return rows
    if isinstance(value, list):
        rows = []
        for index, child in enumerate(value, start=1):
            if isinstance(child, dict):
                child_label = str(child.get("heading") or child.get("label") or index)
                child_value = child.get("text", child)
                rows.extend(_flatten_visible_value(f"{label} · {child_label}", child_value))
            else:
                rows.extend(_flatten_visible_value(f"{label} · {index}", child))
        return rows
    return [{"label": label, "value": str(value)}]


def _visible_entity_fields(site: str, entity: Mapping[str, Any]) -> list[dict[str, str]]:
    if site == "egomart":
        rating, count = rating_of(dict(entity))
        rows = [
            {"label": "品牌", "value": str(entity.get("brand") or "未注明")},
            {"label": "价格", "value": str(entity.get("price") or "未提供")},
            {"label": "平均评分", "value": f"{rating:g} / 5（{count:,} 条评分）"},
        ]
        for key, value in visible_attributes(dict(entity)).items():
            rows.extend(_flatten_visible_value(str(key).replace("_", " "), value))
        return rows

    ignored = {
        "entity_id", "title", "name", "summary", "description", "source",
        "facts", "image_path", "thumbnail_path", "photo_path",
        "image_provenance", "text_provenance",
    }
    rows = []
    for key, value in entity.items():
        if key not in ignored and not key.endswith("_provenance"):
            rows.extend(_flatten_visible_value(str(key).replace("_", " "), value))
    return rows


def _experiment_view(task: dict[str, Any]) -> dict[str, Any]:
    condition = PRIMARY_CONDITION[task["track"]]
    session = RunSession(task, condition)
    payload = session.initial_payload()
    event = session.advance(session.event_step) if session.has_pending_scheduled_event() else []
    return {
        "condition": condition,
        "instruction": payload["instruction"],
        "visible_packets": sorted(initial_visible_packets(task, condition)),
        "clarification_available": payload["clarification_available"],
        "scheduled_event": event[0] if event else None,
    }


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
        scope: str,
    ) -> None:
        self.root = root
        self.reviewer_id = reviewer_id
        self.spec_path = spec_path
        self.spec = spec
        self.tasks = tasks
        self.ordered_ids = ordered_ids
        self.scope = scope

    @classmethod
    def open(
        cls,
        root: str | Path,
        reviewer_id: str,
        *,
        spec: str = REVIEW_SPEC,
        include_all_tasks: bool = False,
    ) -> "ReviewWorkspace":
        workspace = Path(root).resolve()
        if not REVIEWER_ID.fullmatch(reviewer_id):
            raise ReviewWorkspaceError(
                "reviewer id must use letters, numbers, underscores, or hyphens"
            )
        spec_path = resolve_within(workspace, spec, context="construction review spec")
        payload = _load_object(spec_path, "construction review spec")
        cls._validate_spec(payload)
        selection_path = resolve_within(workspace, SELECTION, context="task selection")
        selection = _load_object(selection_path, "task selection")
        if selection.get("status") == "paused_for_glass_use_redesign":
            raise ReviewWorkspaceError(
                "当前旧任务集已暂停人工审核；请先按新的 Glass-Use 准入标准重新构造任务"
            )
        selected = _selected_task_ids(selection.get("primary"))
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
        if not set(selected) <= set(tasks):
            missing = sorted(set(selected) - set(tasks))
            raise ReviewWorkspaceError(
                f"selected executable tasks are missing: {missing}"
            )
        if include_all_tasks:
            ordered_ids = [task["id"] for task in loaded]
            review_tasks = tasks
            scope = "all"
        else:
            ordered_ids = selected
            review_tasks = {task_id: tasks[task_id] for task_id in selected}
            scope = "primary"
        return cls(
            workspace, reviewer_id, spec_path, payload,
            review_tasks, ordered_ids, scope,
        )

    @staticmethod
    def _validate_spec(spec: Mapping[str, Any]) -> None:
        if spec.get("schema_version") != 1:
            raise ReviewWorkspaceError("construction review spec schema_version must be 1")
        policy = spec.get("reviewer_policy")
        expected = {
            "required_independent_reviews": 2,
            "reviewers_must_not_see_each_others_submission": True,
            "disagreement_outcome": "reject",
            "model_output_may_influence_decision": False,
        }
        if policy != expected:
            raise ReviewWorkspaceError("construction review policy is invalid")
        for track in SUPPORTED_ARMS:
            _check_definitions(spec, track)

    def view(
        self,
        task_id: str | None = None,
        *,
        track: str | None = None,
        site: str | None = None,
        status: str | None = None,
    ) -> dict[str, Any]:
        self._ensure_review_active()
        if task_id is None:
            return self._progress(track=track, site=site, status=status)
        try:
            task = self.tasks[task_id]
        except KeyError as exc:
            raise ReviewWorkspaceError(f"unknown review task: {task_id}") from exc
        return self._case(task)

    def _ensure_review_active(self) -> None:
        selection = _load_object(self.root / SELECTION, "task selection")
        if selection.get("status") == "paused_for_glass_use_redesign":
            raise ReviewWorkspaceError(
                "当前旧任务集已暂停人工审核；请先按新的 Glass-Use 准入标准重新构造任务"
            )
        if self.scope == "all":
            current_ids = [
                task["id"] for task in load_tasks(self.root / TASKS)
            ]
        else:
            current_ids = _selected_task_ids(selection.get("primary"))
        if current_ids != self.ordered_ids:
            raise StaleReviewError("任务清单已经变化，请重启人工审核网站")

    def submit(
        self,
        task_id: str,
        response: Mapping[str, Any],
        *,
        expected_revision: str,
    ) -> dict[str, Any]:
        case = self.view(task_id)
        if case["revision"] != expected_revision:
            raise StaleReviewError("审核材料已经变化，请重新查看后提交")
        destination = self._submission_path(task_id)
        if destination.exists():
            raise AlreadySubmittedError("这名审核者已经提交过该任务")
        submission = self._validated_response(case, response)
        write_json_atomic(destination, submission)
        return {
            "task_id": task_id,
            "reviewer": self.reviewer_id,
            "decision": submission["decision"],
            "submission": destination.relative_to(self.root).as_posix(),
            "sha256": _file_sha256(destination),
        }

    def _progress(
        self, *, track: str | None, site: str | None, status: str | None,
    ) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        counts = {"todo": 0, "submitted": 0, "stale": 0, "invalid": 0}
        for task_id in self.ordered_ids:
            task = self.tasks[task_id]
            own_status = self._own_status(task)
            counts[own_status] += 1
            if track and task["track"] != track:
                continue
            if site and task["site"] != site:
                continue
            if status and own_status != status:
                continue
            rows.append({
                "id": task_id,
                "track": task["track"],
                "site": task["site"],
                "status": own_status,
            })
        return {
            "reviewer": self.reviewer_id,
            "scope": self.scope,
            "total": len(self.ordered_ids),
            "counts": counts,
            "tasks": rows,
        }

    def _case(self, task: dict[str, Any]) -> dict[str, Any]:
        source_record, source_dataset, source_video = self._source_material(task)
        site_data = load_site_data(self.root, task["site"], require_assets=False)
        by_id = {entity["entity_id"]: entity for entity in site_data["entities"]}
        roles = _referenced_entities(task, source_record)
        missing_entity_ids = sorted(set(roles) - set(by_id))
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
            entities.append({
                "roles": role,
                **entity,
                "visual_path": visual_path if visual_exists else None,
                "visible_fields": _visible_entity_fields(task["site"], entity),
            })
        review_warnings = []
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
            *[
                str(item) for item in source_record.get("blocking_checks", [])
                if str(item).strip()
            ],
            *[
                str(item) for item in source_record.get("unresolved", [])
                if str(item).strip()
            ],
        ]
        if pending_source_items:
            message = (
                f"来源记录仍有 {len(pending_source_items)} 项未解决，完成前不能保留。"
            )
            review_warnings.append(message)
            release_blockers.append(message)
        if missing_entity_ids:
            message = (
                "来源画面中的任务相关对象没有对应的网站实体："
                + "、".join(missing_entity_ids)
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
            "thumbnail_path", "channel_title", "published_at",
            "duration_seconds", "view_count",
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
            not entity.get("location") or not entity.get("photo_path")
            for entity in entities
        ):
            message = (
                "地图网站还没有完整的坐标、地点照片和地图路线视图。"
                "这道题只能作为候选，不能直接通过正式发布审核。"
            )
            review_warnings.append(message)
            release_blockers.append(message)
        artifacts = self._artifact_hashes(task, source_record, source_video)
        checks = _check_definitions(self.spec, task["track"])
        experiment = _experiment_view(task)
        revision = canonical_sha256({
            "policy": self.spec["reviewer_policy"],
            "checks": checks,
            "artifacts": artifacts,
            "experiment": experiment,
        })
        return {
            "task": task,
            "focus": _task_focus(task),
            "design_profile": _design_profile(task),
            "scenario_evidence": scenario_evidence,
            "motivation": motivation,
            "pending_source_items": pending_source_items,
            "checks": checks,
            "experiment": experiment,
            "entities": entities,
            "review_warnings": review_warnings,
            "release_blockers": release_blockers,
            "site_entity_count": len(site_data["entities"]),
            "source_record": source_record,
            "source_dataset": source_dataset,
            "source_video": source_video,
            "reviewed_artifacts": artifacts,
            "revision": revision,
            "own_status": self._own_status(task, revision=revision),
        }

    def _artifact_hashes(
        self,
        task: Mapping[str, Any],
        source_record: Mapping[str, Any],
        source_video: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        site_data = load_site_data(self.root, task["site"])
        entity_by_id = {
            entity["entity_id"]: entity for entity in site_data["entities"]
        }
        entity_visuals = [
            visual_path
            for entity_id in _referenced_entities(task, source_record)
            if entity_id in entity_by_id
            if (visual_path := entity_visual_path(
                task["site"], entity_by_id[entity_id]
            )) is not None
            if resolve_within(
                self.root,
                visual_path,
                context=f"{task['id']} review website image",
                must_exist=False,
            ).is_file()
        ]
        references = [
            self.spec_path.relative_to(self.root).as_posix(),
            f"data/tasks/{task['track']}/{task['id']}.json",
            task["source"]["dossier"],
            task["source"]["frame_provenance"],
            SITE_DATA_FILES[task["site"]],
            *([str(source_video["path"])] if source_video else []),
            *(
                [task["source"]["task_spec"]]
                if "task_spec" in task["source"] else []
            ),
            *[
                frame
                for packet in task["visual_packets"]
                for frame in packet["frame_paths"]
            ],
            *entity_visuals,
        ]
        return {
            reference: _file_sha256(resolve_within(
                self.root, reference, context=f"{task['id']} review artifact"
            ))
            for reference in dict.fromkeys(references)
        }

    def _source_material(
        self, task: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        dossier_path = resolve_within(
            self.root, task["source"]["dossier"], context="source dossier"
        )
        dossier = _load_object(dossier_path, f"{task['id']} source dossier")
        records = [
            record for record in dossier.get("tasks", [])
            if isinstance(record, dict) and record.get("task_id") == task["id"]
        ]
        if len(records) != 1:
            raise ReviewWorkspaceError(f"{task['id']}: expected one source record")
        dataset = dossier.get("dataset", {})
        if not isinstance(dataset, dict):
            dataset = {}
        source = records[0].get("source", {})
        source_video: dict[str, Any] | None = None
        source_root = dataset.get("source_root")
        media_id = source.get("media_id") if isinstance(source, dict) else None
        if isinstance(source_root, str) and isinstance(media_id, str):
            base = resolve_within(
                self.root, source_root, context="source video directory", must_exist=False
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

    def _submission_path(self, task_id: str) -> Path:
        return self.root / SUBMISSIONS / task_id / f"{self.reviewer_id}.json"

    def _own_status(
        self, task: Mapping[str, Any], *, revision: str | None = None,
    ) -> str:
        path = self._submission_path(str(task["id"]))
        if not path.is_file():
            return "todo"
        try:
            payload = _load_object(path, str(path))
        except ReviewWorkspaceError:
            return "invalid"
        current = revision or self._case(dict(task))["revision"]
        return "submitted" if payload.get("revision") == current else "stale"

    def _validated_response(
        self, case: Mapping[str, Any], response: Mapping[str, Any],
    ) -> dict[str, Any]:
        task = case["task"]
        decision = response.get("decision")
        note = str(response.get("note", "")).strip()
        if decision not in {"admit", "reject"}:
            raise ReviewWorkspaceError("请选择保留或指出问题")
        if decision == "admit" and case["release_blockers"]:
            raise ReviewWorkspaceError(
                "这道题仍有未解决的验收问题，目前不能保留："
                + str(case["release_blockers"][0])
            )
        if decision == "reject" and not note:
            raise ReviewWorkspaceError("发现问题时请简要说明")
        granularity = task["goal"]["target_granularity"]
        if granularity not in GRANULARITIES:
            raise ReviewWorkspaceError("任务中的目标粒度无效")
        unresolved = [note] if decision == "reject" else []
        return {
            "schema_version": 5,
            "reviewer": self.reviewer_id,
            "task_id": task["id"],
            "decision": decision,
            "note": note,
            "target_granularity": granularity,
            "unresolved": unresolved,
            "reviewed_artifacts": case["reviewed_artifacts"],
            "revision": case["revision"],
            "submitted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
