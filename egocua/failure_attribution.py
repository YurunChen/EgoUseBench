"""PULSE failure attribution for valid Ego CUA model runs.

The Checker owns observable behavior facts.  This module is the only place that
combines run qualification with evidence-grounded trajectory reviews to publish
one evidence-linked failure stage.
"""

from __future__ import annotations

import fcntl
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .checker import EVALUATOR_VERSION, behavioral_symptoms
from .environment import public_site_data
from .run_ledger import load_run_record
from .schema import acceptable_states_of
from .visual_sampling import packet_asset_paths
from .workspace import resolve_within, write_json_atomic

PULSE_POLICY_VERSION = "pulse-v6"
FINAL_STATE_CHECKER_VERSIONS = frozenset({"3.0", "3.1", "3.2", EVALUATOR_VERSION})
FAILURE_SOURCE_PATH = Path("artifacts/reports/task-qualification.json")
FAILURE_EVIDENCE_PATH = Path("data/reviews/failure-attribution.json")
FAILURE_REPORT_PATH = Path("artifacts/reports/failure-attribution-formal.json")
PULSE_STAGES = ("P", "U", "L", "S", "E")
REVIEW_PACKET_FIELDS = (
    "run_id",
    "task_id",
    "model",
    "track",
    "site",
    "run_artifact",
    "scoring_goal",
    "checker_version",
)
PULSE_LABELS = {
    "P": {
        "name": "Perception Failure", "zh": "感知失败",
        "criterion": "A query or action explicitly adopts a visual property or relation contradicted by already delivered frames. A broad query or a wrong webpage selection alone is insufficient.",
        "guidance": "行为明确采用了与已送达画面矛盾的信息。宽泛搜索或选错网页对象本身不足以证明感知错误。",
    },
    "U": {
        "name": "Understanding and Update Failure", "zh": "意图与更新失败",
        "criterion": "The agent commits to a referent left explicitly ambiguous by the actual request despite available clarification, or its behavior explicitly retains an obsolete requirement after an update. Not asking alone is insufficient; establish the unresolved intent or obsolete requirement and its effect.",
        "guidance": "任务确需消歧却未经澄清作出选择，或收到更新后明确沿用旧要求。单独没有提问不构成此类。",
    },
    "L": {
        "name": "Linking Failure", "zh": "关联失败",
        "criterion": "The agent selects a webpage entity incompatible with the active requirements, or explicitly rejects a compatible candidate as irrelevant. Establish this from public facts and observable behavior, not merely an empty final selection or absence from the answer list. Do not assume perception or intent were correct; prefer an earlier directly supported P or U error when one exists.",
        "guidance": "选中不符合当前要求的网页对象，或明确把符合要求的候选误判为无关。仅最终未选择或不在答案清单中不足以判断。更早已有明确感知或意图错误时，以更早错误为主。",
    },
    "S": {
        "name": "Action and State-Control Failure", "zh": "动作与状态控制失败",
        "criterion": "Observable action execution or state editing fails: the agent issues invalid or non-executable actions under its recorded protocol, fails to recover from explicit action errors, or evidenced correct target choices are not realized by add/remove/quantity/retain operations. Wrong entity choice belongs to L unless an earlier P or U is directly supported. Distinguish the performing agent's invalid output from a Judge/API processing failure.",
        "guidance": "动作格式不合法、执行报错后未恢复，或正确目标未通过增删、数量、保留操作形成正确状态。目标对象选错先考虑关联失败。",
    },
    "E": {
        "name": "End-state Failure", "zh": "结果核验失败",
        "criterion": "An independently observable completion or verification error causes the failure, such as continuing after a correct state and destroying it without recovery. Wrong final state, budget exhaustion, or saying done alone is insufficient. An earlier uncorrected state-edit error remains S rather than being relabeled E at termination.",
        "guidance": "有直接证据表明完成判断造成失败，例如状态已正确，继续操作将其破坏。终态错误、步数耗尽或声称完成本身不足；更早的状态编辑错误仍归S。",
    },
}

RUN_VERDICTS = (
    "success",
    "invalid_run",
    "benchmark_fault",
    "valid_model_failure",
    "unknown",
)
ATTRIBUTION_METHODS = ("trajectory_review", "human_adjudication")
CANDIDATE_SOURCES = ("coding_agent", "llm_judge", "human")
CONFIDENCE_VALUES = ("high", "low", "adjudicated")
UNCERTAINTY_REASONS = ("stage_boundary", "evidence_gap")
ATTRIBUTION_STATUSES = (
    "excluded",
    "coding_agent_pending",
    "rule_evaluation_pending",
    "evidence_insufficient",
    "resolved",
)
ATTRIBUTION_FIELDS = frozenset(
    {
        "schema_version",
        "policy_version",
        "run_id",
        "task_id",
        "model",
        "track",
        "site",
        "checker_version",
        "scoring_goal",
        "run_verdict",
        "pulse_stage",
        "mechanism",
        "first_decisive_step",
        "candidate_stage",
        "candidate_mechanism",
        "candidate_step",
        "candidate_source",
        "candidate_confidence",
        "candidate_unknown_boundary",
        "observable_facts",
        "secondary_symptoms",
        "attribution_method",
        "confidence",
        "attribution_status",
        "review_required",
        "rule_evidence",
        "qualification_evidence",
        "task_sha256",
        "site_data_sha256",
    }
)
ATTRIBUTION_OPTIONAL_FIELDS = frozenset({"run_artifact", "setting"})
UNKNOWN_BOUNDARIES = tuple(
    f"{left}/{right}"
    for index, left in enumerate(PULSE_STAGES)
    for right in PULSE_STAGES[index + 1 :]
)
MECHANISMS = {
    "P": ("object_grounding", "action_state_grounding", "spatiotemporal_grounding"),
    "U": ("ambiguity_resolution", "update_integration", "constraint_retention"),
    "L": ("retrieval", "candidate_discrimination", "detail_verification"),
    "S": ("action_grounding", "state_repair", "feedback_recovery"),
    "E": ("premature_completion", "completion_judgment", "post_success_overrun"),
}
BENCHMARK_FAULT_POLICY_VERSION = "benchmark-fault-certificate-v1"
BENCHMARK_FAULT_PREDICATES = (
    "checker_replay_mismatch",
    "task_invariant_failure",
    "site_invariant_failure",
    "event_delivery_failure",
    "artifact_integrity_failure",
)


def _benchmark_fault_certificate(
    qualification: str | Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(qualification, Mapping):
        return None
    if qualification.get("verdict") != "benchmark_fault":
        return None
    if qualification.get("policy_version") != BENCHMARK_FAULT_POLICY_VERSION:
        raise ValueError("benchmark fault requires the registered policy_version")
    if qualification.get("predicate") not in BENCHMARK_FAULT_PREDICATES:
        raise ValueError("benchmark fault requires a registered predicate")
    facts = qualification.get("facts")
    if not isinstance(facts, Mapping):
        raise ValueError("benchmark fault certificate requires predicate facts")
    predicate = str(qualification["predicate"])
    if predicate == "checker_replay_mismatch":
        valid_facts = (
            isinstance(facts.get("input_sha256"), str)
            and bool(facts.get("input_sha256"))
            and facts.get("recorded_verdict") in {"pass", "fail", "unknown"}
            and facts.get("replayed_verdict") in {"pass", "fail", "unknown"}
            and facts.get("recorded_verdict") != facts.get("replayed_verdict")
        )
    elif predicate in {"task_invariant_failure", "site_invariant_failure"}:
        valid_facts = (
            isinstance(facts.get("check_name"), str)
            and bool(facts.get("check_name"))
            and facts.get("check_status") == "fail"
            and "expected" in facts
            and "observed" in facts
            and facts.get("expected") != facts.get("observed")
        )
    elif predicate == "event_delivery_failure":
        valid_facts = (
            facts.get("event_required") is True
            and isinstance(facts.get("event_id"), str)
            and bool(facts.get("event_id"))
            and facts.get("delivery_status") in {"missing", "corrupt"}
        )
    else:
        valid_facts = (
            isinstance(facts.get("artifact_path"), str)
            and bool(facts.get("artifact_path"))
            and facts.get("integrity_check")
            in {
                "required_file",
                "json_parse",
                "sha256",
            }
            and "expected" in facts
            and "observed" in facts
            and facts.get("expected") != facts.get("observed")
        )
    if not valid_facts:
        raise ValueError("benchmark fault review lacks the required predicate facts")
    evidence_refs = qualification.get("evidence_refs")
    if (
        not isinstance(evidence_refs, list)
        or not evidence_refs
        or any(not isinstance(ref, str) or not ref.strip() for ref in evidence_refs)
    ):
        raise ValueError("benchmark fault certificate requires evidence_refs")
    return qualification


def _run_verdict(
    checked: Mapping[str, Any],
    qualification: str | Mapping[str, Any] | None,
) -> str:
    if _benchmark_fault_certificate(qualification) is not None:
        return "benchmark_fault"
    if isinstance(qualification, Mapping):
        declared = qualification.get("verdict")
        if declared in {"task_or_site_defect", "benchmark_fault"}:
            return "unknown"
        if declared == "invalid_run":
            return "invalid_run"
        if declared not in {None, "valid_model_failure"}:
            raise ValueError(f"invalid qualification verdict: {declared!r}")
    if isinstance(qualification, str) and qualification in {
        "task_or_site_defect",
        "benchmark_fault",
    }:
        return "unknown"
    if qualification == "invalid_run" or checked.get("run_valid") is not True:
        return "invalid_run"
    success = checked.get("e2e_success")
    if success is True:
        return "success"
    if success is False:
        return "valid_model_failure"
    return "unknown"


def _observable_facts(checked: Mapping[str, Any]) -> list[str]:
    facts = ["run_valid", "task_unsuccessful"]
    boolean_facts = {
        "target_opened": "target_opened",
        "target_selected": "target_selected",
        "release_delivered": "update_delivered",
        "asked": "clarification_requested",
        "valid_question": "valid_clarification",
        "answer_delivered": "clarification_answer_delivered",
        "answer_used": "clarification_answer_used",
        "goal_state_correct": "goal_state_correct",
        "submission_completed": "submission_completed",
        "premature_submission": "premature_submission",
        "stale_final": "stale_final_state",
        "missing_after_update": "missing_required_state_after_update",
        "false_update": "unsupported_update",
    }
    facts.extend(
        label for field, label in boolean_facts.items() if checked.get(field) is True
    )
    return facts


def _base_record(
    checked: Mapping[str, Any], verdict: str, record: Mapping[str, Any]
) -> dict[str, Any]:
    is_failure = verdict == "valid_model_failure"
    meta = record.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    evaluator = checked.get("evaluator")
    evaluator = evaluator if isinstance(evaluator, Mapping) else {}
    return {
        "schema_version": 1,
        "policy_version": PULSE_POLICY_VERSION,
        "run_id": str(checked.get("run_id") or meta.get("run_id") or ""),
        "task_id": str(checked.get("task_id") or record.get("task_id") or ""),
        "model": str(checked.get("model") or meta.get("model") or "unknown"),
        "track": str(checked.get("track") or record.get("track") or "unknown"),
        "site": str(checked.get("site") or meta.get("site") or "unknown"),
        "setting": str((record.get("condition") or {}).get("name", "unknown")),
        "checker_version": evaluator.get("version"),
        "scoring_goal": deepcopy(evaluator["goal"]) if "goal" in evaluator else {
            "acceptable_states": deepcopy(evaluator.get("acceptable_states"))
        },
        "run_verdict": verdict,
        "pulse_stage": None,
        "mechanism": None,
        "first_decisive_step": None,
        "candidate_stage": None,
        "candidate_mechanism": None,
        "candidate_step": None,
        "candidate_source": None,
        "candidate_confidence": None,
        "candidate_unknown_boundary": None,
        "observable_facts": _observable_facts(checked) if is_failure else [],
        "secondary_symptoms": behavioral_symptoms(checked) if is_failure else [],
        "attribution_method": None,
        "confidence": "low" if is_failure else None,
        "attribution_status": "coding_agent_pending" if is_failure else "excluded",
        "review_required": False,
        "rule_evidence": None,
        "qualification_evidence": None,
        "task_sha256": meta.get("task_sha256"),
        "site_data_sha256": meta.get("site_data_sha256"),
    }


def _require_final_state_checker(version: str | None) -> None:
    if version not in FINAL_STATE_CHECKER_VERSIONS:
        supported = ", ".join(sorted(FINAL_STATE_CHECKER_VERSIONS))
        raise ValueError(
            f"v6 attribution requires final-state Checker versions {supported}"
        )


def attribute_run(
    checked: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    qualification: str | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one PULSE record from Checker facts and an optional validity verdict."""

    _require_final_state_checker(checked.get("evaluator", {}).get("version"))
    result = _base_record(checked, _run_verdict(checked, qualification), record)
    certificate = _benchmark_fault_certificate(qualification)
    if certificate is not None:
        result["qualification_evidence"] = deepcopy(dict(certificate))
    elif isinstance(qualification, Mapping) and result["run_verdict"] == "unknown":
        result["observable_facts"] = ["task_validity_pending"]
        if qualification.get("reason"):
            result["observable_facts"].append(qualification["reason"])
    validate_attribution(result)
    return result


def apply_assessments(
    rows: Sequence[Mapping[str, Any]],
    assessments: Sequence[Mapping[str, Any]],
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Apply independent trajectory judgments without trusting hidden reasoning."""

    by_run: dict[str, Mapping[str, Any]] = {}
    for assessment_source in assessments:
        if not isinstance(assessment_source, Mapping):
            raise ValueError("Trajectory assessment must be an object")
        if assessment_source.get("reviewer_kind") not in {"coding_agent", "llm_judge"}:
            raise ValueError(
                "assessment requires reviewer_kind coding_agent or llm_judge"
            )
        run_id = str(assessment_source.get("run_id") or "")
        if not run_id:
            raise ValueError("Trajectory assessment requires run_id")
        if run_id in by_run:
            raise ValueError(f"duplicate Trajectory assessment: {run_id}")
        by_run[run_id] = assessment_source

    known_runs = {str(row.get("run_id") or "") for row in rows}
    unknown_runs = sorted(set(by_run) - known_runs)
    if unknown_runs:
        raise ValueError(
            f"Trajectory assessment references unknown runs: {unknown_runs}"
        )

    resolved: list[dict[str, Any]] = []
    for source in rows:
        row = deepcopy(dict(source))
        validate_attribution(row)
        matched_assessment = by_run.get(str(row.get("run_id") or ""))
        if matched_assessment is None:
            resolved.append(row)
            continue
        if row["run_verdict"] != "valid_model_failure":
            resolved.append(row)
            continue
        if matched_assessment.get("policy_version") != PULSE_POLICY_VERSION:
            raise ValueError(
                "Trajectory assessment requires a matching v6 protocol and scoring evidence"
            )
        if matched_assessment.get("checker_version") != row["checker_version"]:
            raise ValueError("Trajectory assessment checker_version mismatch")

        if matched_assessment["reviewer_kind"] == "llm_judge":
            resolved.append(_apply_judge_assessment(row, matched_assessment, root))
            continue

        stage = matched_assessment.get("candidate_stage")
        mechanism = matched_assessment.get("candidate_mechanism")
        confidence = matched_assessment.get("candidate_confidence")
        boundary = matched_assessment.get("candidate_unknown_boundary")
        step = matched_assessment.get("candidate_step")
        facts = matched_assessment.get("observable_facts")
        if stage is not None and stage not in PULSE_STAGES:
            raise ValueError(f"invalid trajectory candidate_stage: {stage!r}")
        if (stage is None and mechanism is not None) or (
            stage is not None and mechanism not in MECHANISMS[stage]
        ):
            raise ValueError(f"invalid {stage} mechanism: {mechanism!r}")
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(f"invalid trajectory confidence: {confidence!r}")
        if boundary is not None and boundary not in UNKNOWN_BOUNDARIES:
            raise ValueError(f"invalid candidate_unknown_boundary: {boundary!r}")
        if step is not None and (type(step) is not int or step < 0):
            raise ValueError("Trajectory candidate_step must be non-negative")
        if confidence == "high" and (stage is None or step is None):
            raise ValueError(
                "high-confidence review requires a stage and decisive step"
            )
        if (
            not isinstance(facts, list)
            or not facts
            or any(not isinstance(fact, str) or not fact.strip() for fact in facts)
        ):
            raise ValueError("Trajectory assessment requires observable facts")

        row.update(
            {
                "candidate_stage": stage,
                "candidate_mechanism": mechanism,
                "candidate_step": step,
                "candidate_source": matched_assessment["reviewer_kind"]
                if stage
                else None,
                "candidate_confidence": confidence,
                "candidate_unknown_boundary": boundary,
                "pulse_stage": None,
                "mechanism": None,
                "first_decisive_step": None,
                "observable_facts": list(
                    dict.fromkeys(
                        [
                            *row["observable_facts"],
                            *(fact.strip() for fact in facts),
                        ]
                    )
                ),
                "attribution_method": None,
                "confidence": "low",
                "attribution_status": "coding_agent_pending",
                "review_required": False,
                "rule_evidence": None,
            }
        )
        review = matched_assessment.get("trajectory_review")
        if review is not None:
            certificate = verify_trajectory_review(row, review, root)
            row.update(
                {
                    "attribution_method": "trajectory_review",
                    "rule_evidence": certificate,
                    "attribution_status": "evidence_insufficient",
                    "review_required": review.get("uncertainty_reason")
                    != "evidence_gap",
                }
            )
            if stage is not None and confidence == "high" and boundary is None:
                row.update(
                    {
                        "pulse_stage": stage,
                        "mechanism": mechanism,
                        "first_decisive_step": step,
                        "attribution_method": "trajectory_review",
                        "confidence": "high",
                        "attribution_status": "resolved",
                        "review_required": False,
                        "rule_evidence": certificate,
                    }
                )
        validate_attribution(row)
        resolved.append(row)
    return resolved


def resolve_evidence_reference(
    context: Mapping[str, Any], ref: Mapping[str, Any]
) -> tuple[Any, int | None]:
    kind = ref.get("kind")
    index = ref.get("index")
    if kind in {"action", "web", "frame"} or (kind == "state" and index is not None):
        if type(index) is not int or index < 0:
            raise ValueError("Evidence index must be a nonnegative integer")
    if kind == "action":
        value = context["actions"][index]
        return value, value.get("step")
    if kind == "web":
        value = context["web_observations"][index]
        return value, (value.get("metadata") or {}).get("step_number")
    if kind == "frame":
        packet_index = ref.get("packet_index")
        if type(packet_index) is not int or packet_index < 0:
            raise ValueError("Frame reference requires packet_index")
        packet = context["task_context"]["visual_packets"][packet_index]
        return packet["frame_paths"][index], packet["released_at_step"]
    if kind == "entity":
        entity = next(
            (
                item
                for item in context["site_context"]["entities"]
                if str(item["entity_id"]) == ref["id"]
            ),
            None,
        )
        if entity is None:
            raise KeyError(ref["id"])
        return entity, None
    if kind == "goal":
        if ref.get("field") != "acceptable_final_states":
            raise ValueError("Unknown scoring goal field")
        return context["expected_states"], None
    if kind == "state":
        field = ref.get("field")
        if index is None:
            if (
                field not in {"initial_state", "actual_final_state"}
                or context[field] is None
            ):
                raise ValueError("Historical state is unavailable")
            return context[field], None
        if field not in {"state_before", "state_after"}:
            raise ValueError("Action state requires state_before or state_after")
        action = context["actions"][index]
        return action[field], action.get("step")
    raise ValueError(f"Unknown evidence kind: {kind!r}")


def _validated_reference(
    context: Mapping[str, Any], reference: Mapping[str, Any]
) -> tuple[Any, int | None]:
    if not isinstance(reference, Mapping):
        raise ValueError("Evidence reference must be an object")
    try:
        value, observed_step = resolve_evidence_reference(context, reference)
    except (IndexError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Evidence reference is not in this historical run: {reference}"
        ) from exc
    if "observed_step" in reference and reference["observed_step"] != observed_step:
        raise ValueError("Evidence observation/delivery step mismatch")
    if "value" in reference and reference["value"] != value:
        raise ValueError("Evidence reference does not match the recorded fact")
    return value, observed_step


def validate_evidence_references(context: Mapping[str, Any], refs: Any) -> None:
    """Validate evidence identity and timing, not the Judge's interpretation."""
    if not isinstance(refs, list) or not refs:
        raise ValueError("Evidence references must be a nonempty list")
    for reference in refs:
        _validated_reference(context, reference)


def _judgment_evidence_steps(
    context: Mapping[str, Any], refs: Any
) -> dict[str, list[int | None]]:
    if not isinstance(refs, list) or not refs:
        raise ValueError("Evidence references must be a nonempty list")
    pointer_fields = {
        "action": {"kind", "index", "observed_step"},
        "web": {"kind", "index", "observed_step"},
        "frame": {"kind", "packet_index", "index", "observed_step"},
        "entity": {"kind", "id"},
        "goal": {"kind", "field"},
        "state": {"kind", "field", "index", "observed_step"},
    }
    steps: dict[str, list[int | None]] = defaultdict(list)
    for reference in refs:
        _, observed_step = _validated_reference(context, reference)
        if set(reference) - pointer_fields[reference["kind"]]:
            raise ValueError(
                "judgment evidence uses pointer-only references without copied content"
            )
        steps[reference["kind"]].append(observed_step)
    return steps


def validate_judgment(context: Mapping[str, Any], judgment: Any) -> None:
    """Validate the six-field diagnosis and its archived evidence, not its semantics."""
    fields = {"stage", "step", "confidence", "reason", "evidence", "uncertainty"}
    if not isinstance(judgment, Mapping) or set(judgment) != fields:
        raise ValueError(
            "judgment requires exactly stage, step, confidence, reason, evidence, uncertainty"
        )
    stage, step = judgment["stage"], judgment["step"]
    confidence, uncertainty = judgment["confidence"], judgment["uncertainty"]
    if stage is not None and (not isinstance(stage, str) or stage not in PULSE_STAGES):
        raise ValueError("invalid judgment stage")
    if step is not None and (type(step) is not int or step < 0):
        raise ValueError("judgment step must be a nonnegative integer or null")
    if confidence not in CONFIDENCE_VALUES or uncertainty not in (
        None,
        "stage_boundary",
        "evidence_gap",
        "task_validity",
    ):
        raise ValueError("invalid judgment confidence or uncertainty")
    if (stage is None and (confidence != "low" or uncertainty is None)) or (
        stage is not None and uncertainty is not None
    ):
        raise ValueError("uncertain judgment requires null stage and low confidence")
    if confidence == "high" and step is None:
        raise ValueError("high-confidence judgment requires a decisive step")
    if not isinstance(judgment["reason"], str) or not judgment["reason"].strip():
        raise ValueError("judgment requires a nonempty reason")
    evidence_steps = _judgment_evidence_steps(context, judgment["evidence"])
    if step is not None and step not in [
        *evidence_steps["action"],
        *evidence_steps["web"],
    ]:
        raise ValueError(
            "decisive step must cite its corresponding action or web observation"
        )
    if stage == "P" and (
        step is None
        or not evidence_steps["frame"]
        or any(frame_step > step for frame_step in evidence_steps["frame"])
    ):
        raise ValueError(
            "perception judgment requires frames delivered before its decisive step"
        )


def verify_judge_assessment(
    row: Mapping[str, Any], assessment: Mapping[str, Any], root: Path | None
) -> Mapping[str, Any]:
    if root is None:
        raise ValueError("LLM Judge assessment requires historical workspace")
    for field in ("judge_model", "prompt_version", "usage"):
        if field not in assessment:
            raise ValueError(f"LLM Judge assessment requires {field}")
    allowed = {
        "run_id",
        "policy_version",
        "checker_version",
        "reviewer_kind",
        "judge_model",
        "prompt_version",
        "usage",
        "judgment",
        "review_history",
        "dispute_review",
    }
    if set(assessment) - allowed:
        raise ValueError("LLM Judge assessment stores only metadata and judgment")
    judgment = assessment.get("judgment")
    context = historical_review_context(row, root)
    identity = context["run_identity"]
    if identity["run_valid"] is not True:
        raise ValueError("LLM Judge assessment does not match a valid archived run")
    validate_judgment(context, judgment)
    if "dispute_review" in assessment:
        _validate_dispute_review(context, assessment["dispute_review"])
    return judgment


def _validate_dispute_review(context: Mapping[str, Any], review: Any) -> None:
    if (
        not isinstance(review, Mapping)
        or set(review) != {"verdict", "reviewer_kind", "judgment"}
        or review["reviewer_kind"] != "coding_agent"
        or review["verdict"]
        not in {"old_supported", "judge_supported", "both_revised", "unknown"}
    ):
        raise ValueError("invalid Coding Agent dispute review")
    validate_judgment(context, review["judgment"])


def _apply_judge_assessment(
    source: Mapping[str, Any], assessment: Mapping[str, Any], root: Path | None
) -> dict[str, Any]:
    judgment = verify_judge_assessment(source, assessment, root)
    if judgment["uncertainty"] == "task_validity":
        return qualify_attribution_row(
            source,
            {
                "verdict": "task_or_site_defect",
                "reason": judgment["reason"],
                "evidence_refs": judgment["evidence"],
            },
            root,
        )
    row = deepcopy(dict(source))
    stage, confidence = judgment["stage"], judgment["confidence"]
    resolved = stage is not None and confidence == "high"
    row.update(
        candidate_stage=stage,
        candidate_mechanism=None,
        candidate_step=judgment["step"],
        candidate_source="llm_judge" if stage is not None else None,
        candidate_confidence=confidence,
        candidate_unknown_boundary=None,
        pulse_stage=stage if resolved else None,
        mechanism=None,
        first_decisive_step=judgment["step"] if resolved else None,
        attribution_method="trajectory_review",
        confidence="high" if resolved else "low",
        attribution_status="resolved" if resolved else "evidence_insufficient",
        review_required=not resolved,
        rule_evidence={
            "policy_version": "trajectory-judge-v2",
            "reason": judgment["reason"],
            "evidence_refs": deepcopy(judgment["evidence"]),
            "uncertainty_reason": judgment["uncertainty"],
        },
    )
    validate_attribution(row)
    return row


def _record_task_validity_question(
    evidence: dict[str, Any], assessment: Mapping[str, Any]
) -> None:
    if assessment["reviewer_kind"] != "llm_judge":
        return
    judgment = assessment["judgment"]
    if judgment["uncertainty"] != "task_validity":
        return
    qualifications = evidence.setdefault("run_qualifications", {})
    previous = qualifications.get(assessment["run_id"])
    if previous is None or previous.get("verdict") in {
        "valid_model_failure",
        "task_or_site_defect",
    }:
        qualifications[assessment["run_id"]] = {
            "verdict": "task_or_site_defect",
            "reason": judgment["reason"],
            "evidence_refs": deepcopy(judgment["evidence"]),
            "review_method": "llm_judge",
            **({"review_history": [previous]} if previous else {}),
        }


def save_assessment(
    root: str | Path,
    row: Mapping[str, Any],
    assessment: Mapping[str, Any],
    *,
    affected: bool = False,
) -> dict[str, Any]:
    """Save a cited assessment once; replace only explicitly affected evidence."""
    workspace = Path(root).resolve()
    validate_attribution(row)
    if assessment.get("run_id") != row["run_id"]:
        raise ValueError("assessment run_id mismatch")
    path = workspace / FAILURE_EVIDENCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {
                "schema_version": 1,
                "assessments": [],
                "ratings": [],
                "run_qualifications": {},
            }
        )
        existing = next(
            (
                item
                for item in evidence["assessments"]
                if item["run_id"] == row["run_id"]
            ),
            None,
        )
        if existing is not None and not affected:
            return deepcopy(existing)
        current_row = qualify_attribution_row(
            row, evidence.get("run_qualifications", {}).get(row["run_id"]), workspace
        )
        if current_row["run_verdict"] != "valid_model_failure":
            raise ValueError("new assessment requires a currently valid model failure")
        saved = deepcopy(dict(assessment))
        apply_assessments([current_row], [saved], workspace)
        if (
            saved["reviewer_kind"] == "coding_agent"
            and saved.get("trajectory_review") is None
        ):
            raise ValueError("saved Coding Agent assessment requires trajectory_review")
        _record_task_validity_question(evidence, saved)
        if existing is not None:
            history = deepcopy(existing.get("review_history", []))
            history.append(
                {
                    key: deepcopy(value)
                    for key, value in existing.items()
                    if key != "review_history"
                }
            )
            saved["review_history"] = history
            evidence["assessments"] = [
                saved if item["run_id"] == row["run_id"] else item
                for item in evidence["assessments"]
            ]
        else:
            evidence["assessments"].append(saved)
        write_json_atomic(path, evidence)
        return deepcopy(saved)


def verify_trajectory_review(
    row: Mapping[str, Any],
    review: Mapping[str, Any],
    root: Path | None,
) -> dict[str, Any]:
    """Check citation identity, not the semantic accuracy of the interpretation."""
    if root is None:
        raise ValueError("trajectory review requires workspace root")
    if not isinstance(review, Mapping):
        raise ValueError("trajectory_review must be an object")
    reference = row.get("run_artifact")
    if not isinstance(reference, str) or not reference:
        raise ValueError("trajectory review requires the source run_artifact")
    record = load_run_record(
        resolve_within(
            root,
            reference,
            context="PULSE trace",
            must_exist=False,
        )
    )
    if record.get("meta", {}).get("run_id") != row["run_id"]:
        raise ValueError("trajectory review run_id mismatch")
    for field in ("task_id", "track"):
        if record.get(field) != row[field]:
            raise ValueError(f"trajectory review {field} mismatch")
    if record.get("meta", {}).get("model") != row["model"]:
        raise ValueError("trajectory review model mismatch")
    if record.get("run_status", {}).get("valid") is not True:
        raise ValueError("trajectory review requires a valid source run")
    for field in ("reason", "earlier_stages", "failure_effect"):
        if not isinstance(review.get(field), str) or not review[field].strip():
            raise ValueError(f"trajectory review requires {field}")
    uncertainty = review.get("uncertainty_reason")
    if uncertainty is not None:
        if uncertainty not in UNCERTAINTY_REASONS:
            raise ValueError(f"invalid uncertainty_reason: {uncertainty!r}")
        if row["candidate_stage"] is not None:
            raise ValueError("resolved review cannot declare uncertainty_reason")
    citations = review.get("citations")
    if not isinstance(citations, list):
        raise ValueError("trajectory review requires citations")
    steps = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("trajectory citation must be an object")
        index = citation.get("action_index")
        actions = record.get("actions", [])
        if type(index) is not int or not 0 <= index < len(actions):
            raise ValueError("trajectory citation action_index is invalid")
        action = actions[index]
        if citation.get("action") != action:
            raise ValueError("trajectory citation does not match source action")
        steps.append(action.get("step"))
    trace_citations = review.get("trace_citations", [])
    steps.extend(_trace_citation_steps(record, trace_citations))
    if row["candidate_step"] is not None and row["candidate_step"] not in steps:
        raise ValueError("decisive step must have a source citation")
    return {
        **{
            field: deepcopy(review[field])
            for field in (
                "reason",
                "earlier_stages",
                "failure_effect",
                "citations",
            )
        },
        "policy_version": "trajectory-review-v1",
        "run_artifact": reference,
        **({"uncertainty_reason": uncertainty} if uncertainty is not None else {}),
        **({"trace_citations": deepcopy(trace_citations)} if trace_citations else {}),
    }


def _trace_citation_steps(record: Mapping[str, Any], citations: Any) -> list[int]:
    """Verify browser actions absent from the native website action ledger."""
    if not isinstance(citations, list):
        raise ValueError("trace_citations must be a list")
    trace = record.get("trace", {})
    history = trace.get("history", []) if isinstance(trace, Mapping) else trace
    if not isinstance(history, list):
        history = []
    steps = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError("trace citation must be an object")
        index = citation.get("history_index")
        if type(index) is not int or not 0 <= index < len(history):
            raise ValueError("trace citation history_index is invalid")
        entry = history[index]
        step = entry.get("metadata", {}).get("step_number")
        actions = (entry.get("model_output") or {}).get("action")
        expected = {
            "history_index": index,
            "step": step,
            "actions": actions,
            "result": entry.get("result"),
        }
        if type(step) is not int or step < 0 or not actions or citation != expected:
            raise ValueError("trace citation does not match source browser action")
        steps.append(step)
    return steps


def _validate_rule_evidence(row: Mapping[str, Any]) -> None:
    evidence = row.get("rule_evidence")
    if evidence is None:
        return
    if not isinstance(evidence, Mapping):
        raise ValueError("rule_evidence must be an object")
    policy = evidence.get("policy_version")
    if policy == "human-adjudication-v1":
        if (row.get("attribution_method") != "human_adjudication"
                or row.get("candidate_source") != "human"
                or row.get("confidence") != "adjudicated"
                or row.get("candidate_confidence") != "adjudicated"
                or row.get("pulse_stage") != row.get("candidate_stage")
                or not evidence.get("reviewer_id")
                or not evidence.get("reason")):
            raise ValueError("invalid human adjudication evidence")
        return
    if policy == "trajectory-judge-v2":
        if (
            row.get("attribution_method") != "trajectory_review"
            or not evidence.get("reason")
            or not isinstance(evidence.get("evidence_refs"), list)
            or not evidence["evidence_refs"]
        ):
            raise ValueError("invalid minimal Judge evidence")
        if row.get("pulse_stage") is not None and (
            row.get("candidate_source") != "llm_judge"
            or row["pulse_stage"] != row.get("candidate_stage")
            or row.get("first_decisive_step") != row.get("candidate_step")
            or row.get("candidate_confidence") != "high"
        ):
            raise ValueError("invalid resolved Judge evidence")
        return
    if policy == "trajectory-review-v1":
        if (
            row.get("attribution_method") != "trajectory_review"
            or not isinstance(evidence.get("citations"), list)
            or not evidence.get("reason")
            or not evidence.get("earlier_stages")
            or not evidence.get("failure_effect")
        ):
            raise ValueError("invalid trajectory review")
        if row.get("pulse_stage") is not None and (
            row.get("candidate_source") not in CANDIDATE_SOURCES
            or row.get("pulse_stage") != row.get("candidate_stage")
            or row.get("first_decisive_step") != row.get("candidate_step")
            or row.get("candidate_confidence") != "high"
            or row.get("candidate_unknown_boundary") is not None
            or type(row.get("candidate_step")) is not int
            or not (evidence.get("citations") or evidence.get("trace_citations"))
        ):
            raise ValueError("invalid resolved trajectory review")
        return
    raise ValueError("formal attribution requires a trajectory review")


def validate_attribution(row: Mapping[str, Any]) -> None:
    """Reject malformed or logically contradictory PULSE records."""

    missing = ATTRIBUTION_FIELDS - set(row)
    unknown = set(row) - ATTRIBUTION_FIELDS - ATTRIBUTION_OPTIONAL_FIELDS
    if missing or unknown:
        raise ValueError(
            "PULSE record fields disagree with schema; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if row.get("policy_version") != PULSE_POLICY_VERSION:
        raise ValueError(f"PULSE policy_version must be {PULSE_POLICY_VERSION}")
    _require_final_state_checker(row.get("checker_version"))
    scoring_goal = row.get("scoring_goal")
    if not isinstance(scoring_goal, dict):
        raise ValueError("PULSE record requires the Checker scoring_goal")
    acceptable_states_of(scoring_goal, "PULSE scoring_goal")
    verdict = row.get("run_verdict")
    if verdict not in RUN_VERDICTS:
        raise ValueError(f"invalid run_verdict: {verdict!r}")
    stage = row.get("pulse_stage")
    if stage is not None and stage not in PULSE_STAGES:
        raise ValueError(f"invalid pulse_stage: {stage!r}")
    if verdict != "valid_model_failure" and stage is not None:
        raise ValueError("pulse_stage is only valid for a valid_model_failure")
    mechanism = row.get("mechanism")
    if stage is None and mechanism is not None:
        raise ValueError("mechanism requires pulse_stage")
    if (
        stage is not None
        and mechanism not in MECHANISMS[stage]
        and not (mechanism is None and row.get("candidate_source") in {"llm_judge", "human"})
    ):
        raise ValueError(f"invalid {stage} mechanism: {mechanism!r}")
    candidate = row.get("candidate_stage")
    if candidate is not None and candidate not in PULSE_STAGES:
        raise ValueError(f"invalid candidate_stage: {candidate!r}")
    candidate_mechanism = row.get("candidate_mechanism")
    if candidate is None and candidate_mechanism is not None:
        raise ValueError("candidate_mechanism requires candidate_stage")
    if (
        candidate is not None
        and candidate_mechanism not in MECHANISMS[candidate]
        and not (
            candidate_mechanism is None and row.get("candidate_source") in {"llm_judge", "human"}
        )
    ):
        raise ValueError(
            f"invalid {candidate} candidate mechanism: {candidate_mechanism!r}"
        )
    candidate_source = row.get("candidate_source")
    if candidate is None and candidate_source is not None:
        raise ValueError("candidate_source requires candidate_stage")
    if candidate is not None and candidate_source not in CANDIDATE_SOURCES:
        raise ValueError(f"invalid candidate_source: {candidate_source!r}")
    candidate_confidence = row.get("candidate_confidence")
    if (
        candidate_confidence is not None
        and candidate_confidence not in CONFIDENCE_VALUES
    ):
        raise ValueError(f"invalid candidate_confidence: {candidate_confidence!r}")
    candidate_boundary = row.get("candidate_unknown_boundary")
    if candidate_boundary is not None and candidate_boundary not in UNKNOWN_BOUNDARIES:
        raise ValueError(f"invalid candidate_unknown_boundary: {candidate_boundary!r}")
    method = row.get("attribution_method")
    if method is not None and method not in ATTRIBUTION_METHODS:
        raise ValueError(f"invalid attribution_method: {method!r}")
    confidence = row.get("confidence")
    if confidence is not None and confidence not in CONFIDENCE_VALUES:
        raise ValueError(f"invalid confidence: {confidence!r}")
    status = row.get("attribution_status")
    if status not in ATTRIBUTION_STATUSES:
        raise ValueError(f"invalid attribution_status: {status!r}")
    qualification_evidence = row.get("qualification_evidence")
    if verdict == "benchmark_fault":
        if _benchmark_fault_certificate(qualification_evidence) is None:
            raise ValueError("benchmark_fault requires a rule certificate")
    elif qualification_evidence is not None:
        raise ValueError("qualification_evidence is only valid for benchmark_fault")
    _validate_rule_evidence(row)
    if verdict == "valid_model_failure":
        pending_agent = stage is None and status == "coding_agent_pending"
        pending_rule = stage is None and status == "rule_evaluation_pending"
        evidence_gap = (row.get("rule_evidence") or {}).get(
            "uncertainty_reason"
        ) == "evidence_gap"
        if (
            stage is None
            and not pending_agent
            and not pending_rule
            and (
                confidence != "low"
                or (row.get("review_required") is not True and not evidence_gap)
            )
        ):
            raise ValueError(
                "unattributed model failure must be low-confidence and reviewed"
            )
        if pending_agent and row.get("review_required") is not False:
            raise ValueError("Coding Agent pending work must not enter human review")
        if pending_rule and row.get("review_required") is not False:
            raise ValueError("rule evaluation pending work must not enter human review")
        if (
            stage is not None
            and confidence == "high"
            and row.get("review_required") is not False
        ):
            raise ValueError("high-confidence attribution must not enter human review")
        if stage is not None and (
            confidence not in {"high", "adjudicated"}
            or status != "resolved"
            or row.get("review_required") is not False
        ):
            raise ValueError("formal attribution must be high-confidence and resolved")
        if row.get("review_required") is True and status != "evidence_insufficient":
            raise ValueError("human review requires unresolved rule evidence")
        if confidence == "high" and status != "resolved":
            raise ValueError("high-confidence attribution must be resolved")
        if stage is not None:
            rule_evidence = row.get("rule_evidence")
            if not isinstance(rule_evidence, Mapping):
                raise ValueError("formal attribution requires rule_evidence")
            if rule_evidence.get("policy_version") not in {
                "trajectory-review-v1",
                "trajectory-judge-v2",
                "human-adjudication-v1",
            }:
                raise ValueError("formal attribution has unknown rule_evidence policy")
            if method not in ATTRIBUTION_METHODS:
                raise ValueError(
                    "formal attribution requires a cited trajectory assessment"
                )
    elif row.get("review_required") is not False or status != "excluded":
        raise ValueError("non-model failures must be excluded from PULSE review")


def _released_visual_packets(
    record: Mapping[str, Any],
    task: Mapping[str, Any] | None,
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select only initially provided or actually delivered historical frames."""
    missing: list[str] = []
    asset_root = root
    archive = record.get("meta", {}).get("runtime", {}).get("visual_asset_root")
    if archive is not None:
        reference = Path(archive)
        if reference.is_absolute():
            reference = reference.resolve().relative_to(root.resolve())
        asset_root = (root.resolve() if reference == Path(".") else resolve_within(
            root, reference, context="historical visual archive", must_exist=False
        ))
    definitions = (task or {}).get("visual_packets", [])
    frame_packets = {
        frame: packet["id"]
        for packet in definitions
        for frame in packet_asset_paths(packet)
    }
    session = (task or {}).get("visual_session", {})
    for frame in packet_asset_paths(session):
        frame_packets.setdefault(frame, "stream")
    initial = record.get("initial_context")
    releases: list[tuple[int, str, Mapping[str, Any]]] = []
    recorded_inputs = record.get("meta", {}).get("released_visual_inputs")
    if recorded_inputs is not None:
        releases = [
            (packet["step"], "recorded_released_visual_input", packet)
            for packet in recorded_inputs
        ]
    elif isinstance(initial, Mapping):
        releases.extend(
            (0, "recorded_initial_context", packet)
            for packet in initial.get("observations", [])
        )
    elif task is not None:
        condition = record.get("condition", {})
        scope = condition.get("scope")
        if condition.get("frame_interval_seconds") is not None:
            missing.append("initial_visual_delivery")
        elif scope in {"full", "first_frame", "initial_only"}:
            for packet in definitions:
                if packet["id"] == "V0" or (
                    scope == "full"
                    and condition.get("schedule") == "upfront"
                    and packet["id"] == "V1"
                ):
                    paths = packet.get("frame_paths", [])
                    if scope == "first_frame":
                        paths = paths[:1]
                    releases.append(
                        (
                            0,
                            "reconstructed_initial_from_task_and_condition",
                            {**packet, "frame_paths": paths},
                        )
                    )
        elif scope != "none":
            missing.append("initial_visual_delivery")
    else:
        missing.append("initial_visual_delivery")
    for event in record.get("actions", []) if recorded_inputs is None else []:
        if event.get("type") != "visual_context_update":
            continue
        payload = event.get("payload")
        step = event.get("step")
        if (
            not isinstance(payload, Mapping)
            or not isinstance(payload.get("frame_paths"), list)
            or type(step) is not int
        ):
            missing.append("visual_update_delivery_payload")
            continue
        releases.append((step, "recorded_visual_context_update", payload))
    packets = []
    for step, source, payload in releases:
        paths = []
        retained_indices = []
        packet_id = payload.get("id")
        for index, reference in enumerate(payload.get("frame_paths", [])):
            if reference not in frame_packets or (
                payload.get("id") is not None
                and payload["id"] != frame_packets[reference]
            ):
                missing.append(f"frame_not_in_historical_task:{reference}")
                continue
            path = resolve_within(
                asset_root,
                reference,
                context="released historical frame",
                must_exist=False,
            )
            if path.is_file():
                paths.append(path.relative_to(root.resolve()).as_posix())
                retained_indices.append(index)
                if packet_id is None:
                    packet_id = frame_packets[reference]
            else:
                missing.append(f"historical_frame:{reference}")
        if paths:
            packets.append(
                {
                    "id": packet_id,
                    "frame_paths": paths,
                    "released_at_step": step,
                    "release_source": source,
                    **{
                        key: payload[key]
                        for key in (
                            "frame_interval_seconds",
                            "frame_repeat_to_interval_seconds",
                            "wearer_frames_per_image",
                            "wearer_jpeg_quality",
                            "wearer_max_image_edge",
                        )
                        if key in payload
                    },
                    **(
                        {
                            "timestamps_seconds": [
                                payload["timestamps_seconds"][index]
                                for index in retained_indices
                            ]
                        }
                        if "timestamps_seconds" in payload
                        else {}
                    ),
                }
            )
    return packets, missing


def _entity_ids(value: Any) -> set[str]:
    if isinstance(value, list):
        return {item for child in value for item in _entity_ids(child)}
    if not isinstance(value, Mapping):
        return set()
    result: set[str] = set()
    for key, item in value.items():
        if key == "entity_id" and isinstance(item, str):
            result.add(item)
        elif key.endswith("entity_ids") or key == "candidate_ids":
            if isinstance(item, list):
                result.update(str(entity_id) for entity_id in item)
        result.update(_entity_ids(item))
    return result


def referenced_public_entities(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Give Judge and reviewers the same linked, historical public entity facts."""
    site = context.get("site_context")
    if not site:
        return []
    linked: set[str] = set()
    for key in ("task_context", "expected_states", "actual_final_state", "actions"):
        linked.update(_entity_ids(context.get(key)))
    entities = {str(item["entity_id"]): item
                for item in public_site_data(site["site"], site)["entities"]}
    return [entities.get(entity_id, {"entity_id": entity_id, "status": "absent"})
            for entity_id in sorted(linked)]


def historical_review_context(
    row: Mapping[str, Any], root: Path | None
) -> dict[str, Any]:
    """Expose recorded evidence, never substitute changed task or website content."""
    missing: list[str] = []
    if root is None or not row.get("run_artifact"):
        return {"missing_evidence": ["source_run_artifact"]}
    directory = resolve_within(
        root, row["run_artifact"], context="historical review", must_exist=False
    )
    record = load_run_record(directory)
    if record["meta"]["run_id"] != row["run_id"] or record["task_id"] != row["task_id"]:
        raise ValueError("historical review run identity mismatch")
    if (row.get("track") is not None and row["track"] != record.get("track")) or (
        row.get("model") is not None and row["model"] != record["meta"].get("model")
    ):
        raise ValueError("historical review run identity mismatch")
    task_context: dict[str, Any] = {}
    historical_task = None
    snapshot = record.get("task_snapshot")
    if isinstance(snapshot, dict):
        if snapshot.get("id") != record["task_id"]:
            raise ValueError("historical task snapshot identity mismatch")
        historical_task = snapshot
        task_context = {
            "instructions": snapshot.get("instructions", {}),
            "source": "recorded_task_snapshot",
        }
    archived_task = directory / "task.json"
    if not task_context and archived_task.is_file():
        task = json.loads(archived_task.read_text(encoding="utf-8"))
        if task["id"] != record["task_id"]:
            raise ValueError("archived task snapshot identity mismatch")
        historical_task = task
        task_context["instructions"] = task.get("instructions", {})
        task_context["source"] = "archived_task_snapshot"
    if not task_context:
        missing.append("historical_task_snapshot")
    if historical_task is not None:
        task_context["reference"] = {
            key: deepcopy(historical_task[key])
            for key in ("goal", "graft", "refresh", "clarification")
            if key in historical_task
        }
    packets, missing_frames = _released_visual_packets(record, historical_task, root)
    task_context["visual_packets"] = packets
    missing.extend(missing_frames)
    history = []
    trace = record.get("trace")
    source_history = trace.get("history", []) if isinstance(trace, Mapping) else trace
    if not isinstance(source_history, list) or not source_history:
        missing.append("historical_web_observations")
        source_history = []
    for item in source_history:
        if not isinstance(item, Mapping):
            missing.append("malformed_historical_web_observation")
            continue
        state_message = item.get("state_message")
        if isinstance(state_message, str):
            state_message = re.sub(
                r"<(agent_history|agent_state)>.*?</\1>",
                "",
                state_message,
                flags=re.S,
            )
        model_output = item.get("model_output")
        history.append(
            {
                "metadata": item.get("metadata"),
                "state": item.get("state"),
                "state_message": state_message,
                "actions": model_output.get("action")
                if isinstance(model_output, Mapping)
                else item.get("actions"),
                "result": item.get("result"),
                "response_text": item.get("response_text"),
            }
        )
    initial_context = record.get("initial_context")
    actual_initial_input = {
        key: deepcopy(initial_context[key])
        for key in ("instruction", "clarification_available")
        if isinstance(initial_context, Mapping) and key in initial_context
    }
    if (
        not isinstance(actual_initial_input.get("instruction"), str)
        or not actual_initial_input["instruction"].strip()
    ):
        missing.append("initial_instruction")
    if "live_state" not in record:
        missing.append("recorded_end_state")
    site_snapshot = record["meta"].get("runtime", {}).get("site_snapshot")
    site_context = None
    site_reference = None
    if site_snapshot:
        reference = Path(site_snapshot)
        if reference.is_absolute():
            reference = reference.resolve().relative_to(root.resolve())
        path = resolve_within(
            root, reference, context="historical site snapshot", must_exist=False
        )
        if path.is_file():
            site_context = json.loads(path.read_text(encoding="utf-8"))
            site_reference = path.relative_to(root.resolve()).as_posix()
            if site_context["site"] != record["meta"]["site"]:
                raise ValueError("historical site snapshot identity mismatch")
        else:
            missing.append("historical_site_snapshot")
    else:
        missing.append("historical_site_snapshot")
    scoring_goal = row.get("scoring_goal")
    expected_states = None
    if isinstance(scoring_goal, dict):
        expected_states = deepcopy(
            acceptable_states_of(scoring_goal, "PULSE scoring_goal")
        )
    else:
        missing.append("checker_scoring_goal")
    settings = trace.get("agent_settings", {}) if isinstance(trace, Mapping) else {}
    requests = trace.get("llm_requests", []) if isinstance(trace, Mapping) else []
    agent_protocol = {key: deepcopy(settings[key]) for key in (
        "system_prompt", "system_prompt_policy", "response_format",
        "max_actions_per_step", "max_output_tokens", "viewport",
    ) if key in settings}
    request_outcomes = [{key: deepcopy(request[key]) for key in (
        "step", "success", "max_output_tokens", "response_finish_reason", "model_output_error",
    ) if key in request} for request in requests]
    return {
        "task_context": task_context,
        "agent_protocol": agent_protocol,
        "request_outcomes": request_outcomes,
        "recorded_steps": record.get("steps"),
        "max_steps": record["meta"].get("runtime", {}).get("max_steps"),
        "run_identity": {
            "run_id": record["meta"]["run_id"],
            "task_id": record["task_id"],
            "track": record.get("track"),
            "model": record["meta"].get("model"),
            "run_valid": record.get("run_status", {}).get("valid"),
        },
        "actual_initial_input": actual_initial_input,
        "condition": deepcopy(record.get("condition")),
        "expected_states": expected_states,
        "scoring_goal_source": (
            "checker_evaluator" if expected_states is not None else "missing"
        ),
        "initial_state": record.get("initial_state"),
        "actual_final_state": record.get("live_state"),
        "end_state_source": "recorded_live_state"
        if "live_state" in record
        else "missing",
        "actions": record.get("actions", []),
        "web_observations": history,
        "site_context": site_context,
        "site_reference": site_reference,
        "site_entity_count": len(site_context["entities"])
        if site_context is not None
        else None,
        "missing_evidence": missing,
    }


def verify_fault_references(row: Mapping[str, Any], root: Path | None) -> None:
    certificate = row.get("qualification_evidence")
    if certificate is None:
        return
    if root is None or not row.get("run_artifact"):
        raise ValueError(
            "benchmark fault verification requires source artifact and root"
        )
    if certificate.get("source_run_id") != row["run_id"]:
        raise ValueError("benchmark fault requires matching source_run_id")
    if certificate.get("review_method") not in {"coding_agent", "human"}:
        raise ValueError(
            "benchmark fault requires coding_agent or human semantic review; no deterministic validator is registered"
        )
    if not certificate.get("reviewer_id"):
        raise ValueError("benchmark fault requires reviewer_id")
    directory = resolve_within(
        root, row["run_artifact"], context="fault source", must_exist=False
    )
    record = load_run_record(directory)
    if record["meta"]["run_id"] != row["run_id"]:
        raise ValueError("benchmark fault source artifact mismatch")
    snapshot = record.get("task_snapshot", {})
    if snapshot and snapshot["id"] != record["task_id"]:
        raise ValueError("fault task snapshot identity mismatch")
    frame_references = {
        reference
        for packet in snapshot.get("visual_packets", [])
        for reference in packet_asset_paths(packet)
    }
    site_snapshot = record["meta"].get("runtime", {}).get("site_snapshot")
    frozen_site = (root / site_snapshot).resolve() if site_snapshot else None
    for reference in certificate["evidence_refs"]:
        path = resolve_within(root, reference, context="fault evidence")
        if path.is_relative_to(directory):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in frame_references or path == frozen_site:
            continue
        raise ValueError("benchmark fault evidence is not bound to the historical run")


def qualify_attribution_row(
    source: Mapping[str, Any],
    qualification: str | Mapping[str, Any] | None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Apply the current validity review without inventing a new score."""
    row = deepcopy(dict(source))
    if qualification is None:
        return row
    checked = {
        "run_valid": row["run_verdict"] != "invalid_run",
        "e2e_success": row["run_verdict"] == "success",
    }
    verdict = _run_verdict(checked, qualification)
    if verdict != row["run_verdict"]:
        row.update(
            run_verdict=verdict,
            pulse_stage=None,
            mechanism=None,
            first_decisive_step=None,
            candidate_stage=None,
            candidate_mechanism=None,
            candidate_step=None,
            candidate_source=None,
            candidate_confidence=None,
            candidate_unknown_boundary=None,
            observable_facts=["task_validity_pending"] if verdict == "unknown" else [],
            secondary_symptoms=[],
            attribution_method=None,
            confidence=None,
            attribution_status="excluded",
            review_required=False,
            rule_evidence=None,
            qualification_evidence=deepcopy(
                _benchmark_fault_certificate(qualification)
            ),
        )
        validate_attribution(row)
        verify_fault_references(row, root)
    return row


