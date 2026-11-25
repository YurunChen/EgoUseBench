"""Derive and validate Coding-Agent and human selective-review routes."""

from __future__ import annotations

import math
import random
from copy import deepcopy
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from .final_state import evaluate_final_state
from .run_ledger import canonical_sha256, task_sha256
from .construct_validity import CONFIDENCE_POLICY_VERSION, EVIDENCE_DIMENSIONS

HIGH_CONFIDENCE = "high"
LOW_CONFIDENCE = "low"
REJECTED_CONFIDENCE = "rejected"
UNASSESSED_CONFIDENCE = "unassessed"
CONFIDENCE_LEVELS = {
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    REJECTED_CONFIDENCE,
    UNASSESSED_CONFIDENCE,
}


def high_confidence_sample(
    tasks: Mapping[str, Mapping[str, Any]],
    confidence_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose an independent, reproducible quality-control sample of high decisions."""

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for task_id, task in sorted(tasks.items()):
        row = confidence_rows.get(task_id, {})
        if row.get("level") != HIGH_CONFIDENCE or row.get("task_sha256") != task_sha256(
            task
        ):
            continue
        confidence_review_required(row)
        strata[(str(task["track"]), str(task["site"]))].append(task_id)
    population = sum(len(group) for group in strata.values())
    sample_size = min(population, max(10, math.ceil(population * 0.2), len(strata)))
    rng = random.Random(0)
    for group in strata.values():
        rng.shuffle(group)
    selected: list[str] = []
    while len(selected) < sample_size:
        for key in sorted(strata):
            if strata[key] and len(selected) < sample_size:
                selected.append(strata[key].pop())
    return {
        "policy": "high-confidence-qc-v1",
        "population": population,
        "sample_size": sample_size,
        "fraction": 0.2,
        "minimum": 10,
        "seed": 0,
        "task_ids": selected,
        "purpose": "quality_control_not_admission",
        "review_recording": "construction_submission_with_quality_control_purpose",
    }


def confidence_review_required(row: Mapping[str, Any]) -> bool:
    """Fail closed when one confidence row violates its routing contract."""

    if row.get("policy_version") != CONFIDENCE_POLICY_VERSION:
        raise ValueError("confidence row uses a stale policy")
    level = row.get("level")
    if not isinstance(level, str) or level not in CONFIDENCE_LEVELS:
        raise ValueError(f"invalid confidence level: {level!r}")
    required = row.get("human_review_required")
    if not isinstance(required, bool):
        raise TypeError("confidence row must declare human_review_required")
    allowed = {
        HIGH_CONFIDENCE: {"automatic_construction_admission": False},
        LOW_CONFIDENCE: {
            "repair": False,
            "evidence_collection_pending": False,
            "human_review_pending": True,
            "human_review_complete": False,
        },
        REJECTED_CONFIDENCE: {"repair": False},
        UNASSESSED_CONFIDENCE: {"coding_agent_retry_pending": False},
    }
    route = row.get("route")
    if not isinstance(route, str):
        raise ValueError(f"invalid confidence route: {route!r}")
    expected = allowed[level].get(route)
    if expected is None or required is not expected:
        raise ValueError(
            f"confidence level {level!r} is incompatible with route {route!r}"
        )
    return required


def _low_row(task_hash: str, *, reasons: list[str]) -> dict[str, Any]:
    return {
        "task_sha256": task_hash,
        "policy_version": CONFIDENCE_POLICY_VERSION,
        "level": LOW_CONFIDENCE,
        "agent_verdict": None,
        "agent_confidence": "low",
        "assessment_count": 0,
        "assessment_consensus": False,
        "uncertainty_reasons": reasons,
        "human_review_required": True,
        "route": "human_review_pending",
    }


def unassessed_confidence(task_hash: str, *, reason: str) -> dict[str, Any]:
    return {
        "task_sha256": task_hash,
        "policy_version": CONFIDENCE_POLICY_VERSION,
        "level": UNASSESSED_CONFIDENCE,
        "agent_verdict": None,
        "agent_confidence": None,
        "assessment_count": 0,
        "assessment_consensus": False,
        "assessment_errors": [reason],
        "uncertainty_reasons": [],
        "human_review_required": False,
        "route": "coding_agent_retry_pending",
    }


def blind_state_matches_expected(value: Any, expected_goal: Mapping[str, Any] | None) -> bool:
    if not isinstance(value, dict):
        return False
    if "acceptable_states" in value and "selected_entity_ids" not in value:
        endings = value["acceptable_states"]
        return (
            isinstance(endings, list)
            and bool(endings)
            and all(
                blind_state_matches_expected(ending, expected_goal)
                for ending in endings
            )
        )
    fields = value.get("fields", {})
    if (
        set(value) - {"selected_entity_ids", "fields", "submission_completed"}
        or not isinstance(value.get("selected_entity_ids"), list)
        or any(
            not isinstance(entity_id, str) for entity_id in value["selected_entity_ids"]
        )
        or not isinstance(fields, dict)
    ):
        return False
    if expected_goal is None:
        return True
    return evaluate_final_state(expected_goal, {"selected_entity_ids": value["selected_entity_ids"], "fields": fields})


def review_evidence_reuse_allowed(result: Mapping[str, Any]) -> bool:
    """Keep history reusable unless the current review explicitly reopened it."""
    return result.get("status") != "in_progress" and (
        result.get("status") != "pending"
        or result.get("pending_reason")
        in {
            "review_materials_changed",
            "answer_boundary_changed",
            "initial_assessment",
            "new_task_requires_assessment",
            "review_materials_missing",
        }
    )


def unavailable_assessment_confidence(
    task: Mapping[str, Any], result: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe missing evidence without claiming a material change from hashes alone."""
    from .construct_validity import POLICY

    completed = [
        source
        for source in [result, *result.get("history", [])]
        if source.get("status") == "completed"
        and len(source.get("assessments", [])) == 2
    ]
    if result.get("status") == "in_progress" or (
        completed and not review_evidence_reuse_allowed(result)
    ):
        reason = "本题审核已明确重开或正在进行；保留旧评语，等待当前审核完成。"
    elif not completed:
        reason = "当前记录及其历史中未找到已完成的两阶段具体评语；需先找回依据，再补查缺失部分。"
    elif all(
        source.get("materials_sha256") != packet.get("materials_sha256")
        for source in completed
    ):
        reason = "已找到历史具体评语，但尚无依据证明其审核材料与当前材料等价；需核对原材料并确定受影响部分。"
    elif any(
        source.get("materials_sha256") == packet.get("materials_sha256")
        and "answer_boundary" not in source
        for source in completed
    ):
        reason = "旧具体评语已保留；初末答案边界尚未登记前向跟踪基线，不代表需要重新审核。"
    else:
        reason = "已找到材料匹配的历史评语，但评语有效性或答案边界尚未通过核验；需检查原记录。"
    row = unassessed_confidence(task_sha256(task), reason=reason)
    if completed:
        row["historical_assessments"] = [
            {
                "assessment_report": POLICY["assessment"]["report"],
                "task_id": task["id"],
                "materials_sha256": source.get("materials_sha256"),
                "expected_states_sha256": source.get("expected_states_sha256"),
                "answer_boundary": source.get("answer_boundary"),
                "assessments_sha256": canonical_sha256(source["assessments"]),
            }
            for source in completed
        ]
    return row


def reusable_review_evidence(
    task: Mapping[str, Any], result: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Find completed task-local evidence without mutating the review archive."""
    from .construct_validity import PASS_NAMES
    from .confidence_packets import answer_boundary

    if not review_evidence_reuse_allowed(result):
        return None
    current_hash = packet["materials_sha256"]
    retained_packet = packet.get("retained_instruction_packet")
    if retained_packet is not None:
        from .confidence_packets import validate_review_packet

        validate_review_packet(retained_packet)
    candidates = [result, *reversed(result.get("history", []))]
    for source in candidates:
        source_hash = source.get("materials_sha256")
        comparison_hash = (
            retained_packet["materials_sha256"]
            if retained_packet is not None and source_hash != current_hash
            else current_hash
        )
        if source_hash != comparison_hash:
            continue
        boundary = source.get("answer_boundary")
        if source.get("status") != "completed" or not isinstance(boundary, Mapping):
            continue
        verification = source.get("material_verification")
        if verification is not None:
            try:
                retained = verified_blind_evidence(task, result, verification)
            except ValueError:
                continue
            if (
                verification["current_materials_sha256"] != comparison_hash
                or source["assessments"][:1] != retained["assessments"][:1]
            ):
                continue
        if (
            route_assessment(task_sha256(task), source)["level"]
            == UNASSESSED_CONFIDENCE
        ):
            continue
        answers_unchanged = boundary == answer_boundary(task)
        return {
            "source": deepcopy(dict(source)),
            "recheck_passes": ([PASS_NAMES[1]] if not answers_unchanged else []),
            "pending_reason": "answer_boundary_changed"
            if not answers_unchanged
            else None,
        }
    return None


def verified_blind_evidence(
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve a concrete, task-local factual verification to its original blind pass."""
    from .construct_validity import PASS_NAMES, validate_assessment_verdict

    if not isinstance(verification, Mapping):
        raise ValueError("material verification must be an object")
    current_passes = result.get("assessments", [])
    current_blind_is_referenced = (
        result.get("status") == "in_progress"
        and len(current_passes) == 1
        and result.get("materials_sha256")
        == verification.get("source_materials_sha256")
        and canonical_sha256(current_passes[0])
        == verification.get("source_assessment_sha256")
    )
    if not review_evidence_reuse_allowed(result) and not current_blind_is_referenced:
        raise ValueError("review was explicitly reopened or is in progress")
    if result.get("task_id") != task["id"] or verification.get("task_id") != task["id"]:
        raise ValueError("material verification must belong to this task")
    reviewer = verification.get("reviewer")
    if (
        not isinstance(reviewer, Mapping)
        or set(reviewer) != {"agent_id", "context_id"}
        or any(
            not isinstance(value, str) or not value.strip()
            for value in reviewer.values()
        )
    ):
        raise ValueError("material verification requires the actual reviewer identity")
    changes = verification.get("changes")
    if (
        not isinstance(changes, list)
        or not changes
        or any(not isinstance(change, str) or not change.strip() for change in changes)
    ):
        raise ValueError("material verification must describe the actual change scope")
    checks = verification.get("evidence")
    if not isinstance(checks, Mapping) or set(checks) != set(EVIDENCE_DIMENSIONS):
        raise ValueError("material verification must cover every factual dimension")
    source_found = False
    for source in [result, *result.get("history", [])]:
        passes = source.get("assessments", [])
        if (
            source.get("materials_sha256")
            != verification.get("source_materials_sha256")
            or source.get("status") not in {"completed", "in_progress"}
            or len(passes) not in {1, 2}
            or (len(passes) == 2 and source.get("status") != "completed")
            or canonical_sha256(passes[0])
            != verification.get("source_assessment_sha256")
        ):
            continue
        source_found = True
        partial = len(passes) == 1
        if partial:
            if (
                passes[0].get("status") != "completed"
                or passes[0].get("pass") != PASS_NAMES[0]
            ):
                raise ValueError(
                    "source blind review is incomplete or has the wrong role"
                )
            validate_assessment_verdict(passes[0]["verdict"], pass_name=PASS_NAMES[0])
        elif (
            route_assessment(task_sha256(task), source)["level"]
            == UNASSESSED_CONFIDENCE
        ):
            raise ValueError("source review has no valid completed two-stage body")
        blind = passes[0]["verdict"]
        if partial:
            if not blind_state_matches_expected(
                blind.get("derived_final_state"), task["goal"]
            ):
                raise ValueError(
                    "source blind answer does not match a current legal ending"
                )
        elif blind["decision"] == "repair":
            if not (
                (blind.get("issues") or blind.get("uncertainty_reasons"))
                and any(
                    status in {"fail", "uncertain"}
                    for status in blind["gates"].values()
                )
            ):
                raise ValueError(
                    "repair inheritance requires a concrete original blind concern"
                )
        elif blind["decision"] != "admit":
            raise ValueError("source blind review does not support inheritance")
        elif not blind_state_matches_expected(
            blind.get("derived_final_state"), task["goal"]
        ):
            raise ValueError(
                "source blind answer does not match a current legal ending"
            )
        prior_facts = {
            fact
            for assessment in passes
            for facts in assessment["verdict"]["evidence"].values()
            for fact in facts
            if isinstance(fact, str)
        }
        for check in checks.values():
            if not isinstance(check, Mapping) or check.get("conclusion") != "supported":
                raise ValueError(
                    "unresolved factual differences cannot inherit a blind pass"
                )
            for field in ("source_evidence", "current_evidence"):
                facts = check.get(field)
                if (
                    not isinstance(facts, list)
                    or not facts
                    or any(
                        not isinstance(fact, str) or not fact.strip() for fact in facts
                    )
                ):
                    raise ValueError(
                        "each factual check needs specific old and current evidence"
                    )
        if all(
            set(check["source_evidence"]) <= prior_facts for check in checks.values()
        ):
            return deepcopy(dict(source))
    if source_found:
        raise ValueError("source evidence must quote the retained task's actual review")
    raise ValueError("the referenced original blind review body was not found")


def _apply_blind_resolutions(
    blind: Mapping[str, Any], resolved: set[str]
) -> dict[str, Any]:
    """Project each resolved concern without rewriting the sealed review."""
    from .construct_validity import blind_concerns

    current = deepcopy(dict(blind))
    for dimension in current["gates"]:
        if f"gates.{dimension}" in resolved:
            current["gates"][dimension] = "pass"
    for field in ("uncertainty_reasons", "issues"):
        current[field] = [
            text
            for index, text in enumerate(blind.get(field, []))
            if f"{field}.{index}" not in resolved
        ]
    if resolved == set(blind_concerns(blind)):
        current.update(decision="admit", confidence="high")
    return current


def route_assessment(
    task_hash: str,
    result: Mapping[str, Any],
    *,
    expected_goal: Mapping[str, Any] | None = None,
    assessment_body_reused: bool = False,
) -> dict[str, Any]:
    from .construct_validity import (
        PASS_NAMES,
        ConstructValidityError,
        combine_pass_gates,
        validate_blind_resolutions,
        validate_assessment_verdict,
    )

    if assessment_body_reused:
        return unassessed_confidence(
            task_hash,
            reason="assessment body was reused unchanged across different tasks",
        )
    if "proposed_entities" in result:
        return unassessed_confidence(
            task_hash,
            reason="assessment record contains answer-bearing entity roles",
        )
    if "site_audit" in result:
        return unassessed_confidence(
            task_hash,
            reason="assessment record uses an answer-bearing site audit",
        )
    assessments = result.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != 2:
        return unassessed_confidence(
            task_hash, reason="two completed Coding-Agent assessments are required"
        )

    resolution_reviews = result.get("resolution_reviews", [])
    if not isinstance(resolution_reviews, list) or not all(
        isinstance(review, dict) for review in resolution_reviews
    ):
        return unassessed_confidence(
            task_hash, reason="blind resolution reviews are malformed"
        )
    signals: list[str] = []
    uncertainty_reasons: list[str] = []
    pass_gate_statuses: list[dict[str, str]] = []
    for index, assessment in enumerate(assessments):
        if not isinstance(assessment, dict) or assessment.get("status") != "completed":
            return unassessed_confidence(
                task_hash,
                reason=f"Coding-Agent assessment {index + 1} did not complete",
            )
        if assessment.get("pass") != PASS_NAMES[index]:
            return unassessed_confidence(
                task_hash,
                reason=f"Coding-Agent assessment {index + 1} has the wrong pass role",
            )
        verdict = assessment.get("verdict")
        if not isinstance(verdict, dict):
            return unassessed_confidence(
                task_hash, reason=f"Coding-Agent assessment {index + 1} is malformed"
            )
        decision = verdict.get("decision")
        confidence = verdict.get("confidence")
        issues = verdict.get("issues", [])
        try:
            admission_gates = validate_assessment_verdict(
                verdict,
                pass_name=PASS_NAMES[index],
                blind_verdict=assessments[0]["verdict"] if index else None,
            )
            if index == 0:
                second_assessment = assessments[1]
                if not isinstance(second_assessment, Mapping) or not isinstance(
                    second_assessment.get("verdict"), Mapping
                ):
                    raise ConstructValidityError("second assessment is malformed")
                second = second_assessment["verdict"]
                resolutions = second.get("blind_resolutions")
                if resolution_reviews:
                    latest = resolution_reviews[-1]
                    if set(latest) != {"reviewer", "blind_resolutions"} or latest[
                        "reviewer"
                    ] != second.get("reviewer"):
                        raise ConstructValidityError(
                            "resolution reviewer must be the second-pass reviewer"
                        )
                    resolutions = latest["blind_resolutions"]
                if resolutions is not None:
                    validate_assessment_verdict(
                        second, pass_name=PASS_NAMES[1], blind_verdict=verdict
                    )
                    resolved = validate_blind_resolutions(resolutions, verdict, second)
                    if expected_goal is not None and blind_state_matches_expected(
                        verdict.get("derived_final_state"), expected_goal
                    ):
                        verdict = _apply_blind_resolutions(verdict, resolved)
                        decision = verdict["decision"]
                        confidence = verdict["confidence"]
                        issues = verdict["issues"]
                        admission_gates = validate_assessment_verdict(
                            verdict, pass_name=PASS_NAMES[0]
                        )
        except ConstructValidityError:
            return unassessed_confidence(
                task_hash, reason=f"Coding-Agent assessment {index + 1} is malformed"
            )
        if index == 1 and any(
            item["action"] == "repair" for item in verdict.get("follow_up", [])
        ):
            decision = "repair"
        reasons = verdict["uncertainty_reasons"]
        uncertainty_reasons.extend(reasons)
        pass_gate_statuses.append(admission_gates)

        if (
            index == 0
            and decision == "admit"
            and not blind_state_matches_expected(
                verdict.get("derived_final_state"), expected_goal
            )
        ):
            uncertainty_reasons.append(
                "blind reconstruction does not match an expected final state"
            )
            continue
        if index == 1:
            checks = verdict.get("counterexample_checks")
            if not isinstance(checks, Mapping):
                return unassessed_confidence(
                    task_hash,
                    reason="adversarial counterexample statuses are malformed",
                )
            if decision == "admit" and "uncertain" in checks.values():
                uncertainty_reasons.append(
                    "adversarial assessment left a counterexample unresolved"
                )
                continue
            if decision == "admit" and any(
                value != "pass" for value in checks.values()
            ):
                return unassessed_confidence(
                    task_hash,
                    reason="adversarial assessment contradicts its admit verdict",
                )

        if (
            confidence == "high"
            and decision == "repair"
            and "fail" in admission_gates.values()
            and issues
        ):
            signals.append("high_repair")
            continue
        if confidence == "low" or reasons:
            uncertainty_reasons.extend(issues)
            if not reasons:
                uncertainty_reasons.append(
                    f"Coding-Agent assessment {index + 1} reported low confidence"
                )
            continue
        if decision == "admit":
            if issues:
                uncertainty_reasons.extend(issues)
            elif "uncertain" in admission_gates.values():
                uncertainty_reasons.append(
                    f"Coding-Agent assessment {index + 1} left an admission gate unresolved"
                )
            elif all(value == "pass" for value in admission_gates.values()):
                signals.append("high_admit")
            else:
                uncertainty_reasons.append(
                    f"Coding-Agent assessment {index + 1} has inconsistent fields"
                )
        else:
            uncertainty_reasons.append(
                f"Coding-Agent assessment {index + 1} has inconsistent fields"
            )

    assessment_hash = canonical_sha256(assessments)
    combined_gates = combine_pass_gates(pass_gate_statuses)
    follow_up = assessments[1]["verdict"].get("follow_up", [])
    actions = {item["action"] for item in follow_up}
    if "repair" in actions:
        return {
            **_low_row(task_hash, reasons=[item["reason"] for item in follow_up]),
            "agent_verdict": "repair",
            "assessment_count": 2,
            "agent_assessments_sha256": assessment_hash,
            "admission_gates": combined_gates,
            "follow_up": deepcopy(follow_up),
            "human_review_required": False,
            "route": "repair",
        }
    if "high_repair" not in signals and follow_up:
        collecting = "collect_evidence" in actions
        return {
            **_low_row(
                task_hash,
                reasons=list(
                    dict.fromkeys(
                        [*uncertainty_reasons, *(item["reason"] for item in follow_up)]
                    )
                ),
            ),
            "assessment_count": 2,
            "agent_assessments_sha256": assessment_hash,
            "admission_gates": combined_gates,
            "follow_up": deepcopy(follow_up),
            "human_review_required": not collecting,
            "route": "evidence_collection_pending"
            if collecting
            else "human_review_pending",
        }
    if signals == ["high_admit", "high_admit"]:
        return {
            "task_sha256": task_hash,
            "policy_version": CONFIDENCE_POLICY_VERSION,
            "level": HIGH_CONFIDENCE,
            "agent_verdict": "admit",
            "agent_confidence": "high",
            "assessment_count": 2,
            "assessment_consensus": True,
            "agent_assessments_sha256": assessment_hash,
            "admission_gates": combined_gates,
            "uncertainty_reasons": [],
            "human_review_required": False,
            "route": "automatic_construction_admission",
        }
    if "high_repair" in signals:
        return {
            "task_sha256": task_hash,
            "policy_version": CONFIDENCE_POLICY_VERSION,
            "level": REJECTED_CONFIDENCE,
            "agent_verdict": "repair",
            "agent_confidence": "high",
            "assessment_count": 2,
            "assessment_consensus": signals == ["high_repair", "high_repair"]
            and any(
                all(statuses[gate] == "fail" for statuses in pass_gate_statuses)
                for gate in combined_gates
            ),
            "agent_assessments_sha256": assessment_hash,
            "admission_gates": combined_gates,
            "uncertainty_reasons": [],
            "human_review_required": False,
            "route": "repair",
        }
    reasons = (
        list(dict.fromkeys(uncertainty_reasons))
        if uncertainty_reasons
        else ["the blind reconstruction and adversarial passes disagree"]
    )
    return {
        **_low_row(task_hash, reasons=reasons),
        "assessment_count": 2,
        "agent_assessments_sha256": assessment_hash,
        "admission_gates": combined_gates,
    }


def resolve_human_review(
    row: dict[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine a Coding-Agent route with the current verified human submission."""

    if review.get("human_review_reuse"):
        row = {**row, "human_review_reuse": deepcopy(review["human_review_reuse"])}
    if row["level"] not in {LOW_CONFIDENCE, HIGH_CONFIDENCE} or row["route"] in {
        "repair",
        "evidence_collection_pending",
    }:
        return row
    review_status = review["status"]
    purpose = review.get("purpose", "admission")
    if review_status not in {"pending", "admit", "reject", "invalid"}:
        raise ValueError(f"unsupported human review status: {review_status!r}")
    if row["level"] == HIGH_CONFIDENCE:
        if review_status == "reject":
            return {
                **row,
                "level": REJECTED_CONFIDENCE,
                "route": "repair",
                "quality_control_status": "reject",
            }
        if review_status == "invalid":
            return {**row, "live_human_review_status": "invalid"}
        return (
            {**row, "quality_control_status": review_status}
            if purpose == "quality_control"
            else row
        )
    if review_status == "pending":
        return row
    resolved = {**row, "human_review_required": False}
    resolved.pop("live_human_review_status", None)
    if review_status == "admit":
        resolved["route"] = "human_review_complete"
    elif review_status == "reject":
        resolved["level"] = REJECTED_CONFIDENCE
        resolved["route"] = "repair"
    elif review_status == "invalid":
        resolved["human_review_required"] = True
        resolved["route"] = "human_review_pending"
        resolved["live_human_review_status"] = "invalid"
    return resolved
