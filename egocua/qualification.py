"""Development qualification evidence from a real task-performing Agent.

Qualification and difficulty answer different questions.  This module only records
whether one current task version completed in the registered runtime.  A failed
trial is a diagnosis request, not a difficulty label and not an automatic rejection.
"""

from __future__ import annotations

from .final_state import goal_entity_ids

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .checker import behavioral_failure_mode
from .schema import clarification_basis, task_transition

QUALIFICATION_POLICY_VERSION = "task-runtime-development-screen-v2"

_HARD_PROVIDER_REFUSAL_MARKERS = (
    "insufficient_balance",
    "insufficient balance",
    "billing_hard_limit_reached",
    "quota_exceeded",
    "余额不足",
    "请充值后再使用",
)


def trace_has_provider_refusal(trace: Any) -> bool:
    """Return whether a provider hard-refused any part of the recorded trial."""

    if not isinstance(trace, Mapping):
        return False
    serialized = json.dumps(trace, ensure_ascii=False).lower()
    return any(marker in serialized for marker in _HARD_PROVIDER_REFUSAL_MARKERS)


def trace_has_model_action(trace: Any) -> bool:
    """Return whether Browser-Use produced a real post-navigation model action."""

    if not isinstance(trace, Mapping):
        return False
    history = trace.get("history")
    if not isinstance(history, list):
        return False
    for row in history:
        if not isinstance(row, Mapping):
            continue
        metadata = row.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("step_number") == 0:
            continue
        output = row.get("model_output")
        if isinstance(output, Mapping) and output.get("action"):
            return True
    return False


def trace_has_model_response(trace: Any) -> bool:
    """Distinguish received malformed/empty output from a request with no response."""
    if not isinstance(trace, Mapping):
        return False
    requests = trace.get("llm_requests", [])
    if not isinstance(requests, list):
        return False
    if any(
        isinstance(row, Mapping) and row.get("provider_service_error") is True
        for row in requests
    ):
        return False
    return trace_has_model_action(trace) or any(
        isinstance(row, Mapping)
        and (row.get("success") is True or row.get("model_output_error") is True)
        for row in requests
    )


def qualification_family(
    task: Mapping[str, Any],
    source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the shared mechanism that should be canaried before sibling trials."""

    track = str(task["track"])
    transition = task_transition(task)
    if track == "graft":
        initial = task.get("graft", {}).get("initial_goal", {})
        initial_shape = sorted(
            {
                len(state.get("selected_entity_ids", []))
                for state in initial.get("acceptable_states", [])
                if isinstance(state, Mapping)
            }
        )
    elif track == "refresh":
        initial_shape = [1]
    else:
        initial_shape = [2]

    construct = (
        source_record.get("construct", {}) if isinstance(source_record, Mapping) else {}
    )
    if not isinstance(construct, Mapping):
        construct = {}
    goal = task.get("goal", {})
    states = goal.get("acceptable_states", []) if isinstance(goal, Mapping) else []
    final_shape = sorted(
        {
            len(state.get("selected_entity_ids", []))
            for state in states
            if isinstance(state, Mapping)
        }
    )
    field_names = sorted(
        {
            str(name)
            for state in states
            if isinstance(state, Mapping)
            for name in (state.get("fields", {}) or {})
        }
    )
    return {
        "site": str(task["site"]),
        "track": track,
        "transition": transition,
        "clarification_basis": clarification_basis(task),
        "target_granularity": str(goal.get("target_granularity", "")),
        "initial_selection_shape": initial_shape,
        "final_selection_shape": final_shape,
        **({"quantity_normalization": True} if goal.get("unit_equivalences") or task.get("graft", {}).get("initial_goal", {}).get("unit_equivalences") else {}),
        "field_names": field_names,
        "evidence_type": str(construct.get("evidence_type") or "unspecified"),
        "commitment_surface": str(
            construct.get("commitment_surface")
            or {
                "egomart": "cart",
                "egomedia": "media_save",
                "egomaps": "place_identity",
            }.get(str(task["site"]), "unknown")
        ),
    }


def qualification_family_key(family: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(family),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _compact_actions(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    keep = {
        "type",
        "step",
        "query",
        "results",
        "shown_results",
        "page",
        "sort",
        "entity_id",
        "entity_ids",
        "quantity",
        "slot",
        "valid",
        "limit_reached",
        "field",
        "value",
        "state_before",
        "state_after",
    }
    actions = record.get("actions", [])
    if not isinstance(actions, list):
        return []
    return [
        {key: action[key] for key in keep if key in action}
        for action in actions
        if isinstance(action, Mapping)
    ]


def _outcome_symptom(checked: Mapping[str, Any]) -> str:
    mode = behavioral_failure_mode(checked)
    return {
        "stale_after_update": "stale_commitment_after_update",
        "submitted_before_update": "commitment_before_required_update",
        "incorrect_change_after_update": "unsupported_update",
        "wrong_final_state": "final_state_mismatch",
        "other_checked_failure": "checker_or_execution_mismatch",
    }.get(mode, mode)


def failure_diagnostic(
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    checked: Mapping[str, Any],
    *,
    family: Mapping[str, Any],
    related_task_ids: list[str],
    site_entities: list[Mapping[str, Any]],
    run_artifact: str,
) -> dict[str, Any]:
    """Build the smallest packet needed for task-vs-model failure diagnosis."""

    expected_ids = goal_entity_ids(task["goal"])
    final_state = record.get("live_state")
    actual_ids = {
        str(entity_id)
        for entity_id in (
            final_state.get("selected_entity_ids", [])
            if isinstance(final_state, Mapping)
            else []
        )
    }
    action_rows = _compact_actions(record)
    touched_ids = {
        str(action["entity_id"])
        for action in action_rows
        if action.get("entity_id") is not None
    }
    relevant_ids = expected_ids | actual_ids | touched_ids
    entities = []
    for entity in site_entities:
        entity_id = str(entity.get("entity_id", ""))
        if entity_id not in relevant_ids:
            continue
        entities.append(
            {
                key: entity[key]
                for key in (
                    "entity_id",
                    "name",
                    "title",
                    "brand",
                    "kind",
                    "address",
                    "attributes",
                    "summary",
                )
                if key in entity
            }
        )
    return {
        "task_id": str(task["id"]),
        "run_artifact": run_artifact,
        "trajectory_reference": run_artifact,
        "task_context": {
            "instructions": task.get("instructions", {}),
            "visual_packets": task.get("visual_packets", []),
            "track_change": task.get(str(task.get("track")), {}),
        },
        "observed_symptom": _outcome_symptom(checked),
        "expected_states": task.get("goal", {}).get("acceptable_states", []),
        "scoring_goal": task["goal"],
        "actual_final_state": final_state,
        "end_state_source": "recorded_live_state"
        if final_state is not None
        else "missing",
        "checker": {
            key: checked.get(key)
            for key in (
                "run_valid",
                "run_invalid_reasons",
                "e2e_success",
                "goal_state_correct",
                "policy_pass",
                "release_delivered",
                "asked",
                "valid_question",
                "timely_question",
                "answer_used",
                "premature_commitment",
                "premature_submission",
                "stale_final",
                "stale_commitment",
                "false_update",
            )
        },
        "actions": action_rows,
        "relevant_site_entities": entities,
        "family": dict(family),
        "related_task_ids": related_task_ids,
        "required_decision": "task_or_site_defect | valid_model_failure | invalid_run",
    }


def qualification_outcome(checked_run: Mapping[str, Any]) -> dict[str, Any]:
    """Route one real-model trial without interpreting failure as task difficulty."""

    run_valid = checked_run.get("run_valid")
    success = checked_run.get("e2e_success")
    status = "invalid_run"
    next_action = "rerun_protocol_or_infrastructure_failure"
    if run_valid is True and success is True:
        status = "trajectory_review_required"
        next_action = "review_development_trajectory"
    elif run_valid is True and success is False:
        status = "triage_required"
        next_action = "classify_task_defect_vs_valid_model_failure"
    return {
        "policy_version": QUALIFICATION_POLICY_VERSION,
        "status": status,
        "next_action": next_action,
        "e2e_success": success if isinstance(success, bool) else None,
        "run_valid": run_valid,
        "steps": checked_run.get("steps"),
        "premature_commitment": checked_run.get("premature_commitment"),
        "stale_commitment": checked_run.get("stale_commitment"),
        "false_update": checked_run.get("false_update"),
        "asked": checked_run.get("asked"),
        "policy_pass": checked_run.get("policy_pass"),
    }
