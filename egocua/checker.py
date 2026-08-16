from __future__ import annotations

from collections import Counter
from typing import Any

from .final_state import evaluate_final_state
from .protocol import DYNAMIC_EVENT_STEP, clarification_question_is_valid
from .schema import (
    acceptable_states_of,
    goal_for_arm,
    selected_entity_sets_of,
    single_entity_target,
    task_subtype,
)

EVIDENCE_PASS = "PASS"
EVIDENCE_FAIL = "FAIL"
EVIDENCE_UNKNOWN = "UNKNOWN"
EVALUATOR_VERSION = "2.0"

SEARCH_ACTIONS = {"search"}
OPEN_ACTIONS = {"open_entity", "open_product", "open_document", "open_video", "open_place"}
SELECTION_ACTIONS = {
    "select", "save", "select_entity", "deselect_entity",
    "add_to_cart", "remove_from_cart", "update_cart",
    "select_route", "select_destination",
}
CANDIDATE_COMMITTING_ACTIONS = SELECTION_ACTIONS | {"submit"}
ANSWER_USING_ACTIONS = OPEN_ACTIONS | SELECTION_ACTIONS | {"submit"}


def _action_entity_ids(action: dict[str, Any]) -> list[str]:
    plural = action.get("entity_ids", action.get("item_ids"))
    if isinstance(plural, list):
        return [value for value in plural if isinstance(value, str)]
    for key in ("entity_id", "item_id", "document_id", "video_id", "place_id"):
        value = action.get(key)
        if isinstance(value, str) and value:
            return [value]
    return []


def _candidate_targets(action: dict[str, Any], candidate_ids: set[str]) -> set[str]:
    return set(_action_entity_ids(action)) & candidate_ids


def acceptable_states(task: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    return acceptable_states_of(goal_for_arm(task, arm), f"{task['id']}/{arm}")


def _multiset_contains(container: list[str], required: list[str]) -> bool:
    return not (Counter(required) - Counter(container))


def _repeat_rate(actions: list[dict[str, Any]]) -> float:
    signatures = [
        (
            action.get("type"),
            tuple(_action_entity_ids(action)),
            str(action.get("query", "")).strip().casefold(),
        )
        for action in actions
        if action.get("type") in SEARCH_ACTIONS | OPEN_ACTIONS | SELECTION_ACTIONS
    ]
    if not signatures:
        return 0.0
    counts = Counter(signatures)
    return sum(count - 1 for count in counts.values()) / len(signatures)


def _workload(actions: list[dict[str, Any]]) -> dict[str, int]:
    searches = [action for action in actions if action.get("type") in SEARCH_ACTIONS]
    queries = {str(action.get("query", "")).strip().casefold() for action in searches}
    opened = [
        entity_id
        for action in actions if action.get("type") in OPEN_ACTIONS
        for entity_id in _action_entity_ids(action)
    ]
    mutations = [action for action in actions if action.get("type") in SELECTION_ACTIONS]
    return {
        "search_actions": len(searches),
        "distinct_search_queries": len(queries),
        "entity_detail_opens": len(opened),
        "distinct_entities_opened": len(set(opened)),
        "selection_mutations": len(mutations),
        "selection_corrections": sum(
            action.get("type") in {"deselect_entity", "remove_from_cart", "update_cart"}
            for action in mutations
        ),
    }


def _submission_status(final_state: dict[str, Any]) -> bool | None:
    value = final_state.get("submission_completed")
    return value if isinstance(value, bool) else None


def _required_event_type(task: dict[str, Any], arm: str) -> str | None:
    return {
        ("graft", "injected"): "instruction_injection",
        ("refresh", "stream"): "visual_context_update",
    }.get((task["track"], arm))


def _invalid_reasons(
    task: dict[str, Any], record: dict[str, Any], actions: list[dict[str, Any]]
) -> list[str]:
    reasons: list[str] = []
    run_status = record.get("run_status", {})
    if run_status.get("valid") is False:
        reasons.append(str(run_status.get("reason") or "declared_invalid"))
    final_state = record.get("final_state")
    if not isinstance(final_state, dict):
        reasons.append("missing_final_state")
    else:
        if not isinstance(final_state.get("selected_entity_ids"), list):
            reasons.append("missing_selected_entity_ids")
        if not isinstance(final_state.get("fields"), dict):
            reasons.append("missing_fields")
        if _submission_status(final_state) is None:
            reasons.append("missing_submission_status")

    required_event = _required_event_type(task, record["arm"])
    if required_event:
        delivered = [action for action in actions if action.get("type") == required_event]
        if len(delivered) != 1:
            reasons.append(f"expected_one_{required_event}")
        elif delivered[0].get("step", 0) < 2:
            reasons.append(f"{required_event}_before_agent_action")
        else:
            event = delivered[0]
            trigger_reason = event.get("trigger_reason")
            if trigger_reason != "task_timeline":
                reasons.append(f"invalid_{required_event}_trigger_reason")
            elif event.get("step") != DYNAMIC_EVENT_STEP:
                reasons.append(f"invalid_{required_event}_task_timeline_step")
            if any(
                action.get("type") == "submit"
                and action.get("step", 0) < event.get("step", 0)
                for action in actions
            ):
                reasons.append("submission_before_required_event")
    return reasons


def check_run(task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    arm = record["arm"]
    goal = goal_for_arm(task, arm)
    states = acceptable_states(task, arm)
    acceptable_ids = {
        entity_id for state in states for entity_id in state["selected_entity_ids"]
    }
    final_state = record.get("final_state", {})
    final_state = final_state if isinstance(final_state, dict) else {}
    submitted_entities = final_state.get("selected_entity_ids", [])
    submitted_entities = submitted_entities if isinstance(submitted_entities, list) else []
    submitted = set(submitted_entities)
    submission_completed = _submission_status(final_state)
    goal_state_correct = evaluate_final_state(
        goal, {**final_state, "submission_completed": True}
    )
    actions = record.get("actions", [])
    actions = actions if isinstance(actions, list) else []
    invalid_reasons = _invalid_reasons(task, record, actions)
    evidence_complete = (
        isinstance(final_state.get("selected_entity_ids"), list)
        and isinstance(final_state.get("fields"), dict)
        and submission_completed is not None
    )
    outcome_pass = evaluate_final_state(goal, final_state)
    evidence_status = (
        EVIDENCE_PASS if outcome_pass else EVIDENCE_FAIL
    ) if evidence_complete else EVIDENCE_UNKNOWN
    run_valid = not invalid_reasons
    result: dict[str, Any] = {
        "task_id": task["id"],
        "track": task["track"],
        "site": task["site"],
        "subtype": task_subtype(task),
        "arm": arm,
        "replicate_id": record.get("meta", {}).get("replicate_id"),
        "run_id": record.get("meta", {}).get("run_id"),
        "agent": record.get("meta", {}).get("agent"),
        "model": record.get("meta", {}).get("model"),
        "task_sha256": record.get("meta", {}).get("task_sha256"),
        "asset_sha256": record.get("meta", {}).get("asset_sha256"),
        "site_data_sha256": record.get("meta", {}).get("site_data_sha256"),
        "run_valid": run_valid,
        "run_invalid_reasons": invalid_reasons,
        "evidence_status": evidence_status,
        "evaluator": {
            "version": EVALUATOR_VERSION,
            "predicate": "exact_submitted_entities_fields_and_submission",
            "acceptable_states": states,
            "observed_state": final_state,
        },
        "goal_state_correct": goal_state_correct,
        "submission_completed": submission_completed,
        "policy_pass": True,
        "e2e_success": outcome_pass if run_valid and evidence_complete else None,
        "target_opened": any(
            action.get("type") in OPEN_ACTIONS
            and bool(set(_action_entity_ids(action)) & acceptable_ids)
            for action in actions
        ),
        "target_selected": any(
            action.get("type") in SELECTION_ACTIONS
            and bool(set(_action_entity_ids(action)) & acceptable_ids)
            for action in actions
        ),
        "steps": record.get("steps", 0),
        "repeat_rate": _repeat_rate(actions),
        **_workload(actions),
    }
    required_event = _required_event_type(task, arm)
    dynamic_events = [
        action for action in actions if required_event and action.get("type") == required_event
    ]
    result["dynamic_event_required"] = required_event is not None
    result["event_trigger_reason"] = (
        dynamic_events[0].get("trigger_reason") if len(dynamic_events) == 1 else None
    )

    if task["track"] == "graft":
        initial_options = selected_entity_sets_of(
            task["graft"]["initial_goal"], f"{task['id']}/graft/initial_goal"
        )
        final_options = selected_entity_sets_of(task["goal"], f"{task['id']}/goal")
        final_state_requires_a = all(
            any(_multiset_contains(option, initial) for initial in initial_options)
            for option in final_options
        )
        result["final_state_requires_a"] = final_state_requires_a
        result["final_state_contains_a"] = (
            any(_multiset_contains(submitted_entities, initial) for initial in initial_options)
            if final_state_requires_a else None
        )
        result["integration_success"] = result["e2e_success"]
    elif task["track"] == "refresh":
        refresh = task["refresh"]
        final_target_id = single_entity_target(task, arm)
        result["false_update"] = (
            bool(submitted) and submitted != {refresh["initial_entity_id"]}
            if refresh["policy"] == "retain" and arm == "stream" else None
        )
        update_events = [action for action in actions if action.get("type") == "visual_context_update"]
        final_selections = [
            action for action in actions
            if action.get("type") in SELECTION_ACTIONS
            and final_target_id in _action_entity_ids(action)
        ]
        post_update = (
            [
                selection for selection in final_selections
                if selection.get("step", 0) >= update_events[0].get("step", 0)
            ] if update_events else []
        )
        result["post_update_target_action_latency"] = (
            post_update[0].get("step", 0) - update_events[0].get("step", 0)
            if update_events and post_update else None
        )
        result["post_update_target_action_responded"] = (
            result["post_update_target_action_latency"] is not None
        )
    else:
        indexed_actions = list(enumerate(actions))
        asks = [
            (index, action) for index, action in indexed_actions
            if action.get("type") == "clarification_answer" and not action.get("limit_reached")
        ]
        ask_index, ask = asks[0] if asks else (None, None)
        candidate_ids = set(task["clarification"]["candidate_ids"])
        candidate_commitments = [
            (index, action, _candidate_targets(action, candidate_ids))
            for index, action in indexed_actions
            if action.get("type") in CANDIDATE_COMMITTING_ACTIONS
            and (
                action.get("type") == "submit"
                or _candidate_targets(action, candidate_ids)
            )
        ]
        first_commitment_index = (
            candidate_commitments[0][0] if candidate_commitments else None
        )
        timely_question = bool(
            ask_index is not None
            and (first_commitment_index is None or ask_index < first_commitment_index)
        )
        hidden_target = single_entity_target(task, arm)
        candidate_references = [
            (index, action, _candidate_targets(action, candidate_ids))
            for index, action in indexed_actions
            if action.get("type") in ANSWER_USING_ACTIONS
            and _candidate_targets(action, candidate_ids)
        ]
        post_answer_actions = [
            (action, targets) for index, action, targets in candidate_references
            if ask_index is not None and index > ask_index
        ]
        valid_question = bool(
            ask is not None
            and ask.get("valid") is True
            and clarification_question_is_valid(
                task, str(ask.get("slot", "")), str(ask.get("question", ""))
            )
        )
        answer_delivered = bool(
            valid_question
            and ask is not None
            and ask.get("payload") == task["clarification"]["canonical_answer"]
        )
        answer_used = bool(
            answer_delivered and timely_question and post_answer_actions
            and post_answer_actions[0][0].get("type") in ANSWER_USING_ACTIONS
            and post_answer_actions[0][1] == {hidden_target}
        )
        question_attempts = [
            action for action in actions if action.get("type") == "clarification_answer"
        ]
        result.update({
            "asked": bool(asks),
            "question_attempts": len(question_attempts),
            "question_limit_exceeded": any(
                action.get("limit_reached") is True for action in question_attempts
            ),
            "valid_question": valid_question,
            "unnecessary_ask": arm == "oracle" and bool(asks),
            "timely_question": timely_question,
            "answer_delivered": answer_delivered,
            "answer_used": answer_used,
        })
        if arm == "interactive":
            result["policy_pass"] = bool(
                result["valid_question"]
                and result["timely_question"]
                and result["answer_used"]
            )
            if result["e2e_success"] is not None:
                result["e2e_success"] = bool(
                    result["e2e_success"] and result["policy_pass"]
                )
    return result
