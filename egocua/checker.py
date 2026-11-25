from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from .controlled_metrics import controlled_run_metrics
from .messages import INITIAL_STEP, initial_context, release_label_step
from .final_state import evaluate_final_state, goal_entity_ids, normalized_entity_counts
from .protocol import (
    NATIVE_COMMITMENT_ACTIONS,
    RELEASE_DELIVERY_DESCRIPTIONS,
    SELECTION_ACTIONS,
    first_native_commitment_step,
    first_release_trigger,
    STAGE_RELEASE_TRIGGERS,
)
from .run_ledger import validate_run_record_shape
from .schema import (
    acceptable_states_of,
    clarification_basis,
    goal_for_condition,
    task_subtype,
)
from .timeline import (
    COMPOSITE_TRACK,
    SCHEDULED_EVENT_TYPES,
    CompositeInput,
    Condition,
    Timeline,
    TimelineError,
    composite_input,
    is_update_event,
    reveals_final_target,
)
from .truth import Truth

EVIDENCE_PASS = "PASS"
EVIDENCE_FAIL = "FAIL"
EVIDENCE_UNKNOWN = "UNKNOWN"
EVALUATOR_VERSION = "3.3"

SEARCH_ACTIONS = {"search"}
OPEN_ACTIONS = {
    "open_entity",
    "open_product",
    "open_document",
    "open_video",
    "open_place",
}
CANDIDATE_COMMITTING_ACTIONS = SELECTION_ACTIONS | {"submit"}
ANSWER_USING_ACTIONS = OPEN_ACTIONS | SELECTION_ACTIONS | {"submit"}


def behavioral_symptoms(checked: Mapping[str, Any]) -> list[str]:
    """Summarize observable outcomes without claiming a causal stage."""

    symptoms: list[str] = []
    if checked.get("track") == "clarify" and checked.get("asked") is False:
        symptoms.append("clarification_not_requested")
    if checked.get("premature_commitment") is True:
        symptoms.append("premature_commitment")
    if checked.get("premature_submission") is True:
        symptoms.append("submitted_before_update")
    if checked.get("stale_commitment") is True or checked.get("stale_final") is True:
        symptoms.append("stale_after_update")
    if (
        checked.get("false_update") is True
        and checked.get("goal_state_correct") is False
    ):
        symptoms.append("incorrect_change_after_update")
    if checked.get("goal_state_correct") is False:
        symptoms.append("wrong_final_state")
    return symptoms


def behavioral_failure_mode(checked: Mapping[str, Any]) -> str:
    """Return the primary observable failure symptom for operational reports."""

    if checked.get("run_valid") is not True or checked.get("e2e_success") is None:
        return "invalid_run"
    if checked.get("e2e_success") is True:
        return "success"
    priority = (
        "clarification_not_requested",
        "stale_after_update",
        "submitted_before_update",
        "incorrect_change_after_update",
        "wrong_final_state",
    )
    symptoms = set(behavioral_symptoms(checked))
    return next(
        (name for name in priority if name in symptoms), "other_checked_failure"
    )


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


def acceptable_states(
    task: dict[str, Any],
    condition: Condition,
) -> list[dict[str, Any]]:
    return acceptable_states_of(
        goal_for_condition(task, condition), f"{task['id']}/{condition.name}"
    )


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
        for action in actions
        if action.get("type") in OPEN_ACTIONS
        for entity_id in _action_entity_ids(action)
    ]
    mutations = [
        action for action in actions if action.get("type") in SELECTION_ACTIONS
    ]
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


def _submission_status(final_state: Any) -> bool | None:
    if not isinstance(final_state, dict):
        return None
    value = final_state.get("submission_completed")
    return value if isinstance(value, bool) else None


def _default_release_step(record: dict[str, Any]) -> int:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise TimelineError("run record meta must be an object")
    value = meta.get("default_release_step")
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TimelineError(
            "run record meta.default_release_step must be a positive integer"
        )
    return value


def _condition_from_record(record: dict[str, Any]) -> Condition:
    return Condition.from_dict(record.get("condition"))


def _reference_steps(record: dict[str, Any]) -> int | None:
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise TimelineError("run record meta must be an object")
    if "reference_steps" not in meta:
        raise TimelineError("run record meta.reference_steps is required")
    value = meta.get("reference_steps")
    if value is not None and (
        not isinstance(value, int) or isinstance(value, bool) or value < 1
    ):
        raise TimelineError(
            "run record meta.reference_steps must be null or a positive integer"
        )
    return value


def _release_trigger(record: dict[str, Any]) -> str:
    meta = record.get("meta")
    runtime = meta.get("runtime") if isinstance(meta, dict) else None
    value = runtime.get("release_trigger") if isinstance(runtime, dict) else None
    if value is None:
        return "timeline_step"
    if not isinstance(value, str) or value not in RELEASE_DELIVERY_DESCRIPTIONS:
        raise TimelineError(f"unknown release_trigger {value!r}")
    return value


def _timeline_from_record(
    task: dict[str, Any],
    record: dict[str, Any],
    condition: Condition,
) -> Timeline:
    timeline = Timeline.compile(
        task,
        condition,
        default_release_step=_default_release_step(record),
        reference_steps=_reference_steps(record),
    )
    trace = record.get("trace")
    settings = trace.get("agent_settings") if isinstance(trace, Mapping) else None
    policy = (
        settings.get("system_prompt_policy") if isinstance(settings, Mapping) else None
    )
    if (
        task["track"] == "graft"
        and condition.schedule == "upfront"
        and condition.scope != "initial_only"
        and policy in {"wearer-session-v5", "browsergym-verbatim-input-v4"}
    ):
        # These archived protocols delivered bundled text; score their real input.
        timeline = replace(timeline, instruction=str(task["instructions"]["bundled"]))
    return timeline


def _invalid_reasons(
    record: dict[str, Any],
    actions: list[dict[str, Any]],
    timeline: Timeline,
    release_trigger: str,
) -> list[str]:
    reasons: list[str] = []
    run_status = record.get("run_status", {})
    if run_status.get("valid") is False:
        reasons.append(str(run_status.get("reason") or "declared_invalid"))
    initial_site_state = record["meta"]["runtime"].get("initial_site_state")
    if "initial_context" in record and record["initial_context"] != initial_context(
        timeline.initial(), initial_site_state
    ):
        reasons.append("invalid_initial_context")
    if initial_site_state is not None and record.get("initial_state") != {
        **initial_site_state,
        "submission_completed": False,
    }:
        reasons.append("invalid_initial_site_state")
    for action in actions:
        if action.get("type") != "clarification_answer" or "payload" not in action:
            continue
        if timeline.offer is None:
            reasons.append("unexpected_clarification_answer")
        elif not timeline.offer.recorded_answer_is_consistent(action):
            reasons.append("invalid_clarification_answer_payload")
    final_state = record.get("live_state")
    if not isinstance(final_state, dict):
        reasons.append("missing_live_state")
    else:
        if not isinstance(final_state.get("selected_entity_ids"), list):
            reasons.append("missing_selected_entity_ids")
        if not isinstance(final_state.get("fields"), dict):
            reasons.append("missing_fields")

    skipped_refs, clock_offset = _standby_skips(actions)
    woken = any(
        action.get("type") == "standby" and int(action["step"]) < record["steps"]
        for action in actions
    )
    expected_events = [
        release.as_event()
        for release in timeline.releases
        if release.ref not in skipped_refs and (woken or not release.on_standby)
    ]
    delivered_events = [
        action for action in actions if action.get("type") in SCHEDULED_EVENT_TYPES
    ]
    stage_trigger = first_release_trigger(actions, release_trigger)
    for event_type in sorted(SCHEDULED_EVENT_TYPES):
        expected = sorted(
            (event for event in expected_events if event["type"] == event_type),
            key=lambda event: int(event["step"]),
        )
        delivered = sorted(
            (event for event in delivered_events if event.get("type") == event_type),
            key=lambda event: int(event.get("step", 0)),
        )
        # A stream batch due at or after the final step had no delivery opportunity.
        expected = expected[: len(delivered)] + [
            event
            for event in expected[len(delivered) :]
            if event["trigger_reason"] != "frame_cadence"
            or int(event["step"]) < record["steps"] + clock_offset
        ]
        if not expected and delivered:
            reasons.append(f"unexpected_{event_type}")
            continue
        if len(delivered) != len(expected):
            # Delivery happens at the next Agent step. A terminal commitment has
            # no delivery opportunity and remains a measured model outcome.
            if (
                release_trigger
                in ({"first_native_commitment"} | STAGE_RELEASE_TRIGGERS)
                and not delivered
                and (
                    stage_trigger is None
                    or int(stage_trigger[1].get("step", 0)) >= record["steps"]
                )
            ):
                continue
            if len(expected) == 1:
                reasons.append(f"expected_one_{event_type}")
            elif expected:
                reasons.append(f"expected_{len(expected)}_{event_type}_events")
            continue
        for registered, actual in zip(expected, delivered, strict=True):
            if (
                "payload" in actual or release_trigger in STAGE_RELEASE_TRIGGERS
            ) and actual.get("payload") != registered["payload"]:
                reasons.append(f"invalid_{event_type}_payload")
            if actual.get("update") != registered.get("update"):
                reasons.append(f"invalid_{event_type}_update_mark")
            if actual.get("composite_event_id") != registered.get("composite_event_id"):
                reasons.append(f"invalid_{event_type}_composite_event_id")
            actual_index = actions.index(actual)
            prior_native_commitment = first_native_commitment_step(
                actions[:actual_index]
            )
            if actual.get("step", 0) < 1:
                reasons.append(f"{event_type}_before_agent_action")
            elif (
                release_trigger in STAGE_RELEASE_TRIGGERS
                and registered["trigger_reason"] != "frame_cadence"
            ):
                prior_trigger = first_release_trigger(
                    actions[:actual_index], release_trigger
                )
                if actual.get("trigger_reason") != release_trigger:
                    reasons.append(f"invalid_{event_type}_trigger_reason")
                elif prior_trigger is None:
                    reasons.append(f"{event_type}_without_stage_trigger")
                elif (
                    actual.get("step") != prior_trigger[1].get("step", 0) + 1
                    or actual.get("trigger_action_index") != prior_trigger[0]
                    or actual.get("trigger_action_step") != prior_trigger[1].get("step")
                ):
                    reasons.append(f"invalid_{event_type}_stage_delivery")
            elif actual.get("trigger_reason") == "first_native_commitment":
                if release_trigger != "first_native_commitment":
                    reasons.append(f"invalid_{event_type}_trigger_reason")
                elif prior_native_commitment is None:
                    reasons.append(f"{event_type}_before_native_commitment")
            elif actual.get("trigger_reason") != registered["trigger_reason"]:
                reasons.append(f"invalid_{event_type}_trigger_reason")
            elif actual.get("source_step", actual.get("step")) != registered["step"]:
                reasons.append(f"invalid_{event_type}_task_timeline_step")
    return reasons


def _sufficiency_step(
    timeline: Timeline,
    actions: list[dict[str, Any]],
    release_trigger: str,
) -> int | None:
    """Return the registered evidence-sufficiency boundary for this run."""

    registered_sufficiency = timeline.sufficiency_step(actions)
    if (
        release_trigger in ({"first_native_commitment"} | STAGE_RELEASE_TRIGGERS)
        and timeline.releases
        and registered_sufficiency is not None
        and registered_sufficiency != 0
    ):
        delivered = [
            int(action.get("step", 0)) for action in actions if is_update_event(action)
        ]
        if delivered:
            return min(delivered)
        trigger = first_release_trigger(actions, release_trigger)
        commitment_step = trigger[1].get("step") if trigger else None
        return commitment_step + 1 if commitment_step is not None else None

    return registered_sufficiency


def _state_signature(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, dict):
        return None
    selected = value.get("selected_entity_ids")
    fields = value.get("fields")
    if (
        not isinstance(selected, list)
        or not all(isinstance(item, str) and item for item in selected)
        or not isinstance(fields, dict)
        or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in fields.items()
        )
    ):
        return None
    return tuple(sorted(Counter(selected).items())), tuple(sorted(fields.items()))


def _standby_skips(actions: list[dict[str, Any]]) -> tuple[set[str], int]:
    """Stream releases a standby passed over, and the source-clock lead it left."""

    standbys = [action for action in actions if action.get("type") == "standby"]
    skipped = {ref for action in standbys for ref in action["skipped_refs"]}
    if not standbys:
        return skipped, 0
    last = standbys[-1]
    return skipped, int(last["resume_source_step"]) - (int(last["step"]) + 1)


def _false_update(
    actions: list[dict[str, Any]],
    observed_state: dict[str, Any] | None,
    goal: dict[str, Any],
) -> bool | None:
    """Detect an observed legal-to-illegal edit after an actual retained-scene release."""

    release_index = next(
        (
            index
            for index, action in enumerate(actions)
            if action.get("type") == "visual_context_update"
            and is_update_event(action)
        ),
        None,
    )
    terminal = _state_signature(observed_state)
    if release_index is None or terminal is None:
        return None

    previous = None
    for action in actions[:release_index]:
        if action.get("type") in NATIVE_COMMITMENT_ACTIONS:
            previous = _state_signature(action.get("state_after"))

    incomplete = False
    for action in actions[release_index + 1 :]:
        mutation = action.get("type") in NATIVE_COMMITMENT_ACTIONS
        if (
            not mutation
            and "state_before" not in action
            and "state_after" not in action
        ):
            continue
        before = _state_signature(action.get("state_before"))
        after = _state_signature(action.get("state_after"))
        if before is None or after is None:
            incomplete = True
            previous = None
            continue
        if previous is not None and before != previous:
            incomplete = True
        if before != after:
            if not mutation:
                incomplete = True
            elif evaluate_final_state(
                goal, action["state_before"]
            ) and not evaluate_final_state(goal, action["state_after"]):
                return True
        previous = after

    if incomplete or (previous is not None and previous != terminal):
        return None
    return False


def timing_diagnostics(
    task: dict[str, Any],
    condition: Condition,
    actions: list[dict[str, Any]],
    observed_state: dict[str, Any] | None,
    *,
    default_release_step: int,
    reference_steps: int | None,
    release_trigger: str = "timeline_step",
) -> dict[str, Any]:
    """Track-agnostic commitment-timing fields derived from Truth and the Timeline."""

    timeline = Timeline.compile(
        task,
        condition,
        default_release_step=default_release_step,
        reference_steps=reference_steps,
    )
    truth = Truth.of(task)
    final_ids = truth.final_referent_ids
    final_choices = [
        set(goal_entity_ids({**task["goal"], "acceptable_states": [state]}))
        for state in truth.final_acceptable_states
    ]
    t_star = _sufficiency_step(timeline, actions, release_trigger)
    observed_ids = (
        {
            value
            for value in observed_state["selected_entity_ids"]
            if isinstance(value, str)
        }
        if isinstance(observed_state, dict)
        and isinstance(observed_state.get("selected_entity_ids"), list)
        else None
    )

    selections = [
        action for action in actions if action.get("type") in SELECTION_ACTIONS
    ]
    commitments = [
        action
        for action in actions
        if action.get("type") in CANDIDATE_COMMITTING_ACTIONS
    ]
    first_commitment_step = int(commitments[0].get("step", 0)) if commitments else None
    submits = [action for action in actions if action.get("type") == "submit"]

    premature_entities: list[str] = []
    premature_commitment: bool | None = None
    premature_submission: bool | None = None
    if t_star is not None:
        # Graft changes the user's goal. Before B arrives, actions allowed by A are
        # valid even when B later retracts them; failing to remove them afterwards
        # is stale, not premature. Refresh and Clarify instead ask whether the agent
        # committed to a losing referent before the evidence became sufficient.
        valid_before_sufficiency = (
            truth.initial_referent_ids if truth.track == "graft" else final_ids
        )
        for action in selections:
            if action.get("type") in {"deselect_entity", "remove_from_cart"}:
                continue
            if int(action.get("step", 0)) >= t_star:
                continue
            premature_entities.extend(
                entity_id
                for entity_id in _action_entity_ids(action)
                if entity_id not in valid_before_sufficiency
            )
        premature_commitment = bool(premature_entities)
        premature_submission = any(int(a.get("step", 0)) < t_star for a in submits)
    premature_set = set(premature_entities)
    revised = None
    if premature_commitment and observed_ids is not None:
        deselected = {
            entity_id
            for action in actions
            if action.get("type") in {"deselect_entity", "remove_from_cart"}
            for entity_id in _action_entity_ids(action)
        }
        revised = (
            bool(premature_set)
            and premature_set <= deselected
            and not (premature_set & observed_ids)
        )
    invalidated = truth.invalidated_referent_ids
    stale_final = (
        bool(invalidated & observed_ids)
        if invalidated and observed_ids is not None
        else None
    )
    new_obligations = (
        [choice - truth.initial_referent_ids for choice in final_choices]
        if truth.initial_referent_ids is not None
        else []
    )
    missing_after_update = (
        not any(required <= observed_ids for required in new_obligations)
        if any(new_obligations)
        and condition.scope == "full"
        and observed_ids is not None
        else None
    )
    if task["goal"].get("unit_equivalences") and isinstance(observed_state, dict):
        actual = normalized_entity_counts(task["goal"], observed_state.get("selected_entity_ids", []))
        initial_counts = [
            normalized_entity_counts(truth.initial_goal, state["selected_entity_ids"])
            for state in truth.initial_acceptable_states or ()
        ] if truth.initial_goal is not None else []
        final_counts = [
            normalized_entity_counts(task["goal"], state["selected_entity_ids"])
            for state in truth.final_acceptable_states
        ]
        if initial_counts:
            stale_final = all(
                bool((actual - after) & before)
                for before in initial_counts for after in final_counts
            )
            missing_after_update = not any(
                all(actual[key] >= quantity for key, quantity in after.items()
                    if before[key] != quantity)
                for before in initial_counts for after in final_counts
            ) if condition.scope == "full" else None
    false_update = None
    if (
        truth.later_release_should_preserve_referent
        and condition.schedule == "online"
        and condition.scope == "full"
    ):
        false_update = _false_update(actions, observed_state, task["goal"])

    # Recovery after the first release (InterruptBench SR(k) / AgentChangeBench GSRT):
    # steps from the first delivered release to the first commitment on a final-goal
    # entity at or after it. None when nothing was released or the Agent never got there.
    release_steps = [
        int(action.get("step", 0))
        for action in actions
        if reveals_final_target(action)
    ]
    post_release_latency: int | None = None
    if release_steps:
        first_release = min(release_steps)
        on_target = [
            int(action.get("step", 0))
            for action in selections
            if action.get("type") not in {"deselect_entity", "remove_from_cart"}
            and int(action.get("step", 0)) >= first_release
            and set(_action_entity_ids(action)) & final_ids
        ]
        post_release_latency = min(on_target) - first_release if on_target else None
    waits = [int(a.get("step", 0)) for a in actions if a.get("type") == "wait"]
    productive = [
        int(a.get("step", 0))
        for a in actions
        if a.get("type") in SEARCH_ACTIONS | OPEN_ACTIONS
    ]
    return {
        "condition": condition.as_dict(),
        "sufficiency_step": t_star,
        "first_commitment_step": first_commitment_step,
        "premature_commitment": premature_commitment,
        "premature_to_tempting": (
            bool(premature_set & truth.tempting_referent_ids)
            if premature_commitment is not None
            else None
        ),
        "premature_submission": premature_submission,
        "revised": revised,
        "stale_final": stale_final,
        "missing_after_update": missing_after_update,
        "false_update": false_update,
        "final_in_tempting": (
            bool(truth.tempting_referent_ids & observed_ids)
            if truth.tempting_referent_ids and observed_ids is not None
            else None
        ),
        "release_delivered": bool(release_steps),
        "post_release_latency": post_release_latency,
        "waits_before_sufficiency": (
            sum(step < t_star for step in waits) if t_star is not None else None
        ),
        "waits_after_sufficiency": (
            sum(step >= t_star for step in waits) if t_star is not None else None
        ),
        "productive_actions_before_sufficiency": (
            sum(step < t_star for step in productive) if t_star is not None else None
        ),
    }


def check_run(task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Score the historical task snapshot when the run contains one."""
    if "task_snapshot" in record:
        snapshot = record["task_snapshot"]
        if not isinstance(snapshot, dict) or snapshot.get("id") != record["task_id"]:
            raise TimelineError("task snapshot does not belong to run")
        task = snapshot
    return _check_run(task, record)


def rescore_run(task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Score an explicitly registered replacement key without altering the archive.

    The study must first establish that every released input is unchanged.
    """
    if task.get("id") != record.get("task_id"):
        raise TimelineError("scoring task does not belong to run")
    return _check_run(task, record)


def _check_run(task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    try:
        validate_run_record_shape(record, require_complete=True)
    except ValueError as exc:
        raise TimelineError(str(exc)) from exc
    condition = _condition_from_record(record)
    timeline = _timeline_from_record(task, record, condition)
    goal = goal_for_condition(task, condition)
    states = acceptable_states(task, condition)
    acceptable_ids = goal_entity_ids(goal)
    final_state = record.get("live_state", {})
    final_state = final_state if isinstance(final_state, dict) else {}
    submission_completed = _submission_status(record.get("final_state", {}))
    goal_state = record.get("live_state")
    goal_state_correct = (
        evaluate_final_state(goal, goal_state) if isinstance(goal_state, dict) else None
    )
    actions = record.get("actions", [])
    actions = actions if isinstance(actions, list) else []
    release_trigger = _release_trigger(record)
    invalid_reasons = _invalid_reasons(record, actions, timeline, release_trigger)
    evidence_complete = isinstance(
        final_state.get("selected_entity_ids"), list
    ) and isinstance(final_state.get("fields"), dict)
    outcome_pass = goal_state_correct is True
    evidence_status = (
        (EVIDENCE_PASS if outcome_pass else EVIDENCE_FAIL)
        if evidence_complete
        else EVIDENCE_UNKNOWN
    )
    run_valid = not invalid_reasons

    result: dict[str, Any] = {
        "task_id": task["id"],
        "track": task["track"],
        "site": task["site"],
        "subtype": task_subtype(task),
        "clarification_basis": clarification_basis(task),
        "condition": condition.as_dict(),
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
            "predicate": "exact_terminal_live_entities_and_fields",
            "acceptable_states": states,
            "goal": goal,
            "observed_state": final_state,
        },
        "goal_state_correct": goal_state_correct,
        "submission_completed": submission_completed,
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
    dynamic_events = [action for action in actions if is_update_event(action)]
    result["dynamic_event_required"] = any(
        is_update_event(release.as_event()) for release in timeline.releases
    )
    result["event_trigger_reason"] = (
        dynamic_events[0].get("trigger_reason") if len(dynamic_events) == 1 else None
    )
    # A recorded no-op is not a commitment; historical traces may lack snapshots.
    commitment_step = first_native_commitment_step(
        action for action in actions if (
            (before := _state_signature(action.get("state_before"))) is None
            or before != _state_signature(action.get("state_after"))
        )
    )
    result["update_after_commitment"] = (
        None
        if not dynamic_events
        else commitment_step is not None
        and commitment_step < int(dynamic_events[0]["step"])
    )
    result["woken_from_standby"] = (
        None
        if not dynamic_events
        else any(
            action.get("type") == "standby"
            and int(action["step"]) < int(dynamic_events[0]["step"])
            for action in actions
        )
    )

    # --- clarification diagnostics (unified into all results) ---
    _clarify_diagnostics(task, condition, actions, result)
    if task["track"] == COMPOSITE_TRACK:
        result.update(_composite_delivery(task, timeline, actions, int(record["steps"])))

    # --- timing diagnostics ---
    diagnostics = timing_diagnostics(
        task,
        condition,
        actions,
        goal_state,
        default_release_step=_default_release_step(record),
        reference_steps=_reference_steps(record),
        release_trigger=release_trigger,
    )
    for key, value in diagnostics.items():
        result.setdefault(key, value)
    result["collateral_damage"] = _collateral_damage({**record, "task_snapshot": task})

    # --- unified policy_pass (§2.4): not premature and (asked ⇒ ask_before_commitment) ---
    premature = result.get("premature_commitment")
    asked = result.get("asked", False)
    ask_before = result.get("ask_before_commitment", True)
    if premature is None:
        result["policy_pass"] = True
    else:
        result["policy_pass"] = not premature and (not asked or ask_before)

    return result


def _collateral_damage(record: dict[str, Any]) -> bool | None:
    """Whether still-required valid content held before the update is gone at the end.

    None when the update was not delivered or the before/after states cannot decide it.
    """
    lost = controlled_run_metrics(record)["valid_item_loss_count"]
    return None if lost is None else lost > 0


def _composite_delivery(
    task: Mapping[str, Any], timeline: Timeline, actions: list[dict[str, Any]], last_step: int
) -> dict[str, Any]:
    definition = composite_input(task)
    if definition.explicit_events:
        details = _composite_event_details(timeline, definition, actions, last_step)
        output = {"composite_event_delivery": details}
        seen = composite_event_coverage(output)
        return {
            **output,
            "composite_delivery": {
                kind: all(seen[event_id] for event_id, row in details.items() if row["kind"] == kind)
                for kind in sorted(definition.mechanisms)
            },
        }
    delivered = {
        "R": any(
            event.get("type") == "visual_context_update" and is_update_event(event)
            for event in actions
        ),
        "G": any(
            event.get("type") == "instruction_injection" and is_update_event(event)
            for event in actions
        ),
        "C": any(
            event.get("type") == "clarification_answer"
            and event.get("valid") is True
            and int(event["step"]) < last_step
            for event in actions
        ),
    }
    return {
        "composite_delivery": {
            kind: delivered[kind] for kind in ("R", "G", "C") if kind in definition.mechanisms
        },
    }


def composite_event_coverage(checked: Mapping[str, Any]) -> dict[str, bool]:
    """Whether the model received each required event, including controlled previews."""

    return {
        event_id: detail["status"] in {"delivered", "provided_upfront"}
        for event_id, detail in checked.get("composite_event_delivery", {}).items()
    }


def _composite_event_details(
    timeline: Timeline, definition: CompositeInput, actions: list[dict[str, Any]], last_step: int
) -> dict[str, dict[str, Any]]:
    recorded = {
        event_id: action
        for action in actions
        if isinstance((event_id := action.get("composite_event_id")), str)
        and (is_update_event(action) or action.get("type") == "clarification_answer")
    }
    registered = {
        release.composite_event_id: release
        for release in timeline.releases if release.composite_event_id is not None
    }
    skipped, _ = _standby_skips(actions)
    wake_steps = {
        (int(action["step"]) + 1, int(action["resume_source_step"]))
        for action in actions if action.get("type") == "standby"
    }
    preview = timeline.condition.composite_upfront
    details = {}
    for update in definition.updates:
        event_id = update.event_id
        action = recorded.get(event_id)
        release = registered.get(event_id)
        upfront = preview in {"both", {"R": "refresh", "G": "graft"}[update.mechanism]}
        if upfront:
            status, step = "provided_upfront", INITIAL_STEP
        elif action is not None:
            status, step = "delivered", action["step"]
        else:
            status = "skipped_on_standby" if release and release.ref in skipped else "not_delivered"
            step = None
        details[event_id] = {
            "kind": update.mechanism,
            "status": status,
            "step": step,
            "source_seconds": update.onset_seconds,
            "source_step": release.step if release else None,
            "replay_step": action["step"] if action else None,
            "woken_from_standby": (
                update.mechanism == "G" and action is not None
                and (action.get("step"), action.get("source_step")) in wake_steps
            ),
        }
    event_id = definition.clarification_id
    if event_id is not None:
        answer = recorded.get(event_id)
        observed = answer is not None and answer.get("valid") is True and answer["step"] < last_step
        details[event_id] = {
            "kind": "C",
            "status": "delivered" if observed else "not_delivered",
            "step": release_label_step(answer) if observed else None,
        }
    return details


def _clarify_diagnostics(
    task: dict[str, Any],
    condition: Condition,
    actions: list[dict[str, Any]],
    result: dict[str, Any],
) -> None:
    """Add clarification-related fields for all tracks (mostly None for non-clarify)."""

    indexed_actions = list(enumerate(actions))
    asks = [
        (index, action)
        for index, action in indexed_actions
        if action.get("type") == "clarification_answer"
        and not action.get("limit_reached")
    ]
    question_attempts = [
        action for action in actions if action.get("type") == "clarification_answer"
    ]
    asked = bool(asks)
    result["asked"] = asked
    result["question_attempts"] = len(question_attempts)
    result["question_limit_exceeded"] = any(
        action.get("limit_reached") is True for action in question_attempts
    )

    if task["track"] != "clarify":
        # For non-clarify tracks, set fields to None/defaults
        result["valid_question"] = None
        result["timely_question"] = None
        result["ask_before_commitment"] = None
        composite_offer = task["track"] == COMPOSITE_TRACK and "clarification" in task
        result["answer_delivered"] = (
            any(ask.get("valid") is True for _, ask in asks) if composite_offer else None
        )
        result["answer_used"] = None
        result["over_ask"] = asked if condition.ask == "open" and not composite_offer else None
        result["ask_step"] = int(asks[0][1].get("step", 0)) if asks else None
        return

    ask_index, ask = asks[0] if asks else (None, None)
    candidate_ids = set(task["clarification"]["candidate_ids"])
    final_target_sets = {
        frozenset(state["selected_entity_ids"])
        for state in acceptable_states(task, condition)
    }
    final_target_ids = set().union(*final_target_sets)
    commitment_ids = candidate_ids | final_target_ids
    first_commitment_index = next(
        (
            index
            for index, action in indexed_actions
            if action.get("type") in CANDIDATE_COMMITTING_ACTIONS
        ),
        None,
    )
    timely_question = bool(
        ask_index is not None
        and (first_commitment_index is None or ask_index < first_commitment_index)
    )
    candidate_references = [
        (index, action, _candidate_targets(action, commitment_ids))
        for index, action in indexed_actions
        if action.get("type") in ANSWER_USING_ACTIONS
        and _candidate_targets(action, commitment_ids)
    ]
    post_answer_actions = [
        (action, targets)
        for index, action, targets in candidate_references
        if ask_index is not None and index > ask_index
    ]
    valid_question = bool(ask is not None and ask.get("valid") is True)
    answer_delivered = bool(
        valid_question
        and ask is not None
        and ask.get("payload") == task["clarification"]["canonical_answer"]
    )
    answer_used = bool(
        answer_delivered
        and timely_question
        and post_answer_actions
        and post_answer_actions[0][0].get("type") in ANSWER_USING_ACTIONS
        and frozenset(post_answer_actions[0][1]) in final_target_sets
    )

    is_upfront = condition.schedule == "upfront"
    over_ask = is_upfront and asked

    result.update(
        {
            "valid_question": valid_question,
            "timely_question": timely_question,
            "ask_before_commitment": timely_question,  # unified name
            "answer_delivered": answer_delivered,
            "answer_used": answer_used,
            "over_ask": over_ask,
            "ask_step": int(ask.get("step", 0)) if ask is not None else None,
        }
    )
