"""Evidence-based process metrics and source-clustered controlled comparisons."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any, TypeGuard

from .final_state import evaluate_final_state, normalized_entity_counts
from .protocol import NATIVE_COMMITMENT_ACTIONS
from .timeline import reveals_final_target
from .truth import Truth


def _state(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    entities, fields = value.get("selected_entity_ids"), value.get("fields")
    if not isinstance(entities, list) or not all(
        isinstance(x, str) and x for x in entities
    ):
        return None
    if not isinstance(fields, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in fields.items()
    ):
        return None
    return {"selected_entity_ids": sorted(entities), "fields": dict(fields)}


def rule_baseline_scores(
    task: Mapping[str, Any], initial_state: Mapping[str, Any]
) -> dict[str, float | None]:
    """Score the start state and an unmodified, verified initial answer."""
    final_goal = task["goal"]
    scores: dict[str, float | None] = {
        "no_action": float(evaluate_final_state(final_goal, initial_state)),
        "keep_initial": None,
    }
    if task["track"] == "clarify" or (
        task["track"] == "graft" and not task["graft"].get("initial_goal")
    ):
        return scores
    initial_states = Truth.of(task).initial_acceptable_states
    if not initial_states:
        return scores
    scores["keep_initial"] = mean(
        evaluate_final_state(final_goal, state) for state in initial_states
    )
    return scores


def _step(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _frame_count(record: Mapping[str, Any]) -> int | None:
    context = record.get("initial_context")
    if not isinstance(context, Mapping) or not isinstance(
        context.get("observations"), list
    ):
        return None
    packets = list(context["observations"])
    packets.extend(
        action.get("payload")
        for action in record.get("actions", [])
        if action.get("type") == "visual_context_update"
    )
    count = 0
    for packet in packets:
        if not isinstance(packet, Mapping) or not isinstance(
            packet.get("frame_paths"), list
        ):
            return None
        if not all(isinstance(path, str) and path for path in packet["frame_paths"]):
            return None
        count += len(packet["frame_paths"])
    return count


@dataclass
class _StateHistory:
    complete: bool
    release_step: int | None
    before_update: dict[str, Any] | None
    points: list[tuple[int, dict[str, Any]]]


def _state_history(record: Mapping[str, Any]) -> _StateHistory:
    current = _state(record.get("initial_state"))
    condition = record.get("condition", {})
    release = 0 if condition.get("schedule") == "upfront" else None
    history = _StateHistory(current is not None, release, current, [])
    if release == 0 and current is not None:
        history.points.append((0, current))
    last_step = 0
    for action in record.get("actions", []):
        step = action.get("step")
        if not _step(step) or step < last_step:
            history.complete = False
            continue
        last_step = step
        if reveals_final_target(action):
            history.release_step = step
            history.before_update = current if history.complete else None
            history.points = [(step, current)] if current is not None else []
        before, after = (
            _state(action.get("state_before")),
            _state(action.get("state_after")),
        )
        has_snapshot = "state_before" in action or "state_after" in action
        if has_snapshot:
            if before is None or after is None or before != current:
                history.complete = False
            current = after
            if after is not None and history.release_step is not None:
                history.points.append((step, after))
        elif action.get("type") in NATIVE_COMMITMENT_ACTIONS:
            # Old traces and unrecorded no-ops cannot establish exact repair timing.
            history.complete = False
    ending, ending_step = _state(record.get("live_state")), record.get("steps")
    if (
        ending is None
        or current != ending
        or not _step(ending_step)
        or ending_step < last_step
    ):
        history.complete = False
    elif history.release_step is not None:
        history.points.append((ending_step, ending))
    return history


def _state_change_errors(
    task: Mapping[str, Any],
    before: Mapping[str, Any] | None,
    ending: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "obsolete_residual_count": None,
        "valid_item_loss_count": None,
        "obsolete_residual_opportunity": None,
        "valid_item_loss_opportunity": None,
    }
    if "graft" not in task and "refresh" not in task:
        return {**result, "state_change_status": "not_applicable"}
    if before is None or ending is None:
        return {**result, "state_change_status": "missing_state_evidence"}
    goal = task["goal"]
    solved = evaluate_final_state(goal, ending)
    if solved:
        result.update(obsolete_residual_count=0, valid_item_loss_count=0)
    track = task["track"]
    if (
        track == "graft" and "initial_goal" not in task["graft"]
        or track == "refresh" and "initial_acceptable_states" not in task.get("truth", {})
    ):
        return {**result, "state_change_status": "missing_initial_goal"}
    truth = Truth.of(task)
    if truth.initial_acceptable_states is None:
        return {**result, "state_change_status": "missing_initial_goal"}
    initial_goal = truth.initial_goal or goal
    initial_states = [_state(value) for value in truth.initial_acceptable_states]
    final_states = [_state(value) for value in truth.final_acceptable_states]
    if (
        not initial_states
        or not final_states
        or any(value is None for value in initial_states + final_states)
    ):
        return {**result, "state_change_status": "invalid_goal"}
    held, final = (
        normalized_entity_counts(goal, before["selected_entity_ids"]),
        normalized_entity_counts(goal, ending["selected_entity_ids"]),
    )
    initial_held = normalized_entity_counts(initial_goal, before["selected_entity_ids"])
    if evaluate_final_state(initial_goal, before):
        valid_holdings = [held]
    elif initial_held != held:
        return {**result, "state_change_status": "ambiguous_package_equivalences"}
    else:
        valid_holdings = [
            held & normalized_entity_counts(initial_goal, initial["selected_entity_ids"])
            for initial in initial_states if initial is not None
        ]
    if not held:
        return {
            "obsolete_residual_count": 0,
            "valid_item_loss_count": 0,
            "obsolete_residual_opportunity": False,
            "valid_item_loss_opportunity": False,
            "state_change_status": "known",
        }
    unique_holdings = {
        frozenset(counts.items()): counts for counts in valid_holdings
    }
    expected_states = [
        normalized_entity_counts(goal, acceptable["selected_entity_ids"])
        for acceptable in final_states if acceptable is not None
    ]
    obsolete, lost = set(), set()
    remove_possible, keep_possible = set(), set()
    for valid_held in unique_holdings.values():
        for expected in expected_states:
            remove, keep = valid_held - expected, valid_held & expected
            remove_possible.add(bool(remove))
            keep_possible.add(bool(keep))
            if not solved:
                obsolete.add(frozenset((remove & final).items()))
                lost.add(frozenset((keep - final).items()))
    for key, alternatives in (
        ("obsolete_residual_count", obsolete),
        ("valid_item_loss_count", lost),
    ):
        if not solved and len(alternatives) == 1:
            result[key] = sum(count for _, count in alternatives.pop())
    for key, possible in (
        ("obsolete_residual_opportunity", remove_possible),
        ("valid_item_loss_opportunity", keep_possible),
    ):
        if len(possible) == 1:
            result[key] = possible.pop()
    status = (
        "known"
        if all(
            result[key] is not None
            for key in ("obsolete_residual_count", "valid_item_loss_count")
        )
        else "ambiguous_acceptable_states"
    )
    return {**result, "state_change_status": status}


def _retain_selection_regression(
    task: Mapping[str, Any], history: _StateHistory
) -> tuple[str, bool | None]:
    """Did a previously correct selection become invalid after the change?"""
    before = history.before_update
    if history.release_step is None:
        return "update_not_delivered", None
    if not history.complete or before is None:
        return "incomplete_state_history", None
    goal = task["goal"]
    if not before["selected_entity_ids"] or not evaluate_final_state(goal, before):
        return "no_correct_prior_selection", None
    held = normalized_entity_counts(goal, before["selected_entity_ids"])
    regressed = any(
        bool(held - normalized_entity_counts(goal, state["selected_entity_ids"]))
        and not evaluate_final_state(goal, state)
        for _, state in history.points
    )
    return "scorable", regressed


def controlled_run_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Measure delivered frames and complete state repair from recorded snapshots.

    Timing starts at the last actually delivered update (or zero for upfront).
    Missing state transitions never fall back to inferred actions or submit state.
    Selection-loss metrics concern entities and quantities, not goal-field errors.
    """
    result: dict[str, Any] = {
        "delivered_frame_count": _frame_count(record),
        "release_step": None,
        "first_full_recovery_step": None,
        "recovery_steps_after_release": None,
        "regressed_after_recovery": None,
        "obsolete_residual_count": None,
        "valid_item_loss_count": None,
        "obsolete_residual_opportunity": None,
        "valid_item_loss_opportunity": None,
        "selection_held_at_update": None,
        "state_change_status": "missing_task_snapshot",
        "recovery_status": "missing_task_snapshot",
    }
    task = record.get("task_snapshot")
    if not isinstance(task, Mapping) or not isinstance(task.get("goal"), Mapping):
        return result
    valid = record.get("run_status", {}).get("valid")
    if valid is not True:
        result.update(recovery_status="invalid_run", state_change_status="invalid_run")
        return result
    if not isinstance(record.get("actions"), list):
        result.update(
            recovery_status="missing_action_history",
            state_change_status="missing_action_history",
        )
        return result
    history = _state_history(record)
    result["release_step"] = history.release_step
    if task.get("refresh", {}).get("policy") == "retain":
        status, regressed = _retain_selection_regression(task, history)
        result.update(
            retain_selection_regression=regressed,
            retain_selection_regression_status=status,
        )
    if history.release_step is None:
        result.update(
            recovery_status="update_not_delivered",
            state_change_status="update_not_delivered",
        )
        return result
    if history.release_step == 0:
        result["state_change_status"] = "no_in_run_update"
    else:
        if history.before_update is not None:
            result["selection_held_at_update"] = bool(
                history.before_update["selected_entity_ids"]
            )
        result.update(
            _state_change_errors(
                task, history.before_update, _state(record.get("live_state"))
            )
        )
    if not history.complete:
        result["recovery_status"] = "incomplete_state_history"
        return result
    matches = [evaluate_final_state(task["goal"], state) for _, state in history.points]
    if not any(matches):
        result["recovery_status"] = "not_recovered"
        result["regressed_after_recovery"] = False
        return result
    first = matches.index(True)
    step = history.points[first][0]
    result.update(
        recovery_status="recovered",
        first_full_recovery_step=step,
        recovery_steps_after_release=step - history.release_step,
        regressed_after_recovery=not all(matches[first:]),
    )
    return result


def trajectory_mechanism(record: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the observed challenge, independently of final-state scoring.

    Only historical task goals and complete recorded state transitions establish
    whether an update required changing a selection. Missing evidence stays unknown.
    """
    result: dict[str, Any] = {
        "status": "unknown",
        "reason": "missing_task_snapshot",
        "update_delivered": None,
        "answer_delivered": None,
        "initial_goal_satisfied": None,
        "final_goal_satisfied_before_update": None,
        "native_state_changed_after_update": None,
    }
    task = record.get("task_snapshot")
    if not isinstance(task, Mapping) or not isinstance(task.get("goal"), Mapping):
        return result
    if record.get("run_status", {}).get("valid") is not True:
        return {**result, "reason": "invalid_run"}
    if not isinstance(record.get("actions"), list):
        return {**result, "reason": "missing_actions"}
    track = task["track"]
    if track == "clarify":
        answer = task["clarification"]["canonical_answer"]
        delivered = any(
            action.get("type") == "clarification_answer"
            and action.get("valid") is True
            and action.get("payload") == answer
            for action in record.get("actions", [])
        )
        upfront = record.get("condition", {}).get("schedule") == "upfront"
        if upfront:
            provided = record.get("initial_context", {}).get("instruction")
            if provided != task["instructions"]["oracle"]:
                return {**result, "reason": "missing_initial_information"}
        return {
            **result,
            "status": "known",
            "reason": "upfront_information"
            if upfront
            else ("answer_delivered" if delivered else "answer_not_delivered"),
            "update_delivered": delivered or upfront,
            "answer_delivered": delivered,
        }
    history = _state_history(record)
    result["update_delivered"] = history.release_step is not None
    if history.release_step is None:
        return {**result, "status": "known", "reason": "update_not_delivered"}
    before = history.before_update
    if before is None or not history.complete:
        return {**result, "reason": "incomplete_state_history"}
    initial_goal = (
        task.get("graft", {}).get("initial_goal")
        if track == "graft"
        else {
            "acceptable_states": task.get("truth", {}).get("initial_acceptable_states")
        }
    )
    if isinstance(initial_goal, Mapping) and initial_goal.get("acceptable_states"):
        result["initial_goal_satisfied"] = evaluate_final_state(initial_goal, before)
    result["final_goal_satisfied_before_update"] = evaluate_final_state(
        task["goal"], before
    )
    actions = record.get("actions", [])
    release_indexes = [
        index for index, action in enumerate(actions) if reveals_final_target(action)
    ]
    after_update = actions[release_indexes[-1] + 1 :] if release_indexes else actions
    result["native_state_changed_after_update"] = any(
        action.get("type") in NATIVE_COMMITMENT_ACTIONS
        and _state(action.get("state_before")) != _state(action.get("state_after"))
        for action in after_update
    )
    if result["final_goal_satisfied_before_update"]:
        reason = (
            "final_goal_already_satisfied"
            if result["native_state_changed_after_update"]
            else "valid_selection_retained"
        )
    elif result["initial_goal_satisfied"] is False:
        reason = "initial_goal_not_satisfied"
    elif result["initial_goal_satisfied"] is True:
        reason = "adjustment_required"
    else:
        return {**result, "reason": "missing_initial_goal"}
    return {**result, "status": "known", "reason": reason}


@dataclass(frozen=True)
class PairedComparison:
    left: str
    right: str
    validated_source_groups: Mapping[str, str]
    bootstrap_samples: int = 2000
    seed: int = 0
    required_conditions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.left or not self.right or self.left == self.right:
            raise ValueError("comparison requires two distinct named conditions")
        if not isinstance(self.bootstrap_samples, int) or self.bootstrap_samples < 1:
            raise ValueError("bootstrap_samples must be positive")
        if not self.validated_source_groups or any(
            not isinstance(key, str)
            or not key
            or not isinstance(value, str)
            or not value
            for key, value in self.validated_source_groups.items()
        ):
            raise ValueError("validated task-to-source groups are required")


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    offset = (len(ordered) - 1) * quantile
    lower = int(offset)
    fraction = offset - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    )


def source_cluster_statistics(
    clusters: Sequence[Sequence[float]],
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> tuple[float | None, list[float] | None]:
    """Estimate the mean of values, resampling each recording as a whole."""
    if not clusters:
        return None, None
    totals = [(sum(cluster), len(cluster)) for cluster in clusters]
    estimate = sum(total for total, _ in totals) / sum(count for _, count in totals)
    # One source supports a point estimate, not between-source uncertainty.
    if len(clusters) == 1:
        return estimate, None
    rng = random.Random(seed)
    estimates = []
    for _ in range(bootstrap_samples):
        sample = [totals[rng.randrange(len(totals))] for _ in totals]
        estimates.append(sum(total for total, _ in sample) / sum(count for _, count in sample))
    return estimate, [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def paired_source_statistics(
    rows: Sequence[Mapping[str, Any]],
    comparison: PairedComparison,
) -> dict[str, Any]:
    """Average replicates within task pairs, then bootstrap independent sources.

    Rows use task_id, condition.name, replicate_id, run_valid and e2e_success.
    Optional pair_id links alternate task IDs within one validated source.
    Joint success requires both conditions to succeed in the same replicate.
    """
    observations: dict[tuple[str, str], dict[str, dict[Any, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    seen: set[tuple[tuple[str, str], str, Any]] = set()
    attempted_replicates: dict[tuple[str, str], set[Any]] = defaultdict(set)
    attempted_tasks = set()
    invalid = 0
    unscorable = 0
    compared = {comparison.left, comparison.right}
    required = compared.union(comparison.required_conditions)
    attempted_runs = 0
    for row in rows:
        condition = row.get("condition", {})
        arm = condition.get("name") if isinstance(condition, Mapping) else condition
        if arm not in required:
            continue
        task_id = row["task_id"]
        if task_id not in comparison.validated_source_groups:
            raise ValueError(f"unvalidated source group for task {task_id}")
        key = (comparison.validated_source_groups[task_id], row.get("pair_id", task_id))
        replicate = row["replicate_id"]
        identity = (key, arm, replicate)
        if identity in seen:
            raise ValueError(f"duplicate task/condition/replicate: {identity}")
        seen.add(identity)
        if arm in compared:
            attempted_tasks.add(task_id)
            attempted_runs += 1
            attempted_replicates[key].add(replicate)
        if row.get("run_valid") is not True:
            invalid += arm in compared
            continue
        if not isinstance(row.get("e2e_success"), bool):
            unscorable += arm in compared
            continue
        observations[key][arm][replicate] = float(row["e2e_success"])
    by_source: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    # Keys read left/right, e.g. "S/F" means only the left condition succeeded.
    quadrants = {"S/S": 0, "S/F": 0, "F/S": 0, "F/F": 0}
    complete_pairs = 0
    paired_replicates = 0
    for (source, _), arms in observations.items():
        if not arms.get(comparison.left) or not arms.get(comparison.right):
            continue
        left, right = arms[comparison.left], arms[comparison.right]
        matching = sorted(
            index for index in left.keys() & right.keys()
            if all(index in arms[arm] for arm in required)
        )
        if not matching:
            continue
        by_source[source].append((
            mean(left[index] for index in matching),
            mean(right[index] for index in matching),
            mean(left[index] * right[index] for index in matching),
        ))
        paired_replicates += len(matching)
        for index in matching:
            quadrants[f"{_outcome(left[index])}/{_outcome(right[index])}"] += 1
        complete_pairs += 1
    clusters = [
        tuple(mean(column) for column in zip(*by_source[source], strict=True))
        for source in sorted(by_source)
    ]
    def estimate(values: list[float]) -> tuple[float | None, list[float] | None]:
        return source_cluster_statistics(
            [(value,) for value in values], comparison.bootstrap_samples, comparison.seed
        )

    left_success, left_ci = estimate([row[0] for row in clusters])
    right_success, right_ci = estimate([row[1] for row in clusters])
    _, difference_ci = estimate([row[1] - row[0] for row in clusters])
    joint, joint_ci = estimate([row[2] for row in clusters])
    expected_sources = len(set(comparison.validated_source_groups.values()))
    expected_tasks = len(comparison.validated_source_groups)
    left_successes = quadrants["S/S"] + quadrants["S/F"]
    return {
        "left": comparison.left,
        "right": comparison.right,
        "difference_direction": "right_minus_left",
        "joint_success_definition": "same_replicate_both_succeed_then_pair_and_source_mean",
        "coverage": {
            "expected_sources": expected_sources,
            "paired_sources": len(clusters),
            "source_fraction": len(clusters) / expected_sources,
            "expected_tasks": expected_tasks,
            "attempted_tasks": len(attempted_tasks),
            "missing_tasks": expected_tasks - len(attempted_tasks),
            "attempted_pairs": len(attempted_replicates),
            "paired_pairs": complete_pairs,
            "unpaired_pairs": len(attempted_replicates) - complete_pairs,
            "joint_paired_sources": len(clusters),
            "joint_paired_pairs": complete_pairs,
            "paired_replicates": paired_replicates,
            "unpaired_replicates": sum(map(len, attempted_replicates.values()))
            - paired_replicates,
            "attempted_runs": attempted_runs,
            "valid_runs": attempted_runs - invalid,
            "invalid_runs": invalid,
            "unscorable_runs": unscorable,
        },
        "left_success": left_success,
        "left_success_ci95": left_ci,
        "right_success": right_success,
        "right_success_ci95": right_ci,
        "difference": (
            right_success - left_success
            if left_success is not None and right_success is not None else None
        ),
        "difference_ci95": difference_ci,
        "joint_success": joint,
        "joint_success_ci95": joint_ci,
        # InterventionBench's Adaptability when left is the reference condition.
        "adaptability": (
            quadrants["S/S"] / left_successes if left_successes else None
        ),
        "adaptability_denominator": left_successes,
        "outcome_quadrants": quadrants,
    }


def _outcome(success: float) -> str:
    return "S" if success else "F"
