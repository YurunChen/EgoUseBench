"""Summarize controlled-experiment runs into condition and paired statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
from functools import partial
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any, cast

from sites.runtime import SiteState

from .checker import check_run, rescore_run
from .controlled_experiments import (
    CONTROLLED_VARIABLES,
    ControlledArm,
    evidence_visibility,
    execution_schedule,
    scoring_key,
)
from .controlled_metrics import (
    PairedComparison,
    controlled_run_metrics,
    paired_source_statistics,
    rule_baseline_scores,
    source_cluster_statistics,
    trajectory_mechanism,
)
from .experiment import protocol_mismatch
from .experiment_results import score_summary
from .execution import request_rejection_kind
from .interaction_settings import INTERACTION_SETTINGS, SETTING_ORDER
from .lineage import task_recording_group
from .run_materials import (
    execution_inputs as _execution_inputs,
    path_independent_inputs as _path_independent_inputs,
    same_public_site as _same_public_site,
)
from .timeline import COMPOSITE_TRACK, OFFLINE_EGO_USE, Condition, composite_input, is_update_event
from .update_count import UPDATE_COUNT


def _rate(values: list[bool | None]) -> dict[str, Any]:
    known = [value for value in values if value is not None]
    return {
        "count": sum(known),
        "denominator": len(known),
        "unknown": len(values) - len(known),
        "rate": sum(known) / len(known) if known else None,
    }


def _state_diagnostic_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for metric in ("obsolete_residual", "valid_item_loss"):
        opportunities = [row["process"][f"{metric}_opportunity"] for row in rows]
        counts = [row["process"][f"{metric}_count"] for row in rows]
        result[metric] = {
            **_rate([value > 0 if value is not None else None for value in counts]),
            "opportunities": _rate(opportunities),
            "when_applicable": _rate([
                count > 0 if count is not None else None
                for opportunity, count in zip(opportunities, counts, strict=True)
                if opportunity is True
            ]),
        }
    return result


def update_state_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe the state actually held at delivery, not a required action path."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not row["run_valid"] or row["track"] not in {"graft", "refresh"}:
            continue
        facts, process = row["trajectory"], row["process"]
        if process["release_step"] == 0:
            group = "upfront"
        elif facts["update_delivered"] is False:
            group = "undelivered"
        elif (facts["status"] != "known" or facts["update_delivered"] is not True
              or process["selection_held_at_update"] is None):
            group = "unknown"
        elif process["selection_held_at_update"] is False:
            group = "no_prior_selection"
        elif facts["final_goal_satisfied_before_update"] is True:
            group = "final_state_already_valid"
        elif facts["initial_goal_satisfied"] is True:
            group = "revision_required"
        elif facts["initial_goal_satisfied"] is False:
            group = "initial_goal_incomplete"
        else:
            group = "unknown"
        grouped[group].append(row)
    return {
        "inference": "descriptive_only",
        "groups": {
            group: {"runs": len(scoped),
                    "success": _rate([row["e2e_success"] for row in scoped]),
                    **_state_diagnostic_summary(scoped)}
            for group, scoped in sorted(grouped.items())
        },
    }


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["run_valid"]]
    summary: dict[str, Any] = {
        "observed": len(rows),
        "valid": len(valid),
        "invalid": len(rows) - len(valid),
        "success": _rate([row["e2e_success"] for row in valid]),
        "stage_release": dict(
            Counter(
                (row.get("stage_release") or {}).get("status", "not_applicable")
                for row in rows
            )
        ),
        "actual_frame_counts": [
            row["process"]["delivered_frame_count"] for row in rows
        ],
        "usage_records": [row["usage"] for row in rows],
        "request_inputs": [row["request_inputs"] for row in rows],
        "trajectory": _trajectory_summary(rows),
        "update_state": update_state_summary(rows),
        **_state_diagnostic_summary(valid),
    }
    summary["success_by_evidence_visibility"] = {
        label: _rate(
            [
                row["e2e_success"]
                for row in valid
                if row["key_evidence_visibility"] == label
            ]
        )
        for label in sorted({row["key_evidence_visibility"] for row in valid})
    }
    if rows[0]["variable"] == "arrival":
        modes = [_arrival_mode(row) for row in valid]
        summary["arrival"] = {
            "delivery_mode": dict(Counter(modes)),
            "undelivered": _rate([mode == "undelivered" for mode in modes]),
            "delivered_and_success": _rate([
                False if mode == "undelivered" else row["e2e_success"]
                for row, mode in zip(valid, modes, strict=True)
            ]),
        }
    if rows[0]["variable"] == "cue":
        summary["cue_by_subtype"] = {
            subtype: _rate([
                row["e2e_success"] for row in valid if row["subtype"] == subtype
            ])
            for subtype in ("replace", "retain")
        }
        retain = [row for row in valid if row["subtype"] == "retain"]
        summary["retain_selection_regression"] = _rate([
            row["process"].get("retain_selection_regression") for row in retain
        ])
        summary["retain_selection_regression_coverage"] = dict(Counter(
            row["process"]["retain_selection_regression_status"] for row in retain
        ))
    if rows[0]["variable"] == "retention":
        summary["history_replacement_reached"] = _rate([
            (row["request_inputs"] or {}).get("older_visual_history_seen")
            for row in valid
        ])
    if rows[0]["variable"] == "redundancy":
        summary["actual_unique_frame_counts"] = [
            row["process"]["delivered_unique_frame_count"] for row in rows
        ]
    return summary


def _arrival_mode(row: dict[str, Any]) -> str:
    woke = row["woken_from_standby"]
    if woke is None:
        return "undelivered"
    return "standby" if woke else "active"


def _request_inputs(record: dict[str, Any]) -> dict[str, Any] | None:
    trace = record.get("trace")
    if not isinstance(trace, dict) or "llm_requests" not in trace:
        return None
    requests = trace["llm_requests"]
    return {
        **(
            {"older_visual_history_seen": any(
                row["older_visual_batch_count"] > 0 for row in requests
            )}
            if requests and all("older_visual_batch_count" in row for row in requests)
            else {}
        ),
        "requests": len(requests),
        "successful_requests": sum(row["success"] for row in requests),
        "wearer_image_count": sum(row["wearer_image_count"] for row in requests),
        "website_screenshot_count": sum(
            row["website_screenshot_count"] for row in requests
        ),
        "requests_with_usage": sum(row["usage"] is not None for row in requests),
    }


def _run_costs(record: dict[str, Any]) -> dict[str, int | float | None]:
    trace = record.get("trace") or {}
    history = trace.get("history")
    requests = _request_inputs(record)
    usage = trace.get("usage") or {}
    complete_usage = requests is not None and (
        requests["requests"] == requests["requests_with_usage"]
    )
    return {
        "browser_operations": sum(
            1 for entry in history
            for action in (entry.get("model_output") or {}).get("action", [])
            if "browser_action" in action
            and action["browser_action"]["code"] != "wait"
        ) if history is not None and record.get("meta", {}).get("agent") == "browsergym" else None,
        "decision_steps": record.get("steps"),
        "model_requests": requests["requests"] if requests is not None else None,
        "wall_time_seconds": record.get("meta", {}).get("duration_seconds"),
        "input_tokens": usage.get("total_prompt_tokens") if complete_usage else None,
        "output_tokens": usage.get("total_completion_tokens") if complete_usage else None,
        "wearer_images": requests["wearer_image_count"] if requests is not None else None,
        "website_images": requests["website_screenshot_count"] if requests is not None else None,
    }


def _paired_costs(
    rows: list[dict[str, Any]], protocol: PairedComparison,
) -> dict[str, Any]:
    required = {protocol.left, protocol.right, *protocol.required_conditions}
    pairs: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        name = row["condition"]["name"]
        if name in required and row["run_valid"] and isinstance(row["e2e_success"], bool):
            key = (row["task_id"], row["replicate_id"])
            if name in pairs[key]:
                raise ValueError(f"duplicate cost observation: {key}, {name}")
            pairs[key][name] = row
    complete = {key: pair for key, pair in pairs.items() if required <= pair.keys()}
    result = {}
    for subset in ("all_valid", "both_successful"):
        selected = {
            key: pair for key, pair in complete.items()
            if subset == "all_valid" or all(pair[c]["e2e_success"] for c in (protocol.left, protocol.right))
        }
        metrics = {}
        names = sorted({name for pair in selected.values() for name in pair[protocol.left]["costs"]})
        for name in names:
            by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for (task_id, _), pair in selected.items():
                values = tuple(pair[c]["costs"][name] for c in (protocol.left, protocol.right))
                if all(value is not None for value in values):
                    by_task[task_id].append(values)
            by_source: dict[str, list[tuple[float, float]]] = defaultdict(list)
            for task_id, values in by_task.items():
                by_source[protocol.validated_source_groups[task_id]].append(tuple(map(mean, zip(*values, strict=True))))
            estimates = [tuple(map(mean, zip(*by_source[source], strict=True))) for source in sorted(by_source)]
            delta, interval = source_cluster_statistics(
                [(right - left,) for left, right in estimates], protocol.bootstrap_samples, protocol.seed
            )
            metrics[name] = {
                "paired_runs": sum(map(len, by_task.values())),
                "missing_pairs": len(selected) - sum(map(len, by_task.values())),
                "sources": len(by_source),
                "left_mean": mean(v[0] for v in estimates) if estimates else None,
                "right_mean": mean(v[1] for v in estimates) if estimates else None,
                "difference": delta, "difference_ci95": interval,
            }
        result[subset] = {"paired_runs": len(selected), "metrics": metrics}
    return result


def _condition_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    conditions = defaultdict(list)
    for row in rows:
        conditions[row["condition"]["name"]].append(row)
    return {name: _arm_summary(conditions[name]) for name in sorted(conditions)}


def _repetition_cell(
    task_ids: set[str], replicates: tuple[int, ...], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rounds = {}
    for replicate in replicates:
        observed = [row for row in rows if row["replicate_id"] == replicate]
        valid = [row for row in observed if row["run_valid"] and isinstance(row["e2e_success"], bool)]
        rounds[str(replicate)] = {
            "planned": len(task_ids), "valid": len(valid),
            "complete": len(valid) == len(task_ids) and bool(task_ids),
            "success": _rate([row["e2e_success"] for row in valid]),
        }
    complete = all(row["complete"] for row in rounds.values())
    rates = [row["success"]["rate"] for row in rounds.values()]
    return {
        "rounds": rounds, "complete": complete,
        "mean_success_rate": mean(rates) if complete else None,
        "round_standard_deviation": stdev(rates) if complete else None,
    }


def repetition_statistics(
    arms: list[ControlledArm], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Do not publish a partial extension as a completed repeated comparison."""
    result = {}
    for arm in arms:
        if len(arm.plan.replicate_ids) < 2:
            continue
        result[arm.name] = {}
        for condition in arm.plan.conditions:
            scoped = [row for row in rows if row["analysis_reference"]["arm"] == arm.name
                      and row["analysis_reference"]["condition"] == condition]
            groups = {"overall": {task["id"] for task in arm.tasks}}
            groups.update({track: {task["id"] for task in arm.tasks if task["track"] == track}
                           for track in sorted({task["track"] for task in arm.tasks})})
            groups.update({f"refresh_{policy}": {task["id"] for task in arm.tasks
                           if task["track"] == "refresh" and task["refresh"]["policy"] == policy}
                           for policy in ("replace", "retain")})
            result[arm.name][condition] = {
                group: _repetition_cell(ids, arm.plan.replicate_ids,
                                       [row for row in scoped if row["task_id"] in ids])
                for group, ids in groups.items() if ids
            }
    return result


def _analysis_mode(arm: ControlledArm) -> str:
    if arm.pair_id:
        return "paired"
    return (
        "within_task"
        if arm.variable
        in {"frequency", "timing", "redundancy", "visual", "cadence",
            INTERACTION_SETTINGS, *CONTROLLED_VARIABLES}
        else "observational"
    )


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["run_valid"]]
    reasons = sorted({row["trajectory"]["reason"] for row in valid})
    fields = (
        "update_delivered",
        "answer_delivered",
        "initial_goal_satisfied",
        "final_goal_satisfied_before_update",
        "native_state_changed_after_update",
    )
    return {
        "known": sum(row["trajectory"]["status"] == "known" for row in valid),
        "unknown": sum(row["trajectory"]["status"] == "unknown" for row in valid),
        "by_reason": {
            reason: _rate(
                [
                    row["e2e_success"]
                    for row in valid
                    if row["trajectory"]["reason"] == reason
                ]
            )
            for reason in reasons
        },
        "by_fact": {
            field: {
                label: _rate(
                    [
                        row["e2e_success"]
                        for row in valid
                        if row["trajectory"][field] is value
                    ]
                )
                for label, value in (
                    ("true", True),
                    ("false", False),
                    ("unknown", None),
                )
            }
            for field in fields
        },
    }


def _research_summary(rows: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    valid = [row for row in rows if row["run_valid"]]
    return {
        "planned": planned,
        "observed": len(rows),
        "valid": len(valid),
        "success": _rate([row["e2e_success"] for row in valid]),
        "trajectory": _trajectory_summary(rows),
    }


def _variable_summaries(
    arms: list[ControlledArm], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    planned: Counter[tuple[str, str]] = Counter()
    levels: Counter[tuple[str, str, str]] = Counter()
    for arm in arms:
        mode = _analysis_mode(arm)
        for scheduled in arm.schedule():
            planned[arm.variable, mode] += 1
            levels[arm.variable, mode, arm.level or scheduled["condition"]] += 1
    results = {}
    for variable in sorted({arm.variable for arm in arms}):
        scoped = [row for row in rows if row["variable"] == variable]
        modes = sorted(mode for var, mode in planned if var == variable)
        results[variable] = {
            **_research_summary(scoped, sum(planned[variable, mode] for mode in modes)),
            "analysis_modes": {
                mode: {
                    **_research_summary(
                        [row for row in scoped if row["analysis_mode"] == mode],
                        planned[variable, mode],
                    ),
                    "levels": {
                        level: _research_summary(
                            [
                                row
                                for row in scoped
                                if row["analysis_mode"] == mode
                                and row["level"] == level
                            ],
                            count,
                        )
                        for (var, analysis, level), count in sorted(levels.items())
                        if var == variable and analysis == mode
                    },
                }
                for mode in modes
            },
        }
    return results


def _grouped_summaries(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {name: _condition_summaries(groups[name]) for name in sorted(groups)}


def _success_summary(
    values: dict[str, float | None], source_groups: dict[str, str]
) -> dict[str, Any]:
    clusters: dict[str, list[float]] = defaultdict(list)
    for task_id, value in values.items():
        if value is not None:
            clusters[source_groups[task_id]].append(value)
    success, interval = source_cluster_statistics(
        [clusters[source] for source in sorted(clusters)]
    )
    return {
        "planned_tasks": len(values),
        "scored_tasks": sum(map(len, clusters.values())),
        "planned_sources": len({source_groups[task_id] for task_id in values}),
        "scored_sources": len(clusters),
        "success": success,
        "success_ci95": interval,
    }


def _formal_statistics(
    arm: ControlledArm, rows: list[dict[str, Any]], source_groups: dict[str, str]
) -> dict[str, Any]:
    tasks = {task["id"]: task for task in arm.tasks}
    observed = {row["task_id"]: row for row in rows if row["variable"] == "formal"}
    if len(observed) != sum(row["variable"] == "formal" for row in rows):
        raise ValueError("formal statistics expect one registered run per task")
    empty_state = SiteState({}).live_state()
    baseline_scores = {
        task_id: rule_baseline_scores(
            task, arm.plan.initial_site_states.get(task_id, empty_state)
        )
        for task_id, task in tasks.items()
    }
    buckets = {"overall": list(tasks)}
    for track in sorted({task["track"] for task in arm.tasks}):
        buckets[track] = [task_id for task_id in tasks if tasks[task_id]["track"] == track]
    main, baselines = {}, {}
    for label, task_ids in buckets.items():
        values: dict[str, float | None] = {}
        invalid, unscorable, unobserved = 0, 0, 0
        for task_id in task_ids:
            row = observed.get(task_id)
            values[task_id] = None
            if row is None:
                unobserved += 1
            elif not row["run_valid"]:
                invalid += 1
            elif row["e2e_success"] is None:
                unscorable += 1
            else:
                values[task_id] = float(row["e2e_success"])
        known = sum(value for value in values.values() if value is not None)
        main[label] = {
            **_success_summary(values, source_groups),
            "invalid_tasks": invalid,
            "unscorable_tasks": unscorable,
            "unobserved_tasks": unobserved,
            "success_bounds": [
                known / len(values),
                (known + invalid + unscorable + unobserved) / len(values),
            ],
        }
        baselines[label] = {}
        for baseline in ("no_action", "keep_initial"):
            values = {task_id: baseline_scores[task_id][baseline] for task_id in task_ids}
            baselines[label][baseline] = {
                **_success_summary(values, source_groups),
                "not_applicable_tasks": sum(value is None for value in values.values()),
            }
    return {"formal_success": main, "rule_baselines": baselines}


def _commitment_group(row: dict[str, Any]) -> str:
    delivered = row["trajectory"]["update_delivered"]
    if delivered is False:
        return "undelivered"
    if delivered is None or row["process"]["selection_held_at_update"] is None:
        return "unknown"
    after = row.get("update_after_commitment")
    if after is None:
        return "unknown"
    return "after_commitment" if after else "before_commitment"


def _commitment_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = ("before_commitment", "after_commitment", "undelivered", "unknown")
    result = {}
    for track in ("graft", "refresh"):
        scoped = [
            row for row in rows if row["variable"] == "formal"
            and row["level"] == "online" and row["track"] == track
        ]
        if not scoped:
            continue
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scoped:
            if row["run_valid"]:
                grouped[_commitment_group(row)].append(row)
        result[track] = {
            "invalid": sum(not row["run_valid"] for row in scoped),
            "groups": {
                group: {
                    "observed": len(grouped[group]),
                    "success": _rate([row["e2e_success"] for row in grouped[group]]),
                    "selection_held_at_update": _rate([
                        row["process"]["selection_held_at_update"] for row in grouped[group]
                    ]),
                    **_state_diagnostic_summary(grouped[group]),
                }
                for group in groups
            },
        }
    return {"inference": "descriptive_only", "by_track": result}


def _observational_summaries(
    rows: list[dict[str, Any]], evidence: dict[str, Any] | None
) -> list[dict[str, Any]]:
    results = []
    for group in (evidence or {}).get("analysis_groups", []):
        if group["status"] != "verified":
            continue
        levels = {}
        for level in sorted({member["level"] for member in group["members"]}):
            ids = {
                member["task_id"]
                for member in group["members"]
                if member["level"] == level
            }
            levels[level] = _condition_summaries(
                [row for row in rows if row["task_id"] in ids]
            )
        results.append(
            {
                "id": group["id"],
                "variable": group["variable"],
                "analysis_mode": "observational",
                "inference": "descriptive_only",
                "levels": levels,
            }
        )
    return results


def _interaction_pairs(
    arms: list[ControlledArm],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    by_setting = {arm.name: arm for arm in arms if arm.variable == INTERACTION_SETTINGS}
    if not by_setting:
        return []
    if by_setting.keys() != set(SETTING_ORDER):
        raise ValueError("interaction comparisons require all three settings")
    tasks = by_setting["online"].tasks
    task_ids = {task["id"] for task in tasks}
    if any({task["id"] for task in arm.tasks} != task_ids for arm in by_setting.values()):
        raise ValueError("interaction settings must contain the same task IDs")
    return [
        (f"{left}::{left}", f"{right}::{right}", tasks)
        for left, right in zip(SETTING_ORDER[:-1], SETTING_ORDER[1:], strict=True)
    ] if tasks else []


def _comparison_pairs(
    arms: list[ControlledArm],
) -> list[tuple[str, str, list[dict[str, Any]]]]:
    pairs = _interaction_pairs(arms)
    for arm in arms:
        if arm.variable in CONTROLLED_VARIABLES:
            if arm.variable == "cue":
                pairs.extend(
                    (f"cue::{left}", f"cue::{right}", arm.tasks)
                    for left, right in zip(tuple(arm.plan.conditions)[:-1], tuple(arm.plan.conditions)[1:], strict=True)
                )
                continue
            base = (
                "full" if arm.variable == "retention" else arm.plan.reference_condition_name
            )
            pairs.extend(
                (f"{arm.name}::{base}", f"{arm.name}::{condition}", arm.tasks)
                for condition in arm.plan.conditions if condition != base
            )
        elif arm.variable == "visual":
            base = arm.plan.reference_condition_name
            for condition in arm.plan.conditions:
                if condition == base:
                    continue
                eligible = [
                    task for task in arm.tasks
                    if {base, condition} <= set(arm.plan.conditions_for(task["track"]))
                ]
                if eligible:
                    pairs.append((f"{arm.name}::{condition}", f"{arm.name}::{base}", eligible))
        elif arm.variable in {"frequency", "cadence"} or arm.name == "redundancy":
            pairs.extend(
                (f"{arm.name}::{left}", f"{arm.name}::{right}", arm.tasks)
                for left, right in combinations(arm.plan.conditions, 2)
            )
    timing = [arm for arm in arms if arm.variable == "timing"]
    pairs.extend(
        (f"{left.name}::online", f"{right.name}::online", left.tasks)
        for left, right in combinations(timing, 2)
    )
    for variable in ("scope", "span", "dependency", UPDATE_COUNT):
        variants = [
            arm for arm in arms if arm.variable == variable and arm.pair_id is not None
        ]
        if variable == UPDATE_COUNT:
            strata = sorted({(arm.direction, arm.pair_id.split("-")[0]) for arm in variants})
            for direction, combination in strata:
                selected = [arm for arm in variants if arm.direction == direction and arm.pair_id.split("-")[0] == combination]
                by_pair = defaultdict(list)
                for arm in selected:
                    by_pair[arm.pair_id].append(arm)
                if any(len(pair) != 2 or {arm.level for arm in pair} != {"fewer", "more"} for pair in by_pair.values()):
                    raise ValueError("update-count requires exactly one fewer and one more arm per pair")
                pairs.append((
                    f"{UPDATE_COUNT}::fewer", f"{UPDATE_COUNT}::more",
                    [task for arm in selected for task in arm.tasks],
                ))
            continue
        levels = {cast(str, arm.level) for arm in variants}
        for left, right in combinations(sorted(levels), 2):
            paired_ids = {arm.pair_id for arm in variants if arm.level == left} & {
                arm.pair_id for arm in variants if arm.level == right
            }
            paired_tasks = [
                task
                for arm in variants
                if arm.pair_id in paired_ids and arm.level in {left, right}
                for task in arm.tasks
            ]
            if paired_tasks:
                pairs.append(
                    (f"{variable}::{left}", f"{variable}::{right}", paired_tasks)
                )
    return pairs


def _update_count_stability(
    rows: list[dict[str, Any]], protocol: PairedComparison, replicates: tuple[int, ...],
) -> dict[str, Any]:
    scoped = [row for row in rows if row["task_id"] in protocol.validated_source_groups
              and row["condition"]["name"] in {protocol.left, protocol.right}]
    rounds = {
        str(replicate): paired_source_statistics(
            [row for row in scoped if row["replicate_id"] == replicate], protocol,
        )
        for replicate in replicates
    }
    expected_tasks = len(protocol.validated_source_groups)
    complete = all(
        row["coverage"]["valid_runs"] == expected_tasks
        and row["coverage"]["paired_replicates"] * 2 == expected_tasks
        for row in rounds.values()
    )
    outcomes: dict[tuple[str, str], dict[int, bool]] = defaultdict(dict)
    for row in scoped:
        if row["run_valid"] and isinstance(row["e2e_success"], bool):
            outcomes[row["task_id"], row["condition"]["name"]][row["replicate_id"]] = row["e2e_success"]
    flips = {}
    for condition in (protocol.left, protocol.right):
        completed = [
            [values[replicate] for replicate in replicates]
            for (_, arm), values in outcomes.items()
            if arm == condition and set(replicates) <= values.keys()
        ]
        flips[condition] = {
            "planned_tasks": expected_tasks // 2, "complete_tasks": len(completed),
            "always_successful": sum(all(values) for values in completed),
            "always_failed": sum(not any(values) for values in completed),
            "changed_outcome": sum(len(set(values)) > 1 for values in completed),
        }
    differences = [row["difference"] for row in rounds.values()]
    return {
        "planned_replicates": list(replicates), "complete": complete,
        "independent_sources": len(set(protocol.validated_source_groups.values())),
        "rounds": rounds,
        "mean_difference": mean(differences) if complete else None,
        "round_difference_standard_deviation": stdev(differences) if complete else None,
        "round_difference_range": [min(differences), max(differences)] if complete else None,
        "outcome_stability": flips,
    }


def _comparison_results(
    arms: list[ControlledArm],
    rows: list[dict[str, Any]],
    sources: dict[str, str],
) -> list[dict[str, Any]]:
    condition_levels = {
        arm.name: tuple(f"{arm.name}::{name}" for name in arm.plan.conditions)
        for arm in arms if arm.variable in {"pace", "retention", "cue"}
    }
    interaction_keys = tuple(f"{name}::{name}" for name in SETTING_ORDER)
    results = []
    for left, right, tasks in _comparison_pairs(arms):
        family = left.split("::", 1)[0]
        is_interaction = left in interaction_keys
        protocol = PairedComparison(
            left, right, {task["id"]: sources[task["id"]] for task in tasks},
            required_conditions=(
                interaction_keys if is_interaction else condition_levels.get(family, ())
            ),
        )

        def summarize(
            selected: list[dict[str, Any]], protocol: PairedComparison = protocol
        ) -> dict[str, Any]:
            groups = {task["id"]: sources[task["id"]] for task in selected}
            estimate = paired_source_statistics(
                [row for row in rows if row["task_id"] in groups],
                replace(protocol, validated_source_groups=groups),
            )
            if (protocol.left in interaction_keys and (
                protocol.left != f"{OFFLINE_EGO_USE}::{OFFLINE_EGO_USE}"
                or any(task["track"] == "clarify" for task in selected)
            )) or (
                protocol.left.startswith("visual::")
                and (
                    protocol.left != "visual::upfront"
                    or any(task["track"] == "clarify" for task in selected)
                )
            ):
                del estimate["adaptability"], estimate["adaptability_denominator"]
            return estimate

        comparison = summarize(tasks)
        if family == UPDATE_COUNT:
            ids = {task["id"] for task in tasks}
            selected_arms = [arm for arm in arms if arm.variable == UPDATE_COUNT and any(task["id"] in ids for task in arm.tasks)]
            comparison.update(
                variable=UPDATE_COUNT,
                direction=selected_arms[0].direction,
                combination=selected_arms[0].pair_id.split("-")[0],
                model=selected_arms[0].plan.model,
                expected_pairs=len({arm.pair_id for arm in selected_arms}),
                discordant_pairs={
                    "more_only": comparison["outcome_quadrants"]["F/S"],
                    "fewer_only": comparison["outcome_quadrants"]["S/F"],
                },
            )
            replicates = selected_arms[0].plan.replicate_ids
            if len(replicates) > 1:
                if any(arm.plan.replicate_ids != replicates for arm in selected_arms):
                    raise ValueError("update-count pairs require the same registered repetitions")
                comparison["repetition_stability"] = _update_count_stability(rows, protocol, replicates)
                comparison["discordant_pair_unit"] = "task_pair_replicate"
            del comparison["adaptability"], comparison["adaptability_denominator"]
        if left == f"{OFFLINE_EGO_USE}::{OFFLINE_EGO_USE}":
            comparison["costs"] = _paired_costs(rows, protocol)
        if family in {"visual", *CONTROLLED_VARIABLES} or is_interaction:
            comparison["by_track"] = {
                track: summarize([task for task in tasks if task["track"] == track])
                for track in sorted({task["track"] for task in tasks})
            }
        if family in {"visual", *CONTROLLED_VARIABLES} or is_interaction:
            subtypes = {
                subtype: selected
                for subtype in ("replace", "retain")
                if (selected := [
                    task for task in tasks
                    if task["track"] == "refresh" and task["refresh"]["policy"] == subtype
                ])
            }
            if subtypes:
                comparison["by_subtype"] = {
                    subtype: summarize(selected) for subtype, selected in subtypes.items()
                }
        results.append(comparison)
    return results


def _referenced_records(
    arms: list[ControlledArm],
    records: list[tuple[str, dict[str, Any]]],
    execution_results: list[dict[str, Any]],
    schedule: list[dict[str, Any]],
) -> list[tuple[str, dict[str, Any], dict[str, Any], bool]]:
    by_experiment = {arm.plan.experiment_id: arm for arm in arms}
    references = {
        (
            reference["experiment_id"],
            reference["task_id"],
            reference["condition"],
            reference["replicate_id"],
        ): execution
        for execution in schedule
        for reference in execution["analysis_references"]
    }
    inherited = {
        outcome["run_id"]: outcome
        for outcome in execution_results
        if outcome.get("source_run") is not None and outcome.get("valid") is True
    }
    seen: set[int] = set()
    resolved: list[tuple[str, dict[str, Any], dict[str, Any], bool]] = []
    for experiment_id, record in records:
        outcome = inherited.get(record["meta"].get("run_id"))
        run = outcome["run"] if outcome else None
        identity = (
            run["experiment_id"] if run else experiment_id,
            run["task_id"] if run else record["task_id"],
            run["condition"] if run else record["condition"]["name"],
            run["replicate_id"] if run else record["meta"]["replicate_id"],
        )
        if identity not in references:
            raise ValueError(f"unregistered run: {identity}")
        execution = references[identity]
        if execution["execution_index"] in seen:
            raise ValueError(f"duplicate physical execution: {identity}")
        seen.add(execution["execution_index"])
        source_reference = next(
            reference
            for reference in execution["analysis_references"]
            if reference["experiment_id"] == identity[0]
            and reference["condition"] == identity[2]
        )
        arm = by_experiment[identity[0]]
        plan = arm.plan
        historical_name = record["condition"]["name"]
        registered_condition = replace(
            plan.condition(identity[2]), name=historical_name
        )
        record_plan = replace(plan, conditions={historical_name: registered_condition})
        mismatch = protocol_mismatch(
            record,
            record_plan,
            outcome["source_run"]["schedule_index"]
            if outcome
            else source_reference["schedule_index"],
        )
        if mismatch:
            raise ValueError(f"{identity}: {mismatch}")
        snapshot = record.get("task_snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("id") != identity[1]:
            raise ValueError(f"{identity}: missing historical task snapshot")
        current = next(task for task in arm.tasks if task["id"] == identity[1])
        rescore = (
            outcome is not None
            and outcome.get("scoring", {}).get("task_source") == "current_frozen_task"
        )
        old_inputs, old_frames = _path_independent_inputs(
            _execution_inputs(snapshot, record_plan, historical_name)
        )
        new_inputs, new_frames = _path_independent_inputs(
            _execution_inputs(current, plan, identity[2])
        )
        if old_inputs != new_inputs or (not outcome and old_frames != new_frames):
            raise ValueError(f"{identity}: task content differs from registered inputs")
        if not rescore and scoring_key(snapshot) != scoring_key(current):
            raise ValueError(
                f"{identity}: task content differs from registered inputs; key changes require explicit frozen-task rescoring"
            )
        if (
            outcome
            and record["meta"].get("provider_seed") != outcome["run"]["provider_seed"]
        ):
            raise ValueError(f"{identity}: wrong_provider_seed")
        if plan.site_snapshots and not _same_public_site(
            (snapshot, current),
            (
                record["meta"]["runtime"].get("site_snapshot"),
                plan.site_snapshot_for(current),
            ),
        ):
            raise ValueError(
                f"{identity}: public site content differs from registered inputs"
            )
        resolved.extend(
            (reference["experiment_id"], record, reference, rescore)
            for reference in execution["analysis_references"]
        )
    return resolved


def unresolved_request_failures(
    execution_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Read each unresolved execution once; an existing valid result takes priority."""
    completed = {
        row["execution_index"] for row in execution_results if row.get("valid") is True
    }
    latest = {row["execution_index"]: row for row in execution_results}
    return [
        {**row, "kind": request_rejection_kind(row["error"]) or "other"}
        for index, row in latest.items()
        if index not in completed and row.get("valid") is False and row.get("error")
    ]


def _request_failure_summary(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    failures = unresolved_request_failures(outcomes)
    return {
        kind: {
            "count": sum(row["kind"] == kind for row in failures),
            "executions": [row for row in failures if row["kind"] == kind],
        }
        for kind in ("moderation", "context_limit", "other")
    }


def _failed_execution_rows(
    arms: list[ControlledArm], schedule: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use the execution ledger for failures with no scorable archive."""
    schedule_by_index = {entry["execution_index"]: entry for entry in schedule}
    arm_by_name = {arm.name: arm for arm in arms}
    latest = {row["execution_index"]: row for row in execution_results}
    failed = []
    for index, outcome in latest.items():
        if outcome.get("valid") is not False:
            continue
        for reference in schedule_by_index[index]["analysis_references"]:
            arm = arm_by_name[reference["arm"]]
            condition = reference["condition"]
            label = (
                f"{arm.variable}::{arm.level}" if arm.pair_id
                else f"{arm.name}::{condition}"
            )
            failed.append({
                **reference,
                "condition": {"name": label},
                "variable": arm.variable,
                "level": arm.level or condition,
                "run_valid": False,
                "e2e_success": None,
                "analysis_reference": reference,
                **({"pair_id": arm.pair_id} if arm.pair_id else {}),
            })
    return failed



def _composite_run_facts(
    task: dict[str, Any], record: dict[str, Any], checked: dict[str, Any]
) -> dict[str, Any]:
    definition = composite_input(task)
    details = dict(checked.get("composite_event_delivery", {}))
    actions = record.get("actions", [])
    for update in definition.updates:
        delivered_index = next(
            (
                index
                for index, action in enumerate(actions)
                if (
                    action.get("composite_event_id") == update.event_id
                    if definition.explicit_events
                    else action.get("type")
                    == {"R": "visual_context_update", "G": "instruction_injection"}[
                        update.mechanism
                    ]
                    and is_update_event(action)
                )
            ),
            None,
        )
        if update.event_id not in details:
            details[update.event_id] = {
                "kind": update.mechanism,
                "status": "delivered"
                if delivered_index is not None
                else "not_delivered",
                "step": actions[delivered_index]["step"]
                if delivered_index is not None
                else None,
                "source_seconds": update.onset_seconds,
            }
        selection = None
        if delivered_index is not None:
            selection = any(
                Counter(action["state_after"]["selected_entity_ids"])
                - Counter(action["state_before"]["selected_entity_ids"])
                for action in actions[:delivered_index]
                if isinstance(action.get("state_before"), dict)
                and isinstance(action.get("state_after"), dict)
            )
        details[update.event_id] = {
            **details[update.event_id],
            "selected_before_update": selection,
        }
    return {
        "combination": "".join(kind for kind in "CRG" if kind in definition.mechanisms),
        "update_count": len(definition.updates),
        "clarification_offered": record.get("initial_context", {}).get(
            "clarification_available"
        ),
        "clarification_received": checked.get("composite_delivery", {}).get("C")
        if definition.clarification
        else None,
        "step_budget_reached": record.get("steps", 0) >= record["meta"]["max_steps"],
        "delivery": checked.get("composite_delivery", {}),
        "events": details,
    }


def _composite_bucket(
    observed: list[dict[str, Any]], replicates: int, ids: list[str],
) -> dict[str, Any]:
    seen = [row for row in observed if row["task_id"] in ids]
    valid = [row for row in seen if row["run_valid"]]
    planned = len(ids) * replicates
    return {
        "planned": planned,
        **({"planned_tasks": len(ids)} if replicates > 1 else {}),
        "observed": len(seen),
        "valid": len(valid),
        "invalid": len(seen) - len(valid),
        "pending": planned - len(seen),
        "success": _rate([row["e2e_success"] for row in valid]),
    }


def _composite_statistics(
    arms: list[ControlledArm],
    rows: list[dict[str, Any]],
    evidence: dict[str, Any] | None,
) -> dict[str, Any]:
    composite = [
        arm for arm in arms if arm.tasks and arm.tasks[0]["track"] == COMPOSITE_TRACK
    ]
    if not composite:
        return {}
    annotations = (evidence or {}).get("change_semantics", {})
    result = {}
    for arm in composite:
        observed = [
            row for row in rows
            if row["analysis_reference"]["arm"] == arm.name
        ]
        tasks = {task["id"]: task for task in arm.tasks}
        groups: dict[str, dict[str, list[str]]] = {
            "by_combination": defaultdict(list),
            "by_update_count": defaultdict(list),
            "by_semantic_type": defaultdict(list),
            "by_choice_relation": defaultdict(list),
        }
        combination_counts: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for task_id, task in tasks.items():
            definition = composite_input(task)
            combination = "".join(
                kind for kind in "CRG" if kind in definition.mechanisms
            )
            groups["by_combination"][combination].append(task_id)
            groups["by_update_count"][str(len(definition.updates))].append(task_id)
            combination_counts[combination][str(len(definition.updates))].append(task_id)
            annotation = annotations.get(task_id)
            if annotation is None:
                continue
            semantic_types = {
                f"{update.mechanism}:{kind}"
                for update in definition.updates
                for kind in annotation["events"][update.event_id]["types"]
            }
            if annotation.get("clarify"):
                semantic_types.add(f"C:{annotation['clarify']['type']}")
            for kind in sorted(semantic_types):
                groups["by_semantic_type"][kind].append(task_id)
            groups["by_choice_relation"][annotation["choice_relation"]["type"]].append(
                task_id
            )

        bucket = partial(_composite_bucket, observed, len(arm.plan.replicate_ids))
        valid = [
            row for row in observed if row["run_valid"] and "composite" in row
        ]
        result[arm.name] = {
            **({"replicate_ids": list(arm.plan.replicate_ids)} if len(arm.plan.replicate_ids) > 1 else {}),
            "overall": bucket(list(tasks)),
            **{
                group: {label: bucket(ids) for label, ids in sorted(buckets.items())}
                for group, buckets in groups.items()
            },
            "by_combination_update_count": {
                combination: {
                    count: bucket(ids) for count, ids in sorted(counts.items())
                }
                for combination, counts in sorted(combination_counts.items())
            },
            "missing_semantic_annotations": sorted(set(tasks) - set(annotations)),
            "semantic_groups_overlap": True,
            "step_budget_reached": _rate([row["composite"]["step_budget_reached"] for row in valid]),
            "actual_update_events": {
                kind: _rate(
                    [
                        detail["status"] in {"delivered", "provided_upfront"}
                        for row in valid
                        for detail in row["composite"]["events"].values()
                        if detail["kind"] == kind
                    ]
                )
                for kind in "RG"
            },
            "clarification_offered": _rate(
                [
                    row["composite"]["clarification_offered"]
                    for row in valid
                    if row["composite"]["clarification_received"] is not None
                ]
            ),
            "clarification_received": _rate(
                [
                    row["composite"]["clarification_received"]
                    for row in valid
                    if row["composite"]["clarification_received"] is not None
                ]
            ),
            "actual_delivery": {
                kind: _rate(
                    [
                        row["composite"]["delivery"][kind]
                        for row in valid
                        if kind in row["composite"]["delivery"]
                    ]
                )
                for kind in "RG"
            },
        }
    return {"composite_statistics": result}


def summarize_records(
    arms: list[ControlledArm],
    records: list[tuple[str, dict[str, Any]]],
    *,
    root: Path,
    evidence: dict[str, Any] | None = None,
    execution_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score historical task snapshots and compare only the registered protocols."""
    schedule = execution_schedule(arms)
    outcomes = execution_results or []
    by_experiment = {arm.plan.experiment_id: arm for arm in arms}
    indexes = {
        arm.name: {
            (row["task_id"], row["condition"], row["replicate_id"]): row[
                "schedule_index"
            ]
            for row in arm.schedule()
        }
        for arm in arms
    }
    rows: list[dict[str, Any]] = []
    groups: dict[str, str] = {}
    materials: dict[tuple[str, str], tuple[Any, ...]] = {}
    paired_sources: dict[str, str] = {}
    seen = set()
    for experiment_id, record, reference, rescore in _referenced_records(
        arms, records, outcomes, schedule
    ):
        arm = by_experiment[experiment_id]
        key = (
            record["task_id"],
            reference["condition"],
            record["meta"]["replicate_id"],
        )
        identity = (experiment_id, *key)
        if identity in seen or key not in indexes[arm.name]:
            raise ValueError(f"duplicate or unregistered run: {identity}")
        seen.add(identity)
        snapshot = record.get("task_snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("id") != record["task_id"]:
            raise ValueError(f"{identity}: missing historical task snapshot")
        scoring_task = (
            next(task for task in arm.tasks if task["id"] == record["task_id"])
            if rescore
            else snapshot
        )
        family = "timing" if arm.variable == "timing" else arm.name
        material_key = (family, snapshot["id"])
        controls = tuple(
            scoring_task.get(field)
            for field in (
                "instructions",
                "goal",
                "visual_packets",
                "graft",
                "refresh",
                "clarification",
                "truth",
            )
        )
        if material_key in materials and materials[material_key] != controls:
            raise ValueError(f"{identity}: task content differs between conditions")
        materials[material_key] = controls
        group = task_recording_group(snapshot, root=root)
        if arm.pair_id:
            if arm.pair_id in paired_sources and paired_sources[arm.pair_id] != group:
                raise ValueError(f"{identity}: source changed within content pair")
            paired_sources[arm.pair_id] = group
        if snapshot["id"] in groups and groups[snapshot["id"]] != group:
            raise ValueError("source changed between experimental arms")
        groups[snapshot["id"]] = group
        historical = check_run(snapshot, record)
        checked = rescore_run(scoring_task, record) if rescore else historical
        process_record = {
            **record,
            "task_snapshot": scoring_task,
            "run_status": {"valid": checked["run_valid"]},
        }
        label = (
            f"{arm.variable}::{arm.level}" if arm.pair_id else f"{arm.name}::{key[1]}"
        )
        row = {
            **score_summary(checked),
            "artifact_directory": record["meta"].get("artifact_directory"),
            "condition": {"name": label},
            "variable": arm.variable,
            "analysis_mode": _analysis_mode(arm),
            "level": arm.level or key[1],
            **evidence_visibility(
                evidence, snapshot["id"], Condition.from_dict(record["condition"])
            ),
            "frame_interval_seconds": record["condition"].get("frame_interval_seconds"),
            "run_id": record["meta"].get("run_id"),
            "analysis_reference": reference,
            "scoring": {
                "task_source": "current_frozen_task"
                if rescore
                else "run.task_snapshot",
                "historical_task_source": "run.task_snapshot",
                "historical_success": historical["e2e_success"],
            },
            "process": controlled_run_metrics(process_record),
            "trajectory": trajectory_mechanism(
                {**record, "run_status": {"valid": checked["run_valid"]}}
            ),
            "stage_release": record["meta"].get("stage_release"),
            "usage": record.get("trace", {}).get("usage")
            if isinstance(record.get("trace"), dict)
            else None,
            "request_inputs": _request_inputs(record),
            "costs": _run_costs(record),
        }
        if snapshot["track"] == COMPOSITE_TRACK:
            row["composite"] = _composite_run_facts(scoring_task, record, checked)
        if arm.pair_id:
            row["pair_id"] = arm.pair_id
        if arm.variable == "redundancy":
            row["process"]["delivered_unique_frame_count"] = len(
                {
                    path
                    for batch in record["meta"]["released_visual_inputs"]
                    for path in batch["frame_paths"]
                }
            )
        rows.append(row)
    for arm in arms:
        for task in arm.tasks:
            if task["id"] not in groups:
                groups[task["id"]] = task_recording_group(task, root=root)
    formal = next((arm for arm in arms if arm.variable == "formal"), None)
    statistic_rows = rows + [
        row for row in _failed_execution_rows(arms, schedule, outcomes)
        if (
            row["experiment_id"],
            row["task_id"],
            row["analysis_reference"]["condition"],
            row["replicate_id"],
        ) not in seen
    ]
    return {
        "analysis_version": 4,
        "expected_runs": sum(len(arm.schedule()) for arm in arms),
        "observed_runs": len(rows),
        "physical_runs": len(records),
        "request_failures": _request_failure_summary(outcomes),
        "repetition_statistics": repetition_statistics(arms, rows),
        "runs": rows,
        "comparisons": _comparison_results(arms, statistic_rows, groups),
        **(
            {
                **_formal_statistics(formal, statistic_rows, groups),
                "by_update_commitment": _commitment_summaries(statistic_rows),
                "update_state_analysis": update_state_summary([
                    row for row in statistic_rows
                    if row["variable"] == "formal" and row["level"] == "online"
                ]),
            } if formal is not None else {}
        ),
        **_composite_statistics(arms, statistic_rows, evidence),
        "conditions": _condition_summaries(rows),
        "by_track": _grouped_summaries(rows, "track"),
        "by_site": _grouped_summaries(rows, "site"),
        "by_variable": _variable_summaries(arms, rows),
        "observational_analysis": _observational_summaries(rows, evidence),
    }
