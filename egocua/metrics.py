from __future__ import annotations

import random
from collections import defaultdict
from statistics import mean
from typing import Any

CLARIFICATION_COSTS = (0.0, 0.02, 0.05, 0.10)


def _avg(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _task_macro_avg(rows: list[dict[str, Any]], field: str) -> float | None:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("__sample_id", row["task_id"])].append(row)
    task_values = [value for group in grouped.values() if (value := _avg(group, field)) is not None]
    return mean(task_values) if task_values else None


def _arm(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["arm"] == name]


def _conditional_rate(rows: list[dict[str, Any]], field: str, denominator: str) -> float | None:
    eligible = [row for row in rows if row.get(denominator)]
    return _task_macro_avg(eligible, field)


def _three_replicate_reliability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = row.get("__sample_id", row["task_id"])
        grouped[(sample_id, row["arm"])].append(row)
    complete = [
        group for group in grouped.values()
        if len(group) == 3 and len({row.get("replicate_id") for row in group}) == 3
    ]
    success_counts = [sum(bool(row.get("e2e_success")) for row in group) for group in complete]
    return {
        "pass_at_1": _task_macro_avg(rows, "e2e_success"),
        "pass_3": (
            mean(float(successes == 3) for successes in success_counts)
            if success_counts else None
        ),
        "success_count_distribution": {
            f"{successes}_of_3": (
                success_counts.count(successes) / len(success_counts) if success_counts else None
            )
            for successes in range(4)
        },
    }


def _graft_metrics(graft: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "dynamic_e2e_success": _task_macro_avg(_arm(graft, "injected"), "e2e_success"),
        "final_state_contains_a_injected": _task_macro_avg(
            _arm(graft, "injected"), "final_state_contains_a"
        ),
        "injected_integration_success": _task_macro_avg(
            _arm(graft, "injected"), "integration_success"
        ),
    }
    if _arm(graft, "bundled"):
        metrics["diagnostic_interruption_cost"] = _difference(
            graft, "bundled", "injected", "e2e_success"
        )
    return metrics


def _refresh_metrics(refresh: list[dict[str, Any]]) -> dict[str, Any]:
    replace = [row for row in refresh if row["subtype"] == "replace"]
    retain = [row for row in refresh if row["subtype"] == "retain"]
    retain_stream = _arm(retain, "stream")
    replace_stream = _arm(replace, "stream")
    responded = [
        row for row in replace_stream if row.get("post_update_target_action_responded")
    ]

    def success_within(row: dict[str, Any], budget: int) -> bool:
        latency = row.get("post_update_target_action_latency")
        return latency is not None and latency <= budget

    metrics = {
        "stream_replace_e2e_success": _task_macro_avg(replace_stream, "e2e_success"),
        "stream_retain_e2e_success": _task_macro_avg(retain_stream, "e2e_success"),
        "final_false_update_outcome_rate": _task_macro_avg(retain_stream, "false_update"),
        "post_update_target_action_response_rate": _task_macro_avg(
            replace_stream, "post_update_target_action_responded"
        ),
        "post_update_target_action_latency_on_response": _task_macro_avg(
            responded, "post_update_target_action_latency"
        ),
        **{
            f"post_update_sr_at_{budget}": (
                _task_macro_avg(
                    [
                        {**row, "__within_budget": success_within(row, budget)}
                        for row in replace_stream
                    ],
                    "__within_budget",
                )
            )
            for budget in (1, 3, 5, 10)
        },
    }
    if _arm(replace, "stale"):
        metrics["diagnostic_replace_adaptation_benefit"] = _difference(
            replace, "stream", "stale", "e2e_success"
        )
    if _arm(refresh, "oracle_latest"):
        metrics["diagnostic_replace_online_refresh_cost"] = _difference(
            replace, "oracle_latest", "stream", "e2e_success"
        )
        metrics["diagnostic_retain_online_robustness_loss"] = _difference(
            retain, "oracle_latest", "stream", "e2e_success"
        )
    return metrics


def _clarify_metrics(clarify: list[dict[str, Any]]) -> dict[str, Any]:
    interactive = _arm(clarify, "interactive")
    oracle = _arm(clarify, "oracle")
    ask_recall = _task_macro_avg(interactive, "asked")
    interactive_ask_rate = _task_macro_avg(interactive, "asked")
    oracle_ask_rate = _task_macro_avg(oracle, "asked")
    interactive_asks = interactive_ask_rate or 0.0
    ask_rate_sum = interactive_asks + (oracle_ask_rate or 0.0)
    ask_precision = interactive_asks / ask_rate_sum if ask_rate_sum else None
    ask_f1 = None
    if ask_precision is not None and ask_recall is not None:
        denominator = ask_precision + ask_recall
        ask_f1 = 2 * ask_precision * ask_recall / denominator if denominator else 0.0
    metrics: dict[str, Any] = {
        "interactive_e2e_success": _task_macro_avg(interactive, "e2e_success"),
        "ask_recall": ask_recall,
        "valid_question_rate": _conditional_rate(interactive, "valid_question", "asked"),
        "timely_question_rate": _conditional_rate(
            interactive, "timely_question", "asked"
        ),
        "answer_utilization": _task_macro_avg(
            [
                row for row in interactive
                if row.get("valid_question") and row.get("timely_question")
            ],
            "answer_used",
        ),
    }
    if oracle:
        metrics.update({
            "diagnostic_ask_precision": ask_precision,
            "diagnostic_ask_f1": ask_f1,
            "diagnostic_oracle_unnecessary_ask_rate": _task_macro_avg(
                oracle, "unnecessary_ask"
            ),
        })
    if _arm(clarify, "forced_execute"):
        metrics["diagnostic_clarification_gain"] = _difference(
            clarify, "interactive", "forced_execute", "e2e_success"
        )
        utility_sensitivity = {}
        for cost in CLARIFICATION_COSTS:
            utility_rows = [
                {
                    **row,
                    "__utility": (
                        float(row["e2e_success"]) - cost * float(bool(row.get("asked")))
                        if row.get("e2e_success") is not None else None
                    ),
                }
                for row in clarify
            ]
            utility_sensitivity[f"lambda_{cost:.2f}"] = _difference(
                utility_rows, "interactive", "forced_execute", "__utility"
            )
        metrics["diagnostic_utility_sensitivity"] = utility_sensitivity
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("run_valid", True)]
    evidence_rows = [
        {
            **row,
            "__evidence_pass": row.get("evidence_status") == "PASS",
            "__evidence_unknown": row.get("evidence_status") == "UNKNOWN",
        }
        for row in rows
        if row.get("evidence_status") in {"PASS", "FAIL", "UNKNOWN"}
    ]
    evidence_pass_rate = _task_macro_avg(evidence_rows, "__evidence_pass")
    evidence_unknown_rate = _task_macro_avg(evidence_rows, "__evidence_unknown")
    report: dict[str, Any] = {
        "runs": len(rows),
        "common": {
            "valid_run_rate": _task_macro_avg(
                [{**row, "__valid": row.get("run_valid", True)} for row in rows], "__valid"
            ),
            "e2e_success": _task_macro_avg(valid_rows, "e2e_success"),
            "goal_state_correctness": _task_macro_avg(valid_rows, "goal_state_correct"),
            "submission_completion": _task_macro_avg(valid_rows, "submission_completed"),
            "target_opened": _task_macro_avg(valid_rows, "target_opened"),
            "target_selected": _task_macro_avg(valid_rows, "target_selected"),
            "steps_on_success": _task_macro_avg(
                [row for row in valid_rows if row.get("e2e_success")], "steps"
            ),
            "repeat_rate": _task_macro_avg(valid_rows, "repeat_rate"),
            "search_actions": _task_macro_avg(valid_rows, "search_actions"),
            "distinct_search_queries": _task_macro_avg(
                valid_rows, "distinct_search_queries"
            ),
            "entity_detail_opens": _task_macro_avg(valid_rows, "entity_detail_opens"),
            "distinct_entities_opened": _task_macro_avg(
                valid_rows, "distinct_entities_opened"
            ),
            "selection_mutations": _task_macro_avg(valid_rows, "selection_mutations"),
            "selection_corrections": _task_macro_avg(valid_rows, "selection_corrections"),
            "evidence": {
                "unknown_rate": evidence_unknown_rate,
                "success_lower_bound": evidence_pass_rate,
                "success_upper_bound": (
                    evidence_pass_rate + evidence_unknown_rate
                    if evidence_pass_rate is not None and evidence_unknown_rate is not None
                    else None
                ),
            },
            "reliability": _three_replicate_reliability(valid_rows),
        },
    }
    by_track = {
        track: [row for row in valid_rows if row["track"] == track]
        for track in ("graft", "refresh", "clarify")
    }
    report["graft"] = _graft_metrics(by_track["graft"])
    report["refresh"] = _refresh_metrics(by_track["refresh"])
    report["clarify"] = _clarify_metrics(by_track["clarify"])

    report["breakdown"] = {
        track: {
            subtype: {
                "runs": len(subtype_rows),
                "e2e_success": _task_macro_avg(subtype_rows, "e2e_success"),
                "goal_state_correctness": _task_macro_avg(subtype_rows, "goal_state_correct"),
            }
            for subtype in sorted({row["subtype"] for row in track_rows})
            for subtype_rows in [[row for row in track_rows if row["subtype"] == subtype]]
        }
        for track, track_rows in by_track.items()
    }
    report["breakdown_by_site"] = {
        site: {
            "runs": len(site_rows),
            "e2e_success": _task_macro_avg(site_rows, "e2e_success"),
            "goal_state_correctness": _task_macro_avg(site_rows, "goal_state_correct"),
        }
        for site in sorted({str(row.get("site", "unspecified")) for row in valid_rows})
        for site_rows in [[row for row in valid_rows if row.get("site", "unspecified") == site]]
    }
    return report


def _difference(rows: list[dict[str, Any]], left: str, right: str, field: str) -> float | None:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = row.get("__sample_id", row["task_id"])
        grouped[sample_id].append(row)
    paired_effects = []
    for group in grouped.values():
        left_value = _avg(_arm(group, left), field)
        right_value = _avg(_arm(group, right), field)
        if left_value is not None and right_value is not None:
            paired_effects.append(left_value - right_value)
    return mean(paired_effects) if paired_effects else None


def bootstrap_intervals(
    rows: list[dict[str, Any]], *, samples: int = 2000, seed: int = 0
) -> dict[str, list[float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task_id"]].append(row)
    task_ids = sorted(grouped)
    if not task_ids:
        return {}
    task_strata = {
        row["task_id"]: (row["track"], row.get("site", "unspecified"), row["subtype"])
        for row in rows
    }
    ids_by_stratum = {
        stratum: [task_id for task_id in task_ids if task_strata[task_id] == stratum]
        for stratum in sorted(set(task_strata.values()))
    }
    rng = random.Random(seed)
    estimates: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled_rows: list[dict[str, Any]] = []
        for (track, site, subtype), stratum_ids in ids_by_stratum.items():
            for draw_index, task_id in enumerate(
                rng.choices(stratum_ids, k=len(stratum_ids))
            ):
                sample_id = f"{track}-{site}-{subtype}-{draw_index}"
                sampled_rows.extend(
                    {**row, "__sample_id": sample_id} for row in grouped[task_id]
                )
        for key, value in _flatten_metrics(aggregate(sampled_rows)).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                estimates[key].append(float(value))
    intervals = {}
    for key, values in estimates.items():
        values.sort()
        count = len(values)
        intervals[key] = [
            values[int(count * 0.025)],
            values[min(count - 1, int(count * 0.975))],
        ]
    return intervals


def _flatten_metrics(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else key
        flattened.update(_flatten_metrics(child, child_prefix))
    return flattened
