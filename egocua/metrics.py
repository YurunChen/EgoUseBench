from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from statistics import mean
from typing import Any

from .controlled_metrics import PairedComparison, paired_source_statistics
from .timeline import Condition, STANDARD_CONDITIONS


def _avg(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def _iqm(values: list[float]) -> float | None:
    """Interquartile mean over per-task values (rliable, Agarwal et al. 2021)."""

    if not values:
        return None
    ordered = sorted(values)
    quarter = len(ordered) // 4
    middle = ordered[quarter : len(ordered) - quarter] or ordered
    return mean(middle)


def _task_iqm(rows: list[dict[str, Any]], field: str) -> float | None:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("__sample_id", row["task_id"])].append(row)
    task_values = [
        value for group in grouped.values() if (value := _avg(group, field)) is not None
    ]
    return _iqm(task_values)


def _task_macro_avg(rows: list[dict[str, Any]], field: str) -> float | None:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("__sample_id", row["task_id"])].append(row)
    task_values = [
        value for group in grouped.values() if (value := _avg(group, field)) is not None
    ]
    return mean(task_values) if task_values else None


def _condition(
    rows: list[dict[str, Any]],
    **factors: Any,
) -> list[dict[str, Any]]:
    factors = {key: value for key, value in factors.items() if key != "name"}
    selected = []
    for row in rows:
        condition = Condition.from_dict(row["condition"])
        if all(getattr(condition, key) == value for key, value in factors.items()):
            selected.append(row)
    return selected


def _conditional_rate(
    rows: list[dict[str, Any]], field: str, denominator: str
) -> float | None:
    eligible = [row for row in rows if row.get(denominator)]
    return _task_macro_avg(eligible, field)


def _three_replicate_reliability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = row.get("__sample_id", row["task_id"])
        grouped[(sample_id, row["condition"]["name"])].append(row)
    complete = [
        group
        for group in grouped.values()
        if len(group) == 3 and len({row.get("replicate_id") for row in group}) == 3
    ]
    success_counts = [
        sum(bool(row.get("e2e_success")) for row in group) for group in complete
    ]
    return {
        "pass_at_1": _task_macro_avg(rows, "e2e_success"),
        "pass_3": (
            mean(float(successes == 3) for successes in success_counts)
            if success_counts
            else None
        ),
        "success_count_distribution": {
            f"{successes}_of_3": (
                success_counts.count(successes) / len(success_counts)
                if success_counts
                else None
            )
            for successes in range(4)
        },
    }


def _clarification_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Clarification quality under condition semantics."""

    clarify = [row for row in rows if row["track"] == "clarify"]
    progressive = _condition(
        clarify, **STANDARD_CONDITIONS["online"].as_dict()
    )
    complete_at_start = _condition(
        clarify, **STANDARD_CONDITIONS["upfront"].as_dict()
    )
    ask_recall = _task_macro_avg(progressive, "asked")
    progressive_ask_rate = ask_recall or 0.0
    upfront_ask_rate = _task_macro_avg(complete_at_start, "asked") or 0.0
    total_asks = progressive_ask_rate + upfront_ask_rate
    ask_precision = progressive_ask_rate / total_asks if total_asks else None
    ask_f1 = None
    if ask_precision is not None and ask_recall is not None:
        denominator = ask_precision + ask_recall
        ask_f1 = 2 * ask_precision * ask_recall / denominator if denominator else 0.0
    return {
        "ask_recall": ask_recall,
        "ask_precision": ask_precision,
        "ask_f1": ask_f1,
        "valid_question_rate": _conditional_rate(
            progressive, "valid_question", "asked"
        ),
        "timely_question_rate": _conditional_rate(
            progressive, "timely_question", "asked"
        ),
        "answer_utilization": _task_macro_avg(
            [
                row
                for row in progressive
                if row.get("valid_question") and row.get("timely_question")
            ],
            "answer_used",
        ),
        "upfront_over_ask_rate": _task_macro_avg(complete_at_start, "over_ask"),
    }


# --- Unified timing and commitment metrics (§2.4) ----------------------------


def _post_release_sr(rows: list[dict[str, Any]], budget: int) -> float | None:
    """SR(k): goal reached with the post-release commitment within k steps (InterruptBench)."""

    eligible = [
        {
            **row,
            "__within": (
                row.get("post_release_latency") is not None
                and row["post_release_latency"] <= budget
                and row.get("goal_state_correct") is True
            ),
        }
        for row in rows
        if row.get("release_delivered") and row.get("goal_state_correct") is not None
    ]
    return _task_macro_avg(eligible, "__within")


def _timing_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate commitment-timing fields across all tracks and conditions."""

    has_timing = [row for row in rows if row.get("sufficiency_step") is not None]
    return {
        "premature_commitment_rate": _task_macro_avg(
            has_timing, "premature_commitment"
        ),
        "premature_to_tempting_rate": _task_macro_avg(
            has_timing, "premature_to_tempting"
        ),
        "premature_submission_rate": _task_macro_avg(
            has_timing, "premature_submission"
        ),
        "stale_final_rate": _task_macro_avg(rows, "stale_final"),
        "missing_after_update_rate": _task_macro_avg(rows, "missing_after_update"),
        "false_update_rate": _task_macro_avg(rows, "false_update"),
        "revised_rate": _conditional_rate(
            has_timing, "revised", "premature_commitment"
        ),
        "over_ask_rate": _task_macro_avg(rows, "over_ask"),
        "policy_pass_rate": _task_macro_avg(rows, "policy_pass"),
        "mean_waits_before_sufficiency": _task_macro_avg(
            has_timing, "waits_before_sufficiency"
        ),
        "mean_waits_after_sufficiency": _task_macro_avg(
            has_timing, "waits_after_sufficiency"
        ),
        "mean_productive_before_sufficiency": _task_macro_avg(
            has_timing, "productive_actions_before_sufficiency"
        ),
        "post_release_recovery_latency": _task_macro_avg(rows, "post_release_latency"),
        **{
            f"post_release_sr_at_{budget}": _post_release_sr(rows, budget)
            for budget in (1, 3, 5, 10)
        },
    }


# --- Headline table -------------------------------------------------------------

HEADLINE_LEGEND = {
    "success": "成功率 SR：结束时网站状态满足任一合法答案的比例（e2e_success）",
    "stale": "过时残留 Stale：现实或意图已否定的对象仍留在结束时网站状态里的比例",
    "collateral_damage": "附带损害 Collateral Damage：变化前已合法选中、最终仍需要的内容在结束时被删或被换的比例",
}


def _headline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per track × condition; the paper's headline numbers."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        condition = row["condition"]["name"]
        groups[(str(row["track"]), str(condition))].append(row)
    table: list[dict[str, Any]] = []
    for (track, condition), subset in sorted(groups.items()):
        table.append(
            {
                "track": track,
                "condition": condition,
                "runs": len(subset),
                "success": _task_macro_avg(subset, "e2e_success"),
                "stale": _task_macro_avg(subset, "stale_final"),
                "collateral_damage": _task_macro_avg(subset, "collateral_damage"),
            }
        )
    return table


# --- Main aggregate -----------------------------------------------------------


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row.get("run_valid") is True]
    clarification = _clarification_policy(valid_rows)
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
        "headline": _headline(valid_rows),
        "headline_legend": HEADLINE_LEGEND,
        "clarification_policy": clarification,
        "common": {
            "valid_run_rate": _task_macro_avg(
                [{**row, "__valid": row.get("run_valid") is True} for row in rows],
                "__valid",
            ),
            "e2e_success": _task_macro_avg(valid_rows, "e2e_success"),
            "e2e_success_iqm": _task_iqm(valid_rows, "e2e_success"),
            "goal_state_correctness": _task_macro_avg(valid_rows, "goal_state_correct"),
            "submission_completion": _task_macro_avg(
                valid_rows, "submission_completed"
            ),
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
            "selection_corrections": _task_macro_avg(
                valid_rows, "selection_corrections"
            ),
            "evidence": {
                "unknown_rate": evidence_unknown_rate,
                "success_lower_bound": evidence_pass_rate,
                "success_upper_bound": (
                    evidence_pass_rate + evidence_unknown_rate
                    if evidence_pass_rate is not None
                    and evidence_unknown_rate is not None
                    else None
                ),
            },
            "reliability": _three_replicate_reliability(valid_rows),
            "timing": _timing_metrics(valid_rows),
        },
    }

    by_track = {
        track: [row for row in valid_rows if row["track"] == track]
        for track in ("graft", "refresh", "clarify")
    }
    report["breakdown"] = {
        track: {
            subtype: {
                "runs": len(subtype_rows),
                "e2e_success": _task_macro_avg(subtype_rows, "e2e_success"),
                "goal_state_correctness": _task_macro_avg(
                    subtype_rows, "goal_state_correct"
                ),
                **_timing_metrics(subtype_rows),
            }
            for subtype in sorted({row["subtype"] for row in track_rows})
            for subtype_rows in [
                [row for row in track_rows if row["subtype"] == subtype]
            ]
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
        for site_rows in [
            [row for row in valid_rows if row.get("site", "unspecified") == site]
        ]
    }

    report["paired_effects"] = _paired_condition_effects(valid_rows)

    return report


def _paired_condition_effects(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Compare treatments within the same task and remaining condition factors."""

    effects: dict[str, dict[str, dict[str, float | None]]] = {}
    pairs = [
        (
            "upfront_minus_online",
            STANDARD_CONDITIONS["upfront"].as_dict(),
            STANDARD_CONDITIONS["online"].as_dict(),
        ),
        (
            "full_minus_initial_only",
            STANDARD_CONDITIONS["online"].as_dict(),
            STANDARD_CONDITIONS["initial_only"].as_dict(),
        ),
        (
            "allowed_minus_denied",
            STANDARD_CONDITIONS["online"].as_dict(),
            STANDARD_CONDITIONS["no_ask"].as_dict(),
        ),
        (
            "full_minus_first_frame",
            STANDARD_CONDITIONS["online"].as_dict(),
            STANDARD_CONDITIONS["first_frame"].as_dict(),
        ),
        (
            "full_minus_text_only",
            STANDARD_CONDITIONS["online"].as_dict(),
            STANDARD_CONDITIONS["text_only"].as_dict(),
        ),
    ]
    for track in sorted({str(row["track"]) for row in rows}):
        track_rows = [row for row in rows if row["track"] == track]
        by_subtype: dict[str, dict[str, float | None]] = {}
        for subtype in sorted({str(row["subtype"]) for row in track_rows}):
            subset = [row for row in track_rows if row["subtype"] == subtype]
            contrasts: dict[str, float | None] = {
                label: difference
                for label, left, right in pairs
                if (
                    difference := _condition_difference(subset, left, right, "e2e_success")
                ) is not None
            }
            if contrasts:
                by_subtype[subtype] = contrasts
        if by_subtype:
            effects[track] = by_subtype
    return effects


def _condition_difference(
    rows: list[dict[str, Any]],
    left: dict[str, Any],
    right: dict[str, Any],
    field: str,
) -> float | None:
    changed_factors = {
        key for key in left.keys() | right.keys()
        if key != "name" and left.get(key) != right.get(key)
    }
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = row.get("__sample_id", row["task_id"])
        controls = tuple(
            (key, value)
            for key, value in asdict(Condition.from_dict(row["condition"])).items()
            if key != "name" and key not in changed_factors
        )
        grouped[(sample_id, controls)].append(row)
    paired_effects: dict[Any, list[float]] = defaultdict(list)
    for (sample_id, _), group in grouped.items():
        left_value = _avg(_condition(group, **left), field)
        right_value = _avg(_condition(group, **right), field)
        if left_value is not None and right_value is not None:
            paired_effects[sample_id].append(left_value - right_value)
    return (
        mean(mean(values) for values in paired_effects.values())
        if paired_effects
        else None
    )


# Each name states the sign of its difference, online minus the reference.
# The upfront contrast also carries InterventionBench's Adaptability.
PAIRED_CONTRASTS = (
    ("online_minus_upfront", "upfront", "online"),
    ("online_minus_text_only", "text_only", "online"),
    ("online_minus_no_ask", "no_ask", "online"),
)
# Stratify this check by subtype: prior solutions can remain valid after an update.
UPDATE_NECESSITY = ("online_minus_initial_only", "initial_only", "online")


def source_intervals(
    rows: list[dict[str, Any]], source_groups: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Paired contrasts with source-clustered intervals, per track.

    The update-necessity check is nested per subtype under each track. Tasks
    that share a recording are one source, so they do not count as independent
    evidence.
    """
    report: dict[str, dict[str, Any]] = {}
    for track in sorted({str(row["track"]) for row in rows}):
        subset = [row for row in rows if row["track"] == track]
        report[track] = _paired_contrasts(subset, source_groups, PAIRED_CONTRASTS)
        by_subtype = {
            subtype: checked
            for subtype in sorted({str(row["subtype"]) for row in subset})
            if (
                checked := _paired_contrasts(
                    [row for row in subset if row["subtype"] == subtype],
                    source_groups,
                    (UPDATE_NECESSITY,),
                )
            )
        }
        if by_subtype:
            report[track]["by_subtype"] = by_subtype
    return report


def _paired_contrasts(
    rows: list[dict[str, Any]],
    source_groups: dict[str, str],
    contrasts: tuple[tuple[str, str, str], ...],
) -> dict[str, Any]:
    groups = {
        row["task_id"]: source_groups.get(row["task_id"], row["task_id"])
        for row in rows
    }
    computed = {
        name: paired_source_statistics(rows, PairedComparison(left, right, groups))
        for name, left, right in contrasts
    }
    return {
        name: contrast
        for name, contrast in computed.items()
        if contrast["coverage"]["paired_pairs"]
    }
