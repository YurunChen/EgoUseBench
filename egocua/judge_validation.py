"""Independent Judge validation experiment for PULSE failure attribution.

Validation assessments live in `judge_validation.assessments`; they never replace
the formal conclusions in `failure-attribution.json::assessments`.
"""

from __future__ import annotations

import math
import fcntl
import random
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .workspace import write_json_atomic
from .failure_attribution import (
    FAILURE_EVIDENCE_PATH,
    PULSE_STAGES,
    apply_assessments,
    validate_attribution,
    verify_judge_assessment,
    qualify_attribution_row,
)


def register_judge_validation(
    root: str | Path,
    selection: Mapping[str, Any],
    *,
    model: str | None,
    prompt_version: str,
) -> dict[str, Any]:
    """Freeze one source-separated Judge validation in the shared evidence file."""
    validate_judge_sample(selection, set(selection["source_groups"]))
    if (model is not None and not model.strip()) or not prompt_version.strip():
        raise ValueError("Judge validation requires model and prompt_version")
    path = Path(root).resolve() / FAILURE_EVIDENCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = (
            json.loads(path.read_text())
            if path.exists()
            else {
                "schema_version": 1,
                "assessments": [],
                "ratings": [],
                "run_qualifications": {},
            }
        )
        configured = {
            **deepcopy(dict(selection)),
            "model": model,
            "prompt_version": prompt_version,
        }
        current = evidence.get("judge_validation")
        if current is not None:
            same_selection = all(
                current[key] == selection[key]
                for key in (
                    "calibration_run_ids",
                    "validation_run_ids",
                    "source_groups",
                )
            )
            same_judge = (
                current["model"] == model
                and current["prompt_version"] == prompt_version
            )
            if same_selection and same_judge:
                return deepcopy(current)
            selected_ids = {
                *current["calibration_run_ids"],
                *current["validation_run_ids"],
                *selection["calibration_run_ids"],
                *selection["validation_run_ids"],
            }
            if not same_selection:
                has_human_review = any(
                    rating["run_id"] in selected_ids
                    for rating in evidence.get("ratings", [])
                )
                if (
                    current["model"] is not None
                    or model is not None
                    or current["assessments"]
                    or has_human_review
                ):
                    raise ValueError(
                        "Judge validation selection is already frozen with different conditions"
                    )
                evidence["judge_validation"] = {
                    **configured,
                    "assessments": [],
                    "validation_started": False,
                }
            else:
                heldout = set(current["validation_run_ids"])
                if (
                    current.get("validation_started")
                    or any(item["run_id"] in heldout for item in current["assessments"])
                    or any(
                        item["run_id"] in heldout
                        for item in evidence.get("ratings", [])
                    )
                ):
                    raise ValueError(
                        "Judge validation model/prompt_version is frozen after held-out validation starts"
                    )
                if current["assessments"]:
                    current.setdefault("calibration_history", []).append(
                        {
                            "model": current["model"],
                            "prompt_version": current["prompt_version"],
                            "assessments": deepcopy(current["assessments"]),
                        }
                    )
                current.update(
                    model=model, prompt_version=prompt_version, assessments=[]
                )
            write_json_atomic(path, evidence)
            return deepcopy(evidence["judge_validation"])
        evidence["judge_validation"] = {
            **configured,
            "assessments": [],
            "validation_started": False,
        }
        write_json_atomic(path, evidence)
        return deepcopy(evidence["judge_validation"])


def begin_judge_validation(root: str | Path) -> dict[str, Any]:
    """Freeze the Judge before exposing any held-out case to an API request."""
    path = Path(root).resolve() / FAILURE_EVIDENCE_PATH
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = json.loads(path.read_text())
        validation = evidence["judge_validation"]
        if not validation["model"]:
            raise ValueError("Choose the Judge before starting held-out validation")
        if not validation.get("validation_started"):
            validation["validation_started"] = True
            write_json_atomic(path, evidence)
        return deepcopy(validation)


def judge_exposed_sources(evidence: Mapping[str, Any]) -> set[str]:
    """Return recording sources already used to tune or inspect Judge behavior."""
    used_sources: set[str] = set()
    reviewed_ids = {item["run_id"] for item in evidence.get("ratings", [])}
    experiments = [*evidence.get("judge_validation_history", [])]
    if evidence.get("judge_validation") is not None:
        experiments.append(evidence["judge_validation"])
    for experiment in experiments:
        heldout = set(experiment["validation_run_ids"])
        judged_ids = {item["run_id"] for item in experiment["assessments"]}
        for calibration in experiment.get("calibration_history", []):
            judged_ids.update(item["run_id"] for item in calibration["assessments"])
        used = (reviewed_ids | judged_ids) & set(experiment["source_groups"])
        if (
            experiment.get("validation_started")
            or (reviewed_ids | judged_ids) & heldout
        ):
            used.update(heldout)
        used_sources.update(experiment["source_groups"][run_id] for run_id in used)
    return used_sources


def start_new_judge_validation(
    root: str | Path, selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep the previous experiment and choose genuinely unused held-out sources."""
    validate_judge_sample(selection, set(selection["source_groups"]))
    path = Path(root).resolve() / FAILURE_EVIDENCE_PATH
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = json.loads(path.read_text())
        previous = evidence["judge_validation"]
        history = evidence.get("judge_validation_history", [])
        used_sources = judge_exposed_sources(evidence)
        fresh_sources = {
            selection["source_groups"][run_id]
            for run_id in selection["validation_run_ids"]
        }
        if fresh_sources & used_sources:
            raise ValueError(
                "New held-out validation must use sources not exposed in previous experiments"
            )
        evidence["judge_validation_history"] = [*history, deepcopy(previous)]
        evidence["judge_validation"] = {
            **deepcopy(dict(selection)),
            "model": None,
            "prompt_version": previous["prompt_version"],
            "assessments": [],
            "validation_started": False,
        }
        write_json_atomic(path, evidence)
        return deepcopy(evidence["judge_validation"])


def save_judge_validation_assessment(
    root: str | Path, row: Mapping[str, Any], assessment: Mapping[str, Any]
) -> dict[str, Any]:
    """Store experimental judgments; task doubts do not change global qualifications."""
    workspace = Path(root).resolve()
    path = workspace / FAILURE_EVIDENCE_PATH
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = json.loads(path.read_text())
        validation = evidence["judge_validation"]
        if (
            assessment.get("reviewer_kind") != "llm_judge"
            or assessment.get("run_id") != row["run_id"]
        ):
            raise ValueError("Judge validation assessment identity mismatch")
        if (
            not validation["model"]
            or assessment.get("judge_model") != validation["model"]
            or assessment.get("prompt_version") != validation["prompt_version"]
        ):
            raise ValueError("Judge validation model/prompt_version mismatch")
        _require_validation_run(validation, row["run_id"])
        if row["run_id"] in validation["validation_run_ids"] and not validation.get(
            "validation_started"
        ):
            raise ValueError(
                "Begin held-out validation before saving a held-out assessment"
            )
        existing = next(
            (
                item
                for item in validation["assessments"]
                if item["run_id"] == row["run_id"]
            ),
            None,
        )
        if existing is not None:
            return deepcopy(existing)
        current_row = qualify_attribution_row(
            row, evidence.get("run_qualifications", {}).get(row["run_id"]), workspace
        )
        if current_row["run_verdict"] != "valid_model_failure":
            raise ValueError(
                "Judge validation requires a currently valid model failure"
            )
        apply_assessments([current_row], [assessment], workspace)
        saved = deepcopy(dict(assessment))
        validation["assessments"].append(saved)
        write_json_atomic(path, evidence)
        return saved


def _require_validation_run(validation: Mapping[str, Any], run_id: str) -> None:
    if run_id not in {
        *validation["calibration_run_ids"],
        *validation["validation_run_ids"],
    }:
        raise ValueError("run is outside the registered Judge validation")


def save_judge_dispute_review(
    root: str | Path, row: Mapping[str, Any], review: Mapping[str, Any]
) -> dict[str, Any]:
    """Attach a Coding Agent note without replacing Judge or human judgments."""
    workspace = Path(root).resolve()
    path = workspace / FAILURE_EVIDENCE_PATH
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = json.loads(path.read_text())
        validation = evidence["judge_validation"]
        _require_validation_run(validation, row["run_id"])
        assessment = next(
            (
                item
                for item in validation["assessments"]
                if item["run_id"] == row["run_id"]
            ),
            None,
        )
        if assessment is None:
            raise ValueError("dispute review requires an existing Judge assessment")
        if (
            assessment.get("reviewer_kind") != "llm_judge"
            or assessment.get("judge_model") != validation["model"]
            or assessment.get("prompt_version") != validation["prompt_version"]
        ):
            raise ValueError("Judge validation model/prompt_version mismatch")
        if "dispute_review" in assessment:
            return deepcopy(assessment["dispute_review"])
        if row["run_verdict"] != "valid_model_failure":
            raise ValueError("dispute review requires an original valid model failure")
        annotated = {**assessment, "dispute_review": deepcopy(dict(review))}
        apply_assessments([row], [annotated], workspace)
        assessment["dispute_review"] = annotated["dispute_review"]
        write_json_atomic(path, evidence)
        return deepcopy(assessment["dispute_review"])


def save_judge_recheck(
    root: str | Path, row: Mapping[str, Any], assessment: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep a targeted Judge recheck separate from the frozen validation results."""
    workspace = Path(root).resolve()
    path = workspace / FAILURE_EVIDENCE_PATH
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = json.loads(path.read_text())
        validation = evidence["judge_validation"]
        _require_validation_run(validation, row["run_id"])
        if (
            assessment.get("reviewer_kind") != "llm_judge"
            or assessment.get("run_id") != row["run_id"]
            or not isinstance(assessment.get("judge_model"), str)
            or not assessment["judge_model"].strip()
            or not isinstance(assessment.get("prompt_version"), str)
            or not assessment["prompt_version"].strip()
        ):
            raise ValueError(
                "Judge recheck requires matching run, model and prompt_version"
            )
        rechecks = validation.get("rechecks", [])
        identity = ("run_id", "judge_model", "prompt_version")
        existing = next(
            (
                item
                for item in rechecks
                if all(item[key] == assessment[key] for key in identity)
            ),
            None,
        )
        if existing is not None:
            if "dispute_review" in assessment and "dispute_review" not in existing:
                original = {
                    key: value
                    for key, value in assessment.items()
                    if key != "dispute_review"
                }
                if original != existing:
                    raise ValueError(
                        "a dispute note cannot change the original Judge recheck"
                    )
                annotated = {**existing, "dispute_review": assessment["dispute_review"]}
                verify_judge_assessment(row, annotated, workspace)
                existing["dispute_review"] = deepcopy(assessment["dispute_review"])
                write_json_atomic(path, evidence)
            return deepcopy(existing)
        current_row = qualify_attribution_row(
            row, evidence.get("run_qualifications", {}).get(row["run_id"]), workspace
        )
        if current_row["run_verdict"] != "valid_model_failure":
            raise ValueError("Judge recheck requires a currently valid model failure")
        apply_assessments([current_row], [assessment], workspace)
        saved = deepcopy(dict(assessment))
        validation["rechecks"] = [*rechecks, saved]
        write_json_atomic(path, evidence)
        return deepcopy(saved)


def select_reliability_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum: int = 10,
    fraction: float = 0.2,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Audit uncertain cases and independently sample confident classifications."""

    failures = [
        dict(row)
        for row in rows
        if row.get("run_verdict") == "valid_model_failure"
        and row.get("attribution_status")
        not in {"coding_agent_pending", "rule_evaluation_pending"}
    ]
    selected = [
        row
        for row in failures
        if row.get("confidence") == "low" and row.get("review_required", True)
    ]
    high = [row for row in failures if row.get("confidence") == "high"]
    target = len(selected) + min(
        len(high), max(minimum, math.ceil(len(high) * fraction))
    )
    selected_ids = {str(row.get("run_id")) for row in selected}
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in high:
        if str(row.get("run_id")) in selected_ids:
            continue
        strata[
            (str(row.get("model")), str(row.get("track")), str(row.get("site")))
        ].append(row)
    rng = random.Random(seed)
    for values in strata.values():
        rng.shuffle(values)
    # Cover every observed stage before balancing model, track and website.
    for stage in PULSE_STAGES:
        for key in sorted(strata):
            candidate = next(
                (row for row in strata[key] if row.get("pulse_stage") == stage), None
            )
            if candidate is not None and len(selected) < target:
                selected.append(candidate)
                strata[key].remove(candidate)
                break
    keys = sorted(strata)
    while len(selected) < target and any(strata.values()):
        for key in keys:
            if strata[key] and len(selected) < target:
                selected.append(strata[key].pop())
    return selected


def select_judge_validation_sample(
    rows: Sequence[Mapping[str, Any]],
    source_groups: Mapping[str, str],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Choose calibration and held-out cases without sharing source recordings."""
    candidates = [
        dict(row) for row in rows if row["run_verdict"] == "valid_model_failure"
    ]
    missing = [
        row["run_id"] for row in candidates if not source_groups.get(row["run_id"])
    ]
    if missing:
        raise ValueError(
            f"validation sample requires source recording groups: {missing}"
        )
    rng = random.Random(seed)
    rng.shuffle(candidates)
    # Balance model/setting cells first, then task type, website, and existing labels.
    strata: dict[tuple[str, str], dict[tuple[str, str, str], list[dict[str, Any]]]] = {}
    for row in candidates:
        cell = (str(row.get("model", "unknown")), str(row.get("setting", "unknown")))
        subgroup = (str(row.get("track")), str(row.get("site")),
                    str(row.get("pulse_stage") or "Unknown"))
        strata.setdefault(cell, {}).setdefault(subgroup, []).append(row)
    cells = {}
    for cell, subgroups in strata.items():
        balanced = []
        while any(subgroups.values()):
            for subgroup in sorted(subgroups):
                if subgroups[subgroup]:
                    balanced.append(subgroups[subgroup].pop())
        cells[cell] = balanced
    ordered = []
    while any(cells.values()):
        for cell in sorted(cells):
            if cells[cell]:
                ordered.append(cells[cell].pop(0))
    group_sizes = Counter(source_groups[row["run_id"]] for row in ordered)
    heldout_anchors = []
    for stage in (*PULSE_STAGES, "Unknown"):
        stage_rows = [
            row for row in ordered if str(row.get("pulse_stage") or "Unknown") == stage
        ]
        if stage_rows:
            heldout_anchors.append(
                min(
                    stage_rows,
                    key=lambda row: group_sizes[source_groups[row["run_id"]]],
                )
            )
    heldout_groups = {source_groups[row["run_id"]] for row in heldout_anchors}
    calibration = []
    calibration_groups: set[str] = set()
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_model[str(row.get("model", "unknown"))].append(row)
    calibration_order = []
    while any(by_model.values()):
        for model in sorted(by_model):
            if by_model[model]:
                calibration_order.append(by_model[model].pop(0))
    for row in calibration_order:
        group = source_groups[row["run_id"]]
        if group in heldout_groups:
            continue
        excluded_count = sum(
            source_groups[item["run_id"]] in calibration_groups | {group}
            for item in ordered
        )
        remaining = [item for item in ordered
                     if source_groups[item["run_id"]] not in calibration_groups | {group}]
        remaining_cells = {(str(item.get("model", "unknown")),
                            str(item.get("setting", "unknown"))) for item in remaining}
        if len(ordered) - excluded_count < 40 or remaining_cells != set(strata):
            continue
        calibration.append(row["run_id"])
        calibration_groups.add(group)
        if len(calibration) == 10:
            break
    validation = list(
        dict.fromkeys(
            [
                *(row["run_id"] for row in heldout_anchors),
                *(
                    row["run_id"]
                    for row in ordered
                    if source_groups[row["run_id"]] not in calibration_groups
                ),
            ]
        )
    )[:40]
    if len(calibration) != 10 or len(validation) != 40:
        raise ValueError(
            "validation requires 10 calibration and 40 source-separated held-out cases"
        )
    selected = {*calibration, *validation}
    return {
        "calibration_run_ids": calibration,
        "validation_run_ids": validation,
        "source_groups": {run_id: source_groups[run_id] for run_id in sorted(selected)},
    }


def validate_judge_sample(selection: Mapping[str, Any], known_runs: set[str]) -> None:
    calibration = selection["calibration_run_ids"]
    validation = selection["validation_run_ids"]
    groups = selection["source_groups"]
    if len(calibration) != 10 or len(validation) != 40:
        raise ValueError(
            "Judge validation requires 10 calibration and 40 held-out cases"
        )
    selected = [*calibration, *validation]
    if len(set(selected)) != 50 or set(selected) - known_runs:
        raise ValueError(
            "Judge validation selection contains duplicate or unknown runs"
        )
    if any(not groups.get(run_id) for run_id in selected):
        raise ValueError("Judge validation selection requires source recording groups")
    if {groups[run_id] for run_id in calibration} & {
        groups[run_id] for run_id in validation
    }:
        raise ValueError("calibration and validation cannot share source recordings")


def reliability_report(
    ratings: Sequence[Mapping[str, Any]],
    automatic: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare independent human review; compute inter-rater metrics only with a pair."""

    for row in automatic:
        validate_attribution(row)
    automatic_run_ids = {str(row.get("run_id") or "") for row in automatic}
    by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_ratings: set[tuple[str, str]] = set()
    for rating in ratings:
        if not isinstance(rating, Mapping):
            raise ValueError("human ratings must be objects")
        run_id = str(rating.get("run_id") or "")
        rater_id = str(rating.get("rater_id") or "")
        if not run_id or run_id not in automatic_run_ids:
            raise ValueError(f"human rating references unknown run: {run_id!r}")
        if not rater_id:
            raise ValueError("human rating requires rater_id")
        key = (run_id, rater_id)
        if key in seen_ratings:
            raise ValueError(f"duplicate human rating: {run_id}/{rater_id}")
        seen_ratings.add(key)
        stage = rating.get("pulse_stage")
        if stage is not None and stage not in PULSE_STAGES:
            raise ValueError(f"invalid human pulse_stage: {stage!r}")
        by_run[run_id].append(rating)
    pairs = []
    for run_id, values in sorted(by_run.items()):
        by_rater = {str(value.get("rater_id") or ""): value for value in values}
        if len(by_rater) == 2:
            ordered = [by_rater[key] for key in sorted(by_rater)]
            pairs.append(
                (
                    run_id,
                    _rating_label(ordered[0]),
                    _rating_label(ordered[1]),
                )
            )
    labels = (*PULSE_STAGES, "Unknown", "TaskIssue", "InvalidRun")
    confusion = {left: {right: 0 for right in labels} for left in labels}
    for _, left, right in pairs:
        confusion[left][right] += 1
    total = len(pairs)
    agreements = sum(left == right for _, left, right in pairs)
    raw = agreements / total if total else None
    if raw is not None:
        left_counts = Counter(left for _, left, _ in pairs)
        right_counts = Counter(right for _, _, right in pairs)
        expected = sum(left_counts[label] * right_counts[label] for label in labels) / (
            total * total
        )
        kappa = (raw - expected) / (1 - expected) if expected < 1 else None
    else:
        kappa = None

    consensus = {run_id: left for run_id, left, right in pairs if left == right}
    auto_rows = {
        str(row.get("run_id")): row
        for row in automatic
        if row.get("confidence") == "high" and row.get("pulse_stage") in PULSE_STAGES
    }
    comparisons = [
        _rating_label(auto_rows[run_id]) == stage
        for run_id, stage in consensus.items()
        if run_id in auto_rows
    ]
    human_labels = {
        run_id: _rating_label(values[0])
        for run_id, values in by_run.items()
        if len(values) == 1
    }
    human_labels.update(consensus)
    human_comparisons = [
        _rating_label(auto_rows[run_id]) == label
        for run_id, label in human_labels.items()
        if run_id in auto_rows
    ]
    all_auto = {str(row["run_id"]): row for row in automatic}
    class_confusion = {left: {right: 0 for right in labels} for left in labels}
    exact_steps = []
    adjacent_steps = []
    for run_id, human_label in human_labels.items():
        automatic_row = all_auto[run_id]
        class_confusion[human_label][_rating_label(automatic_row)] += 1
        values = by_run[run_id]
        steps = {rating.get("first_decisive_step") for rating in values}
        if len(steps) == 1 and None not in steps:
            human_step = next(iter(steps))
            if type(human_step) is not int or human_step < 0:
                raise ValueError("human first_decisive_step must be non-negative")
            auto_step = automatic_row.get("first_decisive_step")
            exact_steps.append(auto_step == human_step)
            adjacent_steps.append(
                type(auto_step) is int and abs(auto_step - human_step) <= 1
            )
    class_total = sum(sum(counts.values()) for counts in class_confusion.values())
    f1_scores = {}
    for label in labels:
        actual = sum(class_confusion[label].values())
        predicted = sum(counts[label] for counts in class_confusion.values())
        if actual + predicted:
            f1_scores[label] = 2 * class_confusion[label][label] / (actual + predicted)
    return {
        "classification_macro_f1": sum(f1_scores.values()) / len(f1_scores)
        if f1_scores
        else None,
        "classification_f1_by_observed_label": f1_scores,
        "classification_cases": class_total,
        "classification_agreement": sum(
            class_confusion[label][label] for label in labels
        )
        / class_total
        if class_total
        else None,
        "automatic_confusion_matrix": class_confusion,
        "decisive_step_comparison": "explicit_human_steps_independent_of_category",
        "decisive_step_cases": len(exact_steps),
        "decisive_step_exact_agreement": sum(exact_steps) / len(exact_steps)
        if exact_steps
        else None,
        "decisive_step_within_one_agreement": sum(adjacent_steps) / len(adjacent_steps)
        if adjacent_steps
        else None,
        "high_confidence_error_rate": 1
        - sum(human_comparisons) / len(human_comparisons)
        if human_comparisons
        else None,
        "reviewed_cases": len(by_run),
        "automatic_vs_human_cases": len(human_comparisons),
        "automatic_vs_human_agreement": (
            sum(human_comparisons) / len(human_comparisons)
            if human_comparisons
            else None
        ),
        "double_rated_cases": total,
        "incomplete_cases": len(by_run) - total,
        "raw_agreement": raw,
        "cohen_kappa": kappa,
        "confusion_matrix": confusion,
        "automatic_vs_consensus_cases": len(comparisons),
        "automatic_vs_consensus_agreement": (
            sum(comparisons) / len(comparisons) if comparisons else None
        ),
    }


def _rating_label(row: Mapping[str, Any]) -> str:
    if row.get("run_verdict") in {"task_or_site_defect", "unknown", "benchmark_fault"}:
        return "TaskIssue"
    if row.get("run_verdict") == "invalid_run":
        return "InvalidRun"
    stage = row.get("pulse_stage")
    return str(stage) if stage in PULSE_STAGES else "Unknown"


def _judge_prediction_source(row: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a frozen prediction, independently of later qualification or review."""
    prediction = deepcopy(dict(row))
    for field in (
        "pulse_stage",
        "mechanism",
        "first_decisive_step",
        "candidate_stage",
        "candidate_mechanism",
        "candidate_step",
        "candidate_source",
        "candidate_confidence",
        "candidate_unknown_boundary",
        "attribution_method",
        "rule_evidence",
        "qualification_evidence",
    ):
        prediction[field] = None
    prediction.update(
        run_verdict="valid_model_failure",
        observable_facts=[],
        secondary_symptoms=[],
        confidence="low",
        attribution_status="coding_agent_pending",
        review_required=False,
    )
    return prediction


def judge_validation_result(
    evidence: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], root: Path | None
) -> dict[str, Any] | None:
    """Evaluate only the frozen Judge's independent held-out judgments."""
    selection = evidence.get("judge_validation")
    if selection is None:
        return None
    validate_judge_sample(selection, {row["run_id"] for row in rows})
    ratings = evidence.get("ratings", [])
    heldout_ids = set(selection["validation_run_ids"])
    validation_assessments = selection["assessments"]
    for assessment in validation_assessments:
        if (
            assessment.get("judge_model") != selection["model"]
            or assessment.get("prompt_version") != selection["prompt_version"]
            or assessment.get("reviewer_kind") != "llm_judge"
        ):
            raise ValueError("Judge validation model/prompt_version mismatch")
    validation_ids = {item["run_id"] for item in validation_assessments}
    selected_ids = {*selection["calibration_run_ids"], *heldout_ids}
    recheck_ids = {item["run_id"] for item in selection.get("rechecks", [])}
    if (validation_ids | recheck_ids) - selected_ids:
        raise ValueError("Judge validation references unregistered runs")
    heldout_assessments = [
        item for item in validation_assessments if item["run_id"] in heldout_ids
    ]
    judged_ids = {item["run_id"] for item in heldout_assessments}
    heldout_rows = apply_assessments(
        [_judge_prediction_source(row) for row in rows if row["run_id"] in judged_ids],
        heldout_assessments,
        root,
    )
    heldout_ratings = [
        (rating.get("review_history") or [rating])[0]
        for rating in ratings if rating["run_id"] in judged_ids
    ]
    judge_validation = reliability_report(heldout_ratings, heldout_rows)
    judge_validation.update(
        model=selection["model"],
        prompt_version=selection["prompt_version"],
        automatic_cases=len(judged_ids),
        rating_basis="first_independent_submission",
        dispute_review_summary={
            "reviewed": sum(
                "dispute_review" in item for item in validation_assessments
            ),
            "verdict_counts": dict(
                Counter(
                    item["dispute_review"]["verdict"]
                    for item in validation_assessments
                    if "dispute_review" in item
                )
            ),
        },
        recheck_cases=len(selection.get("rechecks", [])),
        output_coverage={
            "selected_cases": len(selected_ids),
            "original_output_cases": len(validation_ids),
            "rechecked_only_cases": len(recheck_ids - validation_ids),
            "covered_cases": len(validation_ids | recheck_ids),
            "missing_run_ids": sorted(selected_ids - validation_ids - recheck_ids),
        },
    )
    judge_validation["sample_complete"] = (
        judge_validation["reviewed_cases"] == 40 and len(judged_ids) == 40
    )
    agreement = judge_validation["automatic_vs_human_agreement"]
    judge_validation["accepted"] = (
        judge_validation["sample_complete"]
        and judge_validation["automatic_vs_human_cases"] >= 10
        and agreement is not None
        and agreement >= 0.85
    )
    return judge_validation
