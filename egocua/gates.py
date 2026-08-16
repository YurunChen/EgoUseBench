from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .checker import check_run
from .environment import load_site_data, site_entity_ids
from .experiment import ExperimentPlan, PRIMARY_CONDITION, SUPPORTED_ARMS
from .lineage import LineageError, validate_source_lineage
from .reviews import ReviewAuditError, validate_construction_evidence
from .run_ledger import asset_hashes, canonical_sha256
from .schema import (
    ValidationError,
    acceptable_states_of,
    goal_for_arm,
    selected_entity_sets_of,
    validate_task,
)
from .workspace import resolve_within


def load_validation_evidence(path: str | Path) -> dict[str, dict[str, Any]]:
    source = Path(path)
    payload = json.loads(source.read_text())
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValidationError(f"{source}: validation evidence schema_version must be 1")
    records = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(records, dict):
        raise ValidationError(f"{source}: validation evidence must contain a tasks object")
    return records


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValidationError(f"{context}: missing {key!r}")
    return mapping[key]


def _canonical_states(
    options: Any, context: str,
) -> frozenset[tuple[tuple[str, ...], tuple[tuple[str, str], ...], bool]]:
    states = acceptable_states_of({"acceptable_states": options}, context)
    return frozenset(
        (
            tuple(sorted(state["selected_entity_ids"])),
            tuple(sorted(state["fields"].items())),
            state["submission_completed"],
        )
        for state in states
    )


def _construction_decision(
    task: dict[str, Any], review: dict[str, Any], context: str,
) -> tuple[str, frozenset[tuple[tuple[str, ...], tuple[tuple[str, str], ...], bool]]]:
    if review.get("decision") != "admit":
        raise ValidationError(f"{context}: construction decision must be 'admit'")
    if review.get("source_evidence_supported") is not True:
        raise ValidationError(f"{context}: source evidence is not independently supported")
    if review.get("intervention_isolated") is not True:
        raise ValidationError(f"{context}: intervention is not independently isolated")
    granularity = review.get("target_granularity")
    if granularity not in {"category", "brand", "variant", "exact_instance"}:
        raise ValidationError(f"{context}: invalid target_granularity")
    unresolved = review.get("unresolved")
    if not isinstance(unresolved, list) or unresolved:
        raise ValidationError(f"{context}: construction review has unresolved issues")
    states = _canonical_states(
        review.get("acceptable_states"), f"{context}/acceptable_states"
    )
    expected = _canonical_states(
        task["goal"].get("acceptable_states"), f"{task['id']}/goal"
    )
    if states != expected:
        raise ValidationError(f"{context}: reviewed acceptable states disagree with the task goal")
    return str(granularity), states


def _validate_construction(
    task: dict[str, Any], evidence: dict[str, Any], *, root: str | Path,
) -> set[str]:
    task_id = task["id"]
    construction = _require(evidence, "construction", task_id)
    try:
        validate_construction_evidence(task, construction, root=root)
    except ReviewAuditError as exc:
        raise ValidationError(str(exc)) from exc
    reviews = construction.get("reviews")
    if not isinstance(reviews, list) or len(reviews) < 2:
        raise ValidationError(f"{task_id}: validation requires two construction reviews")
    raw_reviewer_ids = [review.get("reviewer") for review in reviews if isinstance(review, dict)]
    if len(raw_reviewer_ids) != len(reviews) or any(not str(value).strip() for value in raw_reviewer_ids):
        raise ValidationError(f"{task_id}: every construction review needs a reviewer id")
    reviewer_ids = [str(value) for value in raw_reviewer_ids]
    if len(set(reviewer_ids)) != len(reviewer_ids):
        raise ValidationError(f"{task_id}: construction reviewers must be distinct")
    decisions = [
        _construction_decision(task, review, f"{task_id}/construction/{review['reviewer']}")
        for review in reviews
    ]
    if len(set(decisions)) != 1:
        raise ValidationError(f"{task_id}: construction reviewers disagree")
    return set(reviewer_ids)


def _replay_run(
    task: dict[str, Any], arm: str, run_artifact: Any, *, root: str | Path,
    site_data: dict[str, Any], context: str, require_valid: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        artifact = resolve_within(root, run_artifact, context=context, must_exist=False)
        record_path = artifact / "run.json" if artifact.is_dir() else artifact
        if not record_path.is_file():
            raise FileNotFoundError(record_path)
        record = json.loads(record_path.read_text())
    except (TypeError, ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise ValidationError(f"{context}: invalid run artifact: {exc}") from exc
    if not isinstance(record, dict):
        raise ValidationError(f"{context}: run artifact must contain one object")
    if (
        record.get("task_id") != task["id"]
        or record.get("track") != task["track"]
        or record.get("arm") != arm
    ):
        raise ValidationError(f"{context}: run artifact task or arm identity mismatch")
    meta = record.get("meta", {})
    expected_hashes = {
        "task_sha256": canonical_sha256(task),
        "site_data_sha256": canonical_sha256(site_data),
        "asset_sha256": asset_hashes(task, root),
    }
    if any(meta.get(key) != value for key, value in expected_hashes.items()):
        raise ValidationError(f"{context}: run artifact hashes do not match")
    checked = check_run(task, record)
    if require_valid and checked["run_valid"] is not True:
        raise ValidationError(f"{context}: referenced run artifact is protocol-invalid")
    return record, checked


def _success_required(task: dict[str, Any], arm: str) -> bool:
    if task["track"] == "refresh" and task["refresh"]["policy"] == "replace":
        return arm != "stale"
    if task["track"] == "clarify":
        return arm != "forced_execute"
    return True


def _validate_visual_necessity(
    task: dict[str, Any], human: dict[str, Any], evaluators: set[str],
    site_data: dict[str, Any],
) -> None:
    task_id = task["id"]
    visual = _require(human, "visual_necessity", f"{task_id}/human")
    reference_arm = visual.get("reference_arm")
    expected_reference = PRIMARY_CONDITION[task["track"]]
    if reference_arm != expected_reference:
        raise ValidationError(f"{task_id}: visual-necessity reference arm must be {expected_reference}")
    reviews = visual.get("reviews")
    if not isinstance(reviews, list) or len(reviews) < 2:
        raise ValidationError(f"{task_id}: visual necessity requires two reviews")
    reviewer_ids = {review.get("reviewer") for review in reviews if isinstance(review, dict)}
    if len(reviewer_ids) != len(reviews) or not reviewer_ids <= evaluators:
        raise ValidationError(f"{task_id}: visual-necessity reviewers must be distinct human evaluators")
    entity_ids = site_entity_ids(site_data)
    text_choices: list[tuple[str, ...]] = []
    full_choices: list[tuple[str, ...]] = []
    acceptable = {
        tuple(sorted(option))
        for option in selected_entity_sets_of(
            goal_for_arm(task, reference_arm), f"{task_id}/{reference_arm}"
        )
    }
    for review in reviews:
        text_ids = review.get("text_only_choice_ids")
        full_ids = review.get("full_evidence_choice_ids")
        if (
            not isinstance(text_ids, list) or not text_ids
            or not isinstance(full_ids, list) or not full_ids
            or not set(text_ids + full_ids) <= entity_ids
        ):
            raise ValidationError(f"{task_id}: visual-necessity choices must be non-empty entity IDs")
        text_choices.append(tuple(sorted(text_ids)))
        full_choice = tuple(sorted(full_ids))
        if full_choice not in acceptable:
            raise ValidationError(f"{task_id}: full evidence does not identify an acceptable target set")
        full_choices.append(full_choice)
    if len(set(text_choices)) == 1 and len(text_choices[0]) == 1:
        raise ValidationError(f"{task_id}: text-only evidence stably identifies one target")
    if len(set(full_choices)) != 1:
        raise ValidationError(f"{task_id}: full-evidence reviewers disagree on the target")


def _validate_counterfactual_checks(
    task: dict[str, Any], human: dict[str, Any], evaluators: set[str],
) -> None:
    task_id = task["id"]
    checks = human.get("counterfactual_checks")
    if not isinstance(checks, list) or len(checks) < 2:
        raise ValidationError(f"{task_id}: two counterfactual reviews are required")
    reviewer_ids = {check.get("reviewer") for check in checks if isinstance(check, dict)}
    if len(reviewer_ids) != len(checks) or not reviewer_ids <= evaluators:
        raise ValidationError(f"{task_id}: counterfactual reviewers must be distinct human evaluators")
    expected: dict[str, bool]
    if task["track"] == "graft":
        expected = {"b_is_essential": True}
    elif task["track"] == "refresh":
        expected = {
            "v0_v1_identifies_latest": True,
            "v0_only_identifies_latest": task["refresh"]["policy"] == "retain",
        }
    else:
        expected = {"ambiguous_without_answer": True, "unique_with_answer": True}
    for check in checks:
        if any(check.get(key) is not value for key, value in expected.items()):
            raise ValidationError(f"{task_id}: track-specific counterfactual check failed")


def _validate_human(
    task: dict[str, Any], evidence: dict[str, Any], *, root: str | Path,
    site_data: dict[str, Any], constructor_ids: set[str], expected_arms: set[str],
) -> set[str]:
    task_id = task["id"]
    human = _require(evidence, "human", task_id)
    if human.get("exact_harness") is not True:
        raise ValidationError(f"{task_id}: human validation must use the exact benchmark harness")
    arm_runs = human.get("arm_runs")
    if not isinstance(arm_runs, dict) or set(arm_runs) != expected_arms:
        raise ValidationError(f"{task_id}: human validation must cover every experiment arm")
    all_evaluators: set[str] = set()
    for arm in expected_arms:
        runs = arm_runs[arm]
        if not isinstance(runs, list) or len(runs) < 2:
            raise ValidationError(f"{task_id}/{arm}: needs two independent human runs")
        arm_evaluators: set[str] = set()
        for run in runs:
            if not isinstance(run, dict):
                raise ValidationError(f"{task_id}/{arm}: human run entry must be an object")
            evaluator = str(run.get("evaluator", "")).strip()
            duration = run.get("completion_seconds")
            if not evaluator or evaluator in arm_evaluators:
                raise ValidationError(f"{task_id}/{arm}: human evaluators must be distinct")
            if evaluator in constructor_ids:
                raise ValidationError(f"{task_id}: human evaluators must be independent of constructors")
            if not isinstance(duration, (int, float)) or duration <= 0:
                raise ValidationError(f"{task_id}/{arm}: completion_seconds must be positive")
            record, checked = _replay_run(
                task, arm, run.get("run_artifact"), root=root, site_data=site_data,
                context=f"{task_id}/{arm}/human/{evaluator}",
            )
            if record.get("meta", {}).get("agent") != "human":
                raise ValidationError(f"{task_id}/{arm}: human artifact must declare agent='human'")
            if record.get("meta", {}).get("operator_id") != evaluator:
                raise ValidationError(
                    f"{task_id}/{arm}: human artifact operator_id must match its evaluator"
                )
            if checked["submission_completed"] is not True:
                raise ValidationError(f"{task_id}/{arm}: human run did not reach a terminal submission")
            if _success_required(task, arm) and checked["e2e_success"] is not True:
                raise ValidationError(f"{task_id}/{arm}: information-sufficient human run must succeed")
            if task["track"] == "clarify" and arm == "interactive" and not (
                checked["valid_question"]
                and checked["timely_question"]
                and checked["answer_used"]
            ):
                raise ValidationError(f"{task_id}/interactive: human must ask once and use the answer")
            if task["track"] == "clarify" and arm == "oracle" and checked["asked"]:
                raise ValidationError(f"{task_id}/oracle: human asked an unnecessary question")
            arm_evaluators.add(evaluator)
            all_evaluators.add(evaluator)
    if len(all_evaluators) < 2:
        raise ValidationError(f"{task_id}: validation requires two independent human evaluators")
    _validate_visual_necessity(task, human, all_evaluators, site_data)
    _validate_counterfactual_checks(task, human, all_evaluators)
    return all_evaluators


def _validate_evaluator(
    task: dict[str, Any], evidence: dict[str, Any], constructor_ids: set[str],
    human_ids: set[str], *, root: str | Path, site_data: dict[str, Any],
) -> None:
    task_id = task["id"]
    evaluator = _require(evidence, "evaluator", task_id)
    reviewers = evaluator.get("reviewers")
    if not isinstance(reviewers, list) or not reviewers or len(set(reviewers)) != len(reviewers):
        raise ValidationError(f"{task_id}: release requires an evaluator reviewer")
    if set(reviewers) & (constructor_ids | human_ids):
        raise ValidationError(f"{task_id}: evaluator reviewer must be independent")
    cases = evaluator.get("replay_cases")
    expected: dict[str, dict[str, Any]] = {
        "alternative_correct_path": {"run_valid": True, "e2e_success": True},
        "extra_item_rejected": {"run_valid": True, "e2e_success": False},
        "duplicate_quantity_rejected": {"run_valid": True, "e2e_success": False},
    }
    if task["track"] in {"graft", "refresh"}:
        expected.update({
            "early_submission_rejected": {"run_valid": False},
            "missing_event_rejected": {"run_valid": False},
        })
    else:
        expected.update({
            "invalid_slot_detected": {"valid_question": False},
            "second_question_limit_detected": {"question_limit_exceeded": True},
            "late_question_detected": {"timely_question": False},
            "ignored_answer_detected": {"answer_used": False},
        })
    if not isinstance(cases, dict) or set(cases) != set(expected):
        raise ValidationError(f"{task_id}: evaluator replay cases are incomplete")
    for case, expected_result in expected.items():
        entry = cases[case]
        if not isinstance(entry, dict) or not str(entry.get("arm", "")).strip():
            raise ValidationError(f"{task_id}/{case}: evaluator replay needs an arm and artifact")
        arm = entry["arm"]
        if arm not in SUPPORTED_ARMS[task["track"]]:
            raise ValidationError(f"{task_id}/{case}: invalid evaluator replay arm")
        record, checked = _replay_run(
            task, arm, entry.get("run_artifact"), root=root, site_data=site_data,
            context=f"{task_id}/evaluator/{case}", require_valid=False,
        )
        if any(checked.get(key) != value for key, value in expected_result.items()):
            raise ValidationError(f"{task_id}/{case}: evaluator replay did not produce the expected result")
        selected = record.get("final_state", {}).get("selected_entity_ids", [])
        options = selected_entity_sets_of(goal_for_arm(task, arm), f"{task_id}/{case}")
        if case == "extra_item_rejected" and not any(
            set(option) < set(selected) and len(selected) > len(option) for option in options
        ):
            raise ValidationError(f"{task_id}/{case}: replay is not an extra-item case")
        if case == "duplicate_quantity_rejected" and not (
            len(selected) != len(set(selected))
            and any(set(selected) == set(option) for option in options)
        ):
            raise ValidationError(f"{task_id}/{case}: replay is not a duplicate-quantity case")


def _validate_model(
    task: dict[str, Any], evidence: dict[str, Any], plan: ExperimentPlan, *, root: str | Path,
    site_data: dict[str, Any],
) -> None:
    task_id = task["id"]
    model = _require(evidence, "model", task_id)
    if model.get("execution_mode") != "real_model":
        raise ValidationError(f"{task_id}: model validation must use a real model")
    for key in ("provider", "model", "model_version", "adapter"):
        _require(model, key, f"{task_id}/model")
    expected_runtime = {
        "adapter": plan.adapter,
        "model": plan.model,
        "model_version": plan.model_version,
        "max_steps": plan.max_steps,
    }
    mismatches = {
        key: (model.get(key), expected)
        for key, expected in expected_runtime.items()
        if model.get(key) != expected
    }
    if mismatches:
        raise ValidationError(f"{task_id}: model validation disagrees with experiment plan: {mismatches}")
    if model.get("exact_harness") is not True or model.get("vision_input_used") is not True:
        raise ValidationError(f"{task_id}: model validation must use the exact visual harness")
    if model.get("used_for_task_selection") is not False:
        raise ValidationError(f"{task_id}: model results must not select release tasks")
    issues = model.get("unresolved_operational_issues")
    if not isinstance(issues, list) or issues:
        raise ValidationError(f"{task_id}: model validation has unresolved operational issues")
    arm_runs = _require(model, "arm_runs", f"{task_id}/model")
    expected_arms = set(plan.arms_for(task["track"]))
    if set(arm_runs) != expected_arms:
        raise ValidationError(f"{task_id}: model validation must cover every experiment arm")
    for arm in expected_arms:
        runs = arm_runs[arm]
        if not isinstance(runs, list) or not runs:
            raise ValidationError(f"{task_id}/{arm}: missing real-model run")
        verified = False
        for run in runs:
            if not isinstance(run, dict) or run.get("protocol_valid") is not True:
                continue
            if not isinstance(run.get("e2e_success"), bool) or not run.get("run_artifact"):
                continue
            record, checked = _replay_run(
                task, arm, run["run_artifact"], root=root, site_data=site_data,
                context=f"{task_id}/{arm}/model",
            )
            meta = record.get("meta", {})
            expected_meta = {
                "agent": plan.adapter,
                "model": plan.model,
                "model_version": plan.model_version,
                "max_steps": plan.max_steps,
            }
            if any(meta.get(key) != value for key, value in expected_meta.items()):
                raise ValidationError(f"{task_id}/{arm}: run artifact identity or hashes do not match")
            if checked["e2e_success"] != run["e2e_success"]:
                raise ValidationError(f"{task_id}/{arm}: stored model outcome disagrees with checker replay")
            verified = True
            break
        if not verified:
            raise ValidationError(f"{task_id}/{arm}: needs one protocol-valid real-model run")


def validate_task_for_model_run(
    task: dict[str, Any], evidence: dict[str, Any], plan: ExperimentPlan, *, root: str | Path,
) -> None:
    site_data = load_site_data(root, task["site"], require_assets=True)
    validate_task(task, site_data, root=root, require_assets=True)
    try:
        validate_source_lineage(task, root=root)
    except LineageError as exc:
        raise ValidationError(str(exc)) from exc
    if evidence.get("status") not in {"human_verified", "approved"}:
        raise ValidationError(f"{task['id']}: model run requires human-verified task evidence")
    constructor_ids = _validate_construction(task, evidence, root=root)
    human_ids = _validate_human(
        task, evidence, root=root, site_data=site_data, constructor_ids=constructor_ids,
        expected_arms=set(plan.arms_for(task["track"])),
    )
    _validate_evaluator(
        task, evidence, constructor_ids, human_ids, root=root, site_data=site_data,
    )


def validate_task_for_human_run(
    task: dict[str, Any], evidence: dict[str, Any], *, root: str | Path,
) -> None:
    """Allow exact-harness collection only after independent construction admission."""

    site_data = load_site_data(root, task["site"], require_assets=True)
    validate_task(task, site_data, root=root, require_assets=True)
    try:
        validate_source_lineage(task, root=root)
    except LineageError as exc:
        raise ValidationError(str(exc)) from exc
    if evidence.get("status") not in {"constructed", "human_verified", "approved"}:
        raise ValidationError(
            f"{task['id']}: human run requires construction-admitted task evidence"
        )
    _validate_construction(task, evidence, root=root)


def validate_release(
    tasks: Iterable[dict[str, Any]], evidence_by_task: Mapping[str, dict[str, Any]],
    plan: ExperimentPlan, *, root: str | Path,
) -> dict[str, int]:
    counts = {"graft": 0, "refresh": 0, "clarify": 0}
    seen: set[str] = set()
    for task in tasks:
        task_id = task["id"]
        if task_id in seen:
            raise ValidationError(f"duplicate task id: {task_id}")
        seen.add(task_id)
        evidence = evidence_by_task.get(task_id)
        if evidence is None:
            raise ValidationError(f"{task_id}: missing external validation evidence")
        if evidence.get("status") != "approved":
            raise ValidationError(f"{task_id}: release evidence status must be 'approved'")
        site_data = load_site_data(root, task["site"], require_assets=True)
        validate_task(task, site_data, root=root, require_assets=True)
        try:
            validate_source_lineage(task, root=root)
        except LineageError as exc:
            raise ValidationError(str(exc)) from exc
        constructor_ids = _validate_construction(task, evidence, root=root)
        human_ids = _validate_human(
            task, evidence, root=root, site_data=site_data, constructor_ids=constructor_ids,
            expected_arms=set(plan.arms_for(task["track"])),
        )
        _validate_evaluator(
            task, evidence, constructor_ids, human_ids, root=root, site_data=site_data,
        )
        _validate_model(task, evidence, plan, root=root, site_data=site_data)
        counts[task["track"]] += 1
    unexpected = set(evidence_by_task) - seen
    if unexpected:
        raise ValidationError(f"validation evidence contains unknown tasks: {sorted(unexpected)[:5]}")
    return dict(sorted(counts.items()))
