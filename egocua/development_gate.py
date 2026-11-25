"""Development trial admission from immutable runs and recorded trajectory reviews."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Any

from .checker import check_run, composite_event_coverage, rescore_run
from .environment import load_site_data, task_public_site_data
from .execution import RELEASE_SEMANTICS_VERSION
from .experiment import FORMAL_PLAN, TASK_REVIEW_PLAN, ExperimentPlan, load_experiment_plan
from .qualification import trace_has_model_response, trace_has_provider_refusal
from .run_ledger import (
    SITE_RUNTIME_CONTRACTS,
    MODEL_RUN_TRANSPORTS,
    asset_hashes,
    canonical_sha256,
    load_run_record,
    task_sha256,
    task_site_execution_sha256,
)
from .run_materials import (
    clarified_instruction_snapshot,
    execution_inputs,
    path_independent_inputs,
    material_mismatch,
)
from .selection import load_selection, task_ids
from .update_count import VARIANT_SUFFIX, frozen_original
from .sources import source_record, load_dossier
from .sampling_media import prepare_session_tasks
from .lineage import source_media_segments
from .synthesis_reports import REPORTS
from .timeline import ACTION_ALIGNED_REPLAY, COMPOSITE_TRACK, Condition, Timeline, is_update_event
from .workspace import read_json_strict, resolve_within

REVIEWS = Path("data/reviews/task-qualification.json")
REPORT = Path("artifacts/reports/task-qualification.json")
POLICY = "reviewed-development-trial-v1"


@dataclass(frozen=True)
class DevelopmentDecision:
    action: str | None
    reason: str


def _read_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = read_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected an object")
    return payload


def material_identity(task: dict[str, Any], root: Path) -> dict[str, str]:
    """Bind the policy boundary to actual inputs, not filenames or creation dates."""
    plan = load_experiment_plan(root / TASK_REVIEW_PLAN)
    inputs, paths = path_independent_inputs(execution_inputs(task, plan, "online"))
    payload = json.loads(inputs)
    payload["task"].pop("truth", None)
    for key in ("adapter", "model", "model_version"):
        payload["runtime"].pop(key)
    payload.pop("scheduler_seed")
    hashes = asset_hashes(task, root)
    missing = [path for path in paths if path not in hashes]
    if missing:
        raise ValueError(f"{task['id']}: missing visual inputs: {missing}")
    source = source_record(task, root=root).get("source", {})
    source_identity = {
        key: source[key]
        for key in (
            "media_id",
            "original_video_id",
            "media_start_seconds",
            "media_sha256",
        )
        if key in source
    }
    if "media_segments" in source:
        source_identity["media_segments"] = [
            asdict(segment) for segment in source_media_segments(source, context=task["id"])
        ]
    _, dossier = load_dossier(task, root=root)
    source_identity["dataset"] = dossier.get("dataset", {}).get("name")
    site = str(task["site"])
    return {
        "execution_inputs": canonical_sha256(payload),
        "visual_contents": canonical_sha256([hashes[path] for path in paths]),
        "public_site": canonical_sha256(
            {
                "data": task_public_site_data(task, load_site_data(root, site)),
                "runtime_contract": SITE_RUNTIME_CONTRACTS[site],
                "release_semantics": RELEASE_SEMANTICS_VERSION,
            }
        ),
        "source_identity": canonical_sha256(source_identity),
    }


def grandfathered(task: dict[str, Any], root: Path, reviews: Mapping[str, Any]) -> bool:
    review = reviews.get("tasks", {}).get(str(task["id"]), {})
    if review.get("verdict") == "task_or_site_defect" and review.get("evidence"):
        return False
    baseline = reviews.get("legacy_baseline")
    if baseline is None:
        return False
    if not isinstance(baseline, Mapping) or baseline.get("policy") != POLICY:
        raise ValueError("unsupported development legacy baseline")
    rows = baseline.get("tasks")
    if not isinstance(rows, Mapping):
        raise ValueError("development legacy baseline tasks must be an object")
    registered = rows.get(str(task["id"]))
    if registered is None:
        return False
    if not isinstance(registered, Mapping) or set(registered) != {
        "execution_inputs",
        "visual_contents",
        "public_site",
        "source_identity",
    }:
        raise ValueError(f"{task['id']}: invalid development legacy baseline")
    reviewed_task = clarified_instruction_snapshot(task, review) or task
    return registered == material_identity(reviewed_task, root)


def record_mismatch(
    task: dict[str, Any], record: dict[str, Any], root: Path
) -> str | None:
    """Check real released inputs before rescoring a different hidden answer."""
    return _record_mismatch(task, record, root, DevelopmentEvidence(root).trial_plan(task))


def _record_mismatch(
    task: dict[str, Any], record: dict[str, Any], root: Path, plan: ExperimentPlan
) -> str | None:
    if record.get("task_id") != task["id"]:
        return "wrong_task"
    meta = record.get("meta", {})
    runtime = meta.get("runtime", {})
    if meta.get("max_steps") != plan.max_steps:
        return "wrong_execution_budget"
    condition = Condition.from_dict(record["condition"])
    if replace(condition, name="online") != plan.condition("online"):
        return "wrong_condition"
    snapshot = record.get("task_snapshot")
    if not isinstance(snapshot, dict):
        return "missing_task_snapshot"
    if (
        meta.get("agent") not in MODEL_RUN_TRANSPORTS
        or not meta.get("model")
        or not meta.get("model_version")
    ):
        return "missing_model_identity"
    if meta.get("transport") != MODEL_RUN_TRANSPORTS[meta["agent"]]:
        return "wrong_transport"
    if runtime.get("release_trigger", "timeline_step") != plan.release_trigger:
        return "wrong_release_trigger"
    if plan.release_trigger == "first_native_commitment" and task["track"] != "clarify":
        if runtime.get("release_semantics") != RELEASE_SEMANTICS_VERSION:
            return "legacy_release_semantics"
    if (
        meta.get("default_release_step") != plan.default_release_step
        or meta.get("reference_steps") != plan.reference_steps
    ):
        return "wrong_release_schedule"
    if trace_has_provider_refusal(record.get("trace")) or not trace_has_model_response(
        record.get("trace")
    ):
        return "invalid_model_trace"
    for field in ("pre_release_max_steps", "post_release_max_steps"):
        if runtime.get(field) != getattr(plan, field):
            return "wrong_execution_budget"
    if condition.frame_delivery_steps is not None:
        task = prepare_session_tasks(
            [task], root, float(condition.frame_interval_seconds)
        )[0]
    mismatch = material_mismatch(task, record, plan, root=root)
    if mismatch:
        return mismatch
    if check_run(snapshot, record).get("run_valid") is not True:
        return "invalid_run"
    return None


def inherit_unchanged_review(
    task: dict[str, Any],
    record: Mapping[str, Any],
    review: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Reuse a bound run review after input and scoring facts have been verified."""
    snapshot = record.get("task_snapshot")
    if not isinstance(snapshot, dict) or not review.get("run_artifact"):
        return review
    previous_scoring = (
        snapshot.get("goal"),
        snapshot.get("graft", {}).get("initial_goal"),
        snapshot.get("truth"),
    )
    current_scoring = (
        task.get("goal"),
        task.get("graft", {}).get("initial_goal"),
        task.get("truth"),
    )
    if previous_scoring != current_scoring:
        return review
    return {
        **review,
        "task_sha256": task_sha256(task),
        "inherited_review_task_sha256": review.get("task_sha256"),
    }


def rescore_development_run(task: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Use the verified replay schedule with the current task's answer key."""
    snapshot = record["task_snapshot"]
    if "visual_session" in snapshot:
        task = {**task, "visual_session": snapshot["visual_session"]}
    return rescore_run(task, record)


def _review_matches(
    review: Mapping[str, Any], task: dict[str, Any], checked: Mapping[str, Any]
) -> bool:
    evidence = review.get("evidence")
    if review.get("task_sha256") != task_sha256(task) or not review.get("reviewer"):
        return False
    if (
        not isinstance(evidence, list)
        or not evidence
        or not all(isinstance(item, str) and item.strip() for item in evidence)
    ):
        return False
    expected = (
        "valid_model_success"
        if checked.get("e2e_success") is True
        else "valid_model_failure"
    )
    return review.get("verdict") in {expected, "task_or_site_defect", "invalid_run"}


def retained_development_evidence(
    task: dict[str, Any], root: Path, reviews: Mapping[str, Any]
) -> bool:
    """Keep a named affected-scope review as task evidence, never as a new model score."""
    review = reviews.get("tasks", {}).get(str(task["id"]), {})
    task = clarified_instruction_snapshot(task, review) or task
    reuse = review.get("evidence_reuse")
    if not reuse or review.get("task_sha256") != task_sha256(task):
        return False
    if reuse.get("scope") != "task_validation_only":
        raise ValueError("unsupported development evidence reuse scope")
    if not review.get("reviewer") or not all(
        isinstance(reuse.get(key), list)
        and reuse[key]
        and all(isinstance(item, str) and item.strip() for item in reuse[key])
        for key in ("changes", "retained_evidence")
    ):
        raise ValueError(
            "development evidence reuse requires a named, specific impact review"
        )
    record = load_run_record(
        resolve_within(
            root,
            review["run_artifact"],
            context="retained development run",
            must_exist=False,
        )
    )
    snapshot = record["task_snapshot"]
    if any(
        task[key] != snapshot[key]
        for key in ("id", "track", "site", "instructions", "visual_packets")
    ):
        return False
    if task.get("truth", {}).get("site_scope_entity_ids") != snapshot.get(
        "truth", {}
    ).get("site_scope_entity_ids"):
        return False
    checked = rescore_development_run(task, record)
    expected = (
        "valid_model_success" if checked.get("e2e_success") else "valid_model_failure"
    )
    return (
        checked.get("run_valid") is True
        and trace_has_model_response(record.get("trace"))
        and not trace_has_provider_refusal(record.get("trace"))
        and review.get("verdict") == expected
    )


class DevelopmentEvidence:
    """Read development evidence once for one status or qualification request."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    @cached_property
    def reviews(self) -> dict[str, Any]:
        return _read_optional(self.root / REVIEWS)

    @cached_property
    def report(self) -> dict[str, Any]:
        return _read_optional(self.root / REPORT)

    @cached_property
    def smoke(self) -> dict[str, Any]:
        return read_json_strict(self.root / REPORTS["smoke"].path)

    @cached_property
    def formal_plan(self) -> ExperimentPlan:
        return load_experiment_plan(self.root / FORMAL_PLAN)

    @cached_property
    def historical_plan(self) -> ExperimentPlan:
        return load_experiment_plan(self.root / TASK_REVIEW_PLAN)

    @cached_property
    def formal_members(self) -> frozenset[str]:
        return frozenset(task_ids(load_selection(self.root)))

    def _historical_admission_plan(
        self, task_id: str, record: Mapping[str, Any]
    ) -> ExperimentPlan | None:
        recorded = Condition.from_dict(record["condition"])
        if recorded.batch_seconds is None:
            return self.historical_plan
        current = self.formal_plan
        online = current.condition("online")
        if (
            task_id not in self.formal_members
            or recorded.replay_timing is not None
            or online.replay_timing != ACTION_ALIGNED_REPLAY
        ):
            return None
        return replace(
            current,
            conditions={
                **current.conditions,
                "online": replace(online, replay_timing=None),
            },
        )

    def inherited_admission(self, task: dict[str, Any]) -> bool:
        """Keep task-quality evidence only when replay introduces no open risk."""
        condition = self.formal_plan.condition("online")
        if condition.frame_delivery_steps is not None:
            stream = source_record(task, root=self.root).get("stream", {})
            if not stream.get("session_seconds") or not stream.get("review_evidence"):
                return False
        return self._historically_validated(task)

    def _historically_validated(self, task: dict[str, Any]) -> bool:
        """Read existing admission, named reuse, or an unchanged reviewed run."""
        review = self.reviews.get("tasks", {}).get(str(task["id"]), {})
        if review.get("verdict") == "task_or_site_defect" and review.get("evidence"):
            return False
        if (
            grandfathered(task, self.root, self.reviews)
            or retained_development_evidence(task, self.root, self.reviews)
        ) and self.replay_budget_gap(task) is None:
            return True
        operation_task = task
        artifact = review.get("run_artifact")
        if not artifact or review.get("verdict") not in {
            "valid_model_success",
            "valid_model_failure",
        }:
            return False
        try:
            path = resolve_within(
                self.root, artifact, context="historical development run", must_exist=False
            )
            record = load_run_record(path)
            plan = self._historical_admission_plan(task["id"], record)
            if plan is None:
                return False
            budget = record.get("meta", {}).get("max_steps")
            if budget != plan.max_steps:
                history = record.get("trace", {}).get("history") or [{}]
                last_output = history[-1].get("model_output") or {}
                if (
                    type(budget) is not int
                    or record["steps"] >= min(budget, plan.max_steps)
                    or not any("done" in action for action in last_output.get("action", []))
                ):
                    return False
                plan = replace(plan, max_steps=budget)
            if _record_mismatch(task, record, self.root, plan):
                clarified = clarified_instruction_snapshot(task, review)
                if clarified is None or _record_mismatch(clarified, record, self.root, plan):
                    return False
                task = clarified
            checked = rescore_development_run(task, record)
            bound_review = inherit_unchanged_review(task, record, review)
            if self.trajectory_decision(
                task, checked, bound_review, operation_task=operation_task
            ).action is not None:
                return False
            # Already-reviewed standby cases retain their task-quality evidence.
            # A change newly moved outside the active budget needs its own check.
            return self.replay_budget_gap(operation_task) is None or (
                self.replay_budget_gap(operation_task, plan=plan) is not None
            )
        except (OSError, ValueError):
            return False

    def replay_budget_gap(
        self, task: dict[str, Any], *, plan: ExperimentPlan | None = None
    ) -> str | None:
        """A late scheduled change cannot reach an agent that uses its full budget."""
        plan = plan or self.trial_plan(task)
        condition = plan.condition("online")
        if task["track"] == "clarify" or condition.frame_delivery_steps is None:
            return None
        (prepared,) = prepare_session_tasks(
            [task], self.root, float(condition.frame_interval_seconds)
        )
        timeline = Timeline.compile(
            prepared, condition, default_release_step=plan.default_release_step,
            reference_steps=plan.reference_steps,
        )
        update = next(
            (
                r for r in timeline.releases
                if is_update_event(r.as_event()) and r.step > plan.max_steps
            ),
            None,
        )
        if update is not None:
            return (
                f"update scheduled at step {update.step} is outside the "
                f"{plan.max_steps}-step active budget; verify delivery under fixed replay"
            )
        return None

    def trial_plan(self, task: Mapping[str, Any]) -> ExperimentPlan:
        """New update-count materials use the unchanged composite formal budget."""
        if task["track"] == COMPOSITE_TRACK and task["id"].endswith(VARIANT_SUFFIX):
            if task["id"] not in self.update_count_plans:
                original, plan = frozen_original(self.root, task["id"].removesuffix(VARIANT_SUFFIX))
                if plan.max_steps != 30:
                    raise ValueError("update-count development requires the registered 30-step budget")
                self.update_count_plans[task["id"]] = replace(
                    plan, site_snapshots={task["id"]: plan.site_snapshot_for(original)},
                )
            return self.update_count_plans[task["id"]]
        return self.formal_plan

    @cached_property
    def update_count_plans(self) -> dict[str, ExperimentPlan]:
        return {}

    def decision(self, task: dict[str, Any]) -> DevelopmentDecision:
        """Read evidence only; never launches a model or changes the task state."""
        root = self.root
        reviews = self.reviews
        current_review = reviews.get("tasks", {}).get(str(task["id"]), {})
        if (
            isinstance(current_review, Mapping)
            and current_review.get("verdict") == "task_or_site_defect"
            and current_review.get("evidence")
        ):
            if current_review.get("task_sha256") == task_sha256(task):
                return DevelopmentDecision(
                    "repair_task",
                    "current trajectory review records an unresolved task or website defect",
                )
            return DevelopmentDecision(
                "verify_development_evidence",
                "a recorded task defect needs affected-scope verification before legacy admission can be retained",
            )
        if self.inherited_admission(task):
            return DevelopmentDecision(
                None,
                "existing task-validation evidence retained; no current score inferred",
            )
        clarified = clarified_instruction_snapshot(task, current_review)
        if clarified is not None:
            decision = self.decision(clarified)
            if decision.action is None:
                return DevelopmentDecision(
                    None,
                    "reviewed instruction clarification retains task-validation evidence; historical score unchanged",
                )
        rows = reviews.get("tasks", {})
        if not isinstance(rows, Mapping):
            raise ValueError("development review tasks must be an object")
        review = rows.get(str(task["id"]), {})
        if not isinstance(review, Mapping):
            raise ValueError(f"{task['id']}: development review must be an object")
        report = self.report
        reported_artifact = (
            report.get("tasks", {})
            .get(str(task["id"]), {})
            .get("development_screen", {})
            .get("run_artifact")
        )
        artifacts = list(
            dict.fromkeys(
                item for item in (review.get("run_artifact"), reported_artifact) if item
            )
        )
        pending = DevelopmentDecision(
            "run_development_trial",
            "no equivalent development run is recorded for the current execution inputs",
        )
        for artifact in artifacts:
            if (
                artifact == review.get("run_artifact")
                and review.get("verdict") == "invalid_run"
                and _review_matches(review, task, {})
            ):
                pending = DevelopmentDecision(
                    "run_development_trial",
                    "last recorded trial is invalid; preserve its archive and complete only the missing valid trial",
                )
                continue
            try:
                path = resolve_within(
                    root, str(artifact), context="development run", must_exist=False
                )
                record = load_run_record(path)
                mismatch = _record_mismatch(task, record, root, self.trial_plan(task))
            except (OSError, ValueError) as error:
                pending = DevelopmentDecision(
                    "verify_development_evidence",
                    f"recover or verify the referenced development evidence: {error}",
                )
                continue
            if mismatch:
                action = (
                    "verify_development_evidence"
                    if mismatch
                    in {
                        "missing_task_snapshot",
                        "missing_model_identity",
                        "missing_visual_evidence",
                        "missing_current_site_evidence",
                        "missing_site_runtime_contract",
                        "unverified_site_runtime_contract",
                    }
                    else "run_development_trial"
                )
                pending = DevelopmentDecision(
                    action, f"development run cannot yet be reused: {mismatch}"
                )
                continue
            checked = rescore_development_run(task, record)
            bound_review = review if review.get("run_artifact") == artifact else {}
            bound_review = inherit_unchanged_review(task, record, bound_review)
            return self.trajectory_decision(task, checked, bound_review)
        if self.formal_plan.condition("online").frame_delivery_steps is not None:
            stream = source_record(task, root=root).get("stream", {})
            gap = self.replay_budget_gap(task) if stream.get("review_evidence") else None
            if gap and pending.action == "run_development_trial":
                return DevelopmentDecision(pending.action, gap)
        return pending

    def trajectory_decision(
        self,
        task: dict[str, Any],
        checked: Mapping[str, Any],
        review: Mapping[str, Any],
        *,
        operation_task: dict[str, Any] | None = None,
    ) -> DevelopmentDecision:
        """Apply one trajectory-review contract to the lifecycle and batch runner."""
        root = self.root
        if checked.get("run_valid") is not True or checked.get("e2e_success") is None:
            return DevelopmentDecision(
                "run_development_trial", "development run lacks valid scoring evidence"
            )
        if not _review_matches(review, task, checked):
            return DevelopmentDecision(
                "review_development_trial",
                "valid run requires a current trajectory review, including successful runs",
            )
        verdict = review["verdict"]
        if verdict == "task_or_site_defect":
            return DevelopmentDecision(
                "repair_task", "trajectory review confirmed a task or website defect"
            )
        if verdict == "invalid_run":
            return DevelopmentDecision(
                "run_development_trial", "trajectory reviewer found an invalid run"
            )
        coverage = review.get("mechanism_coverage")
        missing_composite: set[str] = set()
        missing_events: set[str] = set()
        if task["track"] == COMPOSITE_TRACK:
            actual = checked.get("composite_delivery")
            if (
                not isinstance(actual, Mapping)
                or not actual
                or not isinstance(coverage, Mapping)
                or coverage.get("by_mechanism") != actual
            ):
                return DevelopmentDecision(
                    "review_development_trial",
                    "record actual delivery separately for each composite mechanism",
                )
            missing_composite = {name for name, delivered in actual.items() if delivered is not True}
            by_event = composite_event_coverage(checked)
            if by_event and coverage.get("by_event") != by_event:
                return DevelopmentDecision(
                    "review_development_trial",
                    "record actual delivery separately for each scheduled composite event",
                )
            missing_events = {name for name, delivered in by_event.items() if not delivered}
        if task["track"] == COMPOSITE_TRACK:
            covered = not missing_composite
        else:
            key = "answer_delivered" if task["track"] == "clarify" else "release_delivered"
            covered = checked.get(key) is True
        if not isinstance(coverage, Mapping) or coverage.get("observed") is not covered:
            return DevelopmentDecision(
                "review_development_trial",
                "record actual mechanism coverage in trajectory review",
            )
        if not covered:
            reference = coverage.get("directed_check")
            if not isinstance(reference, str) or not reference:
                return DevelopmentDecision(
                    "review_development_trial",
                    "uncovered mechanism requires a referenced directed operation check",
                )
            contract = REPORTS["smoke"]
            if reference != contract.path.as_posix():
                return DevelopmentDecision(
                    "verify_development_evidence",
                    "directed coverage must reference the existing task smoke audit",
                )
            try:
                check = self.smoke
            except (OSError, ValueError) as error:
                return DevelopmentDecision(
                    "verify_development_evidence",
                    f"recover the directed operation check: {error}",
                )
            operation_task = operation_task or task
            section = "composite_tasks" if task["track"] == COMPOSITE_TRACK else "tasks"
            row = check.get(section, {}).get(str(operation_task["id"]), {})
            if operation_task["id"].endswith(VARIANT_SUFFIX):
                site_path = self.trial_plan(operation_task).site_snapshot_for(operation_task)
                site_data = read_json_strict(Path(site_path))
            else:
                site_data = load_site_data(root, str(operation_task["site"]))
            site_digest = task_site_execution_sha256(operation_task, site_data)
            if (
                check.get("schema_version") != contract.schema_version
                or row.get("task_sha256") != task_sha256(operation_task)
                or row.get("task_site_execution_sha256") != site_digest
                or row.get("status") != "pass"
                or not isinstance(row.get("protocol_checks"), int)
                or row["protocol_checks"] < 1
                or not missing_composite <= {
                    name
                    for name, passed in row.get("composite_delivery", {}).items()
                    if passed is True
                }
                or not missing_events <= {
                    name
                    for name, passed in row.get("composite_event_coverage", {}).items()
                    if passed is True
                }
            ):
                return DevelopmentDecision(
                    "review_development_trial",
                    "directed operation evidence is missing or stale",
                )
        return DevelopmentDecision(
            None, "current valid development run and trajectory review are complete"
        )
