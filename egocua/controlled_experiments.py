"""Compile research variables onto the existing execution and scoring protocol."""

from __future__ import annotations

import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .checker import check_run, rescore_run
from .messages import DEFAULT_HISTORY_RETENTION, HISTORY_RETENTIONS
from .experiment import (
    FORMAL_PLAN,
    ExperimentPlan,
    expand_runs,
    load_experiment_plan,
    parse_experiment_plan,
    protocol_mismatch,
)
from .environment import load_site_data, site_data_request, site_snapshot
from .interaction_settings import (
    INTERACTION_SETTINGS,
    load_computer_requests,
    setting_conditions,
)
from .lineage import task_recording_group
from .run_materials import (
    execution_inputs as _execution_inputs,
    path_independent_inputs as _path_independent_inputs,
    public_site_materials,
    same_public_site as _same_public_site,
    site_images_match,
    site_evidence_mismatch,
    visual_mismatch,
    wearer_input_paths,
)
from .run_ledger import load_run_record, task_local_content
from .experiment_output import local_timestamp
from .schema import load_tasks, validate_task
from .selection import load_selection, task_ids
from .sampling_media import (
    materialize_sampling_frames,
    prepare_session_tasks,
)
from .synthesis_workflow import (
    COMPOSITE_REVIEWS, SynthesisWorkflow, TaskStatus, load_composite_drafts,
)
from .timeline import COMPOSITE_TRACK, COMPUTER_USE, OBSERVATION, Condition, Timeline, composite_input
from .update_count import (
    REGISTRATION, UPDATE_COUNT, compile_variant, frozen_original, load_cases, source_study,
)
from .workspace import read_json_strict, resolve_within, write_bytes_atomic, write_json_atomic


DEFAULT_EVALUATION_CONCURRENCY = 10
MAX_EVALUATION_CONCURRENCY = 24
COMPOSITE_VALIDATION = "composite-validation"
COMPOSITE_FORMAL = "composite-formal"
EXECUTION_SUPPLEMENT = "execution-supplement"


# Confirmed defects defer a trial. Awaiting review remains visible in its plan.
def compute_task_statuses(root: Path, ids: list[str]) -> dict[str, str]:
    """Read member status from the synthesis interface."""
    if not ids:
        return {}
    status = SynthesisWorkflow(root).status(ids)
    return {task_id: row["status"] for task_id, row in status["tasks"].items()}


@dataclass(frozen=True)
class ControlledArm:
    name: str
    tasks: list[dict[str, Any]]
    plan: ExperimentPlan
    variable: str = "frequency"
    level: str | None = None
    pair_id: str | None = None
    first_run_ids: dict[str, dict[str, str]] = field(default_factory=dict)
    direction: str | None = None

    def schedule(self) -> list[dict[str, Any]]:
        return expand_runs(self.tasks, self.plan)


def sampled_task_ids(arms: list[ControlledArm]) -> dict[str, list[str]]:
    """Record the task IDs actually scheduled for each controlled variable."""
    recorded: dict[str, list[str]] = {}
    for arm in arms:
        if arm.variable == UPDATE_COUNT:
            recorded.setdefault(UPDATE_COUNT, []).extend(task["id"] for task in arm.tasks)
            continue
        if arm.variable in {*CONTROLLED_VARIABLES, COMPOSITE_VALIDATION, COMPOSITE_FORMAL}:
            recorded[arm.variable] = [task["id"] for task in arm.tasks]
    return recorded


def evidence_visibility(
    evidence: dict[str, Any] | None, task_id: str, condition: Condition
) -> dict[str, Any]:
    reviews = [
        row
        for row in (evidence or {}).get("task_reviews", [])
        if row["task_id"] == task_id
    ]
    if not reviews:
        return {
            "key_evidence_visibility": "unreviewed",
            "visibility_review_status": "missing",
        }
    if len(reviews) != 1:
        raise ValueError(f"duplicate visibility review: {task_id}")
    review = reviews[0]
    key = (
        f"interval_{condition.frame_interval_seconds:g}s"
        if condition.frame_interval_seconds is not None
        else condition.scope
    )
    sampling = review.get("sampling_review", {})
    if condition.frame_interval_seconds is not None:
        if sampling.get("status") != "reviewed" or key not in sampling.get(
            "conditions", {}
        ):
            value = None
        else:
            facts = [
                fact["visibility"]
                for fact in sampling["conditions"][key]["facts"]
                if fact.get("required_for_task", True)
            ]
            value = (
                "absent"
                if "absent" in facts
                else "uncertain"
                if "uncertain" in facts
                else "visible"
            )
    else:
        value = review["visibility"].get(key)
    if value is None:
        return {
            "key_evidence_visibility": "unreviewed",
            "visibility_review_status": "missing",
        }
    if value not in {"visible", "absent", "uncertain"}:
        raise ValueError(f"invalid evidence visibility: {value}")
    return {
        "key_evidence_visibility": value,
        "visibility_review_status": review.get("sampling_review", {}).get(
            "status", "missing"
        )
        if condition.frame_interval_seconds is not None
        else review["review_status"],
    }


def study_matrix(
    arms: list[ControlledArm],
    *,
    root: Path,
    evidence: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        tasks = {task["id"]: task for task in arm.tasks}
        for run in arm.schedule():
            task, plan = tasks[run["task_id"]], arm.plan
            timeline = Timeline.compile(
                task,
                plan.condition(run["condition"]),
                default_release_step=plan.default_release_step,
                reference_steps=plan.reference_steps,
            )
            packets = [
                list(item["frame_paths"]) for item in timeline.initial_observations
            ]
            packets.extend(
                list(event.payload["frame_paths"])
                for event in timeline.releases
                if event.kind == OBSERVATION
            )
            rows.append(
                {
                    **run,
                    "arm": arm.name,
                    "variable": arm.variable,
                    "level": arm.level or run["condition"],
                    "pair_id": arm.pair_id,
                    "experiment_id": plan.experiment_id,
                    "frame_interval_seconds": plan.condition(
                        run["condition"]
                    ).frame_interval_seconds,
                    "wearer_jpeg_quality": plan.condition(
                        run["condition"]
                    ).wearer_jpeg_quality,
                    "source_group": task_recording_group(task, root=root),
                    "frame_counts_if_delivered": [len(packet) for packet in packets],
                    "unique_frame_counts_if_delivered": [
                        len(set(packet)) for packet in packets
                    ],
                    "unprepared_frame_count": sum(
                        not (root / path).is_file()
                        for packet in packets
                        for path in packet
                    ),
                    "release_trigger": plan.release_trigger,
                    "max_steps": plan.max_steps,
                    "pre_release_max_steps": plan.pre_release_max_steps,
                    "post_release_max_steps": plan.post_release_max_steps,
                    **evidence_visibility(
                        evidence, task["id"], plan.condition(run["condition"])
                    ),
                }
            )
    return rows


def online_arm(
    variable: str, tasks: list[dict[str, Any]], base: ExperimentPlan
) -> ControlledArm:
    """Run each task once under the plan's online condition."""
    plan = replace(
        base,
        experiment_id=f"{base.experiment_id}-{variable}",
        conditions={"online": base.condition("online")},
        track_conditions={task["track"]: ("online",) for task in tasks},
        reference_condition_name="online",
        track_probe_conditions={},
        replicate_ids=(0,),
    )
    return ControlledArm(variable, tasks, plan, variable)


CONTROLLED_VARIABLES = ("pace", "retention", "cue", "arrival")
PACE_BATCH_SECONDS = (8, 4, 16)
"""Source seconds per step, reference first; each step keeps the same frame count."""


def _variable_conditions(
    variable: str, online: Condition
) -> tuple[list[Condition], tuple[str, ...]]:
    """One input setting changed from online, reference first, and its tracks."""
    if variable == "pace":
        frames = online.batch_seconds / online.frame_interval_seconds
        return [
            replace(online, name=f"pace_{seconds:g}s")
            if seconds == online.batch_seconds
            else replace(
                online,
                name=f"pace_{seconds:g}s",
                batch_seconds=seconds,
                frame_interval_seconds=seconds / frames,
            )
            for seconds in PACE_BATCH_SECONDS
        ], ("graft", "refresh", "clarify")
    if variable == "retention":
        policies = sorted(
            HISTORY_RETENTIONS, key=lambda policy: policy != DEFAULT_HISTORY_RETENTION
        )
        return [
            replace(online, name=policy, visual_history=policy) for policy in policies
        ], ("graft", "refresh")
    if variable == "cue":
        return [
            replace(online, name="silent"),
            replace(online, name="cued", change_cue=True),
            replace(online, name="cued_facts", change_cue=True, change_facts=True),
        ], ("refresh",)
    if variable == "arrival":
        return [
            replace(online, name="concurrent"),
            replace(online, name="after_finish", follow_up_after_finish=True),
        ], ("graft",)
    raise ValueError(f"unknown controlled variable {variable!r}")


def build_variable_arm(
    variable: str, tasks: list[dict[str, Any]], plan: ExperimentPlan
) -> ControlledArm:
    """Run the same tasks under each level of one online input setting."""
    online = plan.condition("online")
    if plan.adapter != "browsergym" or online.visual_history != DEFAULT_HISTORY_RETENTION:
        raise ValueError(
            "controlled variables derive from the current compressed BrowserGym protocol"
        )
    conditions, tracks = _variable_conditions(variable, online)
    if not tasks or any(task["track"] not in tracks for task in tasks):
        raise ValueError(f"{variable} requires a sample of {'/'.join(tracks)} tasks")
    return ControlledArm(
        variable,
        tasks,
        replace(
            plan,
            experiment_id=f"{plan.experiment_id}-{variable}",
            conditions={condition.name: condition for condition in conditions},
            track_conditions={
                task["track"]: tuple(condition.name for condition in conditions)
                for task in tasks
            },
            reference_condition_name=conditions[0].name,
            track_probe_conditions={},
            replicate_ids=(0,),
        ),
        variable,
    )


def load_formal_plan(
    root: Path, plan_path: str, model: str | None = None
) -> ExperimentPlan:
    """Read the registered protocol; the model under test is chosen at freeze."""
    plan = load_experiment_plan(
        resolve_within(root, plan_path, context="formal evaluation plan")
    )
    if model is None:
        return plan
    return replace(plan, model=model, model_version=model)


def attach_cue_facts(tasks: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Freeze reviewed descriptions with their source timestamps, outside formal tasks."""
    materials = read_json_strict(root / "data/experiments/cue-facts.json")
    if materials["schema_version"] != 1:
        raise ValueError("unsupported Cue facts format")
    prepared = []
    for task in tasks:
        row = materials["tasks"][task["id"]]
        if row["review"]["status"] != "reviewed" or not row["review"]["evidence"]:
            raise ValueError(f"{task['id']}: Cue facts need a frame review")
        prepared.append({**task, "visual_session": {
            **task["visual_session"],
            "cue_facts": {key: row[key] for key in ("text", "timestamps_seconds")},
        }})
    return prepared


def prepare_formal_inputs(
    tasks: list[dict[str, Any]], plan: ExperimentPlan, root: Path
) -> list[dict[str, Any]]:
    """Sample each task's replay frames once, at every interval the plan streams."""
    intervals = sorted(
        {
            float(condition.frame_interval_seconds)
            for condition in plan.conditions.values()
            if condition.frame_delivery_steps is not None
        }
    )
    if not intervals:
        return tasks
    return prepare_session_tasks(tasks, root, *intervals)


def load_validation_study(
    root: Path, tasks: list[dict[str, Any]]
) -> list[ControlledArm]:
    """Development trials: candidates once under the formal online protocol."""
    plan = load_formal_plan(root, FORMAL_PLAN)
    return [online_arm("validation", prepare_formal_inputs(tasks, plan, root), plan)]


def load_composite_validation_study(
    root: Path, sample_ids: list[str] | None = None, *, model: str
) -> tuple[list[ControlledArm], None, dict[str, Any]]:
    """Validate and prepare reviewed experimental tasks, without launching a model."""
    if not model.strip():
        raise ValueError("composite validation requires an explicit model")
    root = root.resolve()
    plan = load_formal_plan(root, FORMAL_PLAN, model)
    with site_data_request(root):
        tasks = SynthesisWorkflow(root).composite_trial_inputs(sample_ids, plan)
        materialize_sampling_frames(tasks, root)
        for task in tasks:
            validate_task(
                task,
                load_site_data(root, str(task["site"])),
                root=root,
                require_assets=True,
            )
    arm = online_arm(COMPOSITE_VALIDATION, tasks, plan)
    gate = {
        "composite_semantic_reviews": {
            task["id"]: (COMPOSITE_REVIEWS / f"{task['id']}.json").as_posix()
            for task in tasks
        }
    }
    return [arm], None, gate



def load_composite_formal_study(
    root: Path, sample_ids: list[str] | None = None, *, model: str
) -> tuple[list[ControlledArm], dict[str, Any], dict[str, Any]]:
    """Freeze ready composite members under the separately registered formal budget."""
    root = root.resolve()
    ids = [task["id"] for task in load_composite_drafts(root, sample_ids)]
    status = SynthesisWorkflow(root).status(ids)
    statuses = {task_id: row["status"] for task_id, row in status["tasks"].items()}
    pending = {
        task_id: value
        for task_id, value in statuses.items()
        if value != TaskStatus.READY
    }
    if pending:
        raise ValueError(f"formal composite evaluation requires ready tasks: {pending}")
    [arm], _, gate = load_composite_validation_study(root, ids, model=model)
    budget = arm.plan.composite_max_steps
    if not isinstance(budget, int) or isinstance(budget, bool) or budget != 30:
        raise ValueError(
            "benchmark runtime.composite_max_steps must be registered as 30"
        )
    semantics = {}
    for task in arm.tasks:
        review = status["review_evidence"][task["id"]]
        annotation = review["evidence"].get("change_semantics")
        validate_composite_semantics(task, annotation)
        semantics[task["id"]] = annotation
    formal = online_arm(
        COMPOSITE_FORMAL,
        arm.tasks,
        replace(load_formal_plan(root, FORMAL_PLAN, model), max_steps=budget),
    )
    return (
        [formal],
        {"change_semantics": semantics},
        {
            **gate,
            "formal_task_statuses": statuses,
        },
    )


def load_update_count_study(
    root: Path, sample_ids: list[str] | None = None, *, model: str,
) -> tuple[list[ControlledArm], dict[str, Any], dict[str, Any]]:
    """Prepare delta-reviewed variants against their unchanged frozen originals."""
    root = root.resolve()
    cases = load_cases(root, sample_ids)
    excluded = [case["pair_id"] for case in cases if not case["included"]]
    if sample_ids and excluded:
        raise ValueError(f"excluded update-count pairs requested: {excluded}")
    cases = [case for case in cases if case["included"]]
    if not cases:
        raise ValueError("no included update-count pairs")
    source = source_study(root, model)
    originals, _, _ = load_frozen_study(source)
    originals = [arm for arm in originals if arm.variable == COMPOSITE_FORMAL]
    if len(originals) != 1:
        raise ValueError("update-count requires one frozen formal composite arm")
    original_arm = originals[0]
    if original_arm.plan.model != model or original_arm.plan.max_steps != 30:
        raise ValueError("update-count original model or budget differs")
    tasks = {task["id"]: task for task in original_arm.tasks}
    workflow = SynthesisWorkflow(root)
    status = workflow.status([case["id"] for case in cases])
    arms = []
    audits = {}
    for case in cases:
        audit = status["update_count_audits"][case["pair_id"]]
        if audit["errors"]:
            raise ValueError(f"{case['pair_id']}: update-count audit: {audit['errors']}")
        review = status["review_evidence"].get(case["id"], {})
        if status["tasks"][case["id"]]["next_action"] not in {
            "run_development_trial", "review_development_trial", "verify_development_evidence",
            "ready_for_composite_study",
        } or review.get("verdict") != "accepted":
            raise ValueError(f"{case['id']}: delta review is incomplete")
        original = tasks[case["pair_id"]]
        # Compare both models' frozen inputs with the same registered scene and key.
        reference, _ = frozen_original(root, case["pair_id"])
        if task_local_content(original) != task_local_content(reference):
            raise ValueError(f"{case['pair_id']}: frozen originals differ across models")
        variant = compile_variant(original, case)
        for level, task in (
            ("more" if case["direction"] == "merge" else "fewer", original),
            ("fewer" if case["direction"] == "merge" else "more", variant),
        ):
            name = f"{UPDATE_COUNT}-{case['pair_id']}-{level}"
            plan = replace(
                original_arm.plan, experiment_id=f"{original_arm.plan.experiment_id}-{name}",
                site_snapshots={task["id"]: original_arm.plan.site_snapshot_for(original)},
            )
            arms.append(ControlledArm(
                name, [task], plan, UPDATE_COUNT, level, case["pair_id"],
                direction=case["direction"],
            ))
        audits[case["pair_id"]] = audit
    return arms, {"update_count_audits": audits}, {
        "required_original_reuse": [case["pair_id"] for case in cases],
        "source_registration": str(REGISTRATION),
        "composite_semantic_reviews": {
            case["id"]: str(COMPOSITE_REVIEWS / f"{case['id']}.json") for case in cases
        },
    }


def validate_composite_semantics(task: dict[str, Any], annotation: Any) -> None:
    definition = composite_input(task)
    if not isinstance(annotation, dict) or not isinstance(
        annotation.get("events"), dict
    ):
        raise ValueError(f"{task['id']}: missing reviewed change_semantics")
    if set(annotation["events"]) != {update.event_id for update in definition.updates}:
        raise ValueError(
            f"{task['id']}: change_semantics must cover exactly the registered updates"
        )
    allowed = {
        "R": {"reveal", "retain", "invalidate"},
        "G": {"refine", "augment", "replace", "drop"},
    }
    for update in definition.updates:
        event = annotation["events"][update.event_id]
        types = event.get("types")
        if (
            not isinstance(types, list)
            or not types
            or not set(types) <= allowed[update.mechanism]
            or not event.get("reason")
        ):
            raise ValueError(f"{task['id']}: invalid semantics for {update.event_id}")
    clarification = annotation.get("clarify")
    if definition.clarification:
        if (
            not isinstance(clarification, dict)
            or clarification.get("type")
            not in {"position", "sequence", "variant", "other"}
            or not clarification.get("reason")
        ):
            raise ValueError(f"{task['id']}: missing clarification semantics")
    elif clarification is not None:
        raise ValueError(f"{task['id']}: unexpected clarification semantics")
    relation = annotation.get("choice_relation", {})
    if relation.get("type") not in {
        "same_choice",
        "different_choices",
        "mixed",
    } or not relation.get("reason"):
        raise ValueError(f"{task['id']}: missing choice relation semantics")


def model_variant(arms: list[ControlledArm], model: str) -> list[ControlledArm]:
    """Change only the model of a frozen composite or controlled study."""
    if not model.strip() or any(
        arm.variable not in {COMPOSITE_FORMAL, UPDATE_COUNT, *CONTROLLED_VARIABLES}
        or not arm.plan.frozen_asset_root
        for arm in arms
    ):
        raise ValueError("model variants require a frozen formal composite or controlled study")
    return [
        replace(arm, plan=replace(arm.plan, model=model, model_version=model))
        for arm in arms
    ]


def repeated_controlled_study(
    sources: list[Path], repetitions: int, root: Path,
) -> tuple[list[ControlledArm], dict[str, Any] | None, dict[str, Any]]:
    """Extend frozen input comparisons, sharing identical runs across variables."""
    if repetitions < 2 or not sources or len(set(sources)) != len(sources):
        raise ValueError("repetitions need distinct frozen studies and at least two runs")
    arms: list[ControlledArm] = []
    statuses: dict[str, str] = {}
    source_gates = {}
    pair_audits: dict[str, Any] = {}
    for source in sources:
        if not (source / "plan.json").is_file():
            raise ValueError("repeat only current frozen studies")
        loaded, evidence, gate = load_frozen_study(source)
        definition = read_study(source)
        completed = {row["execution_index"]: row["run_id"]
                     for row in definition.get("execution_results", []) if row.get("valid") is True}
        first_ids: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        for entry in definition["execution_schedule"]:
            if entry["execution_index"] not in completed:
                continue
            for reference in entry["analysis_references"]:
                if reference["replicate_id"] == 0:
                    first_ids[reference["arm"]][reference["task_id"]][reference["condition"]] = completed[entry["execution_index"]]
        if evidence:
            if {arm.variable for arm in loaded} != {UPDATE_COUNT} or set(evidence) != {"update_count_audits"}:
                raise ValueError("repetition source has unmerged experimental evidence")
            for pair_id, audit in evidence["update_count_audits"].items():
                if pair_id in pair_audits and pair_audits[pair_id] != audit:
                    raise ValueError(f"{pair_id}: inconsistent frozen update-count audit")
                pair_audits[pair_id] = audit
        if definition.get("superseded_by") or (gate or {}).get("superseded_by"):
            raise ValueError("cannot extend a superseded study")
        for arm in loaded:
            if arm.variable not in {*CONTROLLED_VARIABLES, UPDATE_COUNT}:
                raise ValueError("repetition extension is limited to online input variables")
            if arm.plan.replicate_ids != (0,):
                raise ValueError("extend the original single-round study, not a repetition export")
            expected = {(run["task_id"], run["condition"]) for run in arm.schedule()}
            saved = {(task_id, condition) for task_id, conditions in first_ids[arm.name].items()
                     for condition in conditions}
            if saved != expected:
                raise ValueError(f"{arm.name}: original first round is incomplete")
            arms.append(replace(arm, first_run_ids=dict(first_ids[arm.name]), plan=replace(
                arm.plan, replicate_ids=tuple(range(repetitions)),
            )))
        for task_id, status in (gate or {}).get("formal_task_statuses", {}).items():
            if task_id in statuses and statuses[task_id] != status:
                raise ValueError(f"inconsistent task qualification: {task_id}")
            statuses[task_id] = status
        source_gates[str(source.resolve())] = gate
    if len({arm.name for arm in arms}) != len(arms):
        raise ValueError("repetition sources must have distinct arm names")
    output_labels(arms)
    by_name = {arm.name: arm for arm in arms}
    for entry in execution_schedule(arms):
        references = entry["analysis_references"]
        first = by_name[references[0]["arm"]]
        task = next(task for task in first.tasks if task["id"] == entry["run"]["task_id"])
        for reference in references[1:]:
            other = by_name[reference["arm"]]
            peer = next(item for item in other.tasks if item["id"] == task["id"])
            if not _same_public_site(
                (task, peer), (first.plan.site_snapshot_for(task), other.plan.site_snapshot_for(peer)),
            ):
                raise ValueError(f"{task['id']}: shared baseline has different website facts")
            if first.plan.frozen_asset_root != other.plan.frozen_asset_root:
                first_root = Path(first.plan.frozen_asset_root) if first.plan.frozen_asset_root else root
                other_root = Path(other.plan.frozen_asset_root) if other.plan.frozen_asset_root else root
                paths = wearer_input_paths([task], first.plan)
                if any((first_root / path).read_bytes() != (other_root / path).read_bytes() for path in paths):
                    raise ValueError(f"{task['id']}: shared baseline has different images")
    return arms, {"update_count_audits": pair_audits} if pair_audits else None, {
        "formal_task_statuses": statuses,
        "repetition_extension": {
            "repetitions": repetitions, "source_studies": source_gates,
            "original_round_reused": True,
        },
    }


def load_formal_study(
    root: Path,
    plan_path: str,
    *,
    with_probes: bool = False,
    interaction_settings: bool = False,
    variable: str | None = None,
    sample_ids: list[str] | None = None,
    model: str | None = None,
) -> tuple[list[ControlledArm], None, dict[str, Any]]:
    """Prepare the formal baseline or one controlled variable on a chosen sample."""
    members = task_ids(load_selection(root))
    if interaction_settings and (with_probes or variable is not None):
        raise ValueError("interaction settings are separate from probes and controlled variables")
    if (variable is None) != (sample_ids is None):
        raise ValueError("a controlled variable needs its sample and vice versa")
    if sample_ids is not None:
        if with_probes or not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise ValueError(
                "a controlled sample needs distinct task IDs and cannot include probes"
            )
        unknown = set(sample_ids) - set(members)
        if unknown:
            raise ValueError(f"sample tasks are not formal members: {sorted(unknown)}")
        members = sample_ids
    definitions = {task["id"]: task for task in load_tasks(root / "data/tasks")}
    statuses = compute_task_statuses(root, members)
    pending = {
        task_id: status
        for task_id, status in statuses.items()
        if status != TaskStatus.READY
    }
    if pending:
        reasons = ", ".join(
            f"{status}={count}" for status, count in sorted(Counter(pending.values()).items())
        )
        raise ValueError(
            f"formal evaluation requires ready members; {len(pending)} of "
            f"{len(members)} are pending ({reasons}). "
            "Run 'egocua synthesize status --summary'."
        )
    plan = load_formal_plan(root, plan_path, model)
    tasks = [definitions[task_id] for task_id in members]
    gate = {"formal_task_statuses": statuses}
    if variable is not None:
        arm = build_variable_arm(variable, tasks, plan)
        prepared = prepare_formal_inputs(tasks, arm.plan, root)
        if variable == "cue":
            prepared = attach_cue_facts(prepared, root)
        return [replace(arm, tasks=prepared)], None, gate
    formal_tasks = prepare_formal_inputs(tasks, plan, root)
    if interaction_settings:
        requests = load_computer_requests(root, members)
        return interaction_setting_arms(formal_tasks, plan, requests), None, gate
    arms = [online_arm("formal", formal_tasks, plan)]
    if with_probes:
        if not plan.track_probe_conditions:
            raise ValueError("--with-probes requires probes registered in the plan")
        arms.append(information_probe_arm(formal_tasks, plan))
    return arms, None, gate


def interaction_setting_arms(
    tasks: list[dict[str, Any]], plan: ExperimentPlan, requests: dict[str, str],
) -> list[ControlledArm]:
    """One arm per setting, with authored text present only in its own arm."""
    arms = []
    for condition in setting_conditions(plan.condition("online")):
        name = condition.name
        arm_tasks = tasks
        if name == COMPUTER_USE:
            arm_tasks = [
                {
                    **{key: value for key, value in task.items() if key != "visual_session"},
                    "instructions": {**task["instructions"], COMPUTER_USE: requests[task["id"]]},
                }
                for task in tasks
            ]
        arm_plan = replace(
            plan,
            experiment_id=f"{plan.experiment_id}-{INTERACTION_SETTINGS}-{name}",
            conditions={name: condition},
            track_conditions={task["track"]: (name,) for task in tasks},
            reference_condition_name=name,
            track_probe_conditions={},
        )
        arms.append(ControlledArm(name, arm_tasks, arm_plan, INTERACTION_SETTINGS))
    return arms


def information_probe_arm(
    formal_tasks: list[dict[str, Any]], plan: ExperimentPlan
) -> ControlledArm:
    """Pair each registered probe with online on the same members, once each."""
    tracks = {task["track"] for task in formal_tasks}
    return ControlledArm(
        "visual",
        formal_tasks,
        replace(
            plan,
            experiment_id=f"{plan.experiment_id}-visual",
            track_conditions={
                track: ("online", *plan.probe_conditions_for(track))
                for track in tracks
            },
            reference_condition_name="online",
            track_probe_conditions={},
            replicate_ids=(0,),
        ),
        "visual",
    )


def scoring_key(task: dict[str, Any]) -> tuple[Any, Any]:
    return task["goal"], task.get("graft", {}).get("initial_goal")


def execution_schedule(arms: list[ControlledArm]) -> list[dict[str, Any]]:
    """One physical execution may satisfy several identical analysis conditions."""
    executions: list[dict[str, Any]] = []
    inputs_to_index: dict[tuple[str, int, str | None], int] = {}
    scoring_keys: dict[str, tuple[Any, Any]] = {}
    for arm in arms:
        tasks = {task["id"]: task for task in arm.tasks}
        for task_id, task in tasks.items():
            key = scoring_key(task)
            if task_id in scoring_keys and scoring_keys[task_id] != key:
                raise ValueError(
                    f"{task_id}: inconsistent frozen scoring key between conditions"
                )
            scoring_keys[task_id] = key
        for run in arm.schedule():
            reference = {
                **run,
                "arm": arm.name,
                "experiment_id": arm.plan.experiment_id,
            }
            key = (
                _execution_inputs(tasks[run["task_id"]], arm.plan, run["condition"]),
                run["replicate_id"],
                arm.first_run_ids.get(run["task_id"], {}).get(run["condition"])
                if run["replicate_id"] == 0 else None,
            )
            if key in inputs_to_index:
                executions[inputs_to_index[key]]["analysis_references"].append(
                    reference
                )
            else:
                inputs_to_index[key] = len(executions)
                executions.append(
                    {
                        "execution_index": len(executions),
                        "run": reference,
                        "analysis_references": [reference],
                    }
                )
    return executions


def read_study(directory: Path) -> dict[str, Any]:
    """Read frozen current plan plus derived progress, or an untouched historical study."""
    if (directory / "plan.json").is_file():
        definition = read_json_strict(directory / "plan.json")
        progress = directory / "results.json"
        if progress.is_file():
            result = read_json_strict(progress)
            definition.update(
                {
                    key: result[key]
                    for key in (
                        "execution_results", "execution_stop", "run_purpose",
                        "superseded_by", "superseded_reason",
                    )
                    if key in result
                }
            )
        return definition
    return read_json_strict(directory / "study.json")


def select_execution(directory: Path, reference: dict[str, Any]) -> int:
    """Resolve a named analysis condition to its unique physical execution."""
    matches = [
        entry["execution_index"]
        for entry in read_study(directory)["execution_schedule"]
        if any(
            all(candidate[key] == value for key, value in reference.items())
            for candidate in entry["analysis_references"]
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one registered execution, found {matches}")
    return matches[0]


def output_labels(arms: list[ControlledArm]) -> dict[str, str]:
    names = {
        "formal": "全题库",
        "validation": "任务校验",
        COMPOSITE_VALIDATION: "组合任务校验",
        COMPOSITE_FORMAL: "组合任务",
        UPDATE_COUNT: "更新次数对照",
        EXECUTION_SUPPLEMENT: "缺项补跑",
        "visual": "信息对照",
        INTERACTION_SETTINGS: "三场景对照",
        "pace": "现实节奏",
        "retention": "视觉历史保留",
        "cue": "变化提示",
        "arrival": "更新到达",
    }
    models = {arm.plan.model for arm in arms}
    if len(models) != 1:
        raise ValueError("one experiment directory requires one registered model")
    variables = sorted({arm.variable for arm in arms})
    if {"formal", COMPOSITE_FORMAL}.intersection(variables):
        category = "正式评测"
    elif {"validation", COMPOSITE_VALIDATION}.intersection(variables):
        category = "任务试跑"
    elif variables == [EXECUTION_SUPPLEMENT]:
        category = "补充评测"
    elif variables == [UPDATE_COUNT] and any(len(arm.plan.replicate_ids) > 1 for arm in arms):
        category = "稳定性复测"
    else:
        category = "变量对照"
    return {
        "category": category,
        "variable": "、".join(names[name] for name in variables),
        "model": models.pop(),
    }


def export_study(
    arms: list[ControlledArm],
    destination: Path,
    *,
    root: Path,
    evidence: dict[str, Any] | None = None,
    gate: dict[str, Any] | None = None,
    reuse_from: Path | tuple[Path, ...] | None = None,
    retain_site_for: tuple[str, ...] = (),
) -> None:
    """Export immutable input copies for the existing runner; never start a model."""
    if destination.exists():
        raise ValueError(f"refusing to overwrite existing study export: {destination}")
    if root is None:
        raise ValueError("study export requires the project root to freeze all inputs")
    if any(arm.variable == UPDATE_COUNT for arm in arms):
        if root is None or any(arm.variable != UPDATE_COUNT for arm in arms):
            raise ValueError("update-count export requires its own study and project root")
        if reuse_from is None:
            reuse_from = source_study(root, arms[0].plan.model)
    if reuse_from is not None and root is None:
        raise ValueError("reusing runs requires the project root and frozen sites")
    reuse_sources = (
        (reuse_from,) if isinstance(reuse_from, Path) else reuse_from or ()
    )
    if reuse_from is not None and not reuse_sources:
        raise ValueError("reuse sources cannot be empty")
    if (gate or {}).get("repetition_extension") and not reuse_sources:
        raise ValueError("a repetition extension must reuse its original first round")
    retained = {}
    if retain_site_for:
        if len(reuse_sources) != 1 or any(arm.plan.frozen_asset_root for arm in arms):
            raise ValueError("retaining a task's site requires a new export with a reuse source")
        old_arms, _, _ = load_frozen_study(reuse_sources[0])
        requested = set(retain_site_for)
        if not requested <= {task["id"] for arm in arms for task in arm.tasks}:
            raise ValueError("cannot retain a site for a task outside the export")
        for task_id in requested:
            paths = {
                arm.plan.site_snapshot_for(task)
                for arm in old_arms
                for task in arm.tasks
                if task["id"] == task_id
            }
            if len(paths) != 1 or None in paths:
                raise ValueError(f"{task_id}: expected one historical site snapshot")
            retained[task_id] = next(iter(paths))
    input_sources = sorted(
        {arm.plan.frozen_asset_root for arm in arms if arm.plan.frozen_asset_root}
    )
    sampling_tasks = [
        task
        for arm in arms
        for task in arm.tasks
        if arm.plan.frozen_asset_root is None
        and (
            "visual_session" in task
            or any("frequency_samples" in packet for packet in task["visual_packets"])
        )
    ]
    if sampling_tasks:
        if root is None:
            raise ValueError(
                "frequency export requires the project root for source extraction"
            )
        materialize_sampling_frames(sampling_tasks, root)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "study"
        staging.mkdir()
        frozen_arms = []
        site_data_cache = {}
        copied_inputs: dict[str, Path] = {}
        for arm in arms:
            snapshots = (
                dict(arm.plan.site_snapshots) if arm.plan.frozen_asset_root else {}
            )
            snapshots.update({
                task["id"]: retained[task["id"]]
                for task in arm.tasks if task["id"] in retained
            })
            snapshot_keys = {
                task["id"] if task["id"] in snapshots else task["site"]
                for task in arm.tasks
            }
            snapshots = {key: path for key, path in snapshots.items() if key in snapshot_keys}
            source_root = (
                Path(arm.plan.frozen_asset_root) if arm.plan.frozen_asset_root else root
            )
            paths: set[str] = set()
            if root is not None:
                by_snapshot = defaultdict(list)
                for task in arm.tasks:
                    site, task_id = task["site"], task["id"]
                    key = task_id if task_id in snapshots else site
                    by_snapshot[(site, key)].append(task)
                for (site, key), scoped_tasks in sorted(by_snapshot.items()):
                    cache_key = (site, snapshots.get(key))
                    if cache_key not in site_data_cache:
                        site_data_cache[cache_key] = (
                            read_json_strict(Path(snapshots[key]))
                            if key in snapshots
                            else load_site_data(root, site, require_assets=True)
                        )
                    data = site_data_cache[cache_key]
                    images = {}
                    for task in scoped_tasks:
                        images.update(public_site_materials(task, data)["images"])
                    if not site_images_match(images, source_root):
                        raise ValueError(
                            f"{site}: public image content differs from source evidence"
                        )
                    paths.update(images)
                    snapshot = site_snapshot(staging / "inputs", site, data)
                    snapshots[key] = (destination / snapshot.relative_to(staging)).as_posix()
            frozen_root = arm.plan.frozen_asset_root
            if root is not None:
                if source_root is None:
                    raise ValueError("study export requires an asset root")
                paths.update(wearer_input_paths(arm.tasks, arm.plan))
                for relative in sorted(paths):
                    source = resolve_within(
                        source_root, relative, context="frozen study input"
                    )
                    previous = copied_inputs.get(relative)
                    if previous == source:
                        continue
                    content = source.read_bytes()
                    if previous is not None:
                        if previous.read_bytes() != content:
                            raise ValueError(f"conflicting frozen input: {relative}")
                        continue
                    target = resolve_within(
                        staging / "inputs",
                        relative,
                        context="frozen study input",
                        must_exist=False,
                    )
                    write_bytes_atomic(target, content)
                    copied_inputs[relative] = source
                frozen_root = (destination / "inputs").as_posix()
            frozen_arms.append(
                replace(
                    arm,
                    plan=replace(
                        arm.plan,
                        site_snapshots=snapshots,
                        frozen_asset_root=frozen_root,
                    ),
                )
            )
        arms = frozen_arms
        for arm in arms:
            for task in arm.tasks:
                write_json_atomic(
                    staging / "tasks" / arm.name / task["track"] / f"{task['id']}.json",
                    task,
                )
        definition: dict[str, Any] = {
            "arms": [
                {
                    "name": arm.name,
                    "variable": arm.variable,
                    "level": arm.level,
                    "pair_id": arm.pair_id,
                    **({"direction": arm.direction} if arm.direction else {}),
                    **({"first_run_ids": arm.first_run_ids} if arm.first_run_ids else {}),
                    "plan": arm.plan.as_dict(),
                }
                for arm in arms
            ],
            "evidence": evidence,
            "task_sample": sampled_task_ids(arms),
            "review_gate": gate,
            "execution_schedule": execution_schedule(arms),
        }
        if input_sources:
            definition["input_source_assets"] = input_sources
        if {arm.variable for arm in arms} == {COMPOSITE_VALIDATION}:
            definition["run_purpose"] = "task_validation"
        if reuse_from is not None:
            frozen = [
                replace(
                    arm,
                    plan=replace(
                        arm.plan,
                        frozen_asset_root=str(staging / "inputs"),
                        site_snapshots={
                            key: str(staging / Path(path).relative_to(destination))
                            for key, path in arm.plan.site_snapshots.items()
                        },
                    ),
                )
                for arm in arms
            ]
            reused_by_index = {}
            for source in reuse_sources:
                for outcome in _reusable_results(frozen, source, root):
                    reused_by_index.setdefault(outcome["execution_index"], outcome)
            definition["execution_results"] = list(reused_by_index.values())
            retained_indexes = {
                entry["execution_index"]
                for entry in definition["execution_schedule"]
                if entry["run"]["task_id"] in retained
            }
            reused_indexes = {
                row["execution_index"] for row in definition["execution_results"]
            }
            if any(arm.variable == UPDATE_COUNT for arm in arms):
                original_indexes = {
                    entry["execution_index"] for entry in definition["execution_schedule"]
                    if not entry["run"]["task_id"].endswith("-update-count")
                    and (not (gate or {}).get("repetition_extension")
                         or entry["run"]["replicate_id"] == 0)
                }
                if not original_indexes <= reused_indexes:
                    raise ValueError("original composite reuse is incomplete; originals must not rerun")
            if (gate or {}).get("repetition_extension"):
                original_indexes = {entry["execution_index"]
                                    for entry in definition["execution_schedule"]
                                    if entry["run"]["replicate_id"] == 0}
                if not original_indexes <= reused_indexes:
                    raise ValueError("original first-round reuse is incomplete; do not rerun it")
            if not retained_indexes <= reused_indexes:
                raise ValueError("retained site did not restore every requested valid execution")
            definition["reuse_from"] = str(reuse_sources[0].resolve())
            if len(reuse_sources) > 1:
                definition["reuse_sources"] = [str(source.resolve()) for source in reuse_sources]
        definition.update(created_at=local_timestamp(), **output_labels(arms))
        reused = definition.pop("execution_results", None)
        write_json_atomic(staging / "plan.json", definition)
        if reused is not None:
            write_json_atomic(staging / "results.json", {"execution_results": reused})
        staging.rename(destination)


def _reusable_results(
    arms: list[ControlledArm], source: Path, root: Path | None
) -> list[dict[str, Any]]:
    """Reference valid unchanged executions without copying or altering archives."""
    if root is None:
        raise ValueError("reusing runs requires the project root and frozen sites")
    source_study = read_study(source)
    if (source_study.get("review_gate") or {}).get("diagnostic_only") is True:
        return []
    old_arms, _, _ = load_frozen_study(source)
    old_by_name = {arm.name: arm for arm in old_arms}
    candidates: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for outcome in source_study.get("execution_results", []):
        if outcome.get("valid") is not True:
            continue
        record = load_run_record(Path(outcome["artifact_directory"]))
        snapshot = record["task_snapshot"]
        if not check_run(snapshot, record)["run_valid"]:
            continue
        run = outcome["run"]
        plan = old_by_name[run["arm"]].plan
        source_run = outcome.get("source_run", run)
        condition = record["condition"]["name"]
        record_plan = replace(
            plan,
            conditions={condition: Condition.from_dict(record["condition"])},
        )
        mismatch = protocol_mismatch(
            record,
            record_plan,
            source_run["schedule_index"],
            require_current_input_policy=True,
        )
        if mismatch or record["meta"].get("provider_seed") != run["provider_seed"]:
            continue
        inputs, frames = _path_independent_inputs(
            _execution_inputs(snapshot, record_plan, condition)
        )
        key = (inputs, run["replicate_id"])
        candidates[key].append(
            {**outcome, "record": record, "source_run": source_run, "frames": frames}
        )
    results = []
    by_name = {arm.name: arm for arm in arms}
    for entry in execution_schedule(arms):
        run = entry["run"]
        arm = by_name[run["arm"]]
        task = next(task for task in arm.tasks if task["id"] == run["task_id"])
        inputs, frames = _path_independent_inputs(
            _execution_inputs(task, arm.plan, run["condition"])
        )
        key = (inputs, run["replicate_id"])
        for candidate in candidates.get(key, []):
            record = candidate["record"]
            expected_first = (
                arm.first_run_ids.get(task["id"], {}).get(run["condition"])
                if run["replicate_id"] == 0 else None
            )
            if expected_first and record["meta"]["run_id"] != expected_first:
                continue
            historical_condition = Condition.from_dict(record["condition"])
            if protocol_mismatch(
                record, replace(arm.plan, conditions={historical_condition.name: historical_condition}),
                candidate["source_run"]["schedule_index"],
            ):
                continue
            if record["meta"].get("provider_seed") != run["provider_seed"]:
                continue
            site_problem = site_evidence_mismatch(record, root)
            if site_problem:
                if site_problem == "missing_current_site_evidence":
                    raise ValueError(
                        f"{task['id']}: missing historical website evidence ({site_problem}); "
                        "recover it before reusing this run; its existing result is unchanged"
                    )
                continue
            snapshots = (
                str(root / record["meta"]["runtime"]["site_snapshot"]),
                arm.plan.site_snapshot_for(task),
            )
            if not _same_public_site((record["task_snapshot"], task), snapshots):
                continue
            visual_problem = visual_mismatch(
                record, (candidate["frames"], frames), root,
                current_root=Path(arm.plan.frozen_asset_root) if arm.plan.frozen_asset_root else None,
            )
            if visual_problem == "missing_visual_evidence":
                raise ValueError(
                    f"{task['id']}: missing historical visual evidence; recover it before "
                    "exporting reused runs; existing results remain valid in their study"
                )
            if visual_problem:
                continue
            results.append(
                {
                    "execution_index": entry["execution_index"],
                    "run": run,
                    "source_run": candidate["source_run"],
                    "source_study": str(source.resolve()),
                    "artifact_directory": candidate["artifact_directory"],
                    "run_id": record["meta"]["run_id"],
                    "valid": True,
                    "success": rescore_run(task, record)["e2e_success"],
                    "scoring": {
                        "task_source": "current_frozen_task",
                        "historical_task_source": "run.task_snapshot",
                        "historical_success": check_run(
                            record["task_snapshot"], record
                        )["e2e_success"],
                    },
                }
            )
            break
    return results


def load_frozen_study(
    directory: Path,
) -> tuple[list[ControlledArm], dict[str, Any] | None, dict[str, Any] | None]:
    definition = read_study(directory)
    current = (directory / "plan.json").is_file()
    arms = []
    for row in definition["arms"]:
        if current:
            plan = parse_experiment_plan(row["plan"], str(directory / "plan.json"))
            tasks = load_tasks(
                directory / "tasks" / row["name"],
                experimental=COMPOSITE_TRACK in plan.track_conditions,
            )
        else:
            plan = load_experiment_plan(directory / row["name"] / "plan.json")
            tasks = load_tasks(directory / row["name"] / "tasks")
        arms.append(
            ControlledArm(
                row["name"], tasks, plan, row["variable"], row["level"], row["pair_id"],
                row.get("first_run_ids", {}),
                row.get("direction"),
            )
        )
    return arms, definition["evidence"], definition.get("review_gate")
