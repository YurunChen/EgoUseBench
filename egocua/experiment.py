from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlsplit


from .messages import model_input_policy
from .schema import EXPERIMENT_TRACKS
from .timeline import COMPOSITE_TRACK, Condition, TimelineError
from .protocol import RELEASE_DELIVERY_DESCRIPTIONS, validate_stage_budget


TASK_REVIEW_PLAN = "configs/pilot.json"
"""The plan whose conditions define construction checks and ad-hoc runs."""

FORMAL_PLAN = "configs/benchmark.json"


class ExperimentPlanError(ValueError):
    """Raised when an experiment plan is not internally consistent."""


@dataclass(frozen=True)
class ExperimentPlan:
    """Which conditions run on which tracks, how often, and with what runtime."""

    experiment_id: str
    track_conditions: dict[str, tuple[str, ...]]
    replicate_ids: tuple[int, ...]
    adapter: str
    model: str
    model_version: str
    max_steps: int
    scheduler_seed: int
    reference_condition_name: str
    conditions: dict[str, Condition]
    default_release_step: int
    reference_steps: int | None
    release_trigger: str = "timeline_step"
    track_probe_conditions: dict[str, tuple[str, ...]] = field(default_factory=dict)
    pre_release_max_steps: int | None = None
    post_release_max_steps: int | None = None
    site_snapshots: dict[str, str] = field(default_factory=dict)
    initial_site_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    composite_max_steps: int | None = None
    frozen_asset_root: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize the same plan contract accepted by parse_experiment_plan."""
        payload: dict[str, Any] = {
            "schema_version": 3,
            "experiment_id": self.experiment_id,
            "default_release_step": self.default_release_step,
            "reference_steps": self.reference_steps,
            "reference_condition": self.reference_condition_name,
            "conditions": [condition.as_dict() for condition in self.conditions.values()],
            "tracks": {key: list(value) for key, value in self.track_conditions.items()},
            "replicate_ids": list(self.replicate_ids),
            "scheduler_seed": self.scheduler_seed,
            "runtime": {
                "adapter": self.adapter,
                "model": self.model,
                "model_version": self.model_version,
                "max_steps": self.max_steps,
                "release_trigger": self.release_trigger,
                "pre_release_max_steps": self.pre_release_max_steps,
                "post_release_max_steps": self.post_release_max_steps,
                "site_snapshots": self.site_snapshots,
            },
        }
        if self.track_probe_conditions:
            payload["probes"] = {
                key: list(value) for key, value in self.track_probe_conditions.items()
            }
        if self.composite_max_steps is not None:
            payload["runtime"]["composite_max_steps"] = self.composite_max_steps
        if self.frozen_asset_root is not None:
            payload["runtime"]["frozen_asset_root"] = self.frozen_asset_root
        if self.initial_site_states:
            payload["runtime"]["initial_site_states"] = self.initial_site_states
        return payload

    def condition(self, name: str) -> Condition:
        try:
            return self.conditions[name]
        except KeyError as exc:
            raise ExperimentPlanError(f"unknown condition {name!r}") from exc

    def conditions_for(self, track: str) -> tuple[str, ...]:
        """The treatment conditions this track is compared across."""

        try:
            return self.track_conditions[track]
        except KeyError as exc:
            raise ExperimentPlanError(
                f"experiment plan does not include track {track!r}"
            ) from exc

    def probe_conditions_for(self, track: str) -> tuple[str, ...]:
        """The diagnostic conditions that establish the task needs its evidence.

        Probes answer whether a task is admissible, not how well an agent does, so
        they stay out of the treatment comparison, out of the smoke reference, and
        out of the release gates.
        """

        self.conditions_for(track)
        return self.track_probe_conditions.get(track, ())

    def enabled_conditions_for(self, track: str) -> tuple[str, ...]:
        return self.conditions_for(track) + self.probe_conditions_for(track)

    def validate_condition(self, track: str, condition_name: str) -> None:
        if condition_name not in self.enabled_conditions_for(track):
            raise ExperimentPlanError(
                f"condition {condition_name!r} is not enabled for track {track!r}"
            )

    def reference_condition_for(self, track: str) -> Condition:
        self.validate_condition(track, self.reference_condition_name)
        return self.condition(self.reference_condition_name)

    def run_parameters(
        self, task: Mapping[str, Any], run: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Translate one registered schedule entry into adapter runtime settings."""
        self.validate_condition(str(task["track"]), str(run["condition"]))
        return {
            "model": self.model,
            "model_version": self.model_version,
            "replicate_id": run["replicate_id"],
            "provider_seed": run["provider_seed"],
            "schedule_index": run["schedule_index"],
            "experiment_id": self.experiment_id,
            **self.protocol_parameters(task),
        }

    def site_snapshot_for(self, task: Mapping[str, Any]) -> str | None:
        """A retained task snapshot takes precedence over its site's default."""
        return self.site_snapshots.get(str(task["id"]), self.site_snapshots.get(str(task["site"])))

    def protocol_parameters(self, task: Mapping[str, Any]) -> dict[str, Any]:
        """Settings every adapter must take from the plan, whoever drives the run."""
        return {
            "max_steps": self.max_steps,
            **({"frozen_asset_root": self.frozen_asset_root} if self.frozen_asset_root else {}),
            "site_snapshot": self.site_snapshot_for(task),
            "initial_site_state": self.initial_site_states.get(str(task["id"])),
            "default_release_step": self.default_release_step,
            "reference_steps": self.reference_steps,
            "release_trigger": self.release_trigger,
            "pre_release_max_steps": self.pre_release_max_steps,
            "post_release_max_steps": self.post_release_max_steps,
        }


def _parse_track_conditions(
    raw: Any, known_conditions: set[str], context: str
) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not raw:
        raise ExperimentPlanError(f"{context} must map tracks to conditions")
    parsed: dict[str, tuple[str, ...]] = {}
    for track, selected in raw.items():
        if track not in EXPERIMENT_TRACKS:
            raise ExperimentPlanError(f"{context}: unsupported track {track!r}")
        if (
            not isinstance(selected, list)
            or not selected
            or len(selected) != len(set(selected))
        ):
            raise ExperimentPlanError(
                f"{context}: {track} conditions must be a non-empty unique list"
            )
        unknown = set(selected) - known_conditions
        if unknown:
            raise ExperimentPlanError(
                f"{context}: {track} references unknown conditions {sorted(unknown)}"
            )
        parsed[track] = tuple(str(name) for name in selected)
    return parsed


def load_experiment_plan(path: str | Path, arm: str | None = None) -> ExperimentPlan:
    source = Path(path)
    payload = json.loads(source.read_text())
    if arm is not None:
        payload = next(row["plan"] for row in payload["arms"] if row["name"] == arm)
    return parse_experiment_plan(payload, source.as_posix())


def parse_experiment_plan(payload: dict[str, Any], source: str) -> ExperimentPlan:
    if not isinstance(payload, dict):
        raise ExperimentPlanError(f"{source}: experiment plan must be a JSON object")
    if payload.get("schema_version") != 3:
        raise ExperimentPlanError(f"{source}: experiment plan schema_version must be 3")
    allowed_fields = {
        "schema_version",
        "experiment_id",
        "default_release_step",
        "reference_steps",
        "conditions",
        "tracks",
        "probes",
        "replicate_ids",
        "runtime",
        "scheduler_seed",
        "reference_condition",
    }
    unknown_fields = set(payload) - allowed_fields
    if unknown_fields:
        raise ExperimentPlanError(
            f"{source}: experiment plan has unknown fields {sorted(unknown_fields)}"
        )

    experiment_id = payload.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ExperimentPlanError(f"{source}: experiment_id must be non-empty")

    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise ExperimentPlanError(f"{source}: conditions must be a non-empty list")
    conditions: dict[str, Condition] = {}
    for entry in raw_conditions:
        try:
            condition = Condition.from_dict(entry)
        except TimelineError as exc:
            raise ExperimentPlanError(f"{source}: invalid condition: {exc}") from exc
        if condition.name in conditions:
            raise ExperimentPlanError(
                f"{source}: duplicate condition {condition.name!r}"
            )
        conditions[condition.name] = condition

    track_conditions = _parse_track_conditions(
        payload.get("tracks"), set(conditions), f"{source}: tracks"
    )
    track_probe_conditions = (
        _parse_track_conditions(payload["probes"], set(conditions), f"{source}: probes")
        if "probes" in payload
        else {}
    )
    unreachable = set(conditions) - {
        name
        for selected in (*track_conditions.values(), *track_probe_conditions.values())
        for name in selected
    }
    if unreachable:
        raise ExperimentPlanError(
            f"{source}: conditions {sorted(unreachable)} are declared but no track runs "
            f"them; list them under tracks or probes, or delete them"
        )
    overlapping = {
        track
        for track, probes in track_probe_conditions.items()
        if set(probes) & set(track_conditions.get(track, ()))
    }
    if overlapping:
        raise ExperimentPlanError(
            f"{source}: tracks {sorted(overlapping)} list the same condition as both a "
            f"treatment and a probe"
        )

    reference_condition_name = payload.get("reference_condition")
    if (
        not isinstance(reference_condition_name, str)
        or not reference_condition_name.strip()
    ):
        raise ExperimentPlanError(f"{source}: reference_condition is required")
    if reference_condition_name not in conditions:
        raise ExperimentPlanError(
            f"{source}: unknown reference_condition {reference_condition_name!r}"
        )
    missing_reference = [
        track
        for track, selected in track_conditions.items()
        if reference_condition_name not in selected
    ]
    if missing_reference:
        raise ExperimentPlanError(
            f"{source}: reference_condition is not enabled for tracks {missing_reference}"
        )

    raw_replicates = payload.get("replicate_ids")
    if (
        not isinstance(raw_replicates, list)
        or not raw_replicates
        or len(raw_replicates) != len(set(raw_replicates))
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in raw_replicates
        )
    ):
        raise ExperimentPlanError(
            f"{source}: replicate_ids must be a non-empty unique list of non-negative integers"
        )

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        raise ExperimentPlanError(f"{source}: runtime must be an object")
    required_runtime_fields = {"adapter", "model", "model_version", "max_steps"}
    optional_runtime_fields = {
        "release_trigger",
        "pre_release_max_steps",
        "post_release_max_steps",
        "site_snapshots",
        "initial_site_states",
        "composite_max_steps",
        "frozen_asset_root",
    }
    if not required_runtime_fields <= set(runtime) or set(runtime) - (
        required_runtime_fields | optional_runtime_fields
    ):
        raise ExperimentPlanError(
            f"{source}: runtime fields must contain {sorted(required_runtime_fields)} and may "
            f"contain {sorted(optional_runtime_fields)}"
        )
    runtime_identity = [
        runtime[field] for field in ("adapter", "model", "model_version")
    ]
    if any(
        not isinstance(value, str) or not value.strip() for value in runtime_identity
    ):
        raise ExperimentPlanError(
            f"{source}: runtime adapter, model, and model_version are required"
        )
    adapter, model, model_version = (str(value) for value in runtime_identity)
    if adapter != "browsergym" and any(c.visual_history for c in conditions.values()):
        raise ExperimentPlanError(f"{source}: visual_history requires BrowserGym")
    max_steps = runtime.get("max_steps")
    if not isinstance(max_steps, int) or isinstance(max_steps, bool) or max_steps < 1:
        raise ExperimentPlanError(f"{source}: runtime.max_steps must be positive")
    composite_max_steps = runtime.get("composite_max_steps")
    if composite_max_steps is not None and (
        not isinstance(composite_max_steps, int)
        or isinstance(composite_max_steps, bool)
        or composite_max_steps < 1
    ):
        raise ExperimentPlanError(f"{source}: runtime.composite_max_steps must be positive")
    frozen_asset_root = runtime.get("frozen_asset_root")
    if frozen_asset_root is not None and (
        not isinstance(frozen_asset_root, str) or not frozen_asset_root.strip()
    ):
        raise ExperimentPlanError(f"{source}: runtime.frozen_asset_root must be a non-empty path")
    release_trigger = runtime.get("release_trigger", "timeline_step")
    if release_trigger not in RELEASE_DELIVERY_DESCRIPTIONS:
        raise ExperimentPlanError(
            f"{source}: unknown runtime.release_trigger {release_trigger!r}"
        )
    pre_release_max_steps = runtime.get("pre_release_max_steps")
    post_release_max_steps = runtime.get("post_release_max_steps")
    try:
        validate_stage_budget(
            release_trigger, (pre_release_max_steps, post_release_max_steps), max_steps
        )
    except ValueError as exc:
        raise ExperimentPlanError(f"{source}: {exc}") from exc
    scheduler_seed = payload.get("scheduler_seed")
    if (
        not isinstance(scheduler_seed, int)
        or isinstance(scheduler_seed, bool)
        or scheduler_seed < 0
    ):
        raise ExperimentPlanError(
            f"{source}: scheduler_seed must be a non-negative integer"
        )
    if "default_release_step" not in payload:
        raise ExperimentPlanError(f"{source}: default_release_step is required")
    default_release_step = payload["default_release_step"]
    if (
        not isinstance(default_release_step, int)
        or isinstance(default_release_step, bool)
        or default_release_step < 1
    ):
        raise ExperimentPlanError(
            f"{source}: default_release_step must be a positive integer"
        )
    if "reference_steps" not in payload:
        raise ExperimentPlanError(f"{source}: reference_steps is required")
    reference_steps = payload["reference_steps"]
    if reference_steps is not None and (
        not isinstance(reference_steps, int)
        or isinstance(reference_steps, bool)
        or reference_steps < 1
    ):
        raise ExperimentPlanError(
            f"{source}: reference_steps must be a positive integer"
        )
    if any(condition.release_at is not None for condition in conditions.values()) and (
        reference_steps is None
    ):
        raise ExperimentPlanError(
            f"{source}: conditions using release_at require reference_steps"
        )

    return ExperimentPlan(
        experiment_id=experiment_id,
        track_conditions=track_conditions,
        track_probe_conditions=track_probe_conditions,
        replicate_ids=tuple(raw_replicates),
        adapter=adapter,
        model=model,
        model_version=model_version,
        max_steps=max_steps,
        scheduler_seed=scheduler_seed,
        reference_condition_name=reference_condition_name,
        conditions=conditions,
        default_release_step=default_release_step,
        reference_steps=reference_steps,
        release_trigger=str(release_trigger),
        pre_release_max_steps=pre_release_max_steps,
        post_release_max_steps=post_release_max_steps,
        site_snapshots=runtime.get("site_snapshots", {}),
        initial_site_states=runtime.get("initial_site_states", {}),
        composite_max_steps=composite_max_steps,
        frozen_asset_root=frozen_asset_root,
    )


def _expand(
    tasks: Iterable[dict[str, Any]],
    plan: ExperimentPlan,
    conditions_of: Callable[[str], tuple[str, ...]],
) -> list[dict[str, Any]]:
    ordered_tasks = sorted(tasks, key=lambda task: task["id"])
    runs = [
        {
            "task_id": task["id"],
            "track": task["track"],
            "condition": condition_name,
            "replicate_id": replicate_id,
            "provider_seed": plan.scheduler_seed + replicate_id,
        }
        for task in ordered_tasks
        for condition_name in conditions_of(task["track"])
        for replicate_id in plan.replicate_ids
    ]
    random.Random(plan.scheduler_seed).shuffle(runs)
    return [{**run, "schedule_index": index} for index, run in enumerate(runs)]


def expand_runs(
    tasks: Iterable[dict[str, Any]], plan: ExperimentPlan
) -> list[dict[str, Any]]:
    """The treatment schedule: what the experiment compares."""

    return _expand(tasks, plan, plan.conditions_for)


def expand_probe_runs(
    tasks: Iterable[dict[str, Any]], plan: ExperimentPlan
) -> list[dict[str, Any]]:
    """The probe schedule: what establishes the tasks need their evidence at all."""

    return _expand(tasks, plan, plan.probe_conditions_for)


def scheduled_indexes(
    tasks: Iterable[dict[str, Any]], plan: ExperimentPlan
) -> dict[tuple[Any, Any, Any], int]:
    """Registered schedule index for every treatment and probe run the plan defines.

    Treatment and probe conditions are disjoint per track, so the two schedules
    share no key and a run can be looked up without knowing which one it came from.
    """

    materialized = list(tasks)
    return {
        (run["task_id"], run["condition"], run["replicate_id"]): run["schedule_index"]
        for schedule in (
            expand_runs(materialized, plan),
            expand_probe_runs(materialized, plan),
        )
        for run in schedule
    }


def protocol_mismatch(
    record: Mapping[str, Any],
    plan: ExperimentPlan,
    schedule_index: int,
    *,
    require_current_input_policy: bool = False,
) -> str | None:
    """Return the first mismatch against the registered run protocol.

    Input-policy comparison is only for deciding whether an old run can stand
    in for a new formal or controlled execution. Historical scoring leaves it
    off, so a recorded v4 run can still be read as its own protocol.
    """

    meta = record.get("meta")
    if not isinstance(meta, Mapping):
        return "missing_meta"
    expected = {
        "agent": plan.adapter,
        "model": plan.model,
        "model_version": plan.model_version,
        "max_steps": plan.max_steps,
        "schedule_index": schedule_index,
        "default_release_step": plan.default_release_step,
        "reference_steps": plan.reference_steps,
    }
    for field_name, value in expected.items():
        if field_name not in meta:
            return f"missing_{field_name}"
        if meta[field_name] != value:
            return f"wrong_{field_name}"
    runtime = meta.get("runtime")
    if not isinstance(runtime, Mapping):
        return "missing_runtime"
    # The official endpoint is a composite requirement, not an asset-storage policy.
    if (
        plan.model == "deepseek-flash"
        and COMPOSITE_TRACK in plan.track_conditions
        and plan.frozen_asset_root is not None
    ):
        endpoint = urlsplit(str(runtime.get("provider_base_url", "")))
        if meta.get("provider") != "deepseek-official" or endpoint.scheme != "https" or endpoint.hostname != "api.deepseek.com":
            return "unverified_official_provider"
    if "release_trigger" not in runtime:
        return "missing_release_trigger"
    if runtime["release_trigger"] != plan.release_trigger:
        return "wrong_release_trigger"
    for field_name in ("pre_release_max_steps", "post_release_max_steps"):
        if runtime.get(field_name) != getattr(plan, field_name):
            return f"wrong_{field_name}"
    condition = record.get("condition")
    if not isinstance(condition, Mapping):
        return "wrong_condition"
    name = condition.get("name")
    if not isinstance(name, str) or name not in plan.conditions:
        return "wrong_condition"
    if condition != plan.condition(name).as_dict():
        return "wrong_condition"
    if require_current_input_policy and plan.adapter == "browsergym":
        return _input_policy_mismatch(record)
    return None


def _input_policy_mismatch(record: Mapping[str, Any]) -> str | None:
    trace = record.get("trace")
    settings = trace.get("agent_settings") if isinstance(trace, Mapping) else None
    if not isinstance(settings, Mapping) or "system_prompt_policy" not in settings:
        return "missing_system_prompt_policy"
    if settings["system_prompt_policy"] != model_input_policy(
        record["track"], record["condition"]
    ):
        return "wrong_system_prompt_policy"
    return None


def validate_run_records(
    tasks: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    plan: ExperimentPlan,
) -> None:
    """Every task and replicate of each submitted condition appears exactly once.

    A scoring pass covers one slice — the treatment matrix, or a probe sweep — so
    completeness is judged against the conditions the records actually carry.
    Whether a release has run every condition it owes is a gate question, not this one.
    """

    materialized_records = list(records)
    actual = [
        (
            record.get("task_id"),
            (record.get("condition") or {}).get("name"),
            record.get("meta", {}).get("replicate_id"),
        )
        for record in materialized_records
    ]
    submitted_conditions = {condition for _, condition, _ in actual}
    expected = {
        key: index
        for key, index in scheduled_indexes(tasks, plan).items()
        if key[1] in submitted_conditions
    }
    duplicates = sorted(key for key, count in Counter(actual).items() if count > 1)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    if duplicates or missing or unexpected:
        raise ExperimentPlanError(
            "run matrix must contain each task/condition/replicate exactly once; "
            f"missing={missing[:5]}, duplicates={duplicates[:5]}, unexpected={unexpected[:5]}"
        )
    invalid_identity = []
    for record, key in zip(materialized_records, actual, strict=True):
        mismatch = protocol_mismatch(record, plan, expected[key])
        if mismatch is not None:
            invalid_identity.append((key, mismatch))
        steps = record.get("steps")
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or not 0 <= steps <= plan.max_steps
        ):
            invalid_identity.append(
                (key, f"wrong_steps: {steps!r}; expected 0..{plan.max_steps}")
            )
    if invalid_identity:
        raise ExperimentPlanError(
            "run records disagree with the registered schedule/runtime; "
            f"mismatches={invalid_identity[:3]}"
        )
