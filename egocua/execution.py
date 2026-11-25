"""One auditable Experiment Run, independent of the Agent Adapter."""

from __future__ import annotations

import asyncio
import time
import threading
from collections.abc import Awaitable, Callable, Coroutine, Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from .protocol import (
    RELEASE_DELIVERY_DESCRIPTIONS,
    STAGE_RELEASE_TRIGGERS,
    RunSession,
    first_release_trigger,
    validate_stage_budget,
)
from .timeline import (
    COMPOSITE_TRACK,
    COMPUTER_USE,
    HUMAN_SOLVABILITY,
    Condition,
    is_update_event,
)
from .environment import (
    create_site_runtime,
    load_site_data,
    site_snapshot,
    task_public_site_data,
)
from .experiment_results import write_experiment_results
from .messages import initial_context
from .final_state import evaluate_final_state
from .frame_inputs import ObservationImage, observation_images
from sites.runtime import ActionTrace
from .run_ledger import SITE_RUNTIME_CONTRACTS, RunLedger, build_run_meta
from .sampling_media import materialize_sampling_frames, prepare_session_tasks
from .workspace import (
    read_json_strict,
    resolve_within,
    write_bytes_atomic,
    write_json_atomic,
)

RELEASE_SEMANTICS_VERSION = "first-native-commitment-history-v2"
_BACKGROUND_SHUTDOWN_SECONDS = 1.0
T = TypeVar("T")


def run_isolated(coroutine: Coroutine[Any, Any, T]) -> T:
    """Finish execution before bounded cleanup of third-party background tasks."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        pending = {task for task in asyncio.all_tasks(loop) if not task.done()}
        for task in pending:
            task.cancel()
        if pending:
            _, pending = loop.run_until_complete(
                asyncio.wait(pending, timeout=_BACKGROUND_SHUTDOWN_SECONDS)
            )
            for task in pending:
                task._log_destroy_pending = False  # type: ignore[attr-defined]
        asyncio.set_event_loop(None)
        loop.close()


class RunStepLimitExceeded(RuntimeError):
    """Raised when an adapter tries to act beyond the registered step budget."""


class RemoteRunError(RuntimeError):
    """Preserve the original exception details received from an isolated worker."""

    def __init__(self, details: dict[str, Any]) -> None:
        self.details = details
        super().__init__(f"{details['type']}: {details['message']}")


def exception_details(error: BaseException) -> dict[str, Any]:
    """Record causes and fallback contexts without request payloads or credentials."""
    if isinstance(error, RemoteRunError):
        return deepcopy(error.details)
    details: dict[str, Any] = {}
    current = details
    seen: set[int] = set()
    while True:
        seen.add(id(error))
        current.update(type=type(error).__name__, message=str(error))
        if isinstance(error, OSError) and error.errno is not None:
            current["errno"] = error.errno
        relation = "cause" if error.__cause__ is not None else "context"
        linked = error.__cause__ if relation == "cause" else error.__context__
        if linked is None or id(linked) in seen:
            break
        nested: dict[str, Any] = {}
        current[relation] = nested
        current = nested
        error = linked
    return details


def request_rejection_kind(details: dict[str, Any]) -> str | None:
    """Identify explicit provider input rejections, not model task failures."""
    messages = []
    current = details
    while current:
        messages.append(current.get("message", "").lower())
        current = current.get("cause") or current.get("context") or {}
    message = "\n".join(messages)
    if any(
        marker in message
        for marker in (
            "data_inspection_failed",
            "datainspectionfailed",
            "prohibited_content",
            "content_filter",
        )
    ):
        return "moderation"
    if any(
        marker in message
        for marker in (
            "maximum context length",
            "longer than the maximum model length",
            "range of input length should be",
        )
    ):
        return "context_limit"
    return None


@dataclass(frozen=True)
class RunConfig:
    """Everything that identifies and constrains one Experiment Run."""

    agent: str
    model: str
    default_release_step: int
    reference_steps: int | None
    release_trigger: str = "timeline_step"
    provider: str = "unspecified"
    model_version: str | None = None
    transport: str = "unspecified"
    max_steps: int | None = 40
    replicate_id: int = 0
    schedule_index: int | None = None
    provider_seed: int | None = None
    asset_root: str | Path = "."
    output_root: str | Path | None = None
    experiment_id: str = "manual"
    experiment_directory: str | Path | None = None
    operator_id: str | None = None
    pre_release_max_steps: int | None = None
    post_release_max_steps: int | None = None
    site_snapshot: str | Path | None = None
    run_purpose: str | None = None
    initial_site_state: dict[str, Any] | None = None
    frozen_asset_root: str | None = None
    provider_base_url: str | None = None

    def __post_init__(self) -> None:
        if self.default_release_step < 1:
            raise ValueError("default_release_step must be positive")
        if self.max_steps is None and self.agent != "human":
            raise ValueError("only human solvability runs may omit a step limit")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.replicate_id < 0:
            raise ValueError("replicate_id must be non-negative")
        if self.reference_steps is not None and self.reference_steps < 1:
            raise ValueError("reference_steps must be positive when provided")
        if self.release_trigger not in RELEASE_DELIVERY_DESCRIPTIONS:
            raise ValueError(f"unknown release_trigger {self.release_trigger!r}")
        validate_stage_budget(
            self.release_trigger,
            (
                self.pre_release_max_steps,
                self.post_release_max_steps,
            ),
            self.max_steps,
        )
        if self.schedule_index is not None and self.schedule_index < 0:
            raise ValueError("schedule_index must be non-negative")
        if self.operator_id is not None and not self.operator_id.strip():
            raise ValueError("operator_id must be non-empty when provided")
        object.__setattr__(self, "asset_root", Path(self.asset_root).resolve())
        if self.output_root is not None:
            object.__setattr__(self, "output_root", Path(self.output_root).resolve())


def require_stream_input(
    condition: Condition,
    release_trigger: str,
    max_steps: int,
    *,
    track: str | None = None,
) -> None:
    """Reject retired execution plans before starting a live run."""
    if condition.schedule != COMPUTER_USE and (
        condition.frame_delivery_steps != 1 or condition.batch_seconds is None
    ):
        raise ValueError("live runs require a fixed-time frame batch at every step")
    if release_trigger != "timeline_step":
        raise ValueError("website events cannot release live task inputs")
    if max_steps != 20 and not (track == COMPOSITE_TRACK and max_steps == 30):
        raise ValueError(
            "live runs use a 20-step budget; formal composite tasks use 30"
        )


def prepare_live_task(
    task: dict[str, Any], condition: Condition, config: RunConfig
) -> dict[str, Any]:
    require_stream_input(
        condition, config.release_trigger, config.max_steps, track=task["track"]
    )
    return prepare_visual_task(task, condition, config.asset_root)


def prepare_visual_task(
    task: dict[str, Any], condition: Condition, asset_root: str | Path
) -> dict[str, Any]:
    """Prepare shared visual materials, independently of an execution budget."""
    if condition.schedule == COMPUTER_USE:
        return task
    if "visual_session" not in task:
        root = Path(asset_root)
        (task,) = prepare_session_tasks(
            [task], root, float(condition.frame_interval_seconds)
        )
        if condition.scope != "none":
            materialize_sampling_frames([task], root)
    return task


@dataclass
class RunContext:
    """The small Interface an Agent Adapter uses while acting."""

    initial_payload: dict[str, Any]
    base_url: str
    site_data: dict[str, Any]
    _session: RunSession
    _state: Any
    _trace: ActionTrace
    _max_steps: int | None
    _release_trigger: str = "timeline_step"
    _partial_trace: Any = field(default_factory=list)
    _pre_release_max_steps: int | None = None
    _post_release_max_steps: int | None = None
    _delivery_step: int | None = None
    _asset_root: Path = Path(".")
    _visual_asset_root: Path | None = None
    _website_image_root: Path | None = None
    _progress_path: Path | None = None
    _progress_lock: Any = field(default_factory=threading.Lock, repr=False)

    def observation_images(self, observation: dict[str, Any]) -> list[ObservationImage]:
        """Archive delivered frames and render the model's images from that copy.

        Runs of one experiment share the archive: a frame is stored when it is
        first delivered and later runs read that same copy.
        """
        archive = self._visual_asset_root
        if archive is None:
            return observation_images(observation, self._asset_root)
        for path in observation["frame_paths"]:
            target = resolve_within(
                archive, path, context="archived frame", must_exist=False
            )
            if not target.is_file():
                source = resolve_within(self._asset_root, path, context="wearer frame")
                write_bytes_atomic(target, source.read_bytes())
        return observation_images(observation, archive)

    @property
    def task(self) -> dict[str, Any]:
        return self._session.task

    @property
    def condition(self) -> Condition:
        return self._session.timeline.condition

    @property
    def current_step(self) -> int:
        return self._trace.current_step

    def advance(self) -> list[dict[str, Any]]:
        """Advance one Agent step and deliver any due protocol event exactly once."""

        if (
            self.condition.schedule == HUMAN_SOLVABILITY
            or self.condition.schedule == COMPUTER_USE
            or self.condition.frame_delivery_steps is not None
        ):
            if self._max_steps is not None and self.current_step >= self._max_steps:
                raise RunStepLimitExceeded(
                    f"run exceeded its {self._max_steps}-step active budget"
                )
            self._trace.current_step += 1
            events = self._session.advance(self.current_step)
            for event in events:
                self._trace.record_event(event)
            return events

        return self._advance_archived_protocol()

    def _advance_archived_protocol(self) -> list[dict[str, Any]]:
        """Deterministic reconstruction for offline checks of old studies."""
        trigger = first_release_trigger(self._trace.actions, self._release_trigger)
        if self._delivery_step is not None and self._post_release_max_steps is not None:
            limit = self._delivery_step + self._post_release_max_steps - 1
        elif trigger is None and self._pre_release_max_steps is not None:
            limit = self._pre_release_max_steps
        else:
            limit = self._max_steps
        if self._trace.current_step >= limit:
            raise RunStepLimitExceeded(f"run exceeded its {limit}-step active budget")
        self._trace.current_step += 1
        action_triggered = self._release_trigger != "timeline_step"
        release_pending = action_triggered and trigger is not None
        events = self._session.advance(
            self._trace.current_step,
            release_pending=release_pending,
            allow_scheduled_release=not action_triggered,
            trigger_reason=self._release_trigger if release_pending else None,
        )
        for event in events:
            if trigger is not None:
                event["trigger_action_index"] = trigger[0]
                event["trigger_action_step"] = trigger[1]["step"]
            self._trace.record_event(event)
        if self._delivery_step is None and any(map(is_update_event, events)):
            self._delivery_step = self.current_step
        return events

    def wait(self) -> None:
        """Record that the Agent deliberately took no site action this step."""

        self._trace.record("wait")

    def standby(self) -> bool:
        """After the Agent finishes: True when a later wearer request will wake it."""

        event = self._session.standby(self._trace.current_step)
        if event is None:
            return False
        self._trace.record_event(event)
        return True

    def ask_user(self, question: str) -> dict[str, Any]:
        """Ask the wearer one free-form question."""

        event = self._session.ask_user(question, self._trace.current_step)
        self._trace.record_event(event)
        return event

    def archive_browser_screenshot(self, image: bytes) -> str | None:
        """Store the exact browser image sent to the model in this run's archive."""
        if self._website_image_root is None:
            return None
        path = self._website_image_root / f"step-{self.current_step}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image)
        return str(path)

    def capture_partial_trace(self, trace: Any) -> None:
        """Keep the latest reconstructable trace if the Adapter later fails."""

        self._partial_trace = trace

    def save_progress(self) -> None:
        """Save an unfinished local checkpoint, separate from a scored Run Artifact."""
        if self._progress_path is None:
            return
        with self._progress_lock:
            write_json_atomic(self._progress_path, {
                "run_status": {"valid": False, "reason": "in_progress"},
                "steps": self.current_step,
                "actions": list(self._trace.actions),
                "live_state": self._state.live_state(),
                "trace": self._partial_trace,
            })

    def track_website_progress(self) -> None:
        """Persist native website actions without introducing action limits."""
        self._trace.on_record = self.save_progress
        self.save_progress()


SyncAdapter = Callable[[RunContext], Any]
AsyncAdapter = Callable[[RunContext], Awaitable[Any]]


def run_experiment(
    task: dict[str, Any], condition: Condition, config: RunConfig, adapter: SyncAdapter
) -> dict[str, Any]:
    """Run a synchronous Agent Adapter behind the Experiment Run seam."""

    lifecycle = _RunLifecycle(task, condition, config)
    with lifecycle.execution() as context:
        lifecycle.adapter_trace = adapter(context)
    return lifecycle.complete(lifecycle.adapter_trace)


async def run_experiment_async(
    task: dict[str, Any], condition: Condition, config: RunConfig, adapter: AsyncAdapter
) -> dict[str, Any]:
    """Run an asynchronous Agent Adapter behind the same Experiment Run seam."""

    lifecycle = _RunLifecycle(task, condition, config)
    with lifecycle.execution() as context:
        lifecycle.adapter_trace = await adapter(context)
    return lifecycle.complete(lifecycle.adapter_trace)


class _RunLifecycle:
    def __init__(self, task: dict[str, Any], condition: Condition, config: RunConfig):
        if condition.schedule == HUMAN_SOLVABILITY and config.agent != "human":
            raise ValueError("human solvability is a human-only condition")
        if config.max_steps is None and condition.schedule != HUMAN_SOLVABILITY:
            raise ValueError(
                "an unlimited run must use the human solvability condition"
            )
        self.task = task
        self.condition = condition
        self.config = config
        self.site_data = (
            read_json_strict(Path(config.site_snapshot))
            if config.site_snapshot is not None
            else load_site_data(config.asset_root, str(task["site"]))
        )
        if self.site_data["site"] != task["site"]:
            raise ValueError("website snapshot belongs to a different site")
        self.initial_site_state = (
            deepcopy(config.initial_site_state)
            if config.initial_site_state is not None
            else {"selected_entity_ids": [], "fields": {}}
        )
        if config.initial_site_state is not None:
            if task["site"] != "egomart" or task["track"] != "graft":
                raise ValueError("initial_site_state requires an EgoMart Graft task")
            if not evaluate_final_state(
                task["graft"]["initial_goal"], self.initial_site_state
            ):
                raise ValueError("initial_site_state does not satisfy the initial goal")
            public = task_public_site_data(task, self.site_data)
            public_ids = {entity["entity_id"] for entity in public["entities"]}
            if set(self.initial_site_state["selected_entity_ids"]) - public_ids:
                raise ValueError(
                    "initial_site_state contains entities outside the public site"
                )
            if set(self.initial_site_state["fields"]) - set(public["supported_fields"]):
                raise ValueError("initial_site_state contains unsupported fields")
        self.session = RunSession(
            task,
            condition,
            event_step=config.default_release_step,
            reference_steps=config.reference_steps,
        )
        if (
            config.release_trigger in STAGE_RELEASE_TRIGGERS
            and not self.session.timeline.releases
        ):
            raise ValueError(
                "stage release experiments require an online task with a pending update"
            )
        self.initial_payload = initial_context(
            self.session.initial_payload(), config.initial_site_state
        )
        self.trace = ActionTrace()
        self.adapter_trace: Any = []
        self.runtime: Any | None = None
        self.started_at = time.time()
        self.meta: dict[str, Any] = {
            "agent": config.agent,
            "model": config.model,
            "provider": config.provider,
            "model_version": config.model_version,
            "transport": config.transport,
            "replicate_id": config.replicate_id,
            "schedule_index": config.schedule_index,
            "provider_seed": config.provider_seed,
            "randomness_control": (
                "provider_seed" if config.provider_seed is not None else "uncontrolled"
            ),
            "max_steps": config.max_steps,
            "operator_id": config.operator_id,
            "default_release_step": config.default_release_step,
            "reference_steps": config.reference_steps,
            "site": task["site"],
            "runtime": {
                **(
                    {"provider_base_url": config.provider_base_url}
                    if config.provider_base_url
                    else {}
                ),
                "site_contract": SITE_RUNTIME_CONTRACTS[task["site"]],
                "release_trigger": config.release_trigger,
                "release_semantics": (
                    "native-stage-release-v1"
                    if config.release_trigger in STAGE_RELEASE_TRIGGERS
                    else RELEASE_SEMANTICS_VERSION
                ),
                "pre_release_max_steps": config.pre_release_max_steps,
                "post_release_max_steps": config.post_release_max_steps,
                "initial_site_state": deepcopy(config.initial_site_state),
                "site_snapshot": str(config.site_snapshot)
                if config.site_snapshot is not None
                else None,
            },
        }
        if config.run_purpose == "task_validation":
            self.meta.update(
                run_purpose=config.run_purpose,
                used_for_task_repair=True,
                valid_as_held_out_difficulty_evidence=False,
            )
        self.ledger = (
            RunLedger.start(
                config.output_root,
                experiment_id=config.experiment_id,
                task_id=task["id"],
                condition=condition,
                replicate_id=config.replicate_id,
                agent=config.agent,
                experiment_directory=config.experiment_directory,
                model=config.model,
                configuration={
                    "max_steps": config.max_steps,
                    "provider_seed": config.provider_seed,
                    "replicate_id": config.replicate_id,
                    "release_trigger": config.release_trigger,
                    "default_release_step": config.default_release_step,
                    "reference_steps": config.reference_steps,
                    "pre_release_max_steps": config.pre_release_max_steps,
                    "post_release_max_steps": config.post_release_max_steps,
                },
            )
            if config.output_root is not None
            else None
        )

    @contextmanager
    def execution(self) -> Iterator[RunContext]:
        try:
            self.prepare()
            with self.open_runtime().server as server:
                context = self.context(server.base_url)
                try:
                    yield context
                except RunStepLimitExceeded:
                    self.adapter_trace = context._partial_trace
                    self.meta["termination_reason"] = "step_limit"
                except BaseException:
                    self.adapter_trace = context._partial_trace
                    raise
        except BaseException as error:
            self.fail(self.adapter_trace, error)
            raise

    def prepare(self) -> None:
        prepared_meta = build_run_meta(
            self.task,
            asset_root=self.config.asset_root,
            agent=self.config.agent,
            model=self.config.model,
            provider=self.config.provider,
            model_version=self.config.model_version,
            transport=self.config.transport,
            replicate_id=self.config.replicate_id,
            provider_seed=self.config.provider_seed,
            max_steps=self.config.max_steps,
        )
        prepared_meta["runtime"].update(self.meta["runtime"])
        if self.ledger and self.config.site_snapshot is None:
            path = site_snapshot(
                self.ledger.experiment.parent, self.task["site"], self.site_data
            )
            prepared_meta["runtime"]["site_snapshot"] = path.as_posix()
        self.meta.update(prepared_meta)

    def context(self, base_url: str) -> RunContext:
        if self.runtime is None:
            raise RuntimeError("run runtime has not been opened")
        return RunContext(
            initial_payload=self.initial_payload,
            base_url=base_url,
            site_data=task_public_site_data(self.task, self.site_data),
            _session=self.session,
            _state=self.runtime.state,
            _trace=self.trace,
            _max_steps=self.config.max_steps,
            _release_trigger=self.config.release_trigger,
            _pre_release_max_steps=self.config.pre_release_max_steps,
            _post_release_max_steps=self.config.post_release_max_steps,
            _asset_root=Path(self.config.frozen_asset_root or self.config.asset_root),
            _visual_asset_root=self._wearer_archive(),
            _website_image_root=self.ledger.directory / "images" / "website"
            if self.ledger
            else None,
            _progress_path=self.ledger.directory / "progress.json"
            if self.ledger
            else None,
        )

    def _wearer_archive(self) -> Path | None:
        """Delivered glasses frames, stored once per experiment directory."""
        if self.config.frozen_asset_root is not None:
            return Path(self.config.frozen_asset_root)
        if self.ledger is None:
            return None
        return self.ledger.experiment / "images" / "wearer"

    def open_runtime(self) -> Any:
        if self.runtime is not None:
            raise RuntimeError("run runtime has already been opened")
        self.runtime = create_site_runtime(
            self.task,
            self.site_data,
            asset_root=self.config.frozen_asset_root or self.config.asset_root,
            trace=self.trace,
            initial_site_state=self.config.initial_site_state,
        )
        return self.runtime

    def complete(self, trace: Any) -> dict[str, Any]:
        record = self._record(trace, valid=True, reason=None)
        if self.ledger:
            self.ledger.complete(record)
            if self.config.experiment_directory is None:
                write_experiment_results(self.ledger.experiment)
        return record

    def fail(self, trace: Any, error: BaseException) -> None:
        record = self._record(trace, valid=False, reason="infrastructure_error")
        if self.ledger:
            self.ledger.fail(record, error)
            if self.config.experiment_directory is None:
                write_experiment_results(self.ledger.experiment)
            error.run_artifact = {  # type: ignore[attr-defined]
                "artifact_directory": record["meta"]["artifact_directory"],
                "run_id": record["meta"]["run_id"],
            }

    def _record(self, trace: Any, *, valid: bool, reason: str | None) -> dict[str, Any]:
        record: dict[str, Any] = {
            "task_id": self.task["id"],
            "track": self.task["track"],
            "condition": self.condition.as_dict(),
            "run_status": {"valid": valid, "reason": reason},
            "initial_state": {
                **deepcopy(self.initial_site_state),
                "submission_completed": False,
            },
            "steps": self.trace.current_step,
            "actions": list(self.trace.actions),
            "final_state": (
                self.runtime.state.final_state()
                if self.runtime is not None
                else {
                    "selected_entity_ids": [],
                    "fields": {},
                    "submission_completed": False,
                }
            ),
        }
        record["trace"] = trace
        record["task_snapshot"] = deepcopy(self.task)
        record["initial_context"] = deepcopy(self.initial_payload)
        if self.runtime is not None:
            record["live_state"] = self.runtime.state.live_state()
        record["meta"] = {
            **self.meta,
            "started_at": self.started_at,
            "duration_seconds": round(time.time() - self.started_at, 1),
            "event_delivery_steps": [
                action["step"]
                for action in self.trace.actions
                if action.get("type")
                in {"instruction_injection", "visual_context_update"}
            ],
            "run_id": self.ledger.run_id if self.ledger else None,
            "artifact_directory": (
                self.ledger.directory.as_posix() if self.ledger else None
            ),
        }
        archive = self._wearer_archive()
        if archive is not None and archive.is_dir():
            record["meta"]["runtime"] = {
                **record["meta"]["runtime"],
                "visual_asset_root": archive.as_posix(),
            }
        if self.config.release_trigger in STAGE_RELEASE_TRIGGERS:
            record["meta"]["stage_release"] = self._stage_release_summary()
        record["meta"]["released_visual_inputs"] = [
            {"step": 0, **deepcopy(observation)}
            for observation in self.initial_payload["observations"]
        ] + [
            {"step": event["step"], **deepcopy(event["payload"])}
            for event in self.session.events
            if event.get("type") == "visual_context_update"
        ]
        return record

    def _stage_release_summary(self) -> dict[str, Any]:
        trigger = first_release_trigger(self.trace.actions, self.config.release_trigger)
        events = [event for event in self.session.events if is_update_event(event)]
        delivery_step = events[0]["step"] if events else None
        post_steps = (
            self.trace.current_step - delivery_step + 1
            if delivery_step is not None
            else 0
        )
        if trigger is None:
            status = "stage_not_reached"
        elif delivery_step is None:
            status = "trigger_not_delivered"
        elif (
            self.config.post_release_max_steps is not None
            and post_steps >= self.config.post_release_max_steps
        ):
            status = "post_release_budget_exhausted"
        else:
            status = "delivered"
        return {
            "status": status,
            "trigger_action_index": trigger[0] if trigger else None,
            "trigger_action": deepcopy(trigger[1]) if trigger else None,
            "delivery_step": delivery_step,
            "post_release_steps": post_steps,
            "post_release_budget_remaining": (
                max(0, self.config.post_release_max_steps - post_steps)
                if self.config.post_release_max_steps is not None
                else None
            ),
        }
