"""One auditable Experiment Run, independent of the Agent Adapter."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .protocol import RunSession
from .environment import create_site_runtime, load_site_data
from .run_ledger import RunLedger, build_run_meta
from .workspace import write_json_atomic


@dataclass(frozen=True)
class RunConfig:
    """Everything that identifies and constrains one Experiment Run."""

    agent: str
    model: str
    provider: str = "unspecified"
    model_version: str | None = None
    transport: str = "unspecified"
    max_steps: int = 40
    replicate_id: int = 0
    schedule_index: int | None = None
    provider_seed: int | None = None
    asset_root: str | Path = "."
    artifacts_root: str | Path | None = None
    experiment_id: str = "manual"
    output_path: str | Path | None = None
    operator_id: str | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.replicate_id < 0:
            raise ValueError("replicate_id must be non-negative")
        if self.schedule_index is not None and self.schedule_index < 0:
            raise ValueError("schedule_index must be non-negative")
        if self.operator_id is not None and not self.operator_id.strip():
            raise ValueError("operator_id must be non-empty when provided")
        object.__setattr__(self, "asset_root", Path(self.asset_root).resolve())
        for field_name in ("artifacts_root", "output_path"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value).resolve())


@dataclass
class RunContext:
    """The small Interface an Agent Adapter uses while acting."""

    initial_payload: dict[str, Any]
    base_url: str
    site_data: dict[str, Any]
    _session: RunSession
    _state: Any
    _partial_trace: Any = field(default_factory=list)

    @property
    def current_step(self) -> int:
        return self._state.current_step

    @property
    def submission_completed(self) -> bool:
        return self._state.submitted

    def advance(self) -> list[dict[str, Any]]:
        """Advance one Agent step and deliver any due protocol event exactly once."""

        self._state.current_step += 1
        events = self._session.advance(self._state.current_step)
        for event in events:
            self._state.actions.append(event)
            packet_id = (
                "V1" if event["type"] == "visual_context_update"
                else event.get("packet")
            )
            if packet_id:
                self._state.reveal(packet_id)
        return events

    def ask_user(self, slot: str, question: str) -> dict[str, Any]:
        """Use the deterministic clarification Interface and record its answer."""

        event = self._session.ask_user(slot, question, self._state.current_step)
        self._state.actions.append(event)
        return event

    def capture_partial_trace(self, trace: Any) -> None:
        """Keep the latest reconstructable trace if the Adapter later fails."""

        self._partial_trace = trace


SyncAdapter = Callable[[RunContext], Any]
AsyncAdapter = Callable[[RunContext], Awaitable[Any]]


def run_experiment(
    task: dict[str, Any], arm: str, config: RunConfig, adapter: SyncAdapter
) -> dict[str, Any]:
    """Run a synchronous Agent Adapter behind the Experiment Run seam."""

    lifecycle = _RunLifecycle(task, arm, config)
    trace: Any = []
    try:
        lifecycle.prepare()
        with lifecycle.runtime.server as server:
            context = lifecycle.context(server.base_url)
            try:
                trace = adapter(context)
            except BaseException:
                trace = context._partial_trace
                raise
    except BaseException as exc:
        lifecycle.fail(trace, exc)
        raise
    return lifecycle.complete(trace)


async def run_experiment_async(
    task: dict[str, Any], arm: str, config: RunConfig, adapter: AsyncAdapter
) -> dict[str, Any]:
    """Run an asynchronous Agent Adapter behind the same Experiment Run seam."""

    lifecycle = _RunLifecycle(task, arm, config)
    trace: Any = []
    try:
        lifecycle.prepare()
        with lifecycle.runtime.server as server:
            context = lifecycle.context(server.base_url)
            try:
                trace = await adapter(context)
            except BaseException:
                trace = context._partial_trace
                raise
    except BaseException as exc:
        lifecycle.fail(trace, exc)
        raise
    return lifecycle.complete(trace)


class _RunLifecycle:
    def __init__(self, task: dict[str, Any], arm: str, config: RunConfig):
        self.task = task
        self.arm = arm
        self.config = config
        self.site_data = load_site_data(config.asset_root, str(task["site"]))
        self.session = RunSession(task, arm)
        self.initial_payload = self.session.initial_payload()
        self.runtime = create_site_runtime(
            task, arm, self.site_data, asset_root=config.asset_root
        )
        self.state = self.runtime.state
        self.state.event_gate = self.session.has_pending_scheduled_event
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
            "max_steps": config.max_steps,
        }
        self.ledger = (
            RunLedger.start(
                config.artifacts_root,
                experiment_id=config.experiment_id,
                task_id=task["id"],
                arm=arm,
                replicate_id=config.replicate_id,
                agent=config.agent,
                compatibility_output=config.output_path,
            )
            if config.artifacts_root is not None else None
        )

    def prepare(self) -> None:
        self.meta = build_run_meta(
            self.task,
            self.site_data,
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
        self.meta["schedule_index"] = self.config.schedule_index
        self.meta["operator_id"] = self.config.operator_id

    def context(self, base_url: str) -> RunContext:
        return RunContext(
            initial_payload=self.initial_payload,
            base_url=base_url,
            site_data=self.site_data,
            _session=self.session,
            _state=self.state,
        )

    def complete(self, trace: Any) -> dict[str, Any]:
        record = self._record(trace, valid=True, reason=None)
        if self.ledger:
            self.ledger.complete(record)
        elif self.config.output_path:
            write_json_atomic(self.config.output_path, record)
        return record

    def fail(self, trace: Any, error: BaseException) -> None:
        record = self._record(trace, valid=False, reason="infrastructure_error")
        if self.ledger:
            self.ledger.fail(record, error)
        elif self.config.output_path:
            write_json_atomic(self.config.output_path, record)

    def _record(
        self, trace: Any, *, valid: bool, reason: str | None
    ) -> dict[str, Any]:
        record = self.state.result_record(valid=valid, reason=reason)
        record["trace"] = trace
        record["meta"] = {
            **self.meta,
            "started_at": self.started_at,
            "duration_seconds": round(time.time() - self.started_at, 1),
            "event_delivery_steps": [
                action["step"] for action in self.state.actions
                if action.get("type") in {
                    "instruction_injection", "visual_context_update"
                }
            ],
            "run_id": self.ledger.run_id if self.ledger else None,
            "artifact_directory": (
                self.ledger.directory.as_posix() if self.ledger else None
            ),
        }
        return record
