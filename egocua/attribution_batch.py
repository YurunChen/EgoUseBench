"""Incremental, single-writer execution of independent trajectory Judges."""

from __future__ import annotations

import fcntl
import signal
from contextlib import contextmanager
from threading import Event, current_thread, main_thread
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import quote
from uuid import uuid4

from .adapters.openai_requests import OpenAIRequestsTransport
from .adapters.openai_responses import OpenAIResponsesTransport
from .failure_attribution import (
    FAILURE_EVIDENCE_PATH,
    FAILURE_REPORT_PATH,
    historical_review_context,
    qualify_attribution_row,
    save_assessment,
)
from .judge_validation import (
    begin_judge_validation,
    judge_exposed_sources,
    register_judge_validation,
    save_judge_validation_assessment,
    select_judge_validation_sample,
    start_new_judge_validation,
)
from .lineage import task_recording_group
from .run_ledger import load_run_record
from .workspace import read_json_strict, resolve_within, write_json_atomic


@dataclass(frozen=True)
class BatchOptions:
    root: Path
    batch_id: str
    report: Path = FAILURE_REPORT_PATH
    run_ids: frozenset[str] = field(default_factory=frozenset)
    affected_run_ids: frozenset[str] = field(default_factory=frozenset)
    concurrency: int = 4
    resume_failed: bool = False
    validation_phase: str | None = None
    new_validation: bool = False

    def __post_init__(self) -> None:
        if not self.batch_id or Path(self.batch_id).name != self.batch_id:
            raise ValueError("batch_id must be a single directory name")
        if self.batch_id in {".", ".."} or self.concurrency < 1:
            raise ValueError("invalid batch_id or concurrency")
        if self.validation_phase not in {None, "calibration", "validation"}:
            raise ValueError("invalid validation phase")
        if self.validation_phase and self.affected_run_ids:
            raise ValueError("validation does not replace current assessments")
        if self.run_ids and not self.affected_run_ids <= self.run_ids:
            raise ValueError("affected run IDs must belong to the selected scope")

    @property
    def directory(self) -> Path:
        return self.root / "artifacts" / "attribution-judges" / self.batch_id


def _selection(
    options: BatchOptions,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    report = read_json_strict(options.root / options.report)
    rows = report["attributions"]
    known = {row["run_id"] for row in rows}
    if len(known) != len(rows):
        raise ValueError("Judge report contains duplicate run IDs")
    unknown = (options.run_ids | options.affected_run_ids) - known
    if unknown:
        raise ValueError(f"unknown report run IDs: {sorted(unknown)}")
    evidence_path = options.root / FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(evidence_path) if evidence_path.exists() else {}
    scope = options.run_ids
    assessments = evidence.get("assessments", [])
    if options.validation_phase:
        validation = evidence["judge_validation"]
        registered = frozenset(validation[f"{options.validation_phase}_run_ids"])
        if scope and not scope <= registered:
            raise ValueError(
                "selected runs are outside the registered validation phase"
            )
        scope = scope or registered
        assessments = validation["assessments"]
    reviewed = {item["run_id"] for item in assessments}
    selected, skipped = [], []
    for source in rows:
        run_id = source["run_id"]
        if scope and run_id not in scope:
            continue
        row = qualify_attribution_row(
            source, evidence.get("run_qualifications", {}).get(run_id), options.root
        )
        reason = None
        if row["run_verdict"] != "valid_model_failure":
            reason = row["run_verdict"]
        elif run_id not in options.affected_run_ids and (
            run_id in reviewed
            or (
                not options.validation_phase
                and row.get("attribution_status") == "resolved"
            )
        ):
            reason = "existing_assessment"
        if reason:
            skipped.append({"run_id": run_id, "reason": reason})
        else:
            selected.append(row)
    return selected, skipped


def prepare_judge_validation(options: BatchOptions) -> dict[str, Any]:
    """Register calibration and held-out samples from historical recording groups."""
    from .attribution_judge import PROMPT_VERSION

    evidence_path = options.root / FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(evidence_path) if evidence_path.exists() else {}
    report = read_json_strict(options.root / options.report)
    known = {row["run_id"] for row in report["attributions"]}
    if options.run_ids - known:
        raise ValueError("validation scope references unknown report runs")
    if options.new_validation and evidence.get("judge_validation") is None:
        raise ValueError("new validation requires an existing registered experiment")
    if evidence.get("judge_validation") is not None and not options.new_validation:
        registered = evidence["judge_validation"]
        selected = {
            *registered["calibration_run_ids"],
            *registered["validation_run_ids"],
        }
        if options.run_ids and not selected <= options.run_ids:
            raise ValueError(
                "registered validation selection differs from requested scope"
            )
        return registered
    rows, groups = [], {}
    excluded_sources = (
        judge_exposed_sources(evidence) if options.new_validation else set()
    )
    for source in report["attributions"]:
        if options.run_ids and source["run_id"] not in options.run_ids:
            continue
        row = qualify_attribution_row(
            source,
            evidence.get("run_qualifications", {}).get(source["run_id"]),
            options.root,
        )
        if row["run_verdict"] != "valid_model_failure":
            continue
        directory = resolve_within(
            options.root,
            row["run_artifact"],
            context="Judge sample run",
            must_exist=False,
        )
        record = load_run_record(directory)
        task = record.get("task_snapshot")
        if task is None:
            task = read_json_strict(directory / "task.json")
        group = task_recording_group(task, root=options.root)
        if group in excluded_sources:
            continue
        groups[row["run_id"]] = group
        rows.append(row)
    selection = select_judge_validation_sample(rows, groups)
    if options.new_validation:
        registered = start_new_judge_validation(options.root, selection)
        return register_judge_validation(
            options.root, registered, model=None, prompt_version=PROMPT_VERSION
        )
    return register_judge_validation(
        options.root, selection, model=None, prompt_version=PROMPT_VERSION
    )


def prepare_judge_batch(options: BatchOptions) -> dict[str, Any]:
    """Inspect candidate archives without constructing a model or transport."""
    from .attribution_judge import inspect_materials

    rows, skipped = _selection(options)
    candidates = []
    for row in rows:
        context = historical_review_context(row, options.root)
        candidates.append(
            {
                "run_id": row["run_id"],
                "task_id": row["task_id"],
                "run_artifact": row.get("run_artifact"),
                "model": row.get("model"),
                "setting": row.get("setting"),
                **inspect_materials(context, options.root),
                "actions": len(context.get("actions", [])),
                "web_observations": len(context.get("web_observations", [])),
                "visual_packets": len(
                    context.get("task_context", {}).get("visual_packets", [])
                ),
            }
        )
    return {"batch_id": options.batch_id, "candidates": candidates, "skipped": skipped}


def _judge_one(row: dict[str, Any], options: BatchOptions, factories: tuple) -> dict:
    judge_factory, transport_factory = factories
    transport = None
    try:
        transport = transport_factory()
        return {
            "status": "completed",
            "assessment": judge_factory(transport).assess(row, options.root),
        }
    except Exception as error:
        return {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "request_log": getattr(error, "request_log", []),
            "usage": getattr(error, "usage", {}),
        }
    finally:
        if transport is not None:
            transport.session.close()


def _finish_case(row: dict, outcome: dict, options: BatchOptions) -> dict:
    archive = dict(outcome)
    if outcome["status"] == "completed":
        assessment = dict(outcome["assessment"])
        archive["request_log"] = assessment.pop("request_log", archive.get("request_log", []))
        try:
            if options.validation_phase:
                save_judge_validation_assessment(options.root, row, assessment)
            else:
                save_assessment(
                    options.root,
                    row,
                    assessment,
                    affected=row["run_id"] in options.affected_run_ids,
                )
        except Exception as error:
            archive.update(
                status="failed", error_type=type(error).__name__, error=str(error)
            )
        archive["assessment"] = assessment
    archive.update(run_id=row["run_id"], task_id=row["task_id"])
    return archive


def judge_transport(
    config_payload: Mapping[str, Any], env_path: Path = Path(".env")
) -> OpenAIRequestsTransport:
    from .attribution_judge import JudgeConfig

    config = JudgeConfig.from_dict(config_payload)
    transport = (OpenAIResponsesTransport if config.api == "responses"
                 else OpenAIRequestsTransport)
    return transport.from_env(env_path, timeout_seconds=config.request_timeout_seconds)


def run_judge_batch(
    options: BatchOptions,
    config_payload: Mapping[str, Any],
    transport_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    """Run each missing case once; persist results and canonical reviews serially."""
    from .attribution_judge import JudgeConfig, TrajectoryJudge

    config = JudgeConfig.from_dict(dict(config_payload))
    options.directory.mkdir(parents=True, exist_ok=True)
    with (options.directory / ".writer.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "this Judge batch already has an active writer"
            ) from error
        return _run_locked(
            options, dict(config_payload), (config, TrajectoryJudge, transport_factory)
        )


def _recover_received(rows: list[dict], batch: dict, options: BatchOptions) -> None:
    """Persist an already paid, archived response after an interrupted save."""
    known = {path for result in batch["results"].values() for path in result["attempts"]}
    pending = {row["run_id"]: row for row in rows}
    for path in sorted(options.directory.glob("run-*.json"), key=lambda item: item.stat().st_mtime):
        reference = path.relative_to(options.root).as_posix()
        if reference in known:
            continue
        archived = read_json_strict(path)
        run_id = archived["run_id"]
        if run_id not in pending or archived["status"] not in {"response_received", "completed"}:
            continue
        assessment = archived["assessment"]
        if (assessment.get("prompt_version") != batch["prompt_version"]
                or assessment.get("judge_model") != batch["config"]["model"]):
            raise ValueError("archived response has a different Judge model or prompt")
        archive = _finish_case(pending.pop(run_id), {**archived, "status": "completed"}, options)
        write_json_atomic(path, archive)
        batch["results"][run_id] = {
            "status": archive["status"], "archive": reference, "attempts": [reference],
        }
        write_json_atomic(options.directory / "batch.json", batch)


def _run_locked(options: BatchOptions, config_payload: dict, runtime: tuple) -> dict:
    from .attribution_judge import PROMPT_VERSION

    config, judge_class, transport_factory = runtime
    report = resolve_within(
        options.root, options.report, context="Judge report", must_exist=True
    )
    identity = {
        "batch_id": options.batch_id,
        "prompt_version": PROMPT_VERSION,
        "report": str(report.relative_to(options.root)),
        "config": config_payload,
        "run_ids": sorted(options.run_ids),
        "affected_run_ids": sorted(options.affected_run_ids),
        "validation_phase": options.validation_phase,
    }
    batch_path = options.directory / "batch.json"
    batch = (
        read_json_strict(batch_path)
        if batch_path.exists()
        else {**identity, "results": {}}
    )
    if any(batch.get(key) != value for key, value in identity.items()):
        raise ValueError("Judge batch settings changed; use a new batch_id")
    if options.validation_phase:
        evidence = read_json_strict(options.root / FAILURE_EVIDENCE_PATH)
        register_judge_validation(
            options.root, evidence["judge_validation"],
            model=config.model, prompt_version=PROMPT_VERSION,
        )
    rows, skipped = _selection(options)
    if not options.validation_phase:
        reused = _reuse_validation_assessments(rows, options, config)
        batch["reused"] = sorted(set(batch.get("reused", [])) | reused)
        rows = [row for row in rows if row["run_id"] not in reused]
    _recover_received(rows, batch, options)
    previous = batch["results"]
    pending = [
        row
        for row in rows
        if row["run_id"] not in previous
        or (options.resume_failed and previous[row["run_id"]]["status"] == "failed")
    ]
    batch["skipped"] = skipped
    write_json_atomic(batch_path, batch)
    factories = (
        lambda transport: judge_class(config, transport),
        transport_factory or (lambda: judge_transport(config_payload)),
    )
    if pending and options.validation_phase == "validation":
        begin_judge_validation(options.root)
    queue = iter(pending)
    batch["interrupted"] = False
    batch.pop("stop_reason", None)
    with _stop_on_interrupt() as stop, ThreadPoolExecutor(max_workers=options.concurrency) as executor:
        futures = {}
        while True:
            while len(futures) < options.concurrency and not stop.is_set():
                row = next(queue, None)
                if row is None:
                    break
                futures[executor.submit(_judge_one, row, options, factories)] = row
            if not futures:
                break
            completed, _ = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
            for future in completed:
                row = futures.pop(future)
                outcome = future.result()
                run_id = row["run_id"]
                attempts = previous.get(run_id, {}).get("attempts", [])
                path = (
                    options.directory / f"run-{quote(run_id, safe='')}-{uuid4().hex}.json"
                )
                write_json_atomic(
                    path,
                    {
                        **outcome,
                        "run_id": run_id,
                        "task_id": row["task_id"],
                        "status": "response_received"
                        if outcome["status"] == "completed"
                        else "failed",
                    },
                )
                archive = _finish_case(row, outcome, options)
                write_json_atomic(path, archive)
                previous[run_id] = {
                    "status": archive["status"],
                    "archive": str(path.relative_to(options.root)),
                    "attempts": [*attempts, str(path.relative_to(options.root))],
                }
                write_json_atomic(batch_path, batch)
                if archive["status"] == "failed" and "insufficient_balance" in archive.get("error", ""):
                    batch["stop_reason"] = "provider_insufficient_balance"
                    stop.set()
        batch["interrupted"] = stop.is_set()
    write_json_atomic(batch_path, batch)
    return batch


@contextmanager
def _stop_on_interrupt():
    """Stop submitting calls on Ctrl-C; archive in-flight replies before returning."""
    stop = Event()
    old_handler = None
    if current_thread() is main_thread():
        old_handler = signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        yield stop
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGINT, old_handler)


def _reuse_validation_assessments(rows: list[dict], options: BatchOptions, config: Any) -> set[str]:
    """Reuse canonical paid judgments and retain their original prompt provenance."""
    evidence = read_json_strict(options.root / FAILURE_EVIDENCE_PATH)
    validation = evidence.get("judge_validation", {})
    candidates = {row["run_id"]: row for row in rows
                  if row["run_id"] not in options.affected_run_ids}
    reused: set[str] = set()
    for assessment in validation.get("assessments", []):
        run_id = assessment["run_id"]
        if (run_id not in candidates
                or assessment.get("judge_model") != config.model
                or not validation.get("prompt_version")
                or assessment.get("prompt_version") != validation["prompt_version"]):
            continue
        save_assessment(options.root, candidates[run_id], assessment)
        reused.add(run_id)
    return reused
