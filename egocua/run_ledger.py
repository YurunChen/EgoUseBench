"""Auditable, atomic artifacts for one Experiment Run."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .workspace import resolve_within, write_json_atomic, write_text_atomic

RUN_ARTIFACT_SCHEMA_VERSION = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _slug(value: str, label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-.")
    if not normalized or normalized != value:
        raise ValueError(f"{label} must contain only letters, digits, '.', '_' or '-': {value!r}")
    return normalized


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def asset_hashes(
    task: dict[str, Any], asset_root: str | Path
) -> dict[str, str]:
    paths = [
        frame
        for packet in task.get("visual_packets", [])
        for frame in packet.get("frame_paths", [])
    ]
    hashes: dict[str, str] = {}
    for relative_path in paths:
        path = resolve_within(
            asset_root, relative_path, context="run asset", must_exist=False
        )
        if path.is_file():
            hashes[str(relative_path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def runtime_provenance(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    revision = None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode == 0:
        revision = result.stdout.strip() or None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": revision,
    }


def build_run_meta(
    task: dict[str, Any], site_data: dict[str, Any], *, asset_root: str | Path, agent: str, model: str,
    provider: str, model_version: str | None, transport: str,
    replicate_id: int, provider_seed: int | None, max_steps: int,
) -> dict[str, Any]:
    return {
        "agent": agent,
        "model": model,
        "provider": provider,
        "model_version": model_version,
        "transport": transport,
        "replicate_id": replicate_id,
        "provider_seed": provider_seed,
        "randomness_control": "provider_seed" if provider_seed is not None else "uncontrolled",
        "max_steps": max_steps,
        "task_sha256": canonical_sha256(task),
        "site": task["site"],
        "site_data_sha256": canonical_sha256(site_data),
        "asset_sha256": asset_hashes(task, asset_root),
        "runtime": runtime_provenance(asset_root),
    }


@dataclass
class RunLedger:
    directory: Path
    run_id: str
    manifest: dict[str, Any]
    compatibility_output: Path | None = None

    @classmethod
    def start(
        cls, artifacts_root: str | Path, *, experiment_id: str, task_id: str,
        arm: str, replicate_id: int, agent: str,
        compatibility_output: str | Path | None = None,
    ) -> "RunLedger":
        experiment_id = _slug(experiment_id, "experiment_id")
        task_id = _slug(task_id, "task_id")
        arm = _slug(arm, "arm")
        agent = _slug(agent, "agent")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex[:8]
        directory = Path(artifacts_root).resolve() / "runs" / experiment_id / run_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "task_id": task_id,
            "arm": arm,
            "replicate_id": replicate_id,
            "agent": agent,
            "status": "running",
            "started_at_utc": _utc_now(),
            "finished_at_utc": None,
        }
        ledger = cls(
            directory=directory,
            run_id=run_id,
            manifest=manifest,
            compatibility_output=Path(compatibility_output) if compatibility_output else None,
        )
        ledger._write_manifest()
        ledger.log("INFO", "run_started")
        return ledger

    def log(self, level: str, event: str, **fields: Any) -> None:
        entry = {"timestamp_utc": _utc_now(), "level": level, "event": event, **fields}
        with (self.directory / "runtime.jsonl").open("a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def complete(self, record: dict[str, Any]) -> None:
        self._finish(record, "completed")

    def fail(self, record: dict[str, Any], error: BaseException) -> None:
        write_json_atomic(self.directory / "error.json", {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        })
        self.log("ERROR", "run_failed", error_type=type(error).__name__, message=str(error))
        self._finish(record, "failed")

    def _finish(self, record: dict[str, Any], status: str) -> None:
        write_json_atomic(self.directory / "run.json", record)
        write_json_atomic(self.directory / "trace.json", record.get("trace", []))
        events = record.get("actions", [])
        events_path = self.directory / "events.jsonl"
        write_text_atomic(events_path, "".join(
            json.dumps({"event_index": index, **event}, ensure_ascii=False) + "\n"
            for index, event in enumerate(events)
        ))
        if self.compatibility_output:
            write_json_atomic(self.compatibility_output, record)
        self.manifest["status"] = status
        self.manifest["finished_at_utc"] = _utc_now()
        self.manifest["record_sha256"] = canonical_sha256(record)
        self.manifest["provenance"] = record.get("meta", {})
        self._write_manifest()
        if status == "completed":
            self.log("INFO", "run_completed")

    def _write_manifest(self) -> None:
        write_json_atomic(self.directory / "manifest.json", self.manifest)


def inspect_run_artifact(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    if directory.is_file():
        directory = directory.parent
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Run Artifact has no manifest.json: {directory}")
    manifest = json.loads(manifest_path.read_text())
    run = json.loads((directory / "run.json").read_text()) if (directory / "run.json").is_file() else {}
    error = json.loads((directory / "error.json").read_text()) if (directory / "error.json").is_file() else None
    events_path = directory / "events.jsonl"
    event_count = sum(1 for line in events_path.read_text().splitlines() if line.strip()) if events_path.is_file() else 0
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "experiment_id": manifest.get("experiment_id"),
        "task_id": manifest.get("task_id"),
        "arm": manifest.get("arm"),
        "replicate_id": manifest.get("replicate_id"),
        "agent": manifest.get("agent"),
        "model": run.get("meta", {}).get("model"),
        "steps": run.get("steps"),
        "event_count": event_count,
        "run_status": run.get("run_status"),
        "final_state": run.get("final_state"),
        "error": ({"type": error.get("type"), "message": error.get("message")} if error else None),
        "directory": directory.resolve().as_posix(),
    }
