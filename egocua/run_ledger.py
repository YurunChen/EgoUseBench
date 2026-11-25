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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .experiment_output import (
    LOCAL_TIMEZONE,
    create_experiment_directory,
    local_timestamp,
)
from .environment import task_public_site_data
from .schema import referenced_entity_ids
from .timeline import HUMAN_SOLVABILITY, Condition
from .visual_sampling import packet_asset_paths
from .workspace import resolve_within, write_json_atomic, read_json_strict

RUN_ARTIFACT_SCHEMA_VERSION = 4
RUN_RECORD_FIELDS = frozenset(
    {
        "task_id",
        "track",
        "condition",
        "run_status",
        "initial_state",
        "steps",
        "actions",
        "final_state",
        "trace",
        "meta",
    }
)
RUN_RECORD_OPTIONAL_FIELDS = frozenset(
    {"live_state", "task_snapshot", "initial_context", "archive"}
)
RUN_META_FIELDS = frozenset(
    {
        "agent",
        "model",
        "provider",
        "model_version",
        "transport",
        "replicate_id",
        "schedule_index",
        "provider_seed",
        "randomness_control",
        "max_steps",
        "site",
        "runtime",
        "operator_id",
        "default_release_step",
        "reference_steps",
        "started_at",
        "duration_seconds",
        "event_delivery_steps",
        "run_id",
        "artifact_directory",
    }
)
RUN_META_OPTIONAL_FIELDS = frozenset(
    {
        "diagnostic_intervention",
        "termination_reason",
        "stage_release",
        "released_visual_inputs",
        "run_purpose",
        "used_for_task_repair",
        "valid_as_held_out_difficulty_evidence",
        "task_sha256",
        "site_data_sha256",
        "asset_sha256",
    }
)
MODEL_RUN_TRANSPORTS = {
    "browser-use": "browser-use-chat-openai-vision-v1",
    "browsergym": "browsergym-openai-vision-v1",
}
SITE_RUNTIME_CONTRACTS = {
    "egomart": "task-neutral-home-bm25-full-page-search-v3",
    "egomedia": "task-neutral-home-task-scoped-public-fields-bm25-idempotent-save-v7",
    "egomaps": "task-neutral-home-task-scoped-public-fields-bm25-idempotent-save-detail-photo-v5",
}
TASK_SITE_FINGERPRINT_PREFIX = "task-site-v1:"
_RUNNING_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "experiment_id",
    "task_id",
    "condition",
    "replicate_id",
    "agent",
    "status",
    "started_at_utc",
    "finished_at_utc",
}
_MANIFEST_OPTIONAL_FIELDS = {"record_sha256", "provenance"}


def _slug(value: str, label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip()).strip("-.")
    if not normalized or normalized != value:
        raise ValueError(
            f"{label} must contain only letters, digits, '.', '_' or '-': {value!r}"
        )
    return normalized


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def task_local_content(task: Mapping[str, Any]) -> dict[str, Any]:
    """Project task content without source-review or shared-file digests."""
    projection = dict(task)
    source = projection.get("source")
    if isinstance(source, Mapping):
        task_source = dict(source)
        task_source.pop("dossier_sha256", None)
        task_source.pop("frame_provenance_sha256", None)
        task_source.pop("record_sha256", None)
        projection["source"] = task_source
    return projection


def task_sha256(task: Mapping[str, Any]) -> str:
    return canonical_sha256(task_local_content(task))


def site_execution_sha256(site: str, site_data: Mapping[str, Any]) -> str:
    """Hash frozen data plus behavior that can change what an Agent can do."""

    contract = SITE_RUNTIME_CONTRACTS.get(site)
    if contract is None:
        return canonical_sha256(site_data)
    return canonical_sha256(
        {
            "site_data": site_data,
            "runtime_contract": contract,
        }
    )


def task_site_execution_sha256(
    task: Mapping[str, Any],
    site_data: Mapping[str, Any],
) -> str:
    """Fingerprint only website inputs that belong to one task.

    A catalog is shared storage, not one indivisible benchmark input. Adding an
    unrelated listing must not invalidate every completed task audit. Referenced
    targets, candidates and declared distractors remain bound, as does the shared
    runtime behavior that can change search or submission semantics.
    """

    task_dict = dict(task)
    site = str(task_dict["site"])
    public_data = task_public_site_data(task_dict, site_data)
    entities = {str(entity["entity_id"]): entity for entity in public_data["entities"]}
    entity_ids = referenced_entity_ids(task_dict)
    missing = sorted(set(entity_ids) - set(entities))
    if missing:
        raise ValueError(
            f"{task_dict.get('id', '<task>')}: site data is missing {missing}"
        )
    projection = {key: value for key, value in public_data.items() if key != "entities"}
    projection["entities"] = [entities[entity_id] for entity_id in entity_ids]
    digest = canonical_sha256(
        {
            "task_site_data": projection,
            "runtime_contract": SITE_RUNTIME_CONTRACTS.get(site),
        }
    )
    return TASK_SITE_FINGERPRINT_PREFIX + digest


def asset_hashes(task: dict[str, Any], asset_root: str | Path) -> dict[str, str]:
    paths = [
        frame
        for packet in task.get("visual_packets", [])
        for frame in packet_asset_paths(packet)
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
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        revision = result.stdout.strip() or None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_revision": revision,
    }


def build_run_meta(
    task: dict[str, Any],
    *,
    asset_root: str | Path,
    agent: str,
    model: str,
    provider: str,
    model_version: str | None,
    transport: str,
    replicate_id: int,
    provider_seed: int | None,
    max_steps: int | None,
) -> dict[str, Any]:
    return {
        "agent": agent,
        "model": model,
        "provider": provider,
        "model_version": model_version,
        "transport": transport,
        "replicate_id": replicate_id,
        "provider_seed": provider_seed,
        "randomness_control": "provider_seed"
        if provider_seed is not None
        else "uncontrolled",
        "max_steps": max_steps,
        "site": task["site"],
        "runtime": runtime_provenance(asset_root),
    }


@dataclass
class RunLedger:
    directory: Path
    run_id: str
    identity: dict[str, Any]

    @property
    def experiment(self) -> Path:
        """The experiment directory that owns this run's ``runs/<id>`` folder."""
        return self.directory.parent.parent

    @classmethod
    def start(
        cls,
        output_root: str | Path,
        *,
        experiment_id: str,
        task_id: str,
        condition: Condition,
        replicate_id: int,
        agent: str,
        experiment_directory: str | Path | None = None,
        model: str = "unspecified",
        configuration: dict[str, Any] | None = None,
    ) -> "RunLedger":
        experiment_id = _slug(experiment_id, "experiment_id")
        task_id = _slug(task_id, "task_id")
        _slug(condition.name, "condition name")
        agent = _slug(agent, "agent")
        run_id = (
            datetime.now(LOCAL_TIMEZONE).strftime("%Y%m%d-%H%M%S-%f")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        experiment = (
            Path(experiment_directory).resolve()
            if experiment_directory is not None
            else create_experiment_directory(
                output_root,
                {
                    "experiment_id": experiment_id,
                    "category": "单次执行",
                    "variable": experiment_id,
                    "model": model,
                    "task_ids": [task_id],
                    "conditions": [condition.as_dict()],
                    "configuration": configuration,
                },
            )
        )
        if not (experiment / "plan.json").is_file():
            raise FileNotFoundError(f"experiment has no plan.json: {experiment}")
        directory = experiment / "runs" / run_id
        directory.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "experiment_id": experiment_id,
            "task_id": task_id,
            "condition": condition.as_dict(),
            "replicate_id": replicate_id,
            "agent": agent,
            "status": "running",
            "started_at": local_timestamp(),
            "finished_at": None,
        }
        ledger = cls(directory=directory, run_id=run_id, identity=manifest)
        ledger._write_running_record()
        ledger.log("INFO", "run_started")
        return ledger

    def log(self, level: str, event: str, **fields: Any) -> None:
        entry = {
            "timestamp": local_timestamp(),
            "level": level,
            "event": event,
            **fields,
        }
        with (self.directory / "runtime.log").open("a") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def complete(self, record: dict[str, Any]) -> None:
        self._finish(record, "completed")

    def fail(self, record: dict[str, Any], error: BaseException) -> None:
        self.identity["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        }
        self.log(
            "ERROR", "run_failed", error_type=type(error).__name__, message=str(error)
        )
        self._finish(record, "failed")

    def _finish(self, record: dict[str, Any], status: str) -> None:
        _validate_record_identity(self.identity, record)
        self.identity["status"] = status
        self.identity["finished_at"] = local_timestamp()
        record["archive"] = self._archive_metadata()
        stored = dict(record)
        if "task_snapshot" in record:
            experiment = self.experiment
            plan = read_json_strict(experiment / "plan.json")
            scope = experiment / "tasks"
            if "arms" in plan:
                arm = next(
                    row
                    for row in plan["arms"]
                    if row["plan"]["experiment_id"] == self.identity["experiment_id"]
                )
                scope = scope / arm["name"]
            task_path = scope / record["track"] / f"{record['task_id']}.json"
            if task_path.is_file():
                if read_json_strict(task_path) != record["task_snapshot"]:
                    raise ValueError(
                        "actual task differs from this experiment's frozen task"
                    )
            else:
                write_json_atomic(task_path, record["task_snapshot"])
            record["archive"]["task_snapshot"] = task_path.relative_to(
                experiment
            ).as_posix()
            stored.pop("task_snapshot")
        write_json_atomic(self.directory / "run.json", stored)
        if status == "completed":
            self.log("INFO", "run_completed")

    def _archive_metadata(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in self.identity.items()
            if key not in {"run_id", "task_id", "condition", "replicate_id", "agent"}
        }

    def _write_running_record(self) -> None:
        write_json_atomic(
            self.directory / "run.json",
            {
                "task_id": self.identity["task_id"],
                "condition": self.identity["condition"],
                "meta": {
                    key: self.identity[key]
                    for key in ("run_id", "replicate_id", "agent")
                },
                "archive": self._archive_metadata(),
            },
        )


def inspect_run_artifact(path: str | Path) -> dict[str, Any]:
    directory = Path(path)
    directory, manifest, condition, run = _load_artifact(
        directory, require_record=False
    )
    error = manifest.get("error") or (
        json.loads((directory / "error.json").read_text())
        if (directory / "error.json").is_file()
        else None
    )
    return {
        "run_id": manifest.get("run_id"),
        "status": manifest.get("status"),
        "experiment_id": manifest.get("experiment_id"),
        "task_id": manifest.get("task_id"),
        "condition": condition.as_dict(),
        "replicate_id": manifest.get("replicate_id"),
        "agent": manifest.get("agent"),
        "model": run.get("meta", {}).get("model"),
        "steps": run.get("steps"),
        "event_count": len(run.get("actions", [])),
        "run_status": run.get("run_status"),
        "live_state": run.get("live_state"),
        "final_state": run.get("final_state"),
        "error": (
            {"type": error.get("type"), "message": error.get("message")}
            if error
            else None
        ),
        "directory": directory.resolve().as_posix(),
    }


def read_run_identity(path: str | Path) -> dict[str, Any]:
    """Read current or preserved historical identity through the same reader."""
    return _load_artifact(Path(path), require_record=False)[1]


def find_run_directories(root: Path) -> set[Path]:
    """Discover current experiments and immutable historical run archives."""
    return {
        path.parent
        for pattern in (
            "experiments/*/runs/*/run.json",
            "artifacts/runs/*/*/manifest.json",
        )
        for path in root.glob(pattern)
    }


def load_run_record(path: str | Path) -> dict[str, Any]:
    """Load one canonical Run Artifact directory after validating its manifest."""

    _, _, _, record = _load_artifact(Path(path), require_record=True)
    return record


def _load_artifact(
    directory: Path,
    *,
    require_record: bool,
) -> tuple[Path, dict[str, Any], Condition, dict[str, Any]]:
    if not directory.is_dir():
        raise ValueError(f"Run Artifact path must be a directory: {directory}")
    record_path = directory / "run.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text())
        if isinstance(record, dict) and "archive" in record:
            required = {
                "schema_version",
                "experiment_id",
                "status",
                "started_at",
                "finished_at",
            }
            fields = set(record["archive"])
            if not required <= fields or fields - required - {"error", "task_snapshot"}:
                raise ValueError("run archive has missing or unsupported fields")
            task_reference = record["archive"].get("task_snapshot")
            if task_reference is not None:
                record["task_snapshot"] = read_json_strict(
                    resolve_within(
                        directory.parent.parent,
                        task_reference,
                        context="frozen run task",
                    )
                )
            archive = {
                **record["archive"],
                "task_id": record["task_id"],
                "condition": record["condition"],
                **{
                    key: record["meta"][key]
                    for key in ("run_id", "replicate_id", "agent")
                },
            }
            condition = Condition.from_dict(record["condition"])
            if archive["status"] == "running":
                if require_record:
                    raise ValueError("run is still running")
                return directory, archive, condition, {}
            if archive["status"] not in {"completed", "failed"}:
                raise ValueError("unknown run archive status")
            _validate_record_identity(archive, record)
            return directory, archive, condition, record
    # Historical archives are evidence; only the current format is written.
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"run has no archive metadata: {directory}")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("Run Artifact manifest must be an object")
    if manifest.get("schema_version") != RUN_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Run Artifact schema_version must be {RUN_ARTIFACT_SCHEMA_VERSION}"
        )
    condition = Condition.from_dict(manifest.get("condition"))
    record_path = directory / "run.json"
    has_record = record_path.is_file()
    if require_record and not has_record:
        raise FileNotFoundError(f"Run Artifact has no run.json: {directory}")
    run = json.loads(record_path.read_text()) if has_record else {}
    if has_record and not isinstance(run, dict):
        raise ValueError("Run Artifact record must be an object")
    if not _RUNNING_MANIFEST_FIELDS <= set(manifest) or set(manifest) - (
        _RUNNING_MANIFEST_FIELDS | _MANIFEST_OPTIONAL_FIELDS
    ):
        raise ValueError("Run Artifact manifest has missing or unsupported fields")
    if has_record:
        if manifest.get("status") not in {"completed", "failed"}:
            raise ValueError(
                "finalized Run Artifact status must be completed or failed"
            )
        _validate_record_identity(manifest, run)
    elif manifest.get("status") != "running":
        raise ValueError("Run Artifact without run.json must have running status")
    return directory, manifest, condition, run


def _validate_record_identity(
    manifest: dict[str, Any],
    record: dict[str, Any],
) -> None:
    condition = Condition.from_dict(record.get("condition"))
    if condition.as_dict() != manifest["condition"]:
        raise ValueError("record condition does not match Run Artifact manifest")
    validate_run_record_shape(record, require_complete=True)
    meta = record["meta"]
    if "task_snapshot" in record:
        snapshot = record["task_snapshot"]
        if not isinstance(snapshot, dict) or snapshot.get("id") != record.get(
            "task_id"
        ):
            raise ValueError("task snapshot does not belong to run")
    identity = {
        "task_id": record["task_id"],
        "replicate_id": meta["replicate_id"],
        "agent": meta["agent"],
        "run_id": meta["run_id"],
    }
    expected = {key: manifest[key] for key in identity}
    if identity != expected:
        raise ValueError("record identity does not match Run Artifact manifest")


def validate_run_record_shape(
    record: dict[str, Any],
    *,
    require_complete: bool,
) -> None:
    fields = set(record)
    unknown = fields - RUN_RECORD_FIELDS - RUN_RECORD_OPTIONAL_FIELDS
    missing = RUN_RECORD_FIELDS - fields if require_complete else set()
    if unknown or missing:
        raise ValueError(
            "run record fields disagree with schema v4; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    if "live_state" in record:
        state = record["live_state"]
        if (
            not isinstance(state, dict)
            or set(state) != {"selected_entity_ids", "fields"}
            or not isinstance(state["selected_entity_ids"], list)
            or any(
                not isinstance(value, str) or not value
                for value in state["selected_entity_ids"]
            )
            or not isinstance(state["fields"], dict)
            or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in state["fields"].items()
            )
        ):
            raise ValueError(
                "run record live_state must contain entity IDs and string fields"
            )
    meta = record.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("Run Artifact record meta must be an object")
    missing_meta = RUN_META_FIELDS - set(meta) if require_complete else set()
    unknown_meta = set(meta) - RUN_META_FIELDS - RUN_META_OPTIONAL_FIELDS
    if unknown_meta or missing_meta:
        raise ValueError(
            "run record meta fields disagree with schema v4; "
            f"missing={sorted(missing_meta)}, unknown={sorted(unknown_meta)}"
        )
    steps = record.get("steps")
    max_steps = meta.get("max_steps")
    unlimited_human = (
        max_steps is None
        and meta.get("agent") == "human"
        and record.get("condition", {}).get("schedule") == HUMAN_SOLVABILITY
    )
    if (
        not isinstance(steps, int)
        or isinstance(steps, bool)
        or steps < 0
        or (
            not unlimited_human
            and (
                not isinstance(max_steps, int)
                or isinstance(max_steps, bool)
                or steps > max_steps
            )
        )
    ):
        raise ValueError("run record steps exceed the declared max_steps")
