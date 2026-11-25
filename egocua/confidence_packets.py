"""Build answer-free materials for Coding-Agent confidence review."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

from .environment import load_site_data, public_site_data, task_public_site_data
from .construction import load_construction_tasks
from .experiment import TASK_REVIEW_PLAN, ExperimentPlan, load_experiment_plan
from .lineage import (
    LineageError, media_segment_covering, resolve_continuous_media_video,
    source_media_segments,
)
from .protocol import RELEASE_DELIVERY_DESCRIPTIONS
from .run_ledger import canonical_sha256, task_sha256
from .run_materials import clarified_instruction_snapshot
from .sources import find_record, screened_records
from .timeline import Timeline
from .workspace import resolve_within, write_json_atomic


PACKET_SCHEMA_VERSION = 1
OUTPUT_DIR = "artifacts/reports/confidence-review-packets"
MATERIALS_POLICY = "blind_release_only_then_expected_state"
MATERIALS_HASH_POLICY = "task_performer_materials_v3"
CONTINUOUS_VIDEO_REQUIREMENT = "continuous_video_interval"
SITE_ACTIONS = {
    "egomart": [
        "search",
        "open_product",
        "add_to_cart",
        "remove_from_cart",
        "submit_cart",
        "wait",
    ],
    "egomedia": [
        "search",
        "open_video",
        "save",
        "deselect_entity",
        "set_fields",
        "submit",
        "wait",
    ],
    "egomaps": [
        "search",
        "open_place",
        "save",
        "deselect_entity",
        "set_fields",
        "submit",
        "wait",
    ],
}


class ConfidencePacketError(ValueError):
    """Raised when blind review materials cannot be built safely."""


def answer_boundary(task: Mapping[str, Any]) -> dict[str, Any]:
    """Project both private answer boundaries without ordering-only differences."""
    def normalized_goal(goal: Mapping[str, Any]) -> dict[str, Any]:
        states = {
            json.dumps(
                {
                    **state,
                    "selected_entity_ids": sorted(state["selected_entity_ids"]),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            for state in goal["acceptable_states"]
        }
        return {
            **goal,
            "acceptable_states": [json.loads(state) for state in sorted(states)],
        }

    initial = None
    if task["track"] == "graft":
        initial = normalized_goal(task["graft"]["initial_goal"])
    elif task["track"] == "refresh":
        initial = normalized_goal({
            "target_granularity": task["goal"]["target_granularity"],
            "acceptable_states": task["truth"]["initial_acceptable_states"],
        })
    return {"initial_goal": initial, "final_goal": normalized_goal(task["goal"])}


def review_materials_sha256(materials: Mapping[str, Any]) -> str:
    """Identify evidence by contents and timing, not its storage location."""

    def contents(value: Any) -> Any:
        if isinstance(value, Mapping):
            has_contents = "sha256" in value or "content_sha256" in value
            return {
                key: contents(child)
                for key, child in value.items()
                if key != "path" or not has_contents
            }
        if isinstance(value, list):
            return [contents(child) for child in value]
        return value

    projection = {
        key: value for key, value in materials.items() if key != "task_sha256"
    }
    return canonical_sha256(contents(projection))


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_object(path: Path, context: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfidencePacketError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfidencePacketError(f"{context}: JSON root must be an object")
    return value


def _source_material(
    task: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = task.get("source")
    if not isinstance(source, Mapping):
        raise ConfidencePacketError(f"{task.get('id')}: source lineage is missing")
    dossier_ref = source.get("dossier")
    if not isinstance(dossier_ref, str):
        raise ConfidencePacketError(f"{task.get('id')}: source dossier is missing")
    dossier_path = resolve_within(root, dossier_ref, context="source dossier")
    dossier = _load_object(dossier_path, str(dossier_path))
    try:
        record = find_record(dossier, str(task["id"]), context=str(dossier_path))
    except (KeyError, ValueError) as exc:
        raise ConfidencePacketError(str(exc)) from exc
    dataset = dossier.get("dataset")
    if not isinstance(dataset, dict):
        raise ConfidencePacketError(f"{task.get('id')}: source dataset is missing")
    return record, dataset


def _continuous_video_ref(
    task: Mapping[str, Any],
    record: Mapping[str, Any],
    dataset: Mapping[str, Any],
    root: Path,
) -> dict[str, Any] | None:
    construct = record.get("construct")
    requirements = (
        construct.get("evidence_requirements", [])
        if isinstance(construct, Mapping)
        else []
    )
    if not isinstance(requirements, list) or any(
        not isinstance(requirement, str) for requirement in requirements
    ):
        raise ConfidencePacketError(
            f"{task.get('id')}: source evidence_requirements must be a list of strings"
        )
    if CONTINUOUS_VIDEO_REQUIREMENT not in requirements:
        return None

    source = record.get("source")
    source_root = dataset.get("source_root")
    if not isinstance(source, Mapping) or not isinstance(source_root, str):
        raise ConfidencePacketError(
            f"{task.get('id')}: continuous-video source metadata is incomplete"
        )

    timestamps = [
        float(timestamp)
        for packet in record.get("packets", [])
        if isinstance(packet, Mapping)
        for timestamp in packet.get("timestamps_seconds", [])
        if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
    ]
    if not timestamps:
        raise ConfidencePacketError(
            f"{task.get('id')}: continuous-video interval has no timestamps"
        )
    source_start = min(timestamps)
    source_end = max(timestamps)
    if "media_segments" not in source:
        media_start = source.get("media_start_seconds", 0)
        if not isinstance(media_start, (int, float)) or isinstance(media_start, bool):
            raise ConfidencePacketError(
                f"{task.get('id')}: media_start_seconds must be numeric"
            )
    try:
        segments = source_media_segments(source, context=str(task.get("id")))
        segment = media_segment_covering(
            segments, (source_start, source_end), context=str(task.get("id"))
        )
        video_path = resolve_continuous_media_video(
            root, source_root, segment.media_id, context=str(task.get("id"))
        )
    except (LineageError, TypeError, ValueError) as exc:
        raise ConfidencePacketError(str(exc)) from exc
    media_start = segment.source_start_seconds
    media_sha256 = segment.media_sha256
    playback_start = source_start - float(media_start)
    playback_end = source_end - float(media_start)
    if playback_start < 0 or playback_end <= playback_start:
        raise ConfidencePacketError(
            f"{task.get('id')}: continuous-video interval is invalid"
        )
    return {
        "path": video_path.relative_to(root).as_posix(),
        "sha256": media_sha256,
        "playback_interval_seconds": [
            round(playback_start, 6),
            round(playback_end, 6),
        ],
        "source_interval_seconds": [round(source_start, 6), round(source_end, 6)],
        "inspection_requirement": "inspect_the_complete_interval_in_order",
    }


def _frame_timestamps(record: Mapping[str, Any]) -> dict[tuple[str, int], float]:
    timestamps: dict[tuple[str, int], float] = {}
    for packet in record.get("packets", []):
        if not isinstance(packet, Mapping):
            continue
        packet_id = str(packet.get("id", ""))
        values = packet.get("timestamps_seconds", [])
        if not isinstance(values, list):
            continue
        for index, value in enumerate(values):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                timestamps[(packet_id, index)] = float(value)
    return timestamps


def _frame_refs(
    observation: Mapping[str, Any],
    timestamps: Mapping[tuple[str, int], float],
    *,
    packet_id: str,
    root: Path,
) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for index, relative in enumerate(observation.get("frame_paths", [])):
        path = resolve_within(root, str(relative), context=f"{packet_id}:{index} frame")
        ref: dict[str, Any] = {
            "ref": f"{packet_id}:{index}",
            "path": str(relative),
            "sha256": file_sha256(path),
        }
        timestamp = timestamps.get((packet_id, index))
        if timestamp is not None:
            ref["timestamp_seconds"] = timestamp
        refs.append(ref)
    return refs


def _observation_refs(
    observations: list[dict[str, Any]],
    task: Mapping[str, Any],
    timestamps: Mapping[tuple[str, int], float],
    root: Path,
) -> list[dict[str, Any]]:
    packet_by_paths = {
        tuple(packet.get("frame_paths", [])): str(packet.get("id"))
        for packet in task.get("visual_packets", [])
        if isinstance(packet, Mapping)
    }
    refs: list[dict[str, Any]] = []
    for observation in observations:
        paths = tuple(observation.get("frame_paths", []))
        packet_id = packet_by_paths.get(paths)
        if packet_id is None and len(paths) == 1:
            packet_id = next(
                (
                    str(packet.get("id"))
                    for packet in task.get("visual_packets", [])
                    if isinstance(packet, Mapping)
                    and packet.get("frame_paths", [])[:1] == list(paths)
                ),
                None,
            )
        if packet_id is None:
            raise ConfidencePacketError(
                f"{task.get('id')}: timeline observation is not a released packet"
            )
        refs.append(
            {
                "packet_id": packet_id,
                "frames": _frame_refs(
                    observation,
                    timestamps,
                    packet_id=packet_id,
                    root=root,
                ),
            }
        )
    return refs


def build_public_snapshots(root: str | Path) -> dict[str, dict[str, Any]]:
    workspace = Path(root).resolve()
    return {
        site: public_site_data(site, load_site_data(workspace, site))
        for site in sorted(SITE_ACTIONS)
    }


def build_review_packet(
    task: dict[str, Any],
    *,
    root: str | Path,
    plan: ExperimentPlan,
    public_site_path: str,
    public_site_sha256: str,
) -> dict[str, Any]:
    """Return the exact blind materials for one reference-condition run."""

    workspace = Path(root).resolve()
    condition = plan.reference_condition_for(str(task["track"]))
    timeline = Timeline.compile(
        task,
        condition,
        default_release_step=plan.default_release_step,
        reference_steps=plan.reference_steps,
    )
    source_record, dataset = _source_material(task, workspace)
    timestamps = _frame_timestamps(source_record)
    continuous_video = _continuous_video_ref(
        task,
        source_record,
        dataset,
        workspace,
    )
    initial = timeline.initial()
    releases: list[dict[str, Any]] = []
    for release in timeline.releases:
        payload: Any = release.payload
        if release.kind == "observation":
            payload = _observation_refs(
                [dict(release.payload)],
                task,
                timestamps,
                workspace,
            )[0]
        releases.append(
            {
                "ref": release.ref,
                "step": release.step,
                "kind": release.kind,
                "payload": payload,
            }
        )

    clarification = None
    if timeline.offer is not None:
        clarification = {
            "available": True,
            "max_questions": timeline.offer.max_questions,
            "delivery": "after_one_valid_disambiguating_question",
            "released_answer": timeline.offer.canonical_answer,
        }

    materials = {
        "task_id": task["id"],
        "task_sha256": task_sha256(task),
        "track": task["track"],
        "site": task["site"],
        "condition": condition.as_dict(),
        "release_trigger": plan.release_trigger,
        "release_delivery": RELEASE_DELIVERY_DESCRIPTIONS[plan.release_trigger],
        "requested_output_shape": {
            "selected_entity_ids": "list[public entity_id]",
            "fields": "object",
        },
        "initial": {
            "instruction": initial["instruction"],
            "observations": _observation_refs(
                list(initial["observations"]),
                task,
                timestamps,
                workspace,
            ),
            "clarification_available": initial["clarification_available"],
        },
        "scheduled_releases": releases,
        "clarification_exchange": clarification,
        "public_site": {
            "path": public_site_path,
            "content_sha256": public_site_sha256,
        },
        "available_actions": [
            *SITE_ACTIONS[str(task["site"])],
            *(["ask_user"] if timeline.offer is not None else []),
        ],
    }
    if continuous_video is not None:
        materials["continuous_source_video"] = continuous_video
    return {
        "schema_version": PACKET_SCHEMA_VERSION,
        "materials_policy": MATERIALS_POLICY,
        "materials_hash_policy": MATERIALS_HASH_POLICY,
        "materials_sha256": review_materials_sha256(materials),
        "materials": materials,
    }


def validate_review_packet(packet: Mapping[str, Any]) -> None:
    """Reject malformed or answer-bearing packet shapes."""

    if packet.get("schema_version") != PACKET_SCHEMA_VERSION:
        raise ConfidencePacketError("unsupported confidence packet schema")
    if packet.get("materials_policy") != MATERIALS_POLICY:
        raise ConfidencePacketError("unsupported confidence packet policy")
    if packet.get("materials_hash_policy") != MATERIALS_HASH_POLICY:
        raise ConfidencePacketError("unsupported confidence packet hash policy")
    materials = packet.get("materials")
    if not isinstance(materials, Mapping):
        raise ConfidencePacketError("confidence packet has no materials")
    if packet.get("materials_sha256") != review_materials_sha256(materials):
        raise ConfidencePacketError("confidence packet materials hash mismatch")
    forbidden = {
        "answer_boundary",
        "initial_goal",
        "final_goal",
        "acceptable_states",
        "initial_acceptable_states",
        "goal",
        "truth",
        "tempting_referent_ids",
        "candidate_ids",
        "valid_slot",
        "oracle",
        "construct",
        "supported_claims",
        "acceptance_status",
        "proposed_entities",
        "roles",
    }

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            leaked = forbidden & set(value)
            if leaked:
                raise ConfidencePacketError(
                    f"confidence packet leaks hidden keys {sorted(leaked)}"
                )
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(materials)


class CurrentReviewMaterials:
    """Read task-local review inputs once per status or report calculation."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._sites: dict[str, dict[str, Any]] = {}
        self._packets: dict[str, dict[str, Any]] = {}

    @cached_property
    def _plan(self) -> ExperimentPlan:
        return load_experiment_plan(self.root / TASK_REVIEW_PLAN)

    @cached_property
    def _index(self) -> dict[str, Any]:
        path = self.root / OUTPUT_DIR / "index.json"
        return _load_object(path, str(path)) if path.is_file() else {}

    @cached_property
    def _development_reviews(self) -> dict[str, Any]:
        path = self.root / "data/reviews/task-qualification.json"
        return _load_object(path, str(path)) if path.is_file() else {}

    def packet(self, task: dict[str, Any]) -> dict[str, Any]:
        task_id = str(task["id"])
        if task_id not in self._packets:
            site = str(task["site"])
            if site not in self._sites:
                self._sites[site] = load_site_data(self.root, site)
            snapshot = task_public_site_data(task, self._sites[site])
            self._packets[task_id] = build_review_packet(
                task,
                root=self.root,
                plan=self._plan,
                public_site_path=f"{OUTPUT_DIR}/public-sites/{task_id}.json",
                public_site_sha256=canonical_sha256(snapshot),
            )
            review = self._development_reviews.get("tasks", {}).get(task_id, {})
            previous = clarified_instruction_snapshot(task, review)
            if previous is not None:
                self._packets[task_id]["retained_instruction_packet"] = build_review_packet(
                    previous,
                    root=self.root,
                    plan=self._plan,
                    public_site_path=f"{OUTPUT_DIR}/public-sites/{task_id}.json",
                    public_site_sha256=canonical_sha256(snapshot),
                )
        return self._packets[task_id]

    def stored_packet_matches(self, task: dict[str, Any]) -> bool:
        row = self._index.get("tasks", {}).get(task["id"])
        if not isinstance(row, Mapping) or (
            row.get("task_sha256") != task_sha256(task)
            or row.get("answer_boundary") != answer_boundary(task)
            or not isinstance(row.get("path"), str)
        ):
            return False
        path = resolve_within(
            self.root, row["path"], context="review packet", must_exist=False
        )
        if not path.is_file():
            return False
        saved = _load_object(path, str(path))
        validate_review_packet(saved)
        current = self.packet(task)
        if (
            row.get("materials_sha256") != current["materials_sha256"]
            or saved["materials_sha256"] != current["materials_sha256"]
        ):
            return False
        public = saved["materials"].get("public_site", {})
        relative = public.get("path")
        if not isinstance(relative, str):
            return False
        public_path = resolve_within(
            self.root, relative, context="review public site", must_exist=False
        )
        return (
            public_path.is_file()
            and canonical_sha256(_load_object(public_path, str(public_path)))
            == current["materials"]["public_site"]["content_sha256"]
        )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_packets(
    root: str | Path = ".",
    *,
    output_dir: str = OUTPUT_DIR,
    plan_path: str = TASK_REVIEW_PLAN,
    task_ids: list[str] | None = None,
) -> dict:
    workspace = Path(root).resolve()
    output = resolve_within(
        workspace,
        output_dir,
        context="confidence review packet directory",
        must_exist=False,
    )
    public_dir = output / "public-sites"
    public_dir.mkdir(parents=True, exist_ok=True)
    plan = load_experiment_plan(
        resolve_within(workspace, plan_path, context="experiment plan")
    )
    index_path = output / "index.json"
    previous = json.loads(index_path.read_text()) if index_path.exists() else {}
    retained_ids = set(previous.get("tasks", {}))
    source_records = {record["task_id"]: record for _, record in screened_records(workspace)}
    candidate_ids = set(source_records)
    tasks = load_construction_tasks(
        workspace, sorted((retained_ids & candidate_ids) | set(task_ids or []))
    )
    current_ids = {task["id"] for task in tasks}
    requested = current_ids if task_ids is None else set(task_ids)
    sites = {str(task["site"]) for task in tasks}
    site_data = {site: load_site_data(workspace, site) for site in sites}
    public_refs: dict[str, dict[str, str]] = {
        task_id: row
        for task_id, row in previous.get("public_sites", {}).items()
        if task_id in current_ids
    }
    rows: dict[str, dict[str, Any]] = {
        task_id: {
            **row,
            "plan_sha256": row.get(
                "plan_sha256", previous.get("plan", {}).get("sha256")
            ),
        }
        for task_id, row in previous.get("tasks", {}).items()
        if task_id in current_ids
    }
    requested |= {task["id"] for task in tasks} - set(rows)
    for task in tasks:
        task_id = str(task["id"])
        if task_id not in requested:
            continue
        snapshot = task_public_site_data(task, site_data[str(task["site"])])
        public_path = public_dir / f"{task_id}.json"
        write_json_atomic(public_path, snapshot)
        public_ref = {
            "path": public_path.relative_to(workspace).as_posix(),
            "sha256": _file_sha256(public_path),
            "content_sha256": canonical_sha256(snapshot),
        }
        public_refs[task_id] = public_ref
        packet = build_review_packet(
            task,
            root=workspace,
            plan=plan,
            public_site_path=public_ref["path"],
            public_site_sha256=public_ref["content_sha256"],
        )
        validate_review_packet(packet)
        path = output / f"{task_id}.json"
        write_json_atomic(path, packet)
        rows[task_id] = {
            "path": path.relative_to(workspace).as_posix(),
            "sha256": _file_sha256(path),
            "task_sha256": packet["materials"]["task_sha256"],
            "materials_sha256": packet["materials_sha256"],
            "answer_boundary": answer_boundary(task),
            "plan_sha256": _file_sha256(workspace / plan_path),
        }
        source_record = source_records.get(task_id)
        if source_record is not None and "media_segments" in source_record["source"]:
            rows[task_id]["source_segments_dossier"] = task["source"]["dossier"]

    index = {
        "schema_version": 1,
        "materials_policy": "blind_release_only_then_expected_state",
        "materials_hash_policy": MATERIALS_HASH_POLICY,
        "scope_policy": "task_local_public_catalog",
        "plan": {
            "path": plan_path,
            "sha256": _file_sha256(workspace / plan_path),
        },
        "public_sites": public_refs,
        "summary": {"tasks": len(rows), "sites": len(sites)},
        "tasks": dict(sorted(rows.items())),
    }
    write_json_atomic(output / "index.json", index)
    return index
