from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

from egocua.checker import check_run
from egocua.construction import compile_task_from_site, opaque_item_id
from egocua.environment import entity_visual_path, load_site_data
from egocua.experiment import SUPPORTED_ARMS
from egocua.lineage import validate_source_lineage
from egocua.protocol import DYNAMIC_EVENT_STEP
from egocua.schema import goal_for_arm, load_tasks, task_subtype, validate_task
from egocua.workspace import resolve_within, write_json_atomic


TASKS = "data/tasks"
REVIEW_SPEC = "data/reviews/construction.json"
REVIEW_SUBMISSIONS = "data/reviews/submissions"
SELECTION = "data/selection.json"
SITE_DATA_FILES = {
    "egomart": "sites/egomart/products.json",
    "egodocs": "sites/egodocs/documents.json",
    "egomedia": "sites/egomedia/videos.json",
    "egomaps": "sites/egomaps/places.json",
}
MEDIA_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi"}
STATUS_VALUES = {
    "task_verified",
    "needs_source_review",
    "needs_human_review",
    "fail",
}
LEAKAGE = re.compile(
    r"\b(?:V0|V1|visual[_ -]?packet|keyframe|annotation|ground truth|"
    r"selected_entity_ids|submission_completed|correct answer|wrong answer|"
    r"a_only|oracle_latest|forced_execute|instruction_injection|"
    r"visual_context_update)\b|\b[pml]_[0-9a-f]{12}\b",
    re.IGNORECASE,
)
PRERECORDED_INPUT_LANGUAGE = re.compile(
    r"\b(?:this clip|my video|in this sequence|later in the sequence|"
    r"visible in the video)\b",
    re.IGNORECASE,
)
REVIEW_WORDING_FLAGS = re.compile(
    r"\b(?:I am showing|shown in my video|this clip|in this sequence)\b",
    re.IGNORECASE,
)
SITUATED_ACTIVITY = re.compile(
    r"\b(?:hold|holding|handle|handling|use|using|work|working|install|repair|"
    r"sew|cook|walk|move|pick(?:ing)? up|put(?:ting)? down|drop|wipe|wear)\w*\b",
    re.IGNORECASE,
)
ORDINARY_WEB_FILTER = re.compile(
    r"\b(?:rating|stars?|price|cheaper|lower-priced|budget|reviews?|views?|"
    r"popular|shipping|delivery|discount)\b",
    re.IGNORECASE,
)
PHYSICAL_UPDATE = re.compile(
    r"\b(?:while I (?:keep|am still)|I(?:'m| am) still|holding|wearing|"
    r"installing|repairing|sewing|cooking|walking|moving|putting|picking|"
    r"working on|using (?:this|that|the))\b",
    re.IGNORECASE,
)
REQUIRED_CONSTRUCT_FIELDS = {
    "research_question",
    "divergence_source",
    "event_origin",
    "belief_before",
    "belief_after",
    "required_behavior_change",
    "no_update_failure",
    "overupdate_failure",
    "visual_dependency_claim",
    "task_blueprint",
}
CLARIFICATION_QUESTIONS = {
    "position": "Which side or position do you mean?",
    "appearance": "Which appearance do you mean?",
    "brand": "Which brand do you mean?",
    "variant": "Which variant do you mean?",
    "sequence": "Which one in the sequence do you mean?",
}


def _read_object(path: Path, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context}: cannot read JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context}: JSON root must be an object")
    return payload


def _selected_task_ids(selection: Mapping[str, Any]) -> set[str]:
    task_ids: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str):
            task_ids.add(value)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    visit(selection.get("primary"))
    return task_ids


def _check(ok: bool, *, details: Mapping[str, Any] | None = None,
           errors: Iterable[str] = ()) -> dict[str, Any]:
    messages = [str(error) for error in errors if str(error)]
    return {
        "status": "pass" if ok and not messages else "fail",
        **({"details": dict(details)} if details is not None else {}),
        **({"errors": messages} if messages else {}),
    }


def _source_material(
    task: dict[str, Any], root: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source = task["source"]
    dossier_path = resolve_within(
        root, source["dossier"], context=f"{task['id']} source dossier"
    )
    dossier = _read_object(dossier_path, f"{task['id']} source dossier")
    records = [
        record for record in dossier.get("tasks", [])
        if isinstance(record, dict) and record.get("task_id") == task["id"]
    ]
    if len(records) != 1:
        raise ValueError(
            f"{task['id']}: expected one source record, found {len(records)}"
        )
    provenance_path = resolve_within(
        root,
        source["frame_provenance"],
        context=f"{task['id']} frame provenance",
    )
    provenance = _read_object(provenance_path, f"{task['id']} frame provenance")
    return dossier, records[0], dossier_path, provenance_path


def _claim_errors(record: Mapping[str, Any]) -> list[str]:
    packets = record.get("packets", [])
    available = {
        f"{packet.get('id')}:{index}"
        for packet in packets if isinstance(packet, dict)
        for index, _ in enumerate(packet.get("frame_paths", []))
    }
    claims = record.get("supported_claims")
    if not isinstance(claims, list) or not claims:
        return ["source record has no supported_claims"]
    errors: list[str] = []
    for index, claim in enumerate(claims):
        references = claim.get("supported_by") if isinstance(claim, dict) else None
        if (
            not isinstance(claim, dict)
            or not str(claim.get("text", "")).strip()
            or not isinstance(references, list)
            or not references
        ):
            errors.append(f"supported_claims/{index} is incomplete")
            continue
        unknown = sorted(set(references) - available)
        if unknown:
            errors.append(
                f"supported_claims/{index} references unknown frames {unknown}"
            )
    return errors


def _construct_check(
    task: Mapping[str, Any], source_record: Mapping[str, Any], *, is_primary: bool,
) -> dict[str, Any]:
    """Check construct prerequisites without pretending to replace human judgment."""

    errors: list[str] = []
    construct = source_record.get("construct")
    if not isinstance(construct, dict):
        return _check(False, errors=["source record has no construct contract"])
    missing = sorted(
        field for field in REQUIRED_CONSTRUCT_FIELDS
        if field not in construct or construct[field] in (None, "", [], {})
    )
    if missing:
        errors.append(f"construct contract is missing {missing}")

    raw_source = source_record.get("source", {})
    source = raw_source if isinstance(raw_source, dict) else {}
    evidence_text = " ".join([
        *(
            str(value) for value in source.get("original_annotations", [])
        ),
        *(
            str(claim.get("text", ""))
            for claim in source_record.get("supported_claims", [])
            if isinstance(claim, dict)
        ),
        *(
            str(value) for value in task.get("instructions", {}).values()
        ),
    ])
    risk_reasons: list[str] = []
    ongoing_activity_candidate = bool(SITUATED_ACTIVITY.search(evidence_text))
    if not ongoing_activity_candidate:
        risk_reasons.append(
            "no automatic evidence that the wearer is still acting in the physical scene"
        )

    if task["track"] == "graft":
        update = str(task.get("instructions", {}).get("b", ""))
        if ORDINARY_WEB_FILTER.search(update) and not PHYSICAL_UPDATE.search(update):
            risk_reasons.append(
                "the added instruction appears to be only an ordinary website filter"
            )
        if len(task.get("visual_packets", [])) == 1:
            risk_reasons.append(
                "a single initial visual packet may make this equivalent to static photo grounding"
            )
    elif task["track"] == "refresh":
        packet_ids = [packet.get("id") for packet in task.get("visual_packets", [])]
        if packet_ids != ["V0", "V1"]:
            errors.append("Refresh must release one ordered V0/V1 pair")
        if construct.get("event_origin") != "observed":
            errors.append("Refresh visual change is not registered as observed")
    else:
        if len(task.get("clarification", {}).get("candidate_ids", [])) != 2:
            errors.append("Clarify must contain exactly two physical candidates")
        if (
            len(task.get("visual_packets", [])) == 1
            and task_subtype(dict(task)) != "sequence"
        ):
            risk_reasons.append(
                "the ambiguity may be reproducible with one uploaded static photograph"
            )

    return _check(
        not errors,
        details={
            "pilot_scope": "primary_seed" if is_primary else "rescreening_candidate",
            "research_question": construct.get("research_question"),
            "visual_dependency_claim": construct.get("visual_dependency_claim"),
            "required_behavior_change": construct.get("required_behavior_change"),
            "ongoing_activity_candidate": ongoing_activity_candidate,
            "construct_risk_reasons": risk_reasons,
            "human_decision_required": True,
            "admission_test": (
                "Reject from the core benchmark if one task-start photo and one complete "
                "instruction preserve the same decision, timing, and web commitment."
            ),
        },
        errors=errors,
    )


def _video_check(
    task: dict[str, Any], dossier: Mapping[str, Any], record: Mapping[str, Any],
    provenance_path: Path, root: Path, *, verify_source_frames: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    dataset = dossier.get("dataset")
    source_root = dataset.get("source_root") if isinstance(dataset, dict) else None
    if not isinstance(source_root, str) or not source_root:
        return _check(False, errors=["source dossier has no dataset.source_root"])
    video_root = resolve_within(
        root, source_root, context=f"{task['id']} source video root",
        must_exist=False,
    )
    provenance = _read_object(provenance_path, f"{task['id']} frame provenance")
    task_frames = [
        frame for frame in provenance.get("frames", [])
        if isinstance(frame, dict) and frame.get("task_id") == task["id"]
    ]
    media_ids = sorted({
        str(frame.get("media_id")) for frame in task_frames
        if str(frame.get("media_id", "")).strip()
    })
    declared_source = record.get("source", {})
    declared_media_ids = {
        str(value) for value in (
            declared_source.get("media_id"),
            declared_source.get("original_video_id"),
        )
        if isinstance(declared_source, dict) and str(value or "").strip()
    }
    if not media_ids:
        errors.append("frame provenance has no media_id for this task")
    if not set(media_ids) <= declared_media_ids:
        errors.append("frame provenance media_id is not declared by the source record")

    resolved_videos: dict[str, str] = {}
    video_digests: dict[str, str] = {}
    video_metadata = provenance.get("videos", {})
    for media_id in media_ids:
        candidates = sorted({
            path for path in video_root.glob(f"{media_id}*")
            if path.is_file() and path.suffix.casefold() in MEDIA_SUFFIXES
        }) if video_root.is_dir() else []
        if len(candidates) != 1:
            errors.append(
                f"media_id {media_id!r} resolves to {len(candidates)} continuous videos"
            )
        else:
            video = candidates[0]
            resolved_videos[media_id] = video.relative_to(root).as_posix()
            try:
                probe = subprocess.run(
                    [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "format=duration:stream=width,height",
                        "-of", "json", str(video),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                probed = json.loads(probe.stdout)
                actual_duration = float(probed["format"]["duration"])
                stream = probed["streams"][0]
                actual_width = int(stream["width"])
                actual_height = int(stream["height"])
            except (subprocess.CalledProcessError, KeyError, IndexError, ValueError) as exc:
                errors.append(f"media_id {media_id!r} cannot be decoded by ffprobe: {exc}")
                actual_duration = 0.0
                actual_width = actual_height = 0
            digest = hashlib.sha256(video.read_bytes()).hexdigest()
            video_digests[media_id] = digest
            declared_digest = (
                declared_source.get("media_sha256")
                if isinstance(declared_source, dict) else None
            )
            if declared_digest is not None and declared_digest != digest:
                errors.append(f"media_id {media_id!r} source video SHA-256 mismatch")

        metadata = video_metadata.get(media_id) if isinstance(video_metadata, dict) else None
        duration = metadata.get("duration_seconds") if isinstance(metadata, dict) else None
        source_start = metadata.get("source_start_seconds") if isinstance(metadata, dict) else None
        if not isinstance(duration, (int, float)) or duration <= 0:
            errors.append(f"media_id {media_id!r} has no valid provenance duration")
            continue
        if not isinstance(source_start, (int, float)) or source_start < 0:
            errors.append(f"media_id {media_id!r} has no valid source_start_seconds")
        if media_id in resolved_videos and actual_duration > 0:
            if abs(actual_duration - float(duration)) > 0.05:
                errors.append(f"media_id {media_id!r} real duration disagrees with provenance")
            declared_width = metadata.get("width") if isinstance(metadata, dict) else None
            declared_height = metadata.get("height") if isinstance(metadata, dict) else None
            if actual_width != declared_width or actual_height != declared_height:
                errors.append(f"media_id {media_id!r} real dimensions disagree with provenance")
        for frame in task_frames:
            if frame.get("media_id") != media_id:
                continue
            timestamp = frame.get("media_timestamp_seconds")
            if not isinstance(timestamp, (int, float)) or not 0 <= timestamp < duration:
                errors.append(
                    f"frame {frame.get('packet_id')}:{frame.get('frame_index')} "
                    f"falls outside media_id {media_id!r}"
                )

    reextract_mismatches: list[str] = []
    if verify_source_frames and not errors:
        jobs = [
            (frame, Path(resolved_videos[str(frame["media_id"])]))
            for frame in task_frames
            if str(frame.get("media_id")) in resolved_videos
        ]

        def reextract(job: tuple[dict[str, Any], Path]) -> str | None:
            frame, relative_video = job
            video = root / relative_video
            key = f"{frame.get('packet_id')}:{frame.get('frame_index')}"
            with tempfile.TemporaryDirectory(prefix="egocua-frame-") as directory:
                destination = Path(directory) / "frame.jpg"
                try:
                    subprocess.run(
                        [
                            "ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
                            "-ss", f"{float(frame['media_timestamp_seconds']):.3f}",
                            "-frames:v", "1", "-q:v", "2", str(destination),
                        ],
                        check=True,
                        capture_output=True,
                    )
                except (subprocess.CalledProcessError, KeyError, TypeError, ValueError):
                    return f"{key} could not be re-extracted"
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                if digest != frame.get("sha256"):
                    return f"{key} re-extracted bytes disagree with provenance"
            return None

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(jobs)))) as executor:
            reextract_mismatches = sorted(
                result for result in executor.map(reextract, jobs) if result is not None
            )
        errors.extend(reextract_mismatches)
    return _check(
        not errors,
        details={
            "media_ids": media_ids,
            "video_paths": resolved_videos,
            "video_sha256": video_digests,
            "source_frame_reextraction": (
                "verified" if verify_source_frames else "skipped"
            ),
            "reextract_mismatches": reextract_mismatches,
        },
        errors=errors,
    )


def _referenced_entity_ids(
    task: Mapping[str, Any], source_record: Mapping[str, Any] | None = None,
) -> set[str]:
    entity_ids = {
        entity_id
        for state in task["goal"]["acceptable_states"]
        for entity_id in state["selected_entity_ids"]
    }
    if task["track"] == "graft":
        entity_ids.update(
            entity_id
            for state in task["graft"]["initial_goal"]["acceptable_states"]
            for entity_id in state["selected_entity_ids"]
        )
    elif task["track"] == "refresh":
        entity_ids.add(task["refresh"]["initial_entity_id"])
        blueprint = (source_record or {}).get("construct", {}).get("task_blueprint", {})
        later_role = blueprint.get("later_distractor_role")
        if isinstance(later_role, str) and later_role:
            entity_ids.add(opaque_item_id(str(task["id"]), later_role))
    else:
        entity_ids.update(task["clarification"]["candidate_ids"])
    return entity_ids


def _goals(task: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        task["goal"],
        *([task["graft"]["initial_goal"]] if task["track"] == "graft" else []),
    ]


def _site_check(
    task: dict[str, Any], site_data: dict[str, Any], root: Path,
    source_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    errors: list[str] = []
    entities = {
        str(entity["entity_id"]): entity for entity in site_data["entities"]
    }
    required_ids = _referenced_entity_ids(task, source_record)
    missing = sorted(required_ids - set(entities))
    if missing:
        errors.append(f"site is missing referenced entities {missing}")

    frame_hashes = {
        hashlib.sha256(
            resolve_within(root, frame, context=f"{task['id']} released frame").read_bytes()
        ).hexdigest()
        for packet in task["visual_packets"]
        for frame in packet["frame_paths"]
    }
    asset_paths: dict[str, str] = {}
    for entity_id in sorted(required_ids - set(missing)):
        visual = entity_visual_path(task["site"], entities[entity_id])
        if not visual:
            errors.append(f"referenced entity {entity_id} has no website image")
            continue
        try:
            asset = resolve_within(
                root, visual, context=f"{task['id']} website image for {entity_id}"
            )
        except (ValueError, FileNotFoundError) as exc:
            errors.append(str(exc))
            continue
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        if digest in frame_hashes:
            errors.append(
                f"website image for {entity_id} duplicates a released source frame"
            )
        asset_paths[entity_id] = visual
        entity = entities[entity_id]
        if task["site"] == "egomedia":
            required = {
                "thumbnail_path", "channel_title", "published_at",
                "duration_seconds", "view_count",
            }
            absent = sorted(required - set(entity))
            if absent:
                errors.append(
                    f"referenced media entity {entity_id} lacks realistic metadata {absent}"
                )
        elif task["site"] == "egomaps" and not entity.get("location"):
            errors.append(f"referenced map entity {entity_id} has no location")

    supported_fields = set(site_data.get("supported_fields", []))
    for goal_index, goal in enumerate(_goals(task)):
        for state_index, state in enumerate(goal["acceptable_states"]):
            selected = [
                entities[entity_id] for entity_id in state["selected_entity_ids"]
                if entity_id in entities
            ]
            for field, value in state["fields"].items():
                if field not in supported_fields:
                    errors.append(
                        f"goal {goal_index}/{state_index} uses unsupported field {field!r}"
                    )
                if not any(
                    isinstance(entity.get("facts"), dict)
                    and entity["facts"].get(field) == value
                    for entity in selected
                ):
                    errors.append(
                        f"goal {goal_index}/{state_index} answer {field!r}={value!r} "
                        "is not stated in a selected website entity"
                    )
    return _check(
        not errors,
        details={
            "referenced_entity_ids": sorted(required_ids),
            "website_images": asset_paths,
        },
        errors=errors,
    )


def _instruction_check(task: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    instructions = task.get("instructions")
    if not isinstance(instructions, dict) or not instructions:
        return _check(False, errors=["instructions must be a non-empty object"])
    leakage: dict[str, list[str]] = {}
    token_counts: dict[str, int] = {}
    for name, value in instructions.items():
        if not isinstance(value, str) or not value.strip():
            errors.append(f"instruction {name!r} is empty")
            continue
        tokens = re.findall(r"\b\w+\b", value, re.UNICODE)
        token_counts[str(name)] = len(tokens)
        if len(tokens) < 3 or len(tokens) > 120:
            errors.append(
                f"instruction {name!r} has implausible basic length ({len(tokens)} tokens)"
            )
        matches = sorted({match.group(0) for match in LEAKAGE.finditer(value)})
        if matches:
            leakage[str(name)] = matches
            errors.append(f"instruction {name!r} leaks internal terms {matches}")
        prerecorded = sorted({
            match.group(0) for match in PRERECORDED_INPUT_LANGUAGE.finditer(value)
        })
        if prerecorded:
            errors.append(
                f"instruction {name!r} describes prerecorded benchmark input {prerecorded}"
            )
        if "[ANNOTATE" in value.upper() or "TODO" in value.upper():
            errors.append(f"instruction {name!r} contains an unfinished placeholder")
    return _check(
        not errors,
        details={
            "token_counts": token_counts,
            "leakage_matches": leakage,
            "wording_flags": {
                str(name): sorted({
                    match.group(0)
                    for match in REVIEW_WORDING_FLAGS.finditer(str(value))
                })
                for name, value in instructions.items()
                if REVIEW_WORDING_FLAGS.search(str(value))
            },
            "semantic_naturalness_requires_human_review": True,
        },
        errors=errors,
    )


def _positive_record(
    task: dict[str, Any], arm: str, state: Mapping[str, Any],
    *, clarification_question: str | None = None,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    step = 1
    if (task["track"], arm) == ("graft", "injected"):
        actions.append({
            "type": "instruction_injection",
            "step": DYNAMIC_EVENT_STEP,
            "trigger_reason": "task_timeline",
        })
        step = DYNAMIC_EVENT_STEP + 1
    elif (task["track"], arm) == ("refresh", "stream"):
        actions.append({
            "type": "visual_context_update",
            "step": DYNAMIC_EVENT_STEP,
            "trigger_reason": "task_timeline",
        })
        step = DYNAMIC_EVENT_STEP + 1
    elif (task["track"], arm) == ("clarify", "interactive"):
        slot = task["clarification"]["valid_slot"]
        actions.append({
            "type": "clarification_answer",
            "step": 1,
            "slot": slot,
            "question": clarification_question or CLARIFICATION_QUESTIONS[slot],
            "valid": True,
            "payload": task["clarification"]["canonical_answer"],
        })
        step = 2
    actions.extend(
        {"type": "select_entity", "step": step, "entity_id": entity_id}
        for entity_id in state["selected_entity_ids"]
    )
    actions.append({
        "type": "submit",
        "step": step + 1,
        "entity_ids": list(state["selected_entity_ids"]),
    })
    return {
        "task_id": task["id"],
        "track": task["track"],
        "arm": arm,
        "run_status": {"valid": True, "reason": None},
        "steps": step + 1,
        "actions": actions,
        "final_state": copy.deepcopy(dict(state)),
    }


def _checker_check(
    task: dict[str, Any], site_data: dict[str, Any],
    source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    positive_runs = 0
    negative_cases = 0
    canonical_question = None
    if task["track"] == "clarify":
        raw_question = (
            (source_record or {}).get("construct", {})
            .get("task_blueprint", {})
            .get("canonical_question")
        )
        if isinstance(raw_question, str) and raw_question.strip():
            canonical_question = raw_question.strip()
    all_site_ids = [str(entity["entity_id"]) for entity in site_data["entities"]]
    for arm in SUPPORTED_ARMS[task["track"]]:
        goal = goal_for_arm(task, arm)
        state = goal["acceptable_states"][0]
        record = _positive_record(
            task, arm, state, clarification_question=canonical_question
        )
        checked = check_run(task, record)
        if not (
            checked["run_valid"]
            and checked["evidence_status"] == "PASS"
            and checked["e2e_success"] is True
            and checked["policy_pass"] is True
        ):
            errors.append(f"production checker rejects valid {arm} path")
        positive_runs += 1

        acceptable_ids = {
            entity_id
            for option in goal["acceptable_states"]
            for entity_id in option["selected_entity_ids"]
        }
        decoy = next(
            (entity_id for entity_id in all_site_ids if entity_id not in acceptable_ids),
            None,
        )
        if decoy is None:
            errors.append(f"{arm} has no website decoy for negative checker replay")
            continue
        mutations = {
            "wrong_entity": {**state, "selected_entity_ids": [decoy]},
            "target_plus_extra": {
                **state,
                "selected_entity_ids": [*state["selected_entity_ids"], decoy],
            },
            "duplicate_entity": {
                **state,
                "selected_entity_ids": [
                    *state["selected_entity_ids"], state["selected_entity_ids"][0]
                ],
            },
            "not_submitted": {**state, "submission_completed": False},
            "extra_field": {
                **state,
                "fields": {**state["fields"], "unexpected_field": "unexpected"},
            },
        }
        for name, final_state in mutations.items():
            mutated = copy.deepcopy(record)
            mutated["final_state"] = final_state
            rejected = check_run(task, mutated)
            if not (
                rejected["evidence_status"] == "FAIL"
                and rejected["e2e_success"] is False
            ):
                errors.append(f"production checker accepts {arm}/{name}")
            negative_cases += 1

        if (task["track"], arm) in {
            ("graft", "injected"), ("refresh", "stream")
        }:
            missing_event = copy.deepcopy(record)
            missing_event["actions"] = [
                action for action in missing_event["actions"]
                if action["type"] not in {
                    "instruction_injection", "visual_context_update"
                }
            ]
            if check_run(task, missing_event)["run_valid"] is not False:
                errors.append(f"production checker accepts {arm}/missing_event")
            negative_cases += 1

        if (task["track"], arm) == ("clarify", "interactive"):
            wrong_target = next(
                candidate for candidate in task["clarification"]["candidate_ids"]
                if candidate not in state["selected_entity_ids"]
            )
            ignored_answer = copy.deepcopy(record)
            ignored_answer["actions"] = [
                ignored_answer["actions"][0],
                {"type": "select_entity", "step": 2, "entity_id": wrong_target},
                ignored_answer["actions"][-1],
            ]
            if check_run(task, ignored_answer)["policy_pass"] is not False:
                errors.append("production checker accepts interactive/ignored_answer")
            negative_cases += 1
    return _check(
        not errors,
        details={
            "positive_paths": positive_runs,
            "adversarial_paths": negative_cases,
            "arms": list(SUPPORTED_ARMS[task["track"]]),
            "clarification_question_replayed": canonical_question,
        },
        errors=errors,
    )


def _current_review_state(
    task: Mapping[str, Any], root: Path, *, extra_artifacts: Iterable[str] = (),
) -> dict[str, Any]:
    spec_path = root / REVIEW_SPEC
    required = 2
    if spec_path.is_file():
        spec = _read_object(spec_path, "construction review spec")
        policy = spec.get("reviewer_policy", {})
        if isinstance(policy, dict) and isinstance(
            policy.get("required_independent_reviews"), int
        ):
            required = policy["required_independent_reviews"]
    submissions_root = root / REVIEW_SUBMISSIONS / str(task["id"])
    current_admits: list[str] = []
    current_rejects: list[str] = []
    stale_or_invalid: list[str] = []
    required_artifacts = {
        REVIEW_SPEC,
        f"data/tasks/{task['track']}/{task['id']}.json",
        str(task["source"]["dossier"]),
        str(task["source"]["frame_provenance"]),
        SITE_DATA_FILES[str(task["site"])],
        *(
            frame
            for packet in task["visual_packets"]
            for frame in packet["frame_paths"]
        ),
        *extra_artifacts,
    }
    try:
        site_data = load_site_data(root, str(task["site"]))
        entities = {
            str(entity["entity_id"]): entity for entity in site_data["entities"]
        }
        required_artifacts.update(
            visual
            for entity_id in _referenced_entity_ids(task)
            if entity_id in entities
            if (visual := entity_visual_path(str(task["site"]), entities[entity_id]))
        )
    except Exception:
        pass
    for path in sorted(submissions_root.glob("*.json")):
        try:
            submission = _read_object(path, str(path))
            reviewer = str(submission.get("reviewer", ""))
            if (
                submission.get("schema_version") != 5
                or submission.get("task_id") != task["id"]
                or not reviewer
                or reviewer != path.stem
            ):
                raise ValueError("submission identity is invalid")
            artifacts = submission.get("reviewed_artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                raise ValueError("reviewed_artifacts is missing")
            missing_artifacts = sorted(required_artifacts - set(artifacts))
            if missing_artifacts:
                raise ValueError(
                    f"review omitted required artifacts: {missing_artifacts[:5]}"
                )
            for reference, digest in artifacts.items():
                artifact = resolve_within(
                    root, reference, context=f"{task['id']} reviewed artifact"
                )
                if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                    raise ValueError(f"reviewed artifact is stale: {reference}")
            decision = submission.get("decision")
            if decision == "admit":
                current_admits.append(reviewer)
            elif decision == "reject":
                current_rejects.append(reviewer)
            else:
                raise ValueError("review decision is invalid")
        except (ValueError, FileNotFoundError) as exc:
            stale_or_invalid.append(f"{path.name}: {exc}")
    return {
        "required_independent_reviews": required,
        "current_admits": sorted(set(current_admits)),
        "current_rejects": sorted(set(current_rejects)),
        "stale_or_invalid": stale_or_invalid,
        "complete": len(set(current_admits)) >= required and not current_rejects,
    }


def _audit_task(
    task: dict[str, Any], root: Path, *, verify_source_frames: bool,
    is_primary: bool,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}
    site_data: dict[str, Any] | None = None
    dossier: dict[str, Any] | None = None
    source_record: dict[str, Any] | None = None
    provenance_path: Path | None = None

    try:
        site_data = load_site_data(root, task["site"])
        validate_task(task, site_data, root=root, require_assets=True)
        checks["schema_and_frames"] = _check(True, details={
            "frame_count": sum(
                len(packet["frame_paths"]) for packet in task["visual_packets"]
            )
        })
    except Exception as exc:  # one task must not hide the remaining audit rows
        checks["schema_and_frames"] = _check(False, errors=[str(exc)])

    lineage_errors: list[str] = []
    try:
        dossier, source_record, _, provenance_path = _source_material(task, root)
    except Exception as exc:
        lineage_errors.append(str(exc))
    try:
        validate_source_lineage(task, root=root)
    except Exception as exc:
        lineage_errors.append(str(exc))
    if source_record is not None:
        lineage_errors.extend(_claim_errors(source_record))
    checks["source_lineage"] = _check(
        not lineage_errors,
        details={
            "dossier": task.get("source", {}).get("dossier"),
            "frame_provenance": task.get("source", {}).get("frame_provenance"),
            "event_origin": (
                source_record.get("construct", {}).get("event_origin")
                if source_record is not None else None
            ),
        },
        errors=lineage_errors,
    )

    if source_record is not None:
        checks["glass_use_construct_prerequisites"] = _construct_check(
            task, source_record, is_primary=is_primary
        )
    else:
        checks["glass_use_construct_prerequisites"] = _check(
            False, errors=["source construct contract is unavailable"]
        )

    if dossier is not None and source_record is not None and provenance_path is not None:
        try:
            checks["continuous_video"] = _video_check(
                task,
                dossier,
                source_record,
                provenance_path,
                root,
                verify_source_frames=verify_source_frames,
            )
        except Exception as exc:
            checks["continuous_video"] = _check(False, errors=[str(exc)])
    else:
        checks["continuous_video"] = _check(
            False, errors=["source lineage is unavailable"]
        )

    if site_data is not None:
        try:
            checks["site_entities_images_and_answers"] = _site_check(
                task, site_data, root, source_record
            )
        except Exception as exc:
            checks["site_entities_images_and_answers"] = _check(
                False, errors=[str(exc)]
            )
        if source_record is not None:
            try:
                rebuilt = compile_task_from_site(
                    source_record,
                    source_identity=task["source"],
                    site_data=site_data,
                    root=root,
                )
                checks["compiled_from_source_truth"] = _check(
                    rebuilt == task,
                    errors=([] if rebuilt == task else [
                        "task JSON differs from the current source blueprint/site mapping"
                    ]),
                )
            except Exception as exc:
                checks["compiled_from_source_truth"] = _check(
                    False, errors=[str(exc)]
                )
        else:
            checks["compiled_from_source_truth"] = _check(
                False, errors=["source record is unavailable"]
            )
        try:
            checks["production_checker"] = _checker_check(
                task, site_data, source_record
            )
        except Exception as exc:
            checks["production_checker"] = _check(False, errors=[str(exc)])
    else:
        for name in (
            "site_entities_images_and_answers",
            "compiled_from_source_truth",
            "production_checker",
        ):
            checks[name] = _check(False, errors=["site data is unavailable"])

    checks["instruction_form_and_leakage"] = _instruction_check(task)
    failed_checks = sorted(
        name for name, result in checks.items() if result["status"] == "fail"
    )
    integrity_pass = not failed_checks
    video_paths = checks["continuous_video"].get("details", {}).get("video_paths", {})
    review = _current_review_state(
        task,
        root,
        extra_artifacts=(
            str(path) for path in video_paths.values()
            if isinstance(path, str) and path
        ),
    )
    pending_source_items = []
    if source_record is not None:
        pending_source_items = [
            *[str(item) for item in source_record.get("blocking_checks", [])],
            *[str(item) for item in source_record.get("unresolved", [])],
        ]

    if failed_checks or review["current_rejects"]:
        status = "fail"
    elif pending_source_items:
        status = "needs_source_review"
    elif not review["complete"]:
        status = "needs_human_review"
    else:
        status = "task_verified"
    if status not in STATUS_VALUES:
        raise AssertionError(f"unexpected acceptance status: {status}")
    return {
        "task_id": task["id"],
        "track": task["track"],
        "site": task["site"],
        "pilot_scope": "primary_seed" if is_primary else "rescreening_candidate",
        "status": status,
        "integrity_pass": integrity_pass,
        "task_acceptance_ready": status == "task_verified",
        "benchmark_release_ready": False,
        "failed_checks": failed_checks,
        "pending_source_review_items": pending_source_items,
        "human_review": review,
        "checks": checks,
    }


def _batch_quality(
    tasks: list[dict[str, Any]], rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_goal: dict[str, list[str]] = {}
    by_instruction: dict[str, list[str]] = {}
    for task in tasks:
        goal_key = json.dumps(
            {"site": task["site"], "goal": task["goal"]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        by_goal.setdefault(goal_key, []).append(str(task["id"]))
        for value in task.get("instructions", {}).values():
            if isinstance(value, str):
                normalized = " ".join(value.casefold().split())
                by_instruction.setdefault(normalized, []).append(str(task["id"]))
    duplicate_goals = [sorted(ids) for ids in by_goal.values() if len(ids) > 1]
    repeated_instruction_ids = [
        sorted(set(ids)) for ids in by_instruction.values() if len(set(ids)) > 1
    ]
    duplicate_instructions: list[tuple[str, ...]] = sorted(
        {tuple(ids) for ids in repeated_instruction_ids}, key=lambda ids: (len(ids), ids)
    )
    web_filter_updates = [
        row["task_id"] for row in rows
        if any(
            "ordinary website filter" in reason
            for reason in row["checks"]["glass_use_construct_prerequisites"]
            .get("details", {})
            .get("construct_risk_reasons", [])
        )
    ]
    return {
        "duplicate_final_goal_groups": sorted(duplicate_goals),
        "duplicate_instruction_groups": [list(ids) for ids in duplicate_instructions],
        "graft_web_filter_only_risk": sorted(web_filter_updates),
        "interpretation": (
            "These are deterministic diversity warnings, not semantic rejection decisions. "
            "A human must still judge whether two tasks measure meaningfully different situated decisions."
        ),
    }


def build_acceptance_report(
    root: str | Path = ".",
    tasks_path: str | Path = TASKS,
    *,
    verify_source_frames: bool = True,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    task_root = resolve_within(
        workspace, tasks_path, context="task acceptance input", must_exist=False
    )
    tasks = load_tasks(task_root)
    selection = _read_object(workspace / SELECTION, "task selection")
    primary_ids = _selected_task_ids(selection)
    discovered_ids = {str(task["id"]) for task in tasks}
    if not primary_ids <= discovered_ids:
        missing = sorted(primary_ids - discovered_ids)
        raise ValueError(f"primary tasks are missing from the constructed pool: {missing}")
    rows = [
        _audit_task(
            task,
            workspace,
            verify_source_frames=verify_source_frames,
            is_primary=str(task["id"]) in primary_ids,
        )
        for task in tasks
    ]
    statuses = Counter(row["status"] for row in rows)
    tracks = Counter(row["track"] for row in rows)
    sites = Counter(row["site"] for row in rows)
    return {
        "schema_version": 1,
        "scope": "all_constructed_tasks",
        "admission_authority": False,
        "source_frame_reextraction": verify_source_frames,
        "interpretation": (
            "Integrity checks prove file, lineage, website-state and checker consistency only. "
            "Construct validity is reviewed first: a task is rejected from the core benchmark "
            "when one task-start photo plus one complete instruction preserves the same measured "
            "decision, timing, and web commitment. Source and human-review states are pending "
            "gates, not passes. task_verified still requires the registered real-model gate "
            "before benchmark release."
        ),
        "summary": {
            "tasks": len(rows),
            "primary_seed_tasks": len(primary_ids),
            "rescreening_candidates": len(rows) - len(primary_ids),
            "integrity_checks_passed": sum(
                row["integrity_pass"] for row in rows
            ),
            "construct_risk_flagged": sum(
                bool(
                    row["checks"]["glass_use_construct_prerequisites"]
                    .get("details", {})
                    .get("construct_risk_reasons", [])
                )
                for row in rows
            ),
            "statuses": {
                status: statuses.get(status, 0) for status in sorted(STATUS_VALUES)
            },
            "tracks": dict(sorted(tracks.items())),
            "sites": dict(sorted(sites.items())),
        },
        "batch_quality": _batch_quality(tasks, rows),
        "tasks": {row["task_id"]: row for row in rows},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit every constructed task against source video, released frames, frozen "
            "website truth, exact goals, and the production checker."
        )
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument(
        "--output", default="artifacts/reports/task-acceptance.json"
    )
    args = parser.parse_args()
    workspace = Path(args.root).resolve()
    report = build_acceptance_report(workspace, args.tasks)
    output = resolve_within(
        workspace, args.output, context="task acceptance output", must_exist=False
    )
    write_json_atomic(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if report["summary"]["statuses"]["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
