"""Blind human audit workspace for reviewing PULSE trajectory classifications."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Any

from .failure_attribution import (
    FAILURE_EVIDENCE_PATH,
    FAILURE_REPORT_PATH,
    PULSE_LABELS,
    PULSE_STAGES,
    REVIEW_PACKET_FIELDS,
    historical_review_context,
    referenced_public_entities,
)
from .environment import entity_visual_path
from .frame_inputs import observation_images
from .workspace import write_json_atomic


class FailureReviewError(ValueError):
    pass


def _load_evidence(path: Path) -> dict[str, Any]:
    evidence = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {
            "schema_version": 1,
            "assessments": [],
            "counterfactuals": {},
            "ratings": [],
        }
    )
    if not isinstance(evidence, dict) or not isinstance(evidence.get("ratings"), list):
        raise FailureReviewError("PULSE human ratings are malformed")
    return evidence


def save_adjudications(root: str | Path, decisions: dict[str, dict[str, str]]) -> None:
    """Record final human decisions without changing the original assessments."""
    workspace = Path(root).resolve()
    report = json.loads((workspace / FAILURE_REPORT_PATH).read_text(encoding="utf-8"))
    rows = {row["run_id"]: row for row in report["attributions"]}
    for run_id, decision in decisions.items():
        row = rows.get(run_id)
        if (row is None or row["run_verdict"] != "valid_model_failure"
                or row["attribution_status"] == "resolved"
                or decision.get("stage") not in PULSE_STAGES
                or not decision.get("reviewer_id") or not decision.get("reason")):
            raise FailureReviewError(f"invalid human adjudication: {run_id}")
    path = workspace / FAILURE_EVIDENCE_PATH
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        evidence = _load_evidence(path)
        adjudications = evidence.setdefault("adjudications", {})
        for run_id, decision in decisions.items():
            previous = adjudications.get(run_id)
            if previous is not None and previous != decision:
                raise FailureReviewError(f"adjudication already exists: {run_id}")
            adjudications[run_id] = decision
        write_json_atomic(path, evidence)


class FailureReviewWorkspace:
    def __init__(
        self,
        root: Path,
        reviewer_id: str,
        packets: dict[str, dict[str, Any]],
        evidence: dict[str, Any],
        source_exclusions: list[dict[str, Any]] | None = None,
    ) -> None:
        self.root = root
        self.reviewer_id = reviewer_id
        self.packets = packets
        self.evidence = evidence
        self.source_exclusions = source_exclusions or []
        self.scope = "all"

    @classmethod
    def open(
        cls, root: str | Path, reviewer_id: str, *, scope: str = "all",
    ) -> "FailureReviewWorkspace":
        if scope not in {"all", "validation"}:
            raise FailureReviewError("未知的归因复核范围")
        workspace = Path(root).resolve()
        report_path = workspace / FAILURE_REPORT_PATH
        if not report_path.is_file():
            if scope == "validation":
                raise FailureReviewError("尚未生成留出复核材料")
            return cls(
                workspace,
                reviewer_id,
                {},
                {
                    "schema_version": 1,
                    "assessments": [],
                    "counterfactuals": {},
                    "ratings": [],
                },
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        packets = report.get("human_review", {}).get("packets", [])
        if not isinstance(packets, list):
            raise FailureReviewError("PULSE report review packets are malformed")
        by_run = {
            str(packet.get("run_id")): {
                key: packet[key] for key in REVIEW_PACKET_FIELDS if key in packet
            }
            for packet in packets
            if isinstance(packet, dict) and packet.get("run_id")
        }
        evidence = _load_evidence(workspace / FAILURE_EVIDENCE_PATH)
        if scope == "validation":
            required = set(evidence.get("judge_validation", {}).get("validation_run_ids", []))
            if not required:
                raise FailureReviewError("尚未登记独立留出样本")
            missing = required - by_run.keys()
            if missing:
                raise FailureReviewError(f"留出材料缺少 {len(missing)} 条运行，需先补齐")
            by_run = {key: value for key, value in by_run.items() if key in required}
        exclusions = report.get("source_exclusions", [])
        if not isinstance(exclusions, list):
            raise FailureReviewError("source exclusions are malformed")
        if scope == "validation":
            exclusions = [row for row in exclusions if row.get("run_id") in required]
        result = cls(workspace, reviewer_id, by_run, evidence, exclusions)
        result.scope = scope
        return result

    def _own_ratings(self) -> dict[str, dict[str, Any]]:
        return {
            str(row.get("run_id")): row
            for row in self.evidence["ratings"]
            if isinstance(row, dict) and row.get("rater_id") == self.reviewer_id
        }

    def overview(self) -> dict[str, Any]:
        reviewed = set(self._own_ratings())
        tasks = [
            {
                **{key: value for key, value in packet.items() if key != "model"},
                "status": "submitted" if run_id in reviewed else "todo",
            }
            for run_id, packet in sorted(self.packets.items())
        ]
        return {
            "reviewer": self.reviewer_id,
            "scope": self.scope,
            "total": len(tasks),
            "submitted": sum(row["status"] == "submitted" for row in tasks),
            "todo": sum(row["status"] == "todo" for row in tasks),
            "tasks": tasks,
            "source_blocked": len(self.source_exclusions),
            "source_exclusions": self.source_exclusions,
        }

    def case(self, run_id: str) -> dict[str, Any]:
        packet = self.packets.get(run_id)
        if packet is None:
            raise FailureReviewError(f"unknown PULSE review run: {run_id}")
        diagnostic = historical_review_context(packet, self.root)
        diagnostic.get("run_identity", {}).pop("model", None)
        wearer_inputs = []
        for packet_index, visual_packet in enumerate(diagnostic.get("task_context", {}).get("visual_packets", [])):
            for image in observation_images(visual_packet, self.root):
                wearer_inputs.append({
                    "packet_index": packet_index, "step": visual_packet["released_at_step"],
                    "label": image.label,
                    "data_url": f"data:{image.media_type};base64,{image.data}",
                })
        web_screenshots = {}
        for index, web in enumerate(diagnostic.get("web_observations", [])):
            screenshot = (web.get("state") or {}).get("screenshot_path")
            if screenshot:
                path = Path(screenshot)
                if path.is_absolute():
                    path = path.resolve().relative_to(self.root)
                web_screenshots[index] = path.as_posix()
        site = diagnostic.get("site_context")
        entity_evidence = [{"record": entity,
                            "image_path": entity_visual_path(site["site"], entity)}
                           for entity in referenced_public_entities(diagnostic)] if site else []
        return {
            "scope": self.scope,
            "packet": {
                **{key: value for key, value in packet.items() if key != "model"},
                "diagnostic": diagnostic,
            },
            "wearer_inputs": wearer_inputs,
            "web_screenshots": web_screenshots,
            "entity_evidence": entity_evidence,
            "rating": self._own_ratings().get(run_id),
            "stages": [
                {"value": stage, **PULSE_LABELS[stage]} for stage in PULSE_STAGES
            ],
        }

    def submit(
        self,
        run_id: str,
        stage: str | None,
        evidence: str,
        *,
        first_decisive_step: int | None = None,
        run_verdict: str = "valid_model_failure",
    ) -> None:
        if run_id not in self.packets:
            raise FailureReviewError(f"unknown PULSE review run: {run_id}")
        if stage not in {*PULSE_STAGES, None}:
            raise FailureReviewError(f"invalid PULSE stage: {stage!r}")
        if first_decisive_step is not None and (
            type(first_decisive_step) is not int or first_decisive_step < 0
        ):
            raise FailureReviewError("关键步骤必须是非负整数")
        if run_verdict not in {
            "valid_model_failure",
            "task_or_site_defect",
            "invalid_run",
        }:
            raise FailureReviewError("运行判断无效")
        if run_verdict != "valid_model_failure" and stage is not None:
            raise FailureReviewError("任务疑点或无效运行不填写失败类别")
        if not evidence.strip():
            raise FailureReviewError("请填写可观察证据")
        destination = self.root / FAILURE_EVIDENCE_PATH
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Keep the lock on a stable file: atomic replacement changes the JSON inode.
        with destination.with_suffix(".json.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            latest = _load_evidence(destination)
            ratings = [
                row
                for row in latest["ratings"]
                if not (
                    isinstance(row, dict)
                    and row.get("run_id") == run_id
                    and row.get("rater_id") == self.reviewer_id
                )
            ]
            previous = next(
                (
                    row
                    for row in latest["ratings"]
                    if row.get("run_id") == run_id
                    and row.get("rater_id") == self.reviewer_id
                ),
                None,
            )
            rating = {
                "run_id": run_id,
                "rater_id": self.reviewer_id,
                "pulse_stage": stage,
                "evidence": evidence.strip(),
                "first_decisive_step": first_decisive_step,
                "run_verdict": run_verdict,
            }
            if previous is not None:
                rating["review_history"] = [
                    *previous.get("review_history", []),
                    {
                        key: value
                        for key, value in previous.items()
                        if key != "review_history"
                    },
                ]
            ratings.append(rating)
            latest["ratings"] = ratings
            write_json_atomic(destination, latest)
            self.evidence = latest
