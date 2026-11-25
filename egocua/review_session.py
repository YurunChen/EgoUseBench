"""Reuse a live review projection without moving task-state ownership into the UI."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from threading import RLock
from typing import Any

from .environment import (
    SITE_DATA_FILES,
    entity_visual_path,
    load_site_data,
    site_data_request,
    task_public_site_data,
)
from .experiment import TASK_REVIEW_PLAN
from .review_workspace import SELECTION, SUBMISSIONS, ReviewWorkspace
from .synthesis_reports import REPORTS
from .sources import find_record
from .synthesis_workflow import SynthesisWorkflow, status_summary
from .workspace import read_json_strict


def _stamp(path: Path) -> tuple[int, int] | None:
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    return info.st_mtime_ns, info.st_size


class ReviewSession:
    """Keep the shared lifecycle result; refresh only owners whose files changed."""

    def __init__(self, workspace: ReviewWorkspace) -> None:
        self._lock = RLock()
        self._all = workspace
        self._dependencies: dict[Path, set[str]] = {}
        self._stamps: dict[Path, tuple[int, int] | None] = {}
        self._lifecycle: dict[str, Any] | None = None
        self._records: dict[Path, tuple[Any, dict[str, Any]]] = {}

    def _inputs(self, workspace: ReviewWorkspace) -> dict[Path, set[str]]:
        root = workspace.root
        tasks = workspace.tasks
        members = set(workspace.ordered_ids)
        owners: dict[Path, set[str]] = defaultdict(set)
        shared = [
            workspace.spec_path,
            root / SELECTION,
            root / TASK_REVIEW_PLAN,
            root / "data/instructions/authored.json",
            root / "data/reviews/task-qualification.json",
            root / "artifacts/reports/confidence-review-packets/index.json",
            *(
                root / REPORTS[name].path
                for name in ("acceptance", "smoke", "role", "assessment")
            ),
        ]
        for path in shared:
            owners[path].update(members)
        sites = {task["site"] for task in tasks.values()}
        for site in sites:
            site_members = {tid for tid, task in tasks.items() if task["site"] == site}
            for path in (root / "sites" / site).glob("*.json"):
                owners[path].update(site_members)
            data = load_site_data(root, site, require_assets=False)
            all_entity_ids = [entity["entity_id"] for entity in data["entities"]]
            entity_owners: dict[str, set[str]] = defaultdict(set)
            for tid in site_members:
                for entity_id in (
                    tasks[tid].get("truth", {}).get("site_scope_entity_ids")
                    or all_entity_ids
                ):
                    entity_owners[entity_id].add(tid)
            for entity in data["entities"]:
                affected = entity_owners.get(entity["entity_id"])
                if not affected:
                    continue
                image = entity_visual_path(site, entity)
                if image:
                    owners[root / image].update(affected)
                for preview in entity.get("public_previews", []):
                    owners[root / preview["image_path"]].update(affected)
        for tid, task in tasks.items():
            paths = [root / "data/tasks" / task["track"] / f"{tid}.json"]
            paths.extend(
                root / path
                for packet in task["visual_packets"]
                for path in packet["frame_paths"]
            )
            paths.extend(
                root / task["source"][key]
                for key in ("dossier", "frame_provenance")
                if key in task.get("source", {})
            )
            submissions = root / SUBMISSIONS / tid
            paths.extend([submissions, *submissions.glob("*.json")])
            for path in paths:
                owners[path].add(tid)
        return dict(owners)

    def _record_paths(self, workspace: ReviewWorkspace) -> set[Path]:
        root = workspace.root
        return {
            root / "data/instructions/authored.json",
            root / "data/reviews/task-qualification.json",
            root / "artifacts/reports/confidence-review-packets/index.json",
            *(
                root / REPORTS[name].path
                for name in ("acceptance", "smoke", "role", "assessment")
            ),
            *(
                root / task["source"][key]
                for task in workspace.tasks.values()
                for key in ("dossier", "frame_provenance")
                if key in task.get("source", {})
            ),
            *(
                root / SITE_DATA_FILES[site]
                for site in {t["site"] for t in workspace.tasks.values()}
            ),
        }

    def _task_records(
        self, path: Path, workspace: ReviewWorkspace
    ) -> tuple[Any, dict[str, Any]]:
        if not path.is_file():
            return None, {}
        if path in {workspace.root / value for value in SITE_DATA_FILES.values()}:
            site = next(
                site
                for site, value in SITE_DATA_FILES.items()
                if path == workspace.root / value
            )
            data = load_site_data(workspace.root, site, require_assets=False)
            return None, {
                tid: task_public_site_data(task, data)
                for tid, task in workspace.tasks.items()
                if task["site"] == site
            }
        document = read_json_strict(path)
        relative = path.relative_to(workspace.root).as_posix()
        source_tasks = {
            tid: task
            for tid, task in workspace.tasks.items()
            if task.get("source", {}).get("dossier") == relative
        }
        if source_tasks:
            return {key: value for key, value in document.items() if key != "tasks"}, {
                tid: find_record(
                    document, task["source"].get("task_id", tid), context=relative
                )
                for tid, task in source_tasks.items()
            }
        frame_tasks = {
            tid: task
            for tid, task in workspace.tasks.items()
            if task.get("source", {}).get("frame_provenance") == relative
        }
        if frame_tasks:
            header = {
                key: value
                for key, value in document.items()
                if key not in {"frames", "videos", "packet_sampling"}
            }
            rows = {}
            for tid, task in frame_tasks.items():
                paths = {
                    frame
                    for packet in task["visual_packets"]
                    for frame in packet["frame_paths"]
                }
                frames = [
                    frame
                    for frame in document["frames"]
                    if frame["frame_path"] in paths
                ]
                media_ids = {
                    frame["media_id"] for frame in frames if "media_id" in frame
                }
                rows[tid] = {
                    "frames": frames,
                    "videos": {
                        key: value
                        for key, value in document.get("videos", {}).items()
                        if key in media_ids
                    },
                    "sampling": [
                        row
                        for row in document.get("packet_sampling", [])
                        if row["task_id"] == task.get("source", {}).get("task_id", tid)
                    ],
                }
            return header, rows
        if relative == "data/instructions/authored.json":
            return document["schema_version"], document["instructions"]
        if relative == "data/reviews/task-qualification.json":
            baseline = document.get("legacy_baseline", {})
            current = document["tasks"]
            inherited = baseline.get("tasks", {})
            return baseline.get("policy"), {
                tid: (current.get(tid), inherited.get(tid))
                for tid in current.keys() | inherited.keys()
            }
        if path == workspace.root / REPORTS["assessment"].path:
            return document["schema_version"], {
                row["task_id"]: row for row in document["results"]
            }
        return document["schema_version"], document["tasks"]

    def current(self) -> tuple[dict[str, ReviewWorkspace], dict[str, Any]]:
        with self._lock, site_data_request(self._all.root):
            changed = {
                path
                for path in self._dependencies
                if _stamp(path) != self._stamps[path]
            }
            if self._lifecycle is None or changed:
                all_workspace = ReviewWorkspace.open(
                    self._all.root,
                    self._all.reviewer_id,
                    include_all_tasks=True,
                )
                affected: set[str] = set()
                records = dict(self._records)
                for path in changed:
                    if path not in records:
                        affected.update(self._dependencies[path])
                        continue
                    before_global, before_rows = records[path]
                    after_global, after_rows = self._task_records(path, all_workspace)
                    records[path] = after_global, after_rows
                    if before_global != after_global:
                        affected.update(self._dependencies[path])
                    else:
                        affected.update(
                            tid
                            for tid in before_rows.keys() | after_rows.keys()
                            if before_rows.get(tid) != after_rows.get(tid)
                        )
                if self._lifecycle is None:
                    affected.update(self._all.ordered_ids)
                    records = {
                        path: self._task_records(path, all_workspace)
                        for path in self._record_paths(all_workspace)
                    }
                for path in self._record_paths(all_workspace) - records.keys():
                    records[path] = self._task_records(path, all_workspace)
                if all_workspace.tasks != self._all.tasks:
                    for site in {task["site"] for task in all_workspace.tasks.values()}:
                        path = all_workspace.root / SITE_DATA_FILES[site]
                        records[path] = self._task_records(path, all_workspace)
                dependencies = self._inputs(all_workspace)
                stamps = {path: _stamp(path) for path in dependencies}
                current_ids = set(all_workspace.ordered_ids)
                affected &= current_ids
                if (
                    self._lifecycle is None
                    or set(self._lifecycle["tasks"]) != current_ids
                ):
                    affected = current_ids
                update = (
                    SynthesisWorkflow(self._all.root).status(affected)
                    if affected
                    else {
                        "schema_version": self._lifecycle["schema_version"],
                        "tasks": {},
                        "review_evidence": {},
                    }
                )
                previous = self._lifecycle or {"tasks": {}, "review_evidence": {}}
                tasks = {
                    tid: row
                    for tid, row in previous["tasks"].items()
                    if tid in current_ids and tid not in affected
                }
                tasks.update(update["tasks"])
                evidence = {
                    tid: row
                    for tid, row in previous["review_evidence"].items()
                    if tid in current_ids and tid not in affected
                }
                evidence.update(update["review_evidence"])
                lifecycle = {
                    "schema_version": update["schema_version"],
                    "tasks": tasks,
                    "review_evidence": evidence,
                    "summary": status_summary(tasks),
                }
                self._all = all_workspace
                self._dependencies = dependencies
                self._stamps = stamps
                self._lifecycle = lifecycle
                self._records = records
            return {
                "all": self._all,
                "pending": self._all.pending(self._lifecycle),
                "quality": self._all,
            }, self._lifecycle
