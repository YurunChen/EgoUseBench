from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from egocua.selection import SELECTION_PATH
from tools.build import build_final_task_set
from tools.build.build_final_task_set import build_release_plan
from tests.helpers import task


def test_package_names_the_workflow_as_its_only_admission_authority(
    tmp_path, monkeypatch
) -> None:
    definition = task("graft")
    task_id = definition["id"]
    selection = {"task_set": {"graft": [task_id], "refresh": [], "clarify": []}}
    selection_path = tmp_path / SELECTION_PATH
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection))
    source_path = tmp_path / "data/tasks/graft" / f"{task_id}.json"
    retired_path = source_path.with_name("retired-task.json")
    monkeypatch.setattr(
        build_final_task_set,
        "_compiled_tasks",
        lambda _: {
            task_id: (definition, source_path),
            "retired-task": ({**definition, "id": "retired-task"}, retired_path),
        },
    )
    monkeypatch.setattr(
        build_final_task_set,
        "build_confidence_report",
        lambda _: {
            "summary": {},
            "tasks": {task_id: {"route": "coding_agent_high_confidence"}},
        },
    )
    monkeypatch.setattr(
        build_final_task_set,
        "build_difficulty_report",
        lambda _: {
            "summary": {},
            "tasks": {task_id: {"structural": {"band": "unclassified", "burdens": []}}},
        },
    )

    workflow = Mock(spec=build_final_task_set.SynthesisWorkflow)
    workflow.release_evidence.return_value = {}
    monkeypatch.setattr(build_final_task_set, "SynthesisWorkflow", lambda _: workflow)

    build_final_task_set.build(tmp_path)

    manifest = json.loads(
        (tmp_path / build_final_task_set.FINAL_ROOT / "manifest.json").read_text()
    )
    assert manifest["admission_authority"] == "SynthesisWorkflow.status"
    assert not list((tmp_path / build_final_task_set.FINAL_ROOT).rglob(retired_path.name))
    workflow.release_evidence.assert_called_once_with([task_id])


def _task(task_id: str, track: str, subtype: str) -> dict:
    return {
        "task_id": task_id,
        "track": track,
        "subtype": subtype,
        "difficulty": {
            "structural": {
                "band": "compound",
                "burdens": ["change_handling", "compositional_discrimination"],
                "design_risks": [],
            }
        },
    }


def test_release_plan_contains_every_declared_task_without_reserve_split() -> None:
    selection = {
        "task_set": {
            "graft": ["graft-01"],
            "refresh": ["refresh-01"],
            "clarify": [],
        }
    }
    tasks = [
        _task("graft-01", "graft", "refine"),
        _task("refresh-01", "refresh", "replace"),
    ]

    plan = build_release_plan(selection, tasks)

    assert plan["policy"] == "all_task_set_members_passed_lifecycle"
    assert plan["summary"]["release_tasks"] == 2
    assert plan["summary"]["open_slots"] == 0
    assert plan["summary"]["reserve_tasks"] == 0
    assert plan["task_ids"] == ["graft-01", "refresh-01"]


def test_release_plan_rejects_missing_or_extra_tasks() -> None:
    selection = {"task_set": {"graft": ["graft-01"], "refresh": [], "clarify": []}}

    with pytest.raises(ValueError, match="does not match"):
        build_release_plan(selection, [])


def test_package_refreshes_confidence_before_release_validation(
    tmp_path, monkeypatch
) -> None:
    selection = {"task_set": {"graft": ["graft-01"], "refresh": [], "clarify": []}}
    current = {"tasks": {"graft-01": {"route": "human_review_complete"}}}
    selection_path = tmp_path / SELECTION_PATH
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text(json.dumps(selection))
    confidence_path = tmp_path / build_final_task_set.CONFIDENCE_REPORT
    confidence_path.parent.mkdir(parents=True)
    confidence_path.write_text(json.dumps({"tasks": {"graft-01": {"route": "repair"}}}))
    monkeypatch.setattr(
        build_final_task_set, "_compiled_tasks", lambda _: {"graft-01": None}
    )
    monkeypatch.setattr(
        build_final_task_set, "build_confidence_report", lambda _: current
    )

    def validate_release(_workflow, selected):
        assert set(selected) == {"graft-01"}
        assert json.loads(confidence_path.read_text()) == current
        raise ValueError("release validation reached with current confidence")

    monkeypatch.setattr(
        build_final_task_set.SynthesisWorkflow,
        "release_evidence",
        validate_release,
    )
    with pytest.raises(
        ValueError, match="release validation reached with current confidence"
    ):
        build_final_task_set.build(tmp_path)
    assert not (tmp_path / build_final_task_set.FINAL_ROOT).exists()
