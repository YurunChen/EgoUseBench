from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import egocua.attribution_sources as sources
from egocua.workspace import write_json_atomic
from tests.test_failure_attribution import _checked


@pytest.fixture
def current_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "study"
    records, stored, executions = {}, [], []
    settings = ("computer_use", "offline_ego_use", "online")
    for index, setting in enumerate(settings):
        run_id = f"run-{index}"
        alias = "compressed" if setting == "online" else setting
        records[run_id] = {
            "task_id": "task-1", "track": "graft", "condition": {"name": alias},
            "meta": {"run_id": run_id, "model": "model-a"},
            "task_snapshot": {"id": "task-1"},
        }
        stored.append(_checked(run_id=run_id))
        executions.append({
            "run_id": run_id, "artifact_directory": f"runs/{run_id}", "valid": True,
            "run": {"arm": setting, "condition": setting, "task_id": "task-1"},
            "scoring": {"task_source": "current_frozen_task"},
        })
    write_json_atomic(directory / "plan.json", {"execution_results": executions})
    write_json_atomic(directory / "results.json", {"runs": stored})
    monkeypatch.setattr(sources, "load_run_record", lambda path: deepcopy(records[path.name]))
    monkeypatch.setattr(sources, "load_frozen_study", lambda path: ([
        SimpleNamespace(name=setting, tasks=[{"id": "task-1"}]) for setting in settings
    ], None, None))
    monkeypatch.setattr(sources, "rescore_run", lambda task, record: _checked(run_id=record["meta"]["run_id"]))
    return tmp_path


def test_study_source_scopes_human_adjudications(current_study: Path) -> None:
    evidence = {"adjudications": {
        "run-0": {"stage": "L", "reviewer_id": "user", "reason": "Review."},
        "outside": {"stage": "S", "reviewer_id": "user", "reason": "Review."},
    }}
    _, scoped = sources.study_source(["study"], evidence, current_study)
    assert list(scoped["adjudications"]) == ["run-0"]


def test_current_study_uses_registered_setting_not_reused_condition_alias(current_study):
    source, _ = sources.study_source(["study"], {}, current_study)
    rows = source["failure_attributions"]
    assert [row["setting"] for row in rows] == ["computer_use", "offline_ego_use", "online"]
    assert source["study_coverage"][0]["settings"]["online"] == {"runs": 1, "failures": 1}
    assert all(row["scoring_goal"]["acceptable_states"] for row in rows)


def test_current_study_rejects_changed_recorded_outcome(current_study):
    path = current_study / "study/results.json"
    from egocua.workspace import read_json_strict

    result = read_json_strict(path)
    result["runs"][0]["e2e_success"] = True
    write_json_atomic(path, result)
    with pytest.raises(ValueError, match="differs from recorded result"):
        sources.study_source(["study"], {}, current_study)


def test_current_study_rejects_duplicate_result_rows(current_study):
    path = current_study / "study/results.json"
    from egocua.workspace import read_json_strict

    result = read_json_strict(path)
    result["runs"].append(deepcopy(result["runs"][0]))
    write_json_atomic(path, result)
    with pytest.raises(ValueError, match="duplicate result"):
        sources.study_source(["study"], {}, current_study)
