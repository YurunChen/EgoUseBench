from __future__ import annotations

import json

import pytest

from egocua.retest_scope import (
    COMPLETION_ANOMALY_POLICY_VERSION,
    binomial_upper_tail,
    build_completion_anomaly_plan,
    completion_anomaly,
    load_repair_task_ids,
)
from egocua.synthesis_workflow import SynthesisWorkflow
from tests.helpers import task


def test_perfect_completion_does_not_request_repair() -> None:
    complete = {"valid_runs": 3, "expected_runs": 3, "complete": True}

    assert completion_anomaly({**complete, "successes": 3}) == {
        "status": "observed",
        "modify_task": False,
        "valid_runs": 3,
        "successes": 3,
        "success_rate": 1.0,
    }
    assert completion_anomaly({**complete, "successes": 2})["modify_task"] is False
    assert binomial_upper_tail(2, 3, 0.30) == 0.216


def test_incomplete_calibration_never_modifies_a_task() -> None:
    decision = completion_anomaly(
        {
            "valid_runs": 2,
            "expected_runs": 3,
            "successes": 2,
            "complete": False,
        }
    )

    assert decision["status"] == "insufficient_evidence"
    assert decision["modify_task"] is False


def test_completion_report_does_not_generate_repair_scope(tmp_path) -> None:
    calibration = {
        "tasks": {
            "over-easy": {
                "task_sha256": "a",
                "track": "graft",
                "site": "egomedia",
                "subtype": "refine",
                "valid_runs": 3,
                "expected_runs": 3,
                "successes": 3,
                "complete": True,
            },
            "ordinary": {
                "task_sha256": "b",
                "track": "graft",
                "site": "egomedia",
                "subtype": "refine",
                "valid_runs": 3,
                "expected_runs": 3,
                "successes": 2,
                "complete": True,
            },
            "pending": {
                "task_sha256": "c",
                "track": "refresh",
                "site": "egomart",
                "subtype": "replace",
                "valid_runs": 1,
                "expected_runs": 3,
                "successes": 1,
                "complete": False,
            },
        }
    }

    plan = build_completion_anomaly_plan(calibration, source_path="calibration.json")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    assert plan["policy_version"] == COMPLETION_ANOMALY_POLICY_VERSION
    assert plan["repair_task_ids"] == []
    assert load_repair_task_ids(path) == frozenset()


def test_explicit_historical_repair_scope_remains_readable(tmp_path) -> None:
    path = tmp_path / "old-plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy_version": "task-local-exact-binomial-v1",
                "repair_task_ids": ["historical-task"],
            }
        )
    )
    assert load_repair_task_ids(path) == frozenset({"historical-task"})


def test_one_valid_run_is_an_observation_not_a_difficulty_verdict() -> None:
    result = completion_anomaly(
        {
            "valid_runs": 1,
            "expected_runs": 1,
            "successes": 1,
            "complete": True,
        }
    )
    assert result == {
        "status": "observed",
        "modify_task": False,
        "valid_runs": 1,
        "successes": 1,
        "success_rate": 1.0,
    }


@pytest.mark.parametrize("counts", [(-1, 0, 0), (2, 1, 1), (0, 2, 1)])
def test_inconsistent_completion_counts_are_rejected(counts) -> None:
    successes, valid, expected = counts
    with pytest.raises(ValueError, match="completion counts"):
        completion_anomaly(
            {
                "successes": successes,
                "valid_runs": valid,
                "expected_runs": expected,
                "complete": True,
            }
        )


def test_empty_completion_has_no_success_rate() -> None:
    result = completion_anomaly({})
    assert result["status"] == "insufficient_evidence"
    assert result["success_rate"] is None


def test_qualification_gates_only_the_selected_task(tmp_path, monkeypatch) -> None:
    selected = task("graft")
    selected["id"] = "selected"

    def status(workflow, task_ids):
        assert task_ids == ["selected"]
        return {"tasks": {"selected": {"next_action": "run_development_trial"}}}

    monkeypatch.setattr(SynthesisWorkflow, "status", status)
    SynthesisWorkflow(tmp_path).require_development_inputs([selected["id"]])
