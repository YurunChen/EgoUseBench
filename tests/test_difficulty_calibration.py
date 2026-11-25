from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from egocua.difficulty_calibration import aggregate_calibration, failure_mode, protocol_mismatch
from egocua.experiment import (
    ExperimentPlan,
    expand_runs,
    load_experiment_plan,
    scheduled_indexes,
)
from tests.helpers import PLAN, materialize_assets, products, task


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/archive/difficulty-calibration-qwen3-5-flash.json"


def test_calibration_plan_is_online_commitment_triggered_and_one_shot() -> None:
    plan = load_experiment_plan(PLAN_PATH)

    assert plan.model == "qwen3.5-flash"
    assert plan.default_release_step == 12
    assert plan.release_trigger == "first_native_commitment"
    assert plan.replicate_ids == (0,)
    assert all(
        plan.conditions_for(track) == ("online",)
        for track in (
            "graft",
            "refresh",
            "clarify",
        )
    )


def _protocol_record(plan: ExperimentPlan, schedule_index: int) -> dict[str, Any]:
    return {
        "condition": plan.condition("online").as_dict(),
        "meta": {
            "agent": plan.adapter,
            "model": plan.model,
            "model_version": plan.model_version,
            "transport": "browsergym-openai-vision-v1",
            "max_steps": plan.max_steps,
            "schedule_index": schedule_index,
            "default_release_step": plan.default_release_step,
            "reference_steps": plan.reference_steps,
            "runtime": {"release_trigger": plan.release_trigger},
        },
    }


def test_calibration_protocol_binds_release_timing_and_runtime() -> None:
    plan = load_experiment_plan(PLAN_PATH)
    definition = task("graft")
    schedule_index = scheduled_indexes([definition], plan)[
        (definition["id"], "online", 0)
    ]
    record = _protocol_record(plan, schedule_index)

    assert protocol_mismatch(record, plan, schedule_index) is None
    record["meta"]["default_release_step"] = 2
    assert (
        protocol_mismatch(record, plan, schedule_index) == "wrong_default_release_step"
    )
    record["meta"]["default_release_step"] = plan.default_release_step
    record["meta"]["runtime"]["release_trigger"] = "timeline_step"
    assert protocol_mismatch(record, plan, schedule_index) == "wrong_release_trigger"


@pytest.mark.parametrize(
    "field",
    [
        "agent",
        "model",
        "model_version",
        "max_steps",
        "schedule_index",
        "default_release_step",
        "reference_steps",
        "runtime",
    ],
)
def test_calibration_rejects_missing_protocol_fields(field: str) -> None:
    record = _protocol_record(PLAN, 0)
    del record["meta"][field]

    assert protocol_mismatch(record, PLAN, 0) == f"missing_{field}"


def test_calibration_rejects_missing_release_mechanism() -> None:
    record = _protocol_record(PLAN, 0)
    record["meta"]["runtime"] = {}

    assert protocol_mismatch(record, PLAN, 0) == "missing_release_trigger"


def test_calibration_still_requires_browsergym_transport() -> None:
    record = _protocol_record(PLAN, 0)
    record["meta"]["transport"] = "other-transport"

    assert protocol_mismatch(record, PLAN, 0) == "wrong_transport"


def test_calibration_rejects_other_registered_conditions() -> None:
    record = _protocol_record(PLAN, 0)
    record["condition"] = PLAN.condition("upfront").as_dict()

    assert protocol_mismatch(record, PLAN, 0) == "wrong_condition"


def test_calibration_rejects_changed_condition_factors() -> None:
    record = _protocol_record(PLAN, 0)
    record["condition"]["scope"] = "none"

    assert protocol_mismatch(record, PLAN, 0) == "wrong_condition"


def test_calibration_aggregate_excludes_missing_runs_from_pass_rate() -> None:
    rows: list[Mapping[str, Any]] = [
        {
            "expected_runs": 3,
            "valid_runs": 2,
            "successes": 1,
            "complete": False,
            "pass_all_resets": None,
            "pass_any_reset": None,
            "replicates": {
                "0": {"status": "valid", "failure_mode": "success"},
                "1": {"status": "valid", "failure_mode": "wrong_final_state"},
                "2": {"status": "missing"},
            },
        }
    ]

    summary = aggregate_calibration(rows)

    assert summary["pass_at_1"] == 0.5
    assert summary["complete_tasks"] == 0
    assert summary["pass_all_resets"] is None


def test_calibration_failure_mode_uses_checker_facts_not_error_text() -> None:
    checked = {
        "track": "graft",
        "run_valid": True,
        "e2e_success": False,
        "stale_commitment": True,
    }

    assert failure_mode(checked) == "stale_after_update"


@pytest.mark.parametrize(
    ("task_args", "single_task_id", "batch_scope"),
    [
        (["--task-id", "task-a"], "task-a", None),
        (
            ["--task-id", "task-a", "--task-id", "task-b"],
            None,
            frozenset({"task-a", "task-b"}),
        ),
    ],
)
def test_qualification_cli_limits_trial_to_named_candidates(
    tmp_path, monkeypatch, task_args, single_task_id, batch_scope
):
    from tools import run_task_qualification as runner

    loaded = []

    def load_candidates(root, ids):
        loaded.extend(ids)
        return [{"id": key} for key in ids]

    monkeypatch.setattr(runner, "load_construction_tasks", load_candidates)

    async def run_batch(**kwargs):
        assert kwargs["task_id"] == single_task_id
        assert kwargs["requested_task_ids"] == batch_scope
        assert kwargs["concurrency"] == 2
        return {"status": "complete", "report": {"summary": {}}}

    monkeypatch.setattr(runner, "_run_batch", run_batch)
    assert runner.main(["--root", str(tmp_path), "--concurrency", "2", *task_args]) == 0
    assert loaded == (["task-a"] if batch_scope is None else ["task-a", "task-b"])


def test_qualification_cli_rejects_duplicate_named_candidates(capsys):
    from tools import run_task_qualification as runner

    with pytest.raises(SystemExit, match="2"):
        runner.main(["--task-id", "task-a", "--task-id", "task-a"])
    assert "--task-id values must be unique" in capsys.readouterr().err


@pytest.mark.parametrize(
    "next_action", ["run_development_trial", "build_release", "resolve_source_review"]
)
def test_development_trials_share_admission_and_the_frozen_executor(
    tmp_path, monkeypatch, next_action
):
    from egocua.controlled_experiments import online_arm
    from egocua.study_execution import EvaluationSettings
    from egocua.synthesis_workflow import SynthesisWorkflow
    from tools import run_task_qualification as runner

    definition = task("graft")
    row = {"development_screen": {"status": "not_run"}}
    monkeypatch.setattr(
        runner,
        "build_report",
        lambda *args, **kwargs: {"tasks": {definition["id"]: row}},
    )

    def status(workflow, ids):
        assert ids == [definition["id"]]
        return {
            "tasks": {
                definition["id"]: {
                    "next_action": next_action,
                    "reason": "source evidence",
                }
            }
        }

    monkeypatch.setattr(SynthesisWorkflow, "status", status)
    monkeypatch.setattr(
        runner,
        "load_validation_study",
        lambda root, tasks: [online_arm("validation", tasks, PLAN)],
    )
    exported, executed = [], []
    monkeypatch.setattr(
        runner,
        "export_study",
        lambda arms, directory, **kwargs: exported.append((arms, directory)),
    )

    async def execute(directory, settings, root):
        executed.append(settings)

    monkeypatch.setattr(runner, "run_exported_evaluation", execute)
    monkeypatch.setattr(runner, "read_study", lambda directory: {"execution_results": []})

    def run_batch():
        return asyncio.run(
            runner._run_batch(root=tmp_path, tasks=[definition], task_id=definition["id"])
        )

    if next_action == "resolve_source_review":
        with pytest.raises(ValueError, match="resolve_source_review"):
            run_batch()
        assert exported == executed == []
        return
    result = run_batch()
    ((arms, directory),) = exported
    assert [arm.name for arm in arms] == ["validation"]
    assert directory.parent == tmp_path / "experiments"
    assert executed == [
        EvaluationSettings(concurrency=1, run_purpose="task_validation")
    ]
    assert result["status"] == "batch_finished"


@pytest.mark.parametrize("has_action", [True, False])
def test_calibration_report_accepts_saved_run_inputs_without_legacy_hashes(
    tmp_path, has_action
):
    from egocua.execution import RunConfig, run_experiment
    from tools.audit.report_difficulty_calibration import build_report

    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    payload = json.loads(PLAN_PATH.read_text())
    payload["default_release_step"] = 2
    payload["runtime"]["release_trigger"] = "timeline_step"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload))
    plan = load_experiment_plan(path)
    parameters = plan.run_parameters(definition, expand_runs([definition], plan)[0])
    config = RunConfig(
        agent="browsergym",
        transport="browsergym-openai-vision-v1",
        asset_root=tmp_path,
        output_root=tmp_path / "experiments",
        **parameters,
    )

    def execute(context):
        for observation in context.initial_payload["observations"]:
            context.observation_images(observation)
        context.advance()
        context.advance()
        actions = [{"wait": {"seconds": 1}}] if has_action else []
        return {"history": [{"model_output": {"action": actions}}]}

    record = run_experiment(definition, plan.condition("online"), config, execute)
    assert "task_sha256" not in record["meta"]
    report = build_report(tmp_path, path)
    assert report["summary"]["valid_runs"] == int(has_action)
    assert report["summary"]["rejected_run_artifacts"] == {}
    if not has_action:
        replicate = report["tasks"][definition["id"]]["replicates"]["0"]
        assert replicate["status"] == "invalid"
        assert replicate["run_invalid_reasons"] == ["missing_model_action"]
        assert (tmp_path / replicate["run_artifact"] / "run.json").is_file()
        attribution = report["failure_attributions"][0]
        assert attribution["run_verdict"] == "invalid_run"
        assert attribution["checker_version"]
    frame = definition["visual_packets"][0]["frame_paths"][0]
    (tmp_path / frame).write_bytes(b"changed source")
    report = build_report(tmp_path, path)
    assert report["summary"]["valid_runs"] == 0
    assert report["summary"]["rejected_run_artifacts"] == {"changed_visual_inputs": 1}
