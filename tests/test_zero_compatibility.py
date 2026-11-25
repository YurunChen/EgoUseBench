from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from egocua.checker import check_run
from egocua.experiment import ExperimentPlanError, load_experiment_plan
from egocua.run_ledger import (
    RunLedger,
    inspect_run_artifact,
    load_run_record,
)
from egocua.timeline import Condition, Timeline, TimelineError
from tests.helpers import complete_run_meta, complete_run_record, item_id, task


ONLINE = {
    "name": "online",
    "schedule": "online",
    "scope": "full",
    "ask": "allowed",
    "release_at": None,
}


def test_experiment_plan_rejects_schema_2(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema_version": 2}))

    with pytest.raises(ExperimentPlanError, match="schema_version must be 3"):
        load_experiment_plan(plan_path)


def test_experiment_plan_rejects_obsolete_fields(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/pilot.json").read_text())
    payload["arms"] = {"graft": ["stream"]}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ExperimentPlanError, match="unknown fields.*arms"):
        load_experiment_plan(plan_path)


def test_experiment_plan_requires_a_reference_condition(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/pilot.json").read_text())
    del payload["reference_condition"]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ExperimentPlanError, match="reference_condition is required"):
        load_experiment_plan(plan_path)


@pytest.mark.parametrize("field", ["default_release_step", "reference_steps"])
def test_experiment_plan_requires_explicit_timing_fields(
    tmp_path: Path,
    field: str,
) -> None:
    payload = json.loads(Path("configs/pilot.json").read_text())
    del payload[field]
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ExperimentPlanError, match=f"{field} is required"):
        load_experiment_plan(plan_path)


def test_experiment_plan_rejects_unknown_runtime_fields(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/pilot.json").read_text())
    payload["runtime"]["arm"] = "stream"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))

    with pytest.raises(ExperimentPlanError, match="runtime fields must contain"):
        load_experiment_plan(plan_path)


def test_reference_condition_name_is_owned_by_the_plan(tmp_path: Path) -> None:
    payload = json.loads(Path("configs/pilot.json").read_text())
    payload["reference_condition"] = "primary"
    for condition in payload["conditions"]:
        if condition["name"] == "online":
            condition["name"] = "primary"
    for names in payload["tracks"].values():
        names[names.index("online")] = "primary"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload))

    plan = load_experiment_plan(plan_path)

    assert plan.reference_condition_for("graft").name == "primary"


def test_timeline_rejects_string_conditions() -> None:
    with pytest.raises(TimelineError, match="Condition object"):
        Timeline.compile(
            task("refresh"),
            cast(Condition, "stream"),
            default_release_step=2,
            reference_steps=None,
        )


def test_checker_rejects_arm_only_run_records() -> None:
    record = {
        "arm": "stream",
        "actions": [],
        "final_state": {
            "selected_entity_ids": [item_id("refresh", 1)],
            "fields": {},
            "submission_completed": True,
        },
    }

    with pytest.raises(TimelineError, match="unknown=.*arm"):
        check_run(task("refresh"), record)


def test_checker_rejects_arm_even_with_a_condition() -> None:
    with pytest.raises(TimelineError, match="unknown=.*arm"):
        check_run(task("refresh"), {"condition": ONLINE, "arm": "stream"})


def test_checker_rejects_obsolete_nested_metadata() -> None:
    with pytest.raises(TimelineError, match="meta fields.*arm"):
        check_run(
            task("refresh"),
            complete_run_record(
                {
                    "condition": ONLINE,
                    "meta": {"arm": "stream"},
                }
            ),
        )


def test_checker_rejects_missing_timeline_metadata() -> None:
    record = complete_run_record({"condition": ONLINE})
    del record["meta"]["default_release_step"]
    with pytest.raises(TimelineError, match="default_release_step"):
        check_run(task("refresh"), record)


def test_checker_rejects_partial_run_records() -> None:
    record = complete_run_record({"condition": ONLINE})
    del record["track"]

    with pytest.raises(TimelineError, match="missing=.*track"):
        check_run(task("refresh"), record)


def test_inspector_rejects_old_run_artifacts(tmp_path: Path) -> None:
    artifact = tmp_path / "run"
    artifact.mkdir()
    (artifact / "manifest.json").write_text(json.dumps({"schema_version": 3}))

    with pytest.raises(ValueError, match="schema_version must be 4"):
        inspect_run_artifact(artifact)


def test_run_artifact_loader_rejects_direct_record_files(tmp_path: Path) -> None:
    condition = Condition.from_dict(ONLINE)
    ledger = RunLedger.start(
        tmp_path,
        experiment_id="pilot",
        task_id="refresh-01",
        condition=condition,
        replicate_id=0,
        agent="test",
    )
    ledger.complete(
        complete_run_record(
            {
                "task_id": "refresh-01",
                "track": "refresh",
                "condition": ONLINE,
                "run_status": {"valid": True, "reason": None},
                "initial_state": {
                    "selected_entity_ids": [],
                    "fields": {},
                    "submission_completed": False,
                },
                "steps": 0,
                "actions": [],
                "final_state": {
                    "selected_entity_ids": [],
                    "fields": {},
                    "submission_completed": False,
                },
                "trace": [],
                "meta": complete_run_meta(
                    replicate_id=0,
                    agent="test",
                    run_id=ledger.run_id,
                    artifact_directory=ledger.directory.as_posix(),
                ),
            }
        )
    )

    with pytest.raises(ValueError, match="must be a directory"):
        load_run_record(ledger.directory / "run.json")

    record_path = ledger.directory / "run.json"
    record = json.loads(record_path.read_text())
    record["arm"] = "stream"
    record_path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="run record fields disagree with schema v4"):
        inspect_run_artifact(ledger.directory)


def test_run_artifact_rejects_obsolete_manifest_fields(tmp_path: Path) -> None:
    condition = Condition.from_dict(ONLINE)
    ledger = RunLedger.start(
        tmp_path,
        experiment_id="pilot",
        task_id="refresh-01",
        condition=condition,
        replicate_id=0,
        agent="test",
    )
    path = ledger.directory / "run.json"
    record = json.loads(path.read_text())
    record["archive"]["arm"] = "stream"
    path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="archive has missing or unsupported fields"):
        inspect_run_artifact(ledger.directory)


def test_run_artifact_rejects_empty_record_files(tmp_path: Path) -> None:
    ledger = RunLedger.start(
        tmp_path,
        experiment_id="pilot",
        task_id="refresh-01",
        condition=Condition.from_dict(ONLINE),
        replicate_id=0,
        agent="test",
    )
    (ledger.directory / "run.json").write_text("{}")

    with pytest.raises(ValueError, match="no archive metadata"):
        load_run_record(ledger.directory)


def test_run_artifact_rejects_incomplete_metadata(tmp_path: Path) -> None:
    ledger = RunLedger.start(
        tmp_path,
        experiment_id="pilot",
        task_id="refresh-01",
        condition=Condition.from_dict(ONLINE),
        replicate_id=0,
        agent="test",
    )
    record = complete_run_record(
        {
            "task_id": "refresh-01",
            "track": "refresh",
            "condition": ONLINE,
            "meta": complete_run_meta(
                replicate_id=0,
                agent="test",
                run_id=ledger.run_id,
                artifact_directory=ledger.directory.as_posix(),
            ),
        }
    )
    del record["meta"]["model"]

    with pytest.raises(ValueError, match="missing=.*model"):
        ledger.complete(record)
