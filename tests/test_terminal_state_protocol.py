"""Terminal scoring, native state history, and non-destructive migration."""

import asyncio
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from egocua.checker import EVALUATOR_VERSION, check_run, rescore_run
from egocua.execution import RunConfig, run_experiment, run_experiment_async
from egocua.final_state import evaluate_final_state
from egocua.schema import goal_for_condition
from egocua.timeline import TimelineError
from sites.runtime import SiteState
from tests.conditions import CONDITIONS
from tests.helpers import complete_run_record, materialize_site, products, task
from tools.audit.rescore_terminal_state import rescore_experiment


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_every_ablation_preserves_the_final_user_goal(track):
    definition = task(track)
    for condition in CONDITIONS.values():
        assert goal_for_condition(definition, condition) == definition["goal"]


def test_checker_uses_historical_goal_for_outcome_and_diagnostics():
    historical = task("refresh")
    current = deepcopy(historical)
    current["goal"]["acceptable_states"][0]["selected_entity_ids"] = ["new-answer"]
    run = complete_run_record(
        {
            "task_id": historical["id"],
            "track": historical["track"],
            "condition": CONDITIONS["upfront"].as_dict(),
            "task_snapshot": historical,
            "meta": {"default_release_step": 2, "reference_steps": None},
            "actions": [],
            "steps": 1,
            "live_state": {
                "selected_entity_ids": historical["goal"]["acceptable_states"][0][
                    "selected_entity_ids"
                ],
                "fields": {},
            },
        }
    )
    checked = check_run(current, run)
    assert checked["e2e_success"] is True
    assert checked["goal_state_correct"] is True
    assert (
        checked["evaluator"]["acceptable_states"]
        == historical["goal"]["acceptable_states"]
    )
    assert checked["missing_after_update"] is False
    original = deepcopy(run)
    assert rescore_run(current, run)["e2e_success"] is False
    assert check_run(current, run)["e2e_success"] is True
    assert run == original


@pytest.mark.parametrize("snapshot", [None, {"id": "different-task"}])
def test_checker_rejects_invalid_historical_snapshot(snapshot):
    definition = task("graft")
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": definition["track"],
            "condition": CONDITIONS["upfront"].as_dict(),
            "task_snapshot": snapshot,
            "meta": {"default_release_step": 2, "reference_steps": None},
            "actions": [],
            "steps": 1,
        }
    )
    with pytest.raises(TimelineError, match="snapshot does not belong"):
        check_run(definition, run)


@pytest.mark.parametrize("trigger", ["unregistered", "", 1])
def test_checker_rejects_unregistered_release_trigger(trigger):
    definition = task("graft")
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": definition["track"],
            "condition": CONDITIONS["upfront"].as_dict(),
            "meta": {"runtime": {"release_trigger": trigger}},
        }
    )
    with pytest.raises(TimelineError, match="unknown release_trigger"):
        check_run(definition, run)


@pytest.mark.parametrize(
    "selected,fields,success",
    [
        (["a", "a", "b"], {"mode": "walking"}, True),
        (["b", "a", "a"], {"mode": "walking"}, True),
        (["a", "b"], {"mode": "walking"}, False),
        (["a", "a", "b", "old"], {"mode": "walking"}, False),
        (["a", "a", "b"], {}, False),
        (["a", "a", "b"], {"mode": "driving"}, False),
        (["alternative"], {}, True),
    ],
)
def test_outcomes_check_quantities_fields_and_verified_alternatives(
    selected, fields, success
):
    goal = {
        "acceptable_states": [
            {"selected_entity_ids": ["a", "a", "b"], "fields": {"mode": "walking"}},
            {"selected_entity_ids": ["alternative"], "fields": {}},
        ]
    }
    assert (
        evaluate_final_state(goal, {"selected_entity_ids": selected, "fields": fields})
        is success
    )


def test_state_changes_record_before_and_after_without_rewriting_prior_evidence():
    state = SiteState({"entities": []})
    state.save("old")
    state.commit()
    state.deselect("old")
    state.save("new")
    state.set_field("mode", "walking")
    changes = [
        action
        for action in state.actions
        if action["state_before"] != action["state_after"]
    ]
    assert len(changes) == 4
    assert changes[0]["state_before"] == {"selected_entity_ids": [], "fields": {}}
    for previous, current in zip(changes[:-1], changes[1:], strict=True):
        assert previous["state_after"] == current["state_before"]
    assert changes[-1]["state_after"] == state.live_state()
    assert state.final_state()["selected_entity_ids"] == ["old"]
    assert changes[0]["state_after"] == {"selected_entity_ids": ["old"], "fields": {}}


def test_unchanged_quantity_and_search_keep_explicit_state_evidence():
    state = SiteState({"entities": []}, selected_entity_ids=["old"])
    before = state.live_state()
    state.set_quantity("old", 1)
    state.log("search", query="replacement")
    assert len(state.actions) == 2
    for action in state.actions:
        assert action["state_before"] == before
        assert action["state_after"] == before
    state.deselect("old")
    assert state.actions[0]["state_after"] == before


def test_withdrawing_all_selections_accepts_an_empty_website_state():
    goal = {"acceptable_states": [{"selected_entity_ids": [], "fields": {}}]}
    assert evaluate_final_state(goal, {"selected_entity_ids": [], "fields": {}})
    assert not evaluate_final_state(
        goal, {"selected_entity_ids": ["old"], "fields": {}}
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_step_limit_reads_terminal_state_and_remains_valid(tmp_path, asynchronous):
    definition = task("clarify")
    materialize_site(products("clarify"), tmp_path)
    config = RunConfig(
        agent="fixture",
        model="fixture",
        default_release_step=2,
        reference_steps=None,
        max_steps=1,
        asset_root=tmp_path,
        output_root=tmp_path / "artifacts",
    )

    def perform(context):
        context.advance()
        expected = definition["goal"]["acceptable_states"][0]
        for entity in expected["selected_entity_ids"]:
            context._state.save(entity)
        for key, value in expected["fields"].items():
            context._state.set_field(key, value)
        context.advance()

    async def async_perform(context):
        return perform(context)

    if asynchronous:
        run = asyncio.run(
            run_experiment_async(
                definition, CONDITIONS["no_ask"], config, async_perform
            )
        )
    else:
        run = run_experiment(definition, CONDITIONS["no_ask"], config, perform)
    checked = check_run(definition, run)
    assert run["meta"]["termination_reason"] == "step_limit"
    assert checked["run_valid"] is True
    assert checked["e2e_success"] is True
    assert checked["asked"] is False


def test_offline_rescore_uses_historical_snapshot_without_altering_run(tmp_path):
    definition = task("graft")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="fixture",
        model="fixture",
        default_release_step=2,
        reference_steps=None,
        asset_root=tmp_path,
        output_root=tmp_path / "artifacts",
    )

    def adapter(context):
        context.advance()
        expected = definition["goal"]["acceptable_states"][0]
        for entity in expected["selected_entity_ids"]:
            context._state.save(entity)
        return []

    run = run_experiment(definition, CONDITIONS["upfront"], config, adapter)
    directory = Path(run["meta"]["artifact_directory"])
    original = {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }
    summary = {
        "tasks": {
            definition["id"]: {
                "artifact": str(directory),
                "checked": {"run_id": run["meta"]["run_id"]},
            }
        }
    }
    result = rescore_experiment(summary, tmp_path)
    assert result["summary"]["successes"] == 1
    assert result["evaluator_version"] == EVALUATOR_VERSION
    assert original == {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    }

    # Explicitly model an old archive that never recorded terminal state.
    legacy = deepcopy(run)
    legacy.pop("live_state")
    with patch(
        "tools.audit.rescore_terminal_state.load_run_record", return_value=legacy
    ):
        unavailable = rescore_experiment(summary, tmp_path)
    assert unavailable["summary"]["success_rate"] is None
    assert unavailable["tasks"][definition["id"]]["reasons"] == [
        "missing_terminal_live_state"
    ]
