"""Registered initial carts are real website state, not model actions."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from urllib.request import urlopen

import pytest

from egocua.checker import check_run
from egocua.execution import RunConfig, run_experiment_async
from egocua.run_materials import execution_inputs, material_mismatch
from tests.conditions import CONDITIONS
from tests.helpers import (
    PLAN,
    item_id,
    materialize_assets,
    materialize_site,
    products,
    task,
)


def test_registered_cart_is_visible_before_first_action_and_update(tmp_path):
    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    selected = item_id("graft", 1)
    state = {"selected_entity_ids": [selected], "fields": {}}
    config = RunConfig(
        agent="test",
        model="test",
        asset_root=tmp_path,
        default_release_step=1,
        reference_steps=None,
        initial_site_state=state,
    )

    async def execute(context):
        assert context._state.live_state() == state
        assert not context._trace.actions
        assert context.initial_payload["initial_site_state"] == state
        with urlopen(f"{context.base_url}/cart") as response:
            page = response.read().decode()
        entity = next(
            row for row in context.site_data["entities"] if row["entity_id"] == selected
        )
        assert entity["name"] in page
        events = context.advance()
        assert [event["payload"] for event in events] == [
            definition["instructions"]["b"]
        ]
        assert not any(
            action["type"] == "select_entity" for action in context._trace.actions
        )
        context._state.deselect(selected)
        context._state.select(item_id("graft", 0))
        return []

    record = asyncio.run(
        run_experiment_async(definition, CONDITIONS["online"], config, execute)
    )
    assert record["initial_state"] == {**state, "submission_completed": False}
    assert record["meta"]["runtime"]["initial_site_state"] == state
    mutation = next(
        action for action in record["actions"] if action["type"] == "deselect_entity"
    )
    assert mutation["state_before"] == state
    assert check_run(definition, record)["e2e_success"] is True
    changed = deepcopy(record)
    changed["initial_context"]["initial_site_state"]["selected_entity_ids"] = []
    assert (
        "invalid_initial_context"
        in check_run(definition, changed)["run_invalid_reasons"]
    )
    changed = deepcopy(record)
    changed["initial_state"]["selected_entity_ids"] = []
    assert (
        "invalid_initial_site_state"
        in check_run(definition, changed)["run_invalid_reasons"]
    )
    changed = deepcopy(record)
    changed["initial_context"]["initial_cart_contents"] = ["fake target hint"]
    assert (
        "invalid_initial_context"
        in check_run(definition, changed)["run_invalid_reasons"]
    )
    assert state == {"selected_entity_ids": [selected], "fields": {}}


@pytest.mark.parametrize(
    "state",
    [
        {"selected_entity_ids": ["not-public"], "fields": {}},
        {"selected_entity_ids": [], "fields": {}},
        {"selected_entity_ids": [item_id("graft", 0)] * 2, "fields": {}},
        {"selected_entity_ids": [item_id("graft", 0)], "fields": {"unknown": "yes"}},
    ],
)
def test_invalid_initial_cart_fails_before_adapter(tmp_path, state):
    definition = task("graft")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="test",
        model="test",
        asset_root=tmp_path,
        default_release_step=1,
        reference_steps=None,
        initial_site_state=state,
    )

    async def execute(_context):
        pytest.fail("invalid initial state must not reach the adapter")

    with pytest.raises(ValueError, match="initial_site_state"):
        asyncio.run(
            run_experiment_async(definition, CONDITIONS["online"], config, execute)
        )


def test_default_run_starts_empty(tmp_path):
    definition = task("graft")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="test",
        model="test",
        asset_root=tmp_path,
        default_release_step=1,
        reference_steps=None,
    )

    async def execute(context):
        assert context._state.live_state() == {"selected_entity_ids": [], "fields": {}}
        assert "initial_site_state" not in context.initial_payload
        return []

    record = asyncio.run(
        run_experiment_async(definition, CONDITIONS["online"], config, execute)
    )
    assert record["initial_state"]["selected_entity_ids"] == []


def test_initial_cart_changes_registered_inputs_and_prevents_empty_run_reuse(tmp_path):
    definition = task("graft")
    state = {"selected_entity_ids": [item_id("graft", 0)], "fields": {}}
    plan = replace(PLAN, initial_site_states={definition["id"]: state})
    assert (
        plan.run_parameters(
            definition,
            {
                "condition": "online",
                "replicate_id": 0,
                "provider_seed": 1,
                "schedule_index": 0,
            },
        )["initial_site_state"]
        == state
    )
    assert execution_inputs(definition, plan, "online") != execution_inputs(
        definition, PLAN, "online"
    )
    record = {
        "task_id": definition["id"],
        "task_snapshot": deepcopy(definition),
        "meta": {"runtime": {}},
    }
    assert (
        material_mismatch(definition, record, plan, root=tmp_path)
        == "changed_initial_site_state"
    )




def test_plan_loads_registered_initial_cart(tmp_path):
    import json

    from egocua.experiment import load_experiment_plan
    from tests.helpers import plan_payload

    definition = task("graft")
    state = {"selected_entity_ids": [item_id("graft", 0)], "fields": {}}
    payload = plan_payload()
    payload["runtime"]["initial_site_states"] = {definition["id"]: state}
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(payload))
    plan = load_experiment_plan(source)
    assert plan.initial_site_states == {definition["id"]: state}
