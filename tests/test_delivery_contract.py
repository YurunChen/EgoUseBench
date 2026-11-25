from copy import deepcopy
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from egocua.checker import check_run
from egocua.execution import RunConfig, run_experiment
from egocua.protocol import RELEASE_DELIVERY_DESCRIPTIONS, STAGE_RELEASE_TRIGGERS
from tests.conditions import CONDITIONS
from tests.helpers import item_id, materialize_site, products, task


def _add_item(context, entity_id):
    request = Request(
        context.base_url + "/cart/add",
        data=urlencode({"item_id": entity_id}).encode(),
        method="POST",
    )
    with urlopen(request) as response:
        response.read()


@pytest.fixture(params=["graft", "refresh", "clarify"])
def delivery_run(request, tmp_path):
    definition = task(request.param)
    materialize_site(products(request.param), tmp_path)

    def execute(trigger):
        config = RunConfig(
            agent="fixture-adapter",
            model="fixture-model",
            default_release_step=2,
            reference_steps=None,
            release_trigger=trigger,
            asset_root=tmp_path,
        )

        def adapter(context):
            context.advance()
            if request.param == "clarify":
                context.ask_user("Which variant do you mean?")
            with urlopen(context.base_url + "/search?q=Product") as response:
                response.read()
            with urlopen(
                context.base_url + "/product/" + item_id(request.param, 0)
            ) as response:
                response.read()
            _add_item(context, item_id(request.param, 0))
            _add_item(context, item_id(request.param, 1))
            context.advance()
            return []

        return run_experiment(definition, CONDITIONS["online"], config, adapter)

    return definition, execute


@pytest.mark.parametrize("trigger", RELEASE_DELIVERY_DESCRIPTIONS)
def test_real_delivery_content_is_checked_independently_of_trigger(
    delivery_run, trigger
):
    definition, execute = delivery_run
    if definition["track"] == "clarify" and trigger in STAGE_RELEASE_TRIGGERS:
        with pytest.raises(ValueError, match="pending update"):
            execute(trigger)
        return
    run = execute(trigger)
    assert check_run(definition, run)["run_valid"]
    event = next(
        action
        for action in run["actions"]
        if action["type"]
        in {"instruction_injection", "visual_context_update", "clarification_answer"}
    )
    event["payload"] = (
        {"frame_paths": ["not-delivered.jpg"]}
        if event["type"] == "visual_context_update"
        else "Not the registered instruction or answer"
    )

    checked = check_run(definition, run)

    assert checked["run_valid"] is False
    assert f"invalid_{event['type']}_payload" in checked["run_invalid_reasons"]
    assert checked["e2e_success"] is None


@pytest.mark.parametrize("field", ["instruction", "observations"])
def test_initial_context_is_checked_against_compiled_timeline(delivery_run, field):
    definition, execute = delivery_run
    run = execute("timeline_step")
    run["initial_context"][field] = "Not the registered initial input"

    checked = check_run(definition, run)

    assert checked["run_invalid_reasons"] == ["invalid_initial_context"]


def test_partial_historical_event_records_keep_existing_scoring(delivery_run):
    definition, execute = delivery_run
    run = execute("first_native_commitment")
    original = check_run(definition, run)
    historical = deepcopy(run)
    historical.pop("initial_context")
    for event in historical["actions"]:
        if event["type"] in {
            "instruction_injection",
            "visual_context_update",
            "clarification_answer",
        }:
            event.pop("payload")

    checked = check_run(definition, historical)

    assert checked["run_valid"] == original["run_valid"]
    assert checked["e2e_success"] == original["e2e_success"]
