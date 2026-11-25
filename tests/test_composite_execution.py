from dataclasses import replace
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pytest

from egocua.checker import check_run
from egocua.execution import RunConfig, run_experiment
from egocua.schema import ValidationError, load_tasks, validate_task
from egocua.workspace import write_json_atomic
from tests.helpers import item_id, materialize_site, products, site_data
from tests.test_composite_timeline import ONLINE, composite


def task_definition(mechanisms: str) -> dict:
    definition = composite(mechanisms)
    protected = item_id("graft", 0)
    final = item_id("graft", 1)
    pool = definition["visual_session"]["frequency_samples"]
    pool["sources"] = [{"source_video": "source.mp4", "window_seconds": [0, 31]}]
    definition.update(
        id="test-composite",
        schema_version=2,
        site="egomart",
        source={"window_seconds": [0, 31]},
        visual_packets=[{"id": "V0", "frame_paths": pool["frame_paths"][:8]}],
        goal={"target_granularity": "category", "acceptable_states": [
            {"selected_entity_ids": [protected, final], "fields": {}},
        ]},
    )
    if "C" in mechanisms:
        definition["clarification"].update(
            candidate_ids=[protected, item_id("graft", 3)], valid_slot="position"
        )
    return definition


def cart_action(context, action, item):
    with urlopen(
        f"{context.base_url}/cart/{action}",
        data=urlencode({"item_id": item}).encode(),
        timeout=3,
    ) as response:
        assert response.status == 200


@pytest.mark.parametrize("final_indices", [(0, 3), (1, 2)])
def test_composite_clarification_candidates_are_not_final_role_constraints(final_indices):
    definition = task_definition("CG")
    definition["goal"]["acceptable_states"] = [{
        "selected_entity_ids": [item_id("graft", i) for i in final_indices],
        "fields": {},
    }]

    validate_task(definition, site_data("graft"))


@pytest.mark.parametrize("mechanisms", ["RG", "CG", "CR", "RCG"])
def test_composite_load_and_validate_as_experimental(tmp_path, mechanisms):
    definition = task_definition(mechanisms)
    write_json_atomic(tmp_path / "composite/test-composite.json", definition)
    with pytest.raises(ValidationError, match="no task JSON"):
        load_tasks(tmp_path)
    assert load_tasks(tmp_path, experimental=True) == [definition]
    validate_task(definition, site_data("graft"))


@pytest.mark.parametrize("upfront", [None, "refresh", "graft", "both"])
def test_composite_uses_real_cart_and_common_terminal_goal(tmp_path, upfront):
    definition = task_definition("RG")
    materialize_site(products("graft"), tmp_path)
    old, keep, final = (item_id("graft", i) for i in (2, 0, 1))
    condition = replace(ONLINE, composite_upfront=upfront)
    config = RunConfig("scripted", "fixture", 4, None, max_steps=20, asset_root=tmp_path)

    def perform(context):
        context.advance()
        cart_action(context, "add", keep)
        cart_action(context, "add", old)
        context.advance()
        context.advance()
        cart_action(context, "remove", old)
        cart_action(context, "add", final)
        return []

    record = run_experiment(definition, condition, config, perform)
    result = check_run(definition, record)
    assert result["run_valid"], result["run_invalid_reasons"]
    assert result["e2e_success"]
    assert record["live_state"]["selected_entity_ids"] == [keep, final]


def test_waking_for_graft_preserves_existing_cart_and_scores_correctly(tmp_path):
    definition = task_definition("RG")
    materialize_site(products("graft"), tmp_path)
    old, keep, final = (item_id("graft", i) for i in (2, 0, 1))
    config = RunConfig("scripted", "fixture", 4, None, max_steps=20, asset_root=tmp_path)

    def perform(context):
        context.advance()
        cart_action(context, "add", keep)
        cart_action(context, "add", old)
        assert context.standby()
        wake = context.advance()
        assert {event["type"] for event in wake} == {"instruction_injection", "visual_context_update"}
        assert {event["source_step"] for event in wake} == {3}
        cart_action(context, "remove", old)
        cart_action(context, "add", final)
        return []

    record = run_experiment(definition, ONLINE, config, perform)
    result = check_run(definition, record)
    assert result["run_valid"], result["run_invalid_reasons"]
    assert result["e2e_success"]
    assert record["live_state"]["selected_entity_ids"] == [keep, final]


def test_composite_answer_is_not_counted_as_over_asking(tmp_path: Path):
    definition = task_definition("RCG")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig("scripted", "fixture", 4, None, max_steps=20, asset_root=tmp_path)

    def perform(context):
        context.advance()
        assert context.ask_user("Which one?")["valid"]
        context.advance()
        context.advance()
        return []

    result = check_run(definition, run_experiment(definition, ONLINE, config, perform))
    assert result["run_valid"], result["run_invalid_reasons"]
    assert not result["e2e_success"]
    assert result["answer_delivered"] is True
    assert result["over_ask"] is None


def test_composite_facts_do_not_count_search_as_selection_or_offer_as_answer(tmp_path):
    from egocua.study_summary import _composite_run_facts

    definition = task_definition("RCG")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig("scripted", "fixture", 4, None, max_steps=30, asset_root=tmp_path)

    def perform(context):
        context.advance()
        context._state.log("search", query="tool")
        context.advance()
        context.advance()
        context.ask_user("Which one?")
        return []

    record = run_experiment(definition, ONLINE, config, perform)
    checked = check_run(definition, record)
    facts = _composite_run_facts(definition, record, checked)
    assert facts["update_count"] == 2
    assert facts["clarification_offered"] is True
    assert facts["clarification_received"] is False
    assert facts["events"]["R"]["selected_before_update"] is False
    assert facts["events"]["G"]["selected_before_update"] is False


def test_composite_facts_preserve_repeated_event_delivery():
    from egocua.protocol import RunSession
    from egocua.study_summary import _composite_run_facts
    from tests.test_composite_timeline import repeated_composite
    from tests.helpers import complete_run_record

    definition = repeated_composite()
    session = RunSession(definition, ONLINE, 4, None)
    session.advance(1)
    session.ask_user("Which one?", 1)
    session.standby(1)
    session.advance(2)
    session.standby(2)
    session.advance(3)
    record = complete_run_record({
        "task_id": definition["id"], "track": definition["track"],
        "condition": ONLINE.as_dict(), "actions": session.events, "steps": 3,
    })
    facts = _composite_run_facts(definition, record, check_run(definition, record))
    assert facts["update_count"] == 4
    assert facts["combination"] == "CRG"
    assert facts["delivery"] == {"R": False, "G": True, "C": True}
    assert facts["events"]["R1"]["status"] == "skipped_on_standby"
    assert facts["events"]["R2"]["selected_before_update"] is None
    assert facts["events"]["G2"]["step"] == 3
