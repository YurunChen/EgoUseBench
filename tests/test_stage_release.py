from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from egocua.checker import check_run
from egocua.experiment import load_experiment_plan, ExperimentPlanError
from egocua.execution import RunConfig, RunContext, run_experiment
from egocua.protocol import first_release_trigger
from tests.conditions import CONDITIONS
from tests.helpers import item_id, materialize_site, products, task


def _post(context: RunContext, path: str, entity: str) -> None:
    request = Request(context.base_url + path, urlencode({"item_id": entity}).encode())
    urlopen(request).read()


def _config(
    root: Path,
    trigger: str,
    *,
    max_steps: int = 40,
    pre_release_max_steps: int | None = None,
    post_release_max_steps: int | None = None,
) -> RunConfig:
    return RunConfig(
        agent="local-fixture",
        model="none",
        default_release_step=1,
        reference_steps=None,
        asset_root=root,
        release_trigger=trigger,
        max_steps=max_steps,
        pre_release_max_steps=pre_release_max_steps,
        post_release_max_steps=post_release_max_steps,
    )


@pytest.mark.parametrize("trigger", ["first_search", "first_detail", "first_selection"])
def test_native_stage_is_delivered_next_step_once(tmp_path: Path, trigger: str) -> None:
    materialize_site(products("graft"), tmp_path)

    def adapter(context: RunContext) -> list:
        assert context.advance() == []
        if trigger == "first_search":
            urlopen(context.base_url + "/search?q=Product").read()
        elif trigger == "first_detail":
            urlopen(context.base_url + "/product/" + item_id("graft", 0)).read()
        else:
            _post(context, "/cart/add", item_id("graft", 0))
        events = context.advance()
        assert len(events) == 1
        assert events[0]["trigger_reason"] == trigger
        assert events[0]["trigger_action_index"] == 0
        assert events[0]["step"] == 2
        assert context.advance() == []
        return []

    record = run_experiment(
        task("graft"), CONDITIONS["online"], _config(tmp_path, trigger), adapter
    )
    assert record["meta"]["stage_release"]["status"] == "delivered"
    assert record["meta"]["event_delivery_steps"] == [2]
    assert check_run(task("graft"), record)["run_valid"]


def test_empty_and_zero_result_search_do_not_release(tmp_path: Path) -> None:
    materialize_site(products("graft"), tmp_path)

    def adapter(context: RunContext) -> list:
        context.advance()
        urlopen(context.base_url + "/search?q=").read()
        assert context.advance() == []
        urlopen(context.base_url + "/search?q=nonexistentxyzzzzz").read()
        assert context.advance() == []
        return []

    record = run_experiment(
        task("graft"), CONDITIONS["online"], _config(tmp_path, "first_search"), adapter
    )
    assert record["meta"]["stage_release"]["status"] == "stage_not_reached"
    assert check_run(task("graft"), record)["run_valid"]


def test_missing_detail_does_not_release(tmp_path: Path) -> None:
    materialize_site(products("graft"), tmp_path)

    def adapter(context: RunContext) -> list:
        context.advance()
        with pytest.raises(HTTPError):
            urlopen(context.base_url + "/product/nonexistent").read()
        assert context.advance() == []
        return []

    record = run_experiment(
        task("graft"), CONDITIONS["online"], _config(tmp_path, "first_detail"), adapter
    )
    assert record["meta"]["stage_release"]["status"] == "stage_not_reached"


@pytest.mark.parametrize(
    "action",
    [
        {"type": "select_entity", "entity_id": "x"},
        {
            "type": "select_entity",
            "state_before": {"selected_entity_ids": ["x"]},
            "state_after": {"selected_entity_ids": ["x", "x"]},
        },
        {
            "type": "deselect_entity",
            "state_before": {"selected_entity_ids": ["x"]},
            "state_after": {"selected_entity_ids": []},
        },
    ],
)
def test_selection_requires_new_entity_state_evidence(action: dict) -> None:
    assert first_release_trigger([action], "first_selection") is None


def test_two_selections_waits_for_two_distinct_entities(tmp_path: Path) -> None:
    materialize_site(products("graft"), tmp_path)

    def adapter(context: RunContext) -> list:
        context.advance()
        _post(context, "/cart/add", item_id("graft", 0))
        assert context.advance() == []
        _post(context, "/cart/add", item_id("graft", 1))
        events = context.advance()
        assert len(events) == 1
        assert events[0]["trigger_reason"] == "two_selections"
        assert context.advance() == []
        return []

    record = run_experiment(
        task("graft"),
        CONDITIONS["online"],
        _config(tmp_path, "two_selections"),
        adapter,
    )
    assert record["meta"]["event_delivery_steps"] == [3]
    assert check_run(task("graft"), record)["run_valid"]


def test_two_units_of_one_product_do_not_trigger_two_selections():
    action = {
        "type": "select_entity",
        "state_before": {"selected_entity_ids": ["x"]},
        "state_after": {"selected_entity_ids": ["x", "x"]},
    }
    assert first_release_trigger([action], "two_selections") is None


@pytest.mark.parametrize("phase", ["not_reached", "undelivered", "exhausted"])
def test_stage_budget_records_distinct_outcomes(tmp_path: Path, phase: str) -> None:
    materialize_site(products("graft"), tmp_path)

    def adapter(context: RunContext) -> list:
        context.advance()
        if phase == "not_reached":
            context.advance()
            context.advance()
        urlopen(context.base_url + "/search?q=Product").read()
        if phase == "undelivered":
            return []
        assert context.advance()
        context.advance()
        context.advance()
        return []

    config = _config(
        tmp_path,
        "first_search",
        max_steps=4,
        pre_release_max_steps=2,
        post_release_max_steps=2,
    )
    record = run_experiment(task("graft"), CONDITIONS["online"], config, adapter)
    expected = {
        "not_reached": "stage_not_reached",
        "undelivered": "trigger_not_delivered",
        "exhausted": "post_release_budget_exhausted",
    }
    assert record["meta"]["stage_release"]["status"] == expected[phase]
    assert (
        record["steps"] == {"not_reached": 2, "undelivered": 1, "exhausted": 3}[phase]
    )
    assert check_run(task("graft"), record)["run_valid"]


def test_trigger_at_pre_budget_boundary_still_delivers(tmp_path: Path) -> None:
    materialize_site(products("graft"), tmp_path)

    def adapter(context: RunContext) -> list:
        context.advance()
        context.advance()
        urlopen(context.base_url + "/search?q=Product").read()
        assert context.advance()[0]["step"] == 3
        context.advance()
        return []

    config = _config(
        tmp_path,
        "first_search",
        max_steps=4,
        pre_release_max_steps=2,
        post_release_max_steps=2,
    )
    record = run_experiment(task("graft"), CONDITIONS["online"], config, adapter)
    assert record["meta"]["stage_release"]["post_release_steps"] == 2
    malformed = deepcopy(record)
    malformed["actions"][-1]["trigger_action_index"] = 99
    assert not check_run(task("graft"), malformed)["run_valid"]


def test_refresh_sparse_inputs_are_recorded_per_packet(tmp_path: Path) -> None:
    materialize_site(products("refresh"), tmp_path)

    def adapter(context: RunContext) -> list:
        context.advance()
        urlopen(context.base_url + "/search?q=Product").read()
        context.advance()
        return []

    record = run_experiment(
        task("refresh"),
        CONDITIONS["first_frame"],
        _config(tmp_path, "first_search"),
        adapter,
    )
    packets = record["meta"]["released_visual_inputs"]
    assert len(packets) == 2
    assert all(len(packet["frame_paths"]) == 1 for packet in packets)
    assert [packet["step"] for packet in packets] == [0, 2]


@pytest.mark.parametrize(
    "options",
    [
        {"pre_release_max_steps": 20},
        {"pre_release_max_steps": 20, "post_release_max_steps": 10},
        {"pre_release_max_steps": True, "post_release_max_steps": 39},
    ],
)
def test_stage_budget_configuration_fails_early(tmp_path: Path, options: dict) -> None:
    with pytest.raises(ValueError):
        _config(tmp_path, "first_search", **options)


def test_upfront_cannot_silently_become_stage_experiment(tmp_path: Path) -> None:
    materialize_site(products("graft"), tmp_path)
    with pytest.raises(ValueError, match="pending update"):
        run_experiment(
            task("graft"),
            CONDITIONS["upfront"],
            _config(tmp_path, "first_search"),
            lambda _: [],
        )




def test_stage_plan_parses_and_rejects_partial_budgets(tmp_path: Path) -> None:
    runtime = {
        "adapter": "browsergym",
        "model": "none",
        "model_version": "none",
        "max_steps": 40,
        "release_trigger": "first_selection",
        "pre_release_max_steps": 20,
        "post_release_max_steps": 20,
    }
    payload = {
        "schema_version": 3,
        "experiment_id": "stage-test",
        "default_release_step": 2,
        "reference_steps": None,
        "conditions": [CONDITIONS["online"].as_dict()],
        "tracks": {"graft": ["online"]},
        "reference_condition": "online",
        "replicate_ids": [0],
        "scheduler_seed": 1,
        "runtime": runtime,
    }
    source = tmp_path / "plan.json"
    source.write_text(json.dumps(payload))
    plan = load_experiment_plan(source)
    assert plan.release_trigger == "first_selection"
    assert plan.pre_release_max_steps == plan.post_release_max_steps == 20
    del runtime["post_release_max_steps"]
    source.write_text(json.dumps(payload))
    with pytest.raises(ExperimentPlanError, match="both be positive"):
        load_experiment_plan(source)


def test_failed_and_repeated_native_selections_do_not_create_new_trigger(
    tmp_path: Path,
) -> None:
    materialize_site(products("graft"), tmp_path)

    def adapter(context: RunContext) -> list:
        context.advance()
        with pytest.raises(HTTPError):
            _post(context, "/cart/add", "missing")
        assert context.advance() == []
        _post(context, "/cart/add", item_id("graft", 0))
        _post(context, "/cart/add", item_id("graft", 0))
        assert context.advance()
        return []

    record = run_experiment(
        task("graft"),
        CONDITIONS["online"],
        _config(tmp_path, "first_selection"),
        adapter,
    )
    selections = [row for row in record["actions"] if row["type"] == "select_entity"]
    assert first_release_trigger(selections[1:], "first_selection") is None
    assert record["meta"]["stage_release"]["trigger_action"]["step"] == 2


