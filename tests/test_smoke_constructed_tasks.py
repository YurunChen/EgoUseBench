from __future__ import annotations

import copy
from dataclasses import replace
import json
import sys

import pytest

from egocua.checker import check_run
from egocua.protocol import STAGE_RELEASE_TRIGGERS, first_release_trigger
from egocua.run_ledger import task_sha256, task_site_execution_sha256
from tools.audit import smoke_constructed_tasks as smoke
from tools.audit.smoke_constructed_tasks import (
    _broad_queries,
    _covering_states,
    _negative_checks,
    _requested_scope,
    _searchable_text,
)
from tests.conditions import CONDITIONS

from .helpers import (
    PLAN,
    complete_run_record,
    item_id,
    materialize_site,
    products,
    site_data,
    task,
)


def test_formal_smoke_refresh_retains_experimental_operation_evidence(tmp_path, monkeypatch):
    output = tmp_path / "smoke.json"
    composite_row = {"status": "pass", "composite_delivery": {"R": True, "G": True}}
    output.write_text(json.dumps({"composite_tasks": {"RG-01": composite_row}}))
    fresh = {
        "tasks": {"formal-task": {"status": "pass"}},
        "summary": {
            key: 0
            for key in (
                "positive_failures", "preflight_failures", "negative_failures",
                "protocol_failures",
            )
        },
    }
    monkeypatch.setattr(smoke, "run_smoke", lambda *_args, **_kwargs: fresh)
    monkeypatch.setattr(sys, "argv", ["smoke", "--output", str(output)])
    smoke.main()
    saved = json.loads(output.read_text())
    assert saved["tasks"] == fresh["tasks"]
    assert saved["composite_tasks"] == {"RG-01": composite_row}


def test_decoy_excludes_every_acceptable_option_for_current_condition() -> None:
    graft_task = task("graft")
    record = complete_run_record(
        {
            "condition": CONDITIONS["initial_only"].as_dict(),
            "meta": {"default_release_step": 2, "reference_steps": None},
            "actions": [],
            "final_state": {
                "selected_entity_ids": [item_id("graft", 0)],
                "fields": {},
                "submission_completed": True,
            },
        }
    )

    checks = _negative_checks(graft_task, record, site_data("graft")["entities"])
    decoy_check = next(check for check in checks if check["mutation"] == "decoy_only")

    assert decoy_check["evidence_status"] == "FAIL"
    assert decoy_check["e2e_success"] is False


def test_preflight_query_reaches_target_with_a_real_neighbor() -> None:
    entities = site_data("graft")["entities"]
    target = entities[0]

    query, matches = _broad_queries(
        "egomart",
        target,
        entities,
        {target["entity_id"]},
    )[0]

    assert query
    assert target["entity_id"] in matches
    assert any(entity_id != target["entity_id"] for entity_id in matches)


def test_preflight_query_ignores_capture_metadata() -> None:
    entity = {
        "name": "Eight Inch Frying Pan",
        "brand": "Example",
        "attributes": {
            "listing_source": "Amazon.ca",
            "listing_type": "Organic",
            "diameter": "8 inches",
        },
    }

    searchable = _searchable_text("egomart", entity)

    assert "organic" not in searchable
    assert "amazon.ca" not in searchable
    assert "8 inches" in searchable


def test_search_preflight_follows_visible_pagination_past_third_page(
    monkeypatch,
) -> None:
    visited = []

    def request(base_url, path):
        visited.append(path)
        page = int(path.rsplit("=", 1)[1])
        if page == 4:
            return '<a href="/video/target">Open</a>'
        return f'<a href="/search?q=video&amp;page={page + 1}">Next</a>'

    monkeypatch.setattr(smoke, "_request", request)

    pages = smoke._search_pages("http://local", "video", "/video/target")

    assert len(pages) == 4
    assert visited[-1] == "/search?q=video&page=4"


@pytest.mark.parametrize("target_present", [False, True])
def test_search_preflight_stops_at_target_or_final_page(monkeypatch, target_present):
    visited = []
    page = '<a href="/video/target">Open</a>' if target_present else "No results"

    def request(base_url, path):
        visited.append(path)
        return page

    monkeypatch.setattr(smoke, "_request", request)

    assert smoke._search_pages("http://local", "video", "/video/target") == [page]
    assert len(visited) == 1


def test_search_preflight_follows_storefront_next_link(monkeypatch):
    pages = {
        "/search?q=spoon&page=1": '<a href="/search?q=spoon&amp;page=2">Next ›</a>',
        "/search?q=spoon&page=2": '<a href="/product/target">Open</a>',
    }
    monkeypatch.setattr(smoke, "_request", lambda base_url, path: pages[path])

    assert len(smoke._search_pages("http://local", "spoon", "/product/target")) == 2


def test_incremental_smoke_selects_only_requested_site() -> None:
    tasks = [
        {"id": "mart-a", "site": "egomart"},
        {"id": "media-a", "site": "egomedia"},
    ]

    assert _requested_scope(
        tasks,
        requested_ids=set(),
        requested_sites={"egomart"},
    ) == {"mart-a"}


@pytest.fixture
def incremental_smoke(monkeypatch, tmp_path):
    tasks = [{**task("graft"), "id": "old-task"}]
    executed_ids = []

    def scripted(current_task, condition, **kwargs):
        executed_ids.append(current_task["id"])
        if kwargs.get("route_checks") is not None:
            kwargs["route_checks"].append({"search_neighbor_count": 1})
        return {
            "live_state": kwargs.get("expected_state", {}),
            "steps": 1,
            "duplicate": kwargs.get("duplicate_first", False),
        }

    monkeypatch.setattr(smoke, "load_construction_tasks", lambda _, task_ids=None, **kwargs: tasks)
    monkeypatch.setattr(smoke, "load_experiment_plan", lambda _: PLAN)
    monkeypatch.setattr(smoke, "load_site_data", lambda *_: site_data("graft"))
    monkeypatch.setattr(smoke, "_run_scripted", scripted)
    monkeypatch.setattr(
        smoke,
        "check_run",
        lambda _, record: {
            "run_valid": True,
            "evidence_status": "FAIL" if record["duplicate"] else "PASS",
            "e2e_success": not record["duplicate"],
            "run_invalid_reasons": [],
        },
    )
    monkeypatch.setattr(smoke, "_negative_checks", lambda *_: [])
    monkeypatch.setattr(smoke, "_release_protocol_checks", lambda *_: {})

    previous = smoke.run_smoke("tasks", root=tmp_path, plan_path="plan")
    tasks.append({**task("graft"), "id": "new-task"})
    executed_ids.clear()
    return tasks, previous, executed_ids


def test_incremental_smoke_adds_new_task_and_preserves_old_results(
    incremental_smoke,
    tmp_path,
) -> None:
    tasks, previous, executed_ids = incremental_smoke
    original_report = copy.deepcopy(previous)

    report = smoke.run_smoke(
        "tasks",
        root=tmp_path,
        plan_path="plan",
        task_ids=["new-task"],
        previous_report=previous,
    )

    assert set(executed_ids) == {"new-task"}
    assert set(report["tasks"]) == {"old-task", "new-task"}
    assert report["tasks"]["new-task"]["status"] == "pass"
    assert report["task_sha256"]["new-task"] == task_sha256(tasks[1])
    assert report["tasks"]["old-task"] == previous["tasks"]["old-task"]
    assert [run for run in report["runs"] if run["task_id"] == "old-task"] == previous[
        "runs"
    ]
    assert [
        row for row in report["negative_checks"] if row["task_id"] == "old-task"
    ] == previous["negative_checks"]
    assert report["summary"]["tasks"] == 2
    assert previous == original_report


def test_incremental_smoke_retains_unrequested_candidate_evidence(
    incremental_smoke, tmp_path
) -> None:
    tasks, previous, executed_ids = incremental_smoke
    tasks.pop(0)
    previous["protocol_checks"]["old-task"] = {"two_selections": True}
    original_report = copy.deepcopy(previous)

    report = smoke.run_smoke(
        "formal-tasks",
        root=tmp_path,
        plan_path="plan",
        task_ids=["new-task"],
        previous_report=previous,
    )

    assert set(executed_ids) == {"new-task"}
    assert report["tasks"]["old-task"] == previous["tasks"]["old-task"]
    assert report["task_sha256"]["old-task"] == previous["task_sha256"]["old-task"]
    assert report["protocol_checks"]["old-task"] == {"two_selections": True}
    assert [run for run in report["runs"] if run["task_id"] == "old-task"]
    assert [
        row for row in report["negative_checks"] if row["task_id"] == "old-task"
    ]
    assert report["audit_scope"]["preserved_task_count"] == 1
    assert report["summary"]["tasks"] == 2
    assert previous == original_report

    full_report = smoke.run_smoke(
        "formal-tasks", root=tmp_path, plan_path="plan", previous_report=previous
    )
    assert set(full_report["tasks"]) == {"new-task"}
    assert "old-task" not in full_report["protocol_checks"]


def test_incremental_smoke_keeps_old_hashes_for_staleness_detection(
    incremental_smoke,
    tmp_path,
) -> None:
    tasks, previous, _ = incremental_smoke
    tasks[0]["instructions"] = {**tasks[0]["instructions"], "a": "Changed request"}

    report = smoke.run_smoke(
        "tasks",
        root=tmp_path,
        plan_path="plan",
        task_ids=["new-task"],
        previous_report=previous,
    )

    assert report["task_sha256"]["old-task"] == previous["task_sha256"]["old-task"]
    assert report["tasks"]["old-task"] == previous["tasks"]["old-task"]
    assert report["task_sha256"]["old-task"] != task_sha256(tasks[0])


def test_incremental_smoke_requires_previous_report(
    incremental_smoke, tmp_path
) -> None:
    with pytest.raises(ValueError, match="requires an existing complete report"):
        smoke.run_smoke(
            "tasks",
            root=tmp_path,
            plan_path="plan",
            task_ids=["new-task"],
        )


@pytest.mark.parametrize("missing_section", ["runs", "tasks", "task_sha256"])
def test_incremental_smoke_rejects_missing_preserved_task_evidence(
    incremental_smoke,
    tmp_path,
    missing_section,
) -> None:
    _, previous, _ = incremental_smoke
    previous.pop(missing_section)

    with pytest.raises(ValueError, match="covering every task not being rerun"):
        smoke.run_smoke(
            "tasks",
            root=tmp_path,
            plan_path="plan",
            task_ids=["new-task"],
            previous_report=previous,
        )


def test_covering_states_avoids_cartesian_repetition() -> None:
    states = [
        {"selected_entity_ids": [left, right], "fields": {}}
        for left in ("a", "b")
        for right in ("x", "y", "z")
    ]

    covered = _covering_states(states)

    assert len(covered) == 3
    assert set().union(*(set(state["selected_entity_ids"]) for state in covered)) == {
        "a",
        "b",
        "x",
        "y",
        "z",
    }


@pytest.fixture
def replacement_smoke(tmp_path):
    materialize_site(products("graft"), tmp_path)
    definition = task("graft")
    definition["graft"] = {
        "transition": "replace",
        "initial_goal": {
            "target_granularity": "variant",
            "acceptable_states": [
                {
                    "selected_entity_ids": [item_id("graft", 0), item_id("graft", 1)],
                    "fields": {},
                }
            ],
        },
    }
    definition["goal"]["acceptable_states"] = [
        {
            "selected_entity_ids": [item_id("graft", 1), item_id("graft", 2)],
            "fields": {},
        }
    ]
    plan = replace(
        PLAN,
        track_conditions={"graft": ("online",)},
        release_trigger="two_selections",
        pre_release_max_steps=20,
        post_release_max_steps=20,
    )
    return definition, plan


def test_online_only_clarify_smoke_checks_denied_tool_without_plan_condition(
    monkeypatch, tmp_path
):
    definition = task("clarify")
    materialize_site(products("clarify"), tmp_path)
    plan = replace(
        PLAN,
        track_conditions={"clarify": ("online",)},
        track_probe_conditions={},
        conditions={"online": CONDITIONS["online"]},
    )
    monkeypatch.setattr(smoke, "load_construction_tasks", lambda _, task_ids=None, **kwargs: [definition])
    monkeypatch.setattr(smoke, "load_experiment_plan", lambda _: plan)

    report = smoke.run_smoke("tasks", root=tmp_path, plan_path="plan")

    assert report["summary"]["successful_correct_runs"] == 1
    assert report["summary"]["protocol_failures"] == []
    assert report["protocol_checks"][definition["id"]] == {
        "first_question_gets_the_answer": True,
        "second_question_returns_limit_response": True,
        "no_ask_rejects_ask_user": True,
    }
    assert report["runs"][0]["asked"] is True
    assert report["runs"][0]["answer_delivered"] is True
    assert set(plan.conditions) == {"online"}


@pytest.mark.parametrize(
    "trigger",
    ["timeline_step", "first_native_commitment", *sorted(STAGE_RELEASE_TRIGGERS)],
)
def test_scripted_smoke_uses_registered_trigger_with_native_actions(
    replacement_smoke, tmp_path, trigger
):
    definition, plan = replacement_smoke
    plan = replace(
        plan,
        release_trigger=trigger,
        pre_release_max_steps=20 if trigger in STAGE_RELEASE_TRIGGERS else None,
        post_release_max_steps=20 if trigger in STAGE_RELEASE_TRIGGERS else None,
    )

    record = smoke._run_scripted(
        definition,
        CONDITIONS["online"],
        root=tmp_path,
        plan=plan,
        output_root=None,
    )

    assert record["meta"]["runtime"]["release_trigger"] == trigger
    assert (
        record["meta"]["runtime"]["pre_release_max_steps"] == plan.pre_release_max_steps
    )
    assert (
        record["meta"]["runtime"]["post_release_max_steps"]
        == plan.post_release_max_steps
    )
    assert record["live_state"] == definition["goal"]["acceptable_states"][0]
    assert check_run(definition, record)["e2e_success"] is True
    assert all(smoke._release_protocol_checks(definition, record, plan).values())
    releases = [a for a in record["actions"] if a["type"] == "instruction_injection"]
    assert len(releases) == 1
    if trigger == "two_selections":
        native_trigger = first_release_trigger(record["actions"], trigger)
        assert native_trigger is not None
        assert native_trigger[1]["state_before"]["selected_entity_ids"] == [
            item_id("graft", 0)
        ]
        assert native_trigger[1]["state_after"]["selected_entity_ids"] == [
            item_id("graft", 0),
            item_id("graft", 1),
        ]
        assert releases[0]["step"] == native_trigger[1]["step"] + 1
        assert record["steps"] > releases[0]["step"]


@pytest.mark.parametrize("remaining_count", [0, 1])
def test_two_selection_smoke_releases_before_removing_initial_choices(
    replacement_smoke, tmp_path, remaining_count
):
    definition, plan = replacement_smoke
    definition["graft"]["transition"] = "drop"
    definition["goal"]["acceptable_states"][0]["selected_entity_ids"] = [
        item_id("graft", 1)
    ][:remaining_count]

    record = smoke._run_scripted(
        definition,
        CONDITIONS["online"],
        root=tmp_path,
        plan=plan,
        output_root=None,
    )

    assert check_run(definition, record)["e2e_success"] is True
    assert all(smoke._release_protocol_checks(definition, record, plan).values())
    release_index = next(
        i
        for i, action in enumerate(record["actions"])
        if action["type"] == "instruction_injection"
    )
    assert all(
        i > release_index
        for i, action in enumerate(record["actions"])
        if action["type"] == "deselect_entity"
    )


@pytest.mark.parametrize("fault", ["early", "duplicate", "missing", "wrong_trigger"])
def test_smoke_protocol_evidence_rejects_broken_release_records(
    replacement_smoke, tmp_path, fault
):
    definition, plan = replacement_smoke
    record = smoke._run_scripted(
        definition,
        CONDITIONS["online"],
        root=tmp_path,
        plan=plan,
        output_root=None,
    )
    release = next(a for a in record["actions"] if a["type"] == "instruction_injection")
    if fault == "early":
        first_selection = next(
            a for a in record["actions"] if a["type"] == "select_entity"
        )
        release["step"] = first_selection["step"] + 1
    elif fault == "duplicate":
        record["actions"].append(copy.deepcopy(release))
    elif fault == "missing":
        record["actions"].remove(release)
    else:
        record["meta"]["runtime"]["release_trigger"] = "timeline_step"

    assert not all(smoke._release_protocol_checks(definition, record, plan).values())


def test_smoke_report_uses_snapshot_and_keeps_positive_negative_protocol_evidence(
    replacement_smoke, monkeypatch, tmp_path
):
    definition, plan = replacement_smoke
    snapshot = site_data("graft")
    snapshot["entities"][0]["name"] = "Snapshot Product"
    snapshot_path = tmp_path / "site-snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot))
    plan = replace(plan, site_snapshots={"egomart": str(snapshot_path)})
    monkeypatch.setattr(smoke, "load_construction_tasks", lambda _, task_ids=None, **kwargs: [definition])
    monkeypatch.setattr(smoke, "load_experiment_plan", lambda _: plan)

    report = smoke.run_smoke("tasks", root=tmp_path, plan_path="plan")

    row = report["tasks"][definition["id"]]
    assert row["status"] == "pass"
    assert row["task_site_execution_sha256"] == task_site_execution_sha256(
        definition, snapshot
    )
    assert row["protocol_checks"] > 0
    assert report["summary"]["successful_correct_runs"] == 1
    assert report["summary"]["preflight_failures"] == []
    assert report["summary"]["negative_failures"] == []
    assert report["summary"]["protocol_failures"] == []
    assert {row["mutation"] for row in report["negative_checks"]} >= {
        "decoy_only",
        "target_plus_extra",
        "missing_required_event",
        "missing_live_state",
        "duplicate_quantity_through_site",
    }
