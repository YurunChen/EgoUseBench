"""The three RQ2 settings share a goal but expose different task inputs."""

from dataclasses import replace

import pytest

from egocua.controlled_experiments import interaction_setting_arms, online_arm
from egocua.execution import RunConfig, prepare_live_task
from egocua.interaction_settings import MATERIALS, SETTING_ORDER, load_computer_requests
from egocua.protocol import ProtocolError, RunSession
from egocua.run_materials import execution_inputs
from egocua.study_summary import _comparison_results
from egocua.timeline import ACTION_ALIGNED_REPLAY
from egocua.workspace import write_json_atomic
from tests.helpers import PLAN, task


@pytest.fixture
def setting_study(tmp_path):
    definitions = [task(track) for track in ("graft", "refresh", "clarify")]
    for definition in definitions:
        definition["visual_session"] = {
            "frequency_samples": {
                "window_seconds": [100, 123],
                "timestamps_seconds": list(range(100, 124)),
                "frame_paths": [f"frames/{i}.jpg" for i in range(24)],
            },
            "reviewed_seconds": [101, 112],
            **(
                {"update_onset_seconds": 112}
                if definition["track"] != "clarify" else {}
            ),
        }
    write_json_atomic(
        tmp_path / MATERIALS,
        {"schema_version": 1, "computer_use": {
            row["id"]: f"Standalone request for {row['track']}." for row in definitions
        }},
    )
    condition = replace(
        PLAN.condition("online"), ask="open", frame_interval_seconds=1,
        frame_delivery_steps=1, batch_seconds=8, replay_timing=ACTION_ALIGNED_REPLAY,
        visual_history="compressed", wearer_frames_per_image=4,
    )
    plan = replace(PLAN, conditions={"online": condition}, replicate_ids=(0,))
    requests = load_computer_requests(tmp_path, [row["id"] for row in definitions])
    arms = interaction_setting_arms(definitions, plan, requests)
    return arms, online_arm("formal", definitions, plan)


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_isolated_inputs_and_unchanged_online(setting_study, track):
    arms, baseline = setting_study
    assert tuple(arm.name for arm in arms) == SETTING_ORDER
    original = next(row for row in baseline.tasks if row["track"] == track)
    sessions = []
    for arm in arms:
        definition = next(row for row in arm.tasks if row["track"] == track)
        assert definition["goal"] == original["goal"]
        assert definition["site"] == original["site"]
        assert arm.plan.max_steps == baseline.plan.max_steps
        condition = arm.plan.condition(arm.name)
        config = RunConfig("browsergym", "fixture", 4, None, max_steps=20)
        assert prepare_live_task(definition, condition, config) is definition
        sessions.append(RunSession(definition, condition, 4, None))
    computer, offline, online = sessions
    assert "visual_session" not in computer.task
    assert computer.initial_payload() == {
        "instruction": f"Standalone request for {track}.",
        "observations": [],
        "clarification_available": False,
    }
    assert all(computer.advance(step) == [] for step in range(1, 21))
    with pytest.raises(ProtocolError, match="clarification is not available"):
        computer.ask_user("Which one?", 1)

    offline_input = offline.initial_payload()
    assert offline_input["clarification_available"]
    assert [
        time for batch in offline_input["observations"] for time in batch["timestamps_seconds"]
    ] == list(range(100, 124))
    assert all(offline.advance(step) == [] for step in range(1, 21))
    if track == "graft":
        assert offline_input["instruction"] == (
            'At source t=100 s: "Find A"\n\nAt source t=112 s: "Also apply B"'
        )
    elif track == "refresh":
        assert offline_input["instruction"] == original["instructions"]["base"]
    else:
        assert offline_input["instruction"] == original["instructions"]["ambiguous"]
        assert offline.ask_user("Which one?", 1)["payload"] == original[
            "clarification"
        ]["canonical_answer"]

    assert len(online.initial_payload()["observations"]) == 1
    assert online.advance(2)[0]["type"] == "visual_context_update"
    online_task = next(row for row in arms[-1].tasks if row["track"] == track)
    assert "computer_use" not in online_task["instructions"]
    assert execution_inputs(online_task, arms[-1].plan, "online") == execution_inputs(
        original, baseline.plan, "online"
    )


def test_computer_requests_reject_missing_tasks_and_site_ids(tmp_path):
    path = tmp_path / MATERIALS
    write_json_atomic(path, {
        "schema_version": 1,
        "computer_use": {"one": "Add p_a1b2c3d4e5f6."},
    })
    with pytest.raises(ValueError, match="missing requests"):
        load_computer_requests(tmp_path, ["two"])
    with pytest.raises(ValueError, match="invalid Computer Use request"):
        load_computer_requests(tmp_path, ["one"])


def test_paired_results_only_use_complete_triplets(setting_study):
    arms, baseline = setting_study
    rows = [
        {
            "task_id": row["id"], "condition": {"name": f"{arm.name}::{arm.name}"},
            "replicate_id": 0, "run_valid": True, "e2e_success": arm.name != "online",
            "costs": {},
        }
        for arm in arms for row in arm.tasks
        if (arm.name, row["track"]) != ("computer_use", "refresh")
    ]
    sources = {row["id"]: row["id"] for row in baseline.tasks}
    first, second = _comparison_results(arms, rows, sources)
    assert first["coverage"]["paired_pairs"] == second["coverage"]["paired_pairs"] == 2
    assert "adaptability" not in first["by_track"]["graft"]
    assert second["by_track"]["graft"]["adaptability"] == 0
    assert "adaptability" not in second["by_track"]["clarify"]
    assert second["by_track"]["refresh"]["coverage"]["paired_pairs"] == 0
