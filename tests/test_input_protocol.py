"""Offline checks for the wearer-session input protocol."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from egocua.adapters.browsergym import ModelInput, note_invalid_action
from egocua.experiment import protocol_mismatch
from egocua.messages import (
    COMPRESSED_HISTORY_NOTE,
    GRAFT_UPFRONT_INPUT_POLICY,
    MODEL_INPUT_POLICY,
    model_input_policy,
    glasses_received,
    release_label_step,
    release_text,
    wearer_answered,
    wearer_said,
)
from egocua.protocol import ProtocolError, RunSession
from egocua.timeline import Condition, Timeline, TimelineError
from tests.helpers import PLAN, complete_run_record, task


def test_release_labels_quote_the_payload_and_name_the_step():
    speech = {"type": "instruction_injection", "step": 4, "payload": 'Keep "A"'}
    images = {"type": "visual_context_update", "step": 4, "payload": {"frame_paths": []}}
    answer = {"type": "clarification_answer", "step": 3, "payload": "The blue one."}

    assert release_label_step(speech) == 4
    assert release_label_step(answer) == 4
    assert release_text(speech, 4) == 'The wearer said before step 4: "Keep "A""'
    assert release_text(images, 4) == glasses_received(4)
    assert release_text(answer, 4) == wearer_answered("The blue one.", 4)
    assert "added this requirement" not in release_text(speech, 4)
    assert "more recent view" not in release_text(images, 4)


def test_initial_and_later_speech_use_the_same_label():
    assert wearer_said("Find this.", 1).startswith('The wearer said before step 1: "')


def test_invalid_action_feedback_is_not_pinned():
    context = SimpleNamespace(
        initial_payload={
            "instruction": "Find this.",
            "observations": [],
            "clarification_available": False,
        }
    )
    inputs = ModelInput(context)
    entry: dict = {"result": []}
    note_invalid_action(entry, ValueError("not json"))
    assert entry["result"] == [
        {
            "error": "Your last response was not a valid action: not json",
            "is_done": False,
        }
    ]
    assert all("not a valid action" not in part["text"] for part in inputs.parts)


def _browser_record(policy: str | None) -> dict:
    record = {
        "track": "graft",
        "condition": PLAN.condition("online").as_dict(),
        "meta": {
            "agent": "browsergym",
            "model": PLAN.model,
            "model_version": PLAN.model_version,
            "max_steps": PLAN.max_steps,
            "schedule_index": 0,
            "default_release_step": PLAN.default_release_step,
            "reference_steps": PLAN.reference_steps,
            "runtime": {"release_trigger": PLAN.release_trigger},
        },
    }
    if policy is not None:
        record["trace"] = {"agent_settings": {"system_prompt_policy": policy}}
    return record


def test_formal_reuse_rejects_the_hinted_input_policy():
    plan = replace(PLAN, adapter="browsergym")
    hinted = _browser_record("browsergym-verbatim-input-v4")
    current = _browser_record(MODEL_INPUT_POLICY)
    missing = _browser_record(None)

    assert protocol_mismatch(hinted, plan, 0) is None
    assert (
        protocol_mismatch(hinted, plan, 0, require_current_input_policy=True)
        == "wrong_system_prompt_policy"
    )
    assert (
        protocol_mismatch(missing, plan, 0, require_current_input_policy=True)
        == "missing_system_prompt_policy"
    )
    assert protocol_mismatch(current, plan, 0, require_current_input_policy=True) is None


def test_freezing_basic_inputs_does_not_change_the_registered_provider_contract():
    plan = replace(PLAN, adapter="browsergym", model="deepseek-flash", model_version="deepseek-flash")
    record = _browser_record(MODEL_INPUT_POLICY)
    record["meta"].update(model=plan.model, model_version=plan.model_version, provider="openai-compatible")
    assert protocol_mismatch(record, plan, 0) is None
    assert protocol_mismatch(record, replace(plan, frozen_asset_root="/study/inputs"), 0) is None


def test_frozen_composite_deepseek_still_requires_the_official_endpoint():
    plan = replace(
        PLAN, adapter="browsergym", model="deepseek-flash", model_version="deepseek-flash",
        track_conditions={"composite": ("online",)}, frozen_asset_root="/study/inputs",
    )
    record = _browser_record(MODEL_INPUT_POLICY)
    record["track"] = "composite"
    record["meta"].update(model=plan.model, model_version=plan.model_version, provider="openai-compatible")
    assert protocol_mismatch(record, plan, 0) == "unverified_official_provider"
    record["meta"]["provider"] = "deepseek-official"
    record["meta"]["runtime"]["provider_base_url"] = "https://api.deepseek.com"
    assert protocol_mismatch(record, plan, 0) is None
    record["meta"]["runtime"]["provider_base_url"] = "https://other.example.com"
    assert protocol_mismatch(record, plan, 0) == "unverified_official_provider"


def test_continuous_delivery_releases_unlabeled_batches():
    definition = task("refresh")
    definition["visual_session"] = {
        "frequency_samples": {
            "window_seconds": [0, 4],
            "timestamps_seconds": [0, 1, 2, 3, 4],
            "frame_paths": [f"frames/s{index}.jpg" for index in range(5)],
        }
    }
    condition = Condition(
        "cadence",
        "online",
        "full",
        "allowed",
        None,
        frame_interval_seconds=1,
        frame_delivery_steps=2,
    )
    timeline = Timeline.compile(
        definition, condition, default_release_step=3, reference_steps=None
    )

    assert timeline.initial_observations[0]["frame_paths"] == ["frames/s0.jpg"]
    assert [release.step for release in timeline.releases] == [2, 4, 6, 8]
    assert [release.ref for release in timeline.releases] == ["S1", "S2", "S3", "S4"]
    assert timeline.sufficiency_step() is None


def _stream_task(track: str, onset: float | None = None) -> dict:
    definition = task(track)
    definition["visual_session"] = {
        "frequency_samples": {
            "window_seconds": [0, 4],
            "timestamps_seconds": [0, 1, 2, 3, 4],
            "frame_paths": [f"frames/s{index}.jpg" for index in range(5)],
        }
    }
    if onset is not None:
        definition["visual_session"]["update_onset_seconds"] = onset
    return definition


STREAM = Condition(
    "stream",
    "online",
    "full",
    "allowed",
    None,
    frame_interval_seconds=1,
    frame_delivery_steps=2,
)


def test_text_only_replay_keeps_graft_on_source_time_without_sending_images():
    condition = replace(STREAM, scope="none", frame_delivery_steps=1, batch_seconds=1)
    timeline = Timeline.compile(
        _stream_task("graft", onset=2.5),
        condition,
        default_release_step=1,
        reference_steps=None,
    )
    assert not timeline.initial_observations
    assert [(r.kind, r.step, r.streamed) for r in timeline.releases] == [
        ("instruction", 3, True)
    ]


def test_only_fixed_replay_can_start_a_live_run():
    from egocua.execution import require_stream_input
    from tests.conditions import CONDITIONS

    condition = replace(STREAM, frame_delivery_steps=1, batch_seconds=1)
    require_stream_input(condition, "timeline_step", 20)
    with pytest.raises(ValueError, match="frame batch"):
        require_stream_input(CONDITIONS["online"], "timeline_step", 20)
    with pytest.raises(ValueError, match="website events"):
        require_stream_input(condition, "first_native_commitment", 20)
    with pytest.raises(ValueError, match="20-step"):
        require_stream_input(condition, "timeline_step", 40)


def test_graft_follow_up_request_follows_the_session_clock():
    timeline = Timeline.compile(
        _stream_task("graft", onset=2.5),
        STREAM,
        default_release_step=3,
        reference_steps=None,
    )

    follow_up = [r for r in timeline.releases if r.kind == "instruction"]
    assert [(r.step, r.streamed, r.update) for r in follow_up] == [(6, True, True)]
    assert follow_up[0].payload == "Also apply B"
    assert not any(r.update for r in timeline.releases if r.kind == "observation")


def test_streamed_graft_requires_a_reviewed_follow_up_onset():
    with pytest.raises(TimelineError, match="update_onset_seconds"):
        Timeline.compile(
            _stream_task("graft"), STREAM, default_release_step=3, reference_steps=None
        )


@pytest.mark.parametrize("track", ["graft", "refresh"])
def test_change_in_the_opening_batch_is_rejected(track):
    with pytest.raises(TimelineError, match="opening batch"):
        Timeline.compile(
            _stream_task(track, onset=0),
            STREAM,
            default_release_step=3,
            reference_steps=None,
        )


def test_refresh_marks_only_the_batch_that_carries_the_change():
    from egocua.timeline import is_update_event

    timeline = Timeline.compile(
        _stream_task("refresh", onset=2.5),
        STREAM,
        default_release_step=3,
        reference_steps=None,
    )

    events = [release.as_event() for release in timeline.releases]
    assert [is_update_event(event) for event in events] == [False, False, True, False]


def test_agent_actions_never_hold_the_stream():
    from egocua.protocol import RunSession

    session = RunSession(_stream_task("graft", onset=2.5), STREAM, 3, None)
    delivered = {
        step: [
            event["type"]
            for event in session.advance(
                step, release_pending=False, allow_scheduled_release=False
            )
        ]
        for step in range(1, 9)
    }

    assert delivered[2] == delivered[4] == delivered[8] == ["visual_context_update"]
    assert delivered[6] == ["visual_context_update", "instruction_injection"]
    assert delivered[1] == delivered[3] == delivered[5] == delivered[7] == []


def test_skipped_wait_frames_never_reach_the_stream():
    definition = _stream_task("refresh", onset=3.5)
    definition["visual_session"]["skipped_seconds"] = [0, 3]
    timeline = Timeline.compile(
        definition, STREAM, default_release_step=3, reference_steps=None
    )

    delivered = [timeline.initial_observations[0]["frame_paths"]] + [
        r.payload["frame_paths"] for r in timeline.releases
    ]
    assert delivered == [["frames/s0.jpg"], ["frames/s3.jpg"], ["frames/s4.jpg"]]
    assert [r.update for r in timeline.releases] == [False, True]


def test_finishing_before_the_follow_up_stands_by_until_it_arrives():
    from egocua.protocol import RunSession

    session = RunSession(_stream_task("graft", onset=2.5), STREAM, 3, None)
    assert session.advance(1) == []
    standby = session.standby(1)
    woken = session.advance(2)

    assert standby == {
        "type": "standby",
        "step": 1,
        "skipped_refs": ["S1", "S2"],
        "resume_source_step": 6,
    }
    assert [(e["type"], e["step"], e["source_step"]) for e in woken] == [
        ("visual_context_update", 2, 6),
        ("instruction_injection", 2, 6),
    ]
    assert session.standby(2) is None
    assert session.advance(3) == []
    assert [e["source_step"] for e in session.advance(4)] == [8]


@pytest.mark.parametrize("track", ["refresh", "clarify"])
def test_only_a_later_wearer_request_wakes_a_finished_agent(track):
    from egocua.protocol import RunSession

    onset = 2.5 if track == "refresh" else None
    session = RunSession(_stream_task(track, onset=onset), STREAM, 3, None)
    session.advance(1)

    assert session.standby(1) is None


def test_event_delivery_never_stands_by():
    from egocua.protocol import RunSession

    session = RunSession(task("graft"), PLAN.condition("online"), 3, None)

    assert session.standby(1) is None


def test_checker_accepts_a_run_woken_from_standby():
    from egocua.checker import check_run
    from egocua.protocol import RunSession

    definition = _stream_task("graft", onset=2.5)
    session = RunSession(definition, STREAM, 3, None)
    actions = [*session.advance(1), session.standby(1), *session.advance(2)]
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": "graft",
            "condition": STREAM.as_dict(),
            "steps": 3,
            "actions": actions,
        }
    )
    checked = check_run(definition, run)

    assert checked["run_valid"] is True, checked["run_invalid_reasons"]
    assert checked["woken_from_standby"] is True


REPLAY = replace(STREAM, frame_delivery_steps=1, batch_seconds=1)


@pytest.mark.parametrize(
    ("track", "onset", "last_frame", "instruction"),
    [
        ("graft", 2.5, 3, "Find A\n\nAlso apply B"),
        ("refresh", 2.5, 3, "Find the item selected last"),
        ("clarify", None, 1, "Find the target variant"),
    ],
)
def test_upfront_delivers_the_reviewed_replay_prefix_once(
    track, onset, last_frame, instruction
):
    definition = _stream_task(track, onset)
    definition["visual_session"]["reviewed_seconds"] = [0, 1]
    condition = replace(REPLAY, schedule="upfront", ask="open")
    session = RunSession(definition, condition, 1, None)

    assert session.initial_payload()["instruction"] == instruction
    assert session.initial_payload()["clarification_available"] is True
    assert [
        t
        for batch in session.initial_payload()["observations"]
        for t in batch["timestamps_seconds"]
    ] == list(range(last_frame + 1))
    assert session.advance(1) == []
    assert session.standby(1) is None


def test_upfront_uses_the_same_compressed_history_renderer_before_action_one():
    from egocua.frame_inputs import ObservationImage

    definition = task("graft")
    definition["visual_session"] = {
        "frequency_samples": {
            "window_seconds": [0, 39],
            "timestamps_seconds": list(range(40)),
            "frame_paths": [f"frames/s{index}.jpg" for index in range(40)],
        },
        "reviewed_seconds": [0, 24],
        "update_onset_seconds": 24,
    }
    condition = replace(
        REPLAY, schedule="upfront", batch_seconds=8, visual_history="compressed"
    )
    session = RunSession(definition, condition, 1, None)
    context = SimpleNamespace(
        initial_payload=session.initial_payload(),
        observation_images=lambda batch: [
            ObservationImage("image/jpeg", "", path, [{"path": path}])
            for path in batch["frame_paths"]
        ],
    )
    inputs = ModelInput(context, condition.visual_history)

    assert inputs.older_batch_count == 1
    assert [
        inputs.layouts[index]["layout"][0]["path"] for index in inputs.image_indices
    ] == [
        f"frames/s{index}.jpg"
        for index in [*range(8), 8, 10, 12, 14, *range(16, 32)]
    ]
    assert session.advance(1) == []


def test_graft_initial_only_removes_the_follow_up_without_removing_visual_batches():
    definition = _stream_task("graft", onset=2.5)
    online = RunSession(definition, REPLAY, 1, None)
    no_update = RunSession(definition, replace(REPLAY, scope="initial_only"), 1, None)

    assert no_update.initial_payload() == online.initial_payload()
    assert no_update.timeline.releases == tuple(
        release for release in online.timeline.releases if release.kind == "observation"
    )
    assert no_update.standby(1) is None


def test_refresh_initial_only_delivers_the_prechange_batches_and_stops():
    definition = _stream_task("refresh", onset=2.5)
    online = RunSession(definition, REPLAY, 1, None)
    no_update = RunSession(definition, replace(REPLAY, scope="initial_only"), 1, None)

    assert no_update.initial_payload() == online.initial_payload()
    assert [r.step for r in no_update.timeline.releases] == [1, 2]
    assert no_update.timeline.releases == online.timeline.releases[:2]
    assert no_update.advance(3) == online.advance(3)[:2]
    assert no_update.advance(4) == []


def test_clarify_no_ask_keeps_the_ambiguous_request_and_the_entire_replay():
    definition = _stream_task("clarify")
    online = RunSession(definition, REPLAY, 1, None)
    no_ask = RunSession(definition, replace(REPLAY, ask="denied"), 1, None)

    assert no_ask.initial_payload() == {
        **online.initial_payload(), "clarification_available": False
    }
    assert no_ask.timeline.releases == online.timeline.releases
    with pytest.raises(ProtocolError, match="clarification is not available"):
        no_ask.ask_user("Which one?", 1)


def test_change_cue_says_only_that_the_scene_changed():
    from egocua.protocol import RunSession
    from egocua.timeline import CHANGE_CUE_TEXT, INSTRUCTION, is_update_event

    definition = _stream_task("refresh", onset=2.5)
    silent = Timeline.compile(definition, REPLAY, default_release_step=1, reference_steps=None)
    cued_condition = replace(REPLAY, change_cue=True)
    cued = Timeline.compile(
        definition, cued_condition, default_release_step=1, reference_steps=None
    )

    cue = [release for release in cued.releases if release.kind == INSTRUCTION]
    assert [(r.step, r.payload, r.update) for r in cue] == [(3, CHANGE_CUE_TEXT, False)]
    assert [r for r in cued.releases if r.kind != INSTRUCTION] == list(silent.releases)
    assert not is_update_event(cue[0].as_event())
    assert RunSession(definition, cued_condition, 1, None).standby(1) is None
    with pytest.raises(TimelineError, match="Refresh change"):
        Timeline.compile(
            _stream_task("graft", onset=2.5),
            cued_condition,
            default_release_step=1,
            reference_steps=None,
        )


def test_follow_up_after_finish_waits_for_the_agent_to_finish():
    from egocua.protocol import RunSession

    condition = replace(REPLAY, follow_up_after_finish=True)
    working = RunSession(_stream_task("graft", onset=2.5), condition, 1, None)
    delivered = [event for step in range(1, 6) for event in working.advance(step)]
    assert all(event["type"] == "visual_context_update" for event in delivered)
    assert working.standby(5)["resume_source_step"] == 6
    assert [(e["type"], e["step"], e["source_step"]) for e in working.advance(6)] == [
        ("instruction_injection", 6, 3)
    ]

    early = RunSession(_stream_task("graft", onset=2.5), condition, 1, None)
    early.advance(1)
    assert early.standby(1)["skipped_refs"] == ["S2"]
    assert [e["source_step"] for e in early.advance(2)] == [3, 3]


@pytest.mark.parametrize("finish_step", [None, 4])
def test_checker_accepts_a_held_follow_up_that_never_arrived(finish_step):
    from egocua.checker import check_run
    from egocua.protocol import RunSession

    definition = _stream_task("graft", onset=2.5)
    condition = replace(REPLAY, follow_up_after_finish=True)
    session = RunSession(definition, condition, 1, None)
    actions = [event for step in range(1, 5) for event in session.advance(step)]
    if finish_step is not None:
        actions.append(session.standby(finish_step))
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": "graft",
            "condition": condition.as_dict(),
            "steps": 4,
            "actions": actions,
        }
    )
    checked = check_run(definition, run)

    assert checked["run_valid"] is True, checked["run_invalid_reasons"]


@pytest.mark.parametrize(
    ("commit_step", "expected"), [(None, False), (5, True), (6, False), (7, False)]
)
def test_checker_records_whether_the_update_followed_a_commitment(
    commit_step, expected
):
    from egocua.checker import check_run
    from egocua.protocol import RunSession

    definition = _stream_task("refresh", onset=2.5)
    session = RunSession(definition, STREAM, 3, None)
    actions = [event for step in range(1, 9) for event in session.advance(step)]
    if commit_step is not None:
        actions.append({"type": "set_field", "step": commit_step})
        actions.sort(key=lambda action: action["step"])
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": "refresh",
            "condition": STREAM.as_dict(),
            "steps": 9,
            "actions": actions,
        }
    )

    assert check_run(definition, run)["update_after_commitment"] is expected


def test_batch_seconds_sets_frames_per_step_independently_of_tiling():
    definition = task("clarify")
    definition["visual_session"] = {
        "frequency_samples": {
            "window_seconds": [0, 15],
            "timestamps_seconds": list(range(16)),
            "frame_paths": [f"frames/s{index}.jpg" for index in range(16)],
        }
    }
    condition = replace(
        STREAM, frame_delivery_steps=1, wearer_frames_per_image=4, batch_seconds=8
    )
    timeline = Timeline.compile(
        definition, condition, default_release_step=3, reference_steps=None
    )

    assert len(timeline.initial_observations[0]["frame_paths"]) == 8
    assert [len(r.payload["frame_paths"]) for r in timeline.releases] == [8]
    assert Condition.from_dict(condition.as_dict()) == condition


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frame_delivery_steps": None, "batch_seconds": 8}, "requires frame_delivery"),
        ({"batch_seconds": 2.5}, "multiple of the frame spacing"),
        ({"batch_seconds": 0}, "multiple of the frame spacing"),
    ],
)
def test_batch_seconds_requires_a_stream_and_whole_frames(changes, message):
    with pytest.raises(TimelineError, match=message):
        replace(STREAM, **changes)


def test_stream_context_keeps_opening_key_and_latest_batches_whole():
    from egocua.messages import retained_images

    assert retained_images([2, 2], [False, False]) == [range(2), range(2)]
    kept = retained_images([2] * 7, [False] * 7)
    assert kept[0] == range(2)
    assert kept[1] == kept[2] == range(1, 2)
    assert kept[3:] == [range(2)] * 4
    key = retained_images([2] * 7, [False, False, True, False, False, False, False])
    assert key[1] == range(1, 2)
    assert key[2] == range(2)


def test_reviewed_frames_mark_the_batches_that_carry_them():
    from egocua.messages import KEY_EVIDENCE

    definition = _stream_task("refresh", onset=2.5)
    definition["visual_session"]["reviewed_seconds"] = [0.1, 2.9]
    timeline = Timeline.compile(
        definition, STREAM, default_release_step=3, reference_steps=None
    )

    assert timeline.initial_observations[0][KEY_EVIDENCE] is True
    assert [KEY_EVIDENCE in r.payload for r in timeline.releases] == [
        False,
        False,
        True,
        False,
    ]


def test_a_reviewed_frame_outside_the_stream_is_rejected():
    definition = _stream_task("refresh", onset=2.5)
    definition["visual_session"]["reviewed_seconds"] = [9.0]
    with pytest.raises(TimelineError, match="reviewed frame at 9.0 s"):
        Timeline.compile(
            definition, STREAM, default_release_step=3, reference_steps=None
        )


def test_model_input_sends_only_retained_stream_images():
    from egocua.frame_inputs import ObservationImage

    def images(observation):
        return [
            ObservationImage("image/jpeg", "", f"img {path}", [{"path": path}])
            for path in observation["frame_paths"]
        ]

    context = SimpleNamespace(
        initial_payload={
            "instruction": "Find this.",
            "observations": [{"frame_paths": ["a0", "a1"]}],
            "clarification_available": False,
        },
        observation_images=images,
    )
    inputs = ModelInput(context)
    for step in range(2, 8):
        inputs.release(
            {
                "type": "visual_context_update",
                "step": step,
                "payload": {"frame_paths": [f"b{step}0", f"b{step}1"]},
            }
        )
    inputs.release({"type": "instruction_injection", "step": 8, "payload": "Also B"})

    assert inputs.image_indices == [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    texts = [part["text"] for part in inputs.parts if part["type"] == "text"]
    assert "img b20" not in texts and "img b21" in texts
    assert texts[-1] == 'The wearer said before step 8: "Also B"'
    assert len(inputs.layouts) == 14

    from egocua.messages import KEY_EVIDENCE

    inputs = ModelInput(context)
    for step in range(2, 8):
        payload = {"frame_paths": [f"b{step}0", f"b{step}1"]}
        if step == 2:
            payload[KEY_EVIDENCE] = True
        inputs.release({"type": "visual_context_update", "step": step, "payload": payload})
    assert inputs.image_indices == [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]


def test_stream_conditions_use_their_own_input_policy():
    from egocua.messages import STREAM_INPUT_POLICY

    assert model_input_policy("refresh", STREAM.as_dict()) == STREAM_INPUT_POLICY
    assert model_input_policy("refresh", PLAN.condition("online").as_dict()) == (
        MODEL_INPUT_POLICY
    )


@pytest.mark.parametrize(
    ("steps", "dropped_step", "valid"),
    [(9, None, True), (3, None, True), (9, 4, False)],
)
def test_checker_accepts_streamed_batches_on_their_cadence(steps, dropped_step, valid):
    from egocua.checker import check_run
    from egocua.protocol import RunSession
    from tests.helpers import complete_run_record

    definition = task("refresh")
    definition["visual_session"] = {
        "frequency_samples": {
            "window_seconds": [0, 4],
            "timestamps_seconds": [0, 1, 2, 3, 4],
            "frame_paths": [f"frames/s{index}.jpg" for index in range(5)],
        }
    }
    condition = Condition(
        "cadence",
        "online",
        "full",
        "allowed",
        None,
        frame_interval_seconds=1,
        wearer_frames_per_image=1,
        frame_delivery_steps=2,
    )
    session = RunSession(definition, condition, 3, None)
    events = [
        event
        for step in range(1, steps)
        for event in session.advance(step)
        if event["step"] != dropped_step
    ]
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": "refresh",
            "condition": condition.as_dict(),
            "steps": steps,
            "actions": events,
        }
    )

    assert check_run(definition, run)["run_valid"] is valid


def test_continuous_delivery_requires_a_prepared_pool():
    condition = Condition(
        "cadence",
        "online",
        "full",
        "allowed",
        None,
        frame_interval_seconds=1,
        frame_delivery_steps=2,
    )
    with pytest.raises(TimelineError, match="visual_session.frequency_samples"):
        Timeline.compile(
            task("refresh"), condition, default_release_step=2, reference_steps=None
        )


def test_strategy_hint_round_trips_only_when_enabled():
    hinted = replace(PLAN.condition("online"), name="hinted", strategy_hint=True)
    assert hinted.as_dict()["strategy_hint"] is True
    assert Condition.from_dict(hinted.as_dict()) == hinted
    assert "strategy_hint" not in PLAN.condition("online").as_dict()


def test_unanswered_question_does_not_coach_the_model():
    open_ask = replace(PLAN.condition("online"), name="open", ask="open")
    timeline = Timeline.compile(
        task("graft"), open_ask, default_release_step=2, reference_steps=None
    )
    replies = [
        timeline.answer("Which size?", 2)["payload"],
        timeline.answer("Which size?", 3, questions_used=1)["payload"],
    ]
    assert replies == ["Just what I asked for.", "I've already answered."]
    assert not any("best judgment" in reply for reply in replies)


@pytest.mark.parametrize(
    "later",
    [
        "Also add the matching accessory; keep the first item.",
        "Instead, choose the other size.",
        'Keep the first one, but replace the second with the "large" version.',
    ],
)
def test_graft_upfront_preserves_both_original_requests_in_order(later):
    definition = task("graft")
    original = "Find the item I am using."
    definition["instructions"].update(a=original, b=later, bundled="Unused rewrite")
    timeline = Timeline.compile(
        definition,
        PLAN.condition("upfront"),
        default_release_step=2,
        reference_steps=None,
    )

    assert timeline.initial()["instruction"] == original + "\n\n" + later
    assert not timeline.releases
    assert timeline.sufficiency_step() == 0
    inputs = ModelInput(
        SimpleNamespace(
            initial_payload=timeline.initial(),
            observation_images=lambda observation: [],
        )
    )
    assert inputs.parts[0]["text"] == wearer_said(original + "\n\n" + later, 1)


def test_graft_upfront_reuse_requires_verbatim_combined_input_policy():
    plan = replace(PLAN, adapter="browsergym")
    condition = plan.condition("upfront").as_dict()
    old = _browser_record(MODEL_INPUT_POLICY)
    old.update(track="graft", condition=condition)
    current = _browser_record(GRAFT_UPFRONT_INPUT_POLICY)
    current.update(track="graft", condition=condition)

    assert protocol_mismatch(old, plan, 0) is None
    assert (
        protocol_mismatch(old, plan, 0, require_current_input_policy=True)
        == "wrong_system_prompt_policy"
    )
    assert (
        protocol_mismatch(current, plan, 0, require_current_input_policy=True) is None
    )
    assert (
        model_input_policy("graft", plan.condition("online").as_dict())
        == MODEL_INPUT_POLICY
    )
    assert model_input_policy("refresh", condition) == MODEL_INPUT_POLICY
    assert model_input_policy("clarify", condition) == MODEL_INPUT_POLICY


@pytest.mark.parametrize("policy", [MODEL_INPUT_POLICY, "browsergym-verbatim-input-v4"])
def test_old_graft_upfront_scores_its_original_bundled_context(policy):
    from egocua.checker import check_run

    definition = task("graft")
    timeline = Timeline.compile(
        definition,
        PLAN.condition("upfront"),
        default_release_step=2,
        reference_steps=None,
    )
    initial = timeline.initial()
    initial["instruction"] = definition["instructions"]["bundled"]
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "track": "graft",
            "condition": PLAN.condition("upfront").as_dict(),
            "meta": {"default_release_step": 2, "reference_steps": None},
            "steps": 1,
            "initial_context": initial,
            "live_state": {
                key: value
                for key, value in definition["goal"]["acceptable_states"][0].items()
                if key != "submission_completed"
            },
            "trace": {"agent_settings": {"system_prompt_policy": policy}},
        }
    )
    checked = check_run(definition, run)

    assert checked["run_valid"] is True
    assert checked["e2e_success"] is True
    assert (
        run["initial_context"]["instruction"] == definition["instructions"]["bundled"]
    )


def test_new_graft_upfront_rejects_rewritten_bundled_context():
    from egocua.checker import check_run

    definition = task("graft")
    timeline = Timeline.compile(
        definition,
        PLAN.condition("upfront"),
        default_release_step=2,
        reference_steps=None,
    )
    initial = timeline.initial()
    initial["instruction"] = definition["instructions"]["bundled"]
    run = complete_run_record(
        {
            "task_id": definition["id"],
            "condition": PLAN.condition("upfront").as_dict(),
            "meta": {"default_release_step": 2, "reference_steps": None},
            "steps": 1,
            "initial_context": initial,
            "trace": {
                "agent_settings": {"system_prompt_policy": GRAFT_UPFRONT_INPUT_POLICY}
            },
        }
    )

    assert (
        "invalid_initial_context" in check_run(definition, run)["run_invalid_reasons"]
    )


@pytest.mark.parametrize(
    ("policy", "expected_indices"),
    [
        ("full", list(range(12))),
        ("compressed", [0, 1, 5, 8, 11, 12, 13, 15, 16]),
        ("summary", [0, 1, *range(8, 12)]),
    ],
)
def test_retention_changes_only_older_nonopening_batches(policy, expected_indices):
    from egocua.frame_inputs import ObservationImage
    from egocua.messages import KEY_EVIDENCE

    def observation(step):
        return {"frame_paths": [f"{step}a", f"{step}b"], KEY_EVIDENCE: True}

    context = SimpleNamespace(
        initial_payload={"instruction": "Find this.", "observations": [observation(1)]},
        observation_images=lambda batch: [
            ObservationImage("image/jpeg", "", path, [{"path": path}])
            for path in batch["frame_paths"]
        ],
    )
    inputs = ModelInput(context, policy)
    for step in range(2, 7):
        inputs.release(
            {"type": "visual_context_update", "step": step, "payload": observation(step)}
        )
        if policy == "summary":
            targets = inputs.summary_targets
            response = json.dumps({
                "action": "wait",
                "parameters": {},
                "visual_summaries": dict.fromkeys(targets, f"Observed batch {step}"),
            })
            action_text, error = inputs.split_response(response)
            assert error is None and "visual_summaries" not in action_text
            if step < 4:
                assert not any("Earlier visual batch summary:" in p.get("text", "") for p in inputs.parts)
    assert inputs.older_batch_count == 3
    assert inputs.image_indices == expected_indices
    assert any(p.get("text") == "Earlier visual batch summary: Observed batch 2" for p in inputs.parts) == (policy == "summary")
    texts = [p["text"] for p in inputs.parts if p["type"] == "text"]
    if policy == "compressed":
        assert "2a" in texts and "2b" not in texts
        assert texts.count(COMPRESSED_HISTORY_NOTE) == 3
        assert [layout.get("compressed_history", False) for layout in inputs.layouts[:3]] == [
            False, False, True
        ]


def test_current_retentions_share_a_two_batch_window_and_v2_keeps_four():
    from egocua.messages import HISTORY_RETENTIONS, older_batch_indices

    for policy in HISTORY_RETENTIONS:
        assert older_batch_indices(6, policy) == range(1, 4)
    assert older_batch_indices(6, None) == range(1, 2)


def test_compressed_history_keeps_evenly_spaced_frames():
    from egocua.messages import compressed_history

    batch = {
        "frame_paths": [f"f{i}" for i in range(8)],
        "timestamps_seconds": list(range(8)),
        "wearer_frames_per_image": 4,
    }
    thinned = compressed_history(batch)
    assert thinned["frame_paths"] == ["f0", "f2", "f4", "f6"]
    assert thinned["timestamps_seconds"] == [0, 2, 4, 6]
    assert thinned["wearer_frames_per_image"] == 4
    assert batch["frame_paths"] == [f"f{i}" for i in range(8)]


def test_new_stream_conditions_do_not_export_reviewed_batch_flags():
    from egocua.messages import KEY_EVIDENCE

    definition = _stream_task("refresh", onset=2.5)
    definition["visual_session"]["reviewed_seconds"] = [0.1, 2.9]
    condition = replace(STREAM, batch_seconds=1, visual_history="compressed")
    timeline = Timeline.compile(definition, condition, default_release_step=3, reference_steps=None)
    assert all(KEY_EVIDENCE not in batch for batch in timeline.initial_observations)
    assert all(KEY_EVIDENCE not in release.payload for release in timeline.releases)
    assert Condition.from_dict(condition.as_dict()) == condition
    assert model_input_policy("refresh", condition.as_dict()) != model_input_policy("refresh", STREAM.as_dict())


def test_missing_summary_does_not_discard_a_valid_action():
    from egocua.frame_inputs import ObservationImage

    context = SimpleNamespace(
        initial_payload={"instruction": "Find this.", "observations": [{"frame_paths": ["a"]}]},
        observation_images=lambda batch: [
            ObservationImage("image/jpeg", "", path, [{"path": path}])
            for path in batch["frame_paths"]
        ],
    )
    inputs = ModelInput(context, "summary")
    for step in range(2, 7):
        inputs.release({
            "type": "visual_context_update",
            "step": step,
            "payload": {"frame_paths": [f"batch-{step}"]},
        })
        action = {"action": "wait", "parameters": {}}
        command, error = inputs.split_response(json.dumps(action))
        assert json.loads(command) == action
        assert error is not None
    assert inputs.image_indices == [0, 4, 5]
    assert {"type": "text", "text": "Earlier visual batch summary unavailable."} in inputs.parts


def test_thirty_steps_are_exclusive_to_composite_live_tasks():
    from egocua.execution import require_stream_input

    condition = replace(STREAM, frame_delivery_steps=1, batch_seconds=1)
    require_stream_input(condition, "timeline_step", 30, track="composite")
    for track in ("graft", "refresh", "clarify"):
        with pytest.raises(ValueError, match="20-step"):
            require_stream_input(condition, "timeline_step", 30, track=track)
