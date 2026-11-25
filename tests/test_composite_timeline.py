"""Offline tests for composite source timing and the actual opening input."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PIL import Image

from egocua.adapters.browsergym import ModelInput
from egocua.checker import check_run, composite_event_coverage
from egocua.frame_inputs import observation_images
from egocua.messages import glasses_received, release_label_step
from egocua.protocol import ProtocolError, RunSession
from egocua.timeline import (
    ACTION_ALIGNED_REPLAY,
    COMPOSITE_TRACK,
    Condition,
    Timeline,
    TimelineError,
    is_update_event,
)
from tests.helpers import complete_run_record, task

ONLINE = Condition(
    "online",
    "online",
    "full",
    "open",
    None,
    frame_interval_seconds=1,
    frame_delivery_steps=1,
    batch_seconds=8,
    replay_timing=ACTION_ALIGNED_REPLAY,
)
OPENING = "Get a replacement for this one."
FOLLOW_UP = "Make it two."
ANSWER = "The one on the left."


def composite(mechanisms: str = "RCG") -> dict:
    definition = {
        "id": "composite-fixture",
        "track": COMPOSITE_TRACK,
        "instructions": {"base": OPENING},
        "visual_session": {
            "frequency_samples": {
                "window_seconds": [0, 31],
                "timestamps_seconds": list(range(32)),
                "frame_paths": [f"frame-{i}.jpg" for i in range(32)],
            }
        },
    }
    if "R" in mechanisms:
        definition["visual_session"]["refresh_onset_seconds"] = 8
    if "G" in mechanisms:
        definition["instructions"]["b"] = FOLLOW_UP
        definition["visual_session"]["update_onset_seconds"] = 16
    if "C" in mechanisms:
        definition["clarification"] = {
            "candidate_ids": ["left", "right"],
            "valid_slot": "left",
            "canonical_answer": ANSWER,
            "max_questions": 1,
        }
    return definition


def repeated_composite() -> dict:
    definition = {**task("graft"), **composite("C")}
    definition["visual_session"]["frequency_samples"] = {
        "window_seconds": [0, 47],
        "timestamps_seconds": list(range(48)),
        "frame_paths": [f"frame-{i}.jpg" for i in range(48)],
    }
    definition["composite_events"] = [
        {"kind": "refresh", "onset_seconds": 8},
        {"kind": "graft", "onset_seconds": 16, "request": FOLLOW_UP},
        {"kind": "refresh", "onset_seconds": 24},
        {"kind": "graft", "onset_seconds": 32, "request": "Include the spare as well."},
    ]
    return definition


def test_composite_does_not_merge_updates_or_offer_the_answer_before_asking():
    session = RunSession(composite(), ONLINE, 4, None)
    assert session.initial_payload()["instruction"] == OPENING
    assert ANSWER not in str(session.initial_payload())
    assert FOLLOW_UP not in str(session.initial_payload())
    assert session.advance(1) == []

    (physical,) = session.advance(2)
    assert physical["type"] == "visual_context_update"
    assert is_update_event(physical)
    assert physical["source_step"] == physical["step"] == 2

    answer = session.ask_user("Which one?", 2)
    assert answer["valid"] and answer["payload"] == ANSWER
    assert release_label_step(answer) == 3
    assert session.ask_user("Are you sure?", 3)["limit_reached"]

    frame, follow_up = session.advance(3)
    assert frame["type"] == "visual_context_update"
    assert not is_update_event(frame)
    assert follow_up["type"] == "instruction_injection"
    assert follow_up["source_step"] == follow_up["step"] == 3
    assert is_update_event(follow_up)
    assert follow_up["payload"] == FOLLOW_UP


@pytest.mark.parametrize(("mechanisms", "wakes"), [("CR", False), ("CG", True), ("RCG", True)])
def test_only_graft_wakes_and_standby_never_replays_skipped_frames(mechanisms, wakes):
    session = RunSession(composite(mechanisms), ONLINE, 4, None)
    assert session.advance(1) == []
    standby = session.standby(1)
    if not wakes:
        assert standby is None
        assert session.events == []
        return

    assert standby == {
        "type": "standby",
        "step": 1,
        "skipped_refs": ["S1"],
        "resume_source_step": 3,
    }
    frames, follow_up = session.advance(2)
    assert frames["payload"]["timestamps_seconds"] == list(range(16, 24))
    assert follow_up["payload"] == FOLLOW_UP
    assert follow_up["source_step"] == frames["source_step"] == 3
    assert follow_up["step"] == frames["step"] == 2
    assert not is_update_event(frames)
    assert session.standby(2) is None
    assert session.ask_user("Which one?", 2)["payload"] == ANSWER


def test_graft_before_refresh_keeps_later_physical_evidence_after_wake():
    definition = composite("RG")
    definition["visual_session"].update(refresh_onset_seconds=16, update_onset_seconds=8)
    session = RunSession(definition, ONLINE, 4, None)

    session.advance(1)
    assert session.standby(1)["resume_source_step"] == 2
    assert session.advance(2)[-1]["payload"] == FOLLOW_UP
    (refresh,) = session.advance(3)
    assert is_update_event(refresh)
    assert refresh["payload"]["timestamps_seconds"] == list(range(16, 24))


def test_repeated_events_keep_their_source_batches_and_only_graft_wakes():
    definition = repeated_composite()
    session = RunSession(definition, ONLINE, 4, None)
    assert session.advance(1) == []
    assert session.ask_user("Which one?", 1)["composite_event_id"] == "C1"
    assert session.standby(1)["skipped_refs"] == ["S1"]
    frames, speech = session.advance(2)
    assert (frames["source_step"], speech["composite_event_id"]) == (3, "G1")
    assert session.standby(2)["skipped_refs"] == ["S3"]
    frames, speech = session.advance(3)
    assert (frames["source_step"], speech["composite_event_id"]) == (5, "G2")
    assert session.standby(3) is None

    record = complete_run_record({
        "task_id": definition["id"],
        "track": definition["track"],
        "condition": ONLINE.as_dict(),
        "actions": session.events,
        "steps": 3,
    })
    result = check_run(definition, record)
    assert result["run_valid"], result["run_invalid_reasons"]
    assert result["composite_delivery"] == {"R": False, "G": True, "C": True}
    assert composite_event_coverage(result) == {
        "R1": False, "G1": True, "R2": False, "G2": True, "C1": True,
    }
    details = result["composite_event_delivery"]
    assert details["R1"]["status"] == details["R2"]["status"] == "skipped_on_standby"
    assert details["G1"]["step"] == 2 and details["G2"]["step"] == 3
    assert details["G1"]["woken_from_standby"] and details["G2"]["woken_from_standby"]
    assert details["C1"]["step"] == 2

    speech["composite_event_id"] = "G1"
    corrupted = check_run(definition, record)
    assert "invalid_instruction_injection_composite_event_id" in corrupted["run_invalid_reasons"]


def test_repeated_events_can_preview_all_of_one_mechanism():
    definition = repeated_composite()
    del definition["clarification"]
    condition = replace(ONLINE, composite_upfront="both")
    session = RunSession(definition, condition, 4, None)
    initial = session.initial_payload()
    assert initial["instruction"].count("[Later wearer request") == 2
    assert [
        observation["timestamps_seconds"][0] for observation in initial["observations"]
    ] == [0, 8, 24]
    assert session.standby(1) is None
    for step in range(1, 6):
        session.advance(step)
    record = complete_run_record({
        "task_id": definition["id"],
        "track": definition["track"],
        "condition": condition.as_dict(),
        "actions": session.events,
        "steps": 5,
    })
    result = check_run(definition, record)
    assert result["run_valid"], result["run_invalid_reasons"]
    details = result["composite_event_delivery"]
    assert all(row["status"] == "provided_upfront" for row in details.values())
    assert details["R2"]["replay_step"] == 4
    assert details["G2"]["replay_step"] is None


@pytest.mark.parametrize(
    ("index", "onset", "error"),
    [
        (1, 3, "strictly increasing"),
        (1, 9, "distinct source batches"),
        (3, 48, "falls after the session"),
    ],
)
def test_repeated_events_reject_unsorted_colliding_and_out_of_window_onsets(index, onset, error):
    definition = repeated_composite()
    definition["composite_events"][index]["onset_seconds"] = onset
    with pytest.raises(ProtocolError, match=error):
        RunSession(definition, ONLINE, 4, None)


@pytest.mark.parametrize(
    ("upfront", "preview_frames", "preview_speech"),
    [(None, False, False), ("graft", False, True), ("refresh", True, False), ("both", True, True)],
)
def test_rg_arms_preserve_the_opening_and_regular_visual_replay(
    upfront, preview_frames, preview_speech
):
    condition = replace(ONLINE, composite_upfront=upfront)
    assert Condition.from_dict(condition.as_dict()) == condition
    session = RunSession(composite("RG"), condition, 4, None)
    initial = session.initial_payload()
    expected_instruction = (
        f"{OPENING}\n\n[Later wearer request at source t=16 s]\n{FOLLOW_UP}"
        if preview_speech else OPENING
    )
    assert initial["instruction"] == expected_instruction
    opening, *preview = initial["observations"]
    assert opening["timestamps_seconds"] == list(range(8))
    assert len(preview) == int(preview_frames)

    context = SimpleNamespace(initial_payload=initial, observation_images=lambda _: ())
    text = [part["text"] for part in ModelInput(context).parts]
    assert glasses_received(1) in text
    label = "Glasses preview from source t=8–15 s:"
    assert (label in text) is preview_frames

    assert session.advance(1) == []
    (physical,) = session.advance(2)
    assert physical["payload"]["timestamps_seconds"] == list(range(8, 16))
    assert "source_preview" not in physical["payload"]
    if preview_frames:
        assert preview[0] == {**physical["payload"], "source_preview": True}
    events = session.advance(3)
    assert events[0]["payload"]["timestamps_seconds"] == list(range(16, 24))
    assert len(events) == (1 if preview_speech else 2)
    if not preview_speech:
        assert events[1]["payload"] == FOLLOW_UP

    early = RunSession(composite("RG"), condition, 4, None)
    assert (early.standby(1) is None) is preview_speech


def test_browsergym_preview_contains_the_scheduled_source_images(tmp_path):
    for index in range(16):
        Image.new("RGB", (32, 32), (index, 0, 0)).save(
            tmp_path / f"frame-{index}.jpg"
        )
    condition = replace(ONLINE, composite_upfront="refresh", wearer_frames_per_image=4)
    session = RunSession(composite("RG"), condition, 4, None)
    context = SimpleNamespace(
        initial_payload=session.initial_payload(),
        observation_images=lambda observation: observation_images(observation, tmp_path),
    )

    rendered = ModelInput(context)

    assert [item["observation_label"] for item in rendered.layouts] == [
        glasses_received(1),
        glasses_received(1),
        "Glasses preview from source t=8–15 s:",
        "Glasses preview from source t=8–15 s:",
    ]
    assert [
        frame["timestamp_seconds"]
        for item in rendered.layouts
        for frame in item["layout"]
    ] == list(range(16))
    assert len([part for part in rendered.parts if part["type"] == "image_url"]) == 4


@pytest.mark.parametrize(
    ("mechanisms", "condition", "error"),
    [
        ("R", ONLINE, "at least two mechanisms"),
        ("CG", replace(ONLINE, composite_upfront="both"), "only to Refresh"),
        ("RCG", replace(ONLINE, ask="denied"), "requires a clarification offer"),
        ("RG", replace(ONLINE, scope="initial_only"), "full online fixed-step"),
        ("RG", replace(ONLINE, change_cue=True), "full online fixed-step"),
    ],
)
def test_composite_rejects_incompatible_modes(mechanisms, condition, error):
    with pytest.raises(TimelineError, match=error):
        Timeline.compile(composite(mechanisms), condition,
                         default_release_step=4, reference_steps=None)


def test_composite_rejects_shared_event_batch():
    definition = composite()
    definition["visual_session"]["update_onset_seconds"] = 12
    with pytest.raises(TimelineError, match="distinct source batches"):
        Timeline.compile(definition, ONLINE, default_release_step=4, reference_steps=None)


def test_preview_cannot_be_used_for_a_formal_track():
    condition = replace(ONLINE, composite_upfront="graft")
    with pytest.raises(TimelineError, match="only to composite tasks"):
        Timeline.compile(task("graft"), condition, default_release_step=4, reference_steps=None)
