from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pytest

from egocua.experiment import ExperimentPlanError, load_experiment_plan
from egocua.protocol import RunSession
from egocua.timeline import (
    ACTION_ALIGNED_REPLAY,
    CHANGE_CUE_TEXT,
    REPEATED_QUESTION_REPLY,
    UNINFORMATIVE_REPLY,
    Condition,
    Timeline,
    TimelineError,
)
from tests.conditions import CONDITIONS
from tests.helpers import task


@pytest.mark.parametrize(
    "question", ["Is it the one nearer you?", "Can you clarify?", "你指哪个？"]
)
def test_clarify_answers_any_first_question_with_the_reviewed_answer(question):
    session = RunSession(task("clarify"), CONDITIONS["online"], 2, None)

    answer = session.ask_user(question, 1)

    assert answer["valid"] is True
    assert answer["payload"] == "The target variant."
    repeated = session.ask_user("Which one?", 2)
    assert repeated["limit_reached"] is True
    assert repeated["payload"] == REPEATED_QUESTION_REPLY


@pytest.mark.parametrize("track", ["graft", "refresh"])
def test_open_asking_gives_other_tracks_one_fixed_reply(track):
    open_ask = replace(CONDITIONS["online"], name="open", ask="open")
    session = RunSession(task(track), open_ask, 2, None)

    assert session.initial_payload()["clarification_available"] is True
    answer = session.ask_user("Which size do you want?", 1)
    assert answer == {
        "type": "clarification_answer",
        "step": 1,
        "question": "Which size do you want?",
        "valid": False,
        "payload": UNINFORMATIVE_REPLY,
    }
    assert session.ask_user("Anything else?", 2)["limit_reached"] is True
    allowed = RunSession(task(track), CONDITIONS["online"], 2, None)
    assert allowed.initial_payload()["clarification_available"] is False


@pytest.mark.parametrize(
    ("event", "consistent"),
    [
        ({"valid": True, "payload": "The target variant."}, True),
        ({"valid": False, "payload": "No answer was given."}, True),
        ({"valid": False, "limit_reached": True, "payload": "No further answer is available."}, True),
        ({"valid": False, "payload": "The target variant."}, False),
        ({"valid": True, "payload": UNINFORMATIVE_REPLY}, False),
    ],
)
def test_recorded_answers_are_checked_against_their_recorded_decision(event, consistent):
    timeline = Timeline.compile(
        task("clarify"), CONDITIONS["online"], default_release_step=2, reference_steps=None
    )

    assert timeline.offer.recorded_answer_is_consistent(event) is consistent


@pytest.mark.parametrize("quality", [1, 85, 95])
def test_jpeg_quality_condition_roundtrips(quality):
    condition = replace(
        CONDITIONS["online"], wearer_frames_per_image=4, wearer_jpeg_quality=quality
    )
    assert Condition.from_dict(condition.as_dict()) == condition


@pytest.mark.parametrize("extra", [{}, {"wearer_jpeg_quality": 95}])
def test_existing_condition_defaults_to_quality_95(extra):
    condition = Condition.from_dict({**CONDITIONS["online"].as_dict(), **extra})
    assert condition.wearer_jpeg_quality == 95
    assert condition.as_dict() == CONDITIONS["online"].as_dict()


@pytest.mark.parametrize("quality", [0, 96, True, False, 85.0, "85", None])
def test_condition_rejects_invalid_jpeg_quality(quality):
    with pytest.raises(
        TimelineError, match="wearer_jpeg_quality must be an integer from 1 to 95"
    ):
        Condition.from_dict(
            {**CONDITIONS["online"].as_dict(), "wearer_jpeg_quality": quality}
        )


@pytest.mark.parametrize("schedule", ["online", "upfront"])
def test_refresh_keeps_jpeg_quality_for_both_batches(schedule):
    definition = task("refresh")
    for packet in definition["visual_packets"]:
        packet["timestamps_seconds"] = list(range(len(packet["frame_paths"])))
    condition = replace(
        CONDITIONS[schedule], wearer_frames_per_image=4, wearer_jpeg_quality=85
    )
    timeline = Timeline.compile(
        definition, condition, default_release_step=2, reference_steps=None
    )
    delivered = [
        *timeline.initial_observations,
        *(release.payload for release in timeline.releases),
    ]
    assert delivered == [
        {
            "frame_paths": packet["frame_paths"],
            "timestamps_seconds": packet["timestamps_seconds"],
            "wearer_frames_per_image": 4,
            "wearer_jpeg_quality": 85,
        }
        for packet in definition["visual_packets"]
    ]


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
@pytest.mark.parametrize("timing", [ACTION_ALIGNED_REPLAY, None])
def test_replay_aligns_scheduled_input_with_action_steps(track, timing):
    definition = task(track)
    definition["visual_session"] = {
        "frequency_samples": {
            "window_seconds": [0, 23],
            "timestamps_seconds": list(range(24)),
            "frame_paths": ["frames/view.jpg"] * 24,
        },
        **({"update_onset_seconds": 8} if track != "clarify" else {}),
    }
    condition = replace(
        CONDITIONS["online"],
        frame_interval_seconds=1,
        frame_delivery_steps=1,
        batch_seconds=8,
        replay_timing=timing,
        change_cue=track == "refresh",
    )
    condition = Condition.from_dict(condition.as_dict())
    session = RunSession(definition, condition, 4, None)
    assert session.initial_payload()["observations"][0]["timestamps_seconds"] == list(range(8))
    step = 2 if timing is not None else 1
    if step == 2:
        assert session.advance(1) == []
    frames, *messages = session.advance(step)
    assert frames["payload"]["timestamps_seconds"] == list(range(8, 16))
    assert all(event["step"] == event["source_step"] == step for event in (frames, *messages))
    expected = [definition["instructions"]["b"]] if track == "graft" else []
    if track == "refresh":
        expected = [CHANGE_CUE_TEXT]
    assert [message["payload"] for message in messages] == expected


def test_action_aligned_replay_requires_fixed_batches():
    with pytest.raises(TimelineError, match="fixed-step replay"):
        replace(CONDITIONS["online"], replay_timing=ACTION_ALIGNED_REPLAY)


@pytest.mark.parametrize(
    ("schedule", "batch_seconds"), [("online", None), ("online", 8), ("upfront", 8)]
)
def test_source_packing_preserves_frames_and_delivery_steps(batch_seconds, schedule):
    definition = task("graft")
    definition["visual_session"] = {
        "update_onset_seconds": 8,
        "frequency_samples": {
            "window_seconds": [0, 23],
            "timestamps_seconds": list(range(24)),
            "frame_paths": [f"frames/{index}.jpg" for index in range(24)],
        },
    }
    condition = replace(
        CONDITIONS[schedule], wearer_frames_per_image=4,
        frame_interval_seconds=1, frame_delivery_steps=1, batch_seconds=batch_seconds,
    )
    before = Timeline.compile(
        definition, condition, default_release_step=2, reference_steps=None
    )
    definition["visual_session"]["wearer_frames_per_image"] = 1
    after = Timeline.compile(
        definition, condition, default_release_step=2, reference_steps=None
    )
    observations_before = [
        *before.initial_observations,
        *(r.payload for r in before.releases if r.kind == "observation"),
    ]
    observations_after = [
        *after.initial_observations,
        *(r.payload for r in after.releases if r.kind == "observation"),
    ]
    assert observations_before
    assert len(observations_before) == len(observations_after)
    for old, new in zip(observations_before, observations_after, strict=True):
        assert old["frame_paths"] == new["frame_paths"]
        assert old["timestamps_seconds"] == new["timestamps_seconds"]
        assert old["wearer_frames_per_image"] == 4
        assert new["wearer_frames_per_image"] == 1
    assert [(r.step, r.kind) for r in before.releases] == [
        (r.step, r.kind) for r in after.releases
    ]


def test_action_aligned_graft_standby_skips_batches_and_delivers_at_next_action():
    definition = task("graft")
    definition["visual_session"] = {
        "update_onset_seconds": 24,
        "frequency_samples": {
            "window_seconds": [0, 31],
            "timestamps_seconds": list(range(32)),
            "frame_paths": ["frames/view.jpg"] * 32,
        },
    }
    condition = replace(
        CONDITIONS["online"], frame_interval_seconds=1,
        frame_delivery_steps=1, batch_seconds=8, replay_timing=ACTION_ALIGNED_REPLAY,
    )
    session = RunSession(definition, condition, 4, None)
    assert session.advance(1) == []
    standby = session.standby(1)
    assert standby["resume_source_step"] == 4
    assert standby["skipped_refs"] == ["S1", "S2"]
    frames, follow_up = session.advance(2)
    assert frames["payload"]["timestamps_seconds"] == list(range(24, 32))
    assert follow_up["payload"] == definition["instructions"]["b"]
    assert (follow_up["step"], follow_up["source_step"]) == (2, 4)


class ConditionTest(unittest.TestCase):
    def test_condition_name_must_be_text(self) -> None:
        with self.assertRaisesRegex(TimelineError, "condition name must be non-empty"):
            Condition.from_dict(
                {
                    "name": 3,
                    "schedule": "online",
                    "scope": "full",
                    "ask": "allowed",
                    "release_at": None,
                }
            )

    def test_condition_round_trips_one_explicit_factor_object(self) -> None:
        parsed = Condition.from_dict(
            {
                "name": "release_half",
                "schedule": "online",
                "scope": "full",
                "ask": "allowed",
                "release_at": 0.5,
            }
        )
        self.assertEqual(parsed.release_at, 0.5)
        self.assertEqual(parsed.name, "release_half")
        self.assertEqual(Condition.from_dict(parsed.as_dict()), parsed)
        with self.assertRaises(TimelineError):
            Condition.from_dict({"name": "online", "catalog": "no_neighbors"})
        with self.assertRaises(TimelineError):
            Condition.from_dict({"name": "online", "bogus": 1})
        with self.assertRaises(TimelineError):
            Condition("invalid", "sometime", "full", "allowed", None)
        with self.assertRaises(TimelineError):
            Condition("invalid", "online", "full", "allowed", 1.5)

    def test_as_dict_records_every_factor(self) -> None:
        self.assertEqual(CONDITIONS["initial_only"].as_dict()["schedule"], "online")
        self.assertEqual(CONDITIONS["online"].as_dict()["schedule"], "online")

    def test_release_at_plan_requires_an_explicit_reference_length(self) -> None:
        payload = {
            "schema_version": 3,
            "experiment_id": "timing-sensitivity",
            "default_release_step": 2,
            "reference_condition": "release_half",
            "conditions": [
                {
                    "name": "release_half",
                    "schedule": "online",
                    "scope": "full",
                    "ask": "allowed",
                    "release_at": 0.5,
                }
            ],
            "tracks": {"graft": ["release_half"]},
            "replicate_ids": [0],
            "runtime": {
                "adapter": "fixture",
                "model": "fixture",
                "model_version": "fixture",
                "max_steps": 10,
            },
            "scheduler_seed": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ExperimentPlanError, "reference_steps"):
                load_experiment_plan(path)
            payload["reference_steps"] = 10
            path.write_text(json.dumps(payload))
            plan = load_experiment_plan(path)

        self.assertEqual(plan.reference_steps, 10)


class GraftTimelineTest(unittest.TestCase):
    def test_online_releases_b_at_registered_step(self) -> None:
        timeline = Timeline.compile(
            task("graft"),
            CONDITIONS["online"],
            default_release_step=4,
            reference_steps=None,
        )
        initial = timeline.initial()
        self.assertEqual(initial["instruction"], "Find A")
        self.assertEqual(len(initial["observations"]), 1)
        self.assertEqual(len(initial["observations"][0]["frame_paths"]), 8)
        self.assertFalse(initial["clarification_available"])
        self.assertEqual(timeline.due(3), [])
        (release,) = timeline.due(4)
        self.assertEqual(
            (release.ref, release.kind, release.payload),
            ("B", "instruction", "Also apply B"),
        )
        self.assertEqual(release.as_event()["type"], "instruction_injection")
        self.assertEqual(timeline.sufficiency_step(), 4)

    def test_release_at_scales_with_reference_steps(self) -> None:
        condition = Condition("release_half", "online", "full", "allowed", 0.5)
        timeline = Timeline.compile(
            task("graft"),
            condition,
            default_release_step=2,
            reference_steps=10,
        )
        self.assertEqual([r.step for r in timeline.releases], [5])
        with self.assertRaises(TimelineError):
            Timeline.compile(
                task("graft"),
                condition,
                default_release_step=2,
                reference_steps=None,
            )

    def test_session_delivers_at_the_condition_release_step(self) -> None:
        session = RunSession(
            task("graft"),
            Condition("release_0.1", "online", "full", "allowed", 0.1),
            event_step=2,
            reference_steps=10,
        )

        events = session.advance(1)

        self.assertEqual([event["step"] for event in events], [1])

    def test_session_can_release_after_a_native_commitment(self) -> None:
        session = RunSession(
            task("graft"),
            CONDITIONS["online"],
            event_step=12,
            reference_steps=None,
        )

        events = session.advance(3, release_pending=True)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["step"], 3)
        self.assertEqual(events[0]["trigger_reason"], "first_native_commitment")

    def test_session_can_disable_scheduled_release_for_commitment_trigger(self) -> None:
        session = RunSession(
            task("graft"),
            CONDITIONS["online"],
            event_step=2,
            reference_steps=None,
        )

        self.assertEqual(
            session.advance(3, allow_scheduled_release=False),
            [],
        )
        events = session.advance(
            4,
            release_pending=True,
            allow_scheduled_release=False,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["step"], 4)
        self.assertEqual(events[0]["trigger_reason"], "first_native_commitment")

    def test_upfront_and_initial_only(self) -> None:
        upfront = Timeline.compile(
            task("graft"),
            CONDITIONS["upfront"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(upfront.initial()["instruction"], "Find A\n\nAlso apply B")
        self.assertEqual(upfront.releases, ())
        self.assertEqual(upfront.sufficiency_step(), 0)
        initial_only = Timeline.compile(
            task("graft"),
            CONDITIONS["initial_only"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(initial_only.initial()["instruction"], "Find A")
        self.assertEqual(initial_only.releases, ())
        self.assertIsNone(initial_only.sufficiency_step())

    def test_visual_ablation_preserves_online_instruction_and_release(self) -> None:
        online = Timeline.compile(
            task("graft"),
            CONDITIONS["online"],
            default_release_step=2,
            reference_steps=None,
        )
        first_frame = Timeline.compile(
            task("graft"),
            CONDITIONS["first_frame"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(first_frame.instruction, online.instruction)
        self.assertEqual(first_frame.releases, online.releases)
        self.assertEqual(
            len(first_frame.initial()["observations"][0]["frame_paths"]), 1
        )
        text_only = Timeline.compile(
            task("graft"),
            CONDITIONS["text_only"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(text_only.initial()["observations"], [])
        self.assertEqual(text_only.instruction, online.instruction)
        self.assertEqual(text_only.releases, online.releases)

    def test_visual_ablation_preserves_release_at_intervention(self) -> None:
        for scope in ("full", "first_frame", "none"):
            with self.subTest(scope=scope):
                timeline = Timeline.compile(
                    task("graft"),
                    Condition("half", "online", scope, "allowed", 0.5),
                    default_release_step=2,
                    reference_steps=12,
                )
                self.assertEqual(timeline.instruction, "Find A")
                self.assertEqual(timeline.releases[0].step, 6)
                self.assertEqual(timeline.releases[0].payload, "Also apply B")


class RefreshTimelineTest(unittest.TestCase):
    def test_visual_ablation_only_removes_visual_evidence(self) -> None:
        definition = task("refresh")
        online = Timeline.compile(
            definition,
            CONDITIONS["online"],
            default_release_step=3,
            reference_steps=None,
        )
        for name in ("first_frame", "text_only"):
            with self.subTest(condition=name):
                timeline = Timeline.compile(
                    definition,
                    CONDITIONS[name],
                    default_release_step=3,
                    reference_steps=None,
                )
                self.assertEqual(timeline.instruction, online.instruction)
                self.assertEqual(timeline.offer, online.offer)
                if name == "first_frame":
                    self.assertEqual(len(timeline.releases), 1)
                    release = timeline.releases[0]
                    self.assertEqual((release.ref, release.step), ("V1", 3))
                    self.assertEqual(
                        release.payload["frame_paths"],
                        online.releases[0].payload["frame_paths"][:1],
                    )
                else:
                    self.assertEqual(timeline.releases, ())
                self.assertIsNone(timeline.sufficiency_step())
                frames = [
                    frame
                    for packet in timeline.initial_observations
                    for frame in packet["frame_paths"]
                ]
                expected = online.initial_observations[0]["frame_paths"][:1]
                self.assertEqual(frames, expected if name == "first_frame" else [])

    def test_upfront_first_frame_keeps_one_frame_from_each_packet(self) -> None:
        timeline = Timeline.compile(
            task("refresh"),
            Condition("sparse_upfront", "upfront", "first_frame", "allowed", None),
            default_release_step=3,
            reference_steps=None,
        )
        self.assertEqual(
            [len(p["frame_paths"]) for p in timeline.initial_observations], [1, 1]
        )
        self.assertEqual(timeline.releases, ())
        self.assertIsNone(timeline.sufficiency_step())

    def test_first_frame_preserves_registered_release_and_original_task(self) -> None:
        definition = task("refresh")
        before = json.dumps(definition, sort_keys=True)
        timeline = Timeline.compile(
            definition,
            Condition("sparse_late", "online", "first_frame", "allowed", 0.5),
            default_release_step=3,
            reference_steps=12,
        )
        self.assertEqual(timeline.releases[0].step, 6)
        self.assertEqual(json.dumps(definition, sort_keys=True), before)

    def test_online_releases_v1_and_t_star_depends_on_policy(self) -> None:
        replace = Timeline.compile(
            task("refresh"),
            CONDITIONS["online"],
            default_release_step=3,
            reference_steps=None,
        )
        self.assertEqual(len(replace.initial()["observations"]), 1)
        (release,) = replace.due(3)
        self.assertEqual((release.ref, release.kind), ("V1", "observation"))
        self.assertEqual(release.as_event()["type"], "visual_context_update")
        self.assertEqual(replace.sufficiency_step(), 3)
        retain_task = {
            **task("refresh"),
            "refresh": {"policy": "retain", "initial_entity_id": "p_x"},
        }
        retain = Timeline.compile(
            retain_task,
            CONDITIONS["online"],
            default_release_step=3,
            reference_steps=None,
        )
        self.assertEqual(retain.sufficiency_step(), 0)

    def test_upfront_shows_both_packets_and_initial_only_cannot_determine(self) -> None:
        upfront = Timeline.compile(
            task("refresh"),
            CONDITIONS["upfront"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(len(upfront.initial()["observations"]), 2)
        self.assertEqual(upfront.sufficiency_step(), 0)
        initial_only = Timeline.compile(
            task("refresh"),
            CONDITIONS["initial_only"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(len(initial_only.initial()["observations"]), 1)
        self.assertIsNone(initial_only.sufficiency_step())


class ClarifyTimelineTest(unittest.TestCase):
    def test_visual_ablation_keeps_ambiguous_request_and_answer_access(self) -> None:
        online = Timeline.compile(
            task("clarify"),
            CONDITIONS["online"],
            default_release_step=2,
            reference_steps=None,
        )
        for name in ("first_frame", "text_only"):
            with self.subTest(condition=name):
                timeline = Timeline.compile(
                    task("clarify"),
                    CONDITIONS[name],
                    default_release_step=2,
                    reference_steps=None,
                )
                self.assertEqual(timeline.instruction, online.instruction)
                self.assertEqual(timeline.offer, online.offer)
                self.assertEqual(timeline.releases, online.releases)
                self.assertIsNone(timeline.sufficiency_step())
                self.assertEqual(
                    timeline.answer("Which variant do you mean?", 3),
                    online.answer("Which variant do you mean?", 3),
                )
                observations = timeline.initial()["observations"]
                if name == "first_frame":
                    self.assertEqual(
                        observations[0]["frame_paths"],
                        online.initial()["observations"][0]["frame_paths"][:1],
                    )
                else:
                    self.assertEqual(observations, [])

    def test_online_offers_one_question_and_t_star_is_the_valid_answer(self) -> None:
        timeline = Timeline.compile(
            task("clarify"),
            CONDITIONS["online"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(timeline.initial()["instruction"], "Find this product")
        self.assertTrue(timeline.initial()["clarification_available"])
        self.assertIsNone(timeline.sufficiency_step())
        valid = timeline.answer("Which variant do you mean?", 3)
        self.assertTrue(valid["valid"])
        self.assertEqual(valid["payload"], "The target variant.")
        self.assertEqual(timeline.sufficiency_step([valid]), 3)
        limit = timeline.answer("Which variant?", 4, questions_used=1)
        self.assertTrue(limit["limit_reached"])

    def test_upfront_uses_oracle_text_and_no_ask_removes_offer(self) -> None:
        upfront = Timeline.compile(
            task("clarify"),
            CONDITIONS["upfront"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertEqual(upfront.initial()["instruction"], "Find the target variant")
        self.assertTrue(upfront.initial()["clarification_available"])
        self.assertEqual(upfront.sufficiency_step(), 0)
        no_ask = Timeline.compile(
            task("clarify"),
            CONDITIONS["no_ask"],
            default_release_step=2,
            reference_steps=None,
        )
        self.assertFalse(no_ask.initial()["clarification_available"])
        self.assertEqual(no_ask.instruction, "Find this product")
        self.assertEqual(no_ask.initial_observations, upfront.initial_observations)
        self.assertIsNone(no_ask.sufficiency_step())
        with self.assertRaises(TimelineError):
            no_ask.answer("Which variant?", 1)


if __name__ == "__main__":
    unittest.main()


def test_cue_facts_are_tied_to_the_released_batch_and_do_not_wake():
    definition = task('refresh')
    definition['visual_session'] = {
        'update_onset_seconds': 8,
        'frequency_samples': {
            'window_seconds': [0, 23], 'timestamps_seconds': list(range(24)),
            'frame_paths': ['frames/view.jpg'] * 24,
        },
        'cue_facts': {'text': 'Noodles enter the pan.', 'timestamps_seconds': list(range(8, 16))},
    }
    condition = replace(
        CONDITIONS['online'], frame_interval_seconds=1, frame_delivery_steps=1,
        batch_seconds=8, replay_timing=ACTION_ALIGNED_REPLAY,
        change_cue=True, change_facts=True,
    )
    assert Condition.from_dict(condition.as_dict()) == condition
    timeline = Timeline.compile(definition, condition, default_release_step=2, reference_steps=None)
    cue = next(release for release in timeline.releases if release.ref == 'CUE')
    assert cue.step == 2
    assert cue.payload == CHANGE_CUE_TEXT + ' Noodles enter the pan.'
    assert not cue.update and not cue.on_standby
    definition['visual_session']['cue_facts']['timestamps_seconds'] = list(range(16, 24))
    with pytest.raises(TimelineError, match='delivered change batch'):
        Timeline.compile(definition, condition, default_release_step=2, reference_steps=None)
    with pytest.raises(TimelineError, match='requires change_cue'):
        replace(condition, change_cue=False)
