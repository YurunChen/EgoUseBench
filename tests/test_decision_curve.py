"""Reject unsupported narrowing and reference changes.

Only Refresh-replace can re-establish candidates at the later batch boundary.
Every exclusion needs evidence; retain need not manufacture initial ambiguity.
"""

from __future__ import annotations

import unittest

import pytest

from egocua.decision import UTTERANCE_ROW, DecisionError, parse

FRAMES = ["V0:0", "V0:1", "V0:2"]
ENTITIES = {"p_first", "p_second"}
CANDIDATES = {"first": "p_first", "second": "p_second"}
BOTH = ["first", "second"]


def _goal_state(ids: list[str], **fields: str) -> dict:
    return {"selected_entity_ids": ids, "fields": fields, "submission_completed": True}


def _goal_task(states: list[dict]) -> dict:
    return {
        "track": "graft",
        "graft": {"initial_goal": {"acceptable_states": [_goal_state(["p_first"])]}},
        "goal": {"acceptable_states": states},
    }


@pytest.mark.parametrize(
    "states",
    [
        [_goal_state(["p_first"]), _goal_state(["p_second"])],
        [_goal_state(["p_first", "p_second"])],
        [
            _goal_state(["p_second"], route="walking"),
            _goal_state(["p_second"], route="cycling"),
        ],
    ],
)
def test_goal_reference_preserves_complete_alternative_states(
    states: list[dict],
) -> None:
    task = _goal_task(states)
    declared = _curve(
        _settles_at_last_frame(),
        candidates={
            "first": {"goal_ref": "initial"},
            "second": {"goal_ref": "final"},
        },
    )

    curve = parse(declared, FRAMES, ENTITIES, "task", task=task)

    assert curve.settled_at == FRAMES[-1]
    assert len(curve.candidates) == 2
    assert [
        {
            "selected_entity_ids": list(state.selected_entity_ids),
            "fields": dict(state.fields or {}),
            "submission_completed": True,
        }
        for state in curve.goal_states(curve.final_live)
    ] == states


@pytest.mark.parametrize(
    "states",
    [
        [],
        [
            {
                "selected_entity_ids": ["p_first"],
                "fields": {"quantity": 2},
                "submission_completed": True,
            }
        ],
        [
            {
                "selected_entity_ids": ["p_first"],
                "fields": {},
                "submission_completed": "yes",
            }
        ],
    ],
)
def test_goal_reference_rejects_malformed_owned_states(states: list[dict]) -> None:
    declared = _curve(
        _settles_at_last_frame(),
        candidates={
            "first": "p_first",
            "second": {"goal_ref": "final"},
        },
    )
    with pytest.raises(DecisionError, match="goal_ref"):
        parse(declared, FRAMES, ENTITIES, "task", task=_goal_task(states))


@pytest.mark.parametrize(
    "reference",
    [
        {"goal_ref": "unknown"},
        {"goal_ref": "final", "entities": ["p_first"]},
        {"goal_ref": []},
    ],
)
def test_goal_reference_rejects_unknown_or_duplicate_owners(reference: dict) -> None:
    declared = _curve(
        _settles_at_last_frame(), candidates={"first": "p_first", "second": reference}
    )
    with pytest.raises(DecisionError, match="goal_ref"):
        parse(
            declared,
            FRAMES,
            ENTITIES,
            "task",
            task=_goal_task([_goal_state(["p_second"])]),
        )


def test_goal_reference_needs_a_graft_task() -> None:
    declared = _curve(
        _settles_at_last_frame(),
        candidates={
            "first": "p_first",
            "second": {"goal_ref": "final"},
        },
    )
    with pytest.raises(DecisionError, match="goal_ref"):
        parse(declared, FRAMES, ENTITIES, "task")


def test_goal_reference_matches_quantity_and_fields_without_submission() -> None:
    declared = _curve(
        _settles_at_last_frame(),
        candidates={
            "first": "p_first",
            "second": {"goal_ref": "final"},
        },
    )
    definition = _goal_task(
        [
            {
                "selected_entity_ids": ["p_second", "p_second"],
                "fields": {"route": "walking"},
            }
        ]
    )
    curve = parse(declared, FRAMES, ENTITIES, "task", task=definition)
    (state,) = curve.goal_states(curve.final_live)
    assert state.matches(definition["goal"]["acceptable_states"][0])
    assert not state.matches(
        {"selected_entity_ids": ["p_second"], "fields": {"route": "walking"}}
    )
    assert not state.matches(
        {
            "selected_entity_ids": ["p_second", "p_second"],
            "fields": {"route": "cycling"},
        }
    )


def _curve(observed: list[dict], **overrides) -> dict:
    return {
        "candidates": CANDIDATES,
        "observed": observed,
        "established_by": "tester",
        **overrides,
    }


def _settles_at_last_frame() -> list[dict]:
    return [
        {"frame": "V0:0", "live": BOTH},
        {"frame": "V0:1", "live": BOTH},
        {
            "frame": "V0:2",
            "live": ["second"],
            "narrowed_by": "action_order",
            "why": "the first tool is set down and the second is picked up",
        },
    ]


def _settled_from_first_frame() -> list[dict]:
    return [
        {
            "frame": FRAMES[0],
            "live": ["second"],
            "narrowed_by": "scene_appearance",
            "why": "the original tool is already clear",
        },
        *[{"frame": frame, "live": ["second"]} for frame in FRAMES[1:]],
    ]


class DecisionCurveTest(unittest.TestCase):
    def _parse(self, curve: dict):
        return parse(curve, FRAMES, ENTITIES, "task")

    def _refuses(self, curve: dict, message: str) -> None:
        with self.assertRaisesRegex(DecisionError, message):
            self._parse(curve)

    def test_t_star_and_the_tempting_candidate_are_read_off_the_curve(self) -> None:
        curve = self._parse(_curve(_settles_at_last_frame()))
        self.assertEqual(curve.settled_at, "V0:2")
        self.assertTrue(curve.settled_by_frames)
        self.assertEqual(curve.tempting_roles, ("first",))
        self.assertTrue(curve.is_indexical)

    def test_an_answer_may_settle_what_the_frames_could_not(self) -> None:
        observed = [{"frame": frame, "live": BOTH} for frame in FRAMES]
        observed.append(
            {
                "frame": UTTERANCE_ROW,
                "live": ["second"],
                "narrowed_by": "action_order",
                "why": "the user says the second one",
            }
        )
        curve = self._parse(_curve(observed))
        self.assertEqual(curve.settled_at, UTTERANCE_ROW)
        self.assertFalse(curve.settled_by_frames)
        self.assertEqual(curve.tempting_roles, ("first",))

    def test_a_candidate_may_not_come_back_to_life(self) -> None:
        self._refuses(
            _curve(
                [
                    {"frame": "V0:0", "live": BOTH},
                    {
                        "frame": "V0:1",
                        "live": ["second"],
                        "narrowed_by": "action_order",
                        "why": "put down",
                    },
                    {"frame": "V0:2", "live": BOTH},
                ]
            ),
            "came back to life",
        )

    def test_ruling_a_candidate_out_requires_saying_why(self) -> None:
        self._refuses(
            _curve(
                [
                    {"frame": "V0:0", "live": BOTH},
                    {"frame": "V0:1", "live": BOTH},
                    {"frame": "V0:2", "live": ["second"]},
                ]
            ),
            "needs narrowed_by and why",
        )

    def test_claiming_a_narrowing_that_did_not_happen_is_refused(self) -> None:
        self._refuses(
            _curve(
                [
                    {"frame": "V0:0", "live": BOTH},
                    {
                        "frame": "V0:1",
                        "live": BOTH,
                        "narrowed_by": "action_order",
                        "why": "x",
                    },
                    {
                        "frame": "V0:2",
                        "live": ["second"],
                        "narrowed_by": "action_order",
                        "why": "y",
                    },
                ]
            ),
            "nothing was ruled out",
        )

    def test_a_task_nothing_was_ever_in_doubt_about_is_refused(self) -> None:
        self._refuses(_curve(_settled_from_first_frame()), "nothing to decide")

    def test_refresh_retain_may_preserve_a_referent_clear_from_the_first_frame(
        self,
    ) -> None:
        curve = parse(
            _curve(_settled_from_first_frame()),
            FRAMES,
            ENTITIES,
            "task",
            task={"track": "refresh", "refresh": {"policy": "retain"}},
        )

        self.assertEqual(curve.settled_at, FRAMES[0])
        self.assertEqual(
            [observation.live for observation in curve.observations], [("second",)] * 3
        )

    def test_other_task_types_still_require_an_ambiguous_frame(self) -> None:
        for track, policy in (
            ("graft", "retain"),
            ("clarify", "retain"),
            ("refresh", "replace"),
        ):
            with self.subTest(track=track, policy=policy):
                with self.assertRaisesRegex(DecisionError, "nothing to decide"):
                    parse(
                        _curve(_settled_from_first_frame()),
                        FRAMES,
                        ENTITIES,
                        "task",
                        task={"track": track, "refresh": {"policy": policy}},
                    )

    def test_refresh_retain_still_requires_explanation_for_ruling_out_a_candidate(
        self,
    ) -> None:
        observed = _settled_from_first_frame()
        observed[0].pop("why")

        with self.assertRaisesRegex(DecisionError, "needs narrowed_by and why"):
            parse(
                _curve(observed),
                FRAMES,
                ENTITIES,
                "task",
                task={"track": "refresh", "refresh": {"policy": "retain"}},
            )

    def test_every_released_frame_must_be_read(self) -> None:
        self._refuses(
            _curve([{"frame": "V0:0", "live": BOTH}, {"frame": "V0:1", "live": BOTH}]),
            "must read every released frame",
        )

    def test_a_candidate_the_site_does_not_have_is_refused(self) -> None:
        curve = _curve(_settles_at_last_frame())
        curve["candidates"] = {"first": "p_first", "second": "p_missing"}
        self._refuses(curve, "entities the site does not have")

    def test_the_curve_must_say_who_read_the_frames(self) -> None:
        curve = _curve(_settles_at_last_frame())
        curve.pop("established_by")
        self._refuses(curve, "must name who read the frames")

    def test_an_unknown_narrowing_cause_is_refused(self) -> None:
        observed = _settles_at_last_frame()
        observed[-1]["narrowed_by"] = "vibes"
        self._refuses(_curve(observed), "narrowed_by must be one of")

    def test_appearance_narrowing_is_not_indexical(self) -> None:
        observed = _settles_at_last_frame()
        observed[-1]["narrowed_by"] = "scene_appearance"
        self.assertFalse(self._parse(_curve(observed)).is_indexical)


if __name__ == "__main__":
    unittest.main()


def test_key_events_equal_dense_curve_without_duplicate_judgments():
    first = {"frame": "V0:0", "live": BOTH}
    last = {
        "frame": "V0:2",
        "live": ["second"],
        "narrowed_by": "action_order",
        "why": "The second object was used.",
    }
    sparse = _curve([first, last])
    sparse["recording"] = "key_events"
    dense = _curve([first, {"frame": "V0:1", "live": BOTH}, last])
    assert parse(sparse, FRAMES, ENTITIES, "task") == parse(
        dense, FRAMES, ENTITIES, "task"
    )


@pytest.mark.parametrize(
    "frames",
    [["V0:1"], ["V0:0", "unknown"], ["V0:0", "V0:0"], ["V0:0", "V0:2", "V0:1"]],
)
def test_key_events_reject_invalid_release_references(frames):
    declared = _curve([{"frame": frame, "live": BOTH} for frame in frames])
    declared["recording"] = "key_events"
    with pytest.raises(DecisionError, match="key events"):
        parse(declared, FRAMES, ENTITIES, "task")


def test_key_events_preserve_utterance_resolution():
    declared = _curve(
        [
            {"frame": "V0:0", "live": BOTH},
            {
                "frame": UTTERANCE_ROW,
                "live": ["second"],
                "narrowed_by": "user_utterance",
                "why": "The wearer selected the second object.",
            },
        ]
    )
    declared["recording"] = "key_events"
    curve = parse(declared, FRAMES, ENTITIES, "task")
    assert curve.settled_at == UTTERANCE_ROW
    assert curve.tempting_roles == ("first",)
    assert [row.frame for row in curve.observations] == FRAMES + [UTTERANCE_ROW]


def _refresh_switch_rows() -> list[dict]:
    return [
        {
            "frame": "V0:0",
            "live": ["first"],
            "narrowed_by": "scene_appearance",
            "why": "The initial view clearly shows the first tool.",
        },
        {"frame": "V0:1", "live": ["first"]},
        {
            "frame": "V1:0",
            "live": ["second"],
            "narrowed_by": "action_order",
            "why": "The later batch shows a different tool now being used.",
        },
        {"frame": "V1:1", "live": ["second"]},
    ]


def test_refresh_replace_updates_an_initially_clear_reference() -> None:
    rows = _refresh_switch_rows()
    curve = parse(
        _curve(rows),
        [r["frame"] for r in rows],
        ENTITIES,
        "task",
        task={"track": "refresh", "refresh": {"policy": "replace"}},
    )
    assert curve.settled_at == "V1:0"
    assert curve.final_live == ("second",)
    assert curve.tempting_roles == ("first",)
    assert curve.narrowing_cause == "action_order"


@pytest.mark.parametrize(
    "track,policy",
    [("graft", "replace"), ("clarify", "replace"), ("refresh", "retain")],
)
def test_reference_reset_is_only_for_refresh_replace(track: str, policy: str) -> None:
    rows = _refresh_switch_rows()
    with pytest.raises(DecisionError, match="came back to life"):
        parse(
            _curve(rows),
            [r["frame"] for r in rows],
            ENTITIES,
            "task",
            task={"track": track, "refresh": {"policy": policy}},
        )


def test_refresh_replace_requires_evidence_for_the_new_reference() -> None:
    rows = _refresh_switch_rows()
    rows[2].pop("why")
    with pytest.raises(DecisionError, match="needs narrowed_by and why"):
        parse(
            _curve(rows),
            [r["frame"] for r in rows],
            ENTITIES,
            "task",
            task={"track": "refresh", "refresh": {"policy": "replace"}},
        )


def test_refresh_replace_cannot_reset_inside_a_batch() -> None:
    rows = _refresh_switch_rows()
    rows[-1]["live"] = ["first"]
    with pytest.raises(DecisionError, match="came back to life"):
        parse(
            _curve(rows),
            [r["frame"] for r in rows],
            ENTITIES,
            "task",
            task={"track": "refresh", "refresh": {"policy": "replace"}},
        )
