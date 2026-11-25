from __future__ import annotations

import pytest

from tools.catalog.answer_closure_reviews import (
    ClosureReviewError,
    _accept,
    _reject,
    _validated_decision,
)


def _resolution() -> dict:
    return {
        "core_neighbor_entity_ids": ["core-1"],
        "related_entity_ids": ["rel-1", "rel-2"],
        "ambient_entity_ids": ["amb-1", "amb-2"],
        "rejected": [{"entity_id": "core-1", "ring": "core", "reason": "x"}],
    }


def test_accept_joins_role_without_padding_related_ring() -> None:
    resolution = _resolution()
    blueprint = {"environment_role_ids": {"jar": "answer"}, "equivalent_role_ids": {}}
    _accept(resolution, blueprint, "rel-1", "jar")
    assert blueprint["equivalent_role_ids"] == {"jar": ["rel-1"]}
    assert resolution["related_entity_ids"] == ["rel-2"]
    assert resolution["ambient_entity_ids"] == ["amb-1", "amb-2"]


def test_revised_core_answer_preserves_old_rejection_in_history() -> None:
    blueprint = {"environment_role_ids": {"jar": "answer"}}
    resolution = _resolution()
    old_rejection = resolution["rejected"][0].copy()
    _accept(resolution, blueprint, "core-1", "jar")
    assert resolution["core_neighbor_entity_ids"] == []
    assert resolution["rejected"] == []
    assert resolution["rejection_history"] == [old_rejection]
    assert blueprint["equivalent_role_ids"] == {"jar": ["core-1"]}


def test_reject_records_the_ring_and_replaces_an_earlier_row() -> None:
    resolution = _resolution()
    decision = {
        "reason": "两只装，用户要六只装",
        "failure_predicates": ["website_detail_mismatch"],
    }

    _reject(resolution, "amb-1", decision, ["jar", "3oz"])
    _reject(resolution, "amb-1", decision, ["jar", "3oz"])

    rows = [row for row in resolution["rejected"] if row["entity_id"] == "amb-1"]
    assert len(rows) == 1
    assert rows[0]["ring"] == "ambient"
    assert "jar, 3oz" in rows[0]["why_plausible"]


@pytest.mark.parametrize(
    "decision",
    [
        {
            "decision": "reject",
            "reason": "排名靠后",
            "failure_predicates": ["old_goal_only"],
        },
        {
            "decision": "reject",
            "reason": "两只装",
            "failure_predicates": ["not_a_predicate"],
        },
        {"decision": "accept", "role": "unknown", "reason": "同款"},
        {"decision": "accept", "role": "jar", "reason": " "},
    ],
)
def test_invalid_decisions_are_refused(decision: dict) -> None:
    with pytest.raises(ClosureReviewError):
        _validated_decision("amb-1", decision, {"jar"})


def test_earlier_role_answer_stays_a_classified_final_non_answer() -> None:
    resolution = _resolution()
    blueprint = {
        "environment_role_ids": {"first": "a", "later": "b"},
        "final_goal_roles": [["later"]],
    }

    _accept(resolution, blueprint, "core-1", "first")
    _accept(resolution, blueprint, "x-new", "first")

    assert blueprint["equivalent_role_ids"] == {"first": ["core-1", "x-new"]}
    assert resolution["core_neighbor_entity_ids"] == ["core-1"]
    assert "x-new" in resolution["related_entity_ids"]
    rows = {row["entity_id"]: row for row in resolution["rejected"]}
    assert rows["core-1"]["failure_predicates"] == ["later_change_mismatch"]
    assert rows["core-1"]["why_plausible"]
    assert resolution["rejection_history"][0]["reason"] == "x"


def test_round_listings_are_only_first_returned_by_matching_rounds() -> None:
    from tools.catalog.answer_closure_reviews import round_listings

    resolution = {
        "query_rounds": [
            {"stage": "seed", "result_entity_ids": ["a", "b"]},
            {"stage": "broad_20261031c", "result_entity_ids": ["b", "c", "d"]},
            {"stage": "more_20261031c", "result_entity_ids": ["d", "e"]},
        ]
    }

    assert round_listings(resolution, "_20261031c") == ["c", "d", "e"]


def test_insufficient_public_evidence_is_a_separate_rejection() -> None:
    decision = {
        "decision": "reject",
        "reason": "listing never states the pack count",
        "failure_predicates": ["insufficient_public_evidence"],
    }
    assert _validated_decision("amb-1", decision, {"jar"})["decision"] == "reject"


def _two_role_blueprint() -> dict:
    return {
        "environment_role_ids": {"first": "a", "later": "b"},
        "equivalent_role_ids": {"first": ["x"], "later": ["x"]},
        "final_goal_roles": [["later"]],
    }


def test_exact_roles_withdraw_a_final_accept_and_keep_the_earlier_one() -> None:
    from tools.catalog.answer_closure_reviews import _apply_decision

    resolution = {"window_entity_ids": ["x"], "rejected": []}
    blueprint = _two_role_blueprint()
    decision = _validated_decision(
        "x", {"decision": "accept", "roles": ["first"], "reason": "只符合最初请求"}, {"first", "later"}
    )

    _apply_decision(resolution, blueprint, "x", decision, [])

    assert blueprint["equivalent_role_ids"] == {"first": ["x"]}
    assert resolution["related_entity_ids"] == ["x"]
    assert resolution["rejected"][0]["failure_predicates"] == ["later_change_mismatch"]


def test_rejecting_an_accepted_listing_requires_withdraw() -> None:
    from tools.catalog.answer_closure_reviews import _apply_decision

    raw = {"decision": "reject", "reason": "整份食谱，没有过筛步骤", "failure_predicates": ["wrong_method_same_object"]}
    resolution = {"window_entity_ids": ["x"], "rejected": []}
    blueprint = _two_role_blueprint()
    with pytest.raises(ClosureReviewError):
        _apply_decision(resolution, blueprint, "x", _validated_decision("x", raw, {"first", "later"}), [])

    withdrawn = _validated_decision("x", {**raw, "withdraw": True}, {"first", "later"})
    _apply_decision(resolution, blueprint, "x", withdrawn, [])
    assert blueprint["equivalent_role_ids"] == {}
    assert resolution["related_entity_ids"] == ["x"]
    assert resolution["rejected"][0]["reason"] == raw["reason"]


def _scope_task(final: list[list[str]]) -> dict:
    return {
        "graft": {
            "initial_goal": {
                "acceptable_states": [{"selected_entity_ids": ["pan", "press"]}]
            }
        },
        "goal": {
            "acceptable_states": [
                {"selected_entity_ids": ids, "fields": {}} for ids in final
            ]
        },
    }


def test_local_scope_accepts_only_the_replaced_slot() -> None:
    from tools.catalog.answer_closure_reviews import (
        ClosureReviewError,
        add_scope_equivalent,
    )

    task = _scope_task([["pan", "steel"]])

    assert add_scope_equivalent(task, "iron", "item_2") == 1
    assert ["pan", "iron"] in [
        state["selected_entity_ids"] for state in task["goal"]["acceptable_states"]
    ]
    with pytest.raises(ClosureReviewError, match="scope contract"):
        add_scope_equivalent(task, "copper", "item_1")


def test_global_scope_accepts_either_slot() -> None:
    from tools.catalog.answer_closure_reviews import add_scope_equivalent

    task = _scope_task([["steel-pan", "steel-press"]])

    assert add_scope_equivalent(task, "iron-pan", "item_1") == 1
    assert add_scope_equivalent(task, "iron-press", "item_2") == 2
