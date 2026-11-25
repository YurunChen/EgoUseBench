from __future__ import annotations

import pytest
import json

from tools.catalog.apply_core_neighbor_reviews import (
    _final_answer_ids,
    _reviewed_neighbors,
)
from tools.catalog.propose_core_neighbors import _parse


def _valid_neighbors() -> list[dict[str, object]]:
    return [
        {
            "entity_id": f"entity-{index}",
            "why_plausible": "It shares the target object and tutorial method.",
            "why_rejected": "The public listing shows the wrong variant.",
            "failure_predicates": [predicate],
        }
        for index, predicate in enumerate(
            [
                "wrong_variant_or_branch",
                "wrong_method_same_object",
                "website_detail_mismatch",
                "old_goal_only",
            ]
        )
    ]


@pytest.mark.parametrize(
    ("family", "target_fields", "expected"),
    [
        (
            "graft",
            {"final_goal_roles": [["base", "target"]]},
            {"base-id", "target-id", "target-alt"},
        ),
        (
            "refresh",
            {"final_target_role": "target"},
            {"target-id", "target-alt"},
        ),
        (
            "clarify",
            {"hidden_target_role": "target"},
            {"target-id", "target-alt"},
        ),
    ],
)
def test_final_answer_ids_uses_current_blueprint_equivalents(
    family: str,
    target_fields: dict[str, object],
    expected: set[str],
) -> None:
    record = {
        "task_id": "task",
        "family": family,
        "construct": {
            "task_blueprint": {
                "environment_role_ids": {
                    "base": "base-id",
                    "target": "target-id",
                },
                "equivalent_role_ids": {"target": ["target-alt"]},
                **target_fields,
            },
        },
    }

    assert (
        _final_answer_ids(
            record,
            ["base-id", "target-id", "target-alt", "distractor"],
        )
        == expected
    )


def test_final_answer_ids_ignores_incomplete_graft_states() -> None:
    record = {
        "task_id": "task",
        "family": "graft",
        "construct": {
            "task_blueprint": {
                "environment_role_ids": {
                    "base": "base-id",
                    "target": "target-id",
                },
                "equivalent_role_ids": {"target": ["target-alt"]},
                "final_goal_roles": [["base", "target"]],
            },
        },
    }

    assert _final_answer_ids(record, ["base-id", "target-alt"]) == {
        "base-id",
        "target-alt",
    }


def test_reviewed_neighbors_rejects_model_self_dialogue() -> None:
    rows = _valid_neighbors()
    rows[0]["why_rejected"] = "Wait, let's re-evaluate the answer."

    with pytest.raises(ValueError, match="invalid reviewed core neighbor"):
        _reviewed_neighbors("task", {"core_neighbors": rows}, None)


def test_reviewed_neighbors_rejects_ranking_as_rejection_evidence() -> None:
    rows = _valid_neighbors()
    rows[0]["why_rejected"] = "It ranked below the selected result."

    with pytest.raises(ValueError, match="invalid reviewed core neighbor"):
        _reviewed_neighbors("task", {"core_neighbors": rows}, None)


def test_proposal_and_review_accept_one_real_predicate_without_changing_assessment():
    neighbors = _valid_neighbors()
    for row in neighbors:
        row["failure_predicates"] = ["website_detail_mismatch"]
    assessment = {
        "useful_in_real_life": True,
        "scene_update_website_consistent": True,
        "continuous_change": True,
        "deterministically_verifiable": True,
        "issue": "",
    }
    proposed = _parse(
        json.dumps({"task_assessment": assessment, "core_neighbors": neighbors}),
        {"candidate_rows": neighbors},
    )
    reviewed = _reviewed_neighbors("task", proposed, None)
    assert proposed["task_assessment"] == assessment
    assert len(reviewed) == 4
    assert all(
        row["failure_predicates"] == ["website_detail_mismatch"] for row in reviewed
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("why_rejected", ""),
        ("failure_predicates", []),
        ("failure_predicates", ["invalid_predicate"]),
    ],
)
def test_single_predicate_still_requires_valid_evidence_for_every_neighbor(
    field, value
):
    neighbors = _valid_neighbors()
    for row in neighbors:
        row["failure_predicates"] = ["website_detail_mismatch"]
    neighbors[0][field] = value
    with pytest.raises(ValueError, match="invalid reviewed core neighbor"):
        _reviewed_neighbors("task", {"core_neighbors": neighbors}, None)


def test_review_accepts_supported_neighbors_without_a_fixed_count():
    assert (
        len(
            _reviewed_neighbors(
                "task", {"core_neighbors": _valid_neighbors()[:1]}, None
            )
        )
        == 1
    )
    assert _reviewed_neighbors("task", {"core_neighbors": []}, None) == []


def test_explicit_empty_override_does_not_reuse_old_proposals():
    assert (
        _reviewed_neighbors(
            "task",
            {"core_neighbors": _valid_neighbors()},
            {"replacement_core_neighbors": []},
        )
        == []
    )
