from __future__ import annotations

import pytest

from tools.catalog.apply_landing_reviews import (
    _replace_resolution_binding,
    _require_explicit_repair_override,
    _reviewed_landings,
)


def test_reviewed_binding_cannot_be_rolled_back_by_old_proposal() -> None:
    with pytest.raises(ValueError, match="old landing proposal"):
        _require_explicit_repair_override(
            task_id="task",
            role="target",
            current="repaired-id",
            proposed="old-id",
            prior_review_status="coding_agent_accepted",
            explicit_role_overrides=set(),
        )


def test_explicit_task_local_override_may_change_reviewed_binding() -> None:
    _require_explicit_repair_override(
        task_id="task",
        role="target",
        current="old-id",
        proposed="repaired-id",
        prior_review_status="coding_agent_accepted",
        explicit_role_overrides={"target"},
    )


def test_existing_real_window_preserves_old_listing_when_canonical_changes() -> None:
    resolution = {
        "window_entity_ids": ["old-id", "repaired-id", "neighbor"],
        "related_entity_ids": ["old-id", "neighbor"],
    }

    updated = _replace_resolution_binding(resolution, "old-id", "repaired-id")

    assert updated is resolution
    assert updated["window_entity_ids"] == ["old-id", "repaired-id", "neighbor"]


def test_role_repair_can_replace_stale_proposal_roles() -> None:
    proposal = {
        "role_landings": [
            {"role": "old_role", "asin": "OLD"},
        ]
    }
    override = {
        "replace_role_landings": True,
        "role_landings": {"new_role": "NEW"},
    }

    assert _reviewed_landings(proposal, override) == {"new_role": "NEW"}


def test_partial_role_override_keeps_unmodified_proposal_roles() -> None:
    proposal = {
        "role_landings": [
            {"role": "first", "asin": "OLD-1"},
            {"role": "second", "asin": "OLD-2"},
        ]
    }
    override = {"role_landings": {"second": "NEW-2"}}

    assert _reviewed_landings(proposal, override) == {
        "first": "OLD-1",
        "second": "NEW-2",
    }
