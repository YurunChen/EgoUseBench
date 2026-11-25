from __future__ import annotations

import pytest

from egocua.construct_validity import (
    COUNTEREXAMPLE_GUIDANCE,
    POLICY,
    ConstructValidityError,
    admission_gate_statuses,
    combine_pass_gates,
)
from egocua.checker import check_run
from egocua.execution import RunContext
from egocua.protocol import RunSession
from sites.runtime import ActionTrace, SiteState
from tests.conditions import CONDITIONS
from tests.helpers import complete_run_record, item_id, site_data, task


def test_review_policy_routes_remaining_uncertainty_to_human() -> None:
    assert POLICY["routing"]["low_confidence"] == "human_review_pending"


def _evidence_statuses() -> dict[str, str]:
    return {
        "visual_evidence": "pass",
        "causal_contract": "pass",
        "information_partition": "pass",
        "site_mapping": "pass",
        "answer_boundary": "pass",
        "user_instruction": "pass",
    }


def test_effectiveness_requires_both_visual_evidence_and_site_mapping() -> None:
    statuses = _evidence_statuses()
    statuses["site_mapping"] = "fail"

    result = admission_gate_statuses(statuses)

    assert result == {
        "utility": "pass",
        "effectiveness": "fail",
        "continuous_change": "pass",
        "verifiability": "pass",
    }


def test_counterexample_failure_blocks_verifiability() -> None:
    checks = {
        "no_scene": "pass",
        "site_only": "pass",
        "static_substitute": "pass",
        "early_commit": "pass",
        "tempting_choice": "pass",
        "wrong_state": "fail",
        "undeclared_equivalent": "pass",
        "continuous_interval": "pass",
    }

    result = admission_gate_statuses(
        _evidence_statuses(),
        counterexample_statuses=checks,
    )

    assert result["verifiability"] == "fail"


def test_two_passes_cannot_average_away_one_failed_gate() -> None:
    first = {
        "utility": "pass",
        "effectiveness": "pass",
        "continuous_change": "pass",
        "verifiability": "pass",
    }
    second = {**first, "continuous_change": "fail"}

    assert combine_pass_gates([first, second])["continuous_change"] == "fail"


def test_unknown_evidence_dimension_is_rejected() -> None:
    statuses = _evidence_statuses()
    statuses["extra"] = "pass"

    with pytest.raises(ConstructValidityError, match="malformed"):
        admission_gate_statuses(statuses)


@pytest.mark.parametrize(
    ("track", "transition", "expected"),
    [
        ("graft", "refine", (2, False, True)),
        ("refresh", "retain", (0, False, True)),
        ("refresh", "replace", (2, True, True)),
        ("clarify", "resolve", (2, True, True)),
    ],
)
def test_early_commit_challenge_uses_runtime_information_boundary(
    track: str,
    transition: str,
    expected: tuple[int, bool, bool],
) -> None:
    definition = task(track)
    initial_id = item_id(track, 0)
    if transition == "retain":
        definition["refresh"]["policy"] = "retain"
        definition["instructions"]["base"] = (
            "Keep the first item despite the later view"
        )
        definition["goal"]["acceptable_states"][0]["selected_entity_ids"] = [initial_id]
    trace = ActionTrace()
    state = SiteState(site_data(track), trace=trace)
    session = RunSession(
        definition, CONDITIONS["online"], event_step=2, reference_steps=None
    )
    context = RunContext(
        session.initial_payload(), "", state.data, session, state, trace, 8
    )
    context.advance()
    state.select(initial_id)
    if transition == "retain":
        state.commit()
    context.advance()
    if track == "clarify":
        context.ask_user("Which variant do you mean?")
    final_id = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    if final_id != initial_id:
        state.deselect(initial_id)
        state.select(final_id)
    if transition != "retain":
        state.commit()
    record = complete_run_record(
        {
            "condition": CONDITIONS["online"].as_dict(),
            "run_status": {"valid": True, "reason": None},
            "steps": trace.current_step,
            "actions": trace.actions,
            "final_state": state.final_state(),
            "meta": {"default_release_step": 2, "reference_steps": None},
        }
    )

    checked = check_run(definition, record)

    assert (
        checked["sufficiency_step"],
        checked["premature_commitment"],
        checked["e2e_success"],
    ) == expected
    assert checked["premature_submission"] is False


def test_review_guidance_does_not_demand_blanket_early_rejection() -> None:
    assert "valid A-side commitments" in COUNTEREXAMPLE_GUIDANCE["early_commit"]
    assert (
        "removing V1 need not change the answer"
        in (COUNTEREXAMPLE_GUIDANCE["static_substitute"])
    )


def test_relax_review_compares_legal_sets_not_a_required_choice_change() -> None:
    guidance = COUNTEREXAMPLE_GUIDANCE["static_substitute"]
    assert "removing B must remove newly legal states" in guidance
    assert "every originally legal state remains accepted after B" in guidance
    assert "rather than demand a changed model choice" in guidance


def test_tempting_choice_review_does_not_require_artificial_neighbors() -> None:
    guidance = COUNTEREXAMPLE_GUIDANCE["tempting_choice"]
    assert "Do not invent neighbors" in guidance
    assert "task-specific applicability evidence in answer_boundary" in guidance
    assert "use uncertain when applicability cannot be established" in guidance
    assert "A state accepted after Relax is not a distractor" in guidance


def test_no_scene_shortcut_needs_one_guaranteed_complete_legal_state_not_all_answers() -> None:
    guidance = COUNTEREXAMPLE_GUIDANCE["no_scene"]
    assert "at least one legal complete final state" in guidance
    assert "Recovering every accepted state is not necessary" in guidance
    assert "One legal entity alone is not enough" in guidance
    assert "A guessed accepted answer or a unique title-keyword hit is not proof" in guidance
