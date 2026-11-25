import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from tools.catalog import (
    landing_proposals,
    propose_amazon_landings,
    propose_youtube_landings,
)
from tools.catalog.landing_proposals import LandingProposals, role_sets


@pytest.fixture
def proposal_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://catalog.example/v1")
    post = Mock()
    monkeypatch.setattr(landing_proposals.requests, "post", post)
    return post


@pytest.mark.parametrize(
    "site,identity",
    [(propose_amazon_landings, "asin"), (propose_youtube_landings, "youtube_id")],
)
def test_site_proposals_save_once_and_reuse(
    tmp_path, proposal_environment, site, identity
):
    payload = {
        "task_id": "task-a",
        "role_requirements": [{"role": "target", "required_stage": "after_change"}],
        "before_results": [],
        "after_results": [{identity: "real-result"}],
    }
    answer = {
        "task_assessment": {
            "useful": True,
            "scene_update_website_consistent": True,
            "continuous_change": True,
            "verifiable_from_public_metadata": True,
            "issue": "",
        },
        "role_landings": [
            {"role": "target", identity: "real-result", "reason": "supported"}
        ],
    }
    proposal_environment.return_value.json.return_value = {
        "choices": [{"message": {"content": json.dumps(answer)}}]
    }
    worker = LandingProposals(site.OUTPUT, site._prompt, site._parse)

    assert worker.build(tmp_path, [payload], 2) == {
        "requested": 1,
        "reused": 0,
        "proposed": 1,
        "failed": 0,
    }
    saved = json.loads((tmp_path / site.OUTPUT).read_text())
    assert saved["tasks"]["task-a"]["role_landings"] == answer["role_landings"]
    assert saved["tasks"]["task-a"]["status"] == "proposed_pending_coding_agent_review"
    assert worker.build(tmp_path, [payload], 2)["reused"] == 1
    assert proposal_environment.call_count == 1
    args, kwargs = proposal_environment.call_args
    assert args == ("https://catalog.example/v1/chat/completions",)
    assert kwargs["json"]["messages"][1]["content"] == site._prompt(payload, "")


def test_invalid_output_feedback_is_preserved(tmp_path, proposal_environment):
    proposal_environment.return_value.json.side_effect = [
        {"choices": [{"message": {"content": "invalid"}}]},
        {"choices": [{"message": {"content": '{"accepted": true}'}}]},
    ]
    worker = LandingProposals(
        Path("proposals.json"), lambda payload, feedback: feedback, json_parser
    )

    assert worker.build(tmp_path, [{"task_id": "task-a"}], 1)["proposed"] == 1
    assert proposal_environment.call_count == 2
    assert (
        "Previous response failed validation"
        in proposal_environment.call_args.kwargs["json"]["messages"][1]["content"]
    )
    saved = json.loads((tmp_path / "proposals.json").read_text())
    assert saved["tasks"]["task-a"]["attempts"] == 2


def json_parser(text, payload):
    return json.loads(text)


def test_service_error_is_not_retried(tmp_path, proposal_environment):
    proposal_environment.return_value.raise_for_status.side_effect = requests.HTTPError(
        "503"
    )
    worker = LandingProposals(
        Path("proposals.json"), lambda payload, feedback: "prompt", json_parser
    )

    assert worker.build(tmp_path, [{"task_id": "task-a"}], 1)["failed"] == 1
    assert proposal_environment.call_count == 1
    assert not (tmp_path / "proposals.json").exists()


def test_failed_validation_is_not_saved(tmp_path, proposal_environment):
    proposal_environment.return_value.json.return_value = {
        "choices": [{"message": {"content": "invalid"}}]
    }
    worker = LandingProposals(
        Path("proposals.json"), lambda payload, feedback: "prompt", json_parser
    )

    assert worker.build(tmp_path, [{"task_id": "task-a"}], 1)["failed"] == 1
    assert proposal_environment.call_count == 3
    assert not (tmp_path / "proposals.json").exists()


@pytest.mark.parametrize(
    "transition,initial,final",
    [
        ("augment", ["a"], ["a", "b"]),
        ("drop", ["a", "b"], ["a"]),
        ("replace", ["a"], ["b"]),
    ],
)
def test_shared_role_sets_preserve_transition(transition, initial, final):
    task = {"id": "task-a", "track": "graft", "graft": {"transition": transition}}
    blueprint = {"initial_goal_roles": [initial], "final_goal_roles": [final]}
    assert role_sets(task, blueprint) == (set(initial), set(final))
