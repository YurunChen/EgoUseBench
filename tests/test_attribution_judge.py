from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from egocua.attribution_judge import (
    JUDGE_PROMPT,
    PROMPT_VERSION,
    JudgeConfig,
    JudgeError,
    JudgeMaterials,
    TrajectoryJudge,
)
from egocua.failure_attribution import validate_evidence_references


@pytest.fixture
def context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    for name, colour in (
        ("frame.png", "red"),
        ("website.png", "green"),
        ("wrong.png", "blue"),
        ("correct.png", "yellow"),
    ):
        Image.new("RGB", (12, 12), colour).save(tmp_path / name)
    value = {
        "run_identity": {"run_id": "run-1", "task_id": "task-1"},
        "task_context": {
            "instructions": {"initial": "Save the tool I am using."},
            "reference": {"clarification": {"answer": "The tool on the left."}},
            "visual_packets": [
                {
                    "id": "V0",
                    "released_at_step": 0,
                    "frame_paths": ["frame.png"],
                    "timestamps_seconds": [11.0],
                }
            ],
            "visual_session": {"frame_paths": ["future.png"]},
        },
        "actual_initial_input": {
            "instruction": "Use the left tool; this was given upfront.",
            "clarification_available": True,
        },
        "condition": {"name": "online"},
        "expected_states": [{"selected_entity_ids": ["correct"], "fields": {}}],
        "scoring_goal_source": "checker_evaluator",
        "initial_state": {"selected_entity_ids": []},
        "actual_final_state": {"selected_entity_ids": ["wrong"]},
        "missing_evidence": [],
        "actions": [
            {"type": "search", "step": 1, "query": "red tool"},
            {"type": "open_entity", "step": 2, "entity_id": "wrong"},
            {
                "type": "select",
                "step": 3,
                "entity_id": "wrong",
                "state_before": {"selected_entity_ids": []},
                "state_after": {"selected_entity_ids": ["wrong"]},
            },
        ],
        "web_observations": [
            {
                "metadata": {"step_number": 2},
                "state": {
                    "title": "Details",
                    "screenshot_path": str(tmp_path / "website.png"),
                },
                "state_message": "Wrong tool",
                "actions": [{"click": {"index": 9}}],
                "result": [{"success": True}],
            }
        ],
        "site_context": {
            "site": "egomart",
            "schema_version": 1,
            "supported_fields": [],
            "entities": [
                {
                    "entity_id": "wrong",
                    "item_id": "wrong",
                    "name": "Wrong tool",
                    "attributes": {"material": "plastic"},
                    "image_path": "wrong.png",
                    "image_provenance": {"audit": "not public"},
                },
                {
                    "entity_id": "correct",
                    "item_id": "correct",
                    "name": "Correct tool",
                    "attributes": {"material": "steel"},
                    "image_path": "correct.png",
                },
            ],
        },
    }
    monkeypatch.setattr(
        "egocua.attribution_judge.historical_review_context",
        lambda row, root: deepcopy(value),
    )
    return value


@pytest.fixture
def row() -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "task_id": "task-1",
        "run_verdict": "valid_model_failure",
        "checker_version": "3.1",
    }


def config(**changes: Any) -> JudgeConfig:
    return JudgeConfig.from_dict(
        {
            "model": "offline-fake-judge",
            "max_output_tokens": 1000,
            "max_input_characters": 100000,
            **changes,
        }
    )


def judgment(**changes: Any) -> dict[str, Any]:
    return {
        "stage": "L",
        "step": 3,
        "confidence": "high",
        "reason": "The selected plastic tool contradicts the required steel tool and remains selected.",
        "evidence": [{"kind": "action", "index": 2}, {"kind": "entity", "id": "wrong"}],
        "uncertainty": None,
        **changes,
    }


def decision(**changes: Any) -> dict[str, Any]:
    return {
        "stage": "L", "error_event_id": "action:2", "confidence": "high",
        "reason": judgment()["reason"],
        "evidence_ids": ["action:2", "entity:wrong"], "uncertainty": None,
        **changes,
    }


class ScriptedTransport:
    def __init__(self, *outputs: Any) -> None:
        self.outputs = iter(outputs)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(deepcopy(request))
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        blocks = output if isinstance(output, list) else [{"type": "text", "text": json.dumps(output)}]
        return {
            "content": blocks,
            "raw": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
        }


def _text_blocks(packet: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [json.loads(block["text"]) for block in packet if block["type"] == "text"]


def test_complete_packet_contains_task_answer_and_full_timeline(
    context: dict, row: dict, tmp_path: Path
) -> None:
    packet = JudgeMaterials(row, tmp_path).complete_packet()
    texts = _text_blocks(packet)
    header = texts[0]

    assert header["task_reference_not_agent_input"]["reference"]["clarification"]["answer"] == "The tool on the left."
    assert header["answer_and_scoring_goal"]["acceptable_final_states"] == context["expected_states"]
    assert header["actual_final_state"] == context["actual_final_state"]
    events = [block for block in texts if block.get("timeline_event")]
    assert [event["timeline_event"] for event in events] == [
        "delivered_frames", "action", "web_observation", "action", "action"
    ]
    assert events[0]["frames"][0]["evidence_id"] == "frame:0:0"
    assert events[-1]["record"] == context["actions"][-1]
    assert "future.png" not in json.dumps(packet)
    assert "image_provenance" not in json.dumps(packet)


def test_complete_packet_includes_all_historical_images(
    context: dict, row: dict, tmp_path: Path
) -> None:
    packet = JudgeMaterials(row, tmp_path).complete_packet()
    images = [block for block in packet if block["type"] == "image"]

    # Delivered frame, webpage screenshot, selected entity, and accepted answer entity.
    assert len(images) == 4
    assert all(block["source"]["data"] for block in images)


def test_judge_receives_one_complete_packet_without_tools(
    context: dict, row: dict, tmp_path: Path
) -> None:
    transport = ScriptedTransport(decision())
    result = TrajectoryJudge(config(), transport).assess(row, tmp_path)

    assert result["judgment"] == judgment()
    assert result["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    assert len(result["request_log"]) == 1
    assert result["request_log"][0]["elapsed_seconds"] >= 0
    assert result["request_log"][0]["input_summary"]["unique_images"] == 4
    assert transport.requests[0]["tools"] == []
    assert len(transport.requests[0]["messages"]) == 1
    assert len([block for block in transport.requests[0]["messages"][0]["content"] if block["type"] == "image"]) == 4
    assert "image_provenance" not in json.dumps(transport.requests)
    assert '"data":' not in json.dumps(result["request_log"][0]["request"])


def test_state_only_unknown_is_allowed_after_full_packet(
    context: dict, row: dict, tmp_path: Path
) -> None:
    output = decision(
        stage=None,
        error_event_id=None,
        confidence="low",
        uncertainty="evidence_gap",
        evidence_ids=["state:actual_final_state"],
    )
    result = TrajectoryJudge(config(), ScriptedTransport(output)).assess(row, tmp_path)
    assert result["judgment"]["stage"] is None


def test_scoring_goal_is_citable_context(
    context: dict, row: dict, tmp_path: Path
) -> None:
    output = decision(evidence_ids=["action:2", "entity:wrong", "goal:acceptable_final_states"])
    result = TrajectoryJudge(config(), ScriptedTransport(output)).assess(row, tmp_path)
    assert result["judgment"]["evidence"][-1] == {
        "kind": "goal", "field": "acceptable_final_states"
    }


def test_prompt_uses_actions_to_resolve_visible_candidate_choices() -> None:
    assert "selecting the other" in JUDGE_PROMPT
    assert "intent-resolution error as U" in JUDGE_PROMPT
    assert "reference\nanswers, not messages delivered" in JUDGE_PROMPT


@pytest.mark.parametrize(
    "ref",
    [
        {"kind": "action", "index": 100},
        {"kind": "action", "index": -1},
        {"kind": "action", "index": True},
        {"kind": "web", "index": 0, "observed_step": 1},
        {"kind": "frame", "packet_index": 0, "index": 0, "observed_step": 1},
        {"kind": "frame", "packet_index": 1, "index": 0},
        {"kind": "entity", "id": "not-in-this-run"},
        {"kind": "state", "field": "final_state"},
        {"kind": "state", "index": 0, "field": "state_after"},
        {"kind": "action", "index": 0, "value": {"step": 900}},
    ],
)
def test_rejects_wrong_or_future_references(context: dict, ref: dict) -> None:
    with pytest.raises(ValueError):
        validate_evidence_references(context, [ref])


def test_complete_packet_missing_recorded_image_stops_before_model_call(
    context: dict, row: dict, tmp_path: Path
) -> None:
    context["web_observations"][0]["state"]["screenshot_path"] = "missing.png"
    transport = ScriptedTransport()
    with pytest.raises(JudgeError, match="missing historical website screenshot"):
        TrajectoryJudge(config(), transport).assess(row, tmp_path)
    assert transport.requests == []


def test_input_budget_does_not_truncate_or_send(
    context: dict, row: dict, tmp_path: Path
) -> None:
    transport = ScriptedTransport()
    with pytest.raises(JudgeError, match="exceeds input budget"):
        TrajectoryJudge(config(max_input_characters=10), transport).assess(row, tmp_path)
    assert transport.requests == []


def test_tool_call_is_a_processing_error(
    context: dict, row: dict, tmp_path: Path
) -> None:
    tool_call = [{"type": "tool_use", "id": "x", "name": "read_steps", "input": {}}]
    with pytest.raises(JudgeError, match="without tool calls"):
        TrajectoryJudge(config(), ScriptedTransport(tool_call)).assess(row, tmp_path)


def test_transport_error_not_unknown_or_retry(
    context: dict, row: dict, tmp_path: Path
) -> None:
    transport = ScriptedTransport(ConnectionError("service disconnected"))
    with pytest.raises(JudgeError, match="service disconnected") as caught:
        TrajectoryJudge(config(), transport).assess(row, tmp_path)
    assert len(transport.requests) == 1
    assert len(caught.value.request_log) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "other"},
        {"mechanism": "arbitrary"},
        {"confidence": "0.9"},
        {"error_event_id": "action:99"},
        {"uncertainty": "evidence_gap"},
        {"reason": ""},
    ],
)
def test_malformed_conclusion_is_failed_processing(
    context: dict, row: dict, tmp_path: Path, changes: dict
) -> None:
    with pytest.raises(JudgeError):
        TrajectoryJudge(config(), ScriptedTransport(decision(**changes))).assess(row, tmp_path)


@pytest.mark.parametrize("value", [None, "", " "])
def test_model_must_be_explicit(value: Any) -> None:
    with pytest.raises(ValueError, match="Select a multimodal Judge"):
        config(model=value)


def test_success_is_not_sent_to_judge(context: dict, row: dict, tmp_path: Path) -> None:
    transport = ScriptedTransport()
    with pytest.raises(ValueError, match="valid model failure"):
        TrajectoryJudge(config(), transport).assess({**row, "run_verdict": "success"}, tmp_path)
    assert transport.requests == []


def test_judge_output_contains_exactly_six_fields(
    context: dict, row: dict, tmp_path: Path
) -> None:
    result = TrajectoryJudge(config(), ScriptedTransport(decision())).assess(row, tmp_path)
    assert set(result["judgment"]) == {
        "stage", "step", "confidence", "reason", "evidence", "uncertainty"
    }
    assert result["prompt_version"] == PROMPT_VERSION == "trajectory-judge-v16"


def test_prompt_and_human_rubric_share_all_five_definitions() -> None:
    from egocua.failure_attribution import PULSE_LABELS, PULSE_STAGES

    assert tuple(PULSE_LABELS) == PULSE_STAGES
    for stage, rule in PULSE_LABELS.items():
        assert f"{stage} ({rule['name']}): {rule['criterion']}" in JUDGE_PROMPT
        assert rule["guidance"]
    assert "End-state" in PULSE_LABELS["E"]["name"]
    assert "Understanding" in PULSE_LABELS["U"]["name"]


def test_calibration_guards_actual_requirements_and_reference_contract() -> None:
    assert "Do not import additional constraints" in JUDGE_PROMPT
    assert "canonical clarification establishes the intended referent" in JUDGE_PROMPT
    assert "free choice only when the actual request permits free choice" in JUDGE_PROMPT
    assert "return task_validity rather than inventing" in JUDGE_PROMPT
    assert "not P" in JUDGE_PROMPT
    assert "do not invent task, input, instruction" in JUDGE_PROMPT
    assert "explicitly rejects a compatible candidate" in JUDGE_PROMPT


@pytest.mark.parametrize("stage", ["P", "U", "L", "S", "E"])
def test_all_five_judgments_preserve_cited_evidence(
    context: dict, row: dict, tmp_path: Path, stage: str
) -> None:
    refs = [{"kind": "action", "index": 2}, {"kind": "entity", "id": "wrong"}]
    if stage == "P":
        refs.append({"kind": "frame", "packet_index": 0, "index": 0})
    output = decision(stage=stage, evidence_ids=["action:2", "entity:wrong"] + (["frame:0:0"] if stage == "P" else []))
    result = TrajectoryJudge(config(), ScriptedTransport(output)).assess(row, tmp_path)
    assert result["judgment"] == judgment(stage=stage, evidence=refs)


def test_material_packet_separates_reference_answer_and_hides_model(
    context: dict, row: dict, tmp_path: Path
) -> None:
    context["run_identity"]["model"] = "evaluated-model-secret"
    header = _text_blocks(JudgeMaterials(row, tmp_path).complete_packet())[0]
    assert "evaluated-model-secret" not in json.dumps(header)
    assert header["actual_initial_input"]["instruction"] == context["actual_initial_input"]["instruction"]
    assert "clarification" in header["task_reference_not_agent_input"]["reference"]


def test_same_step_observation_precedes_executed_site_action(
    context: dict, row: dict, tmp_path: Path
) -> None:
    events = [x for x in _text_blocks(JudgeMaterials(row, tmp_path).complete_packet())
              if x.get("step") == 2]
    assert [x["timeline_event"] for x in events] == ["web_observation", "action"]
    assert events[0]["observation_phase"] == "before_action_at_this_step"


@pytest.mark.parametrize("missing", ["initial_instruction", "historical_frame:deleted.jpg"])
def test_recorded_material_gap_prevents_any_judge_request(
    context: dict, row: dict, tmp_path: Path, missing: str
) -> None:
    context["missing_evidence"] = [missing]
    transport = ScriptedTransport()
    with pytest.raises(JudgeError, match=missing):
        TrajectoryJudge(config(), transport).assess(row, tmp_path)
    assert transport.requests == []


def test_asset_inspection_matches_full_material_inputs_without_encoding(
    context: dict, tmp_path: Path
) -> None:
    from egocua.attribution_judge import inspect_materials

    assert inspect_materials(context, tmp_path) == {
        "missing_evidence": [], "historical_image_files": 4
    }
    (tmp_path / "wrong.png").unlink()
    assert "historical public entity image" in inspect_materials(context, tmp_path)["missing_evidence"][0]


def test_repeated_text_is_lossless_and_each_occurrence_keeps_its_step(
    context: dict, row: dict, tmp_path: Path
) -> None:
    repeated = "same fact " * 20 + "\r\n"
    original = repeated + "\r\n  whitespace\t\nlast line"
    context["web_observations"][0]["state_message"] = original
    second = deepcopy(context["web_observations"][0])
    second["metadata"]["step_number"] = 3
    second["state_message"] = original + "\nnew fact\n"
    context["web_observations"].append(second)
    packet = JudgeMaterials(row, tmp_path).complete_packet()
    definitions, restored, steps = {}, [], []
    for event in _text_blocks(packet):
        if event.get("timeline_event") != "web_observation":
            continue
        text = event["record"]["state_message"]
        assert not definitions.keys() & text["new_lines"].keys()
        definitions.update(text["new_lines"])
        restored.append("".join(part if isinstance(part, str) else definitions[str(part)] for part in text["parts"]))
        steps.append(event["step"])
    assert restored == [original, original + "\nnew fact\n"]
    assert steps == [2, 3]
    assert sum(value == repeated for value in definitions.values()) == 1


def test_equal_images_are_reused_without_dropping_observation_or_entity(
    context: dict, row: dict, tmp_path: Path
) -> None:
    (tmp_path / "wrong.png").write_bytes((tmp_path / "website.png").read_bytes())
    packet = JudgeMaterials(row, tmp_path).complete_packet()
    images = [block for block in packet if block["type"] == "image"]
    texts = _text_blocks(packet)
    assert len(images) == 3
    assert len([block for block in texts if "image_ref" in block]) == 1
    assert any(block.get("evidence_id") == "web:0" for block in texts)
    assert any(block.get("evidence_id") == "entity:wrong" for block in texts)


def test_scoring_goal_is_stored_once_but_different_snapshot_goal_is_retained(
    context: dict, row: dict, tmp_path: Path
) -> None:
    context["task_context"]["reference"]["goal"] = {
        "acceptable_states": deepcopy(context["expected_states"]), "target_granularity": "category"
    }
    materials = JudgeMaterials(row, tmp_path)
    header = _text_blocks(materials.complete_packet())[0]
    reference = header["task_reference_not_agent_input"]["reference"]["goal"]
    assert reference["acceptable_states"] == {"goal_ref": "goal:acceptable_final_states"}
    assert reference["target_granularity"] == "category"
    assert header["answer_and_scoring_goal"]["acceptable_final_states"] == context["expected_states"]
    assert context["task_context"]["reference"]["goal"]["acceptable_states"] == context["expected_states"]
    context["task_context"]["reference"]["goal"]["acceptable_states"] = [{"selected_entity_ids": ["old"]}]
    header = _text_blocks(JudgeMaterials(row, tmp_path).complete_packet())[0]
    assert header["task_reference_not_agent_input"]["reference"]["goal"]["acceptable_states"] == [{"selected_entity_ids": ["old"]}]


def test_decisive_event_maps_to_step_and_repeated_build_has_same_ids(
    context: dict, row: dict, tmp_path: Path
) -> None:
    materials = JudgeMaterials(row, tmp_path)
    first = materials.complete_packet()
    assert materials.complete_packet() == first
    output = decision(error_event_id="web:0", evidence_ids=["web:0", "entity:wrong"])
    result = materials.resolve_conclusion(output)
    assert result["step"] == 2
    assert result["evidence"][0] == {"kind": "web", "index": 0}
    assert "step" not in output


@pytest.mark.parametrize("changes", [
    {"error_event_id": "entity:wrong"},
    {"error_event_id": "web:0"},
    {"evidence_ids": ["web:999"]},
    {"evidence_ids": [False]},
    {"step": 3},
    {"error_event_id": None},
])
def test_decision_rejects_invalid_missing_or_non_event_ids(
    context: dict, row: dict, tmp_path: Path, changes: dict
) -> None:
    with pytest.raises(JudgeError):
        TrajectoryJudge(config(), ScriptedTransport(decision(**changes))).assess(row, tmp_path)


def test_perception_still_requires_previously_delivered_frame_ids(
    context: dict, row: dict, tmp_path: Path
) -> None:
    context["task_context"]["visual_packets"][0]["released_at_step"] = 4
    with pytest.raises(JudgeError, match="frames delivered before"):
        TrajectoryJudge(config(), ScriptedTransport(decision(
            stage="P", evidence_ids=["action:2", "frame:0:0"]
        ))).assess(row, tmp_path)


def test_unique_short_text_is_not_expanded_into_a_dictionary(
    context: dict, row: dict, tmp_path: Path
) -> None:
    message = "small unique message\n"
    context["web_observations"][0]["state_message"] = message
    events = _text_blocks(JudgeMaterials(row, tmp_path).complete_packet())
    web = next(event for event in events if event.get("timeline_event") == "web_observation")
    assert web["record"]["state_message"] == message
