import copy

import pytest

from egocua.adapters.openai_requests import OpenAITransportError
from egocua.adapters.openai_responses import OpenAIResponsesTransport
from egocua.attribution_judge import JudgeConfig
from egocua.attribution_batch import judge_transport


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class Session:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append({"url": url, **copy.deepcopy(kwargs)})
        return Response(next(self.outputs))


def test_responses_preserves_tool_reasoning_and_delivers_images_without_archiving_it():
    reasoning = {"type": "reasoning", "id": "reason-1", "encrypted_content": "opaque"}
    call = {"type": "function_call", "call_id": "call-1", "name": "read_frames",
            "arguments": '{"packet_id":"stream:0","start":0,"count":1}'}
    session = Session([
        {"output": [reasoning, call], "usage": {"input_tokens": 10}},
        {"output": [{"type": "message", "content": [
            {"type": "output_text", "text": '{"stage":null}'}]}]},
    ])
    transport = OpenAIResponsesTransport("secret", "https://gateway.example/v1", session=session)
    request = {"model": "gpt-6.1-terra", "system": "Read only.", "max_tokens": 100,
               "tools": [{"name": "read_frames", "description": "Read frames.",
                          "input_schema": {"type": "object"}}],
               "messages": [{"role": "user", "content": [{"type": "text", "text": "Review."}]}]}
    first = transport(request)
    assert first["content"][0]["id"] == "call-1"
    assert "opaque" not in str(first)
    assert first["raw"]["usage"]["input_tokens"] == 10
    request["messages"].extend([
        {"role": "assistant", "content": first["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call-1",
         "content": [{"type": "text", "text": "Frame zero, released at step 0."},
                     {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "YWJj"}}]}]},
    ])
    second = transport(request)
    payload = session.calls[1]["json"]
    assert session.calls[0]["url"] == "https://gateway.example/v1/responses"
    assert payload["store"] is False
    assert reasoning in payload["input"] and call in payload["input"]
    assert any(item.get("type") == "function_call_output" and item["call_id"] == "call-1"
               for item in payload["input"])
    assert payload["input"][-1]["content"][-1]["image_url"] == "data:image/jpeg;base64,YWJj"
    assert second["content"] == [{"type": "text", "text": '{"stage":null}'}]
    assert "opaque" not in str(second)


def test_responses_incomplete_output_is_an_error():
    transport = OpenAIResponsesTransport("secret", "https://gateway.example/v1",
                                        session=Session([{"status": "incomplete", "output": []}]))
    with pytest.raises(OpenAITransportError, match="did not complete"):
        transport({"model": "gpt-6.1-terra", "system": "Read only.", "max_tokens": 1,
                   "tools": [], "messages": [{"role": "user", "content": "Review."}]})


def test_judge_config_keeps_chat_default_and_rejects_unknown_api():
    base = {"model": "judge", "max_rounds": 1, "max_output_tokens": 10,
            "max_images_per_read": 1, "max_input_characters": 1000}
    assert JudgeConfig.from_dict(base).api == "chat_completions"
    assert JudgeConfig.from_dict(base).request_timeout_seconds == 180
    assert JudgeConfig.from_dict({**base, "request_timeout_seconds": 600}).request_timeout_seconds == 600
    assert JudgeConfig.from_dict({**base, "api": "responses"}).api == "responses"
    with pytest.raises(ValueError, match="Judge api"):
        JudgeConfig.from_dict({**base, "api": "unknown"})
    with pytest.raises(ValueError, match="request_timeout_seconds"):
        JudgeConfig.from_dict({**base, "request_timeout_seconds": 0})


def test_judge_transport_uses_selected_api_and_timeout(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(OpenAIResponsesTransport, "from_env",
                        lambda path, **kwargs: calls.append((path, kwargs)) or "transport")
    config = {"model": "judge", "api": "responses", "request_timeout_seconds": 600,
              "max_rounds": 1, "max_output_tokens": 10,
              "max_images_per_read": 1, "max_input_characters": 1000}
    env_path = tmp_path / ".env"
    assert judge_transport(config, env_path) == "transport"
    assert calls == [(env_path, {"timeout_seconds": 600})]
