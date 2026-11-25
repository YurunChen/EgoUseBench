"""Responses transport for one evidence-reading Judge conversation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .openai_requests import OpenAIRequestsTransport, OpenAITransportError


def _content(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for block in blocks:
        if block["type"] == "text":
            result.append({"type": "input_text", "text": block["text"]})
        elif block["type"] == "image":
            source = block["source"]
            result.append({
                "type": "input_image",
                "image_url": f"data:{source['media_type']};base64,{source['data']}",
            })
        else:
            raise OpenAITransportError(f"Unsupported Responses content: {block['type']}")
    return result


def _user_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = message["content"]
    if isinstance(blocks, str):
        return [{"role": "user", "content": [{"type": "input_text", "text": blocks}]}]
    items = []
    ordinary = [block for block in blocks if block["type"] != "tool_result"]
    if ordinary:
        items.append({"role": "user", "content": _content(ordinary)})
    for block in blocks:
        if block["type"] != "tool_result":
            continue
        content = block["content"]
        text = "\n".join(part["text"] for part in content if part["type"] == "text")
        items.append({
            "type": "function_call_output",
            "call_id": block["tool_use_id"],
            "output": text,
        })
        if any(part["type"] == "image" for part in content):
            items.append({"role": "user", "content": [
                {"type": "input_text", "text": f"Visual results for {block['tool_use_id']}"},
                *_content(content),
            ]})
    return items


def _response_blocks(output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    for item in output:
        if item["type"] == "function_call":
            blocks.append({
                "type": "tool_use", "id": item["call_id"], "name": item["name"],
                "input": json.loads(item["arguments"]),
            })
        elif item["type"] == "message":
            blocks.extend({"type": "text", "text": part["text"]}
                          for part in item["content"] if part["type"] == "output_text")
    if not blocks:
        raise OpenAITransportError("Responses returned neither text nor function calls")
    return blocks


@dataclass
class OpenAIResponsesTransport(OpenAIRequestsTransport):
    transport_name = "requests-responses"

    _history: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _message_count: int = field(default=0, init=False, repr=False)
    _last_blocks: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @property
    def endpoint(self) -> str:
        base = self.base_url.rstrip("/").removesuffix("/chat/completions")
        return base if base.endswith("/responses") else f"{base}/responses"

    def __call__(self, request_body: dict[str, Any]) -> dict[str, Any]:
        messages = request_body["messages"]
        if len(messages) <= self._message_count:
            raise OpenAITransportError("Responses conversation must append new messages")
        additions = []
        for message in messages[self._message_count:]:
            if message["role"] == "assistant":
                if not self._last_blocks or message["content"] != self._last_blocks:
                    raise OpenAITransportError("Unexpected assistant continuation")
            elif message["role"] == "user":
                additions.extend(_user_items(message))
            else:
                raise OpenAITransportError("Responses Judge expects user/tool messages")
        payload = {
            "model": request_body["model"], "instructions": request_body["system"],
            "input": [*self._history, *additions], "store": False,
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": request_body["max_tokens"],
            "tools": [{"type": "function", "name": tool["name"],
                       "description": tool["description"],
                       "parameters": tool["input_schema"], "strict": False}
                      for tool in request_body["tools"]],
        }
        response = self._post_payload(payload)
        if response.get("status") in {"failed", "incomplete"}:
            raise OpenAITransportError(f"Responses did not complete: {response.get('status')}")
        output = response["output"]
        blocks = _response_blocks(output)
        self._history = [*payload["input"], *output]
        self._message_count = len(messages)
        self._last_blocks = blocks
        return {"content": blocks, "raw": {
            "id": response.get("id"), "usage": response.get("usage", {}),
        }}
