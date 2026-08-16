"""OpenAI-compatible Chat Completions transport implemented with requests."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]
from dotenv import load_dotenv  # type: ignore[import-untyped]


DEFAULT_TIMEOUT_SECONDS = 180.0


class OpenAITransportError(RuntimeError):
    """Raised when model configuration or an OpenAI-compatible request is invalid."""


@dataclass
class OpenAIRequestsTransport:
    provider = "openai-compatible"
    transport_name = "requests-chat-completions"

    api_key: str = field(repr=False)
    base_url: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    session: Any = field(default_factory=requests.Session, repr=False)

    @classmethod
    def from_env(
        cls, env_path: str | Path = ".env", timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    ) -> OpenAIRequestsTransport:
        path = Path(env_path)
        if path.is_file():
            load_dotenv(path, override=False)
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not api_key or not base_url:
            raise OpenAITransportError(
                "OPENAI_API_KEY and OPENAI_BASE_URL must be set in the selected .env file "
                "or process environment"
            )
        return cls(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)

    @property
    def endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def __call__(self, request_body: dict[str, Any]) -> dict[str, Any]:
        payload = _request_payload(request_body)
        try:
            response = self.session.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            response_payload = response.json()
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "")
            detail = detail.replace(self.api_key, "[REDACTED]")
            suffix = f": {detail[:1000]}" if detail else ""
            raise OpenAITransportError(f"OpenAI-compatible request failed{suffix}") from exc
        except ValueError as exc:
            raise OpenAITransportError("OpenAI-compatible response was not valid JSON") from exc
        return _response_payload(response_payload)


def _request_payload(request_body: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": request_body["model"],
        "messages": _messages(request_body),
    }
    tools = request_body.get("tools", [])
    if tools:
        payload["tools"] = [_tool(tool) for tool in tools]
    max_tokens = request_body.get("max_tokens")
    if max_tokens is not None:
        payload["max_completion_tokens"] = max_tokens
    return payload


def _messages(request_body: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system = request_body.get("system")
    if system:
        messages.append({"role": "system", "content": system})
    for message in request_body.get("messages", []):
        role = message["role"]
        content = message.get("content", "")
        if role == "assistant":
            messages.append(_assistant_message(content))
        elif role == "user" and _contains_tool_results(content):
            messages.extend(_tool_result_messages(content))
        else:
            messages.append({"role": role, "content": _content_parts(content)})
    return messages


def _assistant_message(content: Any) -> dict[str, Any]:
    blocks = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
    text = "\n".join(
        str(block.get("text", ""))
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    tool_calls = [
        {
            "id": block["id"],
            "type": "function",
            "function": {
                "name": block["name"],
                "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
            },
        }
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    message: dict[str, Any] = {"role": "assistant", "content": text or None}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _contains_tool_results(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def _tool_result_messages(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    visual_parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") != "tool_result":
            continue
        result = block.get("content", "")
        messages.append({
            "role": "tool",
            "tool_call_id": block["tool_use_id"],
            "content": _text_content(result),
        })
        visual_parts.extend(_labeled_image_parts(result))
    if visual_parts:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Product images returned by the EgoMart tool:"},
                *visual_parts,
            ],
        })
    return messages


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    text = "\n".join(
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()
    return text or "Tool returned visual results."


def _content_parts(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False)
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            parts.append({"type": "text", "text": str(block)})
        elif block.get("type") == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif block.get("type") == "image":
            parts.append(_image_part(block))
    return parts


def _labeled_image_parts(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    parts: list[dict[str, Any]] = []
    for index, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") != "image":
            continue
        if index and isinstance(content[index - 1], dict):
            label = content[index - 1]
            if label.get("type") == "text":
                parts.append({"type": "text", "text": str(label.get("text", ""))})
        parts.append(_image_part(block))
    return parts


def _image_part(block: dict[str, Any]) -> dict[str, Any]:
    source = block.get("source", {})
    media_type = source.get("media_type", "image/jpeg")
    data = source.get("data", "")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{media_type};base64,{data}"},
    }


def _tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
        },
    }


def _response_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        message = payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise OpenAITransportError("OpenAI-compatible response has no assistant message") from exc

    blocks: list[dict[str, Any]] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        blocks.append({"type": "text", "text": content})
    elif isinstance(content, list):
        blocks.extend(
            {"type": "text", "text": str(part.get("text", ""))}
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError as exc:
            raise OpenAITransportError("Model returned invalid tool-call arguments") from exc
        if not isinstance(arguments, dict):
            raise OpenAITransportError("Model tool-call arguments must be a JSON object")
        blocks.append({
            "type": "tool_use",
            "id": call["id"],
            "name": function["name"],
            "input": arguments,
        })
    return {"content": blocks, "raw": payload}
