from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from egocua.adapters.openai_requests import (
    OpenAIRequestsTransport,
    OpenAITransportError,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict):
        self.response = _FakeResponse(payload)
        self.calls: list[dict] = []

    def post(self, url: str, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


class _FailingSession:
    def post(self, url: str, **kwargs):
        response = requests.Response()
        response.status_code = 401
        response._content = b"gateway rejected secret-key"
        raise requests.HTTPError(response=response)


class OpenAIRequestsTransportTest(unittest.TestCase):
    def test_translates_multimodal_messages_tools_and_tool_calls(self) -> None:
        session = _FakeSession(
            {
                "choices": [
                    {
                        "message": {
                            "content": "I will inspect the item.",
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "type": "function",
                                    "function": {
                                        "name": "open_product",
                                        "arguments": '{"item_id":"item-02"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )
        transport = OpenAIRequestsTransport(
            api_key="secret-key",
            base_url="https://gateway.example/v1",
            session=session,
        )
        response = transport(
            {
                "model": "gpt-5.4-mini",
                "max_tokens": 512,
                "system": "Act carefully.",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Find the item."},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": "YWJj",
                                },
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "call-1",
                                "name": "search",
                                "input": {"query": "snack"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call-1",
                                "content": [
                                    {"type": "text", "text": "One result."},
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": "ZGVm",
                                        },
                                    },
                                ],
                            }
                        ],
                    },
                ],
                "tools": [
                    {
                        "name": "search",
                        "description": "Search the products.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    }
                ],
            }
        )

        call = session.calls[0]
        self.assertEqual(call["url"], "https://gateway.example/v1/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], "Bearer secret-key")
        self.assertEqual(call["json"]["max_completion_tokens"], 512)
        self.assertEqual(
            [message["role"] for message in call["json"]["messages"]],
            ["system", "user", "assistant", "tool", "user"],
        )
        self.assertEqual(
            call["json"]["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,YWJj",
        )
        self.assertEqual(
            call["json"]["messages"][4]["content"][2]["image_url"]["url"],
            "data:image/png;base64,ZGVm",
        )
        self.assertEqual(
            call["json"]["messages"][4]["content"][1]["text"],
            "One result.",
        )
        self.assertEqual(call["json"]["tools"][0]["type"], "function")
        self.assertEqual(response["content"][0]["type"], "text")
        self.assertEqual(
            response["content"][1],
            {
                "type": "tool_use",
                "id": "call-2",
                "name": "open_product",
                "input": {"item_id": "item-02"},
            },
        )

    def test_loads_key_and_url_from_the_selected_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "OPENAI_API_KEY=file-key\n"
                "OPENAI_BASE_URL=https://local.example/v1/chat/completions\n"
            )
            with patch.dict(os.environ, {}, clear=True):
                transport = OpenAIRequestsTransport.from_env(env_path)

        self.assertEqual(transport.api_key, "file-key")
        self.assertEqual(
            transport.endpoint,
            "https://local.example/v1/chat/completions",
        )

    def test_missing_env_configuration_fails_before_a_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(
                    OpenAITransportError,
                    "OPENAI_API_KEY and OPENAI_BASE_URL",
                ):
                    OpenAIRequestsTransport.from_env(env_path)

    def test_provider_error_does_not_expose_the_api_key(self) -> None:
        transport = OpenAIRequestsTransport(
            api_key="secret-key",
            base_url="https://gateway.example/v1",
            session=_FailingSession(),
        )
        with self.assertRaises(OpenAITransportError) as raised:
            transport({"model": "gpt-5.4-mini", "messages": []})

        self.assertNotIn("secret-key", str(raised.exception))
        self.assertIn("[REDACTED]", str(raised.exception))

    def test_network_error_keeps_reason_without_credentials(self) -> None:
        transport = OpenAIRequestsTransport(api_key="secret-key", base_url="https://gateway.example/v1")
        with patch.object(transport.session, "post", side_effect=requests.ConnectionError(
            "RemoteDisconnected via http://name:password@localhost:8080 secret-key"
        )):
            with self.assertRaises(OpenAITransportError) as raised:
                transport({"model": "test", "messages": []})
        detail = str(raised.exception)
        self.assertIn("RemoteDisconnected", detail)
        self.assertNotIn("password", detail)
        self.assertNotIn("secret-key", detail)


if __name__ == "__main__":
    unittest.main()
