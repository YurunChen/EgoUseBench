"""BrowserGym integration uses a local scripted provider, never a paid model."""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
import re
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import httpx
from openai import APIConnectionError, APIStatusError, AsyncOpenAI
from PIL import Image
from pyparsing import ParseBaseException

from egocua.adapters.browsergym import (
    ModelCallStop,
    ModelInput,
    _parse_action,
    _request_with_retries,
    _retry_delay,
    is_request_transport_error,
    run_task as run_live_task,
    system_prompt,
)
from egocua.messages import STRATEGY_HINT, wearer_said
from egocua.adapters.browsergym_environment import BrowserWorkerError, action_set
from egocua.checker import check_run
from egocua.final_state import evaluate_final_state
from egocua.experiment_output import experiment_folder_name
from egocua.interaction_settings import SETTING_ORDER, setting_conditions
from egocua.timeline import ACTION_ALIGNED_REPLAY, COMPUTER_USE
from tests.conditions import CONDITIONS
from tests.helpers import item_id, materialize_site, products, task


async def run_task(definition, condition, **kwargs):
    """Exercise the adapter with the archived event-protocol fixtures below."""
    with patch(
        "egocua.adapters.browsergym.prepare_live_task",
        side_effect=lambda definition, _condition, _config: definition,
    ):
        return await run_live_task(definition, condition, **kwargs)


@pytest.fixture
def browser_assets(tmp_path):
    def create(track):
        definition = task(track)
        materialize_site(products(track), tmp_path)
        for packet in definition["visual_packets"]:
            for index, name in enumerate(packet["frame_paths"]):
                path = tmp_path / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (32, 24), (index * 20, 40, 80)).save(path)
        return definition

    return create


@pytest.fixture
def provider(monkeypatch):
    commands = []
    requests = []
    failures = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            headers = {}
            try:
                command = commands.pop(0)
                if isinstance(command, tuple):
                    status, headers = command
                    self.send_response(status)
                    encoded = b'{"error":{"message":"temporarily unavailable"}}'
                elif command is None:
                    self.send_response(200)
                    encoded = b"null"
                else:
                    content = command(body) if callable(command) else command
                    encoded = json.dumps(completion(content)).encode()
                    self.send_response(200)
            except Exception as error:
                failures.append(error)
                encoded = (
                    b'{"error":{"message":"offline script failed","type":"test_error"}}'
                )
                self.send_response(500)
            for name, value in headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    def completion(content):
        return {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "offline-script",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(content)
                        if isinstance(content, dict)
                        else content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    yield commands, requests
    server.shutdown()
    server.server_close()
    thread.join()
    assert not failures
    assert not commands


def test_ark_uses_shared_chat_interface(browser_assets, provider, tmp_path, monkeypatch):
    monkeypatch.setattr("egocua.adapters.browsergym.ARK_HOST", "127.0.0.1")
    definition = browser_assets("graft")
    commands, requests = provider
    commands.append({"action": "done", "parameters": {}})
    record = asyncio.run(run_task(
        definition, CONDITIONS["upfront"], model="doubao-seed-2-1-pro-260915",
        asset_root=tmp_path, max_steps=1, default_release_step=2,
        reference_steps=None, provider_seed=123,
    ))
    assert record["meta"]["provider"] == "volcengine-ark"
    assert record["trace"]["agent_settings"]["api"] == "chat_completions"
    assert requests[0]["model"] == "doubao-seed-2-1-pro-260915"
    assert requests[0]["seed"] == 123
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["messages"][0]["content"] == record["trace"]["agent_settings"]["system_prompt"]
    assert record["trace"]["usage"]["total_tokens"] == 15
    assert check_run(definition, record)["run_valid"] is True


def test_tpm_limit_waits_for_minute_window_instead_of_retrying_immediately():
    response = httpx.Response(429, request=httpx.Request("POST", "https://example.test"))
    error = APIStatusError("TPM exceeded", response=response,
                           body={"code": "ModelAccountTpmRateLimitExceeded"})
    assert _retry_delay(error, 0) == 60
    response.headers["Retry-After"] = "2"
    assert _retry_delay(error, 0) == 2


@pytest.mark.parametrize("index,slot,key", [(0, "primary", "offline-test-key"), (1, "secondary", "second-test-key"), (2, "tertiary", "third-test-key")])
def test_two_keys_are_assigned_per_task_without_changing_chat_request(
    browser_assets, provider, tmp_path, monkeypatch, index, slot, key
):
    monkeypatch.setenv("OPENAI_API_KEY_SECONDARY", "second-test-key")
    secondary_url = os.environ["OPENAI_BASE_URL"] + "/second-service"
    monkeypatch.setenv("OPENAI_BASE_URL_SECONDARY", secondary_url)
    third_url = os.environ["OPENAI_BASE_URL"] + "/third-service"
    monkeypatch.setenv("OPENAI_BASE_URL_TERTIARY", third_url)
    monkeypatch.setenv("OPENAI_API_KEY_TERTIARY", "third-test-key")
    definition = browser_assets("graft")
    commands, requests = provider
    commands.append({"action": "done", "parameters": {}})
    with patch("openai.AsyncOpenAI", wraps=AsyncOpenAI) as client:
        record = asyncio.run(run_task(
            definition, CONDITIONS["upfront"], model="offline-script", asset_root=tmp_path,
            max_steps=1, default_release_step=2, reference_steps=None, schedule_index=index,
        ))
    assert client.call_args.kwargs["api_key"] == key
    assert record["trace"]["agent_settings"]["service_slot"] == slot
    if slot == "secondary":
        assert record["meta"]["runtime"]["provider_base_url"] == secondary_url
    elif slot == "tertiary":
        assert record["meta"]["runtime"]["provider_base_url"] == third_url
    assert requests[0]["model"] == "offline-script"
    assert requests[0]["response_format"] == {"type": "json_object"}


def test_api_errors_retry_same_step(browser_assets, provider, tmp_path):
    definition = browser_assets("graft")
    commands, requests = provider

    commands.extend((status, {"Retry-After": "0"}) for status in (400, 429, 502))
    commands.append({"action": "done", "parameters": {}})
    record = asyncio.run(run_task(
        definition, CONDITIONS["upfront"], model="offline-script",
        asset_root=tmp_path, max_steps=1, default_release_step=2,
        reference_steps=None,
    ))
    assert len(requests) == 4
    assert all(request == requests[0] for request in requests)
    assert record["steps"] == 1 and len(record["trace"]["history"]) == 1
    request, = record["trace"]["llm_requests"]
    assert request["success"]
    assert [retry["attempt"] for retry in request["retries"]] == [1, 2, 3]
    assert request["retries"][1]["delay_seconds"] == 0
    checked = check_run(definition, record)
    assert checked["run_valid"], checked["run_invalid_reasons"]


def test_connection_deadline_does_not_shorten_model_generation(browser_assets, provider, tmp_path):
    commands, _ = provider
    commands.append({"action": "done", "parameters": {}})
    with patch("openai.AsyncOpenAI", wraps=AsyncOpenAI) as constructor:
        record = asyncio.run(run_task(
            browser_assets("graft"), CONDITIONS["upfront"], model="offline-script",
            asset_root=tmp_path, max_steps=1, default_release_step=2,
            reference_steps=None,
        ))
    timeout = constructor.call_args.kwargs["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 15 and timeout.read == 300
    assert record["trace"]["agent_settings"]["llm_connect_timeout"] == 15


def test_request_progress_reports_attempts_without_input_text(monkeypatch, capsys):
    failure = APIConnectionError(request=httpx.Request("POST", "https://offline.invalid"))
    send = AsyncMock(side_effect=[failure, SimpleNamespace(choices=[object()])])
    monkeypatch.setattr("egocua.adapters.browsergym._retry_delay", lambda *args: 0)
    logged = {"task_id": "progress-case", "model": "offline-model", "step": 7,
              "input": {"secret": "must-not-appear-in-progress"}}
    asyncio.run(_request_with_retries(send, logged, ModelCallStop()))
    output = capsys.readouterr().out
    events = [json.loads(line) for line in output.splitlines()]
    assert [row["model_request"] for row in events] == ["started", "retry", "started", "completed"]
    assert all(row["task_id"] == "progress-case" and row["step"] == 7 for row in events)
    assert events[-1]["attempt"] == 2 and logged["elapsed_seconds"] >= 0
    assert "must-not-appear-in-progress" not in output


def test_null_provider_response_retries_same_step(browser_assets, provider, tmp_path):
    definition = browser_assets("graft")
    commands, requests = provider
    commands.extend([None, {"action": "done", "parameters": {}}])
    with patch("egocua.adapters.browsergym._retry_delay", return_value=0):
        record = asyncio.run(run_task(
            definition, CONDITIONS["upfront"], model="offline-script",
            asset_root=tmp_path, max_steps=1, default_release_step=2,
            reference_steps=None,
        ))
    assert len(requests) == 2 and requests[0] == requests[1]
    assert record["steps"] == 1
    request, = record["trace"]["llm_requests"]
    assert request["success"] and len(request["retries"]) == 1
    assert request["retries"][0]["error"]["type"] == "RemoteProtocolError"


def test_null_provider_response_exhausts_retries_without_stopping_batch():
    send = AsyncMock(return_value=None)
    logged = {}
    stop = ModelCallStop()
    with patch("egocua.adapters.browsergym.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(httpx.RemoteProtocolError, match="empty response") as caught:
            asyncio.run(_request_with_retries(send, logged, stop))
    assert is_request_transport_error(caught.value)
    assert send.await_count == 4
    assert len(logged["retries"]) == 3
    assert stop.error is None


@pytest.mark.parametrize("reason, kind", [
    ("data_inspection_failed", "moderation"),
    ("maximum context length is 262144", "context_limit"),
])
def test_provider_input_rejection_preserves_request_and_other_runs(
    browser_assets, tmp_path, monkeypatch, reason, kind,
):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(400, json={"error": {"message": reason}})

    def client(**options):
        return AsyncOpenAI(http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(respond)), **options)

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://offline.invalid/v1")
    monkeypatch.setattr("openai.AsyncOpenAI", client)
    monkeypatch.setattr("egocua.adapters.browsergym._retry_delay", lambda *args: 0)
    stop = ModelCallStop()
    with pytest.raises(APIStatusError) as caught:
        asyncio.run(run_task(
            browser_assets("graft"), CONDITIONS["upfront"], model="offline-script",
            asset_root=tmp_path, output_root=tmp_path, max_steps=1,
            default_release_step=2, reference_steps=None, service_stop=stop,
        ))
    assert len(requests) == 4 and all(r == requests[0] for r in requests)
    assert stop.error is None
    artifact = Path(caught.value.run_artifact["artifact_directory"])
    record = json.loads((artifact / "run.json").read_text())
    assert record["run_status"]["valid"] is False
    logged, = record["trace"]["llm_requests"]
    assert logged["input_rejection"] == kind and len(logged["retries"]) == 3


@pytest.mark.parametrize("status", [None, 400, 401, 429, 503])
def test_api_exception_retries_are_bounded_independently_of_status(status):
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    error = (
        APIConnectionError(request=request) if status is None else
        APIStatusError("test error", response=httpx.Response(status, request=request),
                       body=None)
    )
    send = AsyncMock(side_effect=error)
    logged = {}
    with patch("egocua.adapters.browsergym.asyncio.sleep", new_callable=AsyncMock) as sleep:
        with pytest.raises(type(error)):
            asyncio.run(_request_with_retries(send, logged, ModelCallStop()))
    assert send.await_count == 4
    assert len(logged["retries"]) == sleep.await_count == 3
    for attempt, call in enumerate(sleep.await_args_list):
        assert 2**attempt <= call.args[0] <= 2**attempt + 0.25


@pytest.mark.parametrize("headers", [{"x-should-retry": "false"}, {"Retry-After": "120"}])
def test_provider_retry_instructions_can_stop_request(headers):
    response = httpx.Response(
        503, headers=headers, request=httpx.Request("POST", "https://example.test")
    )
    send = AsyncMock(side_effect=APIStatusError("test error", response=response, body=None))
    logged = {}
    with pytest.raises(APIStatusError):
        asyncio.run(_request_with_retries(send, logged, ModelCallStop()))
    assert send.await_count == 1
    assert "retries" not in logged


def test_no_completion_choice_retries_same_request():
    completion = SimpleNamespace(choices=[object()])
    send = AsyncMock(side_effect=[SimpleNamespace(choices=[]), completion])
    logged = {}
    with patch("egocua.adapters.browsergym.asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(_request_with_retries(send, logged, ModelCallStop()))
    assert result is completion
    assert send.await_count == 2
    assert len(logged["retries"]) == 1


def current_text(request):
    return "\n".join(
        p["text"] for p in request["messages"][-1]["content"] if p["type"] == "text"
    )


def test_action_aligned_refresh_does_not_show_second_batch_on_first_request(
    browser_assets, provider, tmp_path
):
    definition = browser_assets("refresh")
    condition = replace(
        CONDITIONS["online"],
        frame_interval_seconds=1,
        frame_delivery_steps=1,
        batch_seconds=8,
        wearer_frames_per_image=4,
        visual_history="compressed",
        replay_timing="action_aligned",
    )
    opening, later = (
        packet["frame_paths"][0] for packet in definition["visual_packets"]
    )
    definition["visual_session"] = {
        "update_onset_seconds": 66.5,
        "frequency_samples": {
            "window_seconds": [56, 71.5],
            "timestamps_seconds": list(range(56, 72)),
            "frame_paths": [opening] * 8 + [later] * 8,
        },
    }
    commands, requests = provider
    commands.extend({"action": action, "parameters": {}} for action in ("wait", "done"))

    record = asyncio.run(
        run_live_task(
            definition, condition, model="offline-script", asset_root=tmp_path,
            default_release_step=2, reference_steps=None,
        )
    )

    first, second = requests
    assert [int(t) for t in re.findall(r"source t=(\d+) s", current_text(first))] == list(
        range(56, 64)
    )
    assert [int(t) for t in re.findall(r"source t=(\d+) s", current_text(second))] == list(
        range(56, 72)
    )
    assert [
        sum(p["type"] == "image_url" for p in r["messages"][-1]["content"])
        for r in requests
    ] == [3, 5]  # Two/four wearer grids plus the website screenshot.
    assert "Glasses images received before step 2:" not in current_text(first)
    assert "Glasses images received before step 2:" in current_text(second)
    update, = (
        event for event in record["actions"] if event["type"] == "visual_context_update"
    )
    assert (update["step"], update["source_step"], update["update"]) == (2, 2, True)
    assert check_run(definition, record)["run_valid"]


def test_live_run_wakes_on_source_batch_without_any_browser_action(
    browser_assets, provider, tmp_path
):
    definition = browser_assets("graft")
    condition = replace(
        CONDITIONS["online"], frame_interval_seconds=1,
        frame_delivery_steps=1, batch_seconds=8,
    )
    frame = definition["visual_packets"][0]["frame_paths"][0]
    definition["visual_session"] = {
        "update_onset_seconds": 24,
        "frequency_samples": {
            "window_seconds": [0, 24],
            "timestamps_seconds": list(range(25)),
            "frame_paths": [frame] * 25,
        },
    }
    commands, requests = provider
    commands.extend(
        {"action": "done", "parameters": {"text": "finished"}} for _ in range(2)
    )
    record = asyncio.run(
        run_live_task(
            definition, condition, model="offline-script", asset_root=tmp_path,
            default_release_step=2, reference_steps=None,
        )
    )
    follow_up = next(
        event for event in record["actions"] if event["type"] == "instruction_injection"
    )
    assert (follow_up["step"], follow_up["source_step"]) == (2, 3)
    assert len(requests) == 2
    assert check_run(definition, record)["run_valid"]


def navigate_to(entity):
    def command(request):
        text = current_text(request)
        base = re.search(r"URL: (http://127\.0\.0\.1:\d+)", text)[1]
        return {"action": "goto", "parameters": {"url": f"{base}/product/{entity}"}}

    return command


def click(label):
    def command(request):
        text = current_text(request)
        bid = re.search(
            r"\[([^]]+)\] button ['\"]" + re.escape(label) + r"['\"]", text
        )[1]
        return {"action": "click", "parameters": {"bid": bid}}

    return command


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_real_browser_releases_inputs_and_scores_live_state(
    browser_assets, provider, tmp_path, track
):
    definition = browser_assets(track)
    commands, requests = provider
    old = item_id(track, 0 if track != "graft" else 1)
    final = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    if track == "clarify":
        commands.append(
            {"action": "ask_user", "parameters": {"question": "Which variant do you mean?"}}
        )
    commands.extend(
        [
            navigate_to(old),
            click("Add to Cart"),
            click("Delete"),
            navigate_to(final),
            click("Add to Cart"),
            {"action": "done", "parameters": {"text": "finished"}},
        ]
    )
    record = asyncio.run(
        run_task(
            definition,
            CONDITIONS["online"],
            model="offline-script",
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            default_release_step=2,
            reference_steps=None,
            release_trigger="first_native_commitment",
        )
    )
    checked = check_run(definition, record)
    assert checked["run_valid"] and checked["e2e_success"]
    assert record["meta"]["agent"] == "browsergym"
    assert record["live_state"]["selected_entity_ids"] == [final]
    assert evaluate_final_state(definition["goal"], record["live_state"])
    assert not record["final_state"]["submission_completed"]
    assert all(not h["result"][0].get("error") for h in record["trace"]["history"])
    assert len(requests) == record["steps"]
    for request, logged in zip(requests, record["trace"]["llm_requests"], strict=True):
        content = request["messages"][-1]["content"]
        images = [p for p in content if p["type"] == "image_url"]
        assert len(images) == logged["wearer_image_count"] + 1
        assert images[-1]["image_url"]["detail"] == "high"
        assert Image.open(
            BytesIO(base64.b64decode(images[-1]["image_url"]["url"].split(",", 1)[1]))
        ).size == (1512, 982)
    if track == "refresh":
        assert [r["wearer_image_count"] for r in record["trace"]["llm_requests"]] == [
            4,
            4,
            8,
            8,
            8,
            8,
        ]
    if track == "graft":
        assert definition["instructions"]["b"] not in current_text(requests[1])
        assert definition["instructions"]["b"] in current_text(requests[2])
    if track == "clarify":
        assert definition["clarification"]["canonical_answer"] in current_text(
            requests[1]
        )
    first_image = record["trace"]["history"][0]["state"]["screenshot_path"]
    assert Image.open(first_image).size == (1512, 982)


@pytest.mark.parametrize("setting", SETTING_ORDER)
def test_interaction_settings_send_only_their_own_inputs(
    browser_assets, provider, tmp_path, setting
):
    definition = browser_assets("clarify")
    condition = next(
        condition for condition in setting_conditions(replace(
            CONDITIONS["online"], ask="open", frame_interval_seconds=1,
            frame_delivery_steps=1, batch_seconds=8, visual_history="compressed",
            replay_timing=ACTION_ALIGNED_REPLAY, wearer_frames_per_image=4,
        ))
        if condition.name == setting
    )
    is_computer = setting == COMPUTER_USE
    original = definition["instructions"]["ambiguous"]
    standalone = "Add one large silicone mixing spoon."
    answer = definition["clarification"]["canonical_answer"]
    if is_computer:
        definition["instructions"][COMPUTER_USE] = standalone
    else:
        path = definition["visual_packets"][0]["frame_paths"][0]
        definition["visual_session"] = {
            "frequency_samples": {
                "window_seconds": [0, 15],
                "timestamps_seconds": list(range(16)),
                "frame_paths": [path] * 16,
            },
        }
    commands, requests = provider
    if not is_computer:
        commands.append({"action": "ask_user", "parameters": {"question": "Which one?"}})
    target = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    commands.extend([
        navigate_to(target),
        click("Add to Cart"),
        {"action": "done", "parameters": {"text": "finished"}},
    ])
    record = asyncio.run(run_live_task(
        definition, condition, model="offline-script", asset_root=tmp_path,
        output_root=tmp_path / "artifacts", default_release_step=2,
        reference_steps=None, max_steps=20,
    ))
    checked = check_run(definition, record)
    assert checked["run_valid"] and checked["e2e_success"]
    initial_image_count = {"computer_use": 0, "offline_ego_use": 4, "online": 2}[setting]
    assert record["trace"]["llm_requests"][0]["wearer_image_count"] == initial_image_count
    first_text = current_text(requests[0])
    assert (standalone in first_text) is is_computer
    assert (original in first_text) is not is_computer
    assert answer not in first_text
    if not is_computer:
        assert answer in current_text(requests[1])
    else:
        assert all(row["wearer_image_count"] == 0 for row in record["trace"]["llm_requests"])


@pytest.mark.parametrize(
    "code",
    [
        "click('1');click('2')",
        "open('/etc/passwd')",
        "page.evaluate('1')",
        "upload_file('1', '/tmp/file')",
    ],
)
def test_action_set_rejects_batch_arbitrary_code_and_file_tools(code):
    with pytest.raises((ValueError, NameError, ParseBaseException)):
        action_set().to_python_code(code)


@pytest.mark.parametrize(
    "content", ["{}", "[]", '{"action":1}', '{"action":"ask_user"}', "not json"]
)
def test_malformed_model_output_is_not_a_browser_action(content):
    with pytest.raises(ValueError):
        _parse_action(content)


def test_prompt_does_not_offer_unavailable_clarification():
    assert "ask_user is unavailable" in system_prompt("actions", False)


def test_default_prompt_does_not_teach_the_update_strategy():
    prompt = system_prompt("actions", False)
    assert "Later wearer images or requests may change the target" not in prompt
    assert "retain still-valid choices" not in prompt
    hinted = system_prompt("actions", False, STRATEGY_HINT)
    assert "Later wearer images or requests may change the target" in hinted
    assert "retain still-valid choices" in hinted


@pytest.mark.parametrize("instruction", ["Find it.", "  Keep my wording.\nDo not change it.  "])
def test_initial_request_is_verbatim_and_has_no_harness_instructions(instruction):
    context = SimpleNamespace(
        initial_payload={"instruction": instruction, "observations": [],
                         "clarification_available": False},
    )
    inputs = ModelInput(context)
    labeled = wearer_said(instruction, 1)
    assert inputs.parts == [{"type": "text", "text": labeled}]
    message = inputs.message("URL: http://127.0.0.1:1234/", b"website")
    assert message["content"][0]["text"] == labeled
    assert instruction in labeled
    assert "browser is already open" not in labeled
    assert "browser is already open" in system_prompt("actions", False)
    assert "ask_user is unavailable" in system_prompt("actions", False)


def test_packed_images_keep_actual_bytes_and_order(tmp_path):
    from egocua.frame_inputs import observation_images

    paths = []
    for index in range(8):
        p = tmp_path / f"{index}.jpg"
        Image.new("RGB", (32, 24), (index * 20, 0, 0)).save(p)
        paths.append(p.name)
    first = {
        "frame_paths": paths[:4],
        "wearer_frames_per_image": 4,
        "wearer_jpeg_quality": 85,
    }
    second = {
        "frame_paths": paths[4:],
        "wearer_frames_per_image": 4,
        "wearer_jpeg_quality": 85,
    }
    context = SimpleNamespace(
        initial_payload={
            "instruction": "Find it",
            "observations": [first],
            "clarification_available": False,
        },
        base_url="http://127.0.0.1:1",
        observation_images=lambda packet: observation_images(packet, tmp_path),
    )
    inputs = ModelInput(context)
    inputs.release(
        {"type": "visual_context_update", "step": 2, "payload": second}
    )
    message = inputs.message("website", b"website")
    images = [
        p["image_url"]["url"] for p in message["content"] if p["type"] == "image_url"
    ]
    assert len(images) == 3
    assert images[:2] == [
        f"data:{i.media_type};base64,{i.data}"
        for p in [first, second]
        for i in observation_images(p, tmp_path)
    ]
    assert len(inputs.layouts) == 2


def test_invalid_model_action_and_stale_id_are_visible_without_harness_retry(
    browser_assets, provider, tmp_path
):
    definition = browser_assets("graft")
    commands, requests = provider
    commands.extend(["not json", {"action": "click", "parameters": {"bid": "missing-id"}}, {"action": "done", "parameters": {}}])
    record = asyncio.run(
        run_task(
            definition,
            CONDITIONS["text_only"],
            model="offline-script",
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            default_release_step=2,
            reference_steps=None,
        )
    )
    assert record["run_status"]["valid"]
    assert len(requests) == 3
    assert all(request["response_format"] == {"type": "json_object"} for request in requests)
    assert record["trace"]["agent_settings"]["response_format"] == {"type": "json_object"}
    assert record["trace"]["llm_requests"][0]["model_output_error"]
    assert record["trace"]["history"][1]["result"][0]["error"]
    assert "missing-id" in current_text(requests[2])
    assert not check_run(definition, record)["e2e_success"]


def test_stage_budget_does_not_make_extra_model_request(
    browser_assets, provider, tmp_path
):
    definition = browser_assets("graft")
    commands, requests = provider
    commands.extend([{"action": "wait", "parameters": {}}, {"action": "wait", "parameters": {}}])
    record = asyncio.run(
        run_task(
            definition,
            CONDITIONS["text_only"],
            model="offline-script",
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            max_steps=4,
            default_release_step=2,
            reference_steps=None,
            release_trigger="first_selection",
            pre_release_max_steps=2,
            post_release_max_steps=2,
        )
    )
    assert len(requests) == record["steps"] == 2
    assert record["meta"]["stage_release"]["status"] == "stage_not_reached"
    assert [r["usage"]["total_tokens"] for r in record["trace"]["llm_requests"]] == [
        15,
        15,
    ]


def test_prepared_cart_and_change_are_visible_before_first_decision(
    browser_assets, provider, tmp_path
):
    definition = browser_assets("graft")
    commands, requests = provider
    commands.append({"action": "done", "parameters": {}})
    selected = item_id("graft", 1)
    record = asyncio.run(
        run_task(
            definition,
            CONDITIONS["text_only"],
            model="offline-script",
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            default_release_step=1,
            reference_steps=None,
            release_trigger="timeline_step",
            initial_site_state={"selected_entity_ids": [selected], "fields": {}},
        )
    )
    text = current_text(requests[0])
    assert "graft Product 1" in text
    assert definition["instructions"]["b"] in text
    assert record["live_state"]["selected_entity_ids"] == [selected]
    assert not any(a["type"] == "select_entity" for a in record["actions"])


def test_concurrent_runs_use_separate_browsergym_processes(
    browser_assets, provider, tmp_path
):
    definitions = [browser_assets("graft"), browser_assets("refresh")]
    commands, requests = provider
    commands.extend([{"action": "done", "parameters": {}}, {"action": "done", "parameters": {}}])

    async def run_pair():
        return await asyncio.gather(
            *(
                run_task(
                    d,
                    CONDITIONS["text_only"],
                    model="offline-script",
                    asset_root=tmp_path,
                    output_root=tmp_path / "artifacts",
                    default_release_step=2,
                    reference_steps=None,
                )
                for d in definitions
            )
        )

    records = asyncio.run(run_pair())
    assert len(requests) == 2
    assert len({r["initial_context"]["instruction"] for r in records}) == 2
    urls = [r["trace"]["history"][0]["state"]["url"] for r in records]
    assert len(set(urls)) == 2
    assert all(
        r["run_status"]["valid"] and not r["live_state"]["selected_entity_ids"]
        for r in records
    )


def test_remote_error_keeps_original_cause_and_errno():
    from egocua.execution import exception_details

    detail = {
        "type": "APIConnectionError",
        "message": "connection closed",
        "cause": {"type": "OSError", "message": "offline", "errno": 49},
    }
    error = BrowserWorkerError(detail)
    assert exception_details(error) == detail


@pytest.mark.parametrize("kind", ["empty_choices", "refusal", "content_filter"])
def test_provider_rejection_is_not_a_model_action(kind):
    from egocua.adapters.browsergym import _response_text

    response = SimpleNamespace(
        choices=[]
        if kind == "empty_choices"
        else [
            SimpleNamespace(
                finish_reason="content_filter" if kind == "content_filter" else "stop",
                message=SimpleNamespace(
                    content="", refusal="refused" if kind == "refusal" else None
                ),
            )
        ]
    )
    with pytest.raises(RuntimeError):
        _response_text(response)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:document.title='changed'",
        "data:text/html,<script>alert(1)</script>",
        "file:///etc/passwd",
        "http://127.0.0.1:9999/cart",
        "https://example.invalid/",
    ],
)
def test_navigation_rejects_non_task_origins_before_execution(url):
    from egocua.adapters.browsergym_environment import BrowserActions

    with pytest.raises(ValueError, match="HTTP origin"):
        BrowserActions("http://127.0.0.1:1234/").to_python_code(f"goto({url!r})")


@pytest.mark.parametrize(
    "action", ["noop(1000000000000)", "noop(wait_ms=1001)", "noop(-1)", "noop(1e309)"]
)
def test_browser_wait_is_bounded(action):
    from egocua.adapters.browsergym_environment import BrowserActions

    with pytest.raises(ValueError, match="Waiting must"):
        BrowserActions("http://127.0.0.1:1234/").to_python_code(action)


def test_unavailable_question_error_is_delivered_to_the_model(
    browser_assets, provider, tmp_path
):
    definition = browser_assets("graft")
    commands, requests = provider
    commands.extend(
        [{"action": "ask_user", "parameters": {"question": "Which one?"}}, {"action": "done", "parameters": {}}]
    )
    record = asyncio.run(
        run_task(
            definition,
            CONDITIONS["text_only"],
            model="offline-script",
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            default_release_step=2,
            reference_steps=None,
        )
    )
    assert (
        record["trace"]["history"][0]["result"][0]["error"] == "ask_user is unavailable"
    )
    assert "ask_user is unavailable" in current_text(requests[1])
    assert "Previous action feedback" in current_text(requests[1])


def test_cancellation_stops_before_another_model_call_and_retains_archive(
    browser_assets, provider, tmp_path
):
    from egocua.run_ledger import load_run_record

    definition = browser_assets("graft")
    commands, requests = provider
    commands.append({"action": "noop", "parameters": {"wait_ms": 1000}})

    async def cancel_during_browser_action():
        future = asyncio.create_task(
            run_task(
                definition,
                CONDITIONS["text_only"],
                model="offline-script",
                asset_root=tmp_path,
                output_root=tmp_path / "artifacts",
                default_release_step=2,
                reference_steps=None,
            )
        )
        while not requests:
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.2)
        future.cancel()
        with pytest.raises(asyncio.CancelledError) as interrupted:
            await asyncio.wait_for(future, timeout=6)
        return interrupted.value.run_artifact

    artifact = asyncio.run(cancel_during_browser_action())
    assert len(requests) == 1
    record = load_run_record(artifact["artifact_directory"])
    assert not record["run_status"]["valid"]
    assert record["trace"]["llm_requests"][0]["success"]
    assert record["meta"]["run_id"] == artifact["run_id"]


def test_browser_timeout_is_an_invalid_run_not_a_model_failure(
    browser_assets, provider, tmp_path, monkeypatch
):
    from egocua.adapters.browsergym_environment import BrowserSession

    async def timed_out(self, action):
        raise TimeoutError("BrowserGym did not finish within its action budget")

    monkeypatch.setattr(BrowserSession, "step", timed_out)
    definition = browser_assets("graft")
    commands, requests = provider
    commands.append({"action": "noop", "parameters": {"wait_ms": 0}})
    with pytest.raises(TimeoutError) as stopped:
        asyncio.run(
            run_task(
                definition,
                CONDITIONS["text_only"],
                model="offline-script",
                asset_root=tmp_path,
                output_root=tmp_path / "artifacts",
                default_release_step=2,
                reference_steps=None,
            )
        )
    from egocua.run_ledger import load_run_record

    record = load_run_record(stopped.value.run_artifact["artifact_directory"])
    assert not record["run_status"]["valid"]
    assert len(requests) == 1


def test_javascript_navigation_cannot_modify_real_site_state(
    browser_assets, provider, tmp_path
):
    definition = browser_assets("graft")
    commands, requests = provider
    code = (
        "javascript:fetch('/cart/add', {method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:'item_id="
        + item_id("graft", 0)
        + "'});void(0)"
    )
    commands.extend([{"action": "goto", "parameters": {"url": code}}, {"action": "done", "parameters": {}}])
    record = asyncio.run(
        run_task(
            definition,
            CONDITIONS["text_only"],
            model="offline-script",
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            default_release_step=2,
            reference_steps=None,
        )
    )
    assert record["live_state"]["selected_entity_ids"] == []
    assert "Navigation must" in record["trace"]["history"][0]["result"][0]["error"]
    assert "Navigation must" in current_text(requests[1])
    assert record["run_status"]["valid"]


def test_real_browser_action_deadline_and_shutdown(
    browser_assets, tmp_path, monkeypatch
):
    from egocua.adapters import browsergym_environment as browser
    from egocua.environment import create_site_runtime, load_site_data

    definition = browser_assets("graft")
    runtime = create_site_runtime(
        definition, load_site_data(tmp_path, "egomart"), asset_root=tmp_path
    )

    async def run():
        session = browser.BrowserSession()
        with runtime.server as server:
            try:
                await session.start(server.base_url + "/", "offline timeout check", 0)
                monkeypatch.setattr(browser, "BROWSER_ACTION_TIMEOUT_SECONDS", 0.05)
                with pytest.raises(TimeoutError, match="BrowserGym did not finish"):
                    await session.step("noop(1000)")
            finally:
                await session.close()
        assert session.process.returncode == 0

    asyncio.run(run())


def test_hung_browser_process_is_killed_within_shutdown_budget(monkeypatch):
    import sys
    import signal
    from egocua.adapters import browsergym_environment as browser

    monkeypatch.setattr(browser, "BROWSER_CLOSE_GRACE_SECONDS", 0.05)

    async def run():
        session = browser.BrowserSession()
        session.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        await asyncio.wait_for(session.close(), timeout=2)
        assert session.process.returncode == -signal.SIGKILL

    asyncio.run(run())


def test_named_parameters_compile_literals_without_guessing():
    value = "quoted 'value', newline\n'); open('/tmp/should-not-run')"
    command = _parse_action(
        json.dumps({"action": "fill", "parameters": {"bid": "12", "value": value}})
    )
    code = command["action"]
    import ast

    call = ast.parse(code).body[0].value
    assert call.func.id == "fill"
    assert {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords} == {
        "bid": "12",
        "value": value,
    }
    action_set().to_python_code(code)


@pytest.mark.parametrize(
    "command",
    [
        {"action": "click('12')", "parameters": {}},
        {"action": "click", "parameters": {"wrong_field": "12"}},
        {"action": "fill", "parameters": {"bid": "12"}},
        {"action": "open", "parameters": {"file": "/etc/passwd"}},
        {"action": "click", "parameters": {"bid": "12"}, "extra": "ignored"},
        {"action": "wait", "parameters": {"bid": "12"}},
        {"action": "done", "parameters": {"text": 123}},
    ],
)
def test_named_command_rejects_unknown_or_incomplete_parameters(command):
    with pytest.raises((ValueError, TypeError)):
        _parse_action(json.dumps(command))


def test_browsergym_list_parameter_definitions_are_correct():
    from egocua.adapters.browsergym_environment import action_definitions

    definitions = {row["name"]: row["parameters"] for row in action_definitions()}
    assert definitions["click"]["properties"]["modifiers"] == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert (
        definitions["select_option"]["properties"]["options"]["anyOf"][1]["type"]
        == "array"
    )


def test_graft_upfront_sends_original_requests_and_records_its_input_version(
    browser_assets, provider, tmp_path
):
    from egocua.messages import GRAFT_UPFRONT_INPUT_POLICY

    definition = browser_assets("graft")
    commands, requests = provider
    commands.append({"action": "done", "parameters": {}})

    record = asyncio.run(
        run_task(
            definition,
            CONDITIONS["upfront"],
            model="offline-script",
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            default_release_step=2,
            reference_steps=None,
            release_trigger="first_native_commitment",
        )
    )

    original = "\n\n".join(definition["instructions"][key] for key in ("a", "b"))
    assert record["initial_context"]["instruction"] == original
    assert wearer_said(original, 1) in current_text(requests[0])
    assert definition["instructions"]["bundled"] not in current_text(requests[0])
    assert (
        record["trace"]["agent_settings"]["system_prompt_policy"]
        == GRAFT_UPFRONT_INPUT_POLICY
    )
    assert not any(
        action["type"] == "instruction_injection" for action in record["actions"]
    )
    assert check_run(definition, record)["run_valid"] is True


@pytest.mark.parametrize(
    "policy,image_counts",
    [
        ("full", [4, 6, 8, 10, 12, 12]),
        ("compressed", [4, 6, 7, 8, 9, 9]),
        ("summary", [4, 6, 6, 6, 6, 6]),
    ],
)
def test_live_retention_uses_one_request_per_step(
    policy, image_counts, browser_assets, provider, tmp_path
):
    definition = browser_assets("graft")
    condition = replace(
        CONDITIONS["online"],
        frame_interval_seconds=1,
        frame_delivery_steps=1,
        batch_seconds=8,
        wearer_frames_per_image=4,
        visual_history=policy,
    )
    frame = definition["visual_packets"][0]["frame_paths"][0]
    definition["visual_session"] = {
        "update_onset_seconds": 40,
        "frequency_samples": {
            "window_seconds": [0, 47],
            "timestamps_seconds": list(range(48)),
            "frame_paths": [frame] * 48,
        },
    }
    commands, requests = provider

    def respond(request):
        text = current_text(request)
        step = len(requests)
        result = {"action": "wait" if step < 6 else "done", "parameters": {}}
        targets = re.findall(r"Summarize visual batch ID (\d+)\.", text)
        if targets:
            result["visual_summaries"] = {
                target: f"Observed physical detail {target}." for target in targets
            }
        return result

    commands.extend([respond] * 6)
    record = asyncio.run(
        run_live_task(
            definition, condition, model="offline-script", asset_root=tmp_path,
            default_release_step=2, reference_steps=None,
        )
    )
    assert len(requests) == 6
    assert check_run(definition, record)["run_valid"]
    entries = record["trace"]["llm_requests"]
    assert [entry["older_visual_batch_count"] for entry in entries] == [0, 0, 1, 2, 3, 3]
    assert [entry["wearer_image_count"] for entry in entries] == image_counts
    assert all("Earlier visual batch summary:" not in current_text(r) for r in requests[:2])
    assert (
        "Earlier visual batch summary: Observed physical detail 2."
        in current_text(requests[2])
    ) == (policy == "summary")
    assert all(
        "visual_summaries" not in msg["content"]
        for request in requests for msg in request["messages"]
        if msg["role"] == "assistant"
    )


@pytest.mark.parametrize("dedicated", [False, True])
def test_deepseek_uses_only_an_approved_official_route(monkeypatch, dedicated):
    from egocua.adapters.browsergym import provider_endpoint

    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-gateway-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    if dedicated:
        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-official-key")
        assert provider_endpoint("deepseek-flash") == "https://api.deepseek.com"
    else:
        with pytest.raises(ValueError, match="requires DEEPSEEK_API_KEY"):
            provider_endpoint("deepseek-flash")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
        assert provider_endpoint("deepseek-flash") == "https://api.deepseek.com/v1"
    assert provider_endpoint("gemini-3.8-flash") == ("https://gateway.invalid/v1" if dedicated else "https://api.deepseek.com/v1")


def test_openrouter_requires_its_own_key_and_a_qualified_model(monkeypatch):
    from egocua.adapters.browsergym import provider_endpoint

    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-gateway-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY is not configured"):
        provider_endpoint("openrouter:qwen/qwen3.5-9b")
    with pytest.raises(ValueError, match="openrouter:author/model"):
        provider_endpoint("openrouter:qwen3.5-9b")

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-router-key")
    assert provider_endpoint("openrouter:qwen/qwen3.5-9b") == "https://openrouter.ai/api/v1"
    assert provider_endpoint("gemini-3.8-flash") == "https://gateway.invalid/v1"


@pytest.mark.parametrize("route", [None, "siliconflow/fp8"])
@pytest.mark.parametrize("request_model", ["qwen/qwen3.5-9b", "bytedance/ui-tars-1.5-7b"])
def test_browsergym_sends_openrouter_model_and_credential(browser_assets, monkeypatch, tmp_path, route, request_model):
    model = f"openrouter:{request_model}" + (f"@{route}" if route else "")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-gateway-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-router-key")

    def respond(request):
        assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-router-key"
        body = json.loads(request.content)
        assert body["model"] == request_model
        assert body.get("response_format") == (
            None if request_model == "bytedance/ui-tars-1.5-7b"
            else {"type": "json_object"}
        )
        assert body.get("max_tokens") == (
            512 if request_model == "bytedance/ui-tars-1.5-7b" else None
        )
        if route:
            assert body["provider"] == {
                "only": [route], "allow_fallbacks": False, "require_parameters": True,
            }
            assert "seed" not in body
        else:
            assert "provider" not in body
            assert body["seed"] == 9
        assert any(
            part["type"] == "image_url"
            for part in body["messages"][-1]["content"]
        )
        return httpx.Response(200, json={
            "id": "mock-openrouter",
            "object": "chat.completion",
            "created": 0,
            "model": "qwen/qwen3.5-9b",
            "provider": "SiliconFlow",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"action":"done","parameters":{}}',
                },
                "finish_reason": "stop",
            }],
        })

    def client(**options):
        transport = httpx.MockTransport(respond)
        return AsyncOpenAI(http_client=httpx.AsyncClient(transport=transport), **options)

    monkeypatch.setattr("openai.AsyncOpenAI", client)
    definition = browser_assets("graft")
    record = asyncio.run(run_task(
        definition, CONDITIONS["upfront"], model=model,
        asset_root=tmp_path, max_steps=1, default_release_step=2, reference_steps=None,
        provider_seed=9,
    ))
    assert record["meta"]["model"] == model
    assert record["meta"]["provider"] == "openrouter"
    assert record["meta"]["runtime"]["provider_base_url"] == "https://openrouter.ai/api/v1"
    assert record["trace"]["llm_requests"][0]["upstream_provider"] == "SiliconFlow"
    if route:
        assert record["trace"]["agent_settings"]["openrouter_route"] == route
        assert record["trace"]["agent_settings"]["model_seed_sent"] is False
    assert "test-router-key" not in json.dumps(record)
    assert check_run(definition, record)["run_valid"]


def test_experiment_folder_encodes_the_model_namespace():
    name = experiment_folder_name({
        "category": "评测", "variable": "三场景", "model": "openrouter:qwen/qwen3.5-9b",
    })
    assert name.endswith("_评测_三场景_openrouter%3Aqwen%2Fqwen3.5-9b")
    with pytest.raises(ValueError, match="folder-name segments"):
        experiment_folder_name({
            "category": "../评测", "variable": "三场景", "model": "openrouter:qwen/qwen3.5-9b",
        })


@pytest.mark.parametrize("model, expected_limit", [("ui-tars-7b", 512), ("offline-script", None)])
def test_ui_tars_generation_is_bounded_without_changing_other_models(
    browser_assets, provider, tmp_path, monkeypatch, model, expected_limit,
):
    monkeypatch.setattr("egocua.adapters.browsergym.LLM_MAX_RETRIES", 0)
    definition = browser_assets("graft")
    commands, requests = provider
    commands.append({"action": "done", "parameters": {}})
    record = asyncio.run(run_task(
        definition, CONDITIONS["upfront"], model=model,
        asset_root=tmp_path, max_steps=1, default_release_step=2, reference_steps=None,
    ))
    request, = requests
    assert request.get("max_tokens") == expected_limit
    assert request.get("response_format") == (
        None if expected_limit is not None else {"type": "json_object"}
    )
    assert record["trace"]["agent_settings"].get("max_output_tokens") == expected_limit
    assert record["trace"]["agent_settings"]["response_format"] == request.get("response_format")
    entry, = record["trace"]["llm_requests"]
    assert entry["response_finish_reason"] == "stop"


def test_length_limited_output_is_not_an_api_retry():
    response = SimpleNamespace(choices=[
        SimpleNamespace(finish_reason="length", message=SimpleNamespace(content='{"action":'))
    ])
    send = AsyncMock(return_value=response)
    result = asyncio.run(_request_with_retries(send, {}, ModelCallStop()))
    assert result is response
    send.assert_awaited_once()
    with pytest.raises(ValueError):
        _parse_action(result.choices[0].message.content)
