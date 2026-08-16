"""Minimal multimodal chat-loop agent for protocol and API diagnostics.

Implements the packetized input contract in docs/BENCHMARK_DESIGN.md: the task
instruction and camera observations arrive as one multimodal message, every mid-run event
is an appended user message (with images when the event carries pixels), and the
agent acts on the products through tools backed by plain HTTP against the frozen
EgoMart server, so gating, logging, and judging are shared with the production harness.
Browser-Use remains the v1 production Agent; this loop cannot replace its model runs.

The model transport is injectable. The default uses the requests-based
OpenAI-compatible transport; tests inject a scripted transport.
"""

from __future__ import annotations

import base64
import hashlib
import mimetypes
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from sites.egomart import pages
from ..adapters.openai_requests import OpenAIRequestsTransport
from ..execution import RunConfig, RunContext, run_experiment
from ..schema import CLARIFICATION_SLOTS
from ..workspace import resolve_within

PAGE_TEXT_LIMIT = 3500

SYSTEM_PROMPT = (
    "You are a computer-use agent shopping for a smart-glasses wearer. The wearer's "
    "first-person images define what to buy. Ground every choice in the images, use "
    "the EgoMart tools, add the intended item(s) to the cart, and submit. New messages "
    "may arrive while you work: the wearer's goal can change. Always honor the latest "
    "state of the goal before submitting. Product item IDs are opaque: copy the exact "
    "item_id returned by a tool instead of deriving one from a product name. Do not finish "
    "without submitting the cart. If multiple visual candidates fit and ask_user is available, "
    "ask before choosing; do not guess."
)

Transport = Callable[[dict[str, Any]], dict[str, Any]]


def _image_block(path: Path) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mimetypes.guess_type(path.name)[0] or "image/jpeg",
            "data": base64.b64encode(path.read_bytes()).decode(),
        },
    }


def _audit_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_audit_value(item) for item in value]
    if not isinstance(value, dict):
        return value
    audited = {key: _audit_value(child) for key, child in value.items() if key != "data"}
    if value.get("type") == "image" and isinstance(value.get("source"), dict):
        data = value["source"].get("data")
        if data:
            audited["source"]["sha256"] = hashlib.sha256(base64.b64decode(data)).hexdigest()
    return audited


def _observation_blocks(
    observation: dict[str, Any], asset_root: Path, label: str
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": label}]
    blocks += [
        _image_block(resolve_within(asset_root, frame, context="released observation image"))
        for frame in observation["frame_paths"]
    ]
    return blocks


def _page_text(html_text: str) -> str:
    text = re.sub(r"<style.*?</style>", " ", html_text, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:PAGE_TEXT_LIMIT]


def _product_blocks(
    page_text: str, items: list[dict[str, Any]], asset_root: Path
) -> list[dict[str, Any]]:
    item_records = "\n".join(
        f"- item_id={item['item_id']} | name={item['name']}" for item in items
    )
    records_text = (
        "Actionable product records (copy item_id exactly when calling another tool):\n"
        + item_records
        if item_records else "No actionable product records were returned."
    )
    blocks: list[dict[str, Any]] = [
        {"type": "text", "text": records_text},
        {"type": "text", "text": page_text},
    ]
    for item in items:
        image_path = item.get("image_path")
        if not image_path:
            continue
        try:
            resolved = resolve_within(asset_root, image_path, context="EgoMart product image")
        except (ValueError, FileNotFoundError):
            continue
        blocks.append({
            "type": "text",
            "text": f"Product image for {item['item_id']} — {item['name']}:",
        })
        blocks.append(_image_block(resolved))
    return blocks


def _tools(clarification_available: bool) -> list[dict[str, Any]]:
    tools = [
        {
            "name": "search",
            "description": "Search EgoMart's fixed product catalog. Returns the results page text.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "open_product",
            "description": (
                "Open one product page using the exact opaque item_id returned by search."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
        {
            "name": "add_to_cart",
            "description": (
                "Add an item using the exact opaque item_id returned by search or open_product."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
        {
            "name": "remove_from_cart",
            "description": (
                "Remove a whole cart line using the exact opaque item_id returned by a tool."
            ),
            "input_schema": {
                "type": "object",
                "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            },
        },
        {
            "name": "submit_cart",
            "description": "Submit the current cart as the final answer.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "finish",
            "description": "End the run after the cart has been submitted.",
            "input_schema": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    ]
    if clarification_available:
        tools.append({
            "name": "ask_user",
            "description": (
                "Ask the wearer ONE clarification question. slot must be one of: "
                "position, appearance, brand, variant, sequence."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "slot": {"type": "string", "enum": sorted(CLARIFICATION_SLOTS)},
                    "question": {"type": "string"},
                },
                "required": ["slot", "question"],
            },
        })
    return tools


def _http(base_url: str, path: str, form: dict[str, str] | None = None) -> str:
    if form is None:
        request = urllib.request.Request(base_url + path)
    else:
        payload = "&".join(f"{key}={urllib.parse.quote(value)}" for key, value in form.items())
        request = urllib.request.Request(base_url + path, data=payload.encode(), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read().decode()
    except urllib.error.HTTPError as exc:
        if 400 <= exc.code < 500:
            detail = exc.read().decode(errors="replace")
            return f"Tool request rejected (HTTP {exc.code}). {detail}"
        raise


def run_task(
    task: dict[str, Any], arm: str, *, model: str = "gpt-5.4-mini",
    max_steps: int = 40, replicate_id: int = 0, output_path: str | Path | None = None,
    schedule_index: int | None = None,
    asset_root: str | Path = ".", transport: Transport | None = None,
    artifacts_root: str | Path | None = None, experiment_id: str = "manual",
) -> dict[str, Any]:
    if task.get("site") != "egomart":
        raise ValueError("minimal-loop is an EgoMart diagnostic; use Browser-Use for other sites")
    transport = transport or OpenAIRequestsTransport.from_env()
    asset_root = Path(asset_root)
    config = RunConfig(
        agent="minimal-loop",
        model=model,
        provider=getattr(transport, "provider", "injected"),
        model_version=model,
        transport=getattr(transport, "transport_name", "injected-callable"),
        max_steps=max_steps,
        replicate_id=replicate_id,
        schedule_index=schedule_index,
        asset_root=asset_root,
        artifacts_root=artifacts_root,
        experiment_id=experiment_id,
        output_path=output_path,
    )

    def execute(context: RunContext) -> list[dict[str, Any]]:
        initial = context.initial_payload
        base_url = context.base_url
        products = context.site_data["entities"]
        trace: list[dict[str, Any]] = []
        context.capture_partial_trace(trace)

        def dispatch(name: str, arguments: dict[str, Any]) -> str | list[dict[str, Any]]:
            if name == "search":
                query = arguments.get("query", "")
                page_text = _page_text(_http(base_url, "/search?q=" + urllib.parse.quote(query)))
                shown, _ = pages.search_results(products, query)
                shown, _, _ = pages.paged_results(shown, 1)
                return _product_blocks(page_text, shown, asset_root)
            if name == "open_product":
                item_id = arguments.get("item_id", "")
                page_text = _page_text(_http(base_url, "/product/" + urllib.parse.quote(item_id)))
                item = next((entry for entry in products if entry["item_id"] == item_id), None)
                return _product_blocks(page_text, [item] if item else [], asset_root)
            if name == "add_to_cart":
                return _page_text(_http(base_url, "/cart/add", {"item_id": arguments.get("item_id", "")}))
            if name == "remove_from_cart":
                return _page_text(_http(base_url, "/cart/remove", {"item_id": arguments.get("item_id", "")}))
            if name == "submit_cart":
                return _page_text(_http(base_url, "/submit", {}))
            if name == "ask_user":
                return context.ask_user(
                    arguments.get("slot", ""), arguments.get("question", "")
                )["payload"]
            return "Unknown tool."

        first_content: list[dict[str, Any]] = [{"type": "text", "text": initial["instruction"]}]
        observation_count = len(initial["observations"])
        for index, observation in enumerate(initial["observations"]):
            if observation_count == 1:
                label = "Images captured by the wearer's glasses, in chronological order:"
            elif index == 0:
                label = "Earlier images captured by the wearer's glasses:"
            else:
                label = "More recent images captured by the wearer's glasses:"
            first_content += _observation_blocks(observation, asset_root, label)
        first_content.append({
            "type": "text",
            "text": "Use the EgoMart tools to search its fixed product catalog.",
        })
        messages: list[dict[str, Any]] = [{"role": "user", "content": first_content}]
        tools = _tools(initial["clarification_available"])
        finished = False

        while not finished and context.current_step < max_steps:
            for event in context.advance():
                if event["type"] == "visual_context_update":
                    content = [{
                        "type": "text",
                        "text": "The glasses captured a more recent view:",
                    }]
                    content += _observation_blocks(
                        event["payload"], asset_root,
                        "More recent images captured by the wearer's glasses:",
                    )
                else:
                    content = [{"type": "text", "text": f"The wearer speaks again: {event['payload']}"}]
                messages.append({"role": "user", "content": content})

            request_body = {
                "model": model,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": messages,
                "tools": tools,
            }
            response = transport(request_body)
            trace.append({
                "step": context.current_step,
                "request": _audit_value(request_body),
                "response": _audit_value(response),
            })
            assistant_content = response.get("content", [])
            messages.append({"role": "assistant", "content": assistant_content})
            tool_uses = [block for block in assistant_content if block.get("type") == "tool_use"]
            if not tool_uses:
                break
            results = []
            for block in tool_uses:
                if block["name"] == "finish":
                    finished = True
                    results.append({
                        "type": "tool_result", "tool_use_id": block["id"], "content": "Run ended.",
                    })
                    continue
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": dispatch(block["name"], block.get("input", {})),
                })
            messages.append({"role": "user", "content": results})
        return trace

    return run_experiment(task, arm, config, execute)
