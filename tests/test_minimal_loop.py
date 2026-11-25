from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from typing import Any

from egocua.agents.minimal_loop import run_task
from egocua.run_ledger import read_run_identity
from tests.conditions import CONDITIONS
from tests.helpers import products, item_id, materialize_assets, task


def _tool_use(name: str, arguments: dict | None = None, block_id: str = "tu_1") -> dict:
    return {"type": "tool_use", "id": block_id, "name": name, "input": arguments or {}}


def _scripted_transport(steps: list[list[dict]]):
    calls: dict[str, Any] = {"count": 0, "messages": [], "bodies": []}

    def transport(body: dict) -> dict:
        calls["bodies"].append(body)
        calls["messages"].append(body["messages"])
        index = min(calls["count"], len(steps) - 1)
        calls["count"] += 1
        return {"content": steps[index]}

    return transport, calls


def _materialize(benchmark_task, root: Path) -> None:
    materialize_assets(benchmark_task, products(benchmark_task["track"]), root)


class MinimalLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        # These scripted fixtures reconstruct pre-stream input for archival checks.
        self.enterContext(patch(
            "egocua.agents.minimal_loop.prepare_live_task",
            side_effect=lambda definition, _condition, _config: definition,
        ))

    def test_cli_style_run_writes_canonical_artifact_directory(self) -> None:
        benchmark_task = task("clarify")
        transport, _ = _scripted_transport(
            [
                [_tool_use("finish", {"reason": "done"})],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _materialize(benchmark_task, root)
            record = run_task(
                benchmark_task,
                CONDITIONS["upfront"],
                asset_root=root,
                transport=transport,
                output_root=root / "artifacts",
                experiment_id="test-pilot",
                default_release_step=2,
                reference_steps=None,
            )
            run_directory = Path(record["meta"]["artifact_directory"])
            manifest = read_run_identity(run_directory)

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["run_id"], record["meta"]["run_id"])
        self.assertEqual(record["task_snapshot"], benchmark_task)
        self.assertNotIn("provenance", manifest)

    def test_scripted_run_submits_and_records_metadata(self) -> None:
        benchmark_task = task("clarify")
        transport, calls = _scripted_transport(
            [
                [_tool_use("add_to_cart", {"item_id": item_id("clarify", 1)})],
                [_tool_use("submit_cart")],
                [_tool_use("finish", {"reason": "done"})],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _materialize(benchmark_task, root)
            record = run_task(
                benchmark_task,
                CONDITIONS["upfront"],
                asset_root=root,
                transport=transport,
                replicate_id=7,
                default_release_step=2,
                reference_steps=None,
            )
        self.assertEqual(
            record["final_state"]["selected_entity_ids"], [item_id("clarify", 1)]
        )
        self.assertTrue(record["final_state"]["submission_completed"])
        self.assertEqual(record["meta"]["agent"], "minimal-loop")
        self.assertEqual(record["meta"]["provider"], "injected")
        self.assertEqual(record["meta"]["transport"], "injected-callable")
        self.assertEqual(record["meta"]["replicate_id"], 7)
        self.assertIsNone(record["meta"]["provider_seed"])
        self.assertEqual(len(record["trace"]), 3)
        self.assertEqual(
            record["trace"][0]["response"]["content"][0]["name"], "add_to_cart"
        )
        self.assertEqual(record["task_snapshot"], benchmark_task)
        first_message = calls["messages"][0][0]
        self.assertTrue(
            any(block.get("type") == "image" for block in first_message["content"])
        )
        ask_user = next(
            tool for tool in calls["bodies"][0]["tools"] if tool["name"] == "ask_user"
        )
        self.assertIn("ask before choosing; do not guess", calls["bodies"][0]["system"])
        self.assertEqual(ask_user["input_schema"]["required"], ["question"])
        self.assertNotIn("slot", ask_user["input_schema"]["properties"])
        self.assertIn("wait", [tool["name"] for tool in calls["bodies"][0]["tools"]])

    def test_invalid_item_id_is_returned_to_agent_for_retry(self) -> None:
        benchmark_task = task("clarify")
        transport, calls = _scripted_transport(
            [
                [_tool_use("add_to_cart", {"item_id": "Product 1"}, "invalid")],
                [_tool_use("add_to_cart", {"item_id": item_id("clarify", 1)}, "valid")],
                [_tool_use("submit_cart", block_id="submit")],
                [_tool_use("finish", {"reason": "done"}, "finish")],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _materialize(benchmark_task, root)
            record = run_task(
                benchmark_task,
                CONDITIONS["upfront"],
                asset_root=root,
                transport=transport,
                default_release_step=2,
                reference_steps=None,
            )

        tool_results = [
            block
            for message in calls["messages"][-1]
            if message["role"] == "user"
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        invalid_result = next(
            block for block in tool_results if block["tool_use_id"] == "invalid"
        )
        self.assertIn("HTTP 404", invalid_result["content"])
        self.assertEqual(
            record["final_state"]["selected_entity_ids"], [item_id("clarify", 1)]
        )
        self.assertTrue(record["final_state"]["submission_completed"])

    def test_stream_event_arrives_as_multimodal_message(self) -> None:
        benchmark_task = task("refresh")
        transport, calls = _scripted_transport(
            [
                [_tool_use("search", {"query": "Product"})],
                [_tool_use("add_to_cart", {"item_id": item_id("refresh", 1)})],
                [_tool_use("submit_cart")],
                [_tool_use("finish", {"reason": "done"})],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _materialize(benchmark_task, root)
            record = run_task(
                benchmark_task,
                CONDITIONS["online"],
                asset_root=root,
                transport=transport,
                default_release_step=2,
                reference_steps=None,
            )
        events = [
            a for a in record["actions"] if a.get("type") == "visual_context_update"
        ]
        self.assertEqual(len(events), 1)
        searches = [a for a in record["actions"] if a.get("type") == "search"]
        self.assertGreaterEqual(events[0]["step"], 2)
        self.assertLess(searches[0]["step"], events[0]["step"])
        self.assertEqual(record["meta"]["event_delivery_steps"], [events[0]["step"]])
        update_messages = [
            message
            for run_messages in calls["messages"][-1:]
            for message in run_messages
            if message["role"] == "user"
            and any(
                "glasses images received before step 2"
                in str(block.get("text", "")).lower()
                for block in message["content"]
                if isinstance(block, dict)
            )
        ]
        self.assertEqual(len(update_messages), 1)
        self.assertTrue(
            any(
                block.get("type") == "image"
                for block in update_messages[0]["content"]
                if isinstance(block, dict)
            )
        )
        visible_text = " ".join(
            str(block.get("text", ""))
            for message in calls["messages"][-1]
            for block in message["content"]
            if isinstance(block, dict)
        )
        self.assertNotIn("(A)", visible_text)
        self.assertNotIn("(B)", visible_text)
        self.assertNotIn("V0", visible_text)
        self.assertNotIn("V1", visible_text)
        self.assertNotIn("keyframe", visible_text.lower())
        self.assertTrue(record["final_state"]["submission_completed"])
        self.assertEqual(
            record["final_state"]["selected_entity_ids"], [item_id("refresh", 1)]
        )

    def test_search_tool_returns_product_images(self) -> None:
        benchmark_task = task("graft")
        transport, calls = _scripted_transport(
            [
                [_tool_use("search", {"query": "Product 0"})],
                [_tool_use("finish", {"reason": "inspected"})],
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _materialize(benchmark_task, root)
            image_path = root / f"products/{item_id('graft', 0)}.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"\xff\xd8product-jpeg")
            run_task(
                benchmark_task,
                CONDITIONS["initial_only"],
                asset_root=root,
                transport=transport,
                default_release_step=2,
                reference_steps=None,
            )

        tool_results = [
            block
            for message in calls["messages"][-1]
            if message["role"] == "user"
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "tool_result"
        ]
        search_content = tool_results[0]["content"]
        self.assertTrue(any(block.get("type") == "image" for block in search_content))
        self.assertIn(f"item_id={item_id('graft', 0)}", search_content[0]["text"])


if __name__ == "__main__":
    unittest.main()
