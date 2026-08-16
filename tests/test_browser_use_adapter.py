from __future__ import annotations

import asyncio
import json
import re
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from unittest.mock import patch

from egocua.adapters.browser_use import BrowserUseUnavailable, run_task
from tests.helpers import products, item_id, materialize_site, task


class BrowserUseArtifactTest(unittest.TestCase):
    def test_import_failure_still_writes_a_run_artifact(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            for packet in benchmark_task["visual_packets"]:
                for relative_path in packet["frame_paths"]:
                    path = root / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"frame")

            with patch(
                "egocua.adapters.browser_use._imports",
                side_effect=BrowserUseUnavailable("browser-use unavailable"),
            ), self.assertRaises(BrowserUseUnavailable):
                asyncio.run(run_task(
                    benchmark_task, "a_only", asset_root=root,
                    artifacts_root=root / "artifacts", experiment_id="test-pilot",
                ))

            run_directories = list((root / "artifacts" / "runs").glob("**/manifest.json"))
            self.assertEqual(len(run_directories), 1)
            manifest = json.loads(run_directories[0].read_text())
            error = json.loads((run_directories[0].parent / "error.json").read_text())

        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(error["type"], "BrowserUseUnavailable")

    def test_success_path_delivers_follow_up_and_submits_store_state(self) -> None:
        benchmark_task = task("graft")
        created_agents = []
        created_tools = []
        browser_kwargs = []
        llm_kwargs = []

        class FakeHistory:
            def model_dump(self, *, mode: str) -> dict:
                return {"mode": mode, "history": ["completed"]}

        class FakeTools:
            def __init__(self) -> None:
                self.excluded: list[str] = []
                created_tools.append(self)

            def exclude_action(self, action_name: str) -> None:
                self.excluded.append(action_name)

        class FakeBrowser:
            def __init__(self, **kwargs) -> None:
                browser_kwargs.append(kwargs)

            async def kill(self) -> None:
                return None

        class FakeChatOpenAI:
            def __init__(self, **kwargs) -> None:
                llm_kwargs.append(kwargs)

        class FakeAgent:
            def __init__(self, *, task: str, **_kwargs) -> None:
                self.prompt = task
                self.follow_ups: list[str] = []
                created_agents.append(self)

            def add_new_task(self, message: str) -> None:
                self.follow_ups.append(message)

            async def run(self, *, on_step_start, max_steps: int):
                self.assert_budget(max_steps)
                match = re.search(r"http://127\.0\.0\.1:\d+", self.prompt)
                if match is None:
                    raise AssertionError("EgoMart URL missing from Browser-Use prompt")
                base_url = match.group(0)
                await on_step_start(self)
                urlopen(base_url + "/search?q=Product").read()
                await on_step_start(self)
                payload = urlencode({"item_id": item_id("graft", 0)}).encode()
                urlopen(Request(base_url + "/cart/add", data=payload, method="POST")).read()
                urlopen(Request(base_url + "/submit", data=b"", method="POST")).read()
                return FakeHistory()

            @staticmethod
            def assert_budget(max_steps: int) -> None:
                if max_steps != 40:
                    raise AssertionError(f"unexpected max_steps: {max_steps}")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            for packet in benchmark_task["visual_packets"]:
                for relative_path in packet["frame_paths"]:
                    path = root / relative_path
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"frame")

            with patch(
                "egocua.adapters.browser_use._imports",
                return_value=(
                    FakeAgent,
                    FakeBrowser,
                    FakeChatOpenAI,
                    FakeTools,
                    lambda: "/fake/google-chrome",
                ),
            ), patch("egocua.adapters.browser_use.load_dotenv") as load_dotenv:
                record = asyncio.run(run_task(
                    benchmark_task,
                    "injected",
                    model="fake-openai",
                    provider_seed=17,
                    asset_root=root,
                    artifacts_root=root / "artifacts",
                    experiment_id="browser-success",
                ))
                load_dotenv.assert_called_once_with(root.resolve() / ".env", override=False)

            manifest = json.loads(
                (Path(record["meta"]["artifact_directory"]) / "manifest.json").read_text()
            )

        injection = [
            action for action in record["actions"]
            if action.get("type") == "instruction_injection"
        ]
        self.assertEqual(len(injection), 1)
        self.assertGreaterEqual(injection[0]["step"], 2)
        self.assertEqual(record["final_state"]["selected_entity_ids"], [item_id("graft", 0)])
        self.assertTrue(record["final_state"]["submission_completed"])
        self.assertEqual(record["meta"]["provider_seed"], 17)
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(browser_kwargs[0]["executable_path"], "/fake/google-chrome")
        self.assertTrue(browser_kwargs[0]["headless"])
        self.assertFalse(browser_kwargs[0]["keep_alive"])
        self.assertFalse(browser_kwargs[0]["enable_default_extensions"])
        self.assertFalse(browser_kwargs[0]["captcha_solver"])
        self.assertTrue(llm_kwargs[0]["add_schema_to_system_prompt"])
        self.assertTrue(llm_kwargs[0]["dont_force_structured_output"])
        self.assertEqual(len(created_agents), 1)
        self.assertEqual(
            set(created_tools[0].excluded),
            {
                "search", "evaluate", "upload_file", "read_file", "write_file",
                "replace_file", "save_as_pdf",
            },
        )
        self.assertEqual(len(created_agents[0].follow_ups), 1)
        self.assertIn("wearer added this requirement", created_agents[0].follow_ups[0])
        self.assertNotIn("keyframe", created_agents[0].prompt.lower())
        self.assertNotIn("packet", created_agents[0].follow_ups[0].lower())
