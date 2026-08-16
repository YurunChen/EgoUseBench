from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from egocua.adapters.human import run_task as run_human
from egocua.execution import RunConfig, RunContext, run_experiment
from tests.helpers import products, item_id, materialize_site, task


def _post(base_url: str, path: str, form: dict[str, str] | None = None) -> None:
    payload = urlencode(form or {}).encode()
    urlopen(Request(base_url + path, data=payload, method="POST")).read()


class ExperimentExecutionTest(unittest.TestCase):
    def test_human_adapter_uses_the_same_dynamic_event_and_storefront(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            browser: dict[str, str] = {}
            prompts: list[str] = []

            def open_browser(url: str) -> bool:
                browser["base_url"] = url.removesuffix("/context")
                return True

            def command_reader(prompt: str) -> str:
                prompts.append(prompt)
                base_url = browser["base_url"]
                if len(prompts) == 1:
                    urlopen(base_url + "/search?q=Product").read()
                    return ""
                if len(prompts) == 2:
                    _post(base_url, "/cart/add", {"item_id": item_id("graft", 0)})
                    return ""
                _post(base_url, "/submit")
                return "finish"

            record = run_human(
                benchmark_task,
                "injected",
                evaluator_id="human-a",
                asset_root=root,
                artifacts_root=root / "artifacts",
                experiment_id="human-interface",
                command_reader=command_reader,
                message_writer=lambda _message: None,
                browser_opener=open_browser,
            )

        self.assertEqual(record["meta"]["operator_id"], "human-a")
        self.assertEqual(record["meta"]["transport"], "egocua-browser-terminal-v2")
        self.assertEqual(record["meta"]["event_delivery_steps"], [2])
        self.assertEqual(
            record["final_state"],
            {
                "selected_entity_ids": [item_id("graft", 0)],
                "fields": {},
                "submission_completed": True,
            },
        )

    def test_context_delivers_events_and_lifecycle_records_the_outcome(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                asset_root=root,
                artifacts_root=root / "artifacts",
                experiment_id="execution-interface",
            )

            def adapter(context: RunContext) -> list[dict]:
                self.assertEqual(context.current_step, 0)
                self.assertEqual(context.advance(), [])
                urlopen(context.base_url + "/search?q=Product").read()
                events = context.advance()
                self.assertEqual([event["type"] for event in events], ["instruction_injection"])
                _post(context.base_url, "/cart/add", {"item_id": item_id("graft", 0)})
                _post(context.base_url, "/submit")
                return [{"adapter": "fixture", "status": "complete"}]

            record = run_experiment(benchmark_task, "injected", config, adapter)
            run_directory = Path(record["meta"]["artifact_directory"])
            manifest = json.loads((run_directory / "manifest.json").read_text())

        self.assertEqual(record["meta"]["event_delivery_steps"], [2])
        self.assertEqual(record["final_state"]["selected_entity_ids"], [item_id("graft", 0)])
        self.assertTrue(record["final_state"]["submission_completed"])
        self.assertEqual(record["trace"], [{"adapter": "fixture", "status": "complete"}])
        self.assertEqual(manifest["status"], "completed")

    def test_adapter_failure_preserves_partial_trace_through_the_same_interface(self) -> None:
        benchmark_task = task("clarify")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("clarify"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                asset_root=root,
                artifacts_root=root / "artifacts",
                experiment_id="execution-failure",
            )

            def adapter(context: RunContext) -> list[dict]:
                context.capture_partial_trace([{"step": 0, "status": "request_started"}])
                raise RuntimeError("provider unavailable")

            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                run_experiment(benchmark_task, "oracle", config, adapter)

            run_directory = next((root / "artifacts" / "runs").glob("**/manifest.json")).parent
            manifest = json.loads((run_directory / "manifest.json").read_text())
            record = json.loads((run_directory / "run.json").read_text())
            error = json.loads((run_directory / "error.json").read_text())

        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(record["trace"], [{"step": 0, "status": "request_started"}])
        self.assertFalse(record["run_status"]["valid"])
        self.assertEqual(error["type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
