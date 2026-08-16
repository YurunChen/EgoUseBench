from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from egocua.run_ledger import RunLedger, inspect_run_artifact


class RunLedgerTest(unittest.TestCase):
    def test_completed_run_is_written_as_an_auditable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger.start(
                Path(directory), experiment_id="pilot", task_id="clarify-01",
                arm="interactive", replicate_id=2, agent="minimal-loop",
            )
            record = {
                "task_id": "clarify-01",
                "arm": "interactive",
                "actions": [{"type": "search", "step": 1}],
                "final_state": {
                    "selected_entity_ids": ["item-1"],
                    "fields": {},
                    "submission_completed": True,
                },
                "trace": [{"step": 1}],
                "meta": {"replicate_id": 2},
            }
            ledger.complete(record)

            self.assertTrue((ledger.directory / "manifest.json").is_file())
            self.assertTrue((ledger.directory / "run.json").is_file())
            self.assertTrue((ledger.directory / "trace.json").is_file())
            self.assertTrue((ledger.directory / "events.jsonl").is_file())
            manifest = json.loads((ledger.directory / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["run_id"], ledger.run_id)
            summary = inspect_run_artifact(ledger.directory)
            self.assertEqual(summary["event_count"], 1)
            self.assertEqual(summary["status"], "completed")

    def test_failed_run_preserves_partial_record_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger.start(
                Path(directory), experiment_id="pilot", task_id="graft-01",
                arm="injected", replicate_id=0, agent="browser-use",
            )
            record = {
                "task_id": "graft-01",
                "arm": "injected",
                "actions": [],
                "trace": [],
                "run_status": {"valid": False, "reason": "infrastructure_error"},
                "final_state": {
                    "selected_entity_ids": [],
                    "fields": {},
                    "submission_completed": False,
                },
            }
            error = RuntimeError("provider unavailable")
            ledger.fail(record, error)

            manifest = json.loads((ledger.directory / "manifest.json").read_text())
            failure = json.loads((ledger.directory / "error.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(failure["type"], "RuntimeError")
            self.assertIn("provider unavailable", failure["message"])
            summary = inspect_run_artifact(ledger.directory / "run.json")
            self.assertEqual(summary["error"]["type"], "RuntimeError")
