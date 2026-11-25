from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import pytest

from egocua.run_ledger import (
    RunLedger,
    inspect_run_artifact,
    load_run_record,
    canonical_sha256,
    validate_run_record_shape,
)
from tests.conditions import CONDITIONS
from tests.helpers import complete_run_meta, complete_run_record


def test_historical_run_with_source_record_digest_remains_readable(tmp_path):
    ledger = RunLedger.start(
        tmp_path,
        experiment_id="pilot",
        task_id="historical-task",
        condition=CONDITIONS["online"],
        replicate_id=0,
        agent="browsergym",
    )
    record = complete_run_record(
        {
            "task_id": "historical-task",
            "task_snapshot": {
                "id": "historical-task",
                "source": {"task_id": "historical-task", "record_sha256": "a" * 64},
            },
            "meta": {
                "agent": "browsergym",
                "run_id": ledger.run_id,
                "artifact_directory": ledger.directory.as_posix(),
            },
        }
    )
    ledger.complete(record)
    run_path = ledger.directory / "run.json"
    manifest_path = ledger.directory / "manifest.json"
    current = json.loads(run_path.read_text())
    manifest = {**ledger.identity}
    manifest["started_at_utc"] = manifest.pop("started_at")
    manifest["finished_at_utc"] = manifest.pop("finished_at")
    current.pop("archive")
    current["task_snapshot"] = record["task_snapshot"]
    run_path.write_text(json.dumps(current))
    manifest["record_sha256"] = canonical_sha256(json.loads(run_path.read_text()))
    manifest_path.write_text(json.dumps(manifest))
    original_run, original_manifest = run_path.read_bytes(), manifest_path.read_bytes()

    loaded = load_run_record(ledger.directory)

    assert loaded["task_snapshot"]["source"]["record_sha256"] == "a" * 64
    assert json.loads(manifest_path.read_text())["record_sha256"] == canonical_sha256(
        loaded
    )
    assert run_path.read_bytes() == original_run
    assert manifest_path.read_bytes() == original_manifest


@pytest.mark.parametrize("meta", [None, [], "metadata"])
def test_run_identity_validation_rejects_non_object_metadata(
    tmp_path: Path, meta: object
) -> None:
    ledger = RunLedger.start(
        tmp_path,
        experiment_id="pilot",
        task_id="graft-01",
        condition=CONDITIONS["online"],
        replicate_id=0,
        agent="browsergym",
    )
    record = complete_run_record({"task_id": "graft-01"})
    record["meta"] = meta

    with pytest.raises(ValueError, match="record meta must be an object"):
        ledger.complete(record)


class RunLedgerTest(unittest.TestCase):
    def test_live_state_rejects_malformed_observations(self) -> None:
        state: Any
        for state in (
            None,
            [],
            {},
            {"selected_entity_ids": [], "fields": {}, "submitted": True},
            {"selected_entity_ids": "item", "fields": {}},
            {"selected_entity_ids": [None], "fields": {}},
            {"selected_entity_ids": [], "fields": {"quantity": 2}},
        ):
            with (
                self.subTest(state=state),
                self.assertRaisesRegex(ValueError, "live_state"),
            ):
                validate_run_record_shape(
                    complete_run_record({"live_state": state}), require_complete=True
                )

    def test_completed_run_still_requires_all_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger.start(
                Path(directory),
                experiment_id="pilot",
                task_id="graft-01",
                condition=CONDITIONS["online"],
                replicate_id=0,
                agent="browsergym",
            )
            record = complete_run_record(
                {
                    "task_id": "graft-01",
                    "meta": {
                        "agent": "browsergym",
                        "run_id": ledger.run_id,
                        "artifact_directory": ledger.directory.as_posix(),
                    },
                }
            )
            del record["meta"]["model"]

            with self.assertRaisesRegex(ValueError, "missing=.*model"):
                ledger.complete(record)

    def test_record_condition_must_match_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger.start(
                Path(directory),
                experiment_id="pilot",
                task_id="graft-01",
                condition=CONDITIONS["online"],
                replicate_id=0,
                agent="browsergym",
            )

            with self.assertRaisesRegex(ValueError, "condition does not match"):
                ledger.complete({"condition": CONDITIONS["upfront"].as_dict()})

    def test_completed_run_is_written_as_an_auditable_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger.start(
                Path(directory),
                experiment_id="pilot",
                task_id="clarify-01",
                condition=CONDITIONS["online"],
                replicate_id=2,
                agent="minimal-loop",
            )
            record = {
                "task_id": "clarify-01",
                "track": "clarify",
                "condition": CONDITIONS["online"].as_dict(),
                "run_status": {"valid": True, "reason": None},
                "initial_state": {
                    "selected_entity_ids": [],
                    "fields": {},
                    "submission_completed": False,
                },
                "steps": 1,
                "actions": [{"type": "search", "step": 1}],
                "final_state": {
                    "selected_entity_ids": ["item-1"],
                    "fields": {},
                    "submission_completed": True,
                },
                "trace": [{"step": 1}],
                "meta": complete_run_meta(
                    replicate_id=2,
                    agent="minimal-loop",
                    run_id=ledger.run_id,
                    artifact_directory=ledger.directory.as_posix(),
                ),
            }
            ledger.complete(record)

            self.assertFalse((ledger.directory / "manifest.json").exists())
            self.assertTrue((ledger.directory / "run.json").is_file())
            self.assertFalse((ledger.directory / "trace.json").exists())
            self.assertFalse((ledger.directory / "events.jsonl").exists())
            manifest = json.loads((ledger.directory / "run.json").read_text())[
                "archive"
            ]
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(
                json.loads((ledger.directory / "run.json").read_text())["meta"][
                    "run_id"
                ],
                ledger.run_id,
            )
            summary = inspect_run_artifact(ledger.directory)
            self.assertEqual(summary["event_count"], 1)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["event_count"], len(record["actions"]))

    def test_failed_run_preserves_partial_record_and_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = RunLedger.start(
                Path(directory),
                experiment_id="pilot",
                task_id="graft-01",
                condition=CONDITIONS["online"],
                replicate_id=0,
                agent="browsergym",
            )
            record = {
                "task_id": "graft-01",
                "track": "graft",
                "condition": CONDITIONS["online"].as_dict(),
                "initial_state": {
                    "selected_entity_ids": [],
                    "fields": {},
                    "submission_completed": False,
                },
                "steps": 0,
                "actions": [],
                "trace": [],
                "run_status": {"valid": False, "reason": "infrastructure_error"},
                "final_state": {
                    "selected_entity_ids": [],
                    "fields": {},
                    "submission_completed": False,
                },
                "meta": complete_run_meta(
                    replicate_id=0,
                    agent="browsergym",
                    run_id=ledger.run_id,
                    artifact_directory=ledger.directory.as_posix(),
                ),
            }
            error = RuntimeError("provider unavailable")
            ledger.fail(record, error)

            manifest = json.loads((ledger.directory / "run.json").read_text())[
                "archive"
            ]
            failure = json.loads((ledger.directory / "run.json").read_text())[
                "archive"
            ]["error"]
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(failure["type"], "RuntimeError")
            self.assertIn("provider unavailable", failure["message"])
            summary = inspect_run_artifact(ledger.directory)
            self.assertEqual(summary["error"]["type"], "RuntimeError")
