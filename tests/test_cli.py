from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from egocua.cli import main
from tests.helpers import products, materialize_assets, materialize_task_set, task, validation_evidence


class CliTest(unittest.TestCase):
    def test_run_minimal_rejects_unapproved_task_before_model_call(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_path = root / "evidence.json"
            materialize_assets(benchmark_task, products("graft"), root)
            task_root = materialize_task_set([benchmark_task], root)
            evidence_path.write_text(json.dumps({
                "schema_version": 1,
                "tasks": {
                    benchmark_task["id"]: validation_evidence("graft", status="candidate")
                }
            }))
            with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
                main([
                    "run-minimal", str(task_root), benchmark_task["id"], "a_only",
                    "--root", str(root), "--output", str(root / "record.json"),
                    "--evidence", str(evidence_path),
                ])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
