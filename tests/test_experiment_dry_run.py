from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from egocua.agents.minimal_loop import run_task
from egocua.checker import check_run
from egocua.cli import main
from egocua.experiment import expand_runs, validate_run_records
from egocua.gates import validate_release
from egocua.metrics import aggregate
from egocua.schema import validate_dataset
from tests.helpers import (
    PLAN,
    products,
    item_id,
    materialize_assets,
    materialize_site,
    materialize_task_set,
    task,
    validation_evidence,
)


def _tool_use(name: str, arguments: dict | None = None, block_id: str = "tool") -> dict:
    return {"type": "tool_use", "id": block_id, "name": name, "input": arguments or {}}


def _transport(steps: list[list[dict]]):
    calls = iter(steps)

    def scripted(_body: dict) -> dict:
        return {"content": next(calls)}

    return scripted


def _script(track: str, arm: str) -> list[list[dict]]:
    prefix: list[list[dict]] = []
    if (track, arm) in {("graft", "injected"), ("refresh", "stream")}:
        prefix.append([_tool_use("search", {"query": "Product"}, "search")])
    if track == "clarify" and arm == "interactive":
        prefix.append([
            _tool_use(
                "ask_user",
                {"slot": "variant", "question": "Which variant did you mean?"},
                "ask",
            )
        ])

    selected = {
        "graft": item_id("graft", 0),
        "refresh": item_id("refresh", 0) if arm == "stale" else item_id("refresh", 1),
        "clarify": item_id("clarify", 0) if arm == "forced_execute" else item_id("clarify", 1),
    }[track]
    return prefix + [
        [_tool_use("add_to_cart", {"item_id": selected}, "cart")],
        [_tool_use("submit_cart", block_id="submit")],
        [_tool_use("finish", {"reason": "dry run complete"}, "finish")],
    ]


class ExperimentDryRunTest(unittest.TestCase):
    def test_three_track_matrix_produces_scoreable_run_artifacts(self) -> None:
        tasks = [task(track) for track in ("graft", "refresh", "clarify")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            report_path = root / "score.json"
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps({
                "schema_version": 2,
                "experiment_id": PLAN.experiment_id,
                "arms": {track: list(arms) for track, arms in PLAN.arms.items()},
                "replicate_ids": list(PLAN.replicate_ids),
                "runtime": {
                    "adapter": PLAN.adapter,
                    "model": PLAN.model,
                    "model_version": PLAN.model_version,
                    "max_steps": PLAN.max_steps,
                },
                "scheduler_seed": PLAN.scheduler_seed,
            }))
            materialize_site(
                [item for current in tasks for item in products(current["track"])],
                root,
            )
            for current in tasks:
                for item in products(current["track"]):
                    path = root / item["image_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"asset:{item['item_id']}".encode())
            for benchmark_task in tasks:
                materialize_assets(benchmark_task, products(benchmark_task["track"]), root)
            task_root = materialize_task_set(tasks, root)
            self.assertEqual(
                validate_dataset(tasks, root=root, require_assets=True),
                {"clarify": 1, "graft": 1, "refresh": 1},
            )
            validate_release(
                tasks,
                {
                    current["id"]: validation_evidence(
                        current["track"], task_definition=current, root=root
                    )
                    for current in tasks
                },
                PLAN,
                root=root,
            )

            task_by_id = {benchmark_task["id"]: benchmark_task for benchmark_task in tasks}
            records = []
            for run in expand_runs(tasks, PLAN):
                benchmark_task = task_by_id[run["task_id"]]
                arm = run["arm"]
                records.append(run_task(
                    benchmark_task,
                    arm,
                    model=PLAN.model,
                    max_steps=PLAN.max_steps,
                    replicate_id=run["replicate_id"],
                    schedule_index=run["schedule_index"],
                    asset_root=root,
                    artifacts_root=artifacts,
                    experiment_id="matrix-dry-run",
                    transport=_transport(_script(benchmark_task["track"], arm)),
                ))

            validate_run_records(tasks, records, PLAN)
            run_directories = {
                Path(record["meta"]["artifact_directory"])
                for record in records
            }
            self.assertEqual(len(run_directories), 27)
            for run_directory in run_directories:
                self.assertEqual(
                    json.loads((run_directory / "manifest.json").read_text())["status"],
                    "completed",
                )
                for filename in ("runtime.jsonl", "events.jsonl", "trace.json", "run.json"):
                    self.assertTrue((run_directory / filename).is_file())

            checked = [check_run(task_by_id[record["task_id"]], record) for record in records]
            metrics = aggregate(checked)
            self.assertEqual(metrics["common"]["valid_run_rate"], 1.0)
            self.assertEqual(metrics["graft"]["diagnostic_interruption_cost"], 0.0)
            self.assertEqual(metrics["refresh"]["diagnostic_replace_adaptation_benefit"], 1.0)
            self.assertEqual(metrics["refresh"]["diagnostic_replace_online_refresh_cost"], 0.0)
            self.assertEqual(metrics["clarify"]["diagnostic_clarification_gain"], 1.0)
            self.assertEqual(metrics["clarify"]["ask_recall"], 1.0)
            self.assertEqual(metrics["clarify"]["valid_question_rate"], 1.0)

            record_paths = [str(Path(record["meta"]["artifact_directory"]) / "run.json") for record in records]
            with redirect_stdout(StringIO()):
                main([
                    "score",
                    str(task_root),
                    *record_paths,
                    "--output",
                    str(report_path),
                    "--experiment-id",
                    "matrix-dry-run",
                    "--plan",
                    str(plan_path),
                ])
            report = json.loads(report_path.read_text())
            self.assertEqual(report["metrics"]["runs"], 27)
            self.assertEqual(report["metrics"]["common"]["valid_run_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
