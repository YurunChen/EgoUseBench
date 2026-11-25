from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

from PIL import Image

from egocua.agents.minimal_loop import run_task
from egocua.checker import check_run
from egocua.cli import main
from egocua.experiment import expand_runs, validate_run_records
from egocua.metrics import aggregate
from egocua.schema import validate_dataset
from tests.helpers import (
    PLAN,
    products,
    release_ready_task,
    item_id,
    materialize_assets,
    materialize_site,
    materialize_task_set,
    plan_payload,
)


def _tool_use(name: str, arguments: dict | None = None, block_id: str = "tool") -> dict:
    return {"type": "tool_use", "id": block_id, "name": name, "input": arguments or {}}


def _transport(steps: list[list[dict]]):
    calls = iter(steps)

    def scripted(_body: dict) -> dict:
        return {"content": next(calls)}

    return scripted


def _script(track: str, condition_name: str) -> list[list[dict]]:
    prefix: list[list[dict]] = []
    if (track, condition_name) in {("graft", "online"), ("refresh", "online")}:
        prefix.append([_tool_use("search", {"query": "Product"}, "search")])
    if track == "clarify" and condition_name == "online":
        prefix.append(
            [
                _tool_use(
                    "ask_user",
                    {"slot": "variant", "question": "Which variant did you mean?"},
                    "ask",
                )
            ]
        )

    selected = {
        "graft": item_id("graft", 0),
        "refresh": item_id("refresh", 0)
        if condition_name == "initial_only"
        else item_id("refresh", 1),
        "clarify": item_id("clarify", 0)
        if condition_name == "no_ask"
        else item_id("clarify", 1),
    }[track]
    return prefix + [
        [_tool_use("add_to_cart", {"item_id": selected}, "cart")],
        [_tool_use("submit_cart", block_id="submit")],
        [_tool_use("finish", {"reason": "dry run complete"}, "finish")],
    ]


class ExperimentDryRunTest(unittest.TestCase):
    def test_three_track_matrix_produces_scoreable_run_artifacts(self) -> None:
        tasks = [release_ready_task(track) for track in ("graft", "refresh", "clarify")]
        names = {name for track in PLAN.track_conditions.values() for name in track}
        plan = replace(
            PLAN,
            max_steps=20,
            conditions={
                name: replace(
                    PLAN.condition(name), frame_interval_seconds=1,
                    frame_delivery_steps=1, batch_seconds=8,
                )
                for name in sorted(names)
            },
            track_probe_conditions={},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            report_path = root / "score.json"
            plan_path = root / "plan.json"
            plan_path.write_text(json.dumps(plan_payload(plan)))
            materialize_site(
                [item for current in tasks for item in products(current["track"])],
                root,
            )
            for current in tasks:
                for item in products(current["track"]):
                    path = root / item["image_path"]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"asset:{item['item_id']}".encode())
            for track_index, benchmark_task in enumerate(tasks):
                materialize_assets(
                    benchmark_task, products(benchmark_task["track"]), root
                )
                opening = benchmark_task["visual_packets"][0]["frame_paths"][0]
                later = benchmark_task["visual_packets"][-1]["frame_paths"][-1]
                for frame_index, relative in enumerate(dict.fromkeys((opening, later))):
                    Image.new("RGB", (64, 64), (track_index * 70, frame_index * 90, 40)).save(
                        root / relative
                    )
                benchmark_task["visual_session"] = {
                    "update_onset_seconds": 10,
                    "reviewed_seconds": [10],
                    "frequency_samples": {
                        "window_seconds": [0, 15.5],
                        "timestamps_seconds": list(range(16)),
                        "frame_paths": [opening] * 8 + [later] * 8,
                    },
                }
            task_root = materialize_task_set(tasks, root)
            self.assertEqual(
                validate_dataset(tasks, root=root, require_assets=True),
                {"clarify": 1, "graft": 1, "refresh": 1},
            )

            task_by_id = {
                benchmark_task["id"]: benchmark_task for benchmark_task in tasks
            }
            records = []
            for run in expand_runs(tasks, plan):
                benchmark_task = task_by_id[run["task_id"]]
                condition_name = run["condition"]
                records.append(
                    run_task(
                        benchmark_task,
                        plan.condition(condition_name),
                        model=plan.model,
                        max_steps=plan.max_steps,
                        replicate_id=run["replicate_id"],
                        schedule_index=run["schedule_index"],
                        asset_root=root,
                        output_root=artifacts,
                        experiment_id="matrix-dry-run",
                        default_release_step=plan.default_release_step,
                        reference_steps=plan.reference_steps,
                        transport=_transport(
                            _script(benchmark_task["track"], condition_name)
                        ),
                    )
                )

            validate_run_records(tasks, records, plan)
            run_directories = {
                Path(record["meta"]["artifact_directory"]) for record in records
            }
            self.assertEqual(len(run_directories), 27)
            for run_directory in run_directories:
                self.assertEqual(
                    json.loads((run_directory / "run.json").read_text())["archive"][
                        "status"
                    ],
                    "completed",
                )
                for filename in (
                    "runtime.log",
                    "run.json",
                ):
                    self.assertTrue((run_directory / filename).is_file())

            checked = [
                check_run(task_by_id[record["task_id"]], record) for record in records
            ]
            metrics = aggregate(checked)
            self.assertEqual(metrics["common"]["valid_run_rate"], 1.0)
            self.assertEqual(
                metrics["paired_effects"]["graft"]["refine"]["upfront_minus_online"],
                0.0,
            )
            self.assertEqual(
                metrics["paired_effects"]["refresh"]["replace"][
                    "full_minus_initial_only"
                ],
                1.0,
            )
            self.assertEqual(
                metrics["paired_effects"]["refresh"]["replace"]["upfront_minus_online"],
                0.0,
            )
            self.assertEqual(
                metrics["paired_effects"]["clarify"]["resolve"]["allowed_minus_denied"],
                1.0,
            )
            self.assertEqual(metrics["clarification_policy"]["ask_recall"], 1.0)
            self.assertEqual(
                metrics["clarification_policy"]["valid_question_rate"], 1.0
            )

            record_paths = [record["meta"]["artifact_directory"] for record in records]
            with redirect_stdout(StringIO()):
                main(
                    [
                        "score",
                        str(task_root),
                        *record_paths,
                        "--output",
                        str(report_path),
                        "--experiment-id",
                        "matrix-dry-run",
                        "--plan",
                        str(plan_path),
                    ]
                )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["metrics"]["runs"], 27)
            self.assertEqual(report["metrics"]["common"]["valid_run_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
