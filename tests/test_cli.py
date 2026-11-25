from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pytest

from egocua.cli import _recording_groups, main
from egocua.experiment import load_experiment_plan
from egocua.lineage import LineageError
from tests.helpers import products, materialize_assets, task


def test_score_uses_unique_historical_tasks_without_loading_current_files(tmp_path):
    snapshot = task("graft")
    records = [
        {"task_id": snapshot["id"], "task_snapshot": snapshot},
        {"task_id": snapshot["id"], "task_snapshot": snapshot},
    ]
    with (
        patch("egocua.cli.load_run_record", side_effect=records),
        patch("egocua.cli.load_tasks", side_effect=AssertionError("current task read")),
        patch("egocua.cli.load_experiment_plan") as plan,
        patch("egocua.cli.validate_run_records") as validate,
        patch(
            "egocua.cli.check_run",
            return_value={"evaluator": {"version": "test", "predicate": "final_state"}},
        ) as check,
        patch("egocua.cli.attribute_run", return_value={}),
        patch("egocua.cli.aggregate", return_value={}),
        patch("egocua.cli.source_intervals", return_value={}),
        patch("egocua.cli.task_recording_group", return_value="fixture"),
    ):
        main(
            [
                "score",
                "deleted-task-directory",
                "run-a",
                "run-b",
                "--output",
                str(tmp_path / "results.json"),
            ]
        )
        validate.assert_called_once_with([snapshot], records, plan.return_value)
        assert check.call_count == 2
        check.assert_called_with(snapshot, records[-1])


def test_recording_groups_fall_back_only_when_the_dossier_is_gone():
    missing, broken = task("graft"), task("refresh")

    def recording(item, root):
        if item is missing:
            raise FileNotFoundError(item["id"])
        raise LineageError(f"{item['id']}: source record identity mismatch")

    with patch("egocua.cli.task_recording_group", side_effect=recording):
        assert _recording_groups([missing], ".") == {missing["id"]: missing["id"]}
        with pytest.raises(LineageError, match="identity mismatch"):
            _recording_groups([broken], ".")


def test_synthesis_summary_uses_the_workflow_status(capsys):
    report = {
        "summary": {
            "tasks": 3,
            "status_counts": {"needs_repair": 1, "needs_review": 1, "ready": 1},
            "ready": 1,
            "remaining": 2,
        },
        "tasks": {
            "a": {"next_action": "repair_task"},
            "b": {"next_action": "coding_agent_assess"},
            "c": {"next_action": "build_release"},
        },
    }
    with patch("egocua.cli.SynthesisWorkflow") as workflow:
        workflow.return_value.status.return_value = report
        main(["synthesize", "status", "--summary", "--task-id", "a"])
        workflow.return_value.status.assert_called_once_with(
            ["a"],
            composite=False,
            update_count=False,
        )

    assert json.loads(capsys.readouterr().out) == report["summary"]


def _freeze_task(definition: dict, directory: Path) -> None:
    path = (
        directory
        / "tasks"
        / "formal"
        / definition["track"]
        / f"{definition['id']}.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(definition))


@pytest.mark.parametrize(
    ("command", "adapter", "extra"),
    [
        ("run-human", "egocua.adapters.human.run_task", ["--evaluator", "tester"]),
        ("run-minimal", "egocua.agents.minimal_loop.run_task", []),
    ],
)
def test_manual_runs_take_every_setting_from_the_frozen_experiment(
    tmp_path, command, adapter, extra
):
    plan = load_experiment_plan(Path("configs/pilot.json"))
    benchmark_task = task("graft")
    _freeze_task(benchmark_task, tmp_path)
    with (
        patch("egocua.frozen_run.load_experiment_plan", return_value=plan),
        patch("egocua.cli.validate_task_for_human_run"),
        patch("egocua.cli.validate_task_for_model_run"),
        patch("egocua.adapters.openai_requests.OpenAIRequestsTransport.from_env"),
        patch(adapter, return_value={"final_state": {}, "meta": {}}) as run,
    ):
        main(
            [
                command,
                benchmark_task["id"],
                "online",
                "--plan",
                str(tmp_path / "plan.json"),
                "--arm",
                "formal",
                *extra,
            ]
        )
    kwargs = run.call_args.kwargs
    expected = plan.protocol_parameters(benchmark_task)
    if command == "run-human":
        expected["source_max_steps"] = expected.pop("max_steps")
    assert {key: kwargs[key] for key in expected} == expected
    if command == "run-human":
        assert kwargs["experiment_directory"] is None
        assert kwargs["experiment_id"] == "human-solvability"
        assert kwargs["source_study"] == str(tmp_path)
    else:
        assert kwargs["experiment_directory"] == tmp_path
        assert kwargs["experiment_id"] == plan.experiment_id
    if command == "run-minimal":
        assert kwargs["model"] == plan.model


@pytest.mark.parametrize("command", ["run-browser", "run-minimal", "run-human"])
def test_model_and_human_runs_require_a_frozen_experiment(command):
    with redirect_stderr(StringIO()), pytest.raises(SystemExit) as raised:
        main([command, "task-id", "online", "--evaluator", "tester"])
    assert raised.value.code == 2


class CliTest(unittest.TestCase):
    def test_run_minimal_rejects_missing_asset_before_model_call(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_assets(benchmark_task, products("graft"), root)
            _freeze_task(benchmark_task, root)
            missing_frame = root / benchmark_task["visual_packets"][0]["frame_paths"][0]
            missing_frame.unlink()
            plan = load_experiment_plan(Path("configs/pilot.json"))
            stderr = StringIO()
            with (
                patch("egocua.frozen_run.load_experiment_plan", return_value=plan),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as raised,
            ):
                main(
                    [
                        "run-minimal",
                        benchmark_task["id"],
                        "online",
                        "--plan",
                        str(root / "plan.json"),
                        "--arm",
                        "formal",
                        "--root",
                        str(root),
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(missing_frame.name, stderr.getvalue())

    def test_review_stop_without_server_exits_zero(self) -> None:
        from egocua.cli import _review_runtime_paths, _stop_review_process

        port = 58766
        pidfile, _ = _review_runtime_paths(port)
        pidfile.unlink(missing_ok=True)
        self.assertIn("没有记录中的审核网站", _stop_review_process(port))
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit) as raised:
            main(["review", "--port", str(port)])
        self.assertEqual(raised.exception.code, 2)


@pytest.mark.parametrize("identities", [[None], ["started server", None, None]])
def test_review_stop_only_signals_recorded_process(tmp_path, identities):
    from egocua.cli import _stop_review_process

    pidfile = tmp_path / "server.pid"
    pidfile.write_text(json.dumps({"pid": 12345, "identity": "started server"}))
    with (
        patch(
            "egocua.cli._review_runtime_paths", return_value=(pidfile, tmp_path / "log")
        ),
        patch("egocua.cli._process_identity", side_effect=identities),
        patch("egocua.cli.os.kill") as kill,
    ):
        _stop_review_process(58767)
    assert kill.call_count == (0 if identities == [None] else 1)
    assert not pidfile.exists()


@pytest.mark.parametrize(
    "record", ["12345", '{"pid": 12345, "identity": "old server"}']
)
def test_review_stop_rejects_unverified_or_recycled_pid(tmp_path, record):
    from egocua.cli import _stop_review_process
    from egocua.review_workspace import ReviewWorkspaceError

    pidfile = tmp_path / "server.pid"
    pidfile.write_text(record)
    with (
        patch(
            "egocua.cli._review_runtime_paths", return_value=(pidfile, tmp_path / "log")
        ),
        patch("egocua.cli._process_identity", return_value="new unrelated process"),
        patch("egocua.cli.os.kill") as kill,
        pytest.raises(ReviewWorkspaceError),
    ):
        _stop_review_process(58767)
    kill.assert_not_called()
    assert pidfile.read_text() == record


def test_review_stop_checks_identity_again_before_escalating(tmp_path):
    from egocua.cli import _stop_review_process

    pidfile = tmp_path / "server.pid"
    pidfile.write_text(json.dumps({"pid": 12345, "identity": "server"}))
    with (
        patch(
            "egocua.cli._review_runtime_paths", return_value=(pidfile, tmp_path / "log")
        ),
        patch("egocua.cli._process_identity", side_effect=["server", "recycled"]),
        patch("egocua.cli.os.kill") as kill,
    ):
        _stop_review_process(58767)
    assert kill.call_count == 1
    assert not pidfile.exists()


def test_review_stop_preserves_a_replaced_process_record(tmp_path):
    from egocua.cli import _stop_review_process

    pidfile = tmp_path / "server.pid"
    pidfile.write_text(json.dumps({"pid": 12345, "identity": "old server"}))
    replacement = {"pid": 12346, "identity": "new server"}

    def process_exited(pid):
        pidfile.write_text(json.dumps(replacement))
        return None

    with (
        patch(
            "egocua.cli._review_runtime_paths", return_value=(pidfile, tmp_path / "log")
        ),
        patch("egocua.cli._process_identity", side_effect=process_exited),
        patch("egocua.cli.os.kill") as kill,
    ):
        _stop_review_process(58767)
    kill.assert_not_called()
    assert json.loads(pidfile.read_text()) == replacement


if __name__ == "__main__":
    unittest.main()
