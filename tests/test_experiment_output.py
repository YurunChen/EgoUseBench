import json
import re

from egocua.experiment_output import create_experiment_directory
from egocua.run_ledger import RunLedger, inspect_run_artifact, load_run_record
from tests.conditions import CONDITIONS
from tests.helpers import complete_run_record


def test_experiment_name_and_local_timezone(tmp_path):
    plan = {"category": "变量对照", "variable": "画面频率", "model": "qwen3.5-flash"}
    directory = create_experiment_directory(tmp_path, plan)
    assert re.fullmatch(
        r"\d{8}-\d{6}-\d{6}_变量对照_画面频率_qwen3\.5-flash", directory.name
    )
    assert json.loads((directory / "plan.json").read_text())["created_at"].endswith(
        "+08:00"
    )
    assert create_experiment_directory(tmp_path, plan) != directory


def test_multiple_runs_share_one_frozen_experiment_without_duplicate_exports(tmp_path):
    directory = create_experiment_directory(
        tmp_path,
        {
            "category": "任务试跑",
            "variable": "新增任务",
            "model": "offline",
        },
    )
    plan_bytes = (directory / "plan.json").read_bytes()
    ledgers = []
    for _ in range(2):
        ledger = RunLedger.start(
            tmp_path,
            experiment_id="test",
            task_id="task",
            condition=CONDITIONS["online"],
            replicate_id=0,
            agent="browsergym",
            experiment_directory=directory,
        )
        assert inspect_run_artifact(ledger.directory)["status"] == "running"
        record = complete_run_record(
            {
                "task_id": "task",
                "actions": [{"type": "search", "step": 1}],
                "meta": {
                    "agent": "browsergym",
                    "run_id": ledger.run_id,
                    "artifact_directory": str(ledger.directory),
                },
            }
        )
        ledger.complete(record)
        assert set(p.name for p in ledger.directory.iterdir()) == {
            "run.json",
            "runtime.log",
        }
        assert load_run_record(ledger.directory)["actions"] == record["actions"]
        assert inspect_run_artifact(ledger.directory)["event_count"] == 1
        assert record["archive"]["finished_at"].endswith("+08:00")
        ledgers.append(ledger)
    assert ledgers[0].directory != ledgers[1].directory
    assert (directory / "plan.json").read_bytes() == plan_bytes
    assert len(list(tmp_path.iterdir())) == 1
