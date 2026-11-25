from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from tools import run_human_reference


@pytest.fixture
def sample(tmp_path):
    path = tmp_path / "data/experiments/human-reference.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "selection_seed": 1,
                "tasks": [
                    {
                        "task_id": "case-1",
                        "stratum": "refresh:replace",
                        "hidden_future": "do not display",
                    }
                ],
            }
        )
    )
    return tmp_path


def test_listing_does_not_launch_or_show_materials(sample, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["human-reference", "--root", str(sample)])
    monkeypatch.setattr(
        run_human_reference, "make_server", lambda *a, **k: pytest.fail("launched")
    )
    run_human_reference.main()
    output = capsys.readouterr().out
    assert "任务 01" in output and "case-1" not in output
    assert "refresh:replace" not in output and "do not display" not in output


@pytest.mark.parametrize("preview", [False, True])
def test_launcher_serves_independent_choices_and_keeps_preview_separate(
    sample, monkeypatch, preview
):
    calls = []
    argv = ["human-reference", "--root", str(sample), "--item", "1", "--no-open"]
    argv += ["--preview"] if preview else ["--evaluator", "person-01"]
    monkeypatch.setattr(sys, "argv", argv)
    workspace = SimpleNamespace(case=lambda item: calls.append(("case", item)))
    monkeypatch.setattr(
        run_human_reference,
        "AnswerValidationWorkspace",
        lambda *a: calls.append(("workspace", a)) or workspace,
    )
    monkeypatch.setattr(
        run_human_reference,
        "create_answer_validation_app",
        lambda *a, **k: calls.append(("app", k)) or "app",
    )
    server = SimpleNamespace(
        server_port=19001,
        serve_forever=lambda: None,
        server_close=lambda: calls.append(("closed",)),
    )
    monkeypatch.setattr(run_human_reference, "make_server", lambda *a, **k: server)
    run_human_reference.main()
    args = calls[0][1]
    assert args[1] == ("preview-not-participant" if preview else "person-01")
    assert args[2].is_relative_to(
        sample
        / (".scratch/人工答案核验" if preview else "artifacts/human-answer-validation")
    )
    assert ("case", 1) in calls and ("closed",) in calls


@pytest.mark.parametrize(
    "extra",
    [
        ["--item", "2", "--evaluator", "person"],
        ["--item", "1"],
        ["--preview", "--evaluator", "person"],
    ],
)
def test_invalid_launch_cannot_start_server(sample, monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["human-reference", "--root", str(sample), *extra])
    monkeypatch.setattr(
        run_human_reference, "make_server", lambda *a, **k: pytest.fail("launched")
    )
    with pytest.raises(SystemExit) as result:
        run_human_reference.main()
    assert result.value.code == 2


@pytest.mark.parametrize(
    "answers, expected",
    [
        ("\n\n", ["--preview", "--item", "1"]),
        ("person-02\n3\n", ["--item", "3", "--evaluator", "person-02"]),
    ],
)
@pytest.mark.requires_local_data("启动人工执行.command")
def test_double_click_launcher_uses_project_directory_and_existing_entry(
    tmp_path, answers, expected
):
    import os
    import subprocess
    from pathlib import Path

    root = tmp_path / "project with spaces"
    (root / ".venv/bin").mkdir(parents=True)
    launcher = root / "启动人工执行.command"
    launcher.write_text(
        (Path(__file__).resolve().parents[1] / "启动人工执行.command").read_text()
    )
    python = root / ".venv/bin/python"
    python.write_text('#!/bin/sh\npwd\nprintf "%s\\n" "$@"\n')
    python.chmod(0o755)
    result = subprocess.run(
        ["zsh", str(launcher)],
        input=answers,
        text=True,
        capture_output=True,
        cwd=tmp_path,
        env=os.environ,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert str(root) in lines
    assert lines[-len(expected) :] == expected
    assert "-m" in lines and "tools.run_human_reference" in lines
