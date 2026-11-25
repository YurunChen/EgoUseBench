from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from threading import Lock

import pytest

from egocua import attribution_batch as batch_module
from egocua import attribution_judge
from egocua.attribution_judge import PROMPT_VERSION
from egocua.failure_attribution import PULSE_POLICY_VERSION
from egocua.attribution_batch import BatchOptions, prepare_judge_batch, run_judge_batch
from egocua.workspace import read_json_strict, write_json_atomic

CONFIG = {
    "model": "test-vision",
    "max_rounds": 2,
    "max_output_tokens": 128,
    "max_images_per_read": 2,
    "max_input_characters": 10000,
}


def _row(run_id: str, **overrides: object) -> dict:
    return {
        "run_id": run_id,
        "task_id": "task-" + run_id,
        "run_verdict": "valid_model_failure",
        "run_artifact": "artifacts/runs/" + run_id,
        **overrides,
    }


def _assessment(
    row: dict, usage: dict | None = None, request_log: list | None = None
) -> dict:
    return {
        "run_id": row["run_id"],
        "policy_version": PULSE_POLICY_VERSION,
        "checker_version": "3.0",
        "reviewer_kind": "llm_judge",
        "judge_model": CONFIG["model"],
        "prompt_version": PROMPT_VERSION,
        "usage": usage or {},
        "judgment": {
            "stage": "L",
            "step": 1,
            "confidence": "high",
            "reason": "The selected webpage object does not satisfy the stated requirement.",
            "evidence": [{"kind": "action", "index": 0}],
            "uncertainty": None,
        },
        "request_log": request_log or [],
    }



def _workspace(
    root: Path, rows: list[dict], reviews: list[dict] | None = None
) -> BatchOptions:
    write_json_atomic(
        root / batch_module.FAILURE_REPORT_PATH,
        {
            "attributions": rows,
            "judge_validation": {
                "accepted": True,
                "model": CONFIG["model"],
                "prompt_version": PROMPT_VERSION,
            },
        },
    )
    write_json_atomic(
        root / batch_module.FAILURE_EVIDENCE_PATH, {"assessments": reviews or []}
    )
    return BatchOptions(root=root, batch_id="test-batch")


def test_prepare_preserves_old_reviews_and_excludes_task_questions(
    tmp_path, monkeypatch
):
    options = _workspace(
        tmp_path,
        [_row("old"), _row("missing"), _row("task-question", run_verdict="unknown")],
        [{"run_id": "old"}],
    )
    monkeypatch.setattr(
        batch_module,
        "historical_review_context",
        lambda row, root: {"missing_evidence": ["historical_site_snapshot"]},
    )
    prepared = prepare_judge_batch(options)
    assert [item["run_id"] for item in prepared["candidates"]] == ["missing"]
    assert prepared["candidates"][0]["missing_evidence"] == ["historical_site_snapshot"]
    assert {item["reason"] for item in prepared["skipped"]} == {
        "existing_assessment",
        "unknown",
    }


def test_scope_does_not_implicitly_rejudge_and_unknown_ids_fail(tmp_path, monkeypatch):
    _workspace(tmp_path, [_row("old")], [{"run_id": "old"}])
    scoped = BatchOptions(root=tmp_path, batch_id="scope", run_ids=frozenset({"old"}))
    assert prepare_judge_batch(scoped)["candidates"] == []
    unknown = BatchOptions(
        root=tmp_path, batch_id="scope", run_ids=frozenset({"absent"})
    )
    with pytest.raises(ValueError, match="unknown report"):
        prepare_judge_batch(unknown)


def test_explicit_affected_selects_old_review(tmp_path, monkeypatch):
    _workspace(tmp_path, [_row("old")], [{"run_id": "old"}])
    monkeypatch.setattr(
        batch_module,
        "historical_review_context",
        lambda row, root: {"missing_evidence": []},
    )
    options = BatchOptions(
        root=tmp_path, batch_id="affected", affected_run_ids=frozenset({"old"})
    )
    assert [item["run_id"] for item in prepare_judge_batch(options)["candidates"]] == [
        "old"
    ]


def test_missing_live_model_fails_before_transport(tmp_path):
    options = _workspace(tmp_path, [_row("run")])
    with pytest.raises(ValueError, match="model"):
        run_judge_batch(
            options, {**CONFIG, "model": None}, lambda: pytest.fail("no transport")
        )


def test_independent_calls_close_and_local_failure_is_not_retried(
    tmp_path, monkeypatch
):
    options = _workspace(tmp_path, [_row("one"), _row("two"), _row("broken")])
    calls, closed, saved = [], [], []
    lock = Lock()

    class Judge:
        def __init__(self, config, transport):
            self.transport = transport

        def assess(self, row, root):
            with lock:
                calls.append(row["run_id"])
            if row["run_id"] == "broken":
                error = RuntimeError("request disconnected")
                error.request_log = [{"response": "partial"}]
                raise error
            return _assessment(row, {"input_tokens": 5}, [{"response": "judged"}])

    def transport_factory():
        transport = SimpleNamespace()
        transport.session = SimpleNamespace(close=lambda: closed.append(transport))
        return transport

    monkeypatch.setattr(attribution_judge, "TrajectoryJudge", Judge)
    monkeypatch.setattr(
        batch_module,
        "save_assessment",
        lambda root, row, assessment, **kwargs: saved.append(assessment),
    )
    first = run_judge_batch(options, CONFIG, transport_factory)
    assert sorted(calls) == ["broken", "one", "two"]
    assert len(closed) == 3
    assert len(saved) == 2
    assert all("request_log" not in item for item in saved)
    assert first["results"]["broken"]["status"] == "failed"
    assert read_json_strict(tmp_path / first["results"]["broken"]["archive"])[
        "request_log"
    ] == [{"response": "partial"}]
    run_judge_batch(options, CONFIG, transport_factory)
    assert len(calls) == 3
    retry = BatchOptions(root=tmp_path, batch_id="test-batch", resume_failed=True)
    repeated = run_judge_batch(retry, CONFIG, transport_factory)
    assert calls.count("broken") == 2
    attempts = repeated["results"]["broken"]["attempts"]
    assert len(attempts) == 2
    assert all((tmp_path / item).exists() for item in attempts)


def test_transport_creation_failure_does_not_abort_other_runs(tmp_path, monkeypatch):
    options = _workspace(tmp_path, [_row("run")])

    def broken_factory():
        raise RuntimeError("invalid configuration")

    result = run_judge_batch(options, CONFIG, broken_factory)
    assert result["results"]["run"]["status"] == "failed"


def test_interrupt_stops_new_calls_saves_reply_and_resumes_without_repeating(tmp_path, monkeypatch):
    import signal

    _workspace(tmp_path, [_row(str(i)) for i in range(5)])
    options = BatchOptions(root=tmp_path, batch_id="test-batch", concurrency=1)
    calls = []
    original_handler = signal.getsignal(signal.SIGINT)

    class Judge:
        def __init__(self, config, transport):
            pass

        def assess(self, row, root):
            calls.append(row["run_id"])
            if len(calls) == 1:
                signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return _assessment(row)

    monkeypatch.setattr(attribution_judge, "TrajectoryJudge", Judge)
    monkeypatch.setattr(batch_module, "save_assessment", lambda *args, **kwargs: None)
    def factory():
        return SimpleNamespace(session=SimpleNamespace(close=lambda: None))
    first = run_judge_batch(options, CONFIG, factory)
    assert first["interrupted"] and calls == ["0"]
    assert first["results"]["0"]["status"] == "completed"
    assert signal.getsignal(signal.SIGINT) == original_handler
    resumed = run_judge_batch(options, CONFIG, factory)
    assert not resumed["interrupted"]
    assert calls == [str(i) for i in range(5)]
    assert len(resumed["results"]) == 5


def test_insufficient_balance_stops_before_submitting_more_and_keeps_resume_scope(tmp_path, monkeypatch):
    _workspace(tmp_path, [_row(str(i)) for i in range(4)])
    options = BatchOptions(root=tmp_path, batch_id="test-batch", concurrency=1)
    calls = []

    class Judge:
        def __init__(self, config, transport):
            pass

        def assess(self, row, root):
            calls.append(row["run_id"])
            if row["run_id"] == "0":
                raise RuntimeError('{"error":{"code":"insufficient_balance"}}')
            return _assessment(row)

    monkeypatch.setattr(attribution_judge, "TrajectoryJudge", Judge)
    monkeypatch.setattr(batch_module, "save_assessment", lambda *args, **kwargs: None)

    def factory():
        return SimpleNamespace(session=SimpleNamespace(close=lambda: None))

    first = run_judge_batch(options, CONFIG, factory)
    assert calls == ["0"]
    assert first["interrupted"] and first["stop_reason"] == "provider_insufficient_balance"
    resumed = run_judge_batch(options, CONFIG, factory)
    assert calls == [str(i) for i in range(4)]
    assert not resumed["interrupted"] and "stop_reason" not in resumed
    assert resumed["results"]["0"]["status"] == "failed"


def test_batch_rejects_changed_settings(tmp_path, monkeypatch):
    options = _workspace(tmp_path, [])
    run_judge_batch(options, CONFIG)
    with pytest.raises(ValueError, match="settings changed"):
        run_judge_batch(options, {**CONFIG, "max_rounds": 3})


def test_batch_allows_only_one_writer(tmp_path):
    import fcntl

    options = _workspace(tmp_path, [])
    options.directory.mkdir(parents=True)
    with (options.directory / ".writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="active writer"):
            run_judge_batch(options, CONFIG)


def test_current_qualification_overrides_stale_report(tmp_path):
    from egocua.failure_attribution import attribute_run
    from tests.test_failure_attribution import _checked, _record

    row = attribute_run(_checked(run_id="pending"), _record())
    options = _workspace(tmp_path, [row])
    write_json_atomic(
        tmp_path / batch_module.FAILURE_EVIDENCE_PATH,
        {
            "assessments": [],
            "run_qualifications": {
                "pending": {
                    "verdict": "task_or_site_defect",
                    "reason": "alternative answer needs review",
                }
            },
        },
    )
    prepared = prepare_judge_batch(options)
    assert prepared["candidates"] == []
    assert prepared["skipped"] == [{"run_id": "pending", "reason": "unknown"}]


def test_bulk_does_not_require_human_validation(tmp_path):
    options = _workspace(tmp_path, [])
    result = run_judge_batch(options, CONFIG, lambda: pytest.fail("no cases"))
    assert result["results"] == {}


@pytest.mark.parametrize("matching", [True, False])
def test_reuses_only_same_model_and_prompt_outputs(tmp_path, monkeypatch, matching):
    options = _workspace(tmp_path, [_row("run")])
    path = tmp_path / batch_module.FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(path)
    assessment = _assessment(_row("run"))
    if not matching:
        assessment["judge_model"] = "other-model"
    evidence["judge_validation"] = {"prompt_version": PROMPT_VERSION, "assessments": [assessment]}
    write_json_atomic(path, evidence)
    saved = []
    monkeypatch.setattr(batch_module, "save_assessment", lambda root, row, item: saved.append(item))
    from egocua.attribution_judge import JudgeConfig
    reused = batch_module._reuse_validation_assessments([_row("run")], options, JudgeConfig.from_dict(CONFIG))
    assert reused == ({"run"} if matching else set())
    assert saved == ([assessment] if matching else [])
    assert read_json_strict(path)["judge_validation"]["assessments"] == [assessment]


def test_representation_change_reuses_old_canonical_judgment_with_provenance(tmp_path, monkeypatch):
    options = _workspace(tmp_path, [_row("run")])
    path = tmp_path / batch_module.FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(path)
    assessment = _assessment(_row("run"))
    assessment["prompt_version"] = "trajectory-judge-v15"
    evidence["judge_validation"] = {
        "prompt_version": "trajectory-judge-v15", "assessments": [assessment]
    }
    write_json_atomic(path, evidence)
    saved = []
    monkeypatch.setattr(batch_module, "save_assessment", lambda root, row, item: saved.append(item))
    from egocua.attribution_judge import JudgeConfig
    assert batch_module._reuse_validation_assessments([_row("run")], options, JudgeConfig.from_dict(CONFIG)) == {"run"}
    assert saved[0]["prompt_version"] == "trajectory-judge-v15"
    assert saved[0]["judgment"] == assessment["judgment"]
    assert read_json_strict(path)["judge_validation"]["assessments"] == [assessment]


def test_validation_selection_does_not_replace_current_review(tmp_path, monkeypatch):
    _workspace(
        tmp_path, [_row("run")], [{"run_id": "run", "reviewer_kind": "coding_agent"}]
    )
    path = tmp_path / batch_module.FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(path)
    evidence["judge_validation"] = {
        "calibration_run_ids": ["run"],
        "validation_run_ids": [],
        "assessments": [],
    }
    write_json_atomic(path, evidence)
    monkeypatch.setattr(
        batch_module,
        "historical_review_context",
        lambda row, root: {"missing_evidence": []},
    )
    options = BatchOptions(
        root=tmp_path, batch_id="validation", validation_phase="calibration"
    )
    assert len(prepare_judge_batch(options)["candidates"]) == 1
    evidence["judge_validation"]["assessments"] = [{"run_id": "run"}]
    write_json_atomic(path, evidence)
    assert prepare_judge_batch(options)["candidates"] == []
    assert read_json_strict(path)["assessments"] == [
        {"run_id": "run", "reviewer_kind": "coding_agent"}
    ]


def test_prepare_validation_freezes_fifty_source_separated_runs(tmp_path, monkeypatch):
    from egocua.attribution_batch import prepare_judge_validation

    options = _workspace(
        tmp_path, [_row(f"run-{i}", track="graft", site="egomart") for i in range(60)]
    )
    for i in range(60):
        (tmp_path / "artifacts/runs" / f"run-{i}").mkdir(parents=True)
    monkeypatch.setattr(
        batch_module,
        "load_run_record",
        lambda path: {"task_snapshot": {"id": path.name}},
    )
    monkeypatch.setattr(
        batch_module, "task_recording_group", lambda task, root: task["id"]
    )
    selected = prepare_judge_validation(options)
    assert selected["model"] is None
    assert len(selected["calibration_run_ids"]) == 10
    assert len(selected["validation_run_ids"]) == 40
    assert set(selected["calibration_run_ids"]).isdisjoint(
        selected["validation_run_ids"]
    )
    assert prepare_judge_validation(options) == selected
    assert (
        read_json_strict(tmp_path / batch_module.FAILURE_EVIDENCE_PATH)["assessments"]
        == []
    )


def test_prepare_validation_reports_insufficient_evidence(tmp_path, monkeypatch):
    from egocua.attribution_batch import prepare_judge_validation

    options = _workspace(tmp_path, [])
    with pytest.raises(ValueError, match="validation requires"):
        prepare_judge_validation(options)


def test_cli_prepare_does_not_load_model_configuration(tmp_path, monkeypatch, capsys):
    from tools.audit.judge_failure_attribution import main

    _workspace(tmp_path, [])
    code = main(
        [
            "--root",
            str(tmp_path),
            "--batch-id",
            "offline",
            "--prepare",
            "--config",
            "does-not-exist.json",
        ]
    )
    assert code == 0
    assert '"candidates": 0' in capsys.readouterr().out
    assert (tmp_path / "artifacts/attribution-judges/offline/preparation.json").exists()


def test_duplicate_report_rows_are_rejected(tmp_path):
    options = _workspace(tmp_path, [_row("same"), _row("same")])
    with pytest.raises(ValueError, match="duplicate run"):
        prepare_judge_batch(options)


def test_validation_calls_use_separate_record_and_model_binding(tmp_path, monkeypatch):
    _workspace(tmp_path, [_row("run")], [{"run_id": "run"}])
    path = tmp_path / batch_module.FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(path)
    evidence["judge_validation"] = {
        "calibration_run_ids": ["run"],
        "validation_run_ids": [],
        "assessments": [],
    }
    write_json_atomic(path, evidence)
    bound, saved = [], []
    monkeypatch.setattr(
        batch_module,
        "register_judge_validation",
        lambda root, selection, **kwargs: bound.append(kwargs),
    )
    monkeypatch.setattr(
        batch_module,
        "save_judge_validation_assessment",
        lambda root, row, assessment: saved.append(assessment),
    )
    monkeypatch.setattr(
        batch_module,
        "save_assessment",
        lambda *args, **kwargs: pytest.fail("must preserve main assessments"),
    )

    class Judge:
        def __init__(self, config, transport):
            pass

        def assess(self, row, root):
            return _assessment(row)

    monkeypatch.setattr(attribution_judge, "TrajectoryJudge", Judge)
    validation_options = BatchOptions(
        root=tmp_path, batch_id="independent", validation_phase="calibration"
    )
    result = run_judge_batch(
        validation_options,
        CONFIG,
        lambda: SimpleNamespace(session=SimpleNamespace(close=lambda: None)),
    )
    assert result["results"]["run"]["status"] == "completed"
    assert bound == [{"model": CONFIG["model"], "prompt_version": PROMPT_VERSION}]
    assert len(saved) == 1
    assert read_json_strict(path)["assessments"] == [{"run_id": "run"}]


def test_workers_run_concurrently_and_only_main_thread_saves(tmp_path, monkeypatch):
    from threading import Barrier, current_thread

    options = _workspace(tmp_path, [_row(f"run-{i}") for i in range(4)])
    barrier = Barrier(4, timeout=2)
    worker_names, save_names = set(), set()

    class Judge:
        def __init__(self, config, transport):
            pass

        def assess(self, row, root):
            worker_names.add(current_thread().name)
            barrier.wait()
            return _assessment(row)

    monkeypatch.setattr(attribution_judge, "TrajectoryJudge", Judge)
    monkeypatch.setattr(
        batch_module,
        "save_assessment",
        lambda *args, **kwargs: save_names.add(current_thread().name),
    )
    result = run_judge_batch(
        options,
        CONFIG,
        lambda: SimpleNamespace(session=SimpleNamespace(close=lambda: None)),
    )
    assert len(worker_names) == 4
    assert save_names == {current_thread().name}
    assert all(item["status"] == "completed" for item in result["results"].values())


@pytest.fixture
def successful_judge(monkeypatch):
    class Judge:
        def __init__(self, config, transport):
            pass

        def assess(self, row, root):
            return _assessment(
                row, {"input_tokens": 37}, [{"response": "completed judgement"}]
            )

    monkeypatch.setattr(attribution_judge, "TrajectoryJudge", Judge)
    return lambda: SimpleNamespace(session=SimpleNamespace(close=lambda: None))


def test_response_is_archived_before_canonical_save_failure(
    tmp_path, monkeypatch, successful_judge
):
    options = _workspace(tmp_path, [_row("run")])

    def rejected_save(root, row, assessment, **kwargs):
        archives = list(options.directory.glob("run-*.json"))
        assert len(archives) == 1
        raw = read_json_strict(archives[0])
        assert raw["status"] == "response_received"
        assert raw["assessment"]["request_log"] == [{"response": "completed judgement"}]
        assert raw["assessment"]["usage"] == {"input_tokens": 37}
        raise ValueError("invalid evidence reference")

    monkeypatch.setattr(batch_module, "save_assessment", rejected_save)
    result = run_judge_batch(options, CONFIG, successful_judge)
    archived = read_json_strict(tmp_path / result["results"]["run"]["archive"])
    assert archived["status"] == "failed"
    assert archived["error"] == "invalid evidence reference"
    assert archived["request_log"] == [{"response": "completed judgement"}]
    assert archived["assessment"]["usage"] == {"input_tokens": 37}


def test_interrupted_batch_write_retains_response_and_reuses_canonical_result(
    tmp_path, monkeypatch, successful_judge
):
    options = _workspace(tmp_path, [_row("run")])
    batch_writes = 0

    def interrupted_write(path, value):
        nonlocal batch_writes
        if path.name == "batch.json":
            batch_writes += 1
            if batch_writes == 2:
                raise RuntimeError("interrupted before batch reference was written")
        write_json_atomic(path, value)

    def canonical_save(root, row, assessment, **kwargs):
        path = root / batch_module.FAILURE_EVIDENCE_PATH
        evidence = read_json_strict(path)
        evidence["assessments"].append(assessment)
        write_json_atomic(path, evidence)

    monkeypatch.setattr(batch_module, "save_assessment", canonical_save)
    monkeypatch.setattr(batch_module, "write_json_atomic", interrupted_write)
    with pytest.raises(RuntimeError, match="interrupted before batch"):
        run_judge_batch(options, CONFIG, successful_judge)
    archives = list(options.directory.glob("run-*.json"))
    assert len(archives) == 1
    original_archive = archives[0].read_bytes()
    archived = read_json_strict(archives[0])
    assert archived["request_log"] == [{"response": "completed judgement"}]
    assert archived["assessment"]["usage"] == {"input_tokens": 37}
    assert read_json_strict(options.directory / "batch.json")["results"] == {}
    resumed = run_judge_batch(
        options, CONFIG, lambda: pytest.fail("must not repeat the model call")
    )
    assert resumed["skipped"] == [{"run_id": "run", "reason": "existing_assessment"}]
    assert archives[0].read_bytes() == original_archive


def test_holdout_freezes_before_first_request_but_not_for_empty_pending(
    tmp_path, monkeypatch, successful_judge
):
    _workspace(tmp_path, [_row("run")])
    path = tmp_path / batch_module.FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(path)
    evidence["judge_validation"] = {
        "calibration_run_ids": [],
        "validation_run_ids": ["run"],
        "assessments": [],
        "model": CONFIG["model"],
        "prompt_version": PROMPT_VERSION,
        "validation_started": False,
    }
    write_json_atomic(path, evidence)
    monkeypatch.setattr(
        batch_module, "register_judge_validation", lambda *args, **kwargs: None
    )
    saved = []
    monkeypatch.setattr(
        batch_module,
        "save_judge_validation_assessment",
        lambda *args: saved.append(True),
    )

    def transport_factory():
        assert read_json_strict(path)["judge_validation"]["validation_started"] is True
        return successful_judge()

    options = BatchOptions(
        root=tmp_path, batch_id="holdout", validation_phase="validation"
    )
    result = run_judge_batch(options, CONFIG, transport_factory)
    assert result["results"]["run"]["status"] == "completed"
    assert saved == [True]
    monkeypatch.setattr(
        batch_module,
        "begin_judge_validation",
        lambda root: pytest.fail("empty pending must not freeze"),
    )
    run_judge_batch(
        options, CONFIG, lambda: pytest.fail("completed calls must be reused")
    )


def test_new_validation_uses_unexposed_recording_sources(tmp_path, monkeypatch):
    from egocua.attribution_batch import prepare_judge_validation

    options = _workspace(
        tmp_path,
        [_row(f"run-{i}", track="graft", site="egomart") for i in range(120)],
        [{"run_id": "main-review"}],
    )
    for i in range(120):
        (tmp_path / "artifacts/runs" / f"run-{i}").mkdir(parents=True)
    monkeypatch.setattr(
        batch_module,
        "load_run_record",
        lambda path: {"task_snapshot": {"id": path.name}},
    )
    monkeypatch.setattr(
        batch_module, "task_recording_group", lambda task, root: "source-" + task["id"]
    )
    previous = prepare_judge_validation(options)
    path = tmp_path / batch_module.FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(path)
    evidence["judge_validation"]["validation_started"] = True
    calibration_id = previous["calibration_run_ids"][0]
    evidence["judge_validation"]["assessments"] = [{"run_id": calibration_id}]
    write_json_atomic(path, evidence)
    exposed = batch_module.judge_exposed_sources(evidence)
    fresh = prepare_judge_validation(
        BatchOptions(root=tmp_path, batch_id="fresh", new_validation=True)
    )
    assert len(fresh["calibration_run_ids"]) == 10
    assert len(fresh["validation_run_ids"]) == 40
    assert set(fresh["source_groups"].values()).isdisjoint(exposed)
    actual = read_json_strict(path)
    assert actual["judge_validation_history"][0]["validation_started"] is True
    assert actual["assessments"] == [{"run_id": "main-review"}]
    assert actual["judge_validation"]["model"] is None


def test_new_validation_requires_prepare_command(tmp_path):
    from tools.audit.judge_failure_attribution import main

    with pytest.raises(SystemExit) as error:
        main(["--root", str(tmp_path), "--batch-id", "new", "--new-validation"])
    assert error.value.code == 2


def test_batch_keeps_minimal_judgment_opaque(tmp_path, monkeypatch, successful_judge):
    row = _row("run")
    options = _workspace(tmp_path, [row])
    saved = []
    monkeypatch.setattr(
        batch_module,
        "save_assessment",
        lambda root, source, assessment, **kwargs: saved.append(assessment),
    )
    result = run_judge_batch(options, CONFIG, successful_judge)
    expected = _assessment(row, {"input_tokens": 37})
    expected.pop("request_log")
    assert saved == [expected]
    assert set(saved[0]["judgment"]) == {
        "stage",
        "step",
        "confidence",
        "reason",
        "evidence",
        "uncertainty",
    }
    assert not any(key.startswith("candidate_") for key in saved[0])
    archived = read_json_strict(tmp_path / result["results"]["run"]["archive"])
    assert archived["assessment"] == expected
    assert "judgment" not in archived
    assert archived["request_log"] == [{"response": "completed judgement"}]


def test_batch_cannot_mix_prompt_versions(tmp_path, monkeypatch):
    options = _workspace(tmp_path, [])
    run_judge_batch(options, CONFIG)
    monkeypatch.setattr(attribution_judge, "PROMPT_VERSION", "new-prompt")
    with pytest.raises(ValueError, match="settings changed"):
        run_judge_batch(options, CONFIG, lambda: pytest.fail("no model request"))


def test_changed_validation_batch_does_not_reset_existing_experiment(tmp_path, monkeypatch):
    options = _workspace(tmp_path, [])
    path = tmp_path / batch_module.FAILURE_EVIDENCE_PATH
    evidence = read_json_strict(path)
    evidence["judge_validation"] = {"calibration_run_ids": [], "assessments": []}
    write_json_atomic(path, evidence)
    monkeypatch.setattr(batch_module, "register_judge_validation", lambda *args, **kwargs: None)
    options = BatchOptions(root=tmp_path, batch_id="frozen", validation_phase="calibration")
    run_judge_batch(options, CONFIG)
    monkeypatch.setattr(attribution_judge, "PROMPT_VERSION", "new-prompt")
    monkeypatch.setattr(batch_module, "register_judge_validation", lambda *args, **kwargs: pytest.fail("must not reset"))
    with pytest.raises(ValueError, match="settings changed"):
        run_judge_batch(options, CONFIG)


def test_resume_recovers_archived_response_without_another_paid_call(tmp_path, monkeypatch):
    row = _row("interrupted")
    options = _workspace(tmp_path, [row])
    options.directory.mkdir(parents=True)
    response = {"status": "response_received", "run_id": row["run_id"],
                "task_id": row["task_id"], "assessment": _assessment(row)}
    write_json_atomic(options.directory / "run-interrupted-orphan.json", response)
    saved = []
    monkeypatch.setattr(batch_module, "save_assessment", lambda *args, **kwargs: saved.append(args[2]))
    batch = run_judge_batch(options, CONFIG, lambda: pytest.fail("already paid"))
    assert len(saved) == 1
    assert batch["results"]["interrupted"]["status"] == "completed"
    archive = read_json_strict(options.directory / "run-interrupted-orphan.json")
    assert archive["status"] == "completed"
