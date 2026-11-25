from copy import deepcopy

import pytest

from egocua import run_inspection


@pytest.fixture
def record(monkeypatch):
    record = {
        "run_status": {"valid": True},
        "initial_state": {"selected_entity_ids": []},
        "live_state": {"selected_entity_ids": ["actual"]},
        "task_snapshot": {"id": "historical", "goal": "frozen"},
        "meta": {"model": "recorded-model", "run_id": "recorded-run"},
        "initial_context": {"instruction": "Original request"},
        "actions": [{"type": "select_entity", "entity_id": "actual"}],
        "trace": {
            "agent_settings": {"system_prompt": "Recorded system prompt"},
            "wearer_image_layouts": [{"image_index": 0}],
            "llm_requests": [{"step": 1}],
            "history": [
                {
                    "metadata": {"step_number": 1},
                    "state_message": "Original observation",
                    "state": {"screenshot_path": "step-1.png"},
                }
            ],
        },
    }
    monkeypatch.setattr(run_inspection, "load_run_record", lambda path: record)
    return record


def test_input_view_uses_recorded_text_and_references_without_mutation(record):
    original = deepcopy(record)
    result = run_inspection.read_run_view("archive", "input")
    assert result["initial_context"]["instruction"] == "Original request"
    assert result["agent_settings"]["system_prompt"] == "Recorded system prompt"
    assert result["observations"][0]["text"] == "Original observation"
    assert result["observations"][0]["website"]["screenshot_path"] == "step-1.png"
    assert record == original


def test_actions_and_result_views_use_the_canonical_record(record):
    assert run_inspection.read_run_view("archive", "actions") == record["actions"]
    assert run_inspection.read_run_view("archive", "trace") == record["trace"]
    assert (
        run_inspection.read_run_view("archive", "result")["live_state"]
        == record["live_state"]
    )


def test_score_reads_saved_results_without_rescoring(record, tmp_path):
    from egocua.workspace import write_json_atomic

    directory = tmp_path / "runs" / "recorded-run"
    directory.mkdir(parents=True)
    row = {"run_id": "recorded-run", "e2e_success": True, "evaluator_version": "frozen"}
    write_json_atomic(tmp_path / "results.json", {"runs": [row]})
    del record["task_snapshot"]
    assert run_inspection.read_run_view(directory, "score") == [row]


def test_missing_score_is_not_silently_recomputed(record, tmp_path):
    directory = tmp_path / "runs" / "recorded-run"
    directory.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        run_inspection.read_run_view(directory, "score")


def test_unknown_view_fails_before_loading_an_archive():
    with pytest.raises(ValueError, match="unknown run view"):
        run_inspection.read_run_view("does-not-exist", "other")


def test_historical_score_file_can_be_selected_without_moving_the_run(record, tmp_path):
    from egocua.workspace import write_json_atomic

    results = tmp_path / "historical-results.json"
    row = {"run_id": "recorded-run", "e2e_success": False}
    write_json_atomic(results, {"runs": [row]})
    before = results.read_bytes()
    assert run_inspection.read_run_view("old/archive", "score", results) == [row]
    assert results.read_bytes() == before
