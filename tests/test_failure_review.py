from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

import egocua.failure_review as failure_review
from egocua.failure_review import FailureReviewWorkspace
from egocua.workspace import write_json_atomic


def test_final_adjudication_preserves_existing_reviews(tmp_path: Path) -> None:
    report_path = tmp_path / failure_review.FAILURE_REPORT_PATH
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({"attributions": [{
        "run_id": "run-1", "run_verdict": "valid_model_failure",
        "attribution_status": "evidence_insufficient",
    }]}))
    path = tmp_path / failure_review.FAILURE_EVIDENCE_PATH
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"assessments": [{"run_id": "run-1"}], "ratings": []}))
    decision = {"stage": "L", "reviewer_id": "user", "reason": "Summary-based review."}
    failure_review.save_adjudications(tmp_path, {"run-1": decision})
    saved = json.loads(path.read_text())
    assert saved["assessments"] == [{"run_id": "run-1"}]
    assert saved["adjudications"]["run-1"] == decision
    with pytest.raises(failure_review.FailureReviewError, match="already exists"):
        failure_review.save_adjudications(tmp_path, {"run-1": {**decision, "stage": "S"}})


def test_failure_review_is_blind_and_reviewer_specific(tmp_path: Path) -> None:
    report_path = tmp_path / failure_review.FAILURE_REPORT_PATH
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {
                "human_review": {
                    "packets": [
                        {
                            "run_id": "run-1",
                            "task_id": "task-1",
                            "model": "model-a",
                            "track": "graft",
                            "site": "egomart",
                            "observable_facts": ["target_opened"],
                            "secondary_symptoms": ["wrong_final_state"],
                        }
                    ]
                }
            }
        )
    )

    first = FailureReviewWorkspace.open(tmp_path, "rater-a")
    assert "pulse_stage" not in first.case("run-1")["packet"]
    first.submit("run-1", "L", "Opened a near neighbor without checking the detail.")

    own = FailureReviewWorkspace.open(tmp_path, "rater-a")
    other = FailureReviewWorkspace.open(tmp_path, "rater-b")
    assert own.overview()["submitted"] == 1
    assert other.overview()["submitted"] == 0
    assert other.overview()["todo"] == 1
    assert other.case("run-1")["rating"] is None


def test_failure_review_can_submit_unknown_without_inventing_a_primary_stage(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / failure_review.FAILURE_REPORT_PATH
    report_path.parent.mkdir(parents=True)
    report_path.write_text(
        json.dumps(
            {"human_review": {"packets": [{"run_id": "run-1", "task_id": "task-1"}]}}
        )
    )
    workspace = FailureReviewWorkspace.open(tmp_path, "rater-a")

    workspace.submit("run-1", None, "The available trace does not separate P from U.")

    assert workspace.case("run-1")["rating"]["pulse_stage"] is None


def _review_workspaces(
    root: Path,
) -> tuple[FailureReviewWorkspace, FailureReviewWorkspace]:
    write_json_atomic(
        root / failure_review.FAILURE_REPORT_PATH,
        {
            "human_review": {"packets": [{"run_id": "run-1", "task_id": "task-1"}]},
        },
    )
    return (
        FailureReviewWorkspace.open(root, "rater-a"),
        FailureReviewWorkspace.open(root, "rater-b"),
    )


def test_stale_workspaces_preserve_other_ratings_and_latest_evidence(
    tmp_path: Path,
) -> None:
    first, second = _review_workspaces(tmp_path)
    path = tmp_path / failure_review.FAILURE_EVIDENCE_PATH
    first.submit("run-1", "L", "First reviewer evidence.")
    latest = json.loads(path.read_text())
    latest["assessments"] = [{"run_id": "run-1", "review": "new evidence"}]
    write_json_atomic(path, latest)

    second.submit("run-1", "S", "Second reviewer evidence.")

    saved = json.loads(path.read_text())
    assert {row["rater_id"] for row in saved["ratings"]} == {"rater-a", "rater-b"}
    assert saved["assessments"] == latest["assessments"]


def test_concurrent_submissions_serialize_the_complete_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second = _review_workspaces(tmp_path)
    first_writing = Event()
    second_started = Event()
    second_writing = Event()
    release_first = Event()

    def controlled_write(path: Path, payload: dict) -> None:
        reviewer = payload["ratings"][-1]["rater_id"]
        if reviewer == "rater-a":
            first_writing.set()
            assert release_first.wait(5)
        else:
            second_writing.set()
        write_json_atomic(path, payload)

    def submit_second() -> None:
        second_started.set()
        second.submit("run-1", "S", "Second reviewer evidence.")

    monkeypatch.setattr(failure_review, "write_json_atomic", controlled_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending_first = executor.submit(
            first.submit, "run-1", "L", "First reviewer evidence."
        )
        try:
            assert first_writing.wait(5)
            pending_second = executor.submit(submit_second)
            assert second_started.wait(5)
            assert not second_writing.wait(0.05)
        finally:
            release_first.set()
        pending_first.result(timeout=5)
        pending_second.result(timeout=5)

    saved = json.loads((tmp_path / failure_review.FAILURE_EVIDENCE_PATH).read_text())
    assert {row["rater_id"] for row in saved["ratings"]} == {"rater-a", "rater-b"}


def test_unscorable_sources_remain_visible_without_review_packets(
    tmp_path: Path,
) -> None:
    write_json_atomic(
        tmp_path / failure_review.FAILURE_REPORT_PATH,
        {
            "source_exclusions": [
                {
                    "run_id": "old-run",
                    "task_id": "old-task",
                    "run_artifact": "artifacts/old-run",
                    "status": "not_rescorable",
                    "reasons": ["missing_terminal_live_state"],
                }
            ],
            "human_review": {"packets": []},
        },
    )
    overview = FailureReviewWorkspace.open(tmp_path, "reviewer").overview()
    assert overview["source_blocked"] == 1
    assert overview["total"] == overview["todo"] == overview["submitted"] == 0
    assert overview["source_exclusions"][0]["run_id"] == "old-run"


def test_human_review_saves_decisive_step_and_task_validity(tmp_path: Path) -> None:
    first, _ = _review_workspaces(tmp_path)
    first.submit("run-1", "L", "Wrong webpage entity.", first_decisive_step=12)
    assert first.case("run-1")["rating"]["first_decisive_step"] == 12
    first.submit(
        "run-1",
        None,
        "A reasonable answer may be missing.",
        run_verdict="task_or_site_defect",
    )
    assert first.case("run-1")["rating"]["run_verdict"] == "task_or_site_defect"
    with pytest.raises(failure_review.FailureReviewError, match="不填写失败类别"):
        first.submit(
            "run-1", "L", "Unconfirmed task issue.", run_verdict="task_or_site_defect"
        )


def test_review_overview_does_not_load_runs_and_case_loads_only_requested_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import egocua.failure_attribution as attribution
    from tests.helpers import complete_run_record, task

    definition = task("graft")
    packet = {
        "run_id": "run-1",
        "task_id": definition["id"],
        "track": "graft",
        "model": "model-a",
        "site": "egomart",
        "run_artifact": "artifacts/run",
        "scoring_goal": definition["goal"],
        "checker_version": "3.0",
        "pulse_stage": "P",
        "candidate_stage": "L",
    }
    write_json_atomic(
        tmp_path / failure_review.FAILURE_REPORT_PATH,
        {"human_review": {"packets": [packet]}},
    )
    calls = []
    record = complete_run_record(
        {
            "task_id": definition["id"],
            "track": "graft",
            "task_snapshot": definition,
            "meta": {"run_id": "run-1", "model": "model-a", "site": "egomart"},
            "actions": [],
            "steps": 1,
            "live_state": {"selected_entity_ids": [], "fields": {}},
            "initial_context": {
                "instruction": "Save the item.",
                "observations": [],
                "clarification_available": False,
            },
        }
    )

    def load(directory: Path) -> dict:
        calls.append(directory)
        return record

    monkeypatch.setattr(attribution, "load_run_record", load)
    workspace = FailureReviewWorkspace.open(tmp_path, "human")
    assert workspace.overview()["total"] == 1
    assert calls == []
    case = workspace.case("run-1")
    assert len(calls) == 1
    assert (
        case["packet"]["diagnostic"]["actual_initial_input"]["instruction"]
        == "Save the item."
    )
    assert "pulse_stage" not in case["packet"]
    assert "candidate_stage" not in case["packet"]
    assert "diagnostic" not in workspace.overview()["tasks"][0]


def test_validation_review_only_exposes_registered_heldout_packets(tmp_path):
    report = {'human_review': {'packets': [
        {'run_id': 'heldout', 'task_id': 'task-a', 'pulse_stage': 'P'},
        {'run_id': 'calibration', 'task_id': 'task-b', 'pulse_stage': 'U'},
    ]}}
    write_json_atomic(tmp_path / failure_review.FAILURE_REPORT_PATH, report)
    write_json_atomic(tmp_path / failure_review.FAILURE_EVIDENCE_PATH, {
        'ratings': [], 'judge_validation': {'validation_run_ids': ['heldout']},
    })
    workspace = FailureReviewWorkspace.open(tmp_path, 'new-reviewer', scope='validation')
    assert workspace.overview()['total'] == 1
    assert workspace.overview()['scope'] == 'validation'
    assert workspace.case('heldout')['scope'] == 'validation'
    assert 'pulse_stage' not in workspace.case('heldout')['packet']
    with pytest.raises(failure_review.FailureReviewError, match='unknown'):
        workspace.case('calibration')
    workspace.submit('heldout', None, 'The evidence does not identify a stage.')
    saved = json.loads((tmp_path / failure_review.FAILURE_EVIDENCE_PATH).read_text())
    assert saved['judge_validation']['validation_run_ids'] == ['heldout']
    assert len(saved['ratings']) == 1


def test_validation_review_requires_all_registered_materials(tmp_path):
    write_json_atomic(tmp_path / failure_review.FAILURE_REPORT_PATH,
                      {'human_review': {'packets': []}})
    write_json_atomic(tmp_path / failure_review.FAILURE_EVIDENCE_PATH,
                      {'ratings': [], 'judge_validation': {'validation_run_ids': ['missing']}})
    with pytest.raises(failure_review.FailureReviewError, match='缺少 1'):
        FailureReviewWorkspace.open(tmp_path, 'new-reviewer', scope='validation')


def test_review_hides_model_but_retains_internal_identity(tmp_path, monkeypatch):
    write_json_atomic(tmp_path / failure_review.FAILURE_REPORT_PATH, {
        "human_review": {"packets": [{
            "run_id": "blind", "task_id": "t", "model": "evaluated-model",
        }]}
    })
    monkeypatch.setattr(failure_review, "historical_review_context", lambda *args: {
        "run_identity": {"run_id": "blind", "model": "evaluated-model"},
        "missing_evidence": [],
    })
    workspace = FailureReviewWorkspace.open(tmp_path, "reviewer")
    assert workspace.packets["blind"]["model"] == "evaluated-model"
    assert "evaluated-model" not in json.dumps(workspace.case("blind"))
    assert "evaluated-model" not in json.dumps(workspace.overview())
    assert workspace.case("blind")["stages"][1]["zh"] == "意图与更新失败"


def test_packed_images_do_not_replace_case_identity(tmp_path, monkeypatch):
    from PIL import Image

    Image.new("RGB", (8, 8), "red").save(tmp_path / "frame.png")
    write_json_atomic(tmp_path / failure_review.FAILURE_REPORT_PATH, {
        "human_review": {"packets": [{"run_id": "case-id", "task_id": "task-id"}]}
    })
    monkeypatch.setattr(failure_review, "historical_review_context", lambda *args: {
        "task_context": {"visual_packets": [{
            "id": "V0", "released_at_step": 0, "frame_paths": ["frame.png"],
            "timestamps_seconds": [1.0],
        }]}, "missing_evidence": [],
        "web_observations": [{"state": {"screenshot_path": str(tmp_path / "frame.png")}}],
    })
    case = FailureReviewWorkspace.open(tmp_path, "reviewer").case("case-id")
    assert case["packet"]["run_id"] == "case-id"
    assert case["packet"]["task_id"] == "task-id"
    assert len(case["wearer_inputs"]) == 1
    assert case["wearer_inputs"][0]["data_url"].startswith("data:image/")
    assert case["web_screenshots"] == {0: "frame.png"}
