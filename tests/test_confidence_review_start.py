import json
import hashlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError

import pytest

from egocua.confidence_packets import MATERIALS_HASH_POLICY, answer_boundary, review_materials_sha256
from egocua.confidence_session import (
    _refresh_results,
    open_pass,
    record_assessment,
    start_review,
    start_protocol_recheck,
    retain_verified_blind_review,
    register_answer_boundary_baselines,
)
from egocua import confidence_session
from egocua.confidence import reusable_review_evidence, route_assessment
from egocua.construct_validity import POLICY
from egocua.run_ledger import canonical_sha256, task_sha256
from tests.helpers import task
from egocua import confidence_report as build_task_confidence


def test_refresh_preserves_prior_evidence_and_unselected_rows():
    evidence = [{"path": "old-review.json", "pointer": "/tasks/0"}]
    selected = {
        "task_id": "selected",
        "materials_sha256": "a" * 64,
        "status": "pending",
        "assessments": [],
        "prior_review_evidence": evidence,
    }
    untouched = {"task_id": "untouched", "status": "pending", "assessments": []}
    rows = _refresh_results(
        {
            "selected": {"materials_sha256": "b" * 64},
            "untouched": {"materials_sha256": "c" * 64},
        },
        {"results": [selected, untouched]},
        {},
        {"selected"},
    )
    assert rows[0]["prior_review_evidence"] == evidence
    assert rows[0]["history"][0]["prior_review_evidence"] == evidence
    assert rows[1] == untouched


def test_parallel_task_submissions_preserve_both_reviews(tmp_path, monkeypatch):
    task_ids = ["first", "second"]
    _write_index(
        tmp_path, {task_id: {"materials_sha256": "a" * 64} for task_id in task_ids}
    )
    start_review(tmp_path)
    barrier = Barrier(2)
    overlapping_reads = Barrier(2)
    actual_read = confidence_session._read_object
    actual_write = confidence_session.write_json_atomic
    events = []

    def observed_read(path, context):
        result = actual_read(path, context)
        if path.name == "coding-agent-task-assessments.json":
            events.append("read")
            try:
                overlapping_reads.wait(timeout=0.2)
            except BrokenBarrierError:
                pass
        return result

    def observed_write(path, report):
        events.append("write")
        actual_write(path, report)

    monkeypatch.setattr(confidence_session, "_read_object", observed_read)
    monkeypatch.setattr(confidence_session, "write_json_atomic", observed_write)

    def submit(task_id):
        barrier.wait(timeout=5)
        return record_assessment(
            task_id, "blind_reconstruction", _new_verdict(), root=tmp_path
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        submitted = list(pool.map(submit, task_ids))

    assert all(row["status"] == "in_progress" for row in submitted)
    report = json.loads((tmp_path / confidence_session.REPORT_PATH).read_text())
    assert all(len(row["assessments"]) == 1 for row in report["results"])
    assert report["summary"]["in_progress"] == 2
    assert events == ["read", "write", "read", "write"]


def _write_index(root, rows):
    path = root / "artifacts/reports/confidence-review-packets/index.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "materials_policy": "blind_release_only_then_expected_state",
                "materials_hash_policy": MATERIALS_HASH_POLICY,
                "tasks": rows,
            }
        )
    )


def _verdict(*, adversarial=False):
    verdict = {
        "decision": "admit",
        "confidence": "high",
        "gates": {
            "visual_evidence": "pass",
            "causal_contract": "pass",
            "information_partition": "pass",
            "site_mapping": "pass",
            "answer_boundary": "pass",
            "user_instruction": "pass",
        },
        "uncertainty_reasons": [],
        "evidence": {
            "visual_evidence": ["frame:V0:0"],
            "causal_contract": ["event:update"],
            "information_partition": ["counterfactual:no_scene"],
            "site_mapping": ["entity:target"],
            "answer_boundary": ["window:complete-equivalent-closure"],
            "user_instruction": ["instruction:released"],
        },
        "issues": [],
    }
    if adversarial:
        verdict["counterexample_checks"] = {
            name: "pass"
            for name in (
                "no_scene",
                "site_only",
                "static_substitute",
                "early_commit",
                "tempting_choice",
                "wrong_state",
                "undeclared_equivalent",
                "continuous_interval",
            )
        }
    else:
        verdict["derived_final_state"] = {
            "selected_entity_ids": ["target"],
            "fields": {},
            "submission_completed": True,
        }
    return verdict


def _new_verdict(*, adversarial=False):
    return {
        **_verdict(adversarial=adversarial),
        "reviewer": {"agent_id": "test-reviewer", "context_id": "fresh-test-context"},
    }


@pytest.mark.parametrize(
    "answer", [{}, {"primary_state": {"selected_entity_ids": ["target"], "fields": {}}}]
)
def test_malformed_blind_answer_is_rejected_before_recording(tmp_path, answer):
    _write_review_packet(tmp_path, "task")
    start_review(tmp_path)
    path = tmp_path / confidence_session.REPORT_PATH
    before = path.read_bytes()
    verdict = _new_verdict()
    verdict["derived_final_state"] = answer

    with pytest.raises(ValueError, match="complete website state"):
        record_assessment("task", "blind_reconstruction", verdict, root=tmp_path)

    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "change", ["none", "instruction", "answer", "reviewer", "reference"]
)
def test_append_resolutions_preserves_sealed_reviews_and_rejects_changes(
    tmp_path, monkeypatch, change
):
    from tests.test_task_confidence import _resolved_review

    definition = task("clarify")
    task_id = definition["id"]
    task_path = tmp_path / "data/tasks/clarify/task.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(definition))
    _write_review_packet(tmp_path, task_id)
    report = start_review(tmp_path)
    original_row = report["results"][0]
    row = _resolved_review(definition)
    row["materials_sha256"] = original_row["materials_sha256"]
    second = row["assessments"][1]["verdict"]
    review = {
        "reviewer": second["reviewer"],
        "blind_resolutions": second.pop("blind_resolutions"),
    }
    untouched = {**deepcopy(row), "task_id": "unrelated-task"}
    report["results"] = [row, untouched]
    path = tmp_path / confidence_session.REPORT_PATH
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    monkeypatch.setattr(
        confidence_session.CurrentReviewMaterials,
        "stored_packet_matches",
        lambda self, current_task: (
            current_task["instructions"] == definition["instructions"]
        ),
    )
    monkeypatch.setattr(
        confidence_session.CurrentReviewMaterials,
        "packet",
        lambda self, current_task: {"materials_sha256": row["materials_sha256"]},
    )
    changed = deepcopy(definition)
    if change == "instruction":
        changed["instructions"]["base"] = "Changed user request."
    elif change == "answer":
        changed["goal"]["acceptable_states"][0]["selected_entity_ids"] = ["changed"]
    elif change == "reviewer":
        review["reviewer"] = {"agent_id": "constructor", "context_id": "constructor"}
    elif change == "reference":
        review["blind_resolutions"][0]["evidence"] = ["evidence.visual_evidence.99"]
    task_path.write_text(json.dumps(changed))
    if change != "none":
        with pytest.raises(confidence_session.ConfidenceSessionError):
            confidence_session.append_blind_resolutions(task_id, review, root=tmp_path)
        assert path.read_bytes() == before
        return
    appended = confidence_session.append_blind_resolutions(
        task_id, review, root=tmp_path
    )
    assert appended["assessments"] == row["assessments"]
    assert appended["resolution_reviews"] == [review]
    assert json.loads(path.read_text())["results"][1] == untouched
    assert (
        route_assessment(
            task_sha256(definition),
            appended,
            expected_goal=definition["goal"],
        )["level"]
        == "high"
    )


def _write_review_packet(root, task_id, *, public_site_path=None):
    materials = {"task_id": task_id}
    if public_site_path is not None:
        materials["public_site"] = {"path": public_site_path}
    packet = {
        "schema_version": 1,
        "materials_policy": "blind_release_only_then_expected_state",
        "materials_hash_policy": MATERIALS_HASH_POLICY,
        "materials_sha256": review_materials_sha256(materials),
        "materials": materials,
    }
    path = root / f"artifacts/reports/confidence-review-packets/{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet))
    _write_index(
        root,
        {
            task_id: {
                "task_sha256": "task",
                "path": path.relative_to(root).as_posix(),
                "sha256": "packet",
                "materials_sha256": packet["materials_sha256"],
            },
        },
    )


def _assessment_report(results, *, prompt_version=None):
    return {
        "schema_version": 4,
        "prompt_version": prompt_version or POLICY["assessment"]["prompt_version"],
        "assessor": POLICY["assessment"]["assessor"],
        "pass_names": ["blind_reconstruction", "adversarial_falsification"],
        "materials_policy": "blind_release_only_then_expected_state",
        "assessments_per_task": 2,
        "refresh_policy": "explicit_task_ids",
        "freshness_policy": MATERIALS_HASH_POLICY,
        "results": results,
    }


def test_refresh_preserves_rows_until_task_is_explicitly_invalidated(tmp_path):
    _write_index(
        tmp_path,
        {
            "kept": {
                "task_sha256": "task-a",
                "path": "a.json",
                "sha256": "packet-a",
                "materials_sha256": "a" * 64,
            },
            "changed": {
                "task_sha256": "task-new",
                "path": "b.json",
                "sha256": "packet-b",
                "materials_sha256": "b" * 64,
            },
        },
    )
    output = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    output.write_text(
        json.dumps(
            _assessment_report(
                [
                    {
                        "task_id": "kept",
                        "task_sha256": "task-a",
                        "materials_sha256": "a" * 64,
                        "status": "completed",
                        "review_packet": {"path": "a.json", "sha256": "packet-a"},
                        "assessments": [
                            {"pass": "blind_reconstruction"},
                            {"pass": "adversarial_falsification"},
                        ],
                    },
                    {
                        "task_id": "changed",
                        "task_sha256": "task-old",
                        "status": "completed",
                        "review_packet": {"path": "old.json", "sha256": "packet-old"},
                        "assessments": [
                            {"pass": "blind_reconstruction"},
                            {"pass": "adversarial_falsification"},
                        ],
                    },
                    {
                        "task_id": "removed",
                        "task_sha256": "task-c",
                        "status": "completed",
                        "review_packet": {"path": "c.json", "sha256": "packet-c"},
                        "assessments": [
                            {"pass": "blind_reconstruction"},
                            {"pass": "adversarial_falsification"},
                        ],
                    },
                ]
            )
        )
    )

    report = start_review(
        tmp_path,
        refresh=True,
        invalidation_reasons={"changed": "task target was repaired"},
    )

    rows = {row["task_id"]: row for row in report["results"]}
    assert set(rows) == {"kept", "changed"}
    assert rows["kept"]["status"] == "completed"
    assert rows["kept"]["materials_sha256"] == "a" * 64
    assert "task_sha256" not in rows["kept"]
    assert "review_packet" not in rows["kept"]
    assert rows["changed"]["status"] == "pending"
    assert rows["changed"]["pending_reason"] == (
        "explicit_task_invalidation: task target was repaired"
    )
    assert rows["changed"]["assessments"] == []
    assert rows["changed"]["history"] == [
        {
            "task_sha256": "task-old",
            "status": "completed",
            "assessments": [
                {"pass": "blind_reconstruction"},
                {"pass": "adversarial_falsification"},
            ],
        }
    ]
    assert report["summary"] == {
        "tasks": 2,
        "completed": 1,
        "in_progress": 0,
        "pending": 1,
    }
    assert report["invalidations"] == [
        {
            "task_id": "changed",
            "reason": "task target was repaired",
        }
    ]
    assert [row["task_id"] for row in report["retired_results"]] == ["removed"]
    assert report["retired_results"][0]["status"] == "completed"
    refreshed = start_review(tmp_path, refresh=True, task_ids=[])
    assert refreshed["retired_results"] == report["retired_results"]
    assert refreshed["results"] == report["results"]


def test_refresh_invalidates_only_the_task_whose_materials_changed(tmp_path):
    _write_index(
        tmp_path,
        {
            "task": {
                "task_sha256": "new",
                "path": "new.json",
                "sha256": "new",
                "materials_sha256": "b" * 64,
            },
        },
    )
    output = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    output.write_text(
        json.dumps(
            _assessment_report(
                [
                    {
                        "task_id": "task",
                        "task_sha256": "old",
                        "review_packet": {"path": "old.json", "sha256": "old"},
                        "materials_sha256": "a" * 64,
                        "status": "completed",
                        "assessments": [
                            {"pass": "blind_reconstruction"},
                            {"pass": "adversarial_falsification"},
                        ],
                    }
                ]
            )
        )
    )

    report = start_review(tmp_path, refresh=True)

    row = report["results"][0]
    assert row["status"] == "pending"
    assert row["pending_reason"] == "review_materials_changed"
    assert row["materials_sha256"] == "b" * 64
    assert row["assessments"] == []
    assert row["history"] == [
        {
            "task_sha256": "old",
            "materials_sha256": "a" * 64,
            "status": "completed",
            "assessments": [
                {"pass": "blind_reconstruction"},
                {"pass": "adversarial_falsification"},
            ],
        }
    ]


@pytest.mark.parametrize("existing_history", [False, True])
def test_refresh_keeps_unique_history_without_reusing_historical_passes(
    tmp_path,
    existing_history,
):
    task_id = "changed-review"
    _write_review_packet(tmp_path, task_id)
    snapshot = {
        "materials_sha256": "a" * 64,
        "status": "completed",
        "pending_reason": "prior source correction",
        "assessments": [
            {
                "pass": "blind_reconstruction",
                "status": "completed",
                "verdict": _verdict(),
            },
            {
                "pass": "adversarial_falsification",
                "status": "completed",
                "verdict": _verdict(adversarial=True),
            },
        ],
    }
    previous = {"task_id": task_id, **snapshot}
    if existing_history:
        previous["history"] = [snapshot]
    output = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    output.write_text(json.dumps(_assessment_report([previous])))

    start_review(tmp_path, refresh=True)
    report = start_review(tmp_path, refresh=True)

    row = report["results"][0]
    assert row["history"] == [snapshot]
    assert row["assessments"] == []
    assert report["summary"] == {
        "tasks": 1,
        "completed": 0,
        "in_progress": 0,
        "pending": 1,
    }
    materials = open_pass(task_id, root=tmp_path)
    assert materials["pass"] == "blind_reconstruction"
    assert "expected_states" not in materials

    recorded = record_assessment(
        task_id,
        "blind_reconstruction",
        _new_verdict(),
        root=tmp_path,
    )

    assert recorded["status"] == "in_progress"
    assert len(recorded["assessments"]) == 1
    assert recorded["history"] == [snapshot]


def test_new_task_does_not_get_an_empty_history(tmp_path):
    _write_review_packet(tmp_path, "new-task")

    initial = start_review(tmp_path)
    refreshed = start_review(tmp_path, refresh=True)

    assert "history" not in initial["results"][0]
    assert "history" not in refreshed["results"][0]


def test_refresh_preserves_history_across_further_material_changes(tmp_path):
    packet_row = {
        "task_sha256": "task",
        "path": "task.json",
        "sha256": "packet",
        "materials_sha256": "b" * 64,
    }
    _write_index(tmp_path, {"task": packet_row})
    snapshot = {
        "materials_sha256": "a" * 64,
        "status": "in_progress",
        "assessments": [{"pass": "blind_reconstruction"}],
    }
    output = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    output.write_text(
        json.dumps(
            _assessment_report(
                [
                    {
                        "task_id": "task",
                        **snapshot,
                    }
                ]
            )
        )
    )
    start_review(tmp_path, refresh=True)
    packet_row["materials_sha256"] = "c" * 64
    _write_index(tmp_path, {"task": packet_row})

    report = start_review(tmp_path, refresh=True)
    repeated = start_review(tmp_path, refresh=True)

    assert report["results"][0]["history"] == [
        snapshot,
        {
            "materials_sha256": "b" * 64,
            "status": "pending",
            "assessments": [],
            "pending_reason": "review_materials_changed",
        },
    ]
    assert repeated["results"] == report["results"]


def test_invalidation_requires_refresh(tmp_path):
    _write_index(
        tmp_path,
        {
            "task": {
                "task_sha256": "task-a",
                "path": "a.json",
                "sha256": "packet-a",
                "materials_sha256": "a" * 64,
            },
        },
    )

    with pytest.raises(ValueError, match="requires --refresh"):
        start_review(
            tmp_path,
            invalidation_reasons={"task": "task evidence was repaired"},
        )


def test_invalidation_requires_a_reason(tmp_path):
    _write_index(
        tmp_path,
        {
            "task": {
                "task_sha256": "task-a",
                "path": "a.json",
                "sha256": "packet-a",
                "materials_sha256": "a" * 64,
            },
        },
    )

    with pytest.raises(ValueError, match="needs a non-empty"):
        start_review(tmp_path, refresh=True, invalidation_reasons={"task": ""})


def test_prompt_metadata_change_does_not_reset_completed_task_evidence(tmp_path):
    _write_index(
        tmp_path,
        {
            "task": {
                "task_sha256": "task-a",
                "path": "a.json",
                "sha256": "packet-a",
                "materials_sha256": "a" * 64,
            },
        },
    )
    output = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    original = {
        "task_id": "task",
        "materials_sha256": "a" * 64,
        "status": "completed",
        "assessments": [{"pass": "blind_reconstruction", "status": "completed"}],
    }
    output.write_text(
        json.dumps(_assessment_report([original], prompt_version="old-prompt"))
    )

    refreshed = start_review(tmp_path, refresh=True)

    assert refreshed["results"] == [original]


def test_existing_review_requires_explicit_refresh_or_reset(tmp_path):
    _write_index(
        tmp_path,
        {
            "task": {
                "task_sha256": "task-a",
                "path": "a.json",
                "sha256": "packet-a",
                "materials_sha256": "a" * 64,
            },
        },
    )
    output = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    output.write_text("{}")

    with pytest.raises(FileExistsError, match="--refresh"):
        start_review(tmp_path)


def test_session_reveals_expected_state_only_after_blind_pass_is_frozen(tmp_path):
    task_id = "phase-isolated-task"
    _write_review_packet(tmp_path, task_id)
    definition = task("clarify")
    definition["id"] = task_id
    task_path = tmp_path / "data/tasks/clarify/task.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(definition))
    index_path = tmp_path / "artifacts/reports/confidence-review-packets/index.json"
    index = json.loads(index_path.read_text())
    index["tasks"][task_id]["answer_boundary"] = answer_boundary(definition)
    index_path.write_text(json.dumps(index))
    start_review(tmp_path, reset=True)

    blind = open_pass(task_id, root=tmp_path)

    assert blind["pass"] == "blind_reconstruction"
    assert "expected_goal" not in blind
    assert "answer_boundary" not in blind
    assert "sealed_blind_verdict" not in blind
    assert "advisories" not in blind

    record_assessment(
        task_id,
        "blind_reconstruction",
        _new_verdict(),
        root=tmp_path,
    )
    adversarial = open_pass(task_id, root=tmp_path)

    assert adversarial["pass"] == "adversarial_falsification"
    assert (
        "remove the clarification answer"
        in (adversarial["counterexample_guidance"]["static_substitute"])
    )
    assert adversarial["expected_goal"] == definition["goal"]
    assert adversarial["sealed_blind_verdict"] == _new_verdict()
    definition["goal"]["acceptable_states"][0]["selected_entity_ids"] = [
        "changed-answer"
    ]
    task_path.write_text(json.dumps(definition))
    with pytest.raises(ValueError, match="answer boundary changed"):
        record_assessment(
            task_id,
            "adversarial_falsification",
            _new_verdict(adversarial=True),
            root=tmp_path,
        )


def test_session_rejects_skipping_the_blind_pass(tmp_path):
    _write_review_packet(tmp_path, "ordered-task")
    start_review(tmp_path, reset=True)

    with pytest.raises(ValueError, match="expected 'blind_reconstruction'"):
        record_assessment(
            "ordered-task",
            "adversarial_falsification",
            _new_verdict(adversarial=True),
            root=tmp_path,
        )


@pytest.mark.parametrize("adversarial", [False, True])
@pytest.mark.parametrize("reviewer", [None, {}, {"agent_id": "reviewer"}])
def test_new_submission_requires_reviewer_identity_without_changing_history(
    tmp_path, adversarial, reviewer
):
    definition = task("clarify")
    task_id = definition["id"]
    _write_review_packet(tmp_path, task_id)
    task_path = tmp_path / "data/tasks/clarify/task.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(definition))
    report = start_review(tmp_path)
    row = report["results"][0]
    if adversarial:
        row["answer_boundary"] = answer_boundary(definition)
        row["assessments"] = [
            {
                "pass": "blind_reconstruction",
                "status": "completed",
                "verdict": _verdict(),
            }
        ]
    path = tmp_path / confidence_session.REPORT_PATH
    path.write_text(json.dumps(report))
    verdict = _verdict(adversarial=adversarial)
    if reviewer is not None:
        verdict["reviewer"] = reviewer
    pass_name = "adversarial_falsification" if adversarial else "blind_reconstruction"

    with pytest.raises(ValueError, match="reviewer"):
        record_assessment(task_id, pass_name, verdict, root=tmp_path)

    assert json.loads(path.read_text()) == report


def test_title_similarity_is_advisory_only_after_blind_record_is_sealed(tmp_path):
    definition = task("clarify")
    task_id = definition["id"]
    title = "Detailed red portable kitchen utensil"
    definition["instructions"]["base"] = f"Find {title}."
    target_id = definition["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    (tmp_path / "public.json").write_text(
        json.dumps({"entities": [{"entity_id": target_id, "title": title}]})
    )
    task_path = tmp_path / "data/tasks/clarify/task.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(definition))
    _write_review_packet(tmp_path, task_id, public_site_path="public.json")
    index_path = tmp_path / confidence_session.INDEX_PATH
    index = json.loads(index_path.read_text())
    index["tasks"][task_id]["answer_boundary"] = answer_boundary(definition)
    index_path.write_text(json.dumps(index))
    start_review(tmp_path)

    assert "advisories" not in open_pass(task_id, root=tmp_path)
    blind = _verdict()
    blind["reviewer"] = {"agent_id": "independent", "context_id": "fresh-context"}
    record_assessment(task_id, "blind_reconstruction", blind, root=tmp_path)
    opened = open_pass(task_id, root=tmp_path)
    assert len(opened["advisories"]) == 1
    assert target_id in opened["advisories"][0]
    second = _verdict(adversarial=True)
    second["reviewer"] = blind["reviewer"]
    second["evidence"] = {
        dimension: [{"ref": f"blind_reconstruction.evidence.{dimension}"}]
        for dimension in blind["evidence"]
    }
    result = record_assessment(
        task_id, "adversarial_falsification", second, root=tmp_path
    )

    assert result["status"] == "completed"
    assert result["assessments"][0]["verdict"] == blind


@pytest.mark.parametrize("context_id", ["seen-key", "new-context"])
def test_reopened_blind_review_requires_a_context_not_exposed_to_key(
    tmp_path, context_id
):
    task_id = "reopened-task"
    _write_review_packet(tmp_path, task_id)
    report = start_review(tmp_path)
    previous = _verdict(adversarial=True)
    previous["reviewer"] = {"agent_id": "reviewer", "context_id": "seen-key"}
    report["results"][0]["history"] = [
        {"assessments": [{"pass": "adversarial_falsification", "verdict": previous}]}
    ]
    path = tmp_path / confidence_session.REPORT_PATH
    path.write_text(json.dumps(report))
    verdict = _verdict()
    verdict["reviewer"] = {"agent_id": "reviewer", "context_id": context_id}

    if context_id == "seen-key":
        with pytest.raises(ValueError, match="context has seen the key"):
            record_assessment(task_id, "blind_reconstruction", verdict, root=tmp_path)
        assert json.loads(path.read_text())["results"][0]["assessments"] == []
    else:
        result = record_assessment(
            task_id, "blind_reconstruction", verdict, root=tmp_path
        )
        assert result["assessments"][0]["verdict"]["reviewer"] == verdict["reviewer"]


def test_scoped_refresh_preserves_other_tasks_even_when_their_packets_changed():
    previous = {
        task_id: {
            "task_id": task_id,
            "status": "completed",
            "materials_sha256": "a" * 64,
            "answer_boundary": answer_boundary(task("graft")),
            "assessments": [
                {"pass": "blind_reconstruction", "status": "completed"},
                {"pass": "adversarial_falsification", "status": "completed"},
            ],
        }
        for task_id in ("changed", "untouched")
    }
    index = {
        "changed": {"materials_sha256": "a" * 64, "answer_boundary": answer_boundary(task("refresh"))},
        "untouched": {"materials_sha256": "d" * 64, "answer_boundary": answer_boundary(task("clarify"))},
    }

    rows = _refresh_results(
        index, {"results": list(previous.values())}, {}, {"changed"}
    )
    result = {row["task_id"]: row for row in rows}

    assert result["untouched"] == previous["untouched"]
    assert result["changed"]["assessments"] == previous["changed"]["assessments"][:1]
    assert result["changed"]["status"] == "in_progress"


def _history_review_fixture(tmp_path, monkeypatch, changed=None, *, track="clarify"):
    definition = task(track)
    task_id = definition["id"]
    old_materials = {
        "task_id": task_id,
        "initial": {"instruction": "Find the tool.", "observations": []},
        "scheduled_releases": [],
        "public_site": {"sha256": "public-window"},
        "requested_output_shape": {
            "selected_entity_ids": "list[public entity_id]",
            "fields": "object",
        },
        "release_trigger": "timeline_step",
        "release_delivery": "Release on the scheduled step.",
    }
    old_hash = review_materials_sha256(old_materials)
    blind = _verdict()
    blind["derived_final_state"] = deepcopy(definition["goal"]["acceptable_states"][0])
    source = {
        "materials_sha256": old_hash,
        "answer_boundary": answer_boundary(definition),
        "status": "completed",
        "assessments": [
            {
                "index": 0,
                "pass": "blind_reconstruction",
                "status": "completed",
                "verdict": blind,
            },
            {
                "index": 1,
                "pass": "adversarial_falsification",
                "status": "completed",
                "verdict": _verdict(adversarial=True),
            },
        ],
    }
    materials = deepcopy(old_materials)
    if changed == "instruction":
        materials["initial"]["instruction"] = "Find another tool."
    elif changed == "frames":
        materials["initial"]["observations"] = [{"frame": "changed.jpg"}]
    elif changed == "site":
        materials["public_site"]["sha256"] = "changed-window"
    elif changed == "answer":
        definition["goal"]["acceptable_states"][0]["selected_entity_ids"] = [
            "new-target"
        ]
    elif changed == "initial":
        if track == "graft":
            definition["graft"]["initial_goal"]["acceptable_states"].pop()
        else:
            definition["truth"]["initial_acceptable_states"].append(
                {"selected_entity_ids": ["new-initial"], "fields": {}}
            )
    elif changed == "protocol":
        materials.pop("release_trigger")
        materials.pop("release_delivery")
    packet = {
        "schema_version": 1,
        "materials_policy": "blind_release_only_then_expected_state",
        "materials_hash_policy": MATERIALS_HASH_POLICY,
        "materials": materials,
        "materials_sha256": review_materials_sha256(materials),
    }
    packet_path = (
        tmp_path / f"artifacts/reports/confidence-review-packets/{task_id}.json"
    )
    packet_path.parent.mkdir(parents=True)
    packet_path.write_text(json.dumps(packet))
    _write_index(
        tmp_path,
        {
            task_id: {
                "task_sha256": task_sha256(definition),
                "path": packet_path.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest(),
                "materials_sha256": packet["materials_sha256"],
                "answer_boundary": answer_boundary(definition),
            }
        },
    )
    active = {
        "task_id": task_id,
        "materials_sha256": packet["materials_sha256"],
        "answer_boundary": answer_boundary(definition),
        "status": "pending",
        "pending_reason": "review_materials_changed",
        "assessments": [],
        "history": [source],
    }
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report_path.write_text(
        json.dumps({**_assessment_report([active]), "agent": "fixture"})
    )
    acceptance_path = tmp_path / "artifacts/reports/task-acceptance.json"
    acceptance_path.write_text(
        json.dumps(
            {
                "tasks": {
                    task_id: {
                        "task_sha256": task_sha256(definition),
                        "status": "candidate_ready",
                        "integrity_pass": True,
                        "failed_checks": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(confidence_session, "load_tasks", lambda _: [definition])
    monkeypatch.setattr(confidence_session, "load_construction_tasks", lambda _, task_ids=None: [definition])
    monkeypatch.setattr(build_task_confidence, "load_construction_tasks", lambda _, task_ids=None: [definition])
    monkeypatch.setattr(
        build_task_confidence.CurrentReviewMaterials,
        "packet",
        lambda self, task: packet,
    )
    return definition, active, packet


@pytest.mark.parametrize("track,changed", [
    ("clarify", "answer"), ("graft", "answer"),
    ("graft", "initial"), ("refresh", "initial"),
])
def test_historical_blind_evidence_reopens_only_adversarial_review(
    tmp_path, monkeypatch, track, changed
):
    definition, active, packet = _history_review_fixture(tmp_path, monkeypatch, changed, track=track)
    task_id = definition["id"]
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    before = report_path.read_bytes()

    report = build_task_confidence.build_confidence_report(tmp_path)

    assert report["tasks"][task_id]["recheck_passes"] == ["adversarial_falsification"]
    assert report_path.read_bytes() == before
    start_protocol_recheck(task_id, root=tmp_path)
    row = json.loads(report_path.read_text())["results"][0]
    assert row["assessments"] == active["history"][0]["assessments"][:1]
    assert active["history"][0] in row["history"]
    opened = open_pass(task_id, root=tmp_path)
    assert opened["pass"] == "adversarial_falsification"
    assert opened["expected_goal"] == definition["goal"]
    assert packet["materials_sha256"] == row["materials_sha256"]
    assert opened["answer_boundary"] == answer_boundary(definition)


@pytest.mark.parametrize("track", ["graft", "refresh"])
def test_current_initial_boundary_change_cannot_bypass_report_or_open_second(
    tmp_path, monkeypatch, track
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "initial", track=track
    )
    previous = {"task_id": definition["id"], **active["history"][0]}
    path = tmp_path / confidence_session.REPORT_PATH
    path.write_text(json.dumps({**_assessment_report([previous]), "agent": "fixture"}))
    before = path.read_bytes()
    report = build_task_confidence.build_confidence_report(tmp_path)
    confidence = report["tasks"][definition["id"]]
    assert confidence["level"] == "unassessed"
    assert confidence["recheck_passes"] == ["adversarial_falsification"]
    assert path.read_bytes() == before
    rows = _refresh_results(
        {definition["id"]: {
            "materials_sha256": packet["materials_sha256"],
            "answer_boundary": answer_boundary(definition),
        }},
        {"results": [previous]}, {},
    )
    retained = rows[0]
    assert retained["assessments"] == previous["assessments"][:1]
    assert retained["history"][0]["assessments"] == previous["assessments"]
    retained["answer_boundary"] = previous["answer_boundary"]
    path.write_text(json.dumps(_assessment_report([retained])))
    with pytest.raises(ValueError, match="answer boundary changed"):
        open_pass(definition["id"], root=tmp_path)
    with pytest.raises(ValueError, match="answer boundary changed"):
        record_assessment(
            definition["id"], "adversarial_falsification",
            _new_verdict(adversarial=True), root=tmp_path,
        )


@pytest.mark.parametrize("fault", [None, "source", "boundary", "materials"])
def test_forward_baseline_registration_preserves_verdicts_and_tracks_future_changes(
    tmp_path, monkeypatch, fault
):
    definition, active, packet = _history_review_fixture(tmp_path, monkeypatch, track="graft")
    previous = {"task_id": definition["id"], **deepcopy(active["history"][0])}
    previous.pop("answer_boundary")
    previous["expected_states_sha256"] = "historical-final-only-digest"
    unrelated = {"task_id": "unrelated", "assessments": [], "status": "pending"}
    path = tmp_path / confidence_session.REPORT_PATH
    path.write_text(json.dumps(_assessment_report([previous, unrelated])))
    before = path.read_bytes()
    registration = {
        "task_id": definition["id"],
        "source_assessments": deepcopy(previous["assessments"]),
        "source_materials_sha256": previous["materials_sha256"],
        "answer_boundary": answer_boundary(definition),
    }
    monkeypatch.setattr(
        confidence_session.CurrentReviewMaterials, "stored_packet_matches",
        lambda self, task: fault != "materials",
    )
    if fault == "source":
        registration["source_assessments"][1]["verdict"]["confidence"] = "low"
    elif fault == "boundary":
        registration["answer_boundary"]["initial_goal"]["acceptable_states"] = []
    if fault:
        with pytest.raises(ValueError):
            register_answer_boundary_baselines([registration], root=tmp_path)
        assert path.read_bytes() == before
        return
    assert reusable_review_evidence(definition, previous, packet) is None
    with pytest.raises(ValueError, match="register the forward"):
        _refresh_results(
            {definition["id"]: {
                "materials_sha256": packet["materials_sha256"],
                "answer_boundary": answer_boundary(definition),
            }}, {"results": [previous]}, {},
        )
    assert register_answer_boundary_baselines([registration], root=tmp_path)["registered"] == 1
    report = json.loads(path.read_text())
    row = report["results"][0]
    assert row["assessments"] == previous["assessments"]
    assert report["results"][1] == unrelated
    assert row["history"] == [{key: value for key, value in previous.items() if key != "task_id"}]
    assert row["answer_boundary_registration"]["purpose"] == "forward_change_tracking"
    assert row["answer_boundary_registration"]["source_assessment"]["pointer"] == "/results/0/history/0/assessments"
    assert reusable_review_evidence(definition, row, packet)["recheck_passes"] == []
    definition["graft"]["initial_goal"]["acceptable_states"].pop()
    assert reusable_review_evidence(definition, row, packet)["recheck_passes"] == ["adversarial_falsification"]


@pytest.mark.parametrize("changed", ["instruction", "frames", "site", "protocol"])
def test_changed_task_evidence_does_not_inherit_historical_review(
    tmp_path, monkeypatch, changed
):
    definition, active, packet = _history_review_fixture(tmp_path, monkeypatch, changed)
    assert reusable_review_evidence(definition, active, packet) is None
    report = build_task_confidence.build_confidence_report(tmp_path)
    assert "recheck_passes" not in report["tasks"][definition["id"]]
    with pytest.raises(
        ValueError, match="unchanged blind-review evidence is not proven"
    ):
        start_protocol_recheck(definition["id"], root=tmp_path)


def test_missing_frozen_blind_answer_cannot_be_inherited(tmp_path, monkeypatch):
    definition, active, packet = _history_review_fixture(tmp_path, monkeypatch)
    active["history"][0]["assessments"][0]["verdict"].pop("derived_final_state")
    assert reusable_review_evidence(definition, active, packet) is None


@pytest.mark.parametrize("changed_fact", [None, "site", "frames"])
def test_verified_instruction_packet_retains_both_concrete_passes_only(
    tmp_path, monkeypatch, changed_fact
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    retained = deepcopy(packet)
    retained["materials"]["initial"]["instruction"] = "Find the tool."
    if changed_fact == "site":
        retained["materials"]["public_site"]["sha256"] = "changed-site"
    elif changed_fact == "frames":
        retained["materials"]["initial"]["observations"] = ["changed-frame"]
    retained["materials_sha256"] = review_materials_sha256(retained["materials"])
    packet["retained_instruction_packet"] = retained
    original = deepcopy(active)
    reused = reusable_review_evidence(definition, active, packet)
    if changed_fact is None:
        assert reused["recheck_passes"] == []
        assert reused["source"]["assessments"] == active["history"][0]["assessments"]
    else:
        assert reused is None
    assert active == original


def _factual_verification(definition, active, packet):
    source = active["history"][0]
    evidence = source["assessments"][0]["verdict"]["evidence"]
    fields = {name: name for name in evidence}
    return {
        "task_id": definition["id"],
        "source_materials_sha256": source["materials_sha256"],
        "source_assessment_sha256": canonical_sha256(source["assessments"][0]),
        "current_materials_sha256": packet["materials_sha256"],
        "reviewer": {"agent_id": "incremental-reviewer", "context_id": "seen-key"},
        "changes": [
            "Packet wording changed; each retained semantic fact checked below."
        ],
        "evidence": {
            dimension: {
                "source_evidence": evidence[field],
                "current_evidence": [
                    f"Current packet confirms {dimension}: {evidence[field][0]}"
                ],
                "conclusion": "supported",
            }
            for dimension, field in fields.items()
        },
    }


def test_verified_facts_retain_original_blind_without_old_packet(tmp_path, monkeypatch):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    verification = _factual_verification(definition, active, packet)
    original = deepcopy(active["history"][0])

    row = retain_verified_blind_review(definition["id"], verification, root=tmp_path)

    assert row["status"] == "in_progress"
    assert row["assessments"] == original["assessments"][:1]
    assert "reviewer" not in row["assessments"][0]["verdict"]
    assert original in row["history"]
    assert row["material_verification"] == verification
    assert reusable_review_evidence(definition, row, packet) is None
    assert (
        open_pass(definition["id"], root=tmp_path)["pass"]
        == "adversarial_falsification"
    )

    row = record_assessment(
        definition["id"],
        "adversarial_falsification",
        _new_verdict(adversarial=True),
        root=tmp_path,
    )
    assert row["status"] == "completed"
    assert reusable_review_evidence(definition, row, packet)["recheck_passes"] == []
    row["history"][0]["assessments"][0]["verdict"]["evidence"]["visual_evidence"] = [
        "tampered"
    ]
    assert reusable_review_evidence(definition, row, packet) is None


def test_completed_low_confidence_blind_is_retained_without_clearing_concerns(
    tmp_path, monkeypatch
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    source = active["history"][0]
    blind = source["assessments"][0]["verdict"]
    blind["confidence"] = "low"
    blind["gates"]["answer_boundary"] = "uncertain"
    blind["uncertainty_reasons"] = ["An equivalent listing needs checking."]
    report_path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(report_path.read_text())
    report["results"] = [active]
    report_path.write_text(json.dumps(report))
    verification = _factual_verification(definition, active, packet)

    retained = retain_verified_blind_review(
        definition["id"], verification, root=tmp_path
    )
    assert retained["assessments"] == source["assessments"][:1]
    completed = record_assessment(
        definition["id"],
        "adversarial_falsification",
        _new_verdict(adversarial=True),
        root=tmp_path,
    )
    assert (
        route_assessment(
            task_sha256(definition),
            completed,
            expected_goal=definition["goal"],
        )["level"]
        == "low"
    )
    assert completed["assessments"][0]["verdict"] == blind


def test_resolved_repair_blind_can_be_reused_after_material_changes(
    tmp_path, monkeypatch
):
    from tests.test_task_confidence import _resolved_review

    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    original = _resolved_review(definition)
    original["materials_sha256"] = active["history"][0]["materials_sha256"]
    active["history"] = [original]
    path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(path.read_text())
    report["results"] = [active]
    path.write_text(json.dumps(report))

    retained = retain_verified_blind_review(
        definition["id"],
        _factual_verification(definition, active, packet),
        root=tmp_path,
    )

    assert retained["assessments"] == original["assessments"][:1]
    assert original in retained["history"]
    assert (
        open_pass(definition["id"], root=tmp_path)["pass"]
        == "adversarial_falsification"
    )


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("confidence", ["high", "low"])
def test_verified_partial_blind_continues_only_second_pass(
    tmp_path, monkeypatch, external, confidence
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    original = active["history"][0]
    original["status"] = "in_progress"
    original["assessments"] = original["assessments"][:1]
    blind = original["assessments"][0]["verdict"]
    blind["confidence"] = confidence
    if confidence == "low":
        blind["gates"]["answer_boundary"] = "uncertain"
        blind["uncertainty_reasons"] = ["An equivalent listing needs checking."]
    verification = _factual_verification(definition, active, packet)
    report_path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(report_path.read_text())
    reference = {"path": "original-blind.json", "pointer": ""}
    if external:
        blind["task_id"] = definition["id"]
        (tmp_path / reference["path"]).write_text(json.dumps(blind))
        original.pop("materials_sha256")
        verification.pop("source_materials_sha256")
        verification["source_assessment_sha256"] = canonical_sha256(
            original["assessments"][0]
        )
        active.update(
            history=[],
            pending_reason="initial_assessment",
            prior_review_evidence=[reference],
        )
    report["results"] = [active]
    report_path.write_text(json.dumps(report))

    retained = retain_verified_blind_review(
        definition["id"], verification, root=tmp_path
    )

    assert retained["assessments"] == original["assessments"]
    assert "reviewer" not in retained["assessments"][0]["verdict"]
    assert retained["material_verification"]["reviewer"] == verification["reviewer"]
    assert (
        open_pass(definition["id"], root=tmp_path)["pass"]
        == "adversarial_falsification"
    )
    assert reusable_review_evidence(definition, retained, packet) is None
    if external:
        archived = next(
            row for row in retained["history"] if row.get("source_reference")
        )
        assert archived["source_reference"] == reference
        assert archived["assessments"][0]["verdict"] == blind

    completed = record_assessment(
        definition["id"],
        "adversarial_falsification",
        _new_verdict(adversarial=True),
        root=tmp_path,
    )
    assert completed["status"] == "completed"
    assert reusable_review_evidence(definition, completed, packet) is not None
    assert (
        route_assessment(
            task_sha256(definition),
            completed,
            expected_goal=definition["goal"],
        )["level"]
        == confidence
    )


@pytest.mark.parametrize(
    "defect", ["explicit_reopen", "wrong_task", "incomplete_verdict"]
)
def test_external_partial_blind_rejects_unproven_inheritance(
    tmp_path, monkeypatch, defect
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    source = active["history"][0]
    source.update(status="in_progress", assessments=source["assessments"][:1])
    blind = source["assessments"][0]["verdict"]
    blind["task_id"] = definition["id"]
    verification = _factual_verification(definition, active, packet)
    if defect == "wrong_task":
        blind["task_id"] = "another-task"
    if defect == "incomplete_verdict":
        blind.pop("evidence")
    verification["source_assessment_sha256"] = canonical_sha256(
        source["assessments"][0]
    )
    (tmp_path / "blind.json").write_text(json.dumps(blind))
    source.pop("materials_sha256")
    verification.pop("source_materials_sha256")
    active.update(
        history=[],
        pending_reason="initial_assessment",
        prior_review_evidence=[{"path": "blind.json", "pointer": ""}],
    )
    if defect == "explicit_reopen":
        active["pending_reason"] = "incorrect_visual_evidence"
    report_path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(report_path.read_text())
    report["results"] = [active]
    report_path.write_text(json.dumps(report))
    original = report_path.read_bytes()

    with pytest.raises(ValueError):
        retain_verified_blind_review(definition["id"], verification, root=tmp_path)

    assert report_path.read_bytes() == original


def test_partial_blind_with_wrong_main_answer_cannot_be_inherited(
    tmp_path, monkeypatch
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    source = active["history"][0]
    source.update(status="in_progress", assessments=source["assessments"][:1])
    source["assessments"][0]["verdict"]["derived_final_state"][
        "selected_entity_ids"
    ] = ["wrong"]
    report_path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(report_path.read_text())
    report["results"] = [active]
    report_path.write_text(json.dumps(report))
    original = report_path.read_bytes()

    with pytest.raises(ValueError, match="current legal ending"):
        retain_verified_blind_review(
            definition["id"],
            _factual_verification(definition, active, packet),
            root=tmp_path,
        )

    assert report_path.read_bytes() == original


@pytest.mark.parametrize("source_exists", [True, False])
def test_second_pass_evidence_remains_reusable_after_archiving_same_blind(
    tmp_path, monkeypatch, source_exists
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    earlier = deepcopy(active["history"][0])
    later = deepcopy(earlier)
    later["task_id"] = definition["id"]
    later["history"] = [earlier]
    second = later["assessments"][1]["verdict"]
    second.update(
        decision="repair",
        confidence="low",
        issues=["The public listing does not establish the package count."],
    )
    second["gates"]["answer_boundary"] = "uncertain"
    second["evidence"]["answer_boundary"] = [
        "The detail photo shows one box without a readable package count."
    ]
    verification = _factual_verification(definition, active, packet)
    verification["evidence"]["answer_boundary"]["source_evidence"] = (
        second["evidence"]["answer_boundary"]
        if source_exists
        else ["No review ever recorded this package count."]
    )
    report_path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(report_path.read_text())
    report["results"] = [later]
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()

    if not source_exists:
        with pytest.raises(ValueError, match="source evidence must quote"):
            retain_verified_blind_review(definition["id"], verification, root=tmp_path)
        assert report_path.read_bytes() == before
        return

    retained = retain_verified_blind_review(
        definition["id"], verification, root=tmp_path
    )
    assert retained["assessments"] == earlier["assessments"][:1]
    verdict = deepcopy(second)
    verdict["reviewer"] = {"agent_id": "incremental", "context_id": "seen-key"}
    completed = record_assessment(
        definition["id"], "adversarial_falsification", verdict, root=tmp_path
    )

    reused = reusable_review_evidence(definition, completed, packet)
    assert reused is not None
    assert reused["recheck_passes"] == []
    assert completed["history"][0] == earlier
    assert any(
        snapshot["assessments"] == later["assessments"]
        for snapshot in completed["history"]
    )
    route = build_task_confidence.build_confidence_report(tmp_path)["tasks"][
        definition["id"]
    ]
    assert route["level"] == "low"
    assert route["route"] == "human_review_pending"


def _repair_history_fixture(tmp_path, monkeypatch, confidence):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    for assessment in active["history"][0]["assessments"]:
        verdict = assessment["verdict"]
        verdict.update(
            decision="repair",
            confidence=confidence,
            issues=["V0 shows a bottle but its intended cleaning use is unreadable."],
        )
        verdict["gates"]["visual_evidence"] = (
            "fail" if confidence == "high" else "uncertain"
        )
        verdict["evidence"]["visual_evidence"] = [
            "V0:0 shows the bottle beside the sink; its use label is unreadable."
        ]
        verdict.pop("derived_final_state", None)
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(report_path.read_text())
    report["results"] = [active]
    report_path.write_text(json.dumps(report))
    return definition, active, packet


@pytest.mark.parametrize("confidence", ["high", "low"])
def test_verified_repair_preserves_original_without_requiring_a_legal_answer(
    tmp_path, monkeypatch, confidence
):
    definition, active, packet = _repair_history_fixture(
        tmp_path, monkeypatch, confidence
    )
    original = deepcopy(active["history"][0])

    row = retain_verified_blind_review(
        definition["id"],
        _factual_verification(definition, active, packet),
        root=tmp_path,
    )

    assert row["assessments"] == original["assessments"][:1]
    assert original in row["history"]
    assert "reviewer" not in row["assessments"][0]["verdict"]
    assert (
        open_pass(definition["id"], root=tmp_path)["pass"]
        == "adversarial_falsification"
    )
    verdict = deepcopy(original["assessments"][1]["verdict"])
    verdict["reviewer"] = {"agent_id": "incremental", "context_id": "seen-key"}
    row = record_assessment(
        definition["id"], "adversarial_falsification", verdict, root=tmp_path
    )

    assert reusable_review_evidence(definition, row, packet)["recheck_passes"] == []
    route = build_task_confidence.build_confidence_report(tmp_path)["tasks"][
        definition["id"]
    ]
    expected_route = "repair" if confidence == "high" else "human_review_pending"
    assert route["route"] == expected_route
    assert route["level"] == ("rejected" if confidence == "high" else "low")
    row["assessments"][1]["verdict"] = _new_verdict(adversarial=True)
    assert route_assessment(task_sha256(definition), row)["route"] == expected_route


@pytest.mark.parametrize(
    "defect",
    ["missing_issues", "all_gates_pass", "cross_task_copy"],
)
def test_repair_inheritance_rejects_unsupported_or_copied_defects_without_writes(
    tmp_path, monkeypatch, defect
):
    definition, active, packet = _repair_history_fixture(tmp_path, monkeypatch, "high")
    source = active["history"][0]
    if defect == "missing_issues":
        source["assessments"][0]["verdict"]["issues"] = []
    elif defect == "all_gates_pass":
        source["assessments"][0]["verdict"]["gates"]["visual_evidence"] = "pass"
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(report_path.read_text())
    report["results"] = [active]
    if defect == "cross_task_copy":
        report["results"].append({"task_id": "other-task", **deepcopy(source)})
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()

    with pytest.raises(ValueError):
        retain_verified_blind_review(
            definition["id"],
            _factual_verification(definition, active, packet),
            root=tmp_path,
        )

    assert report_path.read_bytes() == before


def test_conflicting_second_pass_does_not_discard_original_blind_facts(
    tmp_path, monkeypatch
):
    definition, active, packet = _repair_history_fixture(tmp_path, monkeypatch, "high")
    source = active["history"][0]
    source["assessments"][1]["verdict"] = _verdict(adversarial=True)
    path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(path.read_text())
    report["results"] = [active]
    path.write_text(json.dumps(report))

    retained = retain_verified_blind_review(
        definition["id"],
        _factual_verification(definition, active, packet),
        root=tmp_path,
    )
    completed = record_assessment(
        definition["id"],
        "adversarial_falsification",
        _new_verdict(adversarial=True),
        root=tmp_path,
    )

    assert retained["assessments"] == source["assessments"][:1]
    assert route_assessment(task_sha256(definition), completed)["route"] == "repair"


def test_low_confidence_inheritance_preserves_uncertainty_without_inventing_an_issue(
    tmp_path, monkeypatch
):
    definition, active, packet = _repair_history_fixture(tmp_path, monkeypatch, "low")
    for phase in active["history"][0]["assessments"]:
        phase["verdict"]["issues"] = []
        phase["verdict"]["uncertainty_reasons"] = [
            "The bottle's cleaning use is not readable in the released view."
        ]
    path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(path.read_text())
    report["results"] = [active]
    path.write_text(json.dumps(report))

    retained = retain_verified_blind_review(
        definition["id"],
        _factual_verification(definition, active, packet),
        root=tmp_path,
    )

    assert retained["assessments"] == active["history"][0]["assessments"][:1]
    assert retained["assessments"][0]["verdict"]["issues"] == []


@pytest.mark.parametrize(
    "defect",
    [
        "missing_body",
        "wrong_task",
        "missing_entities",
        "missing_instructions",
        "unresolved",
        "fabricated_source",
        "missing_reviewer",
        "stale_materials",
        "explicit_reopen",
        "changed_answer",
        "cross_task_copy",
    ],
)
def test_factual_inheritance_rejects_unproven_evidence_without_writes(
    tmp_path, monkeypatch, defect
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "instruction"
    )
    verification = _factual_verification(definition, active, packet)
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(report_path.read_text())
    if defect == "missing_body":
        report["results"][0]["history"] = []
    elif defect == "wrong_task":
        verification["task_id"] = "other-task"
    elif defect == "missing_entities":
        verification["evidence"].pop("site_mapping")
    elif defect == "missing_instructions":
        verification["evidence"]["user_instruction"]["current_evidence"] = []
    elif defect == "unresolved":
        verification["evidence"]["visual_evidence"]["conclusion"] = "uncertain"
    elif defect == "fabricated_source":
        verification["evidence"]["visual_evidence"]["source_evidence"] = [
            "never observed"
        ]
    elif defect == "missing_reviewer":
        verification.pop("reviewer")
    elif defect == "stale_materials":
        verification["current_materials_sha256"] = "0" * 64
    elif defect == "explicit_reopen":
        report["results"][0]["pending_reason"] = "incorrect_visual_evidence"
    elif defect == "changed_answer":
        definition["goal"]["acceptable_states"][0]["selected_entity_ids"] = ["other"]
        index_path = tmp_path / "artifacts/reports/confidence-review-packets/index.json"
        index = json.loads(index_path.read_text())
        index["tasks"][definition["id"]]["task_sha256"] = task_sha256(definition)
        index_path.write_text(json.dumps(index))
    elif defect == "cross_task_copy":
        report["results"].append(
            {"task_id": "other-task", **deepcopy(active["history"][0])}
        )
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()

    with pytest.raises(ValueError):
        retain_verified_blind_review(definition["id"], verification, root=tmp_path)

    assert report_path.read_bytes() == before


def test_explicit_task_invalidation_does_not_restore_history(tmp_path, monkeypatch):
    definition, active, packet = _history_review_fixture(tmp_path, monkeypatch)
    active["pending_reason"] = "incorrect_visual_evidence"
    assert reusable_review_evidence(definition, active, packet) is None


@pytest.mark.parametrize("reason", ["initial_assessment", "review_materials_changed"])
def test_explicit_invalidation_cannot_reuse_a_system_pending_reason(
    tmp_path, monkeypatch, reason
):
    definition, active, packet = _history_review_fixture(tmp_path, monkeypatch)
    start_review(
        tmp_path,
        refresh=True,
        task_ids=[definition["id"]],
        invalidation_reasons={definition["id"]: reason},
    )
    report_path = tmp_path / confidence_session.REPORT_PATH
    row = json.loads(report_path.read_text())["results"][0]

    assert row["pending_reason"] == f"explicit_task_invalidation: {reason}"
    assert reusable_review_evidence(definition, row, packet) is None
    with pytest.raises(ValueError, match="explicitly reopened"):
        retain_verified_blind_review(
            definition["id"],
            _factual_verification(definition, active, packet),
            root=tmp_path,
        )


def test_identical_historical_evidence_needs_no_new_review(tmp_path, monkeypatch):
    definition, active, packet = _history_review_fixture(tmp_path, monkeypatch)
    active["history"][0]["materials_sha256"] = packet["materials_sha256"]
    report_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(report_path.read_text())
    report["results"] = [active]
    report_path.write_text(json.dumps(report))
    before = report_path.read_bytes()

    confidence = build_task_confidence.build_confidence_report(tmp_path)

    assert confidence["tasks"][definition["id"]]["level"] == "high"
    assert "recheck_passes" not in confidence["tasks"][definition["id"]]
    assert report_path.read_bytes() == before


def test_same_task_history_is_not_counted_as_cross_task_template_reuse():
    completed = {"assessments": [{"pass": "blind"}, {"pass": "adversarial"}]}
    one_task = {**completed, "history": [deepcopy(completed)]}
    fingerprints = build_task_confidence.assessment_fingerprints([one_task])
    assert list(fingerprints.values()) == [1]
    reused = build_task_confidence.assessment_fingerprints([one_task, completed])
    assert list(reused.values()) == [2]


def test_refresh_does_not_claim_old_review_saw_current_materials_without_evidence():
    previous = {
        "task_id": "missing-material-proof",
        "status": "completed",
        "assessments": [
            {"pass": "blind_reconstruction"},
            {"pass": "adversarial_falsification"},
        ],
    }
    rows = _refresh_results(
        {previous["task_id"]: {"materials_sha256": "a" * 64}},
        {"results": [previous]},
        {},
    )
    assert rows[0]["status"] == "pending"
    assert rows[0]["pending_reason"] == "review_materials_missing"
    assert rows[0]["assessments"] == []
    assert rows[0]["history"][0]["assessments"] == previous["assessments"]


def test_answer_change_archives_concern_resolutions_without_reusing_them():
    previous = {
        "task_id": "resolved-task",
        "status": "completed",
        "materials_sha256": "a" * 64,
        "answer_boundary": answer_boundary(task("graft")),
        "assessments": [
            {"pass": "blind_reconstruction"},
            {"pass": "adversarial_falsification"},
        ],
        "resolution_reviews": [
            {
                "reviewer": {"agent_id": "actual", "context_id": "actual"},
                "blind_resolutions": [],
            }
        ],
    }
    rows = _refresh_results(
        {
            previous["task_id"]: {
                "materials_sha256": "a" * 64,
                "answer_boundary": answer_boundary(task("refresh")),
            }
        },
        {"results": [previous]},
        {},
    )
    assert "resolution_reviews" not in rows[0]
    assert rows[0]["history"][0]["resolution_reviews"] == previous["resolution_reviews"]
    assert rows[0]["assessments"] == previous["assessments"][:1]


@pytest.mark.parametrize(
    "scope,expected",
    [(["--task-id", "a", "--task-id", "b"], ["a", "b"]), (["--all-tasks"], None)],
)
def test_session_cli_passes_explicit_refresh_scope(monkeypatch, scope, expected):
    import sys
    from tools.audit import task_confidence_session as cli

    calls = []

    def start(root, **kwargs):
        calls.append(kwargs)
        return {"summary": {}}

    monkeypatch.setattr(cli, "start_review", start)
    monkeypatch.setattr(
        sys, "argv", ["task_confidence_session", "start", "--refresh", *scope]
    )
    cli.main()
    assert calls[0]["task_ids"] == expected


def test_session_cli_refuses_unscoped_refresh(monkeypatch):
    import sys
    from tools.audit import task_confidence_session as cli

    monkeypatch.setattr(
        cli,
        "start_review",
        lambda *args, **kwargs: pytest.fail("must not mutate reviews"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "task_confidence_session",
            "start",
            "--refresh",
            "--invalidate-task",
            "a=repair",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("status", ["needs_repair", "needs_review", "ready"])
def test_session_cli_distinguishes_review_completion_from_task_status(
    monkeypatch, capsys, status
):
    import sys
    from tools.audit import task_confidence_session as cli

    recorded = {
        "task_id": "example",
        "status": "completed",
        "assessments": [{}, {}],
    }
    lifecycle = {"status": status, "next_action": "next", "reason": "evidence"}
    monkeypatch.setattr(cli, "record_assessment", lambda *args, **kwargs: recorded)

    class Workflow:
        def __init__(self, root):
            assert root == "."

        def status(self, task_ids):
            assert task_ids == ["example"]
            return {"tasks": {"example": lifecycle}}

    monkeypatch.setattr(cli, "SynthesisWorkflow", Workflow)
    monkeypatch.setattr(
        sys, "argv", ["task_confidence_session", "record", "example", "pass", "{}"]
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)
    assert output == {
        "task_id": "example",
        "assessment_status": "completed",
        "passes": 2,
        "lifecycle": lifecycle,
    }


def test_current_sealed_blind_survives_key_repair_during_second_pass(
    tmp_path, monkeypatch
):
    definition, active, packet = _history_review_fixture(
        tmp_path, monkeypatch, "answer"
    )
    original = deepcopy(active["history"][0])
    definition["goal"]["acceptable_states"].append(
        deepcopy(original["assessments"][0]["verdict"]["derived_final_state"])
    )
    index_path = tmp_path / confidence_session.INDEX_PATH
    index = json.loads(index_path.read_text())
    index["tasks"][definition["id"]]["task_sha256"] = task_sha256(definition)
    index_path.write_text(json.dumps(index))
    verification = _factual_verification(definition, active, packet)
    active.update(
        materials_sha256=original["materials_sha256"],
        status="in_progress",
        pending_reason="explicit_task_invalidation: new purpose",
        assessments=original["assessments"][:1],
        history=[],
    )
    report_path = tmp_path / confidence_session.REPORT_PATH
    report = json.loads(report_path.read_text())
    report["results"] = [active]
    report_path.write_text(json.dumps(report))

    retained = retain_verified_blind_review(
        definition["id"], verification, root=tmp_path
    )

    assert retained["assessments"] == original["assessments"][:1]
    assert retained["materials_sha256"] == packet["materials_sha256"]
    assert open_pass(definition["id"], root=tmp_path)["pass"] == (
        "adversarial_falsification"
    )


def test_import_blind_keeps_original_and_requires_current_verification(tmp_path):
    from egocua.confidence_session import import_blind_review

    _write_review_packet(tmp_path, 'task')
    start_review(tmp_path)
    packet = 'artifacts/reports/confidence-review-packets/task.json'
    verdict = _new_verdict()
    verdict['derived_final_state'] = {'selected_entity_ids': ['target'], 'fields': {}}
    (tmp_path / 'original-blind.json').write_text(json.dumps(verdict))
    evidence = {'packet': packet, 'verdict': 'original-blind.json'}

    source = import_blind_review('task', evidence, root=tmp_path)
    import_blind_review('task', evidence, root=tmp_path)

    report = json.loads((tmp_path / confidence_session.REPORT_PATH).read_text())
    row = report['results'][0]
    assert row['status'] == 'pending'
    assert row['assessments'] == []
    assert row['history'] == [source]
    assert source['assessments'][0]['verdict'] == verdict
    before = (tmp_path / confidence_session.REPORT_PATH).read_bytes()
    with pytest.raises(ValueError, match='belongs to another task'):
        import_blind_review('other-task', evidence, root=tmp_path)
    assert (tmp_path / confidence_session.REPORT_PATH).read_bytes() == before


def test_segmented_diagnostic_videos_are_revealed_only_after_blind_sealing(tmp_path):
    definition = task("clarify")
    task_id = definition["id"]
    _write_review_packet(tmp_path, task_id)
    task_path = tmp_path / "data/tasks/clarify/task.json"
    task_path.parent.mkdir(parents=True)
    task_path.write_text(json.dumps(definition))
    videos = tmp_path / "videos"
    videos.mkdir()
    segments = []
    for media_id, start in (("part-a", 10), ("part-b", 600)):
        (videos / f"{media_id}.mp4").write_bytes(b"fixture")
        segments.append({
            "media_id": media_id,
            "original_video_id": "original",
            "source_start_seconds": start,
            "source_end_seconds": start + 10,
            "media_sha256": "a" * 64,
        })
    (tmp_path / "source.json").write_text(json.dumps({
        "dataset": {"source_root": "videos"},
        "tasks": [{
            "task_id": task_id,
            "source": {"media_id": "part-a", "original_video_id": "original", "media_segments": segments},
        }],
    }))
    index_path = tmp_path / confidence_session.INDEX_PATH
    index = json.loads(index_path.read_text())
    index["tasks"][task_id]["answer_boundary"] = answer_boundary(definition)
    index["tasks"][task_id]["source_segments_dossier"] = "source.json"
    index_path.write_text(json.dumps(index))
    start_review(tmp_path, reset=True)

    blind = open_pass(task_id, root=tmp_path)
    assert "source_video_segments" not in blind
    assert "part-a.mp4" not in json.dumps(blind)
    record_assessment(task_id, "blind_reconstruction", _new_verdict(), root=tmp_path)
    second = open_pass(task_id, root=tmp_path)
    assert second["source_video_segments"]["unavailable_intervals_seconds"] == [[20, 600]]
    assert len(second["source_video_segments"]["segments"]) == 2
