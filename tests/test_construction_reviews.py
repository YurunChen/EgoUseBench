from __future__ import annotations

import json
from pathlib import Path

import pytest

from egocua.construct_validity import CONFIDENCE_POLICY_VERSION, blind_concerns
from egocua.confidence import resolve_human_review, route_assessment
from egocua.confidence_packets import CurrentReviewMaterials
from egocua.review_materials import ReviewMaterials
from egocua.reviews import (
    audit_construction_reviews,
)
from egocua.run_ledger import task_sha256
from egocua.synthesis_reports import REPORTS
from egocua.synthesis_workflow import SynthesisWorkflow
from tests.helpers import coding_agent_assessment, materialize_assets, products, task


def _workspace(root: Path) -> dict:
    benchmark_task = task("refresh")
    materialize_assets(benchmark_task, products("refresh"), root)
    return benchmark_task


def test_missing_reviews_are_pending(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    for path in (tmp_path / "data/reviews/submissions" / benchmark_task["id"]).glob(
        "*.json"
    ):
        path.unlink()

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["counts"] == {"pending": 1, "admit": 0, "reject": 0, "invalid": 0}
    assert result["tasks"][benchmark_task["id"]]["reason"] == "missing_human_review"


def test_one_current_review_admits(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "admit"
    assert result["all_admitted"] is True


@pytest.mark.parametrize("change", ["repair_history", "named_review", "unresolved"])
def test_source_review_metadata_does_not_expire_human_submission(tmp_path, change):
    definition = _workspace(tmp_path)
    submission = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    original = submission.read_bytes()
    case = ReviewMaterials(tmp_path).case(definition)
    dossier = tmp_path / definition["source"]["dossier"]
    payload = json.loads(dossier.read_text())
    payload["tasks"][0][change] = ["A concrete named source-review note."]
    dossier.write_text(json.dumps(payload))

    current = ReviewMaterials(tmp_path).case(definition)
    audit = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])

    assert current["revision"] == case["revision"]
    assert current["reviewed_artifacts"] == case["reviewed_artifacts"]
    assert not any(
        key.startswith("@source-record/") for key in current["reviewed_artifacts"]
    )
    assert audit["tasks"][definition["id"]]["status"] == "admit"
    assert submission.read_bytes() == original


@pytest.mark.parametrize("change", ["instructions", "frame"])
def test_actual_released_content_change_still_expires_human_submission(
    tmp_path, change
):
    definition = _workspace(tmp_path)
    before = ReviewMaterials(tmp_path).case(definition)
    if change == "instructions":
        definition["instructions"]["base"] = "Choose a different required outcome."
        path = tmp_path / "data/tasks/refresh" / f"{definition['id']}.json"
        path.write_text(json.dumps(definition))
    else:
        frame = tmp_path / definition["visual_packets"][0]["frame_paths"][0]
        frame.write_bytes(b"different actual released pixels")

    after = ReviewMaterials(tmp_path).case(definition)
    audit = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])

    assert after["revision"] != before["revision"]
    assert audit["tasks"][definition["id"]]["status"] == "pending"
    assert audit["tasks"][definition["id"]]["reason"] == "reviewed_materials_changed"


def test_second_review_is_rejected_as_invalid(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    first = (
        tmp_path
        / "data/reviews/submissions"
        / benchmark_task["id"]
        / "annotator-1.json"
    )
    second = first.with_name("annotator-2.json")
    payload = json.loads(first.read_text())
    payload["reviewer"] = "annotator-2"
    second.write_text(json.dumps(payload))

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "invalid"
    assert "expected exactly 1 submission" in " ".join(
        result["tasks"][benchmark_task["id"]]["errors"]
    )


def test_one_failed_check_rejects_the_task(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    path = (
        tmp_path
        / "data/reviews/submissions"
        / benchmark_task["id"]
        / "annotator-1.json"
    )
    submission = json.loads(path.read_text())
    submission["decision"] = "reject"
    submission["note"] = "The evidence is insufficient."
    submission["unresolved"] = ["The evidence is insufficient."]
    path.write_text(json.dumps(submission))

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "reject"


def test_unrelated_site_change_keeps_existing_review(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    site_path = tmp_path / "sites/egomart/products.json"
    site = json.loads(site_path.read_text())
    referenced = {
        benchmark_task["refresh"]["initial_entity_id"],
        *{
            entity_id
            for state in benchmark_task["goal"]["acceptable_states"]
            for entity_id in state["selected_entity_ids"]
        },
    }
    unrelated = next(
        product for product in site["products"] if product["item_id"] not in referenced
    )
    unrelated["name"] = "Changed after review"
    site_path.write_text(json.dumps(site))

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "admit"


def test_referenced_site_entity_change_requires_fresh_review(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    site_path = tmp_path / "sites/egomart/products.json"
    site = json.loads(site_path.read_text())
    referenced = {
        entity_id
        for state in benchmark_task["goal"]["acceptable_states"]
        for entity_id in state["selected_entity_ids"]
    }
    target = next(
        product for product in site["products"] if product["item_id"] in referenced
    )
    target["name"] = "Changed after review"
    site_path.write_text(json.dumps(site))

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "pending"
    assert (
        result["tasks"][benchmark_task["id"]]["reason"] == "reviewed_materials_changed"
    )
    assert "reviewed task materials changed" in " ".join(
        result["tasks"][benchmark_task["id"]]["details"]
    )


def test_default_audit_reads_submissions_without_confidence_routing(
    tmp_path: Path,
) -> None:
    benchmark_task = _workspace(tmp_path)
    confidence_path = tmp_path / "artifacts/reports/task-confidence.json"
    confidence = json.loads(confidence_path.read_text())
    confidence["tasks"][benchmark_task["id"]] = {
        "task_sha256": task_sha256(benchmark_task),
        "policy_version": CONFIDENCE_POLICY_VERSION,
        "level": "high",
        "route": "automatic_construction_admission",
        "human_review_required": False,
    }
    confidence_path.write_text(json.dumps(confidence))

    result = audit_construction_reviews(root=tmp_path)

    assert result["tasks"][benchmark_task["id"]]["status"] == "admit"
    assert result["all_admitted"] is True


def test_default_audit_includes_quality_control_without_task_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark_task = _workspace(tmp_path)
    submission_path = (
        tmp_path
        / "data/reviews/submissions"
        / benchmark_task["id"]
        / "annotator-1.json"
    )
    submission = json.loads(submission_path.read_text())
    submission["purpose"] = "quality_control"
    submission_path.write_text(json.dumps(submission))
    original = submission_path.read_bytes()
    (tmp_path / "artifacts/reports/task-confidence.json").unlink()

    def unexpected_status(*args, **kwargs):
        raise AssertionError("material verification must not read task status")

    monkeypatch.setattr(SynthesisWorkflow, "status", unexpected_status)
    result = audit_construction_reviews(root=tmp_path)

    assert result["tasks"][benchmark_task["id"]]["status"] == "admit"
    assert result["tasks"][benchmark_task["id"]]["purpose"] == "quality_control"
    assert submission_path.read_bytes() == original


def test_spec_change_requires_fresh_review(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    spec_path = tmp_path / "data/reviews/construction.json"
    spec = json.loads(spec_path.read_text())
    spec["checks"]["common"][0]["question"] = "Changed question?"
    spec_path.write_text(json.dumps(spec))

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "pending"
    assert "reviewed task materials changed" in " ".join(
        result["tasks"][benchmark_task["id"]]["details"]
    )


def test_malformed_old_review_is_not_treated_as_pending(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    path = (
        tmp_path
        / "data/reviews/submissions"
        / benchmark_task["id"]
        / "annotator-1.json"
    )
    submission = json.loads(path.read_text())
    submission["revision"] = "old"
    submission["submitted_at_utc"] = "invalid"
    path.write_text(json.dumps(submission))

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "invalid"
    assert (
        "invalid submission time" in result["tasks"][benchmark_task["id"]]["errors"][0]
    )


def test_stale_review_is_preserved_without_admitting_task(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    path = (
        tmp_path
        / "data/reviews/submissions"
        / benchmark_task["id"]
        / "annotator-1.json"
    )
    submission = json.loads(path.read_text())
    submission["revision"] = "old"
    path.write_text(json.dumps(submission))
    original = path.read_bytes()

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == "pending"
    assert result["all_admitted"] is False
    assert path.read_bytes() == original


@pytest.mark.parametrize("stale", [False, True])
def test_granularity_change_distinguishes_old_from_invalid_review(
    tmp_path: Path, stale: bool
) -> None:
    benchmark_task = _workspace(tmp_path)
    path = (
        tmp_path
        / "data/reviews/submissions"
        / benchmark_task["id"]
        / "annotator-1.json"
    )
    submission = json.loads(path.read_text())
    submission["target_granularity"] = "previous-granularity"
    if stale:
        submission["revision"] = "previous-task"
    path.write_text(json.dumps(submission))
    original = path.read_bytes()

    result = audit_construction_reviews(root=tmp_path, task_ids=[benchmark_task["id"]])

    assert result["tasks"][benchmark_task["id"]]["status"] == (
        "pending" if stale else "invalid"
    )
    assert path.read_bytes() == original


def _human_reuse_review(root: Path, definition: dict, outcome: str) -> dict:
    result = coding_agent_assessment(definition)
    result["materials_sha256"] = CurrentReviewMaterials(root).packet(definition)[
        "materials_sha256"
    ]
    second = result["assessments"][1]["verdict"]
    second["reviewer"] = {
        "agent_id": "independent-targeted-reviewer",
        "context_id": "current",
    }
    second["evidence"]["answer_boundary"] = [
        "Compared the original submission and actual task: the legal initial/final choices and request are equivalent; only the aggregate artifact binding changed."
    ]
    second["human_review_reuse"] = [
        {
            "submission": {
                "path": f"data/reviews/submissions/{definition['id']}/annotator-1.json",
                "submitted_at_utc": json.loads(
                    (
                        root
                        / "data/reviews/submissions"
                        / definition["id"]
                        / "annotator-1.json"
                    ).read_text()
                )["submitted_at_utc"],
            },
            "outcome": outcome,
            "retained_conclusions": [
                "The original scene referent and useful web outcome remain supported."
            ],
            "affected_conclusions": ["Only the new listing's material needs review."]
            if outcome == "partial"
            else [],
            "evidence": ["evidence.answer_boundary.0"],
        }
    ]
    if outcome == "partial":
        second["confidence"] = "low"
        second["uncertainty_reasons"] = ["The new listing's material is not disclosed."]
        second["gates"]["answer_boundary"] = "uncertain"
        second["follow_up"] = [
            {
                "action": "human_review",
                "reason": "Only the new listing's material needs review.",
                "evidence": ["evidence.answer_boundary.0"],
            }
        ]
    return result


def _save_human_reuse(root: Path, result: dict) -> None:
    path = root / REPORTS["assessment"].path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": REPORTS["assessment"].schema_version,
                "results": [result],
            }
        )
    )


@pytest.mark.parametrize("outcome", ["retained", "partial", "repair_verified"])
def test_named_scoped_human_reuse_preserves_original_submission(tmp_path, outcome):
    definition = _workspace(tmp_path)
    source = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    submission = json.loads(source.read_text())
    submission["revision"] = "original-aggregate-revision"
    if outcome == "repair_verified":
        submission.update(
            decision="reject",
            note="The original target omitted a legal substitute.",
            unresolved=["The original target omitted a legal substitute."],
        )
    source.write_text(json.dumps(submission))
    before = source.read_bytes()
    _save_human_reuse(tmp_path, _human_reuse_review(tmp_path, definition, outcome))

    result = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])[
        "tasks"
    ][definition["id"]]

    assert result["status"] == ("pending" if outcome == "partial" else "admit")
    assert result["human_review_reuse"][0]["outcome"] == outcome
    assert source.read_bytes() == before


@pytest.mark.parametrize(
    "fault",
    [
        "only_same_frames",
        "new_unknown",
        "changed_answers",
        "no_current_body",
        "wrong_primary",
    ],
)
def test_old_human_admit_cannot_cover_unchecked_current_scope(tmp_path, fault):
    definition = _workspace(tmp_path)
    source = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    submission = json.loads(source.read_text())
    submission["revision"] = "original-aggregate-revision"
    source.write_text(json.dumps(submission))
    before = source.read_bytes()
    result = _human_reuse_review(tmp_path, definition, "retained")
    second = result["assessments"][1]["verdict"]
    if fault == "only_same_frames":
        second["human_review_reuse"][0]["evidence"] = ["evidence.visual_evidence.0"]
    elif fault == "new_unknown":
        second["uncertainty_reasons"] = [
            "An additional accepted listing has not been checked."
        ]
    elif fault == "changed_answers":
        result["answer_boundary"]["final_goal"]["acceptable_states"] = []
    elif fault == "wrong_primary":
        result["assessments"][0]["verdict"]["derived_final_state"][
            "selected_entity_ids"
        ] = ["wrong"]
    else:
        result["status"] = "in_progress"
    _save_human_reuse(tmp_path, result)

    audited = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])

    assert audited["tasks"][definition["id"]]["status"] == (
        "invalid" if fault == "wrong_primary" else "pending"
    )
    assert source.read_bytes() == before


def _add_blind_question(result: dict) -> list[dict]:
    first, second = [item["verdict"] for item in result["assessments"]]
    first["confidence"] = "low"
    first["gates"]["visual_evidence"] = "uncertain"
    first["uncertainty_reasons"] = ["The new referent may change the useful result."]
    first["issues"] = ["The previously unseen edge may identify another object."]
    second["evidence"]["visual_evidence"] = [
        "The edge belongs to the original referent; no new object or ending is introduced."
    ]
    return [
        {
            "concern": concern,
            "original": original,
            "evidence": ["evidence.visual_evidence.0"],
            "reason": "The current public evidence explicitly resolves this concern.",
        }
        for concern, original in blind_concerns(first).items()
    ]


@pytest.mark.parametrize("outcome", ["retained", "repair_verified"])
@pytest.mark.parametrize("unresolved", ["gate", "reason", "issue", "partial"])
def test_whole_human_reuse_cannot_override_unresolved_blind_evidence(
    tmp_path, outcome, unresolved
):
    definition = _workspace(tmp_path)
    source = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    submission = json.loads(source.read_text())
    if outcome == "repair_verified":
        submission.update(
            decision="reject", note="Original defect", unresolved=["Original defect"]
        )
    source.write_text(json.dumps(submission))
    before = source.read_bytes()
    result = _human_reuse_review(tmp_path, definition, outcome)
    resolutions = _add_blind_question(result)
    first, second = [item["verdict"] for item in result["assessments"]]
    if unresolved != "partial":
        if unresolved != "gate":
            first["gates"]["visual_evidence"] = "pass"
        if unresolved != "reason":
            first["uncertainty_reasons"] = []
        if unresolved != "issue":
            first["issues"] = []
    else:
        second["blind_resolutions"] = resolutions[:1]
    _save_human_reuse(tmp_path, result)
    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )
    assert row["route"] == "human_review_pending"

    audited = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])[
        "tasks"
    ][definition["id"]]

    assert audited["status"] == "invalid"
    assert "resolve the sealed concerns" in audited["errors"][0]
    current = resolve_human_review(row, audited)
    assert current["route"] == "human_review_pending"
    assert current["human_review_required"] is True
    assert source.read_bytes() == before


@pytest.mark.parametrize("outcome", ["retained", "repair_verified"])
@pytest.mark.parametrize("mechanism", ["inline", "appended"])
def test_whole_reuse_applies_valid_current_blind_resolutions(
    tmp_path, outcome, mechanism
):
    definition = _workspace(tmp_path)
    source = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    submission = json.loads(source.read_text())
    submission["revision"] = "original-aggregate-revision"
    if outcome == "repair_verified":
        submission.update(
            decision="reject", note="Original defect", unresolved=["Original defect"]
        )
    source.write_text(json.dumps(submission))
    before = source.read_bytes()
    result = _human_reuse_review(tmp_path, definition, outcome)
    resolutions = _add_blind_question(result)
    second = result["assessments"][1]["verdict"]
    if mechanism == "inline":
        second["blind_resolutions"] = resolutions
    else:
        result["resolution_reviews"] = [
            {"reviewer": second["reviewer"], "blind_resolutions": resolutions}
        ]
    _save_human_reuse(tmp_path, result)

    audited = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])[
        "tasks"
    ][definition["id"]]

    assert audited["status"] == "admit"
    assert audited["human_review_reuse"][0]["outcome"] == outcome
    assert source.read_bytes() == before


def test_new_human_decision_can_resolve_a_question_old_reuse_cannot(tmp_path):
    definition = _workspace(tmp_path)
    result = _human_reuse_review(tmp_path, definition, "retained")
    _add_blind_question(result)
    _save_human_reuse(tmp_path, result)
    source = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    submission = json.loads(source.read_text())
    submission["submitted_at_utc"] = "2026-10-18T12:00:01Z"
    source.write_text(json.dumps(submission))

    audited = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])[
        "tasks"
    ][definition["id"]]
    row = route_assessment(
        task_sha256(definition),
        result,
        expected_goal=definition["goal"],
    )

    assert audited["status"] == "admit"
    assert "human_review_reuse" not in audited
    assert resolve_human_review(row, audited)["route"] == "human_review_complete"


def test_partial_reuse_does_not_let_an_old_current_hash_cover_a_new_question(tmp_path):
    definition = _workspace(tmp_path)
    _save_human_reuse(tmp_path, _human_reuse_review(tmp_path, definition, "partial"))
    result = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])
    assert result["tasks"][definition["id"]]["status"] == "pending"


def test_new_actual_human_submission_is_not_locked_by_old_partial_reuse(tmp_path):
    definition = _workspace(tmp_path)
    _save_human_reuse(tmp_path, _human_reuse_review(tmp_path, definition, "partial"))
    source = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    new_submission = json.loads(source.read_text())
    new_submission["submitted_at_utc"] = "2026-10-18T12:00:01Z"
    source.write_text(json.dumps(new_submission))
    result = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])
    assert result["tasks"][definition["id"]]["status"] == "admit"
    assert "human_review_reuse" not in result["tasks"][definition["id"]]


@pytest.mark.parametrize("outcome", ["retained", "repair_verified"])
def test_human_reuse_must_match_the_original_decision(tmp_path, outcome):
    definition = _workspace(tmp_path)
    source = (
        tmp_path / "data/reviews/submissions" / definition["id"] / "annotator-1.json"
    )
    if outcome == "retained":
        original = json.loads(source.read_text())
        original.update(
            decision="reject",
            note="The original request needed repair.",
            unresolved=["The original request needed repair."],
        )
        source.write_text(json.dumps(original))
    _save_human_reuse(tmp_path, _human_reuse_review(tmp_path, definition, outcome))
    result = audit_construction_reviews(root=tmp_path, task_ids=[definition["id"]])
    assert result["tasks"][definition["id"]]["status"] == "invalid"


def test_review_materials_reuses_one_lazy_reader_and_report_snapshot(
    tmp_path, monkeypatch
):
    from egocua import review_materials as module

    definition = _workspace(tmp_path)
    _save_human_reuse(tmp_path, _human_reuse_review(tmp_path, definition, "retained"))
    reads = []
    readers = []
    read = module._load_object

    def counted_read(path, context):
        if context == "Coding-Agent assessment report":
            reads.append(path)
        return read(path, context)

    def current_reader(root):
        reader = CurrentReviewMaterials(root)
        readers.append(reader)
        return reader

    monkeypatch.setattr(module, "_load_object", counted_read)
    monkeypatch.setattr(
        "egocua.confidence_packets.CurrentReviewMaterials", current_reader
    )
    materials = ReviewMaterials(tmp_path)
    first = materials.case(definition)
    (tmp_path / REPORTS["assessment"].path).write_text(
        json.dumps({"schema_version": 4, "results": []})
    )
    second = materials.case(definition)
    assert first["human_review_reuse"] == second["human_review_reuse"]
    assert len(reads) == len(readers) == 1
    assert ReviewMaterials(tmp_path).case(definition)["human_review_reuse"] == []
    assert len(reads) == 2


def test_current_material_reader_is_not_created_without_reuse_claims(
    tmp_path, monkeypatch
):
    definition = _workspace(tmp_path)

    def unexpected_reader(root):
        raise AssertionError("no reuse declaration needs a current-material reader")

    monkeypatch.setattr(
        "egocua.confidence_packets.CurrentReviewMaterials", unexpected_reader
    )
    assert ReviewMaterials(tmp_path).case(definition)["human_review_reuse"] == []
