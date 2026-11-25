from __future__ import annotations

import json
import hashlib
from pathlib import Path
from html import unescape

import pytest

from tests.helpers import unchanged_task_owners as unchanged_task_owners

from egocua.review_web import create_review_app
from egocua.confidence import route_assessment
from egocua.review_workspace import (
    AlreadySubmittedError,
    ReviewWorkspace,
    ReviewWorkspaceError,
    StaleReviewError,
    _visible_entity_fields,
)
from egocua.review_materials import review_submission_matches_case
from egocua.confidence_packets import CurrentReviewMaterials
from egocua.run_ledger import canonical_sha256, task_sha256
from egocua.reviews import audit_construction_reviews
from egocua.synthesis_workflow import REVIEWABLE_SOURCE_CHECKS, SynthesisWorkflow
from tests.helpers import (
    ARCHIVED_CORPUS,
    coding_agent_assessment,
    live_corpus_empty,
    materialize_assets,
    products,
    task,
)

pytestmark = pytest.mark.usefixtures("unchanged_task_owners")


@pytest.mark.parametrize(
    "changed_reference",
    [
        "@review-contract",
        "@task/test-review",
        "@entity/egomart/product-a",
    ],
)
def test_legacy_review_cannot_certify_changed_semantic_material(
    changed_reference: str,
) -> None:
    payload = {
        "reviewed_artifacts": {
            "data/tasks/graft/test-review.json": "unchanged-task",
            "image.jpg": "unchanged-image",
        },
        "revision": "old-review",
    }
    current = {changed_reference: "changed-content", "image.jpg": "unchanged-image"}

    assert not review_submission_matches_case(
        payload,
        {"reviewed_artifacts": current, "revision": "new-review"},
    )


def _pending_workspace(
    root: Path, reviewer: str = "reviewer-a"
) -> tuple[dict, ReviewWorkspace]:
    benchmark_task = task("clarify")
    materialize_assets(benchmark_task, products("clarify"), root)
    directory = root / "data/reviews/submissions" / benchmark_task["id"]
    for path in directory.glob("*.json"):
        path.unlink()
    return benchmark_task, ReviewWorkspace.open(
        root, reviewer, task_ids=[benchmark_task["id"]]
    )


def _response(case: dict, *, failed: str | None = None) -> dict:
    return {
        "decision": "reject" if failed else "admit",
        "note": "The evidence is insufficient." if failed else "",
    }


def _set_low_confidence(root: Path) -> None:
    path = root / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(path.read_text())
    for assessment in report["results"][0]["assessments"]:
        assessment["verdict"]["confidence"] = "low"
        assessment["verdict"]["uncertainty_reasons"] = ["Check the visible candidate"]
    path.write_text(json.dumps(report))


def test_review_evidence_audit_does_not_call_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    case = workspace.view(task_id)
    workspace.submit(task_id, _response(case), expected_revision=case["revision"])

    def unexpected_status(*args, **kwargs):
        raise AssertionError("material verification must not call task status")

    monkeypatch.setattr(SynthesisWorkflow, "status", unexpected_status)
    audit = audit_construction_reviews(root=tmp_path, task_ids=[task_id])
    assert audit["tasks"][task_id]["status"] == "admit"
    assert workspace.review_materials(task_id)["revision"] == case["revision"]


@pytest.mark.parametrize(
    "page", ["/review?view=all", "/review/task/test-graft?view=all"]
)
def test_relax_task_renders_in_review_pages(tmp_path: Path, page: str) -> None:
    definition = task("graft")
    definition["graft"]["transition"] = "relax"
    definition["graft"]["initial_goal"], definition["goal"] = (
        definition["goal"],
        definition["graft"]["initial_goal"],
    )
    materialize_assets(definition, products("graft"), tmp_path)
    workspace = ReviewWorkspace.open(tmp_path, "reviewer-a", include_all_tasks=True)
    client = create_review_app(workspace).test_client()

    response = client.get(page)

    assert response.status_code == 200
    assert "放宽原约束" in response.get_data(as_text=True)


@pytest.mark.parametrize("legacy_byte_hash", [False, True])
@pytest.mark.parametrize("change_question", [False, True])
def test_review_reuses_same_contract_content_without_rewriting_evidence(
    tmp_path: Path, legacy_byte_hash: bool, change_question: bool
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    case = workspace.view(task_id)
    workspace.submit(task_id, _response(case), expected_revision=case["revision"])
    submission = tmp_path / "data/reviews/submissions" / task_id / "reviewer-a.json"
    spec_path = tmp_path / "data/reviews/construction.json"
    if legacy_byte_hash:
        payload = json.loads(submission.read_text())
        payload["reviewed_artifacts"]["@review-contract"] = hashlib.sha256(
            spec_path.read_bytes()
        ).hexdigest()
        payload["revision"] = canonical_sha256(
            {
                "policy": workspace.spec["reviewer_policy"],
                "checks": case["checks"],
                "artifacts": payload["reviewed_artifacts"],
                "experiment": case["experiment"],
            }
        )
        submission.write_text(json.dumps(payload))
    original = submission.read_bytes()
    spec = json.loads(spec_path.read_text())
    if change_question:
        spec["checks"]["common"][0]["question"] = "Changed semantic question?"
    spec_path.write_text(json.dumps(spec, indent=4, sort_keys=True))

    reopened = ReviewWorkspace.open(tmp_path, "reviewer-a", task_ids=[task_id])
    assert reopened.view(task_id)["own_status"] == (
        "stale" if change_question or legacy_byte_hash else "submitted"
    )
    audit = audit_construction_reviews(root=tmp_path, task_ids=[task_id])
    assert audit["tasks"][task_id]["status"] == (
        "pending" if change_question or legacy_byte_hash else "admit"
    )
    assert submission.read_bytes() == original


def test_duplicate_review_decision_is_invalid(tmp_path: Path) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    case = workspace.view(task_id)
    workspace.submit(task_id, _response(case), expected_revision=case["revision"])
    submission = tmp_path / "data/reviews/submissions" / task_id / "reviewer-a.json"
    submission.write_text(
        submission.read_text().replace(
            '"decision": "admit"', '"decision": "reject", "decision": "admit"'
        )
    )

    assert workspace.view(task_id)["own_status"] == "invalid"
    audit = audit_construction_reviews(root=tmp_path, task_ids=[task_id])
    assert audit["tasks"][task_id]["status"] == "invalid"
    assert "duplicate JSON key 'decision'" in audit["tasks"][task_id]["errors"][0]


@pytest.mark.parametrize("outcome", ["retained", "partial", "repair_verified"])
def test_page_uses_shared_scoped_reuse_without_rewriting_old_opinion(tmp_path, outcome):
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    case = workspace.view(task_id)
    workspace.submit(
        task_id,
        _response(
            case, failed="Original defect" if outcome == "repair_verified" else None
        ),
        expected_revision=case["revision"],
    )
    source = tmp_path / "data/reviews/submissions" / task_id / "reviewer-a.json"
    original = json.loads(source.read_text())
    original["revision"] = "original-aggregate-revision"
    original["reviewed_artifacts"] = {
        "@source-record/old-task": "sealed-old-source-record-identity",
        "@entity/egomart/old-product": "sealed-old-nonpublic-metadata-identity",
    }
    source.write_text(json.dumps(original))
    before = source.read_bytes()
    result = coding_agent_assessment(definition)
    result["materials_sha256"] = CurrentReviewMaterials(tmp_path).packet(definition)[
        "materials_sha256"
    ]
    second = result["assessments"][1]["verdict"]
    second["reviewer"] = {
        "agent_id": "named-targeted-reviewer",
        "context_id": "current",
    }
    second["evidence"]["answer_boundary"] = [
        "Compared the original submission and current task: all unchanged conclusions remain valid; the affected scope is explicitly listed."
    ]
    second["human_review_reuse"] = [
        {
            "submission": {
                "path": source.relative_to(tmp_path).as_posix(),
                "submitted_at_utc": original["submitted_at_utc"],
            },
            "outcome": outcome,
            "retained_conclusions": [
                "Original referent and accepted final choices remain supported."
            ],
            "affected_conclusions": ["新增候选的材质尚未披露"]
            if outcome == "partial"
            else [],
            "evidence": ["evidence.answer_boundary.0"],
        }
    ]
    if outcome == "partial":
        second["confidence"] = "low"
        second["gates"]["answer_boundary"] = "uncertain"
        second["uncertainty_reasons"] = [
            "The additional candidate material remains undisclosed."
        ]
    report = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    payload = json.loads(report.read_text())
    payload["results"] = [result]
    report.write_text(json.dumps(payload))
    reopened = ReviewWorkspace.open(tmp_path, "reviewer-a", task_ids=[task_id])

    case = reopened.review_materials(task_id)
    audited = audit_construction_reviews(root=tmp_path, task_ids=[task_id])["tasks"][
        task_id
    ]
    response = (
        create_review_app(reopened)
        .test_client()
        .get(f"/review/task/{task_id}?view=all")
    )

    assert case["own_status"] == ("stale" if outcome == "partial" else "submitted")
    assert audited["status"] == ("pending" if outcome == "partial" else "admit")
    assert case["human_review_reuse"][0]["outcome"] == outcome
    assert response.status_code == 200
    page = response.get_data(as_text=True)
    assert "旧人工意见已保留" in page
    if outcome == "partial":
        assert "仅新增问题待判" in page
        assert "新增候选的材质尚未披露" in page
    else:
        assert "你的意见：已提交" in page
    assert source.read_bytes() == before


def _record_source_pending(root: Path, definition: dict, check: str) -> Path:
    dossier_path = root / definition["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())
    record = next(row for row in dossier["tasks"] if row["task_id"] == definition["id"])
    record["blocking_checks"] = [check]
    record["unresolved"] = ["Source redistribution still needs review."]
    dossier_path.write_text(json.dumps(dossier))
    for name in (
        "task-acceptance",
        "task-smoke",
        "role-recoverability",
        "task-confidence",
    ):
        path = root / f"artifacts/reports/{name}.json"
        report = json.loads(path.read_text())
        row = report["tasks"][definition["id"]]
        row["task_sha256"] = task_sha256(definition)
        if name == "task-acceptance":
            row.update(
                status="needs_source_review", integrity_pass=True, failed_checks=[]
            )
        path.write_text(json.dumps(report))
    path = root / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(path.read_text())
    report["results"][0]["task_sha256"] = task_sha256(definition)
    path.write_text(json.dumps(report))
    return dossier_path


@pytest.mark.parametrize("check", sorted(REVIEWABLE_SOURCE_CHECKS))
def test_human_quality_admit_preserves_source_release_blockers(
    tmp_path: Path,
    check: str,
) -> None:
    definition, _ = _pending_workspace(tmp_path)
    task_id = definition["id"]
    dossier_path = _record_source_pending(tmp_path, definition, check)
    original_source = dossier_path.read_bytes()
    report_paths = [
        tmp_path / f"artifacts/reports/{name}.json"
        for name in (
            "task-acceptance",
            "task-confidence",
            "coding-agent-task-assessments",
        )
    ]
    original_reports = {path: path.read_bytes() for path in report_paths}
    workspace = ReviewWorkspace.open(tmp_path, "reviewer-a", task_ids=[task_id])
    case = workspace.view(task_id)

    assert check in case["pending_source_items"]
    assert case["release_blockers"]
    assert not case["quality_review_blockers"]
    client = create_review_app(workspace).test_client()
    page = client.get(f"/review/task/{task_id}?view=quality").get_data(as_text=True)
    assert 'name="decision" value="admit"' in page
    assert "质量结论不会解除这些发布阻塞" in page

    response = client.post(
        f"/review/task/{task_id}/submit?view=quality",
        data={
            "revision": case["revision"],
            "decision": "admit",
            "note": "",
        },
    )

    assert response.status_code == 302
    assert dossier_path.read_bytes() == original_source
    current = SynthesisWorkflow(tmp_path).status()["tasks"][task_id]
    assert current["status"] == "needs_review"
    assert current["next_action"] == "resolve_source_review"
    assert {path: path.read_bytes() for path in report_paths} == original_reports


@pytest.mark.parametrize(
    "check", ["confirmed_source_contradiction", "unknown_source_failure"]
)
def test_explicit_quality_review_cannot_admit_confirmed_or_unknown_source_errors(
    tmp_path: Path,
    check: str,
) -> None:
    definition, _ = _pending_workspace(tmp_path)
    _record_source_pending(tmp_path, definition, check)
    task_id = definition["id"]
    assert ReviewWorkspace.open(tmp_path, "reviewer-a").ordered_ids == []
    workspace = ReviewWorkspace.open(tmp_path, "reviewer-a", task_ids=[task_id])
    case = workspace.view(task_id)

    assert case["quality_review_blockers"]
    with pytest.raises(ReviewWorkspaceError, match="不能保留"):
        workspace.submit(task_id, _response(case), expected_revision=case["revision"])


@pytest.mark.parametrize("failure", ["missing", "failed", "incomplete_review"])
def test_explicit_quality_review_cannot_bypass_required_audits(
    tmp_path: Path,
    failure: str,
) -> None:
    definition, _ = _pending_workspace(tmp_path)
    task_id = definition["id"]
    name = (
        "coding-agent-task-assessments"
        if failure == "incomplete_review"
        else "task-acceptance"
    )
    path = tmp_path / f"artifacts/reports/{name}.json"
    report = json.loads(path.read_text())
    if failure == "missing":
        del report["tasks"][task_id]
    elif failure == "failed":
        report["tasks"][task_id].update(
            status="failed", integrity_pass=False, failed_checks=["structure"]
        )
    else:
        report["results"][0]["assessments"].pop()
    path.write_text(json.dumps(report))
    workspace = ReviewWorkspace.open(tmp_path, "reviewer-a", task_ids=[task_id])
    case = workspace.view(task_id)

    assert case["quality_review_blockers"]
    with pytest.raises(ReviewWorkspaceError, match="不能保留"):
        workspace.submit(task_id, _response(case), expected_revision=case["revision"])


@pytest.mark.parametrize("metadata", ["fingerprint", "report_version"])
def test_quality_review_keeps_approval_when_only_audit_metadata_changes(
    tmp_path: Path, metadata: str
) -> None:
    definition, _ = _pending_workspace(tmp_path)
    task_id = definition["id"]
    path = tmp_path / "artifacts/reports/task-acceptance.json"
    report = json.loads(path.read_text())
    if metadata == "fingerprint":
        report["tasks"][task_id]["task_sha256"] = "changed-metadata"
    else:
        report["schema_version"] = 999
    path.write_text(json.dumps(report))

    workspace = ReviewWorkspace.open(tmp_path, "reviewer-a", task_ids=[task_id])
    case = workspace.view(task_id)
    assert case["quality_review_blockers"] == []
    workspace.submit(task_id, _response(case), expected_revision=case["revision"])


def test_quality_review_cannot_admit_changed_source_frames(tmp_path: Path) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    dossier_path = tmp_path / definition["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())
    dossier["tasks"][0]["packets"][0]["frame_paths"][0] = "changed-frame.jpg"
    dossier_path.write_text(json.dumps(dossier))

    case = workspace.view(definition["id"])

    assert any("版本不一致" in item for item in case["quality_review_blockers"])
    with pytest.raises(ReviewWorkspaceError, match="版本不一致"):
        workspace.submit(
            definition["id"], _response(case), expected_revision=case["revision"]
        )


def test_source_question_is_readable_without_internal_record_dump(
    tmp_path: Path,
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    dossier_path = tmp_path / definition["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())
    record = next(row for row in dossier["tasks"] if row["task_id"] == definition["id"])
    record["construct"]["event_origin"] = "redesign_same_source"
    record["unresolved"] = [
        {
            "question": "图中对象与教程是否一致？",
            "decisive_reason": "需要比较体型。",
            "source_question_references": ["internal/path#/tasks/0"],
            "action": "human_review",
        }
    ]
    dossier_path.write_text(json.dumps(dossier))
    case = workspace.view(definition["id"])
    assert "图中对象与教程是否一致？；需要比较体型。" in case["pending_source_items"]
    assert not any(
        "source_question_references" in row for row in case["pending_source_items"]
    )
    assert case["scenario_evidence"]["label"] == "沿用原视频重新设计"
    assert "不进入核心" not in case["motivation"]["static_replacement_test"]


def test_workspace_derives_one_case_and_personal_progress(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)

    progress = workspace.view()
    case = workspace.view(benchmark_task["id"])

    assert progress["total"] == 1
    assert progress["submission_counts"]["todo"] == 1
    assert progress["tasks"][0]["preview"]
    assert case["experiment"]["condition"] == "online"
    assert len(case["checks"]) == 3
    assert case["design_profile"] == {
        "initial_options": 2,
        "final_entity_count": 1,
        "required_field_count": 0,
        "change": "确认前有两个合理候选，回答后只剩一个",
    }
    assert case["scenario_evidence"]["label"] == "由原始标注改写"
    assert case["motivation"]["task_risk"] == ""
    assert len(case["scenario_evidence"]["supported_claims"]) == 1
    assert case["own_status"] == "todo"


def test_open_workspace_stops_when_task_membership_changes(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    task_path = (
        tmp_path
        / "data/tasks"
        / benchmark_task["track"]
        / f"{benchmark_task['id']}.json"
    )
    replacement = json.loads(task_path.read_text())
    replacement["id"] = "other-task"
    task_path.with_name("other-task.json").write_text(json.dumps(replacement))
    task_path.unlink()

    with pytest.raises(StaleReviewError, match="任务清单已经变化"):
        workspace.view()


def test_open_workspace_stops_when_selected_membership_changes(tmp_path: Path) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    path = tmp_path / "data/selection.json"
    selection = json.loads(path.read_text())
    selection["task_set"][definition["track"]] = []
    path.write_text(json.dumps(selection))

    with pytest.raises(StaleReviewError, match="任务清单已经变化"):
        workspace.view()


def test_submission_rechecks_current_task_definition(tmp_path: Path) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    case = workspace.view(task_id)
    path = tmp_path / "data/tasks" / definition["track"] / f"{task_id}.json"
    changed = json.loads(path.read_text())
    original_ids = changed["goal"]["acceptable_states"][0]["selected_entity_ids"]
    changed["goal"]["acceptable_states"][0]["selected_entity_ids"] = [
        entity_id
        for entity_id in changed["clarification"]["candidate_ids"]
        if entity_id not in original_ids
    ]
    path.write_text(json.dumps(changed))

    with pytest.raises(StaleReviewError, match="审核材料已经变化"):
        workspace.submit(task_id, _response(case), expected_revision=case["revision"])
    assert not (
        tmp_path / "data/reviews/submissions" / task_id / "reviewer-a.json"
    ).exists()


def test_single_linked_candidate_is_shown_as_a_review_warning(tmp_path: Path) -> None:
    benchmark_task = task("refresh")
    initial = benchmark_task["refresh"]["initial_entity_id"]
    benchmark_task["refresh"]["policy"] = "retain"
    benchmark_task["goal"]["acceptable_states"][0]["selected_entity_ids"] = [initial]
    materialize_assets(benchmark_task, products("refresh"), tmp_path)

    case = ReviewWorkspace.open(
        tmp_path, "reviewer-a", task_ids=[benchmark_task["id"]]
    ).view(benchmark_task["id"])

    assert len(case["entities"]) == 1
    assert "只关联了一个网页候选" in case["review_warnings"][0]


def test_missing_site_asset_blocks_only_its_case_not_the_review_queue(
    tmp_path: Path,
) -> None:
    benchmark_task, _ = _pending_workspace(tmp_path)
    site_records = products("clarify")
    referenced = set(benchmark_task["clarification"]["candidate_ids"])
    record = next(
        item
        for item in site_records
        if item.get("entity_id", item.get("item_id")) in referenced
    )
    missing = tmp_path / record["image_path"]
    missing.unlink()

    workspace = ReviewWorkspace.open(
        tmp_path, "reviewer-b", task_ids=[benchmark_task["id"]]
    )
    case = workspace.view(benchmark_task["id"])

    assert case["release_blockers"]
    assert any("发布材料不完整" in message for message in case["release_blockers"])
    assert case["quality_review_blockers"]
    with pytest.raises(ReviewWorkspaceError, match="不能保留"):
        workspace.submit(
            benchmark_task["id"], _response(case), expected_revision=case["revision"]
        )


def test_submit_is_atomic_final_and_updates_only_personal_progress(
    tmp_path: Path,
) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])

    receipt = workspace.submit(
        benchmark_task["id"], _response(case), expected_revision=case["revision"]
    )

    assert receipt["decision"] == "admit"
    assert workspace.view()["submission_counts"]["submitted"] == 1
    other = ReviewWorkspace.open(
        tmp_path, "reviewer-b", task_ids=[benchmark_task["id"]]
    )
    assert other.view()["submission_counts"]["todo"] == 1
    with pytest.raises(AlreadySubmittedError):
        workspace.submit(
            benchmark_task["id"], _response(case), expected_revision=case["revision"]
        )


def test_stale_submission_can_be_replaced_after_materials_change(
    tmp_path: Path,
) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    workspace.submit(
        benchmark_task["id"], _response(case), expected_revision=case["revision"]
    )
    submission_dir = tmp_path / "data/reviews/submissions" / benchmark_task["id"]
    original = (submission_dir / "reviewer-a.json").read_bytes()
    spec_path = tmp_path / "data/reviews/construction.json"
    spec = json.loads(spec_path.read_text())
    spec["checks"]["common"][0]["question"] = "Changed after the first review?"
    spec_path.write_text(json.dumps(spec))

    refreshed_workspace = ReviewWorkspace.open(
        tmp_path, "reviewer-a", task_ids=[benchmark_task["id"]]
    )
    refreshed = refreshed_workspace.view(benchmark_task["id"])
    assert refreshed["own_status"] == "stale"
    receipt = refreshed_workspace.submit(
        benchmark_task["id"],
        _response(refreshed),
        expected_revision=refreshed["revision"],
    )

    assert receipt["decision"] == "admit"
    assert refreshed_workspace.view(benchmark_task["id"])["own_status"] == "submitted"
    archived = list((submission_dir / "history" / "reviewer-a").glob("*.json"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == original
    assert len(list(submission_dir.glob("*.json"))) == 1
    with pytest.raises(AlreadySubmittedError):
        refreshed_workspace.submit(
            benchmark_task["id"],
            _response(refreshed),
            expected_revision=refreshed["revision"],
        )


def test_web_resubmits_when_review_is_stale(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    workspace.submit(
        benchmark_task["id"], _response(case), expected_revision=case["revision"]
    )
    spec_path = tmp_path / "data/reviews/construction.json"
    spec = json.loads(spec_path.read_text())
    spec["checks"]["common"][0]["question"] = "Changed after the first review?"
    spec_path.write_text(json.dumps(spec))
    refreshed = ReviewWorkspace.open(
        tmp_path, "reviewer-a", task_ids=[benchmark_task["id"]]
    ).view(benchmark_task["id"])
    client = create_review_app(workspace).test_client()

    page = client.get(f"/review/task/{benchmark_task['id']}?view=quality")
    html = page.get_data(as_text=True)
    response = client.post(
        f"/review/task/{benchmark_task['id']}/submit?view=quality",
        data={
            "revision": refreshed["revision"],
            "decision": "admit",
            "note": "",
        },
        follow_redirects=True,
    )

    assert "对应材料已变化" in html
    assert "已经提交" not in html
    assert "答案没问题，保留" in html
    assert response.status_code == 200
    assert "抽查已记录，录取状态不变" in response.get_data(as_text=True)
    assert "已经提交过该任务" not in response.get_data(as_text=True)


def test_material_change_rejects_an_open_page_submission(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    spec_path = tmp_path / "data/reviews/construction.json"
    spec = json.loads(spec_path.read_text())
    spec["checks"]["common"][0]["question"] = "Changed after opening?"
    spec_path.write_text(json.dumps(spec))

    with pytest.raises(StaleReviewError):
        workspace.submit(
            benchmark_task["id"], _response(case), expected_revision=case["revision"]
        )


def test_local_web_flow_shows_task_and_ground_truth_for_review(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    source_dir = tmp_path / "reference/fixture"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / f"fixture-{benchmark_task['track']}.mp4").write_bytes(
        b"fixture video"
    )
    app = create_review_app(workspace)
    client = app.test_client()

    index = client.get("/review")
    page = client.get(f"/review/task/{benchmark_task['id']}?view=quality")

    assert index.status_code == 200
    assert "任务审核" in index.get_data(as_text=True)
    assert "平均评分" in page.get_data(as_text=True)
    assert "基准答案" in page.get_data(as_text=True)
    assert "目标与候选对照" in page.get_data(as_text=True)
    assert "可接受条目" in page.get_data(as_text=True)
    assert "不代表全部都要选择" in page.get_data(as_text=True)
    assert "只看图片和标题" in page.get_data(as_text=True)
    assert "查看判分时要求的最终状态" in page.get_data(as_text=True)
    assert "高置信质量抽查" in page.get_data(as_text=True)
    assert "答案没问题，保留" in page.get_data(as_text=True)
    assert "打开冻结网站核对" in page.get_data(as_text=True)
    assert "最初要求" in page.get_data(as_text=True)
    assert "现实场景依据" in page.get_data(as_text=True)
    assert "现实画面与后续变化是否必要" in page.get_data(as_text=True)
    assert "最关键的反事实检查" in page.get_data(as_text=True)
    assert "画面支持的关键事实" in page.get_data(as_text=True)
    assert "已定位到抽帧前后" in page.get_data(as_text=True)
    assert "结构难度" not in page.get_data(as_text=True)
    assert "先独立判断" not in page.get_data(as_text=True)
    assert "低置信" not in page.get_data(as_text=True)
    assert "校准抽样" not in page.get_data(as_text=True)
    assert client.get("/review/asset?path=../outside").status_code == 404


def test_review_site_switches_between_all_preview_and_human_queue(
    tmp_path: Path,
) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    client = create_review_app(workspace).test_client()

    pending = client.get("/review?view=pending")
    preview = client.get("/review?view=all")

    pending_html = pending.get_data(as_text=True)
    preview_html = preview.get_data(as_text=True)
    assert pending.status_code == 200
    assert "需要人工审核的任务" in pending_html
    assert "补查后仍不确定的具体问题" in pending_html
    assert "不交人工" not in pending_html
    assert 'href="/review?view=all"' in pending_html
    assert preview.status_code == 200
    assert "全部任务" in preview_html
    assert "只读预览" in preview_html
    assert benchmark_task["id"] in preview_html
    assert 'href="/review?view=pending"' in preview_html
    assert 'id="quality-control"' in preview_html
    assert "高置信抽查" in preview_html
    assert "通过不代替其他验收，发现问题则退回该题修复" in preview_html


@pytest.mark.parametrize(
    "page",
    [
        "/review?view=all",
        "/review?view=pending",
        "/review/task/test-clarify?view=all",
        "/review/task/test-clarify?view=pending",
        "/review/task/test-clarify/submit?view=pending",
    ],
)
def test_review_requests_reuse_the_initial_lifecycle(
    tmp_path: Path, monkeypatch, page: str
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    _set_low_confidence(tmp_path)
    current = SynthesisWorkflow(tmp_path).status()
    assert current["tasks"][task_id]["next_action"] == "human_review"
    case = workspace.view(task_id)
    calls = []
    original_status = SynthesisWorkflow.status

    def counted_status(self, task_ids=None, **kwargs):
        calls.append(None if task_ids is None else list(task_ids))
        return original_status(self, task_ids, **kwargs)

    monkeypatch.setattr(SynthesisWorkflow, "status", counted_status)
    app = create_review_app(workspace)
    app.extensions["review_session"].current()
    client = app.test_client()
    assert calls == [[task_id]]
    for _ in range(2):
        calls.clear()
        if "/submit" in page:
            response = client.post(
                page,
                data={"revision": case["revision"], "decision": "invalid"},
            )
            assert response.status_code == 400
        else:
            response = client.get(page)
            assert response.status_code == 200
            assert task_id in response.get_data(as_text=True)
        if "/task/" in page:
            html = response.get_data(as_text=True)
            assert "需要你确认" in html
            assert "<li>Check the visible candidate</li>" in html
        assert calls == []


@pytest.mark.parametrize("view", ["all", "pending"])
def test_human_review_displays_current_uncertainty_reasons(
    tmp_path: Path, monkeypatch, view: str
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    _set_low_confidence(tmp_path)
    questions = ["请确认两个近邻都能使用。", "请核对画面中的接触面。"]
    calls = []
    original_status = SynthesisWorkflow.status

    def counted_status(self, task_ids=None, **kwargs):
        calls.append(list(task_ids))
        current = original_status(self, task_ids, **kwargs)
        assert current["tasks"][definition["id"]]["next_action"] == "human_review"
        current["review_evidence"][definition["id"]]["uncertainty_reasons"] = questions
        return current

    monkeypatch.setattr(SynthesisWorkflow, "status", counted_status)

    response = (
        create_review_app(workspace)
        .test_client()
        .get(f"/review/task/{definition['id']}?view={view}")
    )

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert html.index("需要你确认") < html.index("1. 用户指令")
    for question in questions:
        assert f"<li>{question}</li>" in html
    assert calls == [[definition["id"]]]


@pytest.mark.parametrize("view", ["all", "quality"])
def test_ready_preview_does_not_display_pending_questions(
    tmp_path: Path, monkeypatch, view: str
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    calls = []
    original_status = SynthesisWorkflow.status

    def counted_status(self, task_ids=None, **kwargs):
        calls.append(list(task_ids))
        current = original_status(self, task_ids, **kwargs)
        assert current["tasks"][definition["id"]]["status"] == "ready"
        current["review_evidence"][definition["id"]]["uncertainty_reasons"] = [
            "A prior question must not be shown as pending."
        ]
        return current

    monkeypatch.setattr(SynthesisWorkflow, "status", counted_status)

    response = (
        create_review_app(workspace)
        .test_client()
        .get(f"/review/task/{definition['id']}?view={view}")
    )

    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "需要你确认" not in html
    assert "A prior question must not be shown as pending." not in html
    assert calls == [[definition["id"]]]


def test_quality_review_request_derives_current_lifecycle_once(
    tmp_path: Path, monkeypatch
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    calls = []
    original_status = SynthesisWorkflow.status

    def counted_status(self, task_ids=None, **kwargs):
        calls.append(None if task_ids is None else list(task_ids))
        return original_status(self, task_ids, **kwargs)

    monkeypatch.setattr(SynthesisWorkflow, "status", counted_status)
    response = (
        create_review_app(workspace)
        .test_client()
        .get(f"/review/task/{task_id}?view=quality")
    )

    assert response.status_code == 200
    assert calls == [[task_id]]


@pytest.mark.parametrize(
    "page",
    [
        "/review?view=all",
        "/review?view=pending",
        "/review/task/test-clarify?view=all",
        "/review/task/test-clarify?view=pending",
        "/review/task/test-clarify?view=quality",
        "/review/task/test-clarify/submit?view=pending",
        "/review/task/test-clarify/submit?view=quality",
    ],
)
def test_review_request_loads_and_validates_each_catalog_once(
    tmp_path, monkeypatch, page
):
    from egocua import environment

    definition, workspace = _pending_workspace(tmp_path)
    if "quality" not in page:
        _set_low_confidence(tmp_path)
    case = workspace.view(definition["id"])
    loads, public_entities = [], []
    original_load = environment._load_site_data
    original_public = environment._public_entity_data

    def counted_load(*args, **kwargs):
        loads.append((args[1], kwargs["require_assets"]))
        return original_load(*args, **kwargs)

    def counted_public(site, entity):
        public_entities.append(entity["entity_id"])
        return original_public(site, entity)

    monkeypatch.setattr(environment, "_load_site_data", counted_load)
    monkeypatch.setattr(environment, "_public_entity_data", counted_public)
    app = create_review_app(workspace)
    app.extensions["review_session"].current()
    client = app.test_client()
    for _ in range(2):
        loads.clear()
        public_entities.clear()
        if "/submit" in page:
            response = client.post(
                page, data={"revision": case["revision"], "decision": "invalid"}
            )
            assert response.status_code == 400
        else:
            assert client.get(page).status_code == 200
        assert loads == ([("egomart", False)] if "/task/" in page else [])
        assert len(public_entities) == (
            len(products("clarify")) if "/task/" in page else 0
        )
        assert len(set(public_entities)) == len(public_entities)


def test_site_open_keeps_its_required_asset_validation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from egocua import environment, review_web

    definition, workspace = _pending_workspace(tmp_path)
    loads = []
    original = environment._load_site_data

    def counted(*args, **kwargs):
        loads.append((args[1], kwargs["require_assets"]))
        return original(*args, **kwargs)

    runtime = SimpleNamespace(
        server=SimpleNamespace(
            start=lambda: None,
            base_url="http://frozen-site.invalid",
        )
    )
    monkeypatch.setattr(environment, "_load_site_data", counted)
    monkeypatch.setattr(
        review_web, "create_site_runtime", lambda *args, **kwargs: runtime
    )
    client = create_review_app(workspace).test_client()
    loads.clear()
    response = client.get(f"/review/task/{definition['id']}/site?view=all")
    assert response.status_code == 302
    assert loads == [("egomart", False), ("egomart", True)]


def test_unrelated_queue_change_does_not_expire_an_open_workspace(
    tmp_path: Path, monkeypatch
) -> None:
    definition, _ = _pending_workspace(tmp_path)
    workspace = ReviewWorkspace.open(tmp_path, "reviewer-a")
    assert workspace.ordered_ids == []
    original_status = SynthesisWorkflow.status
    calls = []

    def only_current_view_status(self, task_ids=None, **kwargs):
        calls.append(list(task_ids))
        assert list(task_ids) == []
        return original_status(self, task_ids, **kwargs)

    _set_low_confidence(tmp_path)
    monkeypatch.setattr(SynthesisWorkflow, "status", only_current_view_status)

    assert workspace.view()["total"] == 0
    assert calls == [[]]
    monkeypatch.setattr(SynthesisWorkflow, "status", original_status)
    response = create_review_app(workspace).test_client().get("/review?view=pending")
    assert response.status_code == 200
    assert definition["id"] in response.get_data(as_text=True)


def test_completed_human_review_leaves_pending_queue_on_next_request(
    tmp_path: Path,
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    task_id = definition["id"]
    _set_low_confidence(tmp_path)
    case = workspace.view(task_id)
    client = create_review_app(workspace).test_client()
    response = client.post(
        f"/review/task/{task_id}/submit?view=pending",
        data={"revision": case["revision"], "decision": "admit"},
    )

    assert response.status_code == 302
    assert ReviewWorkspace.open(tmp_path, "reviewer-a").ordered_ids == []
    page = client.get(f"/review/task/{task_id}?view=pending")
    assert page.status_code == 404
    preview = client.get(f"/review/task/{task_id}?view=all")
    assert preview.status_code == 200
    assert "已提交" in preview.get_data(as_text=True)
    assert "需要你确认" not in preview.get_data(as_text=True)
    assert "Check the visible candidate" not in preview.get_data(as_text=True)


def test_task_page_separates_task_status_from_personal_submission(
    tmp_path: Path,
) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    client = create_review_app(workspace).test_client()
    status = SynthesisWorkflow(tmp_path).status()["tasks"][benchmark_task["id"]]

    html = client.get(f"/review/task/{benchmark_task['id']}?view=all").get_data(
        as_text=True
    )

    assert f'class="status {status["status"]}">{status["status_label"]}</span>' in html
    assert "你的意见：尚未提交" in html


def test_quality_sample_excludes_current_defects_even_with_cached_high_confidence(
    tmp_path: Path,
) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    path = tmp_path / "artifacts/reports/task-acceptance.json"
    report = json.loads(path.read_text())
    report["tasks"][benchmark_task["id"]]["status"] = "fail"
    path.write_text(json.dumps(report))
    client = create_review_app(workspace).test_client()

    html = client.get("/review?view=all").get_data(as_text=True)

    assert "从 0 条当前高置信任务中独立抽取 0 条" in html


@pytest.mark.parametrize("decision", ["admit", "reject"])
def test_high_confidence_quality_control_uses_existing_single_task_review(
    tmp_path: Path,
    decision: str,
) -> None:
    from egocua.development_gate import REVIEWS, material_identity

    definitions = [task(track) for track in ("clarify", "refresh")]
    for definition in definitions:
        materialize_assets(definition, products(definition["track"]), tmp_path)
    # This fixture represents legacy tasks on the completed shared catalog.
    development_path = tmp_path / REVIEWS
    development_reviews = json.loads(development_path.read_text())
    development_reviews["legacy_baseline"]["tasks"] = {
        definition["id"]: material_identity(definition, tmp_path)
        for definition in definitions
    }
    development_path.write_text(json.dumps(development_reviews))
    for path in (tmp_path / "data/reviews/submissions").glob("*/*.json"):
        path.unlink()
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    assessments = json.loads(assessment_path.read_text())
    assessments["results"] = [
        {
            **coding_agent_assessment(definition),
            "materials_sha256": CurrentReviewMaterials(tmp_path).packet(definition)[
                "materials_sha256"
            ],
        }
        for definition in definitions
    ]
    assessment_path.write_text(json.dumps(assessments))
    confidence_path = tmp_path / "artifacts/reports/task-confidence.json"
    confidence = json.loads(confidence_path.read_text())
    confidence["tasks"] = {
        definition["id"]: route_assessment(task_sha256(definition), assessment)
        for definition, assessment in zip(
            definitions, assessments["results"], strict=True
        )
    }
    confidence_path.write_text(json.dumps(confidence))
    task_id, untouched_id = [definition["id"] for definition in definitions]
    workspace = ReviewWorkspace.open(tmp_path, "qc-reviewer", include_all_tasks=True)
    client = create_review_app(workspace).test_client()
    before = SynthesisWorkflow(tmp_path).status()
    assert {row["status"] for row in before["tasks"].values()} == {"ready"}
    page = client.get(f"/review/task/{task_id}?view=quality")
    assert page.status_code == 200
    assert "高置信质量抽查" in page.get_data(as_text=True)
    response = client.post(
        f"/review/task/{task_id}/submit?view=quality",
        data={
            "revision": workspace.view(task_id)["revision"],
            "decision": decision,
            "note": "The selected tutorial does not show the required step."
            if decision == "reject"
            else "",
        },
    )
    assert response.status_code == 302
    submission = tmp_path / "data/reviews/submissions" / task_id / "qc-reviewer.json"
    assert json.loads(submission.read_text())["purpose"] == "quality_control"
    after = SynthesisWorkflow(tmp_path).status()
    assert after["tasks"][task_id]["status"] == (
        "ready" if decision == "admit" else "needs_repair"
    )
    assert after["tasks"][untouched_id] == before["tasks"][untouched_id]
    assert not list(
        (tmp_path / "data/reviews/submissions" / untouched_id).glob("*.json")
    )
    if decision == "reject":
        confidence["tasks"][task_id] = after["review_evidence"][task_id]
        confidence_path.write_text(json.dumps(confidence))
        submission.unlink()
        restored = SynthesisWorkflow(tmp_path).status()
        assert restored["tasks"][task_id]["status"] == "ready"
        assert restored["tasks"][untouched_id] == before["tasks"][untouched_id]
        html = client.get("/review?view=all").get_data(as_text=True)
        assert "从 2 条当前高置信任务中独立抽取 2 条" in html


def test_quality_control_rejects_tasks_outside_the_high_confidence_sample(
    tmp_path: Path,
) -> None:
    definition, workspace = _pending_workspace(tmp_path)
    assessment_path = tmp_path / "artifacts/reports/coding-agent-task-assessments.json"
    report = json.loads(assessment_path.read_text())
    for assessment in report["results"][0]["assessments"]:
        assessment["verdict"]["confidence"] = "low"
        assessment["verdict"]["uncertainty_reasons"] = ["Source evidence is unresolved"]
    assessment_path.write_text(json.dumps(report))
    client = create_review_app(workspace).test_client()
    assert (
        client.get(f"/review/task/{definition['id']}?view=quality").status_code == 403
    )
    assert (
        client.post(
            f"/review/task/{definition['id']}/submit?view=quality",
            data={
                "revision": workspace.view(definition["id"])["revision"],
                "decision": "reject",
                "note": "not sampled",
            },
        ).status_code
        == 403
    )


def test_all_task_preview_cannot_submit_a_review(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    client = create_review_app(workspace).test_client()

    preview = client.get(f"/review/task/{benchmark_task['id']}?view=all")
    blocked = client.post(
        f"/review/task/{benchmark_task['id']}/submit?view=all",
        data={
            "revision": workspace.view(benchmark_task["id"])["revision"],
            "decision": "admit",
            "note": "",
        },
    )

    preview_html = preview.get_data(as_text=True)
    assert preview.status_code == 200
    assert "当前是只读预览" in preview_html
    assert "答案没问题，保留" not in preview_html
    assert blocked.status_code == 403
    assert "预览不能提交" in blocked.get_data(as_text=True)
    assert "请切换到人工审核或高置信抽查" in blocked.get_data(as_text=True)


def test_local_web_returns_404_for_removed_task(tmp_path: Path) -> None:
    _, workspace = _pending_workspace(tmp_path)

    response = (
        create_review_app(workspace).test_client().get("/review/task/removed-task")
    )

    assert response.status_code == 404
    assert "unknown review task: removed-task" in response.get_data(as_text=True)


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
def test_current_selected_review_pages_have_complete_answer_entities() -> None:
    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts/reports/task-confidence.json").is_file():
        pytest.skip("task confidence report has not been built")
    workspace = ReviewWorkspace.open(root, "workspace-integration-audit")
    assert workspace.scope == "targeted_confidence_review"
    app = create_review_app(workspace)
    client = app.test_client()

    progress = workspace.view()
    index = client.get("/review")
    index_html = index.get_data(as_text=True)

    assert index.status_code == 200
    lifecycle = SynthesisWorkflow(root).status()["tasks"]
    expected_ids = {
        task_id
        for task_id, row in lifecycle.items()
        if row["next_action"] == "human_review"
    }
    expected_total = len(expected_ids)
    assert progress["total"] == expected_total
    assert set(workspace.ordered_ids) == expected_ids
    for task_id in workspace.ordered_ids:
        case = workspace.view(task_id)
        page = client.get(f"/review/task/{task_id}")

        assert task_id in index_html
        assert page.status_code == 200
        assert not case["quality_review_blockers"]
        if case["pending_source_items"]:
            assert case["release_blockers"]
        site = case["task"]["site"]
        if (
            site == "egomaps"
            and case["task"]["track"] == "graft"
            and case["task"]["graft"]["transition"] == "refine"
        ):
            assert len(case["entities"]) >= 1
            assert all(entity["visual_path"] for entity in case["entities"])
        else:
            assert len(case["entities"]) >= 2
            assert all(entity["visual_path"] for entity in case["entities"])

    rejected_ids = {
        "candidate-clarify-sequence-ego2web-03",
        "candidate-refresh-replace-hdepic-10",
        "candidate-refresh-retain-ego2web-05",
        "candidate-refresh-retain-ego2web-06",
        "candidate-refresh-retain-ego2web-08",
        "seed-refresh-replace-hdepic-03",
    }
    assert all(task_id not in index_html for task_id in rejected_ids)


def test_review_answer_entities_keep_catalog_identity(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    catalog = {item["item_id"]: item for item in products("clarify")}
    case = workspace.view(benchmark_task["id"])

    assert case["entities"]
    for entity in case["entities"]:
        original = catalog[entity["entity_id"]]
        assert entity["name"] == original["name"]
        assert entity.get("attributes", {}) == original.get("attributes", {})


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
@pytest.mark.requires_local_data("data/reviews/construction.json", "data/observations")
def test_refresh_retain_page_shows_initial_target_and_later_distractor() -> None:
    from egocua.schema import load_tasks
    from egocua.sources import source_record

    root = Path(__file__).resolve().parents[1]
    benchmark_task = next(
        row
        for row in load_tasks(root / "data/tasks")
        if row["track"] == "refresh"
        and row["refresh"]["policy"] == "retain"
        and "later_distractor_role"
        in source_record(row, root=root)["construct"]["task_blueprint"]
    )
    blueprint = source_record(benchmark_task, root=root)["construct"]["task_blueprint"]
    initial_id = benchmark_task["refresh"]["initial_entity_id"]
    later_id = blueprint["environment_role_ids"][blueprint["later_distractor_role"]]
    workspace = ReviewWorkspace.open(
        root,
        "workspace-refresh-audit",
        task_ids=[benchmark_task["id"]],
        include_all_tasks=True,
    )
    case = workspace.view(benchmark_task["id"])
    page = (
        create_review_app(workspace)
        .test_client()
        .get(f"/review/task/{benchmark_task['id']}?view=all")
    )
    html = unescape(page.get_data(as_text=True))
    initial = next(entity for entity in case["entities"] if "V0目标" in entity["roles"])
    later = next(
        entity for entity in case["entities"] if "后来出现的对象" in entity["roles"]
    )

    assert page.status_code == 200
    assert initial["entity_id"] == initial_id
    assert later["entity_id"] == later_id
    assert initial["name"] in html
    assert later["name"] in html
    assert "开始时的目标" in html
    assert "后来出现的对象" in html


def test_review_maps_entity_fields_preserve_public_facts() -> None:
    entity = {
        "facts": {
            "listed_city": "Harbor City",
            "route_options": {"walking": "5 min"},
            "entrance": "north side",
        }
    }
    rows = _visible_entity_fields("egomaps", entity)
    visible = " ".join(row["value"] for row in rows)

    assert "Harbor City" in visible
    assert "5 min" in visible
    assert "north side" in visible


def test_web_submission_returns_to_progress(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    app = create_review_app(workspace)
    client = app.test_client()
    data = {
        "revision": case["revision"],
        "decision": "admit",
        "note": "",
    }

    response = client.post(
        f"/review/task/{benchmark_task['id']}/submit?view=quality",
        data=data,
        follow_redirects=True,
    )

    assert response.status_code == 200
    assert "抽查已记录，录取状态不变" in response.get_data(as_text=True)


def test_problem_submission_enters_repair_instead_of_elimination(
    tmp_path: Path,
) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    client = create_review_app(workspace).test_client()

    response = client.post(
        f"/review/task/{benchmark_task['id']}/submit?view=quality",
        data={
            "revision": case["revision"],
            "decision": "reject",
            "note": "The expected product does not match the source object.",
        },
        follow_redirects=True,
    )

    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "已进入修复" in html
    assert "已淘汰" not in html


def test_reject_requires_one_problem_note(tmp_path: Path) -> None:
    benchmark_task, workspace = _pending_workspace(tmp_path)
    case = workspace.view(benchmark_task["id"])
    response = {"decision": "reject", "note": ""}

    with pytest.raises(ValueError, match="发现问题时请简要说明"):
        workspace.submit(
            benchmark_task["id"], response, expected_revision=case["revision"]
        )


def test_review_asset_reads_current_experiment_records(tmp_path):
    _, workspace = _pending_workspace(tmp_path)
    client = create_review_app(workspace).test_client()
    relative = "experiments/example/runs/run-1/run.json"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_text('{"test": "record"}')
    response = client.get("/review/asset", query_string={"path": relative})
    assert response.status_code == 200
    assert response.get_json() == {"test": "record"}
    executable = path.with_suffix(".py")
    executable.write_text("not a review asset")
    assert (
        client.get(
            "/review/asset",
            query_string={"path": executable.relative_to(tmp_path).as_posix()},
        ).status_code
        == 404
    )


def test_segmented_source_page_lists_each_real_video_and_missing_interval(tmp_path):
    definition = task("clarify")
    materialize_assets(definition, products("clarify"), tmp_path)
    dossier_path = tmp_path / definition["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())
    source = dossier["tasks"][0]["source"]
    source.update(media_id="part-a", original_video_id="original", media_start_seconds=10)
    source.pop("media_sha256", None)
    source["media_segments"] = []
    video_root = tmp_path / dossier["dataset"]["source_root"]
    video_root.mkdir(parents=True, exist_ok=True)
    for name, start in (("part-a", 10), ("part-b", 600)):
        (video_root / f"{name}.mp4").write_bytes(b"video fixture")
        source["media_segments"].append({
            "media_id": name,
            "original_video_id": "original",
            "source_start_seconds": start,
            "source_end_seconds": start + 10,
            "media_sha256": "a" * 64,
        })
    dossier_path.write_text(json.dumps(dossier))
    workspace = ReviewWorkspace.open(tmp_path, "reviewer-a", include_all_tasks=True)
    response = create_review_app(workspace).test_client().get(f"/review/task/{definition['id']}?view=all")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert html.count("<video ") == 2
    assert "part-a.mp4" in html and "part-b.mp4" in html
    assert "未下载区间：20.0–600.0 秒" in html
    assert "不代表完整连续视频" in html
