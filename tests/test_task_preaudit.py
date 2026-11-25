import json
from collections import Counter
from pathlib import Path

import pytest

from egocua.run_ledger import task_sha256
from egocua.schema import load_tasks
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty
from tools.build import build_task_preaudit
from tools.build.build_task_preaudit import build_preaudit

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
@pytest.mark.requires_local_data("artifacts/reports/site-audits/index.json")
def test_task_preaudit_covers_current_selection() -> None:
    selection = json.loads((ROOT / "data/selection.json").read_text())
    selected_ids = [
        task_id for task_ids in selection["task_set"].values() for task_id in task_ids
    ]
    compiled = {
        task["id"]: task
        for task in load_tasks(ROOT / "data/tasks")
        if task["id"] in selected_ids
    }
    source_records = {
        record["task_id"]: record
        for reference in selection["source_manifests"]
        for record in json.loads((ROOT / reference).read_text())["tasks"]
        if record["task_id"] in selected_ids
    }
    expected_decisions = {
        task_id: (
            "release_preaudit_pass_with_followups"
            if source_records[task_id].get("unresolved")
            or source_records[task_id].get("blocking_checks")
            else "release_preaudit_pass"
        )
        for task_id in selected_ids
    }
    report = build_preaudit(ROOT)

    assert report["summary"]["tasks"] == len(selected_ids)
    assert report["summary"]["decisions"] == dict(Counter(expected_decisions.values()))
    assert {
        row["task_id"]: row["decision"] for row in report["tasks"]
    } == expected_decisions
    for row in report["tasks"]:
        source = source_records[row["task_id"]]
        assert row["unresolved"] == source.get("unresolved", [])
        assert row["blocking_checks"] == source.get("blocking_checks", [])
    assert report["summary"]["tracks"] == {
        track: len(task_ids)
        for track, task_ids in selection["task_set"].items()
        if task_ids
    }
    assert report["summary"]["sites"] == dict(
        sorted(Counter(compiled[task_id]["site"] for task_id in selected_ids).items())
    )
    assert all(row["structural_pass"] for row in report["tasks"])
    assert all(row["site_realism_ready"] for row in report["tasks"])
    assert all(row["confidence_routing_pending"] for row in report["tasks"])
    assert {row["task_id"] for row in report["tasks"]} == set(selected_ids)
    assert all(
        row["checks"]["supported_claim_frames_resolve"]
        and row["checks"]["scenario_origin_valid"]
        and row["checks"]["natural_user_instruction"]
        for row in report["tasks"]
    )


@pytest.fixture
def preaudit_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    task_id = "preaudit-fixture"
    task = {
        "id": task_id,
        "track": "refresh",
        "site": "egomart",
        "instructions": {"base": "Order one matching cooking utensil."},
        "goal": {
            "acceptable_states": [
                {
                    "selected_entity_ids": ["turner"],
                    "fields": {},
                    "submission_completed": True,
                }
            ]
        },
        "refresh": {"initial_entity_id": "turner"},
    }
    source = {
        "task_id": task_id,
        "status": "video_screened",
        "source": {"media_id": "source-video"},
        "packets": [{"id": "V0", "frame_paths": ["frame.jpg"]}],
        "supported_claims": [
            {
                "text": "A cooking turner is held over the pan.",
                "supported_by": ["V0:0"],
            }
        ],
        "construct": {
            "event_origin": "observed",
            "required_behavior_change": "Keep the matching turner.",
        },
        "unresolved": [],
        "blocking_checks": [],
    }
    payloads = {
        "data/selection.json": {
            "task_set": {"graft": [], "refresh": [task_id], "clarify": []},
            "source_manifests": ["source.json"],
        },
        "source.json": {"tasks": [source]},
        "artifacts/reports/site-audits/index.json": {
            "tasks": {
                task_id: {
                    "task_sha256": task_sha256(task),
                    "audit_path": "site-audit.html",
                }
            }
        },
    }
    for relative_path, payload in payloads.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
    (tmp_path / "frame.jpg").write_bytes(b"source-frame")
    (tmp_path / "product.jpg").write_bytes(b"product-image")
    monkeypatch.setattr(build_task_preaudit, "load_tasks", lambda _: [task])
    monkeypatch.setattr(
        build_task_preaudit,
        "load_site_data",
        lambda *_: {
            "entities": [{"entity_id": "turner", "image_path": "product.jpg"}],
        },
    )
    return tmp_path


@pytest.mark.parametrize(
    ("pending_field", "followup", "expected_decision"),
    [
        (None, [], "release_preaudit_pass"),
        ("unresolved", ["Review details."], "release_preaudit_pass_with_followups"),
        (
            "blocking_checks",
            ["Review details."],
            "release_preaudit_pass_with_followups",
        ),
        (
            "unresolved",
            [
                {
                    "question": "Does this item fit?",
                    "source_question_references": ["source#/evidence"],
                    "resolution": "human_review",
                }
            ],
            "release_preaudit_pass_with_followups",
        ),
        (
            "blocking_checks",
            [{"reason": "missing_public_detail", "reference": "source#/details"}],
            "release_preaudit_pass_with_followups",
        ),
    ],
)
def test_task_preaudit_preserves_source_followups(
    preaudit_workspace: Path,
    pending_field: str | None,
    followup: list,
    expected_decision: str,
) -> None:
    source_path = preaudit_workspace / "source.json"
    source_payload = json.loads(source_path.read_text())
    source = source_payload["tasks"][0]
    if pending_field is not None:
        source[pending_field] = followup
    source_path.write_text(json.dumps(source_payload))

    report = build_preaudit(preaudit_workspace)
    row = report["tasks"][0]

    assert {
        "decision": row["decision"],
        "structural_pass": row["structural_pass"],
        "site_realism_ready": row["site_realism_ready"],
        "unresolved": row["unresolved"],
        "blocking_checks": row["blocking_checks"],
        "summary": report["summary"]["decisions"],
        "admission_authority": report["admission_authority"],
    } == {
        "decision": expected_decision,
        "structural_pass": True,
        "site_realism_ready": True,
        "unresolved": source["unresolved"],
        "blocking_checks": source["blocking_checks"],
        "summary": {expected_decision: 1},
        "admission_authority": False,
    }
