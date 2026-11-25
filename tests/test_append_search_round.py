from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.catalog import append_search_round as module

TASK_ID = "media-task"


def _record() -> dict:
    return {
        "task_id": TASK_ID,
        "family": "graft",
        "construct": {
            "task_blueprint": {
                "site": "egomedia",
                "environment_role_ids": {"lesson": "m_answer"},
                "equivalent_role_ids": {"lesson": ["m_new_answer"]},
                "final_goal_roles": [["lesson"]],
            },
            "catalog_resolution": {
                "query_rounds": [{"stage": "before_change", "query": "serve"}],
                "window_entity_ids": ["m_answer", "m_old"],
                "core_neighbor_entity_ids": ["m_old"],
                "related_entity_ids": [],
                "ambient_entity_ids": [],
                "rejected": [{"entity_id": "m_old", "reason": "reviewed"}],
            },
        },
    }


def _write(root: Path, relative: str, payload: dict) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _capture(task_id: str, stage: str, query: str):
    rows = [{"youtube_id": vid, "title": vid, "rank": i} for i, vid in enumerate(["yt_old", "yt_new", "yt_ans2"], 1)]
    return task_id, stage, {"query": query, "provider": "youtube-search-v1", "results": rows}


def _run(root: Path, manifest: dict, stage: str = "natural_broad"):
    manifest_path = root / "data/sources/manifests/m.json"
    entities = [
        {"entity_id": vid, "title": vid, "source": {"page_url": f"https://www.youtube.com/watch?v={vid}"}}
        for vid in ("m_answer", "m_old", "m_new", "m_new_answer")
    ]
    with (
        patch.object(module, "iter_manifests", return_value=[(manifest_path, manifest)]),
        patch.object(module, "materialize"),
        patch.object(module, "load_site_data", return_value={"entities": entities}),
        patch.dict(module.CAPTURES, {"egomedia": (Path("captures"), _capture)}),
    ):
        return module.append_round(root, task_id=TASK_ID, stage=stage, query="swing lesson")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write(tmp_path, "data/sources/upstream_entity_index.json", {
        "sites": {"egomedia": {"yt_old": "m_old", "yt_new": "m_new", "yt_ans2": "m_new_answer"}}
    })
    _write(tmp_path, f"data/sources/search_snapshots/{TASK_ID}.json", {"rounds": [{"stage": "before_change"}]})
    return tmp_path


def test_appends_round_and_keeps_reviewed_rings(root: Path) -> None:
    manifest = {"tasks": [_record()]}
    result = _run(root, manifest)

    resolution = manifest["tasks"][0]["construct"]["catalog_resolution"]
    assert [row["stage"] for row in resolution["query_rounds"]] == ["before_change", "natural_broad"]
    assert resolution["window_entity_ids"] == ["m_answer", "m_old", "m_new", "m_new_answer"]
    assert resolution["core_neighbor_entity_ids"] == ["m_old"]
    assert resolution["rejected"] == [{"entity_id": "m_old", "reason": "reviewed"}]
    assert resolution["ambient_entity_ids"] == ["m_new"]
    assert [row["entity_id"] for row in result["pending_judgement"]] == ["m_new"]
    snapshot = json.loads((root / f"data/sources/search_snapshots/{TASK_ID}.json").read_text())
    assert [row["entity_id"] for row in snapshot["rounds"][-1]["results"]] == ["m_old", "m_new", "m_new_answer"]


def test_refuses_an_already_declared_stage(root: Path) -> None:
    with pytest.raises(ValueError, match="already declared"):
        _run(root, {"tasks": [_record()]}, stage="before_change")
