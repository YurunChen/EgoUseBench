from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from tools.catalog.catalog_clients import yt_search_json
from tools.catalog.capture_youtube_search_windows import OUTPUT_DIR, capture


@pytest.mark.parametrize(
    "returncode,stdout",
    [(1, '{"id":"valid"}'), (0, '{"id":"valid"}\nbroken'), (0, "[]")],
)
def test_search_rejects_partial_or_malformed_output(returncode, stdout):
    result = subprocess.CompletedProcess([], returncode, stdout, "upstream error")
    with patch("tools.catalog.catalog_clients.subprocess.run", return_value=result):
        with pytest.raises(RuntimeError):
            yt_search_json("knife tutorial", limit=40, flat=True)


def test_search_preserves_complete_order():
    result = subprocess.CompletedProcess([], 0, '{"id":"first"}\n\n{"id":"second"}', "")
    with patch("tools.catalog.catalog_clients.subprocess.run", return_value=result):
        assert yt_search_json("knife tutorial", limit=40, flat=True) == [
            {"id": "first"},
            {"id": "second"},
        ]


def test_capture_replaces_unverified_cache_and_reuses_completed_capture(tmp_path):
    sources = tmp_path / "data/sources"
    sources.mkdir(parents=True)
    (sources / "search_query_reviews.json").write_text(
        json.dumps(
            {
                "tasks": {
                    "media-task": {
                        "before_query": "cut vegetables",
                        "after_query": "julienne",
                    },
                }
            }
        )
    )
    output = tmp_path / OUTPUT_DIR
    output.mkdir(parents=True)
    for stage, query in [
        ("before_change", "cut vegetables"),
        ("after_change", "julienne"),
    ]:
        (output / f"media-task--{stage}.json").write_text(
            json.dumps(
                {
                    "provider": "youtube-search-v1",
                    "query": query,
                    "results": [{"id": "old"}] * 4,
                }
            )
        )
    rows = [{"id": f"video00000{i}", "title": f"Tutorial {i}"} for i in range(4)]
    module = "tools.catalog.capture_youtube_search_windows"
    with (
        patch(
            f"{module}.load_tasks",
            return_value=[{"id": "media-task", "site": "egomedia"}],
        ),
        patch(f"{module}.yt_search_json", return_value=rows) as search,
    ):
        assert capture(tmp_path, concurrency=1)["captures"] == 2
        assert capture(tmp_path, concurrency=1)["captures"] == 0
    assert search.call_count == 2
    assert all(
        json.loads(path.read_text())["collection_status"] == "complete"
        for path in output.glob("*.json")
    )
