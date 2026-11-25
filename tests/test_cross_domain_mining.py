from __future__ import annotations

import json
from pathlib import Path

from tools.mine.mine_cross_domain_extensions import mine


def test_cross_domain_mining_preserves_annotations_without_approving_candidates(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks"
    for track in ("graft", "refresh", "clarify"):
        (tasks / track).mkdir(parents=True)
    task = {
        "id": "graft-example",
        "track": "graft",
        "source": {"dossier": "source.json", "task_id": "graft-example"},
        "visual_packets": [{"id": "V0", "frame_paths": ["frame.jpg"]}],
    }
    (tasks / "graft" / "graft-example.json").write_text(json.dumps(task))
    (tmp_path / "source.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "graft-example",
                        "source": {"media_id": "video-1"},
                    }
                ],
            }
        )
    )
    annotation = "Open the matching tutorial and report its duration."
    ego2web = tmp_path / "ego2web.json"
    ego2web.write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "vid": "video-1",
                    "instruction": annotation,
                    "domain": "Media Retrieval",
                    "visual_clues": ["00:00-00:05"],
                },
                {
                    "id": 8,
                    "vid": "video-1",
                    "instruction": "Buy the matching item.",
                    "domain": "E-Commerce",
                },
            ]
        )
    )

    result = mine(tasks, ego2web, root=tmp_path)

    assert result["summary"] == {
        "candidates": 1,
        "source_tasks": 1,
        "by_workflow": {"media": 1},
        "by_track": {"graft": 1},
    }
    candidate = result["candidates"][0]
    assert candidate["status"] == "mined"
    assert candidate["source"]["original_annotation"] == annotation
    assert candidate["lexical_signals"] == ["structured_report", "open_resource"]
    assert candidate["blocking_checks"]
