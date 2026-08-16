from __future__ import annotations

import json
from pathlib import Path

import pytest

from egocua.reviews import construction_evidence_from_reviews
from tests.helpers import products, materialize_assets, task
from tools.promote_construction_reviews import promote_reviews


def test_admitted_submissions_are_promoted_without_retyping(tmp_path: Path) -> None:
    benchmark_task = task("graft")
    materialize_assets(benchmark_task, products("graft"), tmp_path)
    result = promote_reviews(
        "data/reviews/construction.json",
        "data/tasks",
        "data/validation/construction.json",
        root=str(tmp_path),
    )

    payload = json.loads((tmp_path / "data/validation/construction.json").read_text())
    assert result["promoted"] == 1
    assert payload["tasks"][benchmark_task["id"]] == {
        "status": "constructed",
        "construction": construction_evidence_from_reviews(
            benchmark_task, root=tmp_path
        ),
    }


def test_promotion_output_cannot_escape_the_workspace(tmp_path: Path) -> None:
    benchmark_task = task("graft")
    materialize_assets(benchmark_task, products("graft"), tmp_path)

    with pytest.raises(ValueError, match="escapes the workspace"):
        promote_reviews(
            "data/reviews/construction.json",
            "data/tasks",
            "../outside.json",
            root=str(tmp_path),
        )
