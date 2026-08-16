from __future__ import annotations

import json
from pathlib import Path

from egocua.reviews import audit_construction_reviews
from tests.helpers import materialize_assets, products, task


def _workspace(root: Path) -> dict:
    benchmark_task = task("refresh")
    materialize_assets(benchmark_task, products("refresh"), root)
    return benchmark_task


def test_missing_reviews_are_pending(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    for path in (tmp_path / "data/reviews/submissions" / benchmark_task["id"]).glob("*.json"):
        path.unlink()

    result = audit_construction_reviews(root=tmp_path)

    assert result["counts"] == {"pending": 1, "admit": 0, "reject": 0, "invalid": 0}
    assert result["tasks"][benchmark_task["id"]]["reason"] == "missing_independent_reviews"


def test_two_current_independent_reviews_admit(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)

    result = audit_construction_reviews(root=tmp_path)

    assert result["tasks"][benchmark_task["id"]]["status"] == "admit"
    assert result["all_admitted"] is True


def test_one_failed_check_rejects_the_task(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    path = tmp_path / "data/reviews/submissions" / benchmark_task["id"] / "annotator-2.json"
    submission = json.loads(path.read_text())
    submission["decision"] = "reject"
    submission["note"] = "The evidence is insufficient."
    submission["unresolved"] = ["The evidence is insufficient."]
    path.write_text(json.dumps(submission))

    result = audit_construction_reviews(root=tmp_path)

    assert result["tasks"][benchmark_task["id"]]["status"] == "reject"


def test_site_change_invalidates_existing_reviews(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    site_path = tmp_path / "sites/egomart/products.json"
    site = json.loads(site_path.read_text())
    site["products"][0]["name"] = "Changed after review"
    site_path.write_text(json.dumps(site))

    result = audit_construction_reviews(root=tmp_path)

    assert result["tasks"][benchmark_task["id"]]["status"] == "invalid"
    assert "reviewed artifacts changed" in " ".join(
        result["tasks"][benchmark_task["id"]]["errors"]
    )


def test_spec_change_invalidates_existing_reviews(tmp_path: Path) -> None:
    benchmark_task = _workspace(tmp_path)
    spec_path = tmp_path / "data/reviews/construction.json"
    spec = json.loads(spec_path.read_text())
    spec["checks"]["common"][0]["question"] = "Changed question?"
    spec_path.write_text(json.dumps(spec))

    result = audit_construction_reviews(root=tmp_path)

    assert result["tasks"][benchmark_task["id"]]["status"] == "invalid"
    assert "reviewed artifacts changed" in " ".join(
        result["tasks"][benchmark_task["id"]]["errors"]
    )
