"""Quality review uses frozen materials, not an independent answer-choice score."""

import json
import re
from concurrent.futures import ThreadPoolExecutor

import pytest
from flask import Flask

from egocua.sample_review import SampleReviewWorkspace
from egocua.sample_review_web import register_sample_review
from tests.test_answer_validation import workspace  # noqa: F401


@pytest.fixture
def quality(workspace):  # noqa: F811
    review = SampleReviewWorkspace(workspace.root, "reviewer-a")
    review.materials = workspace.materials
    return review


@pytest.fixture
def app(quality):
    app = Flask("egocua.review_web")
    app.add_url_rule("/review", "review_index", lambda: "original review")
    app.add_url_rule("/review/update-count", "update_count_index", lambda: "updates")
    register_sample_review(app, quality)
    return app


def test_sample_exposes_answers_and_all_materials_without_asking_for_choices(
    app, quality
):
    client = app.test_client()
    index = client.get("/review/sample")
    assert index.status_code == 200
    page = client.get("/review/sample/1")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    assert "The target variant." in html
    assert "预设答案" in html and "合法结果" in html
    assert 'name="choices"' not in html
    assert "通过" in html and "有问题" in html and "无法判断" in html
    assert client.get("/review/sample/1/frame/0/0").mimetype == "image/jpeg"
    assert client.get("/review/sample/1/asset/0").status_code == 200
    assert client.get("/review/sample/1/frame/999/0").status_code == 404
    assert client.get("/review/sample/1/asset/999").status_code == 404
    assert not quality.output.exists()


def test_submit_resume_and_export_keep_quality_separate_from_model_results(
    app, quality
):
    client = app.test_client()
    html = client.get("/review/sample/1").get_data(as_text=True)
    token = re.search(r'name="token" value="([^"]+)"', html)[1]
    assert client.post("/review/sample/1", data={"decision": "pass"}).status_code == 403
    result = client.post("/review/sample/1", data={"token": token, "decision": "pass"})
    assert result.status_code == 302 and result.location.endswith("/sample/2")
    record = quality.record(1)
    assert record["decision"] == "pass"
    assert record["source"] == quality.materials.case(1).source
    assert not (quality.root / "experiments").exists()
    assert not (quality.root / "data/reviews/task-qualification.json").exists()
    reopened = SampleReviewWorkspace(quality.root, quality.reviewer)
    reopened.materials = quality.materials
    assert reopened.record(1) == record
    assert client.get("/review/sample/export").get_json()["records"] == [record]
    assert "已保存：通过" in client.get("/review/sample/1").get_data(as_text=True)
    with pytest.raises(ValueError, match="不能覆盖"):
        quality.submit(1, "problem", "must not replace")


@pytest.mark.parametrize("decision", ["problem", "uncertain"])
def test_issues_need_a_reason_but_are_saved_without_forcing_failure(quality, decision):
    with pytest.raises(ValueError, match="原因"):
        quality.submit(1, decision, "  ")
    quality.submit(1, decision, "The evidence is unclear")
    assert quality.record(1)["decision"] == decision


def test_only_registered_items_are_opened_and_old_material_mismatch_fails(quality):
    with pytest.raises(ValueError, match="登记清单"):
        quality.materials.case(3)
    quality.submit(1, "pass", "")
    path = quality.output / "01.json"
    record = json.loads(path.read_text())
    record["source"] = {}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="材料不同"):
        quality.record(1)


def test_concurrent_submissions_keep_one_first_judgment(quality):
    second = SampleReviewWorkspace(quality.root, quality.reviewer)
    second.materials = quality.materials

    def save(owner):
        try:
            owner.submit(1, "pass", "")
            return "saved"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(save, [quality, second])) == ["rejected", "saved"]


def test_error_preserves_written_note_and_exports_no_fake_records(app):
    client = app.test_client()
    html = client.get("/review/sample/1").get_data(as_text=True)
    token = re.search(r'name="token" value="([^"]+)"', html)[1]
    response = client.post(
        "/review/sample/1", data={"token": token, "note": "check this"}
    )
    assert response.status_code == 400
    assert "check this" in response.get_data(as_text=True)
    assert client.get("/review/sample/export").get_json()["records"] == []
