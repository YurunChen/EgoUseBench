from __future__ import annotations

import json
from pathlib import Path

import pytest

from egocua import review_materials
from egocua.review_materials import ReviewMaterials, review_submission_matches_case
from tests.helpers import materialize_assets, products, task


@pytest.fixture
def materials(tmp_path: Path) -> tuple[dict, ReviewMaterials]:
    definition = task("clarify")
    materialize_assets(definition, products("clarify"), tmp_path)
    return definition, ReviewMaterials(tmp_path)


@pytest.mark.parametrize("metadata", ["source", "image_provenance", "tier"])
def test_entity_binding_ignores_nonpublic_metadata(materials, tmp_path, metadata):
    definition, reader = materials
    before = reader.case(definition)
    path = tmp_path / "sites/egomart/products.json"
    catalog = json.loads(path.read_text())
    for entity in catalog["products"]:
        entity[metadata] = (
            "changed-audit-group"
            if metadata == "tier"
            else {"captured_at": "later", "collector": "another-import"}
        )
    path.write_text(json.dumps(catalog))

    after = reader.case(definition)

    assert before["reviewed_artifacts"] == after["reviewed_artifacts"]
    assert review_submission_matches_case(before, after)


def test_entity_binding_tracks_public_content(materials, tmp_path):
    definition, reader = materials
    before = reader.case(definition)
    path = tmp_path / "sites/egomart/products.json"
    catalog = json.loads(path.read_text())
    for entity in catalog["products"]:
        entity["name"] = "Changed visible product name"
    path.write_text(json.dumps(catalog))

    after = reader.case(definition)

    assert not review_submission_matches_case(before, after)


def test_material_binding_tracks_released_frame_bytes(materials, tmp_path):
    definition, reader = materials
    before = reader.case(definition)
    frame = definition["visual_packets"][0]["frame_paths"][0]
    (tmp_path / frame).write_bytes(b"changed released pixels")

    after = reader.case(definition)

    assert before["reviewed_artifacts"][frame] != after["reviewed_artifacts"][frame]
    assert not review_submission_matches_case(before, after)


def test_source_video_uses_registered_identity_without_reading_bytes(
    materials, tmp_path, monkeypatch
):
    definition, reader = materials
    path = tmp_path / definition["source"]["dossier"]
    dossier = json.loads(path.read_text())
    record = dossier["tasks"][0]
    record["source"]["media_sha256"] = "d" * 64
    path.write_text(json.dumps(dossier))
    video = tmp_path / dossier["dataset"]["source_root"] / "fixture-clarify.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"registered source video")
    original_read_bytes = Path.read_bytes

    def reject_video_scan(self):
        assert self != video, "display must not rescan the registered source video"
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", reject_video_scan)

    before = reader.case(definition)
    reference = video.relative_to(tmp_path).as_posix()
    assert before["reviewed_artifacts"][reference] == "d" * 64
    record["source"]["media_sha256"] = "e" * 64
    path.write_text(json.dumps(dossier))
    after = reader.case(definition)

    assert after["reviewed_artifacts"][reference] == "e" * 64
    assert not review_submission_matches_case(before, after)


def test_missing_registered_video_identity_keeps_existing_binding(
    materials, tmp_path, monkeypatch
):
    definition, reader = materials
    path = tmp_path / definition["source"]["dossier"]
    dossier = json.loads(path.read_text())
    video = tmp_path / dossier["dataset"]["source_root"] / "fixture-clarify.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"source without an admitted identity")
    original_hash = review_materials._file_sha256
    reads = []

    def observed_hash(path):
        reads.append(path)
        return original_hash(path)

    monkeypatch.setattr(review_materials, "_file_sha256", observed_hash)

    case = reader.case(definition)

    assert video in reads
    assert video.relative_to(tmp_path).as_posix() in case["reviewed_artifacts"]
