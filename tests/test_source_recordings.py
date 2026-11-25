import hashlib
import json
from pathlib import Path

import pytest

from egocua.lineage import (
    EGO4D_PARENT_MAPPING,
    LineageError,
    build_source_identity,
    discover_frame_provenance,
    summarize_source_recordings,
    task_recording_group,
    validate_independent_recordings,
)


def test_source_identity_uses_paths_not_mutable_batch_digests(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "frames.json").write_text("{}")

    def unexpected_digest(path):
        pytest.fail("source references must not hash shared manifests")

    monkeypatch.setattr("egocua.lineage.file_sha256", unexpected_digest)
    assert build_source_identity(
        {"task_id": "candidate"},
        dossier_reference="manifest.json",
        provenance_reference="frames.json",
        root=tmp_path,
    ) == {
        "dossier": "manifest.json",
        "task_id": "candidate",
        "frame_provenance": "frames.json",
    }


def test_experiment_manifest_finds_its_declared_local_provenance(
    tmp_path: Path,
) -> None:
    reference = "data/experiments/variants/span/sample/screening_manifest.json"
    manifest = tmp_path / reference
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"tasks": []}))
    sidecar = manifest.parent / "frame_provenance.json"
    sidecar.write_text(json.dumps({"manifest": reference}))

    assert discover_frame_provenance(reference, root=tmp_path) == (
        sidecar.relative_to(tmp_path).as_posix(),
        sidecar,
    )


def dossier(tmp_path: Path, dataset: str) -> Path:
    path = tmp_path / f"{dataset}.json"
    path.write_text(json.dumps({"dataset": {"name": dataset}}))
    return path


def record(task_id: str, media_id: str) -> dict:
    return {"task_id": task_id, "source": {"media_id": media_id}}


@pytest.fixture
def parent_mapping(tmp_path: Path) -> Path:
    path = tmp_path / EGO4D_PARENT_MAPPING
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "clip-a": {"video_uid": "parent", "starting_sec": 0, "ending_sec": 180},
                "clip-b": {
                    "video_uid": "parent",
                    "starting_sec": 900,
                    "ending_sec": 1080,
                },
            }
        )
    )
    return path


def test_nonoverlapping_clips_share_one_original_recording(
    tmp_path: Path,
    parent_mapping: Path,
) -> None:
    manifest = dossier(tmp_path, "Ego2Web")
    records = [(manifest, record("a", "clip-a")), (manifest, record("b", "clip-b"))]

    summary = summarize_source_recordings(records, root=tmp_path)

    assert (summary["source_clusters"], summary["source_clips"]) == (1, 2)


def test_formal_members_reject_different_clips_of_same_parent(tmp_path, parent_mapping):
    manifest = dossier(tmp_path, "Ego2Web")
    entries = [(manifest, record("a", "clip-a")), (manifest, record("b", "clip-b"))]

    with pytest.raises(
        LineageError, match="source cluster Ego4D:parent is shared by a, b"
    ):
        validate_independent_recordings(entries, root=tmp_path)


def test_equal_recording_ids_in_different_datasets_remain_independent(tmp_path):
    entries = [
        (dossier(tmp_path, dataset), record(task_id, "same-name"))
        for dataset, task_id in [("HD-EPIC", "a"), ("EgoLife", "b")]
    ]

    assert (
        validate_independent_recordings(entries, root=tmp_path)["source_clusters"] == 2
    )


def test_collision_retains_clip_intervals_and_mapping_digest(
    tmp_path: Path,
    parent_mapping: Path,
) -> None:
    manifest = dossier(tmp_path, "Ego2Web")
    records = [(manifest, record("a", "clip-a")), (manifest, record("b", "clip-b"))]

    summary = summarize_source_recordings(records, root=tmp_path)

    collision = summary["source_recording_collisions"][0]
    assert collision["recording_id"] == "parent"
    assert [row["clip_interval_seconds"] for row in collision["tasks"]] == [
        [0, 180],
        [900, 1080],
    ]
    assert summary["source_recording_mappings"] == [
        {
            "path": EGO4D_PARENT_MAPPING,
            "sha256": hashlib.sha256(parent_mapping.read_bytes()).hexdigest(),
        }
    ]


def test_clip_labeled_original_does_not_override_canonical_parent(
    tmp_path: Path,
    parent_mapping: Path,
) -> None:
    manifest = dossier(tmp_path, "Ego2Web")
    first, second = record("a", "clip-a"), record("b", "clip-b")
    first["source"]["original_video_id"] = "clip-a"
    second["source"]["original_video_id"] = "clip-b"

    summary = summarize_source_recordings(
        [(manifest, first), (manifest, second)], root=tmp_path
    )

    assert summary["source_clusters"] == 1


def test_egoschema_and_ego2web_share_parent_namespace(
    tmp_path: Path,
    parent_mapping: Path,
) -> None:
    records = [
        (dossier(tmp_path, "Ego2Web"), record("a", "clip-a")),
        (dossier(tmp_path, "EgoSchema"), record("b", "clip-b")),
    ]

    assert summarize_source_recordings(records, root=tmp_path)["source_clusters"] == 1


def test_same_recording_text_in_unrelated_datasets_is_not_a_collision(
    tmp_path: Path,
) -> None:
    records = [
        (dossier(tmp_path, "HD-EPIC"), record("a", "001")),
        (dossier(tmp_path, "CaptainCook4D"), record("b", "001")),
    ]

    assert summarize_source_recordings(records, root=tmp_path)["source_clusters"] == 2


def test_non_ego_source_preserves_declared_original_recording(tmp_path: Path) -> None:
    manifest = dossier(tmp_path, "HD-EPIC")
    first, second = record("a", "clip-a"), record("b", "clip-b")
    for row in (first, second):
        row["source"]["original_video_id"] = "original"

    summary = summarize_source_recordings(
        [(manifest, first), (manifest, second)], root=tmp_path
    )

    assert summary["source_clusters"] == 1


def test_recording_group_does_not_depend_on_task_wording_digest(tmp_path: Path) -> None:
    manifest = dossier(tmp_path, "HD-EPIC")
    payload = json.loads(manifest.read_text())
    payload["tasks"] = [record("a", "original-video")]
    manifest.write_text(json.dumps(payload))
    task = {
        "id": "paired-a",
        "source": {
            "dossier": manifest.name,
            "task_id": "a",
        },
    }

    assert task_recording_group(task, root=tmp_path) == "HD-EPIC:original-video"


def test_missing_parent_mapping_cannot_silently_count_clip_as_recording(
    tmp_path: Path,
) -> None:
    manifest = dossier(tmp_path, "Ego2Web")

    with pytest.raises(LineageError, match="uid_to_ego4d.json: cannot read JSON"):
        summarize_source_recordings([(manifest, record("a", "clip-a"))], root=tmp_path)


def test_unmapped_clip_cannot_silently_count_as_recording(
    tmp_path: Path,
    parent_mapping: Path,
) -> None:
    manifest = dossier(tmp_path, "Ego2Web")

    with pytest.raises(LineageError, match="no Ego4D parent mapping"):
        summarize_source_recordings([(manifest, record("a", "unknown"))], root=tmp_path)


@pytest.mark.parametrize(
    "start,end", [(180, 0), (-1, 180), (0, 0), (False, 180), (0, "180")]
)
def test_invalid_parent_intervals_fail(
    tmp_path: Path, parent_mapping: Path, start, end
) -> None:
    parent_mapping.write_text(
        json.dumps(
            {
                "clip-a": {
                    "video_uid": "parent",
                    "starting_sec": start,
                    "ending_sec": end,
                },
            }
        )
    )
    manifest = dossier(tmp_path, "Ego2Web")

    with pytest.raises(LineageError, match="invalid Ego4D parent clip interval"):
        summarize_source_recordings([(manifest, record("a", "clip-a"))], root=tmp_path)


def test_duplicate_task_is_not_misreported_as_source_collision(tmp_path: Path) -> None:
    manifest = dossier(tmp_path, "HD-EPIC")
    pair = (manifest, record("a", "clip-a"))

    with pytest.raises(LineageError, match="duplicate task"):
        summarize_source_recordings([pair, pair], root=tmp_path)


def test_empty_corpus_does_not_require_upstream_mapping(tmp_path: Path) -> None:
    assert summarize_source_recordings([], root=tmp_path) == {
        "source_clusters": 0,
        "source_clips": 0,
        "source_recording_collisions": [],
        "source_recording_mappings": [],
    }


@pytest.mark.parametrize(
    "source,error",
    [
        (None, "source must be an object"),
        ({}, "source media_id is missing"),
        ({"media_id": "clip", "dataset": "Ego2Web"}, "datasets disagree"),
    ],
)
def test_invalid_source_identity_fails(tmp_path: Path, source, error: str) -> None:
    manifest = dossier(tmp_path, "HD-EPIC")

    with pytest.raises(LineageError, match=error):
        summarize_source_recordings(
            [(manifest, {"task_id": "a", "source": source})], root=tmp_path
        )


def test_missing_dossier_dataset_cannot_create_unnamespaced_cluster(
    tmp_path: Path,
) -> None:
    manifest = dossier(tmp_path, "HD-EPIC")
    manifest.write_text("{}")

    with pytest.raises(LineageError, match="dataset.name is missing"):
        summarize_source_recordings([(manifest, record("a", "clip"))], root=tmp_path)


def test_missing_parent_id_fails(tmp_path: Path, parent_mapping: Path) -> None:
    manifest = dossier(tmp_path, "Ego2Web")
    parent_mapping.write_text(
        json.dumps({"clip-a": {"starting_sec": 0, "ending_sec": 180}})
    )

    with pytest.raises(LineageError, match="parent video_uid is missing"):
        summarize_source_recordings([(manifest, record("a", "clip-a"))], root=tmp_path)


def test_mapping_revision_is_not_hidden_by_stale_cache(
    tmp_path: Path, parent_mapping: Path
) -> None:
    manifest = dossier(tmp_path, "Ego2Web")
    records = [(manifest, record("a", "clip-a")), (manifest, record("b", "clip-b"))]
    before = summarize_source_recordings(records, root=tmp_path)
    mapping = json.loads(parent_mapping.read_text())
    mapping["clip-b"]["video_uid"] = "another-parent"
    parent_mapping.write_text(json.dumps(mapping))

    after = summarize_source_recordings(records, root=tmp_path)

    assert (before["source_clusters"], after["source_clusters"]) == (1, 2)
    assert before["source_recording_mappings"] != after["source_recording_mappings"]
