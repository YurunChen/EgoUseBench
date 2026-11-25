from __future__ import annotations

import json
from pathlib import Path

import pytest

from egocua.sources import (
    SourceError,
    iter_records,
    screened_records,
    task_detail_entity_ids,
)


def test_registered_experiment_source_uses_original_manifest(tmp_path: Path) -> None:
    reference = "data/experiments/variants/span/sample/screening_manifest.json"
    manifest = tmp_path / reference
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"tasks": [{"task_id": "near", "status": "video_screened"}]}))
    selection = tmp_path / "data/selection.json"
    selection.write_text(json.dumps({"source_manifests": [reference], "task_set": {}}))

    assert [(path, row["task_id"]) for path, row in screened_records(tmp_path)] == [
        (manifest, "near")
    ]
    assert json.loads(selection.read_text())["task_set"] == {}


def test_registered_staged_manifest_is_not_loaded_twice(tmp_path: Path) -> None:
    reference = "data/sources/manifests/sample_screening_manifest.json"
    manifest = tmp_path / reference
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"tasks": [{"task_id": "sample", "status": "video_screened"}]}))
    (tmp_path / "data/selection.json").write_text(
        json.dumps({"source_manifests": [reference]})
    )

    assert len(screened_records(tmp_path)) == 1


def test_screened_replacement_draft_does_not_remove_existing_member(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "data/sources/manifests/sample_screening_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": "existing", "status": "video_screened"},
                    {
                        "task_id": "draft",
                        "status": "video_screened",
                        "replaces_candidate_id": "existing",
                    },
                ]
            }
        )
    )

    assert [record["task_id"] for _, record in screened_records(tmp_path)] == [
        "existing",
        "draft",
    ]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "manifest must be a JSON object"),
        ({}, "tasks must be a list"),
        ({"tasks": {}}, "tasks must be a list"),
        (
            {"tasks": [{"task_id": "valid", "status": "rejected"}, None]},
            "task records must be objects",
        ),
    ],
)
def test_invalid_source_manifest_is_not_silently_filtered(
    tmp_path: Path, payload: object, error: str
) -> None:
    manifest = tmp_path / "data/sources/manifests/sample_screening_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(payload))

    with pytest.raises(SourceError, match=error):
        list(iter_records(tmp_path, status="video_screened"))


def test_task_detail_entities_use_current_semantics_and_reviewed_core(
    tmp_path: Path,
) -> None:
    dossier = tmp_path / "data" / "sources" / "manifests" / "sample.json"
    dossier.parent.mkdir(parents=True)
    dossier.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "sample-refresh",
                        "construct": {
                            "catalog_resolution": {
                                "core_neighbor_entity_ids": ["core-a", "core-b"],
                                "related_entity_ids": ["related-a"],
                            },
                        },
                    }
                ],
            }
        )
    )
    task = {
        "id": "sample-refresh",
        "track": "refresh",
        "source": {
            "dossier": dossier.relative_to(tmp_path).as_posix(),
            "task_id": "sample-refresh",
        },
        "goal": {
            "acceptable_states": [
                {
                    "selected_entity_ids": ["final"],
                    "fields": {},
                    "submission_completed": True,
                }
            ],
        },
        "refresh": {"initial_entity_id": "initial"},
        "truth": {"tempting_referent_ids": []},
    }

    assert task_detail_entity_ids(task, root=tmp_path) == [
        "final",
        "initial",
        "core-a",
        "core-b",
    ]
    assert task_detail_entity_ids(
        task,
        root=tmp_path,
        include_related=True,
    ) == ["final", "initial", "core-a", "core-b", "related-a"]
