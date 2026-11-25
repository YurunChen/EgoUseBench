import json
from pathlib import Path

from egocua.selection import task_ids, validate_selection

import pytest
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty

pytestmark = pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)


@pytest.mark.requires_local_data("reference/sources/EgoSchema/repo/uid_to_ego4d.json")
def test_current_pilot_selection_is_complete_and_reports_recording_clusters() -> None:
    root = Path(__file__).resolve().parents[1]

    result = validate_selection("data/selection.json", root=root)

    selection = json.loads((root / "data/selection.json").read_text())
    declared = {track: len(ids) for track, ids in selection["task_set"].items()}

    assert result["status"] == "valid"
    assert result["tasks"] == sum(declared.values())
    shared_recording_excess = sum(
        len(group["tasks"]) - 1 for group in result["source_recording_collisions"]
    )
    assert result["source_clusters"] + shared_recording_excess == result["tasks"]
    assert result["source_clips"] == result["tasks"]
    assert result["track_counts"] == declared


def test_source_surface_mismatches_are_rejected_before_construction() -> None:
    """Records whose source surface does not match the constructed task stay rejected.

    candidate-clarify-sequence-ego2web-03 was on this list and is not any more: it was
    rejected for shopping against a tutorial-search annotation, which stopped being a
    rejection basis once counterfactual recombination became the corpus norm. It is now
    candidate-mart-clarify-sequence-07.
    """

    root = Path(__file__).resolve().parents[1]
    rejected = {
        "candidate-refresh-replace-hdepic-10",
        "candidate-refresh-retain-ego2web-05",
        "candidate-refresh-retain-ego2web-06",
        "candidate-refresh-retain-ego2web-08",
        "candidate-refresh-retain-ego2web-09",
        "graft-p-12",
        "media-graft-c-07",
        "seed-refresh-replace-hdepic-03",
    }
    selection = json.loads((root / "data/selection.json").read_text())
    selected = set(task_ids(selection))
    assert rejected.isdisjoint(selected)

    records = {}
    for path in (root / "data/sources/manifests").glob("*screening_manifest.json"):
        payload = json.loads(path.read_text())
        records.update(
            {
                record["task_id"]: record
                for record in payload.get("tasks", [])
                if record.get("task_id") in rejected
            }
        )
    assert set(records) == rejected
    assert all(record["status"] == "rejected" for record in records.values())
    assert all(record.get("rejection_reason") for record in records.values())
