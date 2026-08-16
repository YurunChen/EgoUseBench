import json
from pathlib import Path

from tools.validate_pilot_selection import validate_selection


def test_current_pilot_selection_is_complete_and_source_independent() -> None:
    root = Path(__file__).resolve().parents[1]

    result = validate_selection(
        "data/selection.json", root=root
    )

    assert result["status"] == "valid"
    assert result["primary_tasks"] == 11
    assert result["source_clusters"] == 11
    assert result["track_counts"] == {"clarify": 4, "graft": 1, "refresh": 6}
    assert result["site_counts"] == {"egomaps": 1, "egomart": 10, "egomedia": 0}


def test_source_surface_mismatches_are_rejected_before_construction() -> None:
    root = Path(__file__).resolve().parents[1]
    rejected = {
        "candidate-clarify-sequence-ego2web-03",
        "candidate-refresh-retain-ego2web-09",
        "graft-p-12",
        "media-graft-c-07",
    }
    selection = json.loads((root / "data/selection.json").read_text())
    selected = {
        task_id
        for task_ids in selection["primary"].values()
        for task_id in task_ids
    }
    assert rejected.isdisjoint(selected)

    records = {}
    for path in (root / "data/sources").glob("*screening_manifest.json"):
        payload = json.loads(path.read_text())
        records.update({
            record["task_id"]: record
            for record in payload.get("tasks", [])
            if record.get("task_id") in rejected
        })
    assert set(records) == rejected
    assert all(record["status"] == "rejected" for record in records.values())
    assert all(record.get("rejection_reason") for record in records.values())
