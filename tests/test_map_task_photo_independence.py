from __future__ import annotations

import hashlib
import json
from pathlib import Path

from egocua.schema import referenced_entity_ids, load_tasks, validate_dataset
from egocua.selection import load_selection, task_ids as selected_task_ids

import pytest
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty

pytestmark = pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)


ROOT = Path(__file__).resolve().parents[1]


def _selection_tasks() -> list[dict]:
    selection = load_selection(ROOT)
    task_ids = set(selected_task_ids(selection))
    return [task for task in load_tasks(ROOT / "data/tasks") if task["id"] in task_ids]


@pytest.mark.requires_local_data("sites/egomaps/static/photos")
def test_map_task_referenced_photos_are_globally_unique() -> None:
    places = {
        place["place_id"]: place
        for place in json.loads((ROOT / "sites/egomaps/places.json").read_text())[
            "places"
        ]
    }
    referenced: set[str] = set()
    for task in _selection_tasks():
        if task["site"] != "egomaps":
            continue
        referenced.update(referenced_entity_ids(task))

    digests: dict[str, str] = {}
    for place_id in referenced:
        place = places[place_id]
        digest = hashlib.sha256((ROOT / place["photo_path"]).read_bytes()).hexdigest()
        assert digest not in digests, (
            f"{digests[digest]} and {place_id} share photo bytes"
        )
        digests[digest] = place_id


@pytest.mark.requires_local_data("sites/egomaps/static/photos")
def test_each_maps_task_has_distinct_referenced_photos() -> None:
    places = {
        place["place_id"]: place
        for place in json.loads((ROOT / "sites/egomaps/places.json").read_text())[
            "places"
        ]
    }
    for task in _selection_tasks():
        if task["site"] != "egomaps":
            continue
        digests = {
            hashlib.sha256(
                (ROOT / places[entity_id]["photo_path"]).read_bytes()
            ).hexdigest()
            for entity_id in referenced_entity_ids(task)
        }
        entity_count = len(referenced_entity_ids(task))
        assert len(digests) == entity_count, task["id"]


@pytest.mark.requires_local_data(
    "sites/egomaps/static/photos",
    "sites/egomedia/static/thumbnails",
    "sites/egomart/static/products",
    "data/observations",
)
def test_validate_dataset_accepts_current_map_photo_independence() -> None:
    tasks = _selection_tasks()
    validate_dataset(tasks, root=ROOT, require_assets=True)
