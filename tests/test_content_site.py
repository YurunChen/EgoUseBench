from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
from typing import Any
from urllib.request import urlopen

import pytest

from egocua.environment import create_site_runtime, load_site_data
from egocua.schema import ValidationError, validate_task
from tests.helpers import item_id, task


ROOT = Path(__file__).resolve().parents[1]
TRIMMER_MEDIA_IDS = (
    "m_5e07554f34a3",
    "m_b93721540bd2",
    "m_af2c1497cd8b",
)


def _egomedia_task() -> dict[str, Any]:
    """Reuse current released frames while testing the fixed media site contract."""

    selection = json.loads((ROOT / "data/selection.json").read_text())
    selected_graft_id = selection["primary"]["graft"][0]
    benchmark_task = json.loads(
        (ROOT / f"data/tasks/graft/{selected_graft_id}.json").read_text()
    )
    benchmark_task["id"] = "test-egomedia-trimmer"
    benchmark_task["site"] = "egomedia"
    benchmark_task["instructions"] = {
        "a": "Find a tutorial for installing this grass trimmer head.",
        "b": "Also make sure it shows how to reload the trimmer line.",
        "bundled": (
            "Find a tutorial that shows both grass trimmer head installation "
            "and line reloading."
        ),
    }
    benchmark_task["graft"]["initial_goal"]["acceptable_states"] = [{
        "selected_entity_ids": [TRIMMER_MEDIA_IDS[0]],
        "fields": {},
        "submission_completed": True,
    }]
    benchmark_task["goal"]["acceptable_states"] = [{
        "selected_entity_ids": list(TRIMMER_MEDIA_IDS),
        "fields": {},
        "submission_completed": True,
    }]
    return benchmark_task


def test_egomedia_exposes_realistic_visible_metadata_and_thumbnail() -> None:
    benchmark_task = _egomedia_task()
    data = load_site_data(ROOT, "egomedia")
    runtime = create_site_runtime(
        benchmark_task, "bundled", data, asset_root=ROOT
    )

    with runtime.server as server:
        search = urlopen(
            server.base_url + "/search?q=grass+trimmer+head"
        ).read().decode()
        thumbnail = urlopen(
            server.base_url + "/thumbnail/m_5e07554f34a3"
        )
        hidden_source = urlopen(
            server.base_url + "/search?q=CC+BY"
        ).read().decode()

    assert "Yard Repair Workshop" in search
    assert "184,732 views" in search
    assert "10:12" in search
    assert "thumbnail/m_5e07554f34a3" in search
    assert thumbnail.status == 200
    assert thumbnail.headers.get_content_type() == "image/jpeg"
    assert "No matching records" in hidden_source


def test_complete_egomedia_seed_assets_pass_release_validation() -> None:
    data = load_site_data(ROOT, "egomedia", require_assets=True)
    benchmark_task = _egomedia_task()
    validate_task(
        benchmark_task,
        data,
        root=ROOT,
        require_assets=True,
    )


def test_trimmer_search_results_have_complete_visible_metadata() -> None:
    benchmark_task = _egomedia_task()
    data = load_site_data(ROOT, "egomedia")
    entity_ids = set(TRIMMER_MEDIA_IDS)
    entities = {
        entity["entity_id"]: entity
        for entity in data["entities"]
        if entity["entity_id"] in entity_ids
    }

    assert set(entities) == entity_ids
    for entity in entities.values():
        assert entity["thumbnail_path"]
        assert entity["channel_title"]
        assert entity["published_at"]
        assert entity["duration_seconds"] > 0
        assert entity["view_count"] >= 0

    runtime = create_site_runtime(
        benchmark_task, "bundled", data, asset_root=ROOT
    )
    with runtime.server as server:
        search = urlopen(
            server.base_url + "/search?q=trimmer"
        ).read().decode()
    for entity_id in entity_ids:
        assert entity_id in search


def test_referenced_egomedia_entity_requires_a_real_thumbnail() -> None:
    benchmark_task = _egomedia_task()
    data = copy.deepcopy(load_site_data(ROOT, "egomedia"))
    target = next(
        entity for entity in data["entities"]
        if entity["entity_id"] == "m_5e07554f34a3"
    )
    target.pop("thumbnail_path")

    with pytest.raises(ValidationError, match="missing thumbnail_path"):
        validate_task(benchmark_task, data, root=ROOT, require_assets=True)


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("channel_title", "missing channel_title"),
        ("published_at", "missing published_at"),
        ("duration_seconds", "missing usable duration_seconds"),
        ("view_count", "missing usable view_count"),
    ),
)
def test_referenced_egomedia_entity_requires_visible_metadata(
    field: str, message: str,
) -> None:
    benchmark_task = _egomedia_task()
    data = copy.deepcopy(load_site_data(ROOT, "egomedia"))
    target = next(
        entity for entity in data["entities"]
        if entity["entity_id"] == "m_5e07554f34a3"
    )
    target.pop(field)

    with pytest.raises(ValidationError, match=message):
        validate_task(benchmark_task, data, root=ROOT, require_assets=True)


def test_referenced_egomaps_entities_require_photo_coordinates_and_routes() -> None:
    benchmark_task = task("clarify")
    benchmark_task["site"] = "egomaps"
    candidate_ids = [item_id("clarify", 0), item_id("clarify", 1)]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for packet in benchmark_task["visual_packets"]:
            for frame_path in packet["frame_paths"]:
                path = root / frame_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"frame")
        entities: list[dict[str, Any]] = []
        for index, entity_id in enumerate(candidate_ids):
            photo_path = f"sites/egomaps/static/photos/{entity_id}.jpg"
            photo = root / photo_path
            photo.parent.mkdir(parents=True, exist_ok=True)
            photo.write_bytes(f"photo-{index}".encode())
            entities.append({
                "entity_id": entity_id,
                "name": f"Place {index}",
                "address": f"{index} Test Street",
                "location": {"latitude": 31.2 + index, "longitude": 121.4 + index},
                "photo_path": photo_path,
                "route_options": {"walking": f"{index + 2} min"},
            })
        site_data: dict[str, Any] = {
            "site": "egomaps",
            "supported_fields": [],
            "entities": entities,
        }
        validate_task(benchmark_task, site_data, root=root, require_assets=True)
        runtime = create_site_runtime(
            benchmark_task,
            "interactive",
            site_data,
            asset_root=root,
        )
        with runtime.server as server:
            search = urlopen(server.base_url + "/search?q=Place").read().decode()
            photo = urlopen(
                server.base_url + f"/thumbnail/{candidate_ids[0]}"
            ).read()
        assert f"thumbnail/{candidate_ids[0]}" in search
        assert photo == b"photo-0"

        for field, message in (
            ("photo_path", "missing photo_path"),
            ("location", "missing location coordinates"),
            ("route_options", "missing usable route_options"),
        ):
            incomplete = copy.deepcopy(site_data)
            incomplete["entities"][0].pop(field)
            with pytest.raises(ValidationError, match=message):
                validate_task(
                    benchmark_task,
                    incomplete,
                    root=root,
                    require_assets=True,
                )
