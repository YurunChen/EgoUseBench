from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

import pytest

from egocua.environment import (
    EnvironmentError,
    SITES,
    entity_visual_path,
    site_data_path,
    _validate_media_preview_assets,
    _verified_publisher_description,
    load_site_data,
    public_site_data,
    task_public_site_data,
    site_data_request,
)
from tests.helpers import item_id, materialize_assets, products, task


@pytest.mark.parametrize(
    "site,catalog,image_field",
    [
        ("egomart", "products.json", "image_path"),
        ("egomedia", "videos.json", "thumbnail_path"),
        ("egomaps", "places.json", "photo_path"),
    ],
)
def test_site_registration_drives_data_and_visual_paths(site, catalog, image_field):
    assert site in SITES
    assert site_data_path(site) == f"sites/{site}/{catalog}"
    assert entity_visual_path(site, {image_field: "image.jpg"}) == "image.jpg"


def _write_site(root: Path, site_products: list[dict[str, Any]]) -> None:
    path = root / "sites/egomart/products.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "site": "egomart",
                "products": site_products,
            }
        )
    )


def test_all_tasks_load_the_same_fixed_site_products(tmp_path: Path) -> None:
    site_products = products("graft")
    _write_site(tmp_path, site_products)

    first = load_site_data(tmp_path, "egomart")
    second = load_site_data(tmp_path, "egomart")

    assert [entity["item_id"] for entity in first["entities"]] == [
        product["item_id"] for product in site_products
    ]
    assert second == first


def test_site_data_request_reads_and_validates_once_then_reloads(tmp_path, monkeypatch):
    from egocua import environment

    _write_site(tmp_path, products("graft"))
    calls = []
    original = environment._load_site_data

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(environment, "_load_site_data", counted)
    path = tmp_path / "sites/egomart/products.json"
    with site_data_request(tmp_path):
        first = load_site_data(tmp_path, "egomart")
        with site_data_request(tmp_path):
            assert load_site_data(tmp_path, "egomart") is first
        payload = json.loads(path.read_text())
        payload["products"][0]["name"] = "source observation"
        path.write_text(json.dumps(payload))
        assert load_site_data(tmp_path, "egomart") is first
        assert calls == ["egomart"]
    with (
        site_data_request(tmp_path),
        pytest.raises(EnvironmentError, match="evaluator language"),
    ):
        load_site_data(tmp_path, "egomart")
    assert calls == ["egomart", "egomart"]


def test_site_data_request_preserves_stricter_asset_and_frame_checks(
    tmp_path, monkeypatch
):
    from egocua import environment

    definition = task("clarify")
    materialize_assets(definition, products("clarify"), tmp_path)
    calls = []
    original = environment._load_site_data

    def counted(*args, **kwargs):
        calls.append(kwargs["require_assets"])
        return original(*args, **kwargs)

    monkeypatch.setattr(environment, "_load_site_data", counted)
    with site_data_request(tmp_path):
        load_site_data(tmp_path, "egomart")
        checked = load_site_data(tmp_path, "egomart", require_assets=True)
        assert load_site_data(tmp_path, "egomart", require_assets=True) is checked
        with pytest.raises(EnvironmentError, match="invalid product image"):
            load_site_data(
                tmp_path,
                "egomart",
                require_assets=True,
                released_frame_paths=[products("clarify")[0]["image_path"]],
            )
    assert calls == [False, True, True]


def test_request_validates_public_fields_before_applying_task_scope(tmp_path):
    entries = products("clarify")
    entries[-1]["internal_annotation"] = "not public metadata"
    _write_site(tmp_path, entries)
    definition = task("clarify")
    definition["truth"] = {"site_scope_entity_ids": [item_id("clarify", 0)]}
    with site_data_request(tmp_path):
        catalog = load_site_data(tmp_path, "egomart")
        with pytest.raises(EnvironmentError, match="non-native public fields"):
            task_public_site_data(definition, catalog)


def test_rejects_duplicate_site_product_ids(tmp_path: Path) -> None:
    site_products = products("graft")
    site_products[1]["item_id"] = site_products[0]["item_id"]
    _write_site(tmp_path, site_products)

    with pytest.raises(EnvironmentError, match="duplicate product IDs"):
        load_site_data(tmp_path, "egomart")


@pytest.mark.parametrize("invalid_id", [None, "", 7, [], {}])
def test_rejects_invalid_site_product_ids(tmp_path: Path, invalid_id: Any) -> None:
    site_products = products("graft")
    site_products[0]["item_id"] = invalid_id
    _write_site(tmp_path, site_products)

    with pytest.raises(EnvironmentError, match="invalid or duplicate product IDs"):
        load_site_data(tmp_path, "egomart")


def test_allows_distinct_real_listings_with_the_same_visible_name(
    tmp_path: Path,
) -> None:
    site_products = products("graft")
    site_products[1]["name"] = f"  {site_products[0]['name'].upper()}  "
    _write_site(tmp_path, site_products)

    data = load_site_data(tmp_path, "egomart")

    assert len(data["entities"]) == len(site_products)


def test_rejects_duplicate_site_product_images(tmp_path: Path) -> None:
    site_products = products("graft")
    site_products[1]["image_path"] = site_products[0]["image_path"]
    site_products[1]["image_provenance"]["sha256"] = site_products[0][
        "image_provenance"
    ]["sha256"]
    _write_site(tmp_path, site_products)

    with pytest.raises(EnvironmentError, match="duplicate product image"):
        load_site_data(tmp_path, "egomart")


def test_rejects_missing_site_store(tmp_path: Path) -> None:
    with pytest.raises(EnvironmentError, match="sites/egomart/products.json"):
        load_site_data(tmp_path, "egomart")


def test_rejects_hidden_benchmark_role_in_public_attributes(tmp_path: Path) -> None:
    site_products = products("graft")
    site_products[0]["attributes"]["role in scene"] = "target"
    _write_site(tmp_path, site_products)

    with pytest.raises(EnvironmentError, match="hidden benchmark role"):
        load_site_data(tmp_path, "egomart")


def test_rejects_evaluator_language_in_public_entity_text(tmp_path: Path) -> None:
    site_products = products("graft")
    site_products[0]["description"] = "A search distractor."
    _write_site(tmp_path, site_products)

    with pytest.raises(EnvironmentError, match="evaluator language"):
        load_site_data(tmp_path, "egomart")


@pytest.mark.parametrize(
    "mutation", [None, "description", "snapshot_hash", "source_url"]
)
@pytest.mark.parametrize(
    "publisher_text",
    [
        "Players deduce the correct answer.",
        "A scarecrow is a decoy used to discourage birds.",
        "This travel edition is not the one shown earlier.",
        "Sellers change, so these may not be the correct ones in my video.",
        "Preschool Benchmark: 11.A.ECa.",
    ],
)
def test_publisher_text_exception_requires_exact_bound_evidence(
    tmp_path: Path, mutation: str | None, publisher_text: str
) -> None:
    payload = {
        "provider": "youtube-video-detail-v1",
        "id": "video-one",
        "title": "Clue rules",
        "description": publisher_text,
        "public_summary": "Clue rules.",
    }
    (tmp_path / "detail.json").write_text(json.dumps(payload))
    entity: dict[str, Any] = {
        "title": payload["title"],
        "description": payload["description"],
        "summary": payload["public_summary"],
        "source": {
            "metadata_snapshot_basis": "frozen_youtube_detail",
            "detail_snapshot_path": "detail.json",
            "page_url": "https://www.youtube.com/watch?v=video-one",
            "metadata_sha256": hashlib.sha256(
                json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
        },
    }
    if mutation == "description":
        entity["description"] += " This is the correct answer to the task."
    elif mutation == "snapshot_hash":
        entity["source"]["metadata_sha256"] = "not-current"
    elif mutation == "source_url":
        entity["source"]["page_url"] = "https://www.youtube.com/watch?v=another"
    assert _verified_publisher_description(tmp_path, entity) is (mutation is None)


def test_task_scope_hides_unregistered_catalog_rows() -> None:
    definition = task("graft")
    semantic_ids = [item_id("graft", 0), item_id("graft", 1)]
    neighbor_id = item_id("graft", 2)
    definition["truth"] = {
        "tempting_referent_ids": [semantic_ids[1]],
        "site_scope_entity_ids": [*semantic_ids, neighbor_id],
    }
    site = {
        "schema_version": 1,
        "site": "egomart",
        "entities": [
            {
                "entity_id": item_id("graft", index),
                "item_id": item_id("graft", index),
                "name": f"Product {index}",
                "brand": "Brand",
                "price": "$1.00",
                "attributes": {},
                "image_path": f"products/{index}.jpg",
            }
            for index in range(4)
        ],
    }

    public = task_public_site_data(definition, site)

    assert [entity["entity_id"] for entity in public["entities"]] == [
        *semantic_ids,
        neighbor_id,
    ]


def _media_preview_entity(**changes: Any) -> dict[str, Any]:
    return {
        "entity_id": "media-one",
        "title": "Public tutorial",
        "source": {"page_url": "https://www.youtube.com/watch?v=video-one"},
        "public_previews": [
            {
                "image_path": "previews/one.jpg",
                "heading": "Public video storyboard",
                "source_url": "https://i.ytimg.com/sb/video-one/storyboard3_L2/M0.jpg",
                "start_sec": 0,
                "end_sec": 30,
                "provenance": {"kind": "youtube_storyboard", "sha256": "private hash"},
                **changes,
            }
        ],
    }


def test_public_media_preview_drops_audit_and_unrecognized_fields() -> None:
    entity = _media_preview_entity(task_role="answer", inference="private decision")

    public = public_site_data("egomedia", {"site": "egomedia", "entities": [entity]})

    assert public["entities"][0]["public_previews"] == [
        {
            "image_path": "previews/one.jpg",
            "heading": "Public video storyboard",
            "source_url": "https://i.ytimg.com/sb/video-one/storyboard3_L2/M0.jpg",
            "start_sec": 0,
            "end_sec": 30,
            "kind": "youtube_storyboard",
        }
    ]


def test_public_media_preserves_full_publisher_description() -> None:
    entity = _media_preview_entity()
    entity["summary"] = "Short summary."
    entity["description"] = "Publisher text. " * 100 + "Decisive public rule."

    public = public_site_data("egomedia", {"site": "egomedia", "entities": [entity]})

    assert public["entities"][0]["description"] == entity["description"]
    assert "source" not in public["entities"][0]


@pytest.mark.parametrize(
    "changes",
    [
        {"source_url": "javascript:alert(1)"},
        {"source_url": "https://i.ytimg.com.evil.example/sb/video-one/page.jpg"},
        {"source_url": "https://i.ytimg.com/sb/video-two/page.jpg"},
        {"heading": "The correct answer"},
        {"start_sec": -1},
        {"end_sec": float("nan")},
        {"end_sec": 0},
        {"start_sec": True},
    ],
)
def test_public_media_preview_rejects_invalid_public_metadata(
    changes: dict[str, Any],
) -> None:
    with pytest.raises(EnvironmentError):
        public_site_data(
            "egomedia",
            {"site": "egomedia", "entities": [_media_preview_entity(**changes)]},
        )


def test_public_media_player_screenshot_allows_an_exact_timestamp() -> None:
    entity = _media_preview_entity(
        source_url="https://www.youtube.com/watch?v=video-one&t=12s",
        start_sec=12,
        end_sec=12,
        provenance={"kind": "canonical_watch_frame"},
    )

    public = public_site_data("egomedia", {"site": "egomedia", "entities": [entity]})

    assert (
        public["entities"][0]["public_previews"][0]["kind"] == "canonical_watch_frame"
    )


def test_public_media_preview_asset_rejects_a_hash_mismatch(tmp_path: Path) -> None:
    entity = _media_preview_entity(image_path="preview.jpg")
    (tmp_path / "preview.jpg").write_bytes(b"original image")

    with pytest.raises(EnvironmentError, match="hash mismatch"):
        _validate_media_preview_assets(entity, tmp_path, set())


def test_public_media_preview_asset_cannot_reuse_a_released_observation(
    tmp_path: Path,
) -> None:
    entity = _media_preview_entity(image_path="released.jpg")

    with pytest.raises(EnvironmentError, match="reuses a released observation"):
        _validate_media_preview_assets(entity, tmp_path, {"released.jpg"})


def test_public_media_player_screenshot_validates_original_hash_and_capture_time(
    tmp_path: Path,
) -> None:
    original = b"original public player screenshot"
    (tmp_path / "preview.png").write_bytes(original)
    entity = _media_preview_entity(
        image_path="preview.png",
        source_url="https://www.youtube.com/watch?v=video-one",
        start_sec=12,
        end_sec=12,
        provenance={
            "kind": "canonical_watch_frame",
            "captured_at": "2026-09-05T00:00:00Z",
            "sha256": hashlib.sha256(original).hexdigest(),
        },
    )

    _validate_media_preview_assets(entity, tmp_path, set())
