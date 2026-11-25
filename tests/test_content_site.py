from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest
from flask.testing import FlaskClient

from egocua.environment import (
    EnvironmentError,
    create_site_runtime,
    load_site_data,
    public_site_data,
    validate_referenced_site_assets,
)
from egocua.run_ledger import (
    canonical_sha256,
    site_execution_sha256,
    task_site_execution_sha256,
)
from egocua.schema import (
    ValidationError,
    load_tasks,
    referenced_entity_ids,
    validate_task,
)
from tests.helpers import (
    ARCHIVED_CORPUS,
    item_id,
    live_corpus_empty,
    task,
)
from sites.content import create_content_app
from sites.egomedia.app import CONFIG as MEDIA_CONFIG
from sites.egomaps.app import CONFIG as MAPS_CONFIG
from sites.egomart.app import create_app as create_mart_app
from sites.egomart import pages as mart_pages
from sites.runtime import SiteState
from sites.content_chrome import media_search_page

ROOT = Path(__file__).resolve().parents[1]


def test_media_search_card_limits_description_without_hiding_title() -> None:
    entity = {"entity_id": "video", "title": "Video", "description": "A" * 200}
    page = media_search_page(MEDIA_CONFIG, "video", [entity])
    assert "A" * 157 + "..." in page
    assert "A" * 200 not in page


TRIMMER_MEDIA_IDS = (
    "m_5e07554f34a3",
    "m_b93721540bd2",
    "m_af2c1497cd8b",
)


@pytest.mark.parametrize("config", [MEDIA_CONFIG, MAPS_CONFIG], ids=["media", "maps"])
def test_repeated_save_keeps_a_single_selected_entity(config) -> None:
    state = SiteState(
        {
            "site": config.site,
            "entities": [{"entity_id": "saved-one", "title": "Public item"}],
        }
    )
    client = create_content_app(state, config).test_client()

    for _ in range(2):
        assert (
            client.post("/select", data={"entity_id": "saved-one"}).status_code == 200
        )
    assert client.post("/submit").status_code == 200

    assert state.final_state()["selected_entity_ids"] == ["saved-one"]


def test_cart_repeated_add_and_quantity_keep_multiple_units() -> None:
    state = SiteState(
        {
            "entities": [
                {
                    "entity_id": "cart-one",
                    "item_id": "cart-one",
                    "name": "Public product",
                    "price": "$1",
                }
            ]
        }
    )
    client = create_mart_app(state).test_client()

    client.post("/cart/add", data={"item_id": "cart-one", "quantity": "2"})
    client.post("/cart/add", data={"item_id": "cart-one"})
    client.post("/submit")

    assert state.final_state()["selected_entity_ids"] == ["cart-one"] * 3


@pytest.mark.parametrize("endpoint", ["/cart/add", "/cart/update"])
@pytest.mark.parametrize(
    "raw_quantity",
    ["²", pytest.param("9" * 5000, id="exceeds_integer_limit"), "", "-1", "1.5", "١"],
)
def test_cart_invalid_quantity_is_rejected_without_changing_state(
    endpoint, raw_quantity
) -> None:
    state = SiteState(
        {
            "entities": [
                {
                    "entity_id": "one",
                    "item_id": "one",
                    "name": "Public product",
                    "price": "$1",
                }
            ]
        },
        selected_entity_ids=["one"] if endpoint == "/cart/update" else [],
    )
    client = create_mart_app(state).test_client()
    original = list(state.live_state()["selected_entity_ids"])

    response = client.post(endpoint, data={"item_id": "one", "quantity": raw_quantity})

    assert response.status_code == 400
    assert state.live_state()["selected_entity_ids"] == original
    assert state.actions == []


@pytest.mark.parametrize("quantity", [0, 1, 3, 6, 7])
def test_cart_quantity_update_preserves_order_and_logs_state(quantity) -> None:
    entities = [
        {
            "entity_id": entity_id,
            "item_id": entity_id,
            "name": entity_id,
            "price": "$1",
        }
        for entity_id in ("first", "second", "third")
    ]
    original = ["first", "second", "third", "second"]
    state = SiteState({"entities": entities}, selected_entity_ids=list(original))
    client = create_mart_app(state).test_client()

    response = client.post(
        "/cart/update", data={"item_id": "second", "quantity": str(quantity)}
    )

    expected = ["first", *(["second"] * quantity), "third"]
    assert response.status_code == 200
    assert state.live_state()["selected_entity_ids"] == expected
    action = state.actions[-1]
    assert action["type"] == ("deselect_entity" if quantity == 0 else "select_entity")
    assert action["state_before"]["selected_entity_ids"] == original
    assert action["state_after"]["selected_entity_ids"] == expected


@pytest.mark.parametrize("raw_page", ["²", "9" * 5000])
def test_cart_search_invalid_page_uses_first_page(raw_page) -> None:
    client = create_mart_app(SiteState({"entities": []})).test_client()

    response = client.get("/search", query_string={"page": raw_page})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "selected,counts",
    [
        ([], []),
        (["second", "first", "second"], [("second", 2), ("first", 1)]),
        (["unknown", "first", "unknown", "first"], [("first", 2)]),
    ],
)
def test_cart_groups_quantity_in_first_selection_order(selected, counts) -> None:
    products = [
        {"entity_id": item_id, "item_id": item_id, "name": name, "price": "$1"}
        for item_id, name in [("first", "First product"), ("second", "Second product")]
    ]
    state = SiteState({"entities": products}, selected_entity_ids=selected)
    client = create_mart_app(state).test_client()
    by_id = {product["item_id"]: product for product in products}
    expected = mart_pages.shell(
        mart_pages.cart_page([(by_id[item_id], count) for item_id, count in counts]),
        cart_count=len(selected),
        skin="cart-page",
    )

    assert client.get("/cart").get_data(as_text=True) == expected


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("/select", {"entity_id": "one"}),
        ("/deselect", {"entity_id": "one"}),
        ("/field", {"name": "notes", "value": "checked"}),
    ],
)
def test_media_state_operations_render_the_current_selection(endpoint, payload) -> None:
    state = SiteState(
        {
            "entities": [{"entity_id": "one", "title": "Public tutorial"}],
            "supported_fields": ["notes"],
        },
        selected_entity_ids=["one"],
    )
    client = create_content_app(state, MEDIA_CONFIG).test_client()

    response = client.post(endpoint, data=payload)

    assert response.data == client.get("/selection").data


@pytest.mark.parametrize("config", [MEDIA_CONFIG, MAPS_CONFIG], ids=["media", "maps"])
@pytest.mark.parametrize("supported_fields", [["notes"], ["notes", "route"]])
def test_saved_field_remains_editable_after_an_incorrect_value(
    config, supported_fields
) -> None:
    state = SiteState(
        {
            "site": config.site,
            "entities": [],
            "supported_fields": supported_fields,
        }
    )
    client = create_content_app(state, config).test_client()

    incorrect = client.post("/field", data={"name": "notes", "value": "wrong"})
    page = incorrect.get_data(as_text=True)

    assert incorrect.status_code == 200
    assert 'action="/field"' in page
    assert '<option value="notes">' in page
    assert "<b>notes</b>: wrong" in page

    corrected = client.post("/field", data={"name": "notes", "value": "correct"})

    assert corrected.status_code == 200
    assert state.live_state()["fields"] == {"notes": "correct"}
    assert "<b>notes</b>: correct" in corrected.get_data(as_text=True)
    assert corrected.data == client.get("/selection").data


def _preview_entity() -> dict[str, Any]:
    return {
        "entity_id": "preview-video",
        "title": "A public tutorial",
        "source": {"page_url": "https://www.youtube.com/watch?v=preview-id"},
        "public_previews": [
            {
                "image_path": "preview.jpg",
                "heading": "Public storyboard <original>",
                "source_url": "https://i.ytimg.com/sb/preview-id/storyboard3_L2/M0.jpg?a=1&b=2",
                "start_sec": 0,
                "end_sec": 30,
                "provenance": {
                    "kind": "youtube_storyboard",
                    "sha256": "private-audit-hash",
                },
                "task_role": "private-answer-role",
            }
        ],
    }


def _preview_client(root: Path, entity: dict[str, Any]) -> FlaskClient:
    data = public_site_data(
        "egomedia",
        {
            "site": "egomedia",
            "entities": [
                entity,
                {"entity_id": "other-video", "title": "Other tutorial"},
            ],
        },
    )
    return create_content_app(
        SiteState(data, asset_root=root), MEDIA_CONFIG
    ).test_client()


def test_media_public_preview_serves_declared_original_image_bytes(
    tmp_path: Path,
) -> None:
    original = b"\xff\xd8\xff\xe0original storyboard bytes"
    (tmp_path / "preview.jpg").write_bytes(original)

    response = _preview_client(tmp_path, _preview_entity()).get(
        "/public-preview/preview-video/0"
    )

    assert (response.status_code, response.mimetype, response.data) == (
        200,
        "image/jpeg",
        original,
    )


def test_media_public_preview_page_displays_source_and_time_without_provenance(
    tmp_path: Path,
) -> None:
    page = (
        _preview_client(tmp_path, _preview_entity())
        .get("/video/preview-video")
        .get_data(as_text=True)
    )

    assert "Public storyboard &lt;original&gt;" in page
    assert "0:00–0:30" in page
    assert "YouTube storyboard source" in page
    assert '<a href="https://' not in page
    assert 'href="/public-preview/preview-video/0"' in page
    assert "private-audit-hash" not in page and "private-answer-role" not in page


def test_media_player_screenshot_is_labeled_separately_at_its_exact_time(
    tmp_path: Path,
) -> None:
    entity = _preview_entity()
    entity["public_previews"][0].update(
        {
            "source_url": "https://www.youtube.com/watch?v=preview-id",
            "start_sec": 12.5,
            "end_sec": 12.5,
            "provenance": {"kind": "canonical_watch_frame"},
        }
    )

    page = (
        _preview_client(tmp_path, entity)
        .get("/video/preview-video")
        .get_data(as_text=True)
    )

    assert "0:12" in page and "YouTube player screenshot source" in page


@pytest.mark.parametrize(
    "url",
    [
        "/public-preview/missing-video/0",
        "/public-preview/other-video/0",
        "/public-preview/preview-video/1",
        "/public-preview/preview-video/-1",
        "/public-preview/preview-video/0/preview.jpg",
    ],
)
def test_media_public_preview_cannot_address_undeclared_assets(
    tmp_path: Path, url: str
) -> None:
    assert _preview_client(tmp_path, _preview_entity()).get(url).status_code == 404


@pytest.mark.parametrize(
    "path", ["../outside.jpg", "/tmp/outside.jpg", "missing.jpg", "notes.txt"]
)
def test_media_public_preview_rejects_unsafe_or_missing_paths(
    tmp_path: Path, path: str
) -> None:
    (tmp_path / "notes.txt").write_text("private notes")
    entity = _preview_entity()
    entity["public_previews"][0]["image_path"] = path

    assert (
        _preview_client(tmp_path, entity)
        .get("/public-preview/preview-video/0")
        .status_code
        == 404
    )


def test_media_public_preview_rejects_a_symlink_outside_the_asset_root(
    tmp_path: Path,
) -> None:
    root = tmp_path / "assets"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(b"not a public image")
    (root / "preview.jpg").symlink_to(outside)

    assert (
        _preview_client(root, _preview_entity())
        .get("/public-preview/preview-video/0")
        .status_code
        == 404
    )


def test_media_detail_without_previews_retains_identical_page_bytes(
    tmp_path: Path,
) -> None:
    entity = _preview_entity()
    entity.pop("public_previews")
    existing_page = _preview_client(tmp_path, entity).get("/video/preview-video").data
    entity["public_previews"] = []

    assert (
        _preview_client(tmp_path, entity).get("/video/preview-video").data
        == existing_page
    )


def test_media_detail_preserves_evidence_beyond_search_summary(tmp_path: Path) -> None:
    entity = _preview_entity()
    entity["summary"] = "Short search-card summary."
    entity["description"] = (
        "Public context. " * 100 + "Letter and word bonuses <both> apply."
    )

    client = _preview_client(tmp_path, entity)
    detail = client.get("/video/preview-video").get_data(as_text=True)

    assert "Letter and word bonuses &lt;both&gt; apply." in detail
    assert "Short search-card summary." not in detail


def test_media_detail_identifies_static_preview(tmp_path: Path) -> None:
    page = (
        _preview_client(tmp_path, _preview_entity())
        .get("/video/preview-video")
        .get_data(as_text=True)
    )
    assert "playback is not available" in page
    assert 'class="yt-player-chrome"' in page
    assert 'class="play"' not in page


def test_media_detail_stays_inside_frozen_site(tmp_path: Path) -> None:
    entity = _preview_entity()
    entity["video_provenance"] = {"source_url": "https://www.youtube.com/watch?v=preview-id"}
    client = _preview_client(tmp_path, entity)
    detail = client.get("/video/preview-video").get_data(as_text=True)
    assert "playback is not available" in detail
    assert 'class="yt-player-chrome"' in detail
    assert "<iframe" not in detail
    assert "<video" not in detail
    assert "youtube.com" not in detail
    assert client.get("/media/preview-video").status_code == 404


def test_frozen_public_sites_hide_playback_and_page_addresses() -> None:
    """Public records have no player or watch-page field.

    Description prose can quote a link that was already in the frozen source
    text. The page does not turn that prose into a player or a watch link.
    """

    forbidden = {"playback_url", "page_url", "watch_url"}
    for site in ("egomart", "egomedia", "egomaps"):
        public = public_site_data(site, load_site_data(ROOT, site))
        for entity in public["entities"]:
            assert forbidden.isdisjoint(_keys(entity))


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key))
            found.update(_keys(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_keys(child))
    return found


def test_public_media_projection_exposes_no_source_or_playback_address() -> None:
    data = {
        "site": "egomedia",
        "entities": [
            {
                "entity_id": "x",
                "source": {"page_url": "https://www.youtube.com/watch?v=4VBN6L_n-AU"},
                "video_provenance": {"source_url": "https://www.youtube.com/watch?v=4VBN6L_n-AU"},
            }
        ],
    }
    entity = public_site_data("egomedia", data)["entities"][0]
    assert entity == {"entity_id": "x"}


def test_product_details_separate_features_and_specifications() -> None:
    page = mart_pages.product_page(
        {
            "item_id": "one",
            "name": "Product",
            "attributes": {"feature_1": "A useful feature", "Material": "Steel"},
        }
    )
    assert "A useful feature" in page and "Product information" in page
    assert "Feature 1:" not in page and "Steel" in page


def _egomedia_task() -> dict[str, Any]:
    """Reuse current released frames while testing the fixed media site contract."""

    if live_corpus_empty():
        pytest.skip(ARCHIVED_CORPUS)
    task_path = ROOT / "data/tasks/graft/candidate-media-graft-c-79.json"
    benchmark_task = json.loads(task_path.read_text())
    benchmark_task["id"] = "test-egomedia-trimmer"
    benchmark_task["site"] = "egomedia"
    benchmark_task["instructions"] = {
        "a": "Find a tutorial for fitting the cutting head onto this strimmer.",
        "b": "Also make sure it covers winding new line onto the spool.",
        "bundled": (
            "Find a tutorial that covers both fitting the cutting head "
            "and winding on new line."
        ),
    }
    benchmark_task["graft"]["initial_goal"]["acceptable_states"] = [
        {
            "selected_entity_ids": [TRIMMER_MEDIA_IDS[0]],
            "fields": {},
            "submission_completed": True,
        }
    ]
    benchmark_task["goal"]["acceptable_states"] = [
        {
            "selected_entity_ids": list(TRIMMER_MEDIA_IDS),
            "fields": {},
            "submission_completed": True,
        }
    ]
    benchmark_task["truth"]["site_scope_entity_ids"] = list(
        dict.fromkeys(
            [
                *benchmark_task["truth"]["site_scope_entity_ids"],
                *TRIMMER_MEDIA_IDS,
            ]
        )
    )
    benchmark_task["truth"]["tempting_referent_ids"] = []
    # The goals just written add an entity, whatever the task copied from did.
    benchmark_task["graft"]["transition"] = "augment"
    return benchmark_task


@pytest.mark.requires_local_data("sites/egomedia/static/thumbnails/m_5e07554f34a3.jpg")
def test_egomedia_exposes_realistic_visible_metadata_and_thumbnail() -> None:
    benchmark_task = _egomedia_task()
    data = load_site_data(ROOT, "egomedia")
    runtime = create_site_runtime(benchmark_task, data, asset_root=ROOT)

    with runtime.server as server:
        search = (
            urlopen(server.base_url + "/search?q=universal+speed+feed+trimmer+head")
            .read()
            .decode()
        )
        thumbnail = urlopen(server.base_url + "/thumbnail/m_5e07554f34a3")
        hidden_source = (
            urlopen(server.base_url + "/search?q=redistribution+rights").read().decode()
        )

    trimmer = next(
        entity for entity in data["entities"] if entity["entity_id"] == "m_5e07554f34a3"
    )
    assert "Chickanic" in search
    assert "1.7M views" in search
    assert trimmer["view_count"] == 1_796_887
    assert "14:34" in search
    assert "TRIMMER HEAD" in search
    assert "thumbnail/m_5e07554f34a3" in search
    assert thumbnail.status == 200
    assert thumbnail.headers.get_content_type() == "image/jpeg"
    assert "redistribution rights are not asserted" not in hidden_source.casefold()


def test_egomedia_hides_report_fields_when_the_task_only_saves_media() -> None:
    benchmark_task = _egomedia_task()
    data = load_site_data(ROOT, "egomedia")
    runtime = create_site_runtime(benchmark_task, data, asset_root=ROOT)
    payload = urlencode({"entity_id": TRIMMER_MEDIA_IDS[0]}).encode()

    with runtime.server as server:
        page = (
            urlopen(
                Request(
                    server.base_url + "/select",
                    data=payload,
                    method="POST",
                )
            )
            .read()
            .decode()
        )

    assert "Report fields" not in page
    assert "Set field" not in page
    assert "Changes are saved." in page
    assert 'action="/submit"' not in page


def test_egomedia_exposes_only_fields_required_by_the_current_task() -> None:
    benchmark_task = json.loads(
        (ROOT / "data/tasks/clarify/seed-media-clarify-variant-01.json").read_text()
    )
    data = load_site_data(ROOT, "egomedia")
    target_id = benchmark_task["goal"]["acceptable_states"][0]["selected_entity_ids"][0]
    runtime = create_site_runtime(benchmark_task, data, asset_root=ROOT)
    payload = urlencode({"entity_id": target_id}).encode()

    with runtime.server as server:
        page = (
            urlopen(
                Request(
                    server.base_url + "/select",
                    data=payload,
                    method="POST",
                )
            )
            .read()
            .decode()
        )

    assert "animal variant" not in page
    assert "release year" not in page
    assert "covered steps" not in page
    assert "Set field" not in page
    assert "Changes are saved." in page
    assert 'action="/submit"' not in page


def test_site_runtime_contracts_invalidate_runs_when_shared_ui_changes() -> None:
    media_data = load_site_data(ROOT, "egomedia")
    mart_data = load_site_data(ROOT, "egomart")

    assert site_execution_sha256("egomedia", media_data) != canonical_sha256(media_data)
    assert site_execution_sha256("egomart", mart_data) != canonical_sha256(mart_data)


def test_task_site_fingerprint_ignores_unrelated_catalog_entries() -> None:
    benchmark_task = next(
        item for item in load_tasks(ROOT / "data/tasks") if item["site"] == "egomart"
    )
    mart_data = load_site_data(ROOT, "egomart")
    with_unrelated = {
        **mart_data,
        "entities": [
            *mart_data["entities"],
            {
                **mart_data["entities"][0],
                "entity_id": "p_unrelated",
                "item_id": "unrelated",
                "name": "Unrelated catalog item",
            },
        ],
    }

    assert task_site_execution_sha256(
        benchmark_task,
        mart_data,
    ) == task_site_execution_sha256(benchmark_task, with_unrelated)


def test_task_site_fingerprint_changes_with_a_referenced_listing() -> None:
    benchmark_task = next(
        item for item in load_tasks(ROOT / "data/tasks") if item["site"] == "egomart"
    )
    mart_data = load_site_data(ROOT, "egomart")
    referenced = set(referenced_entity_ids(benchmark_task))
    changed = {
        **mart_data,
        "entities": [
            {**entity, "name": entity["name"] + " changed"}
            if entity["entity_id"] in referenced
            else entity
            for entity in mart_data["entities"]
        ],
    }

    assert task_site_execution_sha256(
        benchmark_task,
        mart_data,
    ) != task_site_execution_sha256(benchmark_task, changed)


def test_egomart_ranks_natural_multiword_queries_instead_of_returning_zero() -> None:
    benchmark_task = task("refresh")
    data = load_site_data(ROOT, "egomart")
    runtime = create_site_runtime(benchmark_task, data, asset_root=ROOT)

    with runtime.server as server:
        page = (
            urlopen(server.base_url + "/search?q=sweater+cardigan+pastel+color+block")
            .read()
            .decode()
        )

    assert "p_85d682215a07" in page
    assert "No matching" not in page


def test_egomedia_ranks_natural_multiword_queries_instead_of_returning_zero() -> None:
    data = load_site_data(ROOT, "egomedia")
    client = create_content_app(SiteState(data), MEDIA_CONFIG).test_client()
    query = "/search?q=abstract+acrylic+blending+tutorial+brush+technique"
    first = client.get(query).get_data(as_text=True)
    second = client.get(f"{query}&page=2").get_data(as_text=True)

    assert "m_addb33dff4c3" in first + second
    assert "No matching" not in first


def test_egomedia_search_uses_realistic_paged_result_windows() -> None:
    benchmark_task = _egomedia_task()
    data = load_site_data(ROOT, "egomedia")
    runtime = create_site_runtime(benchmark_task, data, asset_root=ROOT)

    with runtime.server as server:
        first = urlopen(server.base_url + "/search?q=trimmer").read().decode()
        second = urlopen(server.base_url + "/search?q=trimmer&page=2").read().decode()

    assert "Page 1 of" in first
    assert "page=2" in first
    assert "Page 2 of" in second
    assert first != second


def test_content_sites_hide_wearer_context_from_upstream_chrome() -> None:
    media_task = _egomedia_task()
    media_data = load_site_data(ROOT, "egomedia")
    media_runtime = create_site_runtime(media_task, media_data, asset_root=ROOT)
    with media_runtime.server as server:
        search = (
            urlopen(server.base_url + "/search?q=universal+speed+feed+trimmer+head")
            .read()
            .decode()
        )
        with pytest.raises(HTTPError) as missing:
            urlopen(server.base_url + "/context")
        assert missing.value.code == 404

    assert "Wearer" not in search
    assert 'href="/context"' not in search

    maps_task = task("clarify")
    maps_task["site"] = "egomaps"
    maps_data = load_site_data(ROOT, "egomaps")
    maps_runtime = create_site_runtime(maps_task, maps_data, asset_root=ROOT)
    with maps_runtime.server as server:
        search = urlopen(server.base_url + "/search?q=winery").read().decode()
        with pytest.raises(HTTPError) as missing:
            urlopen(server.base_url + "/context")
        assert missing.value.code == 404

    assert "Wearer" not in search
    assert 'href="/context"' not in search


def test_mock_runtimes_store_only_public_domain_data() -> None:
    maps_task = task("clarify")
    maps_task["site"] = "egomaps"
    mart_task = task("clarify")
    mart_task["site"] = "egomart"
    cases = [
        (_egomedia_task(), "egomedia"),
        (maps_task, "egomaps"),
        (mart_task, "egomart"),
    ]

    for benchmark_task, site in cases:
        runtime = create_site_runtime(
            benchmark_task,
            load_site_data(ROOT, site),
            asset_root=ROOT,
        )
        assert "content_origin" not in runtime.state.data
        for entity in runtime.state.entities:
            assert "source" not in entity
            assert not any(key.endswith("_provenance") for key in entity)


def test_mock_runtime_rejects_non_native_public_fields() -> None:
    maps_task = task("clarify")
    maps_task["site"] = "egomaps"
    maps_data = copy.deepcopy(load_site_data(ROOT, "egomaps"))
    maps_data["entities"][0]["benchmark_answer"] = True

    with pytest.raises(EnvironmentError, match="non-native public fields"):
        create_site_runtime(maps_task, maps_data, asset_root=ROOT)

    maps_data = copy.deepcopy(load_site_data(ROOT, "egomaps"))
    maps_data["benchmark_answer"] = maps_data["entities"][0]["entity_id"]

    with pytest.raises(EnvironmentError, match="non-native public root fields"):
        create_site_runtime(maps_task, maps_data, asset_root=ROOT)


def test_content_homes_match_upstream_shells() -> None:
    media_task = _egomedia_task()
    media_data = load_site_data(ROOT, "egomedia")
    media_runtime = create_site_runtime(media_task, media_data, asset_root=ROOT)
    with media_runtime.server as server:
        home = urlopen(server.base_url + "/").read().decode()
        search = (
            urlopen(server.base_url + "/search?q=universal+speed+feed+trimmer+head")
            .read()
            .decode()
        )
        watch = urlopen(server.base_url + "/watch/m_5e07554f34a3").read().decode()
    assert "yt-home" in home
    assert "YouTube" in home
    assert "EgoMedia" not in home
    assert "Wearer" not in home
    assert "TRIMMER HEAD" in watch
    assert 'aria-controls="yt-guide"' in home
    assert "yt-guide-open" in home
    assert "m_5e07554f34a3" not in home
    assert "m_5e07554f34a3" in search
    assert "2.4K Comments" in watch
    assert "@rifleman1873" in watch
    assert "head needed to be greased" in watch

    maps_task = task("clarify")
    maps_task["site"] = "egomaps"
    maps_data = load_site_data(ROOT, "egomaps")
    maps_runtime = create_site_runtime(maps_task, maps_data, asset_root=ROOT)
    with maps_runtime.server as server:
        home = urlopen(server.base_url + "/").read().decode()
    assert "maps-home" in home
    assert "Search Google Maps" in home
    assert "EgoMaps" not in home
    assert "Wearer" not in home


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
def test_egomart_home_requires_product_discovery_through_search() -> None:
    benchmark_task = next(
        row for row in load_tasks(ROOT / "data/tasks") if row["site"] == "egomart"
    )
    data = load_site_data(ROOT, "egomart")
    entities = {row["entity_id"]: row for row in data["entities"]}
    target_id = referenced_entity_ids(benchmark_task)[0]
    target = entities[target_id]
    runtime = create_site_runtime(benchmark_task, data, asset_root=ROOT)

    with runtime.server as server:
        home = urlopen(server.base_url + "/").read().decode()
        search = (
            urlopen(server.base_url + "/search?" + urlencode({"q": target["name"]}))
            .read()
            .decode()
        )

    assert target["name"] not in home
    assert target_id not in home
    assert target_id in search


def test_maps_blank_and_detail_pages_do_not_expose_the_global_catalog() -> None:
    maps_task = task("clarify")
    maps_task["site"] = "egomaps"
    maps_data = load_site_data(ROOT, "egomaps")
    first, second = maps_data["entities"][:2]
    first_title = first.get("title") or first["name"]
    second_title = second.get("title") or second["name"]
    runtime = create_site_runtime(maps_task, maps_data, asset_root=ROOT)

    with runtime.server as server:
        home = urlopen(server.base_url + "/").read().decode()
        blank = urlopen(server.base_url + "/search").read().decode()
        detail = (
            urlopen(server.base_url + f"/place/{first['entity_id']}").read().decode()
        )

    assert first_title not in home
    assert first_title not in blank
    assert "No results found" in blank
    assert second_title not in detail
    assert f"/place/{second['entity_id']}" not in detail


def test_maps_omit_unsupported_listing_metadata() -> None:
    maps_task = task("clarify")
    maps_task["site"] = "egomaps"
    maps_data = copy.deepcopy(load_site_data(ROOT, "egomaps"))
    entity = maps_data["entities"][0]
    entity.pop("rating", None)
    entity.pop("ratings_count", None)
    entity.pop("hours", None)
    title = entity.get("title") or entity["name"]
    runtime = create_site_runtime(maps_task, maps_data, asset_root=ROOT)

    with runtime.server as server:
        page = (
            urlopen(server.base_url + "/search?q=" + title.replace(" ", "+"))
            .read()
            .decode()
        )

    assert '<p class="maps-rating">' not in page
    assert '<p class="maps-hours">' not in page
    assert entity["address"] in page


def test_maps_save_keeps_the_place_page_and_exposes_saved_feedback() -> None:
    maps_task = task("clarify")
    maps_task["site"] = "egomaps"
    maps_data = load_site_data(ROOT, "egomaps")
    entity = maps_data["entities"][0]
    title = entity.get("title") or entity["name"]
    runtime = create_site_runtime(maps_task, maps_data, asset_root=ROOT)

    payload = urlencode({"entity_id": entity["entity_id"]}).encode()
    with runtime.server as server:
        detail = (
            urlopen(server.base_url + f"/place/{entity['entity_id']}").read().decode()
        )
        page = (
            urlopen(
                Request(
                    server.base_url + "/select",
                    data=payload,
                    method="POST",
                )
            )
            .read()
            .decode()
        )

    assert '<button class="maps-save" type="submit">Save</button>' in detail
    assert f"<h1>{title}</h1>" in page
    assert "Remove from saved" in page
    assert "Saved places</h1>" not in page


@pytest.mark.requires_local_data("data/observations/candidate-media-graft-c-79")
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

    runtime = create_site_runtime(benchmark_task, data, asset_root=ROOT)
    with runtime.server as server:
        search_first = urlopen(server.base_url + "/search?q=trimmer").read().decode()
        search_second = (
            urlopen(server.base_url + "/search?q=trimmer&page=2").read().decode()
        )
        search_third = (
            urlopen(server.base_url + "/search?q=trimmer&page=3").read().decode()
        )
        weed_first = urlopen(server.base_url + "/search?q=Weed+Wacker").read().decode()
        weed_second = (
            urlopen(server.base_url + "/search?q=Weed+Wacker&page=2").read().decode()
        )
    search = search_first + search_second + search_third
    weed = weed_first + weed_second
    assert "m_5e07554f34a3" in search
    assert "m_b93721540bd2" in search
    # Realistic search indexes the public description as well as the title.
    assert "m_af2c1497cd8b" in search
    assert "m_af2c1497cd8b" in weed
    for entity_id in entity_ids:
        assert entities[entity_id]["channel_title"]


@pytest.fixture
def published_media_entity(tmp_path: Path) -> dict[str, Any]:
    asset = tmp_path / "thumbnail.jpg"
    thumbnail = b"fixture thumbnail"
    asset.write_bytes(thumbnail)
    return {
        "entity_id": "video",
        "title": "How to replace a trimmer head",
        "summary": "A practical trimmer maintenance tutorial.",
        "thumbnail_path": asset.name,
        "channel_title": "Garden Tools",
        "published_at": "2024-08-01",
        "duration_seconds": 180,
        "view_count": 150,
        "source": {
            "page_url": "https://www.youtube.com/watch?v=example",
            "thumbnail_url": "https://i.ytimg.com/vi/example/default.jpg",
            "thumbnail_sha256": hashlib.sha256(thumbnail).hexdigest(),
            "metadata_retrieved_at": "2026-11-20",
            "metadata_sha256": "a" * 64,
        },
    }


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("channel_title", "missing channel_title"),
        ("published_at", "missing published_at"),
        ("duration_seconds", "missing usable duration_seconds"),
        ("view_count", "missing usable view_count"),
        ("thumbnail_path", "missing thumbnail_path"),
        ("source.page_url", "canonical YouTube source missing"),
        ("source.metadata_sha256", "upstream metadata provenance missing"),
    ),
)
def test_referenced_egomedia_entity_requires_visible_metadata(
    published_media_entity: dict[str, Any],
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    target = published_media_entity
    validate_referenced_site_assets("egomedia", [target], root=tmp_path)
    if field.startswith("source."):
        target["source"].pop(field.split(".", 1)[1])
    else:
        target.pop(field)

    with pytest.raises(EnvironmentError, match=message):
        validate_referenced_site_assets("egomedia", [target], root=tmp_path)


def test_referenced_egomaps_entities_require_independent_photo_and_address() -> None:
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
            entities.append(
                {
                    "entity_id": entity_id,
                    "name": f"Place {index}",
                    "address": f"{index} Test Street",
                    "location": {"latitude": 31.2 + index, "longitude": 121.4 + index},
                    "photo_path": photo_path,
                    "source": {
                        "page_url": f"https://www.google.com/maps/search/?api=1&query=Place+{index}",
                        "image_url": f"https://example.invalid/place-{index}.jpg",
                        "sha256": hashlib.sha256(f"photo-{index}".encode()).hexdigest(),
                    },
                    "route_options": {"walking": f"{index + 2} min"},
                }
            )
        site_data: dict[str, Any] = {
            "site": "egomaps",
            "supported_fields": [],
            "entities": entities,
        }
        validate_task(benchmark_task, site_data, root=root, require_assets=True)
        runtime = create_site_runtime(
            benchmark_task,
            site_data,
            asset_root=root,
        )
        with runtime.server as server:
            search = urlopen(server.base_url + "/search?q=Place").read().decode()
            photo = urlopen(server.base_url + f"/thumbnail/{candidate_ids[0]}").read()
        assert "maps-location-preview" in search
        assert f"thumbnail/{candidate_ids[0]}" not in search
        assert "31.200000, 121.400000" in search
        assert photo == b"photo-0"

        for field, message in (
            ("photo_path", "missing photo_path"),
            ("source", "independent place source missing"),
            ("address", "missing address"),
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
        address_only = copy.deepcopy(site_data)
        for entity in address_only["entities"]:
            entity.pop("location")
            entity.pop("route_options")
        validate_task(benchmark_task, address_only, root=root, require_assets=True)
        client = create_content_app(SiteState(address_only), MAPS_CONFIG).test_client()
        detail = client.get(f"/place/{candidate_ids[0]}").get_data(as_text=True)
        assert "Exact map pin unavailable." in detail
        assert "Route information unavailable." in detail
        assert "maps-location-preview maps-location-hero" not in detail
        assert client.post("/select", data={"entity_id": candidate_ids[0]}).status_code == 200

        for field, bad_value, message in (
            ("location", {"latitude": 91, "longitude": 1}, "invalid location coordinates"),
            ("route_options", {}, "missing usable route_options"),
        ):
            malformed = copy.deepcopy(site_data)
            malformed["entities"][0][field] = bad_value
            with pytest.raises(ValidationError, match=message):
                validate_task(benchmark_task, malformed, root=root, require_assets=True)


def test_maps_detail_shows_facts_not_displayed_on_search_cards() -> None:
    entity = {
        "entity_id": "hall",
        "title": "Public Hall",
        "facts": {
            "listed_city": "Harbor City",
            "route_options": {"walking": "5 min"},
            "entrance": "north side",
        },
    }
    client = create_content_app(
        SiteState({"site": "egomaps", "entities": [entity]}), MAPS_CONFIG
    ).test_client()
    search = client.get("/search?q=Public+Hall").get_data(as_text=True)
    detail = client.get("/place/hall").get_data(as_text=True)

    assert "Public Hall" in search
    assert "Harbor City" not in search
    assert "north side" not in search
    assert "Harbor City" in detail
    assert "5 min" in detail
    assert "north side" in detail


def test_maps_detail_exposes_frozen_place_photo_and_retains_location_preview() -> None:
    entity = {
        "entity_id": "market",
        "name": "Neighborhood Market",
        "photo_path": "sites/egomaps/static/photos/market.jpg",
        "address": "Discovery Square",
        "location": {"latitude": 22.31, "longitude": 39.10},
    }
    client = create_content_app(
        SiteState({"site": "egomaps", "entities": [entity]}), MAPS_CONFIG
    ).test_client()

    detail = client.get("/place/market").get_data(as_text=True)

    assert '<img class="maps-place-photo" src="/thumbnail/market"' in detail
    assert 'alt="Neighborhood Market — place image"' in detail
    assert "maps-location-preview maps-location-hero" in detail
    assert "22.310000, 39.100000" in detail


def test_media_detail_displays_linked_manufacturer_text_separately():
    entity = {
        "entity_id": "one",
        "title": "Tutorial",
        "description": "Original video description.",
        "details": {
            "linked_manufacturer_instructions": {
                "publisher": "Manufacturer",
                "source_url": "https://manufacturer.example/application",
                "text": "Cover adjacent parts before application.",
            }
        },
    }
    client = create_content_app(
        SiteState({"site": "egomedia", "entities": [entity]}), MEDIA_CONFIG
    ).test_client()

    page = client.get("/video/one").get_data(as_text=True)

    assert "Original video description." in page
    assert "Linked manufacturer instructions" in page
    assert "https://manufacturer.example/application" in page
    assert "Cover adjacent parts before application." in page


@pytest.mark.parametrize("longitude_alias", ["lng", "lon"])
def test_map_coordinates_require_the_current_catalog_fields(longitude_alias):
    from sites.content_chrome import _map_coordinates

    assert _map_coordinates({"location": {"latitude": 31.2, "longitude": 121.4}}) == (
        31.2,
        121.4,
    )
    assert _map_coordinates({"location": {"lat": 31.2, longitude_alias: 121.4}}) is None


@pytest.mark.parametrize('config', [MEDIA_CONFIG, MAPS_CONFIG], ids=['media', 'maps'])
def test_content_fonts_are_local_and_available_offline(config):
    state = SiteState({'site': config.site, 'entities': [
        {'entity_id': 'public-item', 'title': 'Public item'},
    ]})
    client = create_content_app(state, config).test_client()
    page = client.get('/').get_data(as_text=True)
    assert 'fonts.googleapis.com' not in page
    assert 'fonts.gstatic.com' not in page
    assert 'src:url(/fonts/roboto.ttf)' in page
    with client.get('/fonts/roboto.ttf') as font:
        assert font.status_code == 200
        assert font.data == (ROOT / 'sites/fonts/Roboto.ttf').read_bytes()
