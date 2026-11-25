from pathlib import Path
import json

import pytest

from egocua.run_ledger import task_site_execution_sha256
from egocua.workspace import write_json_atomic
from tests.helpers import task
from tools.catalog import materialize_upstream_search_windows as materializer
from tools.catalog.materialize_upstream_search_windows import (
    _amazon_capture_rows,
    _capture_rows,
    _review_count,
    _stable_id,
)


def test_amazon_capture_rows_accepts_licensed_abo_directory(tmp_path) -> None:
    write_json_atomic(
        tmp_path / materializer.ABO_CAPTURES / "task-a--before_change.json",
        {
            "task_id": "task-a",
            "provider": "amazon_berkeley_objects",
            "captured_at": "2026-09-05T00:00:00Z",
            "results": [{"asin": "B000000001", "title": "Licensed chair"}],
        },
    )

    rows = _amazon_capture_rows(tmp_path, task_ids={"task-a"})

    assert rows["B000000001"]["provider"] == "amazon_berkeley_objects"
    assert rows["B000000001"]["snapshot_paths"] == [
        (materializer.ABO_CAPTURES / "task-a--before_change.json").as_posix()
    ]


def test_amazon_capture_rows_rejects_cross_provider_collision(tmp_path) -> None:
    for directory, provider in (
        (materializer.AMAZON_CAPTURES, "amazon"),
        (materializer.ABO_CAPTURES, "amazon_berkeley_objects"),
    ):
        write_json_atomic(
            tmp_path / directory / "task-a--before_change.json",
            {
                "task_id": "task-a",
                "provider": provider,
                "results": [{"asin": "B000000001", "title": "Chair"}],
            },
        )

    with pytest.raises(ValueError, match="both live and ABO"):
        _amazon_capture_rows(tmp_path, task_ids={"task-a"})


def test_capture_rows_deduplicates_stable_upstream_id(tmp_path) -> None:
    first = {
        "captured_at": "2026-08-29T00:00:00Z",
        "results": [{"asin": "B012345678", "title": "One", "price": None}],
    }
    second = {
        "captured_at": "2026-08-29T01:00:00Z",
        "results": [{"asin": "B012345678", "title": "One", "price": "$9.99"}],
    }
    (tmp_path / "a.json").write_text(__import__("json").dumps(first))
    (tmp_path / "b.json").write_text(__import__("json").dumps(second))

    rows = _capture_rows(tmp_path, Path("."), "asin")

    assert list(rows) == ["B012345678"]
    assert rows["B012345678"]["price"] == "$9.99"
    assert len(rows["B012345678"]["snapshot_paths"]) == 2


def test_stable_id_and_compact_review_count() -> None:
    assert _stable_id("p", "amazon", "B012345678") == _stable_id(
        "p", "amazon", "B012345678"
    )
    assert _review_count("(5.5K)") == 5_500


@pytest.mark.parametrize("materialization_workspace", ["egomart"], indirect=True)
@pytest.mark.parametrize("detail_market", ["www.amazon.ca", "www.amazon.com.au", None])
def test_product_detail_preserves_marketplace_price_and_rating_ownership(
    materialization_workspace,
    detail_market,
) -> None:
    workspace = materialization_workspace
    root = workspace["root"]
    asin = workspace["new_id"]
    capture_path = (
        root / materializer.AMAZON_CAPTURES / "selected-task--before_change.json"
    )
    capture = json.loads(capture_path.read_text())
    for row in capture["results"]:
        row.update(price="$16.93", rating=3.7, review_count=8)
    write_json_atomic(capture_path, capture)
    detail_url = f"https://{detail_market}/dp/{asin}" if detail_market else None
    write_json_atomic(
        root / materializer.AMAZON_DETAILS / f"{asin}.json",
        {
            "asin": asin,
            "page_url": detail_url,
            "title": "Verified same-model knife",
            "price": "$99.00",
            "product_details": {
                "BladeLength": "3.98 Inches",
                "Customer Reviews": "4.2 out of 5 stars (43)",
                "Customer Reviews ‏": "4.2 out of 5 stars (43)",
                "Best Sellers Rank": "#1 in Australian knives",
            },
        },
    )

    materializer.materialize(root, site="egomart", task_ids={"selected-task"})

    catalog = json.loads((root / materializer.AMAZON_CATALOG).read_text())
    entity = next(
        row
        for row in catalog["products"]
        if row["item_id"] == workspace["new_entity_id"]
    )
    assert entity["price"] == (
        "$99.00" if detail_market == "www.amazon.ca" else "$16.93"
    )
    assert entity["rating"] == 3.7
    assert entity["ratings_count"] == 8
    assert entity["attributes"]["BladeLength"] == "3.98 Inches"
    assert not any("Reviews" in key or "Rank" in key for key in entity["attributes"])
    assert entity["text_provenance"].get("detail_source_url") == detail_url


@pytest.mark.parametrize("materialization_workspace", ["egomart"], indirect=True)
@pytest.mark.parametrize(
    "bad_title", ["Amazon.ca", "Amazon.ca: Home & Kitchen", "Robot Check"]
)
def test_generic_detail_title_does_not_replace_search_title(
    materialization_workspace,
    bad_title,
) -> None:
    workspace = materialization_workspace
    root = workspace["root"]
    asin = workspace["new_id"]
    write_json_atomic(
        root / materializer.AMAZON_DETAILS / f"{asin}.json",
        {
            "asin": asin,
            "title": bad_title,
        },
    )

    materializer.materialize(root, site="egomart", task_ids={"selected-task"})

    catalog = json.loads((root / materializer.AMAZON_CATALOG).read_text())
    entity = next(
        row
        for row in catalog["products"]
        if row["item_id"] == workspace["new_entity_id"]
    )
    assert entity["name"] == f"Captured title {asin}"


@pytest.mark.parametrize("materialization_workspace", ["egomart"], indirect=True)
@pytest.mark.parametrize("identity_field", ["page_url", "product_details"])
def test_conflicting_detail_identity_stops_before_catalog_or_image_writes(
    materialization_workspace,
    identity_field,
) -> None:
    workspace = materialization_workspace
    root = workspace["root"]
    asin = workspace["new_id"]
    detail = {"asin": asin, "title": "Unrelated battery"}
    detail[identity_field] = (
        "https://www.amazon.ca/dp/B079NWHSCS"
        if identity_field == "page_url"
        else {"ASIN \u200f": "\u200e B079NWHSCS", "Voltage": "12V"}
    )
    write_json_atomic(root / materializer.AMAZON_DETAILS / f"{asin}.json", detail)
    catalog = root / workspace["catalog_path"]
    original = catalog.read_bytes()

    with pytest.raises(ValueError, match="ASIN mismatch"):
        materializer.materialize(root, site="egomart", task_ids={"selected-task"})

    assert (catalog.read_bytes(), workspace["downloads"]) == (original, [])


@pytest.mark.parametrize("materialization_workspace", ["egomart"], indirect=True)
def test_conflicting_search_identity_stops_materialization(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace
    root = workspace["root"]
    path = root / materializer.AMAZON_CAPTURES / "selected-task--before_change.json"
    capture = json.loads(path.read_text())
    capture["results"][0]["page_url"] = "https://www.amazon.ca/dp/B079NWHSCS"
    write_json_atomic(path, capture)

    with pytest.raises(ValueError, match="source_url ASIN mismatch"):
        materializer.materialize(root, site="egomart", task_ids={"selected-task"})


@pytest.mark.parametrize("materialization_workspace", ["egomart"], indirect=True)
def test_missing_real_title_stops_materialization(materialization_workspace) -> None:
    workspace = materialization_workspace
    root = workspace["root"]
    path = root / materializer.AMAZON_CAPTURES / "selected-task--before_change.json"
    capture = json.loads(path.read_text())
    for row in capture["results"]:
        row["title"] = "Amazon.ca"
    write_json_atomic(path, capture)

    with pytest.raises(ValueError, match="no usable product title"):
        materializer.materialize(root, site="egomart", task_ids={"selected-task"})


@pytest.mark.parametrize("materialization_workspace", ["egomart"], indirect=True)
def test_us_search_capture_keeps_its_actual_marketplace(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace
    root = workspace["root"]
    path = root / materializer.AMAZON_CAPTURES / "selected-task--before_change.json"
    capture = json.loads(path.read_text())
    for row in capture["results"]:
        row["page_url"] = f"https://www.amazon.com/dp/{row['asin']}"
        row.pop("product_page_url", None)
    write_json_atomic(path, capture)

    materializer.materialize(root, site="egomart", task_ids={"selected-task"})

    catalog = json.loads((root / materializer.AMAZON_CATALOG).read_text())
    entity = next(
        row
        for row in catalog["products"]
        if row["item_id"] == workspace["new_entity_id"]
    )
    assert entity["attributes"]["listing_source"] == "amazon.com"
    assert (
        entity["text_provenance"]["source_url"]
        == f"https://www.amazon.com/dp/{workspace['new_id']}"
    )


def test_capture_filter_reads_only_exact_requested_tasks(tmp_path) -> None:
    for task_id, asin in (("task-a", "B000000001"), ("task-b", "B000000002")):
        write_json_atomic(
            tmp_path / f"{task_id}--before_change.json",
            {
                "task_id": task_id,
                "results": [{"asin": asin, "title": task_id}],
            },
        )
    (tmp_path / "task-a-extra--before_change.json").write_text(
        "invalid unselected capture"
    )

    rows = _capture_rows(tmp_path, Path("."), "asin", task_ids={"task-a", "task-b"})

    assert set(rows) == {"B000000001", "B000000002"}


def test_capture_filter_rejects_mislabeled_capture(tmp_path) -> None:
    write_json_atomic(
        tmp_path / "selected--before_change.json",
        {
            "task_id": "another-task",
            "results": [],
        },
    )

    with pytest.raises(ValueError, match="capture task ID disagrees"):
        _capture_rows(tmp_path, Path("."), "asin", task_ids={"selected"})


@pytest.fixture(params=["egomart", "egomedia"])
def materialization_workspace(request, tmp_path, monkeypatch):
    site = request.param
    if site == "egomart":
        capture_dir, catalog_path = (
            materializer.AMAZON_CAPTURES,
            materializer.AMAZON_CATALOG,
        )
        collection, id_key, upstream_key, title_key = (
            "products",
            "item_id",
            "asin",
            "name",
        )
        old_id, shared_id, new_id = "B000000001", "B000000002", "B000000003"
        namespace, prefix = "amazon", "p"

        def source(upstream_id):
            return {
                "text_provenance": {
                    "source_url": f"https://www.amazon.ca/dp/{upstream_id}"
                }
            }
    else:
        capture_dir, catalog_path = (
            materializer.YOUTUBE_CAPTURES,
            materializer.YOUTUBE_CATALOG,
        )
        collection, id_key, upstream_key, title_key = (
            "videos",
            "video_id",
            "youtube_id",
            "title",
        )
        old_id, shared_id, new_id = "old-video", "shared-video", "new-video"
        namespace, prefix = "youtube", "m"

        def source(upstream_id):
            return {
                "source": {"page_url": f"https://www.youtube.com/watch?v={upstream_id}"}
            }

    old_entity = {
        id_key: "existing-old",
        title_key: "Keep this old title",
        **source(old_id),
    }
    shared_entity = {
        id_key: "existing-shared",
        title_key: "Previous shared title",
        **(
            {"rating": 4.8, "ratings_count": 71}
            if site == "egomart"
            else {
                "view_count": 235,
                "duration_seconds": 80,
            }
        ),
        **source(shared_id),
    }
    provenance_key = "text_provenance" if site == "egomart" else "source"
    shared_entity[provenance_key]["snapshot_paths"] = ["existing-capture.json"]
    write_json_atomic(
        tmp_path / catalog_path,
        {
            collection: [old_entity, shared_entity],
            "content_origin": "Existing catalog origin",
        },
    )
    write_json_atomic(
        tmp_path / materializer.ENTITY_INDEX,
        {
            "schema_version": 1,
            "sites": {
                site: {old_id: "existing-old", shared_id: "existing-shared"},
                "egomaps": {"Q1": "existing-place"},
            },
        },
    )
    for task_id, upstream_ids in (
        ("selected-task", [shared_id, new_id]),
        ("unselected-task", [old_id]),
    ):
        write_json_atomic(
            tmp_path / capture_dir / f"{task_id}--before_change.json",
            {
                "task_id": task_id,
                "captured_at": "2026-09-05T00:00:00Z",
                "results": [
                    {
                        upstream_key: upstream_id,
                        "title": f"Captured title {upstream_id}",
                        "image_url": f"https://example.com/{upstream_id}.jpg",
                        "thumbnail_url": f"https://example.com/{upstream_id}.jpg",
                        "rating": None,
                        "review_count": None,
                        "view_count": None,
                        "duration_seconds": None,
                    }
                    for upstream_id in upstream_ids
                ],
            },
        )
    downloads = []

    def download_images(root, jobs, *, concurrency):
        downloads.extend(jobs)
        return {entity_id: "a" * 64 for entity_id, *_ in downloads}

    monkeypatch.setattr(materializer, "_download_images", download_images)
    return {
        "root": tmp_path,
        "site": site,
        "catalog_path": catalog_path,
        "collection": collection,
        "id_key": id_key,
        "title_key": title_key,
        "old_entity": old_entity,
        "old_id": old_id,
        "shared_id": shared_id,
        "shared_entity": shared_entity,
        "provenance_key": provenance_key,
        "new_id": new_id,
        "new_entity_id": _stable_id(prefix, namespace, new_id),
        "downloads": downloads,
    }


def test_incremental_materialization_preserves_unselected_entities_and_catalog_metadata(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace
    materializer.materialize(workspace["root"], site="all", task_ids={"selected-task"})
    payload = json.loads((workspace["root"] / workspace["catalog_path"]).read_text())
    by_id = {
        entity[workspace["id_key"]]: entity
        for entity in payload[workspace["collection"]]
    }

    assert by_id["existing-old"] == workspace["old_entity"]
    assert payload["content_origin"] == "Existing catalog origin"
    assert set(by_id) == {"existing-old", "existing-shared", workspace["new_entity_id"]}
    assert {job[0] for job in workspace["downloads"]} == {workspace["new_entity_id"]}


def test_incremental_materialization_merges_existing_upstream_mappings(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace
    materializer.materialize(workspace["root"], site="all", task_ids={"selected-task"})
    index = json.loads((workspace["root"] / materializer.ENTITY_INDEX).read_text())

    assert index["sites"] == {
        workspace["site"]: {
            workspace["old_id"]: "existing-old",
            workspace["shared_id"]: "existing-shared",
            workspace["new_id"]: workspace["new_entity_id"],
        },
        "egomaps": {"Q1": "existing-place"},
    }


def test_incremental_materialization_reports_changes_to_existing_shared_entities(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace

    report = materializer.materialize(
        workspace["root"], site="all", task_ids={"selected-task"}
    )

    assert report["requested_task_ids"] == ["selected-task"]
    assert len(report["pending_existing_entity_updates"]) == 1
    changed = report["pending_existing_entity_updates"][0]
    assert (changed["site"], changed["upstream_id"], changed["entity_id"]) == (
        workspace["site"],
        workspace["shared_id"],
        "existing-shared",
    )
    assert workspace["title_key"] in changed["changed_fields"]
    assert not {"rating", "ratings_count", "view_count", "duration_seconds"} & set(
        changed["changed_fields"]
    )


def test_incremental_materialization_preserves_shared_facts_and_merges_snapshot_paths(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace
    materializer.materialize(workspace["root"], site="all", task_ids={"selected-task"})
    payload = json.loads((workspace["root"] / workspace["catalog_path"]).read_text())
    shared = next(
        entity
        for entity in payload[workspace["collection"]]
        if entity[workspace["id_key"]] == "existing-shared"
    )
    provenance_key = workspace["provenance_key"]
    snapshot_paths = shared[provenance_key].pop("snapshot_paths")
    expected = {
        **workspace["shared_entity"],
        provenance_key: {
            key: value
            for key, value in workspace["shared_entity"][provenance_key].items()
            if key != "snapshot_paths"
        },
    }

    assert shared == expected
    assert snapshot_paths[0] == "existing-capture.json"
    assert snapshot_paths[1].endswith("selected-task--before_change.json")


def test_incremental_materialization_keeps_existing_task_execution_hash(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace
    definition = task("refresh")
    definition["site"] = workspace["site"]
    definition["refresh"] = {"policy": "retain", "initial_entity_id": "existing-shared"}
    definition["goal"]["acceptable_states"][0]["selected_entity_ids"] = [
        "existing-shared"
    ]
    definition["truth"]["initial_acceptable_states"] = [
        {"selected_entity_ids": ["existing-shared"], "fields": {}}
    ]

    def execution_hash():
        payload = json.loads(
            (workspace["root"] / workspace["catalog_path"]).read_text()
        )
        return task_site_execution_sha256(
            definition,
            {
                "site": workspace["site"],
                "entities": [
                    {
                        **{
                            key: value
                            for key, value in entity.items()
                            if key != "video_id"
                        },
                        "entity_id": entity[workspace["id_key"]],
                    }
                    for entity in payload[workspace["collection"]]
                ],
            },
        )

    before = execution_hash()
    materializer.materialize(workspace["root"], site="all", task_ids={"selected-task"})

    assert execution_hash() == before


def test_unknown_task_filter_fails_before_catalog_or_index_writes(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace
    paths = [
        workspace["root"] / path
        for path in (workspace["catalog_path"], materializer.ENTITY_INDEX)
    ]
    before = [path.read_bytes() for path in paths]

    with pytest.raises(ValueError, match="no captures for requested task IDs"):
        materializer.materialize(
            workspace["root"], site="all", task_ids={"unknown-task"}
        )

    assert [path.read_bytes() for path in paths] == before
    assert not workspace["downloads"]


def test_default_materialization_still_reads_every_capture(
    materialization_workspace,
) -> None:
    workspace = materialization_workspace

    report = materializer.materialize(workspace["root"], site=workspace["site"])

    assert report == {f"{workspace['site']}_entities": 3}
    assert {job[0] for job in workspace["downloads"]} == {
        "existing-old",
        "existing-shared",
        workspace["new_entity_id"],
    }


def test_cli_accepts_repeated_task_ids(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        "sys.argv",
        [
            "materialize",
            "--root",
            str(tmp_path),
            "--task-id",
            "task-a",
            "--task-id",
            "task-b",
        ],
    )

    def capture_call(root, **kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(materializer, "materialize", capture_call)

    materializer.main()

    assert calls == [
        {"site": "all", "concurrency": 12, "task_ids": {"task-a", "task-b"}}
    ]
