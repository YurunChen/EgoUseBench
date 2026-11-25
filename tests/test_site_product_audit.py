from __future__ import annotations

import hashlib
import json
import re
import pytest
from pathlib import Path
from typing import cast

from PIL import Image

from egocua.environment import load_site_data
from egocua.run_ledger import task_sha256, task_site_execution_sha256
from egocua.schema import load_tasks
from sites.egomart.pages import rating_of
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty


ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(live_corpus_empty(), reason=ARCHIVED_CORPUS)
@pytest.mark.requires_local_data("artifacts/reports/site-audits/index.json")
def test_site_audit_index_binds_every_task_to_the_fixed_site() -> None:
    tasks = load_tasks(ROOT / "data/tasks")
    payload = json.loads(
        (ROOT / "artifacts/reports/site-audits/index.json").read_text()
    )
    audit_index = payload["tasks"]

    assert payload["schema_version"] == 4
    active_sites = {task["site"] for task in tasks}
    assert set(payload["sites"]) == active_sites
    site_data = {site: load_site_data(ROOT, site) for site in active_sites}
    site_counts = {site: len(data["entities"]) for site, data in site_data.items()}
    assert {
        site: record["entities"] for site, record in payload["sites"].items()
    } == site_counts
    duplicate_audit = payload["egomart_duplicate_audit"]
    assert duplicate_audit["distinct_images"] == site_counts["egomart"]
    # A complete real Amazon result window legitimately contains repeated titles,
    # variants and shared brand images. Keep the counts visible as diagnostics;
    # do not delete or synthesize listings merely to force them to zero.
    assert duplicate_audit["exact_duplicate_image_groups"] >= 0
    assert duplicate_audit["exact_duplicate_name_groups"] >= 0
    assert 0 <= duplicate_audit["tasks_with_exact_duplicate_references"] <= len(tasks)
    assert set(audit_index) == {task["id"] for task in tasks}
    assert {
        path.stem for path in (ROOT / "artifacts/reports/site-audits").glob("*.jpg")
    } == set(audit_index)
    for task in tasks:
        record = audit_index[task["id"]]
        audit_path = ROOT / record["audit_path"]
        assert (
            hashlib.sha256(audit_path.read_bytes()).hexdigest()
            == record["audit_sha256"]
        )
        assert task_sha256(task) == record["task_sha256"]
        assert record["site"] == task["site"]
        assert (
            task_site_execution_sha256(task, site_data[task["site"]])
            == record["task_site_execution_sha256"]
        )
        assert "site_data_sha256" not in record


def test_global_site_products_do_not_contain_construction_only_metadata() -> None:
    site = json.loads((ROOT / "sites/egomart/products.json").read_text())

    assert site["schema_version"] == 1
    assert site["site"] == "egomart"
    assert site["products"]
    assert len({product["item_id"] for product in site["products"]}) == len(
        site["products"]
    )
    assert all("role" not in product for product in site["products"])
    assert all("tier" not in product for product in site["products"])
    assert all(
        "visible_variant" not in product.get("attributes", {})
        for product in site["products"]
    )


@pytest.mark.requires_local_data("sites/egomart/static/products")
def test_all_current_product_images_exist() -> None:
    site = json.loads((ROOT / "sites/egomart/products.json").read_text())
    referenced = {ROOT / product["image_path"] for product in site["products"]}
    actual = {
        path
        for path in (ROOT / "sites/egomart/static/products").rglob("*")
        if path.is_file()
    }

    # The shared asset directory also serves frozen historical study catalogs.
    assert referenced <= actual


def test_egomart_display_data_uses_only_captured_retail_metadata() -> None:
    products = json.loads((ROOT / "sites/egomart/products.json").read_text())[
        "products"
    ]
    prices = [
        float(match.group())
        for product in products
        if (match := re.search(r"\d+(?:\.\d+)?", str(product.get("price", ""))))
    ]
    declared = [product for product in products if "rating" in product]
    missing = [product for product in products if "rating" not in product]

    assert prices
    assert declared
    assert missing
    assert all(
        rating_of(product)[0] == float(product["rating"]) for product in declared
    )
    assert all(rating_of(product) == (None, None) for product in missing)
    assert all(0 <= float(product["rating"]) <= 5 for product in declared)
    assert all(int(product.get("ratings_count", 0)) >= 0 for product in declared)


@pytest.mark.requires_local_data("sites/egomart/static/products")
def test_egomart_main_images_follow_catalog_image_distribution() -> None:
    products = json.loads((ROOT / "sites/egomart/products.json").read_text())[
        "products"
    ]
    long_sides: list[int] = []
    squareish = 0
    white_background = 0
    for product in products:
        with Image.open(ROOT / product["image_path"]) as image:
            width, height = image.size
            long_sides.append(max(width, height))
            squareish += 0.75 <= width / height <= 1.33
            rgba = image.convert("RGBA")
            corners = [
                cast(tuple[int, int, int, int], rgba.getpixel((0, 0))),
                cast(tuple[int, int, int, int], rgba.getpixel((width - 1, 0))),
                cast(tuple[int, int, int, int], rgba.getpixel((0, height - 1))),
                cast(tuple[int, int, int, int], rgba.getpixel((width - 1, height - 1))),
            ]
            white_background += all(
                alpha < 16 or min(red, green, blue) >= 240
                for red, green, blue, alpha in corners
            )

    total = len(products)
    # Complete real result windows include search-card images (typically 320 px)
    # as well as larger product-detail images. Resolution is therefore an
    # acquisition-integrity floor, not a reason to synthesize or upscale assets.
    assert sum(side >= 300 for side in long_sides) / total >= 0.99
    assert squareish / total >= 0.60
    assert white_background / total >= 0.60
