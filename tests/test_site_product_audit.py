from __future__ import annotations

import hashlib
import json
import re
from statistics import median
from pathlib import Path
from typing import cast

from PIL import Image

from egocua.environment import load_site_data
from egocua.run_ledger import canonical_sha256
from egocua.schema import load_tasks
from sites.egomart.pages import rating_of


ROOT = Path(__file__).parents[1]


def test_site_audit_index_binds_every_task_to_the_fixed_site() -> None:
    tasks = load_tasks(ROOT / "data/tasks")
    payload = json.loads((ROOT / "artifacts/reports/site-audits/index.json").read_text())
    audit_index = payload["tasks"]

    assert payload["schema_version"] == 2
    active_sites = {task["site"] for task in tasks}
    assert set(payload["sites"]) == active_sites
    site_counts = {
        site: len(load_site_data(ROOT, site)["entities"])
        for site in active_sites
    }
    assert {
        site: record["entities"]
        for site, record in payload["sites"].items()
    } == site_counts
    egomart_count = site_counts["egomart"]
    assert payload["egomart_duplicate_audit"] == {
        "distinct_images": egomart_count,
        "exact_duplicate_image_groups": 0,
        "listings_in_duplicate_image_groups": 0,
        "exact_duplicate_name_groups": 0,
        "listings_in_duplicate_name_groups": 0,
        "tasks_with_exact_duplicate_references": 0,
    }
    assert set(audit_index) == {task["id"] for task in tasks}
    assert {path.stem for path in (ROOT / "artifacts/reports/site-audits").glob("*.jpg")} == set(
        audit_index
    )
    for task in tasks:
        task_path = ROOT / "data/tasks" / task["track"] / f"{task['id']}.json"
        record = audit_index[task["id"]]
        audit_path = ROOT / record["audit_path"]
        assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == record["audit_sha256"]
        assert hashlib.sha256(task_path.read_bytes()).hexdigest() == record["task_sha256"]
        assert record["site"] == task["site"]
        assert canonical_sha256(load_site_data(ROOT, task["site"])) == record[
            "site_data_sha256"
        ]


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


def test_product_asset_directory_has_no_orphan_files() -> None:
    site = json.loads((ROOT / "sites/egomart/products.json").read_text())
    referenced = {ROOT / product["image_path"] for product in site["products"]}
    actual = {
        path
        for path in (ROOT / "sites/egomart/static/products").iterdir()
        if path.is_file()
    }

    assert actual == referenced


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int((len(ordered) - 1) * fraction)]


def test_egomart_display_data_matches_retail_metadata_scale() -> None:
    products = json.loads((ROOT / "sites/egomart/products.json").read_text())["products"]
    prices = [
        float(match.group())
        for product in products
        if (match := re.search(r"\d+(?:\.\d+)?", str(product.get("price", ""))))
    ]
    fallback = [product for product in products if "rating" not in product]
    ratings = [rating_of(product)[0] for product in fallback]
    counts = [float(rating_of(product)[1]) for product in fallback]

    assert len(prices) == len(products)
    assert 1 <= _quantile(prices, 0.10) <= 10
    assert 15 <= median(prices) <= 40
    assert 45 <= _quantile(prices, 0.90) <= 180
    assert 3.3 <= _quantile(ratings, 0.10) <= 3.7
    assert 4.3 <= median(ratings) <= 4.6
    assert 4.8 <= _quantile(ratings, 0.90) <= 5.0
    assert _quantile(counts, 0.10) <= 3
    assert 20 <= median(counts) <= 60
    assert 500 <= _quantile(counts, 0.90) <= 1_200
    assert max(counts) >= 10_000


def test_egomart_main_images_follow_catalog_image_distribution() -> None:
    products = json.loads((ROOT / "sites/egomart/products.json").read_text())["products"]
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
    assert sum(side >= 500 for side in long_sides) / total >= 0.95
    assert sum(side >= 1_000 for side in long_sides) / total >= 0.70
    assert squareish / total >= 0.65
    assert white_background / total >= 0.70
