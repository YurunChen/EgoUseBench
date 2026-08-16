from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from egocua.environment import EnvironmentError, load_site_data
from tests.helpers import products


def _write_site(root: Path, site_products: list[dict[str, Any]]) -> None:
    path = root / "sites/egomart/products.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "schema_version": 1,
        "site": "egomart",
        "products": site_products,
    }))


def test_all_tasks_load_the_same_fixed_site_products(tmp_path: Path) -> None:
    site_products = products("graft")
    _write_site(tmp_path, site_products)

    first = load_site_data(tmp_path, "egomart")
    second = load_site_data(tmp_path, "egomart")

    assert [entity["item_id"] for entity in first["entities"]] == [
        product["item_id"] for product in site_products
    ]
    assert second == first


def test_rejects_duplicate_site_product_ids(tmp_path: Path) -> None:
    site_products = products("graft")
    site_products[1]["item_id"] = site_products[0]["item_id"]
    _write_site(tmp_path, site_products)

    with pytest.raises(EnvironmentError, match="duplicate product IDs"):
        load_site_data(tmp_path, "egomart")


def test_rejects_duplicate_site_product_names(tmp_path: Path) -> None:
    site_products = products("graft")
    site_products[1]["name"] = f"  {site_products[0]['name'].upper()}  "
    _write_site(tmp_path, site_products)

    with pytest.raises(EnvironmentError, match="duplicate product name"):
        load_site_data(tmp_path, "egomart")


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
