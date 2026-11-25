from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sites.egomart.app import EgoMartServer
from sites.runtime import SiteState
from sites.egomart.pages import rating_of
from tests.helpers import products, item_id, task


def _state(items, asset_root: Path | None = None) -> SiteState:
    data = {
        "schema_version": 1,
        "site": "egomart",
        "supported_fields": [],
        "entities": items,
    }
    return SiteState(data, asset_root=asset_root) if asset_root else SiteState(data)


def _post(url: str, **fields: str) -> str:
    payload = urlencode(fields).encode()
    return urlopen(Request(url, data=payload, method="POST")).read().decode()


class WebAppTest(unittest.TestCase):
    def _serve(self, benchmark_task, root: Path) -> tuple[SiteState, EgoMartServer]:
        state = _state(products(benchmark_task["track"]), asset_root=root)
        return state, EgoMartServer(state)

    def _materialize_frames(self, benchmark_task, root: Path) -> None:
        for packet in benchmark_task["visual_packets"]:
            for frame in packet["frame_paths"]:
                path = root / frame
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fake-jpeg")

    def _materialize_product_images(self, benchmark_task, root: Path) -> None:
        for item in products(benchmark_task["track"]):
            path = root / item["image_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-product-jpeg")

    def test_store_flow_records_deterministic_state(self) -> None:
        benchmark_task = task("clarify")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            self._materialize_product_images(benchmark_task, root)
            state, server = self._serve(benchmark_task, root)
            with server:
                home = urlopen(server.base_url + "/").read().decode()
                self.assertIn("amazon", home)
                self.assertNotIn("EgoMart", home)
                self.assertNotIn("What the wearer saw", home)
                self.assertNotIn("Visual packet", home)
                self.assertNotIn("V0", home)
                self.assertNotIn("keyframe", home.lower())
                with self.assertRaises(HTTPError) as missing:
                    urlopen(server.base_url + "/context")
                self.assertEqual(missing.exception.code, 404)
                with self.assertRaises(HTTPError) as hidden_frame:
                    urlopen(server.base_url + "/observations/0/images/0")
                self.assertEqual(hidden_frame.exception.code, 404)
                search = (
                    urlopen(server.base_url + "/search?q=Product+1").read().decode()
                )
                self.assertIn("results for", search)
                self.assertNotIn("Wearer Context", search)
                self.assertNotIn("Delivering to", search)
                self.assertIn('value="Product 1"', search)
                product = (
                    urlopen(server.base_url + f"/product/{item_id('clarify', 1)}")
                    .read()
                    .decode()
                )
                self.assertIn("Add to Cart", product)
                self.assertEqual(
                    urlopen(
                        server.base_url + f"/product-image/{item_id('clarify', 1)}"
                    ).status,
                    200,
                )
                _post(
                    server.base_url + "/cart/add",
                    item_id=item_id("clarify", 1),
                    quantity="1",
                )
                confirmation = _post(server.base_url + "/submit")
                self.assertIn("Order placed", confirmation)
            result = state.final_state()
            self.assertEqual(result["selected_entity_ids"], [item_id("clarify", 1)])
            self.assertTrue(result["submission_completed"])
            self.assertIn("search", [action["type"] for action in state.actions])
            self.assertIn("open_entity", [action["type"] for action in state.actions])

    def test_submit_reads_cart_not_parameters_and_rejects_empty(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            state, server = self._serve(benchmark_task, root)
            with server:
                empty = _post(server.base_url + "/submit", item_id=item_id("graft", 0))
                self.assertIn("cart is empty", empty.lower())
                _post(
                    server.base_url + "/cart/add",
                    item_id=item_id("graft", 0),
                    quantity="2",
                )
                _post(
                    server.base_url + "/cart/update",
                    item_id=item_id("graft", 0),
                    quantity="1",
                )
                _post(server.base_url + "/cart/add", item_id=item_id("graft", 2))
                _post(server.base_url + "/cart/remove", item_id=item_id("graft", 2))
                _post(server.base_url + "/cart/add", item_id=item_id("graft", 1))
                _post(server.base_url + "/submit")
            result = state.final_state()
            self.assertEqual(
                result["selected_entity_ids"],
                [item_id("graft", 0), item_id("graft", 1)],
            )
            self.assertTrue(result["submission_completed"])

    def test_submit_preserves_duplicate_quantities_for_exact_evaluation(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            state, server = self._serve(benchmark_task, root)
            with server:
                _post(
                    server.base_url + "/cart/add",
                    item_id=item_id("graft", 0),
                    quantity="2",
                )
                _post(server.base_url + "/submit")

            self.assertEqual(
                state.final_state()["selected_entity_ids"],
                [item_id("graft", 0), item_id("graft", 0)],
            )

    def test_internal_product_annotations_are_not_searchable(self) -> None:
        items = products("graft")
        items[0]["attributes"]["annotation"] = "hiddenprivatetoken"
        state = _state(items)
        with EgoMartServer(state) as server:
            page = (
                urlopen(server.base_url + "/search?q=hiddenprivatetoken")
                .read()
                .decode()
            )
        self.assertIn("No exact matches", page)
        self.assertIn("Try fewer or different search terms", page)
        self.assertEqual(state.actions[-1]["results"], 0)
        self.assertFalse(state.actions[-1]["exact_match"])

    def test_search_uses_hidden_listing_synonyms_without_rendering_them(self) -> None:
        items = products("graft")
        items[0]["attributes"]["search_aliases"] = ["metal strainer"]
        state = _state(items)
        with EgoMartServer(state) as server:
            page = urlopen(server.base_url + "/search?q=metal+strainer").read().decode()

        self.assertIn(items[0]["name"], page)
        self.assertNotIn("search_aliases", page)
        self.assertTrue(state.actions[-1]["exact_match"])

    def test_search_uses_real_pagination_sorting_and_product_pages(self) -> None:
        items = products("graft") + [
            {
                **products("graft")[0],
                "item_id": f"p_extra_{index:02d}",
                "name": f"Extra Product {index:02d}",
                "price": f"${100 + index}.00",
            }
            for index in range(40)
        ]
        state = _state(items)
        with EgoMartServer(state) as server:
            first_page = urlopen(server.base_url + "/search?q=Product").read().decode()
            second_page = (
                urlopen(server.base_url + "/search?q=Product&page=2").read().decode()
            )
            descending = (
                urlopen(server.base_url + "/search?q=Product&sort=price_desc")
                .read()
                .decode()
            )

        self.assertIn("1-48 of 56 results", first_page)
        self.assertIn("49-56 of 56 results", second_page)
        self.assertIn("Next", first_page)
        self.assertNotIn('action="/cart/add"', first_page)
        self.assertIn("View product", first_page)
        self.assertLess(
            descending.index("Extra Product 39"), descending.index("Product 15")
        )
        self.assertEqual(state.actions[-1]["sort"], "price_desc")

    def test_storefront_does_not_host_released_observations(self) -> None:
        benchmark_task = task("refresh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            state, server = self._serve(benchmark_task, root)
            with server:
                for path in (
                    "/context",
                    "/observations/0/images/0",
                    "/observations/1/images/0",
                ):
                    with self.assertRaises(HTTPError) as missing:
                        urlopen(server.base_url + path)
                    self.assertEqual(missing.exception.code, 404)

    def test_explicit_zero_rating_is_not_replaced_with_synthetic_metadata(self) -> None:
        self.assertEqual(
            rating_of({"item_id": "p_000000000000", "rating": 0.0, "ratings_count": 0}),
            (0.0, 0),
        )

    def test_storefront_hides_wearer_context_from_amazon_chrome(self) -> None:
        state = _state(products("graft"))
        with EgoMartServer(state) as server:
            search = urlopen(server.base_url + "/search?q=Product").read().decode()
            detail = (
                urlopen(server.base_url + f"/product/{products('graft')[0]['item_id']}")
                .read()
                .decode()
            )
            cart = urlopen(server.base_url + "/cart").read().decode()
            with self.assertRaises(HTTPError) as missing:
                urlopen(server.base_url + "/context")
            self.assertEqual(missing.exception.code, 404)

        for page in (search, detail, cart):
            self.assertNotIn("Wearer Context", page)
            self.assertNotIn("Delivering to", page)
            self.assertNotIn('href="/context"', page)
            self.assertNotIn("What the wearer saw", page)
        self.assertIn("Results", search)
        self.assertIn("All results", detail)
        self.assertIn("Continue shopping", cart)
        self.assertNotIn("Saved For Later", cart)
        self.assertNotIn("Sign in", search)
        self.assertNotIn("Hello, sign in", search)

    def test_storefront_home_is_amazon_deck_not_wearer_context(self) -> None:
        items = products("graft")
        state = _state(items)
        with EgoMartServer(state) as server:
            home = urlopen(server.base_url + "/").read().decode()
            with self.assertRaises(HTTPError) as missing:
                urlopen(server.base_url + "/context")
            self.assertEqual(missing.exception.code, 404)

        self.assertIn("home-deck", home)
        self.assertIn("Shop", home)
        self.assertIn("Search Amazon", home)
        self.assertNotIn("EgoMart", home)
        self.assertNotIn("What the wearer saw", home)
        self.assertNotIn("Delivering to", home)
        self.assertNotIn("Sign in", home)

    def test_storefront_does_not_show_unimplemented_or_unsourced_commerce_state(
        self,
    ) -> None:
        items = products("graft")
        state = _state(items)
        with EgoMartServer(state) as server:
            search = urlopen(server.base_url + "/search?q=Product").read().decode()
            detail = (
                urlopen(server.base_url + f"/product/{items[0]['item_id']}")
                .read()
                .decode()
            )

        for page in (search, detail):
            self.assertNotIn("✓prime", page)
            self.assertNotIn("FREE delivery", page)
        self.assertNotIn("Buy Now", detail)
        self.assertNotIn("More from", detail)
        self.assertNotIn("In Stock", detail)
        self.assertNotIn("Sold by", detail)
        self.assertNotIn("30-day refund", detail)

    def test_sourced_shipping_cost_is_visible_and_included_in_cart_total(self) -> None:
        items = products("graft")
        items[0]["price"] = "$20.00"
        items[0]["attributes"] = {"shipping_cost": "$4.99"}
        state = _state(items)
        with EgoMartServer(state) as server:
            detail = (
                urlopen(server.base_url + f"/product/{items[0]['item_id']}")
                .read()
                .decode()
            )
            cart = _post(
                server.base_url + "/cart/add",
                item_id=items[0]["item_id"],
                quantity="1",
            )

        self.assertIn("Shipping: <b>$4.99</b>", detail)
        self.assertIn("Order total: <b>$24.99</b>", cart)

    def test_pdp_shows_reviews_dimensions_and_delivery_without_free_shipping(
        self,
    ) -> None:
        items = products("graft")
        sample = {
            "item_id": "p_test_commerce_pdp",
            "name": "Sample Household Item",
            "price": "$12.00",
            "attributes": {
                "dimensions": "4.0 x 3.0 x 2.0 in",
                "delivery_days": 3,
            },
            "reviews": [
                {
                    "author": "Alex M.",
                    "rating": 5,
                    "title": "Works well",
                    "text": "Exactly what I needed for daily use.",
                    "date": "2024-05-01",
                }
            ],
        }
        items.append(sample)
        state = _state(items)
        with EgoMartServer(state) as server:
            detail = (
                urlopen(server.base_url + f"/product/{sample['item_id']}")
                .read()
                .decode()
            )

        self.assertIn('id="reviews"', detail)
        self.assertIn("Customer reviews", detail)
        self.assertIn("Exactly what I needed for daily use.", detail)
        self.assertIn("<b>Dimensions:</b> 4.0 x 3.0 x 2.0 in", detail)
        self.assertIn("Arrives in <b>3</b> days", detail)
        self.assertNotIn("FREE delivery", detail)
        self.assertNotIn("Buy Now", detail)


if __name__ == "__main__":
    unittest.main()
