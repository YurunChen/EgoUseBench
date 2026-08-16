from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sites.egomart.app import EgoMartServer, WebState
from sites.egomart.pages import rating_of
from tests.helpers import products, item_id, task


def _post(url: str, **fields: str) -> str:
    payload = urlencode(fields).encode()
    return urlopen(Request(url, data=payload, method="POST")).read().decode()


class WebAppTest(unittest.TestCase):
    def _serve(self, benchmark_task, arm: str, root: Path, gate=None) -> tuple[WebState, EgoMartServer]:
        state = WebState(benchmark_task, arm, products(benchmark_task["track"]), asset_root=root)
        state.event_gate = gate
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
            state, server = self._serve(benchmark_task, "oracle", root)
            with server:
                context = urlopen(server.base_url + "/context").read().decode()
                self.assertIn("What the wearer saw", context)
                self.assertIn("EgoMart", context)
                self.assertNotIn("Visual packet", context)
                self.assertNotIn("V0", context)
                self.assertNotIn("keyframe", context.lower())
                self.assertEqual(urlopen(server.base_url + "/observations/0/images/0").status, 200)
                search = urlopen(server.base_url + "/search?q=Product+1").read().decode()
                self.assertIn("results for", search)
                product = urlopen(
                    server.base_url + f"/product/{item_id('clarify', 1)}"
                ).read().decode()
                self.assertIn("Add to Cart", product)
                self.assertEqual(
                    urlopen(server.base_url + f"/product-image/{item_id('clarify', 1)}").status,
                    200,
                )
                _post(server.base_url + "/cart/add", item_id=item_id("clarify", 1), quantity="1")
                confirmation = _post(server.base_url + "/submit")
                self.assertIn("Order placed", confirmation)
            result = state.result_record()
            self.assertEqual(result["final_state"]["selected_entity_ids"], [item_id("clarify", 1)])
            self.assertTrue(result["final_state"]["submission_completed"])
            self.assertEqual(result["run_status"], {"valid": True, "reason": None})
            self.assertTrue(state.successful_search_seen)
            self.assertTrue(state.product_seen)

    def test_submit_reads_cart_not_parameters_and_rejects_empty(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            state, server = self._serve(benchmark_task, "a_only", root)
            with server:
                empty = _post(server.base_url + "/submit", item_id=item_id("graft", 0))
                self.assertIn("cart is empty", empty.lower())
                _post(server.base_url + "/cart/add", item_id=item_id("graft", 0), quantity="2")
                _post(server.base_url + "/cart/update", item_id=item_id("graft", 0), quantity="1")
                _post(server.base_url + "/cart/add", item_id=item_id("graft", 2))
                _post(server.base_url + "/cart/remove", item_id=item_id("graft", 2))
                _post(server.base_url + "/cart/add", item_id=item_id("graft", 1))
                _post(server.base_url + "/submit")
            result = state.result_record()
            self.assertEqual(result["final_state"]["selected_entity_ids"], [item_id("graft", 0), item_id("graft", 1)])
            self.assertTrue(result["final_state"]["submission_completed"])

    def test_submit_preserves_duplicate_quantities_for_exact_evaluation(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            state, server = self._serve(benchmark_task, "a_only", root)
            with server:
                _post(
                    server.base_url + "/cart/add",
                    item_id=item_id("graft", 0),
                    quantity="2",
                )
                _post(server.base_url + "/submit")

            self.assertEqual(
                state.result_record()["final_state"]["selected_entity_ids"],
                [item_id("graft", 0), item_id("graft", 0)],
            )

    def test_event_gate_blocks_first_submit_only(self) -> None:
        benchmark_task = task("refresh")
        calls: list[int] = []

        def gate() -> bool:
            calls.append(1)
            return len(calls) == 1

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            state, server = self._serve(benchmark_task, "stream", root, gate=gate)
            with server:
                _post(server.base_url + "/cart/add", item_id=item_id("refresh", 1))
                blocked = _post(server.base_url + "/submit")
                self.assertIn("not placed", blocked)
                self.assertFalse(state.submitted)
                confirmation = _post(server.base_url + "/submit")
                self.assertIn("Order placed", confirmation)
            self.assertTrue(state.submitted)
            self.assertEqual(state.submitted_entity_ids, [item_id("refresh", 1)])
            self.assertEqual([a["type"] for a in state.actions if "submit" in a["type"]],
                             ["submit_blocked", "submit"])

    def test_oracle_latest_reveals_both_packets(self) -> None:
        benchmark_task = task("refresh")
        state = WebState(benchmark_task, "oracle_latest", products("refresh"))
        self.assertEqual(state.visible_packet_ids, {"V0", "V1"})
        self.assertEqual(WebState(benchmark_task, "stream", products("refresh")).visible_packet_ids, {"V0"})

    def test_internal_product_annotations_are_not_searchable(self) -> None:
        benchmark_task = task("graft")
        items = products("graft")
        items[0]["attributes"]["annotation"] = "hidden-target-token"
        state = WebState(benchmark_task, "a_only", items)
        with EgoMartServer(state) as server:
            page = urlopen(
                server.base_url + "/search?q=hidden-target-token"
            ).read().decode()
        self.assertIn("No exact matches", page)
        self.assertIn("Try fewer or different search terms", page)
        self.assertFalse(state.successful_search_seen)
        self.assertEqual(state.actions[-1]["results"], 0)
        self.assertFalse(state.actions[-1]["exact_match"])

    def test_search_uses_real_pagination_sorting_and_product_pages(self) -> None:
        benchmark_task = task("graft")
        items = products("graft") + [
            {
                **products("graft")[0],
                "item_id": f"p_extra_{index:02d}",
                "name": f"Extra Product {index:02d}",
                "price": f"${100 + index}.00",
            }
            for index in range(4)
        ]
        state = WebState(benchmark_task, "a_only", items)
        with EgoMartServer(state) as server:
            first_page = urlopen(server.base_url + "/search?q=Product").read().decode()
            second_page = urlopen(server.base_url + "/search?q=Product&page=2").read().decode()
            descending = urlopen(
                server.base_url + "/search?q=Product&sort=price_desc"
            ).read().decode()

        self.assertIn("1-16 of 20 results", first_page)
        self.assertIn("17-20 of 20 results", second_page)
        self.assertIn("Next", first_page)
        self.assertNotIn('action="/cart/add"', first_page)
        self.assertIn("View product", first_page)
        self.assertLess(descending.index("Extra Product 03"), descending.index("Product 15"))
        self.assertEqual(state.actions[-1]["sort"], "price_desc")

    def test_stream_cannot_fetch_future_images_before_release(self) -> None:
        benchmark_task = task("refresh")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._materialize_frames(benchmark_task, root)
            state, server = self._serve(benchmark_task, "stream", root)
            with server:
                with self.assertRaises(HTTPError) as hidden:
                    urlopen(server.base_url + "/observations/1/images/0")
                self.assertEqual(hidden.exception.code, 404)
                state.reveal("V1")
                self.assertEqual(
                    urlopen(server.base_url + "/observations/1/images/0").status,
                    200,
                )

    def test_explicit_zero_rating_is_not_replaced_with_synthetic_metadata(self) -> None:
        self.assertEqual(
            rating_of({"item_id": "p_000000000000", "rating": 0.0, "ratings_count": 0}),
            (0.0, 0),
        )

    def test_storefront_does_not_show_unimplemented_or_unsourced_commerce_state(self) -> None:
        benchmark_task = task("graft")
        items = products("graft")
        state = WebState(benchmark_task, "a_only", items)
        with EgoMartServer(state) as server:
            search = urlopen(server.base_url + "/search?q=Product").read().decode()
            detail = urlopen(
                server.base_url + f"/product/{items[0]['item_id']}"
            ).read().decode()

        for page in (search, detail):
            self.assertNotIn("✓prime", page)
            self.assertNotIn("FREE delivery", page)
        self.assertNotIn("Buy Now", detail)
        self.assertNotIn("More from", detail)
        self.assertNotIn("In Stock", detail)
        self.assertNotIn("Sold by", detail)
        self.assertNotIn("30-day refund", detail)

    def test_sourced_shipping_cost_is_visible_and_included_in_cart_total(self) -> None:
        benchmark_task = task("graft")
        items = products("graft")
        items[0]["price"] = "$20.00"
        items[0]["attributes"] = {"shipping_cost": "$4.99"}
        state = WebState(benchmark_task, "a_only", items)
        with EgoMartServer(state) as server:
            detail = urlopen(
                server.base_url + f"/product/{items[0]['item_id']}"
            ).read().decode()
            cart = _post(
                server.base_url + "/cart/add",
                item_id=items[0]["item_id"],
                quantity="1",
            )

        self.assertIn("Shipping: <b>$4.99</b>", detail)
        self.assertIn("Order total: <b>$24.99</b>", cart)


if __name__ == "__main__":
    unittest.main()
