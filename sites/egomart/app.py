from __future__ import annotations

import mimetypes
import json
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from flask import Flask, Response, request
from werkzeug.exceptions import BadRequest
from sites._paths import resolve_within
from sites.runtime import ActionTrace, SiteRuntime, SiteServer, SiteState

from . import pages


def _is_https_url(value: Any) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme == "https" and bool(parsed.netloc)


def _unsigned_integer(value: str) -> int:
    try:
        return int(value) if value.isdecimal() else 1
    except ValueError:
        return 1


def _cart_quantity() -> int:
    value = request.form.get("quantity", "1")
    if not value.isascii() or not value.isdecimal():
        raise BadRequest("Quantity must be a whole number.")
    try:
        return int(value)
    except ValueError:
        raise BadRequest("Quantity is too large.") from None


def _cart_entries(
    products: dict[str, dict[str, Any]],
    cart: list[str],
) -> list[tuple[dict[str, Any], int]]:
    return [
        (products[item_id], quantity)
        for item_id, quantity in Counter(cart).items()
        if item_id in products
    ]


def create_app(state: SiteState) -> Flask:
    app = Flask(__name__)
    products = state.entities
    by_id = {product["item_id"]: product for product in products}

    def page(
        body: str,
        status: int = 200,
        *,
        query: str = "",
        skin: str = "",
    ) -> tuple[str, int, dict[str, str]]:
        return (
            pages.shell(
                body,
                cart_count=len(state.selected_entity_ids),
                query=query,
                skin=skin,
            ),
            status,
            {"Content-Type": "text/html; charset=utf-8"},
        )

    @app.get("/")
    def home() -> tuple[str, int, dict[str, str]]:
        return page(pages.home_page(products), skin="home-page")

    @app.get("/search")
    @app.get("/s")
    def search() -> tuple[str, int, dict[str, str]]:
        text = request.args.get("q", request.args.get("k", ""))
        sort_key = request.args.get("sort", "featured")
        if sort_key not in pages.SORT_LABELS:
            sort_key = "featured"
        raw_page = request.args.get("page", "1")
        page_number = _unsigned_integer(raw_page)
        matched, no_match = pages.search_results(products, text)
        ordered = pages.sorted_results(matched, sort_key)
        shown, current_page, page_count = pages.paged_results(ordered, page_number)
        state.log(
            "search",
            query=text,
            results=len(matched),
            shown_results=len(shown),
            page=current_page,
            sort=sort_key,
            exact_match=not no_match,
        )
        return page(
            pages.search_page(
                text,
                shown,
                no_match,
                ordered,
                current_page=current_page,
                page_count=page_count,
                sort_key=sort_key,
            ),
            query=text,
        )

    @app.get("/product-image/<item_id>")
    def product_image(item_id: str) -> Response | tuple[str, int, dict[str, str]]:
        product = by_id.get(item_id)
        if product is None or not product.get("image_path"):
            return page("<h1>Product image missing</h1>", 404)
        try:
            path = resolve_within(
                state.asset_root,
                product["image_path"],
                context="site product image",
            )
        except (ValueError, FileNotFoundError):
            return page("<h1>Product image missing</h1>", 404)
        return Response(
            path.read_bytes(),
            mimetype=mimetypes.guess_type(path.name)[0] or "image/jpeg",
        )

    def product_page(item_id: str) -> tuple[str, int, dict[str, str]]:
        product = by_id.get(item_id)
        if product is None:
            return page("<h1>Unknown product</h1>", 404)
        state.log("open_entity", entity_id=item_id)
        return page(pages.product_page(product))

    app.add_url_rule("/product/<item_id>", "product", product_page, methods=["GET"])
    app.add_url_rule("/dp/<item_id>", "product_dp", product_page, methods=["GET"])

    @app.get("/cart")
    @app.get("/gp/cart/view.html")
    def cart() -> tuple[str, int, dict[str, str]]:
        return page(
            pages.cart_page(_cart_entries(by_id, state.selected_entity_ids)),
            skin="cart-page",
        )

    @app.post("/cart/add")
    def add_to_cart() -> tuple[str, int, dict[str, str]]:
        item_id = request.form.get("item_id", "")
        if item_id not in by_id:
            return page("<h1>Unknown product</h1>", 404)
        if not pages.in_stock(by_id[item_id]):
            state.log("selection_unavailable", entity_id=item_id)
            return page("<h1>Currently unavailable</h1>", 409)
        quantity = _cart_quantity()
        if quantity < 1:
            return page("<h1>Quantity must be at least 1.</h1>", 400)
        state.add_quantity(item_id, quantity)
        return cart()

    @app.post("/cart/update")
    def update_cart() -> tuple[str, int, dict[str, str]]:
        item_id = request.form.get("item_id", "")
        if item_id not in by_id:
            return page("<h1>Unknown product</h1>", 404)
        quantity = _cart_quantity()
        state.set_quantity(item_id, quantity)
        return cart()

    @app.post("/cart/remove")
    def remove_from_cart() -> tuple[str, int, dict[str, str]]:
        item_id = request.form.get("item_id", "")
        if item_id in state.selected_entity_ids:
            state.deselect(item_id)
        return cart()

    @app.post("/submit")
    def submit() -> tuple[str, int, dict[str, str]]:
        if not state.selected_entity_ids:
            state.log("submit_empty")
            return page(pages.submit_empty_page())
        state.commit()
        purchased = [
            by_id[item_id] for item_id in state.submitted_entity_ids if item_id in by_id
        ]
        return page(pages.confirmation_page(purchased))

    return app


class EgoMartServer(SiteServer):
    def __init__(self, state: SiteState, host: str = "127.0.0.1", port: int = 0):
        super().__init__(state, create_app(state), host=host, port=port)


def load_data(
    root: str | Path,
    *,
    require_assets: bool = False,
    released_frame_paths: Any = (),
) -> dict[str, Any]:
    """Load the fixed store and expose normalized entity IDs to the harness."""

    source = Path(root).resolve() / "sites" / "egomart" / "products.json"
    payload = json.loads(source.read_text())
    products = payload.get("products")
    if (
        payload.get("schema_version") != 1
        or payload.get("site") != "egomart"
        or not isinstance(products, list)
        or not products
    ):
        raise ValueError(f"{source}: invalid egomart product store")
    ids = [product.get("item_id") for product in products if isinstance(product, dict)]
    if (
        len(ids) != len(products)
        or any(not isinstance(item_id, str) or not item_id for item_id in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError(f"{source}: invalid or duplicate product IDs")
    image_paths: set[str] = set()
    for product in products:
        image_path = str(product.get("image_path", "")).strip()
        if image_path and image_path in image_paths:
            raise ValueError(f"{source}: duplicate product image {image_path}")
        image_paths.add(image_path)
    if require_assets:
        if any(product.get("placeholder", False) for product in products):
            raise ValueError(f"{source}: release products contain placeholders")
        released = set(released_frame_paths)
        for product in products:
            entity_id = str(product["item_id"])
            if not re.fullmatch(r"p_[0-9a-f]{12}", entity_id):
                raise ValueError(f"{source}: non-opaque product ID {entity_id}")
            image_path = str(product.get("image_path", "")).strip()
            if not image_path or image_path in released:
                raise ValueError(f"{source}: invalid product image for {entity_id}")
            resolved = resolve_within(
                root, image_path, context=f"{entity_id} product image"
            )
            provenance = product.get("image_provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{source}: image provenance missing for {entity_id}")
            if not _is_https_url(provenance.get("source_url")):
                raise ValueError(
                    f"{source}: independent web page missing for product image {entity_id}"
                )
            if not _is_https_url(provenance.get("image_url")):
                raise ValueError(
                    f"{source}: independent web image URL missing for {entity_id}"
                )
            forbidden_origin_fields = {
                "crop_method",
                "retrieval_method",
                "source_recording",
                "timestamp_seconds",
            }
            if forbidden_origin_fields.intersection(provenance):
                raise ValueError(
                    f"{source}: source-video product image forbidden for {entity_id}"
                )
            for key in ("retrieved_at", "license_or_access_note", "sha256"):
                if not str(provenance.get(key, "")).strip():
                    raise ValueError(
                        f"{source}: image provenance {key} missing for {entity_id}"
                    )
            declared_hash = str(provenance.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
                raise ValueError(f"{source}: invalid image hash for {entity_id}")
            if hashlib.sha256(resolved.read_bytes()).hexdigest() != declared_hash:
                raise ValueError(
                    f"{source}: product image hash mismatch for {entity_id}"
                )
    return {
        "schema_version": payload["schema_version"],
        "site": payload["site"],
        "supported_fields": [],
        "entities": [
            {**product, "entity_id": product["item_id"]} for product in products
        ],
    }


def create_runtime(
    data: dict[str, Any],
    asset_root: str | Path,
    *,
    trace: ActionTrace | None = None,
    port: int = 0,
) -> SiteRuntime:
    state = SiteState(data, asset_root=Path(asset_root), trace=trace or ActionTrace())
    return SiteRuntime(state=state, server=EgoMartServer(state, port=port))
