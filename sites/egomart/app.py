from __future__ import annotations

import mimetypes
import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from flask import Flask, Response, request
from egocua.protocol import initial_visible_packets
from egocua.workspace import resolve_within
from sites.runtime import SiteRuntime, SiteServer, register_observation_routes

from . import pages


@dataclass
class WebState:
    task: dict[str, Any]
    arm: str
    products: list[dict[str, Any]]
    asset_root: Path = field(default_factory=Path.cwd)
    current_step: int = 0
    visible_packet_ids: set[str] = field(default_factory=set)
    cart: list[str] = field(default_factory=list)
    submitted_entity_ids: list[str] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    submitted_fields: dict[str, str] = field(default_factory=dict)
    submitted: bool = False
    successful_search_seen: bool = False
    product_seen: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)
    event_gate: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if not self.visible_packet_ids:
            self.visible_packet_ids = initial_visible_packets(self.task, self.arm)

    def reveal(self, packet_id: str) -> None:
        self.visible_packet_ids.add(packet_id)

    def log(self, action_type: str, **payload: Any) -> None:
        self.actions.append({"type": action_type, "step": self.current_step, **payload})

    def cart_entries(self) -> list[tuple[dict[str, Any], int]]:
        entries: list[tuple[dict[str, Any], int]] = []
        seen: set[str] = set()
        for item_id in self.cart:
            if item_id in seen:
                continue
            item = _find_product(self.products, item_id)
            if item is not None:
                seen.add(item_id)
                entries.append((item, self.cart.count(item_id)))
        return entries

    def result_record(
        self, *, valid: bool = True, reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": self.task["id"],
            "track": self.task["track"],
            "arm": self.arm,
            "run_status": {"valid": valid, "reason": reason},
            "initial_state": {
                "selected_entity_ids": [],
                "fields": {},
                "submission_completed": False,
            },
            "steps": self.current_step,
            "actions": self.actions,
            "final_state": {
                "selected_entity_ids": self.submitted_entity_ids,
                "fields": self.submitted_fields,
                "submission_completed": self.submitted,
            },
        }


def create_app(state: WebState) -> Flask:
    app = Flask(__name__)

    def page(body: str, status: int = 200) -> tuple[str, int, dict[str, str]]:
        return pages.shell(body, cart_count=len(state.cart)), status, {
            "Content-Type": "text/html; charset=utf-8"
        }

    register_observation_routes(
        app,
        state,
        page,
        pages.context_page,
        lambda observation_index, image_index: pages.frame_viewer(
            str(observation_index), str(image_index)
        ),
    )

    @app.get("/search")
    @app.get("/s")
    def search() -> tuple[str, int, dict[str, str]]:
        text = request.args.get("q", request.args.get("k", ""))
        sort_key = request.args.get("sort", "featured")
        if sort_key not in pages.SORT_LABELS:
            sort_key = "featured"
        raw_page = request.args.get("page", "1")
        page_number = int(raw_page) if raw_page.isdigit() else 1
        matched, no_match = pages.search_results(state.products, text)
        ordered = pages.sorted_results(matched, sort_key)
        shown, current_page, page_count = pages.paged_results(ordered, page_number)
        if text.strip() and matched:
            state.successful_search_seen = True
        state.log(
            "search",
            query=text,
            results=len(matched),
            shown_results=len(shown),
            page=current_page,
            sort=sort_key,
            exact_match=not no_match,
        )
        return page(pages.search_page(
            text,
            shown,
            no_match,
            ordered,
            current_page=current_page,
            page_count=page_count,
            sort_key=sort_key,
        ))

    @app.get("/product-image/<item_id>")
    def product_image(item_id: str) -> Response | tuple[str, int, dict[str, str]]:
        product = _find_product(state.products, item_id)
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
        product = _find_product(state.products, item_id)
        if product is None:
            return page("<h1>Unknown product</h1>", 404)
        state.product_seen = True
        state.log("open_entity", entity_id=item_id)
        return page(pages.product_page(product))

    app.add_url_rule("/product/<item_id>", "product", product_page, methods=["GET"])
    app.add_url_rule("/dp/<item_id>", "product_dp", product_page, methods=["GET"])

    @app.get("/cart")
    @app.get("/gp/cart/view.html")
    def cart() -> tuple[str, int, dict[str, str]]:
        return page(pages.cart_page(state.cart_entries()))

    @app.post("/cart/add")
    def add_to_cart() -> tuple[str, int, dict[str, str]]:
        item_id = request.form.get("item_id", "")
        if _find_product(state.products, item_id) is None:
            return page("<h1>Unknown product</h1>", 404)
        raw_quantity = request.form.get("quantity", "1")
        quantity = max(1, min(3, int(raw_quantity))) if raw_quantity.isdigit() else 1
        state.cart.extend([item_id] * quantity)
        state.log("select_entity", entity_id=item_id, quantity=quantity)
        return page(pages.cart_page(state.cart_entries()))

    @app.post("/cart/update")
    def update_cart() -> tuple[str, int, dict[str, str]]:
        item_id = request.form.get("item_id", "")
        raw_quantity = request.form.get("quantity", "1")
        quantity = max(0, min(3, int(raw_quantity))) if raw_quantity.isdigit() else 1
        rebuilt: list[str] = []
        placed = False
        for entry in state.cart:
            if entry == item_id:
                if not placed:
                    rebuilt.extend([item_id] * quantity)
                    placed = True
            else:
                rebuilt.append(entry)
        state.cart = rebuilt
        action = "deselect_entity" if quantity == 0 else "select_entity"
        state.log(action, entity_id=item_id, quantity=quantity)
        return page(pages.cart_page(state.cart_entries()))

    @app.post("/cart/remove")
    def remove_from_cart() -> tuple[str, int, dict[str, str]]:
        item_id = request.form.get("item_id", "")
        if item_id in state.cart:
            state.cart = [entry for entry in state.cart if entry != item_id]
            state.log("deselect_entity", entity_id=item_id)
        return page(pages.cart_page(state.cart_entries()))

    @app.post("/submit")
    def submit() -> tuple[str, int, dict[str, str]]:
        if state.event_gate is not None and state.event_gate():
            state.log("submit_blocked", cart=list(state.cart))
            return page(pages.submit_blocked_page())
        if not state.cart:
            state.log("submit_empty")
            return page(pages.submit_empty_page())
        state.submitted_entity_ids = list(state.cart)
        state.submitted = True
        state.submitted_fields = dict(state.fields)
        state.log("submit", entity_ids=list(state.cart), fields={})
        products = [
            product
            for item_id in state.cart
            if (product := _find_product(state.products, item_id)) is not None
        ]
        return page(pages.confirmation_page(products))

    return app


class EgoMartServer(SiteServer):
    def __init__(self, state: WebState, host: str = "127.0.0.1", port: int = 0):
        super().__init__(state, create_app(state), host=host, port=port)


def load_data(
    root: str | Path, *, require_assets: bool = False,
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
    if len(ids) != len(products) or len(ids) != len(set(ids)):
        raise ValueError(f"{source}: invalid or duplicate product IDs")
    names: set[str] = set()
    image_paths: set[str] = set()
    image_hashes: set[str] = set()
    for product in products:
        name = " ".join(str(product.get("name", "")).casefold().split())
        image_path = str(product.get("image_path", "")).strip()
        image_hash = str(product.get("image_provenance", {}).get("sha256", "")).strip()
        if name and name in names:
            raise ValueError(f"{source}: duplicate product name {name}")
        if image_path and image_path in image_paths:
            raise ValueError(f"{source}: duplicate product image {image_path}")
        if image_hash and image_hash in image_hashes:
            raise ValueError(f"{source}: duplicate product image hash {image_hash}")
        names.add(name)
        image_paths.add(image_path)
        image_hashes.add(image_hash)
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
            resolved = resolve_within(root, image_path, context=f"{entity_id} product image")
            provenance = product.get("image_provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"{source}: image provenance missing for {entity_id}")
            if not str(
                provenance.get("source_url") or provenance.get("retrieval_method") or ""
            ).strip():
                raise ValueError(f"{source}: image provenance source missing for {entity_id}")
            for key in ("retrieved_at", "license_or_access_note", "sha256"):
                if not str(provenance.get(key, "")).strip():
                    raise ValueError(f"{source}: image provenance {key} missing for {entity_id}")
            declared_hash = str(provenance.get("sha256", ""))
            if not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
                raise ValueError(f"{source}: invalid image hash for {entity_id}")
            if hashlib.sha256(resolved.read_bytes()).hexdigest() != declared_hash:
                raise ValueError(f"{source}: product image hash mismatch for {entity_id}")
    return {
        "schema_version": payload["schema_version"],
        "site": payload["site"],
        "supported_fields": [],
        "entities": [
            {**product, "entity_id": product["item_id"]}
            for product in products
        ],
    }


def create_runtime(
    task: dict[str, Any], arm: str, data: dict[str, Any], asset_root: str | Path,
    *, port: int = 0,
) -> SiteRuntime:
    state = WebState(task, arm, data["entities"], asset_root=Path(asset_root))
    return SiteRuntime(state=state, server=EgoMartServer(state, port=port))


def _find_product(
    products: list[dict[str, Any]], item_id: str,
) -> dict[str, Any] | None:
    return next((product for product in products if product["item_id"] == item_id), None)
