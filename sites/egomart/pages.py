"""EgoMart storefront pages.

Server-rendered HTML following the common search-result, product-detail, cart,
and checkout structure used by large retail sites. The wordmark is deliberately
de-branded to EgoMart; all state lives in the caller, and these functions are
pure builders.
"""

from __future__ import annotations

import hashlib
import html
import math
import re
from typing import Any
from urllib.parse import quote, urlencode

STORE_NAME = "EgoMart"
INTERNAL_ATTRIBUTES = {"annotation", "rank"}
RESULTS_PER_PAGE = 16
SORT_LABELS = {
    "featured": "Featured",
    "price_asc": "Price: Low to High",
    "price_desc": "Price: High to Low",
    "rating_desc": "Avg. Customer Review",
}

# Deterministic display defaults for records whose source does not provide
# ratings. These quantiles are calibrated from 30,000 Amazon Reviews 2023 item
# metadata records sampled equally across six product categories. Explicit,
# task-scored ratings in products.json always take precedence.
_RATING_QUANTILES = (
    2.8, 3.3, 3.5, 3.7, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4,
    4.4, 4.5, 4.5, 4.6, 4.6, 4.7, 4.8, 4.9, 5.0, 5.0,
)
_RATING_COUNT_QUANTILES = (
    1, 1, 2, 3, 4, 6, 9, 13, 19, 27,
    38, 55, 82, 125, 190, 300, 520, 950, 2_300, 12_000,
)


def _digest(item_id: str) -> bytes:
    return hashlib.md5(item_id.encode()).digest()


def rating_of(item: dict[str, Any]) -> tuple[float, int]:
    digest = _digest(item["item_id"])
    declared_rating = item.get("rating")
    declared_count = item.get("ratings_count")
    rating = float(
        declared_rating
        if declared_rating is not None
        else _RATING_QUANTILES[digest[0] % len(_RATING_QUANTILES)]
    )
    count = int(
        declared_count
        if declared_count is not None
        else _RATING_COUNT_QUANTILES[
            int.from_bytes(digest[1:3], "big") % len(_RATING_COUNT_QUANTILES)
        ]
    )
    return rating, count


def stars(rating: float) -> str:
    return '<span class="stars" title="' + f"{rating} out of 5 stars" + '">' + \
        "★" * round(rating) + "☆" * (5 - round(rating)) + "</span>"


def price_value(item: dict[str, Any] | None) -> float:
    if item is None:
        return 0.0
    match = re.search(r"\d+(?:\.\d+)?", str(item.get("price", "")))
    return float(match.group()) if match else 0.0


def money_value(value: Any) -> float:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else 0.0


def price_html(price: Any) -> str:
    value = re.search(r"\d+(?:\.\d+)?", str(price or ""))
    if not value:
        return '<span class="muted">Price unavailable</span>'
    dollars, _, cents = f"{float(value.group()):.2f}".partition(".")
    return (
        f'<span class="price"><span class="p-sym">$</span>'
        f'<span class="p-whole">{dollars}</span><span class="p-cents">{cents}</span></span>'
    )


def visible_attributes(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in item.get("attributes", {}).items()
        if key not in INTERNAL_ATTRIBUTES
    }


def searchable_text(item: dict[str, Any]) -> str:
    parts = [item["name"], str(item.get("brand", ""))]
    parts += [str(value) for value in visible_attributes(item).values()]
    return " ".join(parts).casefold()


def search_results(items: list[dict[str, Any]], query: str) -> tuple[list[dict[str, Any]], bool]:
    terms = query.casefold().split()

    def hit(item: dict[str, Any], term: str) -> bool:
        haystack = searchable_text(item)
        singular = term[:-1] if term.endswith("s") else term
        return term in haystack or (len(singular) > 2 and singular in haystack)

    matched = [item for item in items if all(hit(item, term) for term in terms)] if terms else items
    no_match = bool(terms) and not matched
    return matched, no_match


def sorted_results(items: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if sort_key == "price_asc":
        return sorted(items, key=lambda item: (price_value(item), item["item_id"]))
    if sort_key == "price_desc":
        return sorted(items, key=lambda item: (-price_value(item), item["item_id"]))
    if sort_key == "rating_desc":
        return sorted(items, key=lambda item: (-rating_of(item)[0], item["item_id"]))
    return list(items)


def paged_results(
    items: list[dict[str, Any]], page_number: int,
) -> tuple[list[dict[str, Any]], int, int]:
    page_count = max(1, math.ceil(len(items) / RESULTS_PER_PAGE))
    current_page = min(max(page_number, 1), page_count)
    start = (current_page - 1) * RESULTS_PER_PAGE
    return items[start:start + RESULTS_PER_PAGE], current_page, page_count


def _image(item: dict[str, Any], css_class: str = "") -> str:
    item_id = quote(item["item_id"])
    alt = html.escape(item["name"])
    cls = f' class="{css_class}"' if css_class else ""
    return f'<img{cls} src="/product-image/{item_id}" alt="{alt}">'


def item_card(item: dict[str, Any]) -> str:
    rating, count = rating_of(item)
    item_id = item["item_id"]
    badge = item.get("badge")
    badge_html = (
        f'<span class="badge{" badge-pick" if "Pick" in str(badge) else ""}">{html.escape(str(badge))}</span>'
        if badge else ""
    )
    unit = item.get("unit_price")
    unit_html = f'<span class="unit-price">({html.escape(str(unit))})</span>' if unit else ""
    return (
        '<article class="card">'
        f'<div class="card-img">{badge_html}<a href="/product/{quote(item_id)}">{_image(item)}</a></div>'
        f'<a class="card-title" href="/product/{quote(item_id)}">{html.escape(item["name"])}</a>'
        f'<div class="rating-row">{stars(rating)}<a class="count" href="/product/{quote(item_id)}">{count:,}</a></div>'
        f'<div class="price-row">{price_html(item.get("price"))}{unit_html}</div>'
        f'<a class="btn-pill" href="/product/{quote(item_id)}">View product</a>'
        "</article>"
    )


def facet_sidebar(items: list[dict[str, Any]]) -> str:
    brands: list[str] = []
    for item in items:
        brand = str(item.get("brand") or "").strip()
        if brand and brand not in brands:
            brands.append(brand)
    brand_rows = "".join(
        f'<li><a href="/search?q={quote(brand)}"><span class="cb"></span>{html.escape(brand)}</a></li>'
        for brand in brands[:9]
    )
    groups: dict[str, list[str]] = {}
    facet_keys = (
        "product_type", "product type", "color", "listing_color",
        "size", "material", "format",
    )
    for item in items:
        for key, value in visible_attributes(item).items():
            if key not in facet_keys:
                continue
            values = groups.setdefault(str(key), [])
            if str(value) not in values:
                values.append(str(value))
    attribute_sections = ""
    for key, values in list(groups.items())[:4]:
        links = "".join(
            f'<li><a href="/search?q={quote(value)}"><span class="cb"></span>{html.escape(value)}</a></li>'
            for value in values[:7]
        )
        label = key.replace("_", " ").title()
        attribute_sections += f"<h3>{html.escape(label)}</h3><ul>{links}</ul>"
    return (
        '<aside class="facets">'
        + (f"<h3>Browse by brand</h3><ul>{brand_rows}</ul>" if brand_rows else "")
        + attribute_sections
        + "</aside>"
    )


def _search_url(query: str, sort_key: str, page_number: int) -> str:
    return "/search?" + urlencode({"q": query, "sort": sort_key, "page": page_number})


def _pagination(query: str, sort_key: str, current_page: int, page_count: int) -> str:
    if page_count <= 1:
        return ""
    start = max(1, current_page - 2)
    end = min(page_count, current_page + 2)
    links: list[str] = []
    if current_page > 1:
        links.append(
            f'<a href="{html.escape(_search_url(query, sort_key, current_page - 1))}">‹ Previous</a>'
        )
    for number in range(start, end + 1):
        css_class = ' class="page-cur"' if number == current_page else ""
        links.append(
            f'<a{css_class} href="{html.escape(_search_url(query, sort_key, number))}">{number}</a>'
        )
    if current_page < page_count:
        links.append(
            f'<a href="{html.escape(_search_url(query, sort_key, current_page + 1))}">Next ›</a>'
        )
    return '<nav class="pagination">' + "".join(links) + "</nav>"


def search_page(
    query: str,
    shown: list[dict[str, Any]],
    no_match: bool,
    matched_items: list[dict[str, Any]],
    *,
    current_page: int,
    page_count: int,
    sort_key: str,
) -> str:
    query_html = f'<span class="accent">"{html.escape(query)}"</span>' if query else "all items"
    banner = (
        f'<p class="results-note">No exact matches for <b>"{html.escape(query)}"</b>. '
        "Try fewer or different search terms.</p>" if no_match else ""
    )
    cards = "".join(item_card(item) for item in shown)
    total = len(matched_items)
    first = (current_page - 1) * RESULTS_PER_PAGE + 1 if shown else 0
    last = first + len(shown) - 1 if shown else 0
    options = "".join(
        f'<option value="{key}"{" selected" if key == sort_key else ""}>{label}</option>'
        for key, label in SORT_LABELS.items()
    )
    return (
        '<div class="results-bar"><span>'
        f"{first}-{last} of {total} results for {query_html}</span>"
        '<form class="sort" action="/search" method="get">'
        f'<input type="hidden" name="q" value="{html.escape(query)}">'
        f'<label>Sort by: <select name="sort">{options}</select></label>'
        '<button type="submit">Apply</button></form></div>'
        + banner
        + f'<div class="listing">{facet_sidebar(matched_items)}'
        f'<div class="results"><div class="grid">{cards}</div>'
        f'{_pagination(query, sort_key, current_page, page_count)}</div></div>'
    )


def product_page(item: dict[str, Any]) -> str:
    rating, count = rating_of(item)
    item_id = item["item_id"]
    attributes = visible_attributes(item)
    shipping_cost = attributes.get("shipping_cost")
    detail_attributes = {
        key: value for key, value in attributes.items() if key != "shipping_cost"
    }
    bullets = "".join(
        f"<li><b>{html.escape(str(key).replace('_', ' ').title())}:</b> "
        f"{html.escape(str(value))}</li>"
        for key, value in detail_attributes.items()
    ) or "<li>No additional specifications are listed.</li>"
    shipping_html = (
        f'<p class="shipping">Shipping: <b>{html.escape(str(shipping_cost))}</b></p>'
        if shipping_cost is not None else ""
    )
    brand = str(item.get("brand") or "").strip()
    brand_html = (
        f'<a class="brand-link" href="/search?q={quote(brand)}">Visit the {html.escape(brand)} Store</a>'
        if brand else ""
    )
    unit = item.get("unit_price")
    unit_html = f'<span class="unit-price">{html.escape(str(unit))}</span>' if unit else ""
    return (
        '<nav class="crumbs"><a href="/context">Wearer Context</a> › '
        '<a href="/search">All results</a></nav>'
        '<div class="pdp">'
        f'<div class="pdp-rail">{_image(item, "thumb")}</div>'
        f'<div class="pdp-image">{_image(item)}</div>'
        '<div class="pdp-main">'
        f"<h1>{html.escape(item['name'])}</h1>"
        f"{brand_html}"
        f'<div class="rating-row">{rating} {stars(rating)} <a class="count" href="#reviews">{count:,} ratings</a></div>'
        '<hr>'
        f'<div class="price-row">{price_html(item.get("price"))}{unit_html}</div>'
        f"<h2>About this item</h2><ul class=\"bullets\">{bullets}</ul>"
        "</div>"
        '<aside class="buybox">'
        f'<div class="price-row">{price_html(item.get("price"))}</div>'
        f'{shipping_html}'
        '<form method="post" action="/cart/add">'
        f'<input type="hidden" name="item_id" value="{html.escape(item_id)}">'
        '<label class="qty">Quantity: <select name="quantity">'
        "<option>1</option><option>2</option><option>3</option></select></label>"
        '<button class="btn-cart" type="submit">Add to Cart</button>'
        "</form>"
        "</aside></div>"
    )


def cart_page(entries: list[tuple[dict[str, Any], int]]) -> str:
    if not entries:
        return (
            f"<h1>Your {STORE_NAME} Cart is empty</h1>"
            '<p><a href="/search">Browse products</a> or check the '
            '<a href="/context">wearer context</a>.</p>'
        )
    rows = ""
    total_units = 0
    subtotal = 0.0
    shipping_total = 0.0
    for item, quantity in entries:
        total_units += quantity
        subtotal += price_value(item) * quantity
        shipping_total += money_value(
            visible_attributes(item).get("shipping_cost")
        ) * quantity
        item_id = item["item_id"]
        options = "".join(
            f'<option value="{value}"{" selected" if value == quantity else ""}>'
            f'{"0 (Delete)" if value == 0 else value}</option>'
            for value in range(0, 4)
        )
        rows += (
            '<div class="cart-row">'
            f'<span class="cb cb-checked"></span>{_image(item)}'
            '<div class="cart-info">'
            f'<a class="cart-title" href="/product/{quote(item_id)}">{html.escape(item["name"])}</a>'
            '<div class="cart-controls">'
            '<form method="post" action="/cart/update">'
            f'<input type="hidden" name="item_id" value="{html.escape(item_id)}">'
            f'<label>Qty: <select name="quantity" onchange="this.form.submit()">{options}</select></label>'
            '<button class="btn-link" type="submit">Update</button></form>'
            '<form method="post" action="/cart/remove">'
            f'<input type="hidden" name="item_id" value="{html.escape(item_id)}">'
            '<button class="btn-link" type="submit">Delete</button></form>'
            "</div></div>"
            f'<div class="cart-price">{price_html(item.get("price"))}</div>'
            "</div>"
        )
    subtotal_html = (
        f'Subtotal ({total_units} item{"s" if total_units != 1 else ""}): <b>${subtotal:,.2f}</b>'
    )
    totals_html = subtotal_html
    if shipping_total:
        totals_html += (
            f'<br>Shipping: <b>${shipping_total:,.2f}</b>'
            f'<br>Order total: <b>${subtotal + shipping_total:,.2f}</b>'
        )
    return (
        '<div class="cart-layout"><div class="cart-main">'
        "<h1>Shopping Cart</h1>"
        f'{rows}<p class="subtotal">{totals_html}</p></div>'
        '<aside class="cart-box">'
        f'<p class="subtotal">{totals_html}</p>'
        '<form method="post" action="/submit">'
        '<button class="btn-checkout" type="submit">Proceed to checkout</button></form>'
        "</aside></div>"
    )


def submit_blocked_page() -> str:
    return (
        "<h1>Order not placed yet</h1>"
        '<p class="results-note">A new update from the wearer is arriving. '
        "Review your messages and the <a href='/context'>context page</a>, "
        "confirm your selection still matches the latest intent, then check out again.</p>"
        '<p><a href="/cart">Return to cart</a></p>'
    )


def submit_empty_page() -> str:
    return "<h1>Your cart is empty</h1><p>Add the intended item before checking out.</p>"


def confirmation_page(items: list[dict[str, Any]]) -> str:
    rows = "".join(
        f'<div class="cart-row">{_image(item)}<div class="cart-info">'
        f'<span class="cart-title">{html.escape(item["name"])}</span>'
        f'<span class="muted">{html.escape(item["item_id"])}</span></div></div>'
        for item in items
    )
    return (
        '<h1 class="confirm">✓ Order placed, thank you!</h1>'
        '<p class="muted">Confirmation shared with the wearer.</p>'
        + rows
    )


def context_page(observations: list[tuple[int, dict[str, Any]]]) -> str:
    galleries = ""
    for display_index, (observation_index, observation) in enumerate(observations):
        if len(observations) == 1:
            heading = "Wearer's view"
        elif display_index == 0:
            heading = "Earlier view"
        else:
            heading = "More recent view"
        thumbs = "".join(
            f'<a href="/observation/{observation_index}/image/{index}">'
            f'<img src="/observations/{observation_index}/images/{index}" '
            f'alt="wearer view image {index + 1}"></a>'
            for index in range(len(observation["frame_paths"]))
        )
        galleries += (
            f'<section class="observation"><h2>{heading}</h2>'
            f'<div class="frames">{thumbs}</div>'
            '<p class="muted">Click any image to view it at full resolution.</p></section>'
        )
    return (
        "<h1>What the wearer saw</h1>"
        "<p>Images are shown in chronological order.</p>"
        + galleries
    )


def frame_viewer(observation_index: str, image_index: str) -> str:
    return (
        f"<h1>Wearer view · image {int(image_index) + 1}</h1>"
        f'<img class="full-frame" src="/observations/{quote(observation_index)}/images/'
        f'{quote(image_index)}" alt="wearer view at full resolution">'
        '<p><a href="/context">Back to context</a></p>'
    )


def shell(body: str, cart_count: int = 0, query: str = "") -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{STORE_NAME}</title>
<style>{_CSS}</style></head><body>
<header class="topbar">
<a class="logo" href="/context">{STORE_NAME}</a>
<form class="searchbox" action="/search">
<input name="q" value="{html.escape(query)}" placeholder="Search {STORE_NAME}">
<button aria-label="search">🔍</button></form>
<a class="cart-link" href="/cart">🛒<b>Cart</b> <span class="cart-count">({cart_count})</span></a>
</header>
<nav class="subnav"><span class="subnav-all">☰ All</span><a href="/context">Wearer Context</a>
<a href="/search">All Products</a><a href="/cart">Cart</a></nav>
<main>{body}</main>
<footer class="footer">© {STORE_NAME}. Frozen deterministic benchmark environment.</footer>
</body></html>"""


_CSS = """
*{box-sizing:border-box}
body{font-family:Arial,sans-serif;font-size:14px;line-height:20px;color:#0f1111;margin:0;background:#fff}
a{color:#2162a1;text-decoration:none} a:hover{text-decoration:underline;color:#c7511f}
.muted{color:#565959;font-size:12px} .muted-inv{color:#ccc;font-size:12px;display:block}
.topbar{background:#131921;color:#fff;display:flex;align-items:center;gap:14px;padding:6px 16px}
.logo{color:#fff;font-size:22px;font-weight:700;letter-spacing:.5px;padding:4px 8px;border:1px solid transparent;border-radius:3px}
.logo:hover{border-color:#fff;color:#fff;text-decoration:none}
.searchbox{flex:1;display:flex;max-width:780px;height:40px;border-radius:6px;overflow:hidden}
.searchbox input{flex:1;padding:0 10px;border:none;font-size:15px}
.searchbox button{background:#febd69;border:none;padding:0 14px;font-size:17px;cursor:pointer}
.searchbox button:hover{background:#f3a847}
.cart-link{color:#fff;display:flex;align-items:flex-end;gap:2px;font-size:15px}
.cart-link:hover{color:#febd69;text-decoration:none}.cart-count{color:#f08804;font-weight:700}
.subnav{background:#232f3e;color:#fff;padding:6px 16px;font-size:14px;display:flex;gap:18px;align-items:center}
.subnav a{color:#ddd}.subnav a:hover{color:#fff}.subnav-all{font-weight:700}
main{max-width:1440px;margin:0 auto;padding:12px 20px;min-height:70vh}
h1{font-size:24px;line-height:32px;font-weight:400}h2{font-size:17px;font-weight:700}
h3{font-size:14px;font-weight:700;margin:12px 0 4px}
.accent{color:#c45500;font-weight:700}
.results-bar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e7e7e7;padding:4px 0 8px;font-size:14px}
.sort{display:flex;align-items:center;gap:6px}.sort select,.sort button{border:1px solid #d5d9d9;border-radius:8px;padding:4px 7px;background:#f0f2f2;font-size:13px}
.results-note{border-bottom:1px solid #e7e7e7;padding-bottom:6px}
.listing{display:flex;gap:24px;margin-top:10px}
.facets{width:220px;flex-shrink:0;font-size:14px}
.facets ul{list-style:none;padding:0;margin:0 0 10px}.facets li{margin:3px 0;display:flex;align-items:center;gap:6px}
.facets a{color:#0f1111;display:flex;align-items:center;gap:6px}
.cb{width:14px;height:14px;border:1px solid #888c8c;border-radius:3px;display:inline-block;background:#fff;flex-shrink:0}
.cb-checked{background:#007185;border-color:#007185;position:relative}
.cb-checked::after{content:"✓";color:#fff;font-size:11px;position:absolute;top:-3px;left:2px}
.results{flex:1}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:16px}
.card{border:1px solid #e7e7e7;border-radius:8px;padding:0 0 12px;display:flex;flex-direction:column;gap:3px;background:#fff;overflow:hidden}
.card-img{position:relative;background:#f7f7f7;padding:14px}
.card-img img{width:100%;aspect-ratio:1;object-fit:contain;mix-blend-mode:multiply}
.badge{position:absolute;top:8px;left:0;background:#232f3e;color:#fff;font-size:11px;padding:3px 8px;border-radius:0 3px 3px 0}
.badge-pick{background:#232f3e}
.card>*:not(.card-img){margin:0 12px}
.card-title{color:#0f1111;font-size:14px;line-height:19px;max-height:57px;overflow:hidden}
.card-title:hover{color:#c7511f}
.stars{color:#de7921;letter-spacing:1px;font-size:14px}
.rating-row{display:flex;align-items:center;gap:5px}.count{font-size:13px;color:#2162a1}
.price-row{display:flex;align-items:baseline;gap:6px}
.price{white-space:nowrap}.p-sym{font-size:12px;vertical-align:top}
.p-whole{font-size:24px;font-weight:500}.p-cents{font-size:12px;vertical-align:top}
.unit-price{color:#565959;font-size:12px}
.btn-pill{background:#ffd814;border:1px solid #fcd200;border-radius:100px;padding:7px 0;width:calc(100% - 24px);font-size:13px;cursor:pointer;margin-top:6px;text-align:center;color:#0f1111}
.btn-pill:hover{background:#f7ca00}
.pagination{display:flex;gap:0;justify-content:center;margin:22px 0;border:1px solid #d5d9d9;border-radius:8px;width:fit-content;margin-inline:auto}
.pagination a{padding:8px 14px;border-right:1px solid #e7e7e7;color:#2162a1}
.pagination a:last-child{border-right:none}.pagination .page-cur{font-weight:700;color:#0f1111;border:1px solid #0f1111;margin:-1px}
.crumbs{font-size:12px;color:#565959;margin-bottom:8px}
.pdp{display:flex;gap:22px;flex-wrap:wrap}
.pdp-rail{width:56px;flex-shrink:0;display:flex;flex-direction:column;gap:8px}
.pdp-rail .thumb{width:56px;height:56px;object-fit:contain;border:2px solid #2162a1;border-radius:6px;background:#f7f7f7;padding:3px}
.pdp-image{width:min(420px,44vw);flex-shrink:0}
.pdp-image img{width:100%;aspect-ratio:1;object-fit:contain;background:#fff}
.pdp-main{flex:1;min-width:280px}
.pdp-main h1{font-size:23px;line-height:30px;margin:0 0 2px}
.brand-link{font-size:13px}
.pdp-main hr{border:none;border-top:1px solid #e7e7e7;margin:10px 0}
.bullets{padding-left:20px}.bullets li{margin:3px 0}
.buybox{width:260px;flex-shrink:0;border:1px solid #d5d9d9;border-radius:8px;padding:16px;height:fit-content;font-size:14px}
.qty{display:block;margin:8px 0}
.qty select,.cart-controls select{border:1px solid #d5d9d9;border-radius:8px;padding:3px 8px;background:#f0f2f2;cursor:pointer}
.btn-cart{width:100%;background:#ffd814;border:1px solid #fcd200;border-radius:100px;padding:9px 0;font-size:14px;cursor:pointer;margin:5px 0}
.btn-cart:hover{background:#f7ca00}
.cart-layout{display:flex;gap:22px;flex-wrap:wrap-reverse;justify-content:flex-end}
.cart-main{flex:1;background:#fff;padding:16px;border-radius:8px}
.cart-main h1{border-bottom:1px solid #e7e7e7;padding-bottom:10px;margin-bottom:0}
.cart-row{display:flex;align-items:flex-start;gap:14px;border-bottom:1px solid #e7e7e7;padding:14px 0}
.cart-row>img{width:160px;height:160px;object-fit:contain;background:#fff;flex-shrink:0}
.cart-info{flex:1;display:flex;flex-direction:column;gap:3px}
.cart-title{color:#0f1111;font-size:16px;line-height:22px}
.cart-title:hover{color:#c7511f}
.cart-controls{display:flex;align-items:center;gap:12px;margin-top:8px;font-size:12px}
.cart-controls form{display:inline}
.btn-link{background:none;border:none;color:#2162a1;font-size:12px;cursor:pointer;padding:0}
.btn-link:hover{text-decoration:underline}
.cart-price{font-size:18px;font-weight:700}
.subtotal{font-size:18px;text-align:right}
.cart-box{width:280px;flex-shrink:0;border:1px solid #d5d9d9;border-radius:8px;padding:18px;height:fit-content}
.cart-box .subtotal{text-align:left;margin-top:0}
.btn-checkout{width:100%;background:#ffd814;border:1px solid #fcd200;border-radius:100px;padding:9px 0;font-size:14px;cursor:pointer}
.btn-checkout:hover{background:#f7ca00}
.observation{margin-bottom:16px}
.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.frames img{width:100%;border-radius:4px}
.full-frame{width:100%;max-width:1280px}
.confirm{color:#007600}
.footer{background:#232f3e;color:#ccc;text-align:center;padding:18px;font-size:12px;margin-top:32px}
"""
