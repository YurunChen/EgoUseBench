"""Amazon storefront pages.

Server-rendered HTML matching Amazon search, product-detail, cart, and checkout
chrome. Sign-in is omitted. Internal site id remains `egomart`.
"""

from __future__ import annotations

import html
import math
import re
from typing import Any
from urllib.parse import quote, urlencode

from sites.search import flatten_public_text, rank_records

STORE_NAME = "Amazon"
WORDMARK = "amazon"
PAGE_TITLE = "Amazon.com"
_AMAZON_MARK = (
    '<svg class="smile" viewBox="0 0 100 12" aria-hidden="true">'
    '<path d="M4 2c22 12 70 12 92-1" fill="none" stroke="#ff9900" stroke-width="3" '
    'stroke-linecap="round"/>'
    '<path d="M90 1l8 4-6 5" fill="none" stroke="#ff9900" stroke-width="3" stroke-linecap="round"/>'
    "</svg>"
)
_SEARCH_ICON = (
    '<svg class="search-icon" viewBox="0 0 24 24" aria-hidden="true">'
    '<circle cx="10.5" cy="10.5" r="6.25" fill="none" stroke="#111" stroke-width="2.4"/>'
    '<path d="M15.2 15.2L21 21" fill="none" stroke="#111" stroke-width="2.4" '
    'stroke-linecap="round"/>'
    "</svg>"
)
_CART_ICON = (
    '<svg class="cart-glyph" viewBox="0 0 38 32" aria-hidden="true">'
    '<path d="M2 4h5l3.2 16.5h18.6L32 10H10" fill="none" stroke="#fff" stroke-width="2.2" '
    'stroke-linejoin="round"/>'
    '<circle cx="15.5" cy="28" r="2.4" fill="#fff"/>'
    '<circle cx="27.5" cy="28" r="2.4" fill="#fff"/>'
    "</svg>"
)
INTERNAL_ATTRIBUTES = {
    "annotation",
    "rank",
    "delivery_days",
    "dimensions",
    "search_aliases",
    "listing_source",
    "listing_type",
}
_SCRAPE_NOISE_ATTRIBUTES = frozenset(
    {
        "customer reviews",
        "customer reviews ‏",
        "best sellers rank",
        "asin",
    }
)
RESULTS_PER_PAGE = 48
SORT_LABELS = {
    "featured": "Featured",
    "price_asc": "Price: Low to High",
    "price_desc": "Price: High to Low",
    "rating_desc": "Avg. Customer Review",
}


def rating_of(item: dict[str, Any]) -> tuple[float | None, int | None]:
    declared_rating = item.get("rating")
    declared_count = item.get("ratings_count")
    return (
        float(declared_rating) if declared_rating is not None else None,
        int(declared_count) if declared_count is not None else None,
    )


def reviews_of(item: dict[str, Any]) -> list[dict[str, Any]]:
    declared = item.get("reviews")
    if isinstance(declared, list) and declared:
        return [review for review in declared if isinstance(review, dict)]
    return []


def delivery_days_of(item: dict[str, Any]) -> int | None:
    declared = item.get("attributes", {}).get("delivery_days")
    if declared is not None:
        try:
            days = int(declared)
        except (TypeError, ValueError):
            days = 0
        if days > 0:
            return days
    return None


def dimensions_of(item: dict[str, Any]) -> str | None:
    attributes = item.get("attributes", {})
    declared = attributes.get("dimensions")
    if declared:
        return str(declared)
    return None


def stars(rating: float) -> str:
    return (
        '<span class="stars" title="'
        + f"{rating} out of 5 stars"
        + '">'
        + "★" * round(rating)
        + "☆" * (5 - round(rating))
        + "</span>"
    )


def _reviews_block(item: dict[str, Any]) -> str:
    reviews = reviews_of(item)
    if not reviews:
        return ""
    rows = ""
    for review in reviews:
        review_rating = float(review.get("rating", 0))
        rows += (
            '<article class="review">'
            f'<div class="review-head">{html.escape(str(review.get("author", "Customer")))} '
            f"{review_rating:g} {stars(review_rating)} · "
            f"{html.escape(str(review.get('date', '')))}</div>"
            f'<div class="review-title"><b>{html.escape(str(review.get("title", "")))}</b></div>'
            f'<p class="review-text">{html.escape(str(review.get("text", "")))}</p>'
            "</article>"
        )
    return f'<section id="reviews" class="reviews"><h2>Customer reviews</h2>{rows}</section>'


def price_value(item: dict[str, Any] | None) -> float | None:
    if item is None:
        return None
    return money_value(item.get("price"))


def money_value(value: Any) -> float | None:
    match = re.search(r"\d[\d,]*(?:\.\d+)?", str(value or ""))
    return float(match.group().replace(",", "")) if match else None


def money_currency(value: Any) -> str:
    label = str(value or "")
    number = re.search(r"\d", label)
    return label[: number.start()].strip() or "$" if number else "$"


def price_html(price: Any) -> str:
    amount = money_value(price)
    if amount is None:
        return '<span class="muted">Price unavailable</span>'
    dollars, _, cents = f"{amount:.2f}".partition(".")
    currency = html.escape(money_currency(price))
    return (
        f'<span class="price"><span class="p-sym">{currency}</span>'
        f'<span class="p-whole">{dollars}</span><span class="p-cents">{cents}</span></span>'
    )


def visible_attributes(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.get("attributes", {}).items()
        if key not in INTERNAL_ATTRIBUTES
        and key.casefold().strip() not in _SCRAPE_NOISE_ATTRIBUTES
    }


def in_stock(item: dict[str, Any]) -> bool:
    return item.get("attributes", {}).get("Availability") != "out_of_stock"


def brand_of(item: dict[str, Any]) -> str:
    attributes = visible_attributes(item)
    return str(
        item.get("brand") or attributes.get("Brand") or attributes.get("brand") or ""
    ).strip()


def searchable_text(item: dict[str, Any]) -> str:
    parts = [item["name"], brand_of(item)]
    parts += [str(value) for value in visible_attributes(item).values()]
    aliases = item.get("attributes", {}).get("search_aliases", [])
    if isinstance(aliases, list):
        parts += [str(alias) for alias in aliases if str(alias).strip()]
    return " ".join(parts).casefold()


def searchable_fields(item: dict[str, Any]) -> list[tuple[str, float]]:
    aliases = item.get("attributes", {}).get("search_aliases", [])
    return [
        (str(item.get("name", "")), 3.0),
        (brand_of(item), 2.0),
        (flatten_public_text(visible_attributes(item)), 1.5),
        (flatten_public_text(aliases), 1.0),
    ]


def search_results(
    items: list[dict[str, Any]], query: str
) -> tuple[list[dict[str, Any]], bool]:
    if not query.strip():
        return list(items), False
    matched = rank_records(items, query, searchable_fields)
    no_match = not matched
    return matched, no_match


def sorted_results(items: list[dict[str, Any]], sort_key: str) -> list[dict[str, Any]]:
    if sort_key in {"price_asc", "price_desc"}:

        def price_key(item: dict[str, Any]) -> tuple[bool, float, str]:
            amount = price_value(item)
            if amount is None:
                return True, 0.0, item["item_id"]
            return (
                False,
                -amount if sort_key == "price_desc" else amount,
                item["item_id"],
            )

        return sorted(items, key=price_key)
    if sort_key == "rating_desc":
        return sorted(
            items,
            key=lambda item: (
                rating_of(item)[0] is None,
                -(rating_of(item)[0] or 0.0),
                item["item_id"],
            ),
        )
    return list(items)


def paged_results(
    items: list[dict[str, Any]],
    page_number: int,
) -> tuple[list[dict[str, Any]], int, int]:
    page_count = max(1, math.ceil(len(items) / RESULTS_PER_PAGE))
    current_page = min(max(page_number, 1), page_count)
    start = (current_page - 1) * RESULTS_PER_PAGE
    return items[start : start + RESULTS_PER_PAGE], current_page, page_count


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
        if badge
        else ""
    )
    unit = item.get("unit_price")
    unit_html = (
        f'<span class="unit-price">({html.escape(str(unit))})</span>' if unit else ""
    )
    rating_html = (
        f'<div class="rating-row">{stars(rating)}<a class="count" href="/product/{quote(item_id)}">{count:,}</a></div>'
        if rating is not None and count is not None
        else ""
    )
    availability_html = (
        "" if in_stock(item) else '<p role="status">Currently unavailable</p>'
    )
    price = item.get("price")
    price_row = (
        f'<div class="price-row">{price_html(price)}{unit_html}</div>'
        if money_value(price) is not None
        else ""
    )
    return (
        '<article class="card">'
        f'<div class="card-img">{badge_html}<a href="/product/{quote(item_id)}">{_image(item)}</a></div>'
        f'<a class="card-title" href="/product/{quote(item_id)}">{html.escape(item["name"])}</a>'
        f"{rating_html}"
        f"{price_row}"
        f"{availability_html}"
        f'<a class="btn-pill" href="/product/{quote(item_id)}">View product</a>'
        "</article>"
    )


def related_search_sidebar(items: list[dict[str, Any]]) -> str:
    brands: list[str] = []
    for item in items:
        brand = brand_of(item)
        if brand and brand not in brands:
            brands.append(brand)
    brand_rows = "".join(
        f'<li><a href="/search?q={quote(brand)}">Search {html.escape(brand)}</a></li>'
        for brand in brands[:9]
    )
    groups: dict[str, list[str]] = {}
    facet_keys = (
        "product_type",
        "product type",
        "color",
        "listing_color",
        "size",
        "material",
        "format",
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
            f'<li><a href="/search?q={quote(value)}">Search {html.escape(value)}</a></li>'
            for value in values[:7]
        )
        label = key.replace("_", " ").title()
        attribute_sections += f"<h3>{html.escape(label)}</h3><ul>{links}</ul>"
    return (
        '<aside class="facets"><h2>Related searches</h2><p>Starts a new search</p>'
        + (f"<h3>Search by brand</h3><ul>{brand_rows}</ul>" if brand_rows else "")
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


def home_page(items: list[dict[str, Any]]) -> str:
    categories = (
        ("Grocery essentials", "tortillas", "Everyday food and pantry staples."),
        ("Kitchen and dining", "spatula", "Cookware, utensils, and tableware."),
        ("Tools and home improvement", "drill", "Hardware and repair supplies."),
        ("Books and learning", "notebook", "Books, paper, and study supplies."),
    )
    cards = "".join(
        _home_category_card(items, name, query, blurb)
        for name, query, blurb in categories
    )
    return (
        '<section class="home-hero"><h1>Find what you need</h1>'
        "<p>Search by product, brand, feature, or use.</p></section>"
        f'<div class="home-deck">{cards}</div>'
    )


def _home_category_card(
    items: list[dict[str, Any]], name: str, query: str, blurb: str
) -> str:
    sample = next(
        (
            item
            for item in items
            if query in searchable_text(item) and money_value(item.get("price")) is not None
        ),
        None,
    )
    preview = (
        f'<a class="home-thumb" href="/product/{quote(sample["item_id"])}">{_image(sample)}</a>'
        if sample is not None
        else ""
    )
    return (
        f'<article class="home-card home-category">{preview}'
        f"<h2>{html.escape(name)}</h2>"
        f"<p>{html.escape(blurb)}</p>"
        f'<a class="home-more" href="/search?q={quote(query)}">Shop {html.escape(name)}</a>'
        "</article>"
    )


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
    query_html = (
        f'<span class="accent">"{html.escape(query)}"</span>' if query else "all items"
    )
    banner = (
        f'<p class="results-note">No exact matches for <b>"{html.escape(query)}"</b>. '
        "Try fewer or different search terms.</p>"
        if no_match
        else ""
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
        f'<label>Sort by: <select name="sort" onchange="this.form.submit()">{options}</select></label>'
        '<button type="submit">Go</button></form></div>'
        + banner
        + f'<div class="listing">{related_search_sidebar(matched_items)}'
        f'<div class="results"><h2 class="results-heading">Results</h2>'
        f'<div class="grid">{cards}</div>'
        f"{_pagination(query, sort_key, current_page, page_count)}</div></div>"
    )


def product_page(item: dict[str, Any]) -> str:
    rating, count = rating_of(item)
    item_id = item["item_id"]
    attributes = visible_attributes(item)
    shipping_cost = attributes.get("shipping_cost")
    detail_attributes = {
        key: value for key, value in attributes.items() if key != "shipping_cost"
    }
    features = [
        value
        for key, value in detail_attributes.items()
        if re.fullmatch(r"feature_\d+", key)
    ]
    specifications = {
        key: value
        for key, value in detail_attributes.items()
        if not re.fullmatch(r"feature_\d+", key)
    }
    bullets = "".join(f"<li>{html.escape(str(value))}</li>" for value in features)
    specs = "".join(
        f"<li><b>{html.escape(str(key).replace('_', ' ').title())}:</b> "
        f"{html.escape(str(value))}</li>"
        for key, value in specifications.items()
    )
    about_html = (
        f'<h2>About this item</h2><ul class="bullets">{bullets}</ul>' if bullets else ""
    )
    specs_html = (
        f'<h2>Product information</h2><ul class="bullets">{specs}</ul>' if specs else ""
    )
    shipping_html = (
        f'<p class="shipping">Shipping: <b>{html.escape(str(shipping_cost))}</b></p>'
        if shipping_cost is not None
        else ""
    )
    delivery_days = delivery_days_of(item)
    delivery_html = (
        f'<p class="delivery">Arrives in <b>{delivery_days}</b> days</p>'
        if delivery_days is not None
        else ""
    )
    brand = brand_of(item)
    brand_html = (
        f'<a class="brand-link" href="/search?q={quote(brand)}">Visit the {html.escape(brand)} Store</a>'
        if brand
        else ""
    )
    unit = item.get("unit_price")
    unit_html = (
        f'<span class="unit-price">{html.escape(str(unit))}</span>' if unit else ""
    )
    dimensions = dimensions_of(item)
    rating_html = (
        f'<div class="rating-row">{rating} {stars(rating)} '
        f'<a class="count" href="#reviews">{count:,} ratings</a></div>'
        if rating is not None and count is not None
        else ""
    )
    dimensions_html = (
        f'<h2>Product details</h2><ul class="bullets">'
        f"<li><b>Dimensions:</b> {html.escape(dimensions)}</li></ul>"
        if dimensions
        else ""
    )
    return (
        '<nav class="crumbs"><a href="/search">All results</a></nav>'
        '<div class="pdp">'
        f'<div class="pdp-rail">{_image(item, "thumb")}</div>'
        f'<div class="pdp-image">{_image(item)}</div>'
        '<div class="pdp-main">'
        f"<h1>{html.escape(item['name'])}</h1>"
        f"{brand_html}"
        f"{rating_html}"
        "<hr>"
        f'<div class="price-row">{price_html(item.get("price"))}{unit_html}</div>'
        f"{about_html}{specs_html}"
        f"{dimensions_html}"
        f"{_reviews_block(item)}"
        "</div>"
        '<aside class="buybox">'
        f'<div class="price-row">{price_html(item.get("price"))}</div>'
        f"{delivery_html}"
        f"{shipping_html}"
        '<form method="post" action="/cart/add">'
        f'<input type="hidden" name="item_id" value="{html.escape(item_id)}">'
        '<label class="qty">Quantity: <input type="number" name="quantity" min="1" step="1" value="1" required></label>'
        + (
            '<button class="btn-cart" type="submit">Add to Cart</button>'
            if in_stock(item)
            else '<button class="btn-cart" type="submit" disabled>Currently unavailable</button>'
        )
        + "</form>"
        "</aside></div>"
    )


def cart_page(entries: list[tuple[dict[str, Any], int]]) -> str:
    if not entries:
        return (
            f'<div class="cart-empty"><h1>Your {STORE_NAME} Cart is empty</h1>'
            '<p class="muted"><a href="/search">Shop today\'s deals</a></p>'
            '<a class="btn-continue" href="/search">Continue shopping</a></div>'
        )
    rows = ""
    total_units = 0
    subtotals: dict[str, float] = {}
    shipping_totals: dict[str, float] = {}
    unknown_price = False
    unknown_shipping = False
    for item, quantity in entries:
        total_units += quantity
        currency = money_currency(item.get("price"))
        amount = price_value(item)
        if amount is None:
            unknown_price = True
        else:
            subtotals[currency] = subtotals.get(currency, 0.0) + amount * quantity
        shipping_cost = visible_attributes(item).get("shipping_cost")
        shipping_amount = (
            money_value(shipping_cost) if shipping_cost is not None else 0.0
        )
        if shipping_amount is None:
            unknown_shipping = True
        else:
            shipping_totals[currency] = (
                shipping_totals.get(currency, 0.0) + shipping_amount * quantity
            )
        item_id = item["item_id"]
        rows += (
            '<div class="cart-row">'
            f'<span class="cb cb-checked"></span>{_image(item)}'
            '<div class="cart-info">'
            f'<a class="cart-title" href="/product/{quote(item_id)}">{html.escape(item["name"])}</a>'
            '<div class="cart-controls">'
            '<form method="post" action="/cart/update">'
            f'<input type="hidden" name="item_id" value="{html.escape(item_id)}">'
            f'<label>Qty: <input type="number" name="quantity" min="0" step="1" value="{quantity}" required></label>'
            '<button class="btn-link" type="submit">Update</button></form>'
            '<form method="post" action="/cart/remove">'
            f'<input type="hidden" name="item_id" value="{html.escape(item_id)}">'
            '<button class="btn-link" type="submit">Delete</button></form>'
            "</div></div>"
            f'<div class="cart-price">{price_html(item.get("price"))}</div>'
            "</div>"
        )
    totals_html = f"Subtotal ({total_units} item{'s' if total_units != 1 else ''}): "
    if unknown_price:
        totals_html += (
            "<b>Price unavailable</b><br>Order total: <b>Price unavailable</b>"
        )
    else:
        totals_html += " + ".join(
            f"<b>{html.escape(currency)}{subtotal:,.2f}</b>"
            for currency, subtotal in subtotals.items()
        )
        if unknown_shipping:
            totals_html += "<br>Shipping: <b>Price unavailable</b><br>Order total: <b>Price unavailable</b>"
        else:
            for currency, shipping_total in shipping_totals.items():
                if shipping_total:
                    symbol = html.escape(currency)
                    totals_html += (
                        f"<br>Shipping: <b>{symbol}{shipping_total:,.2f}</b>"
                        f"<br>Order total: <b>{symbol}{subtotals[currency] + shipping_total:,.2f}</b>"
                    )
    return (
        '<div class="cart-layout"><div class="cart-main">'
        '<div class="cart-head"><h1>Shopping Cart</h1><span class="cart-price-label">Price</span></div>'
        f'{rows}<p class="subtotal">{totals_html}</p></div>'
        '<aside class="cart-box">'
        f'<p class="subtotal">{totals_html}</p>'
        '<p role="status">Cart changes are saved.</p>'
        "</aside></div>"
    )


def submit_empty_page() -> str:
    return (
        "<h1>Your cart is empty</h1><p>Add the intended item before checking out.</p>"
    )


def confirmation_page(items: list[dict[str, Any]]) -> str:
    rows = "".join(
        f'<div class="cart-row">{_image(item)}<div class="cart-info">'
        f'<span class="cart-title">{html.escape(item["name"])}</span>'
        f'<span class="muted">{html.escape(item["item_id"])}</span></div></div>'
        for item in items
    )
    return (
        '<h1 class="confirm">✓ Order placed, thank you!</h1>'
        '<p class="muted">Confirmation shared with the wearer.</p>' + rows
    )


_SUBNAV_LINKS = (
    "Best Sellers",
    "Today's Deals",
    "Gift Cards",
    "Registry",
    "Books",
)


def _subnav_html() -> str:
    links = "".join(
        f'<a href="/search">{html.escape(label)}</a>' for label in _SUBNAV_LINKS
    )
    return f'<nav class="subnav"><a class="subnav-all" href="/search">☰ All</a>{links}</nav>'


def shell(body: str, cart_count: int = 0, query: str = "", skin: str = "") -> str:
    cart_badge = str(cart_count) if cart_count else "0"
    title = f"{PAGE_TITLE} : {query}" if query else PAGE_TITLE
    body_class = f' class="{html.escape(skin)}"' if skin else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{_CSS}</style></head><body{body_class}>
<header class="topbar">
<a class="logo" href="/">{_AMAZON_MARK}{WORDMARK}</a>
<form class="searchbox" action="/search" method="get">
<span class="search-cat" aria-hidden="true">All</span>
<input name="q" value="{html.escape(query)}" placeholder="Search {STORE_NAME}" aria-label="Search">
<button type="submit" aria-label="Go">{_SEARCH_ICON}</button>
</form>
<a class="nav-orders" href="/cart"><span class="nav-line1">Returns</span><span class="nav-line2">&amp; Orders</span></a>
<a class="cart-link" href="/cart">{_CART_ICON}<span class="cart-count">{cart_badge}</span><span class="cart-label">Cart</span></a>
</header>
{_subnav_html()}
<main>{body}</main>
<footer class="footer"><a class="back-top" href="#">Back to top</a>
<div class="footer-grid">
<div><h4>Get to Know Us</h4><a href="/search">About {STORE_NAME}</a></div>
<div><h4>Make Money with Us</h4><a href="/search">Sell products</a></div>
<div><h4>Let Us Help You</h4><a href="/search">Help</a><a href="/cart">Your Orders</a></div>
</div>
<p class="footer-copy">© 1996-2026, Amazon.com, Inc. or its affiliates</p></footer>
</body></html>"""


_CSS = """
*{box-sizing:border-box}
body{font-family:Arial,sans-serif;font-size:14px;line-height:20px;color:#0f1111;margin:0;background:#fff}
body.home-page{background:#e3e6e6}
a{color:#2162a1;text-decoration:none} a:hover{text-decoration:underline;color:#c7511f}
.muted{color:#565959;font-size:12px}
.topbar{background:#131921;color:#fff;display:flex;align-items:center;gap:12px;padding:4px 16px;min-height:60px}
.logo{color:#fff;font-size:22px;font-weight:700;letter-spacing:.5px;padding:4px 8px;border:1px solid transparent;border-radius:3px;display:flex;flex-direction:column;align-items:flex-start;line-height:1;flex-shrink:0}
.logo:hover{border-color:#fff;color:#fff;text-decoration:none}
.smile{width:86px;height:12px;margin-top:2px}
.searchbox{flex:1;display:flex;max-width:880px;height:40px;border-radius:4px;overflow:hidden}
.search-cat{background:#e6e6e6;color:#0f1111;font-size:12px;line-height:40px;padding:0 10px;border-right:1px solid #cdcdcd;white-space:nowrap;flex-shrink:0}
.searchbox input{flex:1;padding:0 12px;border:none;font-size:15px;color:#0f1111;min-width:0}
.searchbox button{background:#febd69;border:none;width:45px;cursor:pointer;display:grid;place-items:center;flex-shrink:0}
.searchbox button:hover{background:#f3a847}
.search-icon{width:22px;height:22px;display:block}
.nav-orders{color:#fff;display:flex;flex-direction:column;line-height:1.15;padding:6px 8px;border:1px solid transparent;border-radius:3px;white-space:nowrap}
.nav-orders:hover{border-color:#fff;color:#fff;text-decoration:none}
.nav-line1{font-size:12px;color:#ccc}.nav-line2{font-size:14px;font-weight:700}
.cart-link{color:#fff;display:flex;align-items:flex-end;gap:2px;padding:6px 8px 4px;border:1px solid transparent;border-radius:3px;position:relative;min-width:72px}
.cart-link:hover{border-color:#fff;color:#fff;text-decoration:none}
.cart-glyph{width:38px;height:32px;display:block}
.cart-label{font-size:14px;font-weight:700;line-height:1;padding-bottom:3px}
.cart-count{position:absolute;left:22px;top:2px;color:#f08804;font-size:16px;font-weight:700;line-height:1}
.subnav{background:#232f3e;color:#fff;padding:0 16px;font-size:14px;display:flex;gap:16px;align-items:center;min-height:39px;overflow-x:auto}
.subnav a{color:#fff;white-space:nowrap;padding:8px 0}.subnav a:hover{color:#fff;text-decoration:underline}
.subnav-all{font-weight:700;padding:8px 0;white-space:nowrap}
main{max-width:1500px;margin:0 auto;padding:12px 18px;min-height:70vh}
body.home-page main{max-width:1480px;padding:14px 18px 48px}
.home-deck{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:20px}
.home-card{background:#fff;border-radius:8px;padding:16px 18px 14px;box-shadow:0 2px 5px rgba(15,17,17,.08);min-height:420px;display:flex;flex-direction:column}
.home-hero{background:linear-gradient(110deg,#fff 0%,#f7ca00 140%);padding:32px;margin-bottom:20px;border-radius:8px}.home-hero h1{font-size:32px;margin:0 0 8px}.home-hero p{font-size:17px;margin:0}.home-category{min-height:180px}.home-category p{color:#565959;line-height:1.5;flex:1}
.home-thumb{display:block;margin:0 0 12px}.home-thumb img{width:100%;height:160px;object-fit:contain;background:#fff}
.home-card h2{font-size:21px;line-height:27px;font-weight:700;margin:0 0 12px}
.home-quad{display:grid;grid-template-columns:1fr 1fr;gap:12px 10px;flex:1}
.home-cell{color:#0f1111;font-size:12px;line-height:16px;display:flex;flex-direction:column;gap:6px}
.home-cell img{width:100%;aspect-ratio:1;object-fit:contain;background:#fff}
.home-cell span{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.home-more{margin-top:12px;font-size:13px;color:#2162a1}
h1{font-size:24px;line-height:32px;font-weight:400;margin:0 0 8px}h2{font-size:17px;font-weight:700;margin:0 0 8px}
h3{font-size:14px;font-weight:700;margin:12px 0 4px}
.accent{color:#c45500;font-weight:700}
.results-bar{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #e7e7e7;padding:4px 0 8px;font-size:14px;flex-wrap:wrap;gap:8px}
.sort{display:flex;align-items:center;gap:6px}.sort select,.sort button{border:1px solid #d5d9d9;border-radius:8px;padding:4px 7px;background:#f0f2f2;font-size:13px;cursor:pointer}
.results-note{border-bottom:1px solid #e7e7e7;padding-bottom:6px;margin:0 0 8px}
.listing{display:flex;gap:20px;margin-top:8px;align-items:flex-start}
.facets{width:220px;flex-shrink:0;font-size:14px;border-right:1px solid #e7e7e7;padding-right:12px}
.facets ul{list-style:none;padding:0;margin:0 0 10px}.facets li{margin:4px 0;display:flex;align-items:center;gap:6px}
.facets a{color:#0f1111;display:flex;align-items:center;gap:6px;font-size:13px}
.results{flex:1;min-width:0}
.results-heading{font-size:21px;line-height:28px;font-weight:700;margin:0 0 12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:18px 14px}
.card{display:flex;flex-direction:column;gap:4px;background:#fff;padding-bottom:8px}
.card-img{position:relative;background:#fff;padding:8px 0}
.card-img img{width:100%;aspect-ratio:1;object-fit:contain}
.badge{position:absolute;top:0;left:0;background:#232f3e;color:#fff;font-size:11px;padding:3px 8px;border-radius:0 3px 3px 0;z-index:1}
.badge-pick{background:#232f3e}
.card-title{color:#0f1111;font-size:14px;line-height:20px;max-height:60px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical}
.card-title:hover{color:#c7511f}
.stars{color:#de7921;letter-spacing:1px;font-size:14px}
.rating-row{display:flex;align-items:center;gap:5px;flex-wrap:wrap}.count{font-size:13px;color:#2162a1}
.price-row{display:flex;align-items:baseline;gap:6px;flex-wrap:wrap}
.price{white-space:nowrap;color:#0f1111}.p-sym{font-size:12px;vertical-align:top;position:relative;top:1px}
.p-whole{font-size:21px;font-weight:400;line-height:1}.p-cents{font-size:12px;vertical-align:top;position:relative;top:1px}
.unit-price{color:#565959;font-size:12px}
.btn-pill{background:#ffd814;border:1px solid #fcd200;border-radius:100px;padding:7px 0;width:100%;max-width:220px;font-size:13px;cursor:pointer;margin-top:4px;text-align:center;color:#0f1111;display:inline-block}
.btn-pill:hover{background:#f7ca00;text-decoration:none;color:#0f1111}
.pagination{display:flex;gap:0;justify-content:center;margin:22px 0;border:1px solid #d5d9d9;border-radius:8px;width:fit-content;margin-inline:auto;background:#fff}
.pagination a{padding:8px 14px;border-right:1px solid #e7e7e7;color:#2162a1}
.pagination a:last-child{border-right:none}.pagination .page-cur{font-weight:700;color:#0f1111;border:1px solid #0f1111;margin:-1px;border-radius:4px}
.crumbs{font-size:12px;color:#565959;margin-bottom:8px}
.pdp{display:flex;gap:22px;flex-wrap:wrap;align-items:flex-start}
.pdp-rail{width:56px;flex-shrink:0;display:flex;flex-direction:column;gap:8px}
.pdp-rail .thumb{width:56px;height:56px;object-fit:contain;border:2px solid #2162a1;border-radius:6px;background:#f7f7f7;padding:3px}
.pdp-image{width:min(420px,44vw);flex-shrink:0}
.pdp-image img{width:100%;aspect-ratio:1;object-fit:contain;background:#fff}
.pdp-main{flex:1;min-width:280px;max-width:680px}
.pdp-main h1{font-size:24px;line-height:32px;margin:0 0 4px;font-weight:400}
.brand-link{font-size:13px;display:inline-block;margin-bottom:6px}
.pdp-main hr{border:none;border-top:1px solid #e7e7e7;margin:10px 0}
.bullets{padding-left:20px;margin:0 0 16px}.bullets li{margin:4px 0}
.reviews{margin-top:18px;padding-top:12px;border-top:1px solid #e7e7e7}
.review{padding:10px 0;border-bottom:1px solid #e7e7e7}
.review-head{font-size:13px;color:#565959;margin-bottom:4px}
.review-title{margin-bottom:4px}.review-text{margin:0;color:#0f1111;font-size:14px}
.buybox{width:260px;flex-shrink:0;border:1px solid #d5d9d9;border-radius:8px;padding:16px;height:fit-content;font-size:14px;background:#fff}
.buybox .p-whole{font-size:28px}
.delivery,.shipping{margin:6px 0;font-size:14px;color:#0f1111}
.qty{display:block;margin:8px 0}
.qty input,.cart-controls input{width:72px;border:1px solid #d5d9d9;border-radius:8px;padding:3px 8px;background:#f0f2f2;font-size:13px;box-shadow:0 2px 5px rgba(15,17,17,.15)}
.btn-cart{width:100%;background:#ffd814;border:1px solid #fcd200;border-radius:100px;padding:9px 0;font-size:14px;cursor:pointer;margin:8px 0 4px;box-shadow:0 2px 5px rgba(213,217,217,.5)}
.btn-cart:hover{background:#f7ca00}
.cart-layout{display:flex;gap:22px;flex-wrap:wrap-reverse;justify-content:flex-start;align-items:flex-start}
body.cart-page{background:#eaeded}
body.cart-page main{padding-top:18px}
.cart-main{flex:1 1 520px;max-width:900px;min-width:min(100%,520px);background:#fff;padding:16px 18px 20px;border-radius:8px}
.cart-head{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;border-bottom:1px solid #e7e7e7;padding-bottom:8px}
.cart-head h1{border:none;padding:0;margin:0;font-size:28px;line-height:36px}
.cart-price-label{color:#565959;font-size:14px;padding-bottom:6px}
.cart-empty{max-width:640px;padding:18px;background:#fff;border-radius:8px}
.btn-continue{display:inline-block;background:#ffd814;border:1px solid #fcd200;border-radius:100px;padding:8px 18px;color:#0f1111;font-size:13px;margin-top:12px}
.btn-continue:hover{background:#f7ca00;text-decoration:none;color:#0f1111}
.cart-row{display:flex;align-items:flex-start;gap:14px;border-bottom:1px solid #e7e7e7;padding:14px 0}
.cart-row>img{width:160px;height:160px;object-fit:contain;background:#fff;flex-shrink:0}
.cart-info{flex:1;display:flex;flex-direction:column;gap:3px}
.cart-title{color:#0f1111;font-size:16px;line-height:22px}
.cart-title:hover{color:#c7511f}
.cart-controls{display:flex;align-items:center;gap:12px;margin-top:8px;font-size:12px;flex-wrap:wrap}
.cart-controls form{display:inline}
.btn-link{background:none;border:none;color:#2162a1;font-size:12px;cursor:pointer;padding:0}
.btn-link:hover{text-decoration:underline}
.cart-price{font-size:18px;font-weight:700;white-space:nowrap}
.subtotal{font-size:18px;text-align:right;margin-top:12px}
.cart-box{width:280px;flex-shrink:0;border:1px solid #d5d9d9;border-radius:8px;padding:18px;height:fit-content;background:#fff}
.cart-box .subtotal{text-align:left;margin-top:0;font-size:18px}
.btn-checkout{width:100%;background:#ffd814;border:1px solid #fcd200;border-radius:100px;padding:9px 0;font-size:14px;cursor:pointer;margin-top:10px;box-shadow:0 2px 5px rgba(213,217,217,.5)}
.btn-checkout:hover{background:#f7ca00}
.confirm{color:#007600;font-size:28px;line-height:36px}
.footer{background:#232f3e;color:#ddd;margin-top:32px}
.back-top{display:block;background:#37475a;color:#fff;text-align:center;padding:14px;font-size:13px}
.back-top:hover{color:#fff;text-decoration:none;background:#485769}
.footer-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:24px;padding:28px 18px;max-width:1000px;margin:0 auto}
.footer-grid h4{color:#fff;font-size:16px;font-weight:700;margin:0 0 8px}
.footer-grid a{display:block;color:#ddd;font-size:14px;margin:4px 0}.footer-grid a:hover{color:#fff}
.footer-copy{text-align:center;padding:18px;font-size:12px;color:#999;border-top:1px solid #3a4553;margin:0}
"""
