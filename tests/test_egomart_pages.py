import pytest

from sites.egomart.app import create_app
from sites.egomart.pages import (
    brand_of,
    related_search_sidebar,
    cart_page,
    money_value,
    product_page,
    searchable_text,
    visible_attributes,
    price_html,
    sorted_results,
)
from sites.runtime import SiteState
from tests.helpers import products


def test_price_preserves_declared_listing_currency():
    assert '<span class="p-sym">US$</span>' in price_html("US$34.99")
    assert '<span class="p-sym">£</span>' in price_html("£4.99")
    assert '<span class="p-sym">$</span>' in price_html("$2.85")


@pytest.mark.parametrize(
    "price,amount,whole,cents",
    [
        ("₹5,999", 5999.0, "5999", "00"),
        ("₹1,25,999.50", 125999.5, "125999", "50"),
        ("US$1,249.99", 1249.99, "1249", "99"),
        ("$0.00", 0.0, "0", "00"),
    ],
)
def test_grouped_price_is_preserved(price, amount, whole, cents):
    assert money_value(price) == amount
    rendered = price_html(price)
    assert f'<span class="p-whole">{whole}</span>' in rendered
    assert f'<span class="p-cents">{cents}</span>' in rendered


def test_missing_price_is_not_displayed_as_free():
    assert "Price unavailable" in price_html(None)


def test_verified_brand_lives_in_public_product_details() -> None:
    item = {
        "name": "Eight Inch Frying Pan",
        "attributes": {
            "Brand": "Example Kitchen",
            "Material": "Steel",
            "listing_source": "Amazon.ca",
            "listing_type": "Organic",
        },
    }

    assert brand_of(item) == "Example Kitchen"
    assert visible_attributes(item) == {
        "Brand": "Example Kitchen",
        "Material": "Steel",
    }
    assert "example kitchen" in searchable_text(item)
    assert "organic" not in searchable_text(item)


def test_out_of_stock_product_cannot_be_added() -> None:
    item = products("graft")[0]
    item["attributes"]["Availability"] = "out_of_stock"
    state = SiteState({"schema_version": 1, "site": "egomart", "entities": [item]})

    response = (
        create_app(state)
        .test_client()
        .post("/cart/add", data={"item_id": item["item_id"], "quantity": "1"})
    )

    assert (response.status_code, state.selected_entity_ids) == (409, [])


def test_out_of_stock_product_disables_native_add_button() -> None:
    item = products("graft")[0]
    item["attributes"]["Availability"] = "out_of_stock"

    assert "disabled>Currently unavailable</button>" in product_page(item)


@pytest.mark.parametrize(
    "price,total",
    [("£0.99", "£1.98"), ("US$3.00", "US$6.00"), ("₹5,999", "₹11,998.00")],
)
def test_cart_keeps_listing_currency(price: str, total: str) -> None:
    item = products("graft")[0]
    item["price"] = price
    item["attributes"].pop("shipping_cost", None)

    assert f"<b>{total}</b>" in cart_page([(item, 2)])


def test_cart_does_not_sum_different_currencies() -> None:
    first, second = products("graft")[:2]
    first["price"], second["price"] = "£0.99", "US$3.00"
    for item in (first, second):
        item["attributes"].pop("shipping_cost", None)

    assert "<b>£0.99</b> + <b>US$3.00</b>" in cart_page([(first, 1), (second, 1)])


@pytest.mark.parametrize("price", [None, "", "Price unavailable"])
def test_unknown_money_is_distinct_from_zero(price) -> None:
    assert money_value(price) is None
    assert money_value("$0.00") == 0.0


@pytest.mark.parametrize("include_known_price", [False, True])
@pytest.mark.parametrize("shipping_cost", [None, "$4.00"])
@pytest.mark.parametrize("price", [None, "", "Price unavailable"])
def test_cart_with_unknown_price_does_not_report_a_complete_total(
    include_known_price: bool, shipping_cost: str | None, price: str | None
) -> None:
    unknown, known = products("graft")[:2]
    unknown["price"] = price
    unknown["attributes"]["shipping_cost"] = shipping_cost
    known["price"] = "$12.00"
    known["attributes"].pop("shipping_cost", None)
    entries = [(unknown, 2)] + ([(known, 1)] if include_known_price else [])

    rendered = cart_page(entries)

    units = 3 if include_known_price else 2
    assert f"Subtotal ({units} items): <b>Price unavailable</b>" in rendered
    assert "Order total: <b>Price unavailable</b>" in rendered
    assert "<b>$0.00</b>" not in rendered
    assert "<b>$12.00</b>" not in rendered
    assert "<b>$8.00</b>" not in rendered


@pytest.mark.parametrize("price,total", [("$0.00", "$8.00"), ("$12.00", "$32.00")])
def test_cart_known_prices_include_quantity_and_shipping(
    price: str, total: str
) -> None:
    item = products("graft")[0]
    item["price"] = price
    item["attributes"]["shipping_cost"] = "$4.00"

    rendered = cart_page([(item, 2)])

    assert "Shipping: <b>$8.00</b>" in rendered
    assert f"Order total: <b>{total}</b>" in rendered
    assert "Price unavailable" not in rendered


def test_unknown_shipping_does_not_become_free_shipping() -> None:
    item = products("graft")[0]
    item["price"] = "$12.00"
    item["attributes"]["shipping_cost"] = "Price unavailable"

    rendered = cart_page([(item, 2)])

    assert "Subtotal (2 items): <b>$24.00</b>" in rendered
    assert "Shipping: <b>Price unavailable</b>" in rendered
    assert "Order total: <b>Price unavailable</b>" in rendered


@pytest.mark.parametrize(
    "sort_key,expected_ids",
    [
        ("price_asc", ["free", "paid", "unknown"]),
        ("price_desc", ["paid", "free", "unknown"]),
    ],
)
def test_price_sort_places_unknown_after_known_prices(sort_key, expected_ids) -> None:
    items = [
        {"item_id": "unknown", "price": "Price unavailable"},
        {"item_id": "paid", "price": "$12.00"},
        {"item_id": "free", "price": "$0.00"},
    ]

    assert [item["item_id"] for item in sorted_results(items, sort_key)] == expected_ids


def test_cart_route_preserves_unknown_price_and_selection() -> None:
    item = products("graft")[0]
    item["price"] = "Price unavailable"
    state = SiteState({"schema_version": 1, "site": "egomart", "entities": [item]})

    response = (
        create_app(state)
        .test_client()
        .post("/cart/add", data={"item_id": item["item_id"], "quantity": "2"})
    )

    assert response.status_code == 200
    assert "Subtotal (2 items): <b>Price unavailable</b>" in response.get_data(
        as_text=True
    )
    assert state.selected_entity_ids == [item["item_id"], item["item_id"]]
    assert item["price"] == "Price unavailable"


def test_related_searches_use_explicit_links_instead_of_checkboxes() -> None:
    rendered = related_search_sidebar([
        {"brand": "A & B", "attributes": {"material": "Wood & Steel"}}
    ])

    assert '<h2>Related searches</h2><p>Starts a new search</p>' in rendered
    assert '<h3>Search by brand</h3>' in rendered
    assert '<a href="/search?q=A%20%26%20B">Search A &amp; B</a>' in rendered
    assert '<a href="/search?q=Wood%20%26%20Steel">Search Wood &amp; Steel</a>' in rendered
    assert 'class="cb"' not in rendered
    assert 'checkbox' not in rendered


def test_related_search_replaces_query_and_resets_paging_and_sort() -> None:
    items = [
        {"item_id": f"light-{i}", "name": "Work light", "brand": "DEWALT", "attributes": {}}
        for i in range(50)
    ]
    state = SiteState({"schema_version": 1, "site": "egomart", "entities": items})
    client = create_app(state).test_client()
    first = client.get("/search?q=work+light&page=2&sort=price_desc").get_data(as_text=True)

    assert '<a href="/search?q=DEWALT">Search DEWALT</a>' in first
    clicked = client.get("/search?q=DEWALT").get_data(as_text=True)
    assert 'value="DEWALT"' in clicked
    assert state.actions[-1]["query"] == "DEWALT"
    assert state.actions[-1]["page"] == 1
    assert state.actions[-1]["sort"] == "featured"


def test_related_search_keeps_products_without_brand_searchable() -> None:
    items = [
        {"item_id": "dcl074", "name": "DEWALT DCL074 Work Light", "attributes": {}},
        {"item_id": "other", "name": "DEWALT Compact Work Light", "brand": "DEWALT", "attributes": {}},
    ]
    state = SiteState({"schema_version": 1, "site": "egomart", "entities": items})
    client = create_app(state).test_client()

    response = client.get("/search?q=DEWALT")

    assert '/product/dcl074' in response.get_data(as_text=True)
    assert client.get("/product/dcl074").status_code == 200


def test_cart_quantity_six_is_native_visible_and_editable() -> None:
    item = products('graft')[0]
    state = SiteState({'schema_version': 1, 'site': 'egomart', 'entities': [item]})
    client = create_app(state).test_client()
    response = client.post('/cart/add', data={'item_id': item['item_id'], 'quantity': '6'})
    assert response.status_code == 200
    assert state.selected_entity_ids == [item['item_id']] * 6
    assert 'name="quantity" min="0" step="1" value="6"' in response.get_data(as_text=True)
    response = client.post('/cart/update', data={'item_id': item['item_id'], 'quantity': '24'})
    assert response.status_code == 200
    assert state.selected_entity_ids == [item['item_id']] * 24
    assert 'value="24"' in response.get_data(as_text=True)
    response = client.post('/cart/update', data={'item_id': item['item_id'], 'quantity': '0'})
    assert response.status_code == 200
    assert state.selected_entity_ids == []


@pytest.mark.parametrize('route', ['/cart/add', '/cart/update'])
@pytest.mark.parametrize('quantity', ['-1', '1.5', 'abc', '', '1e2'])
def test_invalid_cart_quantity_is_rejected_without_changing_cart(route, quantity) -> None:
    item = products('graft')[0]
    state = SiteState({'schema_version': 1, 'site': 'egomart', 'entities': [item]})
    state.add_quantity(item['item_id'], 6)
    response = create_app(state).test_client().post(route, data={'item_id': item['item_id'], 'quantity': quantity})
    assert response.status_code == 400
    assert state.selected_entity_ids == [item['item_id']] * 6


def test_add_zero_is_rejected_and_product_has_no_artificial_maximum() -> None:
    item = products('graft')[0]
    state = SiteState({'schema_version': 1, 'site': 'egomart', 'entities': [item]})
    client = create_app(state).test_client()
    response = client.post('/cart/add', data={'item_id': item['item_id'], 'quantity': '0'})
    assert response.status_code == 400
    assert state.selected_entity_ids == []
    rendered = product_page(item)
    assert 'name="quantity" min="1" step="1" value="1" required' in rendered
    assert 'max="3"' not in rendered
