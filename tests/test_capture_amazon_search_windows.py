from urllib.parse import quote_plus

import pytest

from tools.catalog import capture_amazon_search_windows as search


def test_parse_amazon_search_result_card() -> None:
    body = b"""
    <div data-component-type="s-search-result" data-asin="B012345678">
      <h2><a href="/Example/dp/B012345678"><span>Real Tool</span></a></h2>
      <img class="s-image" src="https://images.example/tool.jpg" />
      <span class="a-price"><span class="a-offscreen">$12.50</span></span>
      <span class="a-icon-alt">4.5 out of 5 stars</span>
    </div>
    """

    assert search._parse(body) == [
        {
            "rank": 1,
            "asin": "B012345678",
            "title": "Real Tool",
            "price": "$12.50",
            "rating": "4.5 out of 5 stars",
            "review_count": None,
            "sponsored": False,
            "page_url": "https://www.amazon.ca/Example/dp/B012345678",
            "image_url": "https://images.example/tool.jpg",
        }
    ]


def test_search_title_is_never_invented_from_url_slug() -> None:
    body = b"""<div data-component-type="s-search-result" data-asin="B0005ZV1CQ">
    <h2><a href="/Morton-Salt-Regular-26/dp/B0005ZV1CQ"><span>Morton</span></a></h2>
    <img class="s-image" src="https://m.media-amazon.com/images/salt.jpg" /></div>"""

    assert search._parse(body)[0]["title"] == "Morton"


@pytest.mark.parametrize(
    "product_heading",
    [
        '<h2><a href="/dp/B012345678"><span>Milano Double Dark 7.5 oz</span></a></h2>',
        '<a href="/dp/B012345678"><h2><span>Milano Double Dark 7.5 oz</span></h2></a>',
    ],
)
def test_product_heading_takes_precedence_over_brand_heading(product_heading) -> None:
    body = f"""<div data-component-type="s-search-result" data-asin="B012345678">
    <h2>Pepperidge Farm</h2>{product_heading}
    <img class="s-image" src="https://images.example/cookies.jpg" /></div>"""

    row = search._parse(body.encode())[0]

    assert row["title"] == "Milano Double Dark 7.5 oz"
    assert row["page_url"] == "https://www.amazon.ca/dp/B012345678"


def _reader_card(rank: int, suffix: str = "") -> str:
    url = (
        f"http://www.amazon.ca/Tool/dp/B{rank:09d}/ref=sr_1_{rank}{suffix}?sr=8-{rank}"
    )
    return f"""[![Image {rank}: Real wooden tool](https://m.media-amazon.com/{rank}.jpg)]({url})

## [Real wooden tool {rank}]({url})

4.5[_4.5 out of 5 stars_](javascript:void(0))[(120)]({url}#customerReviews)

Price, product page[$12.50$12.50]({url})

Add to cart
"""


def _reader_page(cards: str, count: int = 2) -> str:
    return f"""Title: Amazon.ca : wooden tools
URL Source: http://www.amazon.ca/s?k=wooden+tools

# ## 1-{count} of over 200 results for "wooden tools"

## Results

{cards}

## Need help?
"""


def test_reader_preserves_complete_primary_result_cards() -> None:
    rows = search._parse_reader_markdown(
        _reader_page(_reader_card(1) + _reader_card(2))
    )

    assert [
        (
            row["rank"],
            row["asin"],
            row["title"],
            row["price"],
            row["rating"],
            row["review_count"],
        )
        for row in rows
    ] == [
        (1, "B000000001", "Real wooden tool 1", "$12.50", "4.5 out of 5 stars", "120"),
        (2, "B000000002", "Real wooden tool 2", "$12.50", "4.5 out of 5 stars", "120"),
    ]


def test_reader_accepts_observed_primary_modifier_without_dropping_cards() -> None:
    rows = search._parse_reader_markdown(
        _reader_page(_reader_card(1) + _reader_card(2, "_mod_primary_new"))
    )

    assert [row["rank"] for row in rows] == [1, 2]


def test_reader_excludes_recommendation_carousel_not_primary_sponsored_card() -> None:
    recommendation = _reader_card(9).replace("ref=sr_1_9", "ref=sxin_thematic")
    primary = _reader_card(2, "_sspa")
    rows = search._parse_reader_markdown(
        _reader_page(_reader_card(1) + "\n## Highly rated\n" + recommendation + primary)
    )

    assert [(row["rank"], row["sponsored"]) for row in rows] == [(1, False), (2, True)]


def test_reader_does_not_take_rating_from_recommendation_when_main_card_has_none() -> (
    None
):
    first = _reader_card(1).replace(
        "4.5[_4.5 out of 5 stars_](javascript:void(0))[(120)]"
        "(http://www.amazon.ca/Tool/dp/B000000001/ref=sr_1_1?sr=8-1#customerReviews)",
        "",
    )
    recommendation = _reader_card(9).replace("ref=sr_1_9", "ref=sxin_thematic")
    rows = search._parse_reader_markdown(
        _reader_page(first + "\n## Highly rated\n" + recommendation + _reader_card(2))
    )

    assert rows[0]["rating"] is None


@pytest.mark.parametrize(
    "cards",
    [
        _reader_card(1),
        _reader_card(2) + _reader_card(1),
        _reader_card(1) + _reader_card(1),
        _reader_card(1) + _reader_card(2) + _reader_card(3),
    ],
)
def test_reader_rejects_missing_duplicated_reordered_or_extra_primary_ranks(
    cards: str,
) -> None:
    with pytest.raises(ValueError, match="ranks"):
        search._parse_reader_markdown(_reader_page(cards))


def test_reader_rejects_duplicate_asins_instead_of_silently_deduplicating() -> None:
    repeated = _reader_card(2).replace("B000000002", "B000000001")

    with pytest.raises(ValueError, match="ASINs are duplicated"):
        search._parse_reader_markdown(_reader_page(_reader_card(1) + repeated))


def test_reader_requires_actual_linked_title_not_image_alt_or_slug() -> None:
    missing_title = _reader_card(2).replace(
        "## [Real wooden tool 2]", "[Real wooden tool 2]"
    )

    with pytest.raises(ValueError, match="public title"):
        search._parse_reader_markdown(_reader_page(_reader_card(1) + missing_title))


def test_reader_rejects_truncated_page_even_when_last_card_is_present() -> None:
    truncated = _reader_page(_reader_card(1) + _reader_card(2)).replace(
        "## Need help?", ""
    )

    with pytest.raises(ValueError, match="ending"):
        search._parse_reader_markdown(truncated)


def test_reader_uses_declared_range_instead_of_a_fixed_48_card_limit() -> None:
    rows = search._parse_reader_markdown(
        _reader_page("".join(_reader_card(rank) for rank in range(1, 6)), count=5)
    )

    assert len(rows) == 5


def test_reader_unwraps_primary_sponsored_url() -> None:
    destination = "/Tool/dp/B000000002/ref=sr_1_2_sspa?sr=8-2"
    wrapped = "http://www.amazon.ca/sspa/click?url=" + quote_plus(destination)
    second = _reader_card(2, "_sspa").replace(
        "http://www.amazon.ca" + destination, wrapped
    )
    rows = search._parse_reader_markdown(_reader_page(_reader_card(1) + second))

    assert rows[1]["sponsored"] is True


def test_explicit_fallback_records_direct_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_direct(*args: object) -> None:
        raise RuntimeError("HTTP shell has no result cards")

    monkeypatch.setattr(search, "_capture", fail_direct)
    monkeypatch.setattr(
        search,
        "_reader_capture",
        lambda *args: ("task", "before", {"provider": search.READER_PROVIDER}),
    )

    assert search._capture_with_reader("task", "before", "wooden tools")[2][
        "direct_capture_error"
    ] == ("RuntimeError: HTTP shell has no result cards")


def test_successful_direct_capture_does_not_use_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ("task", "before", {"provider": search.PROVIDER})
    monkeypatch.setattr(search, "_capture", lambda *args: expected)

    def unexpected_reader(*args: object) -> None:
        raise AssertionError("reader must not run")

    monkeypatch.setattr(search, "_reader_capture", unexpected_reader)

    assert search._capture_with_reader("task", "before", "wooden tools") == expected


def test_capture_keeps_source_rows_without_unused_response_digest(monkeypatch):
    source_url = "https://www.amazon.ca/s?k=wooden+tools"
    rows = [{"asin": "B012345678", "title": "Real Tool", "rank": 1}]
    monkeypatch.setattr(search, "_fetch", lambda query: (source_url, b"page"))
    monkeypatch.setattr(search, "_parse", lambda body: rows)

    _, _, capture = search._capture("task", "before", "wooden tools")

    assert capture["response_url"] == source_url
    assert capture["results"] == rows
    assert capture["result_count"] == 1
    assert "response_sha256" not in capture
    assert "response_hash_scope" not in capture


@pytest.mark.parametrize("site", ["egomart", "egomedia"])
def test_equal_queries_are_captured_once(tmp_path, monkeypatch, site):
    import json
    from tools.catalog import capture_youtube_search_windows as youtube

    module = search if site == "egomart" else youtube
    reviews = tmp_path / "data/sources/search_query_reviews.json"
    reviews.parent.mkdir(parents=True)
    reviews.write_text(
        json.dumps(
            {
                "tasks": {
                    "same-task": {
                        "before_query": "wooden tool",
                        "after_query": "wooden tool",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(module, "load_tasks", lambda _: [])
    calls = []

    def capture(task_id, stage, query):
        calls.append((task_id, stage, query))
        return task_id, stage, {"result_count": 1, "results": [{"title": "Tool"}]}

    monkeypatch.setattr(module, "_capture", capture)
    result = module.capture(tmp_path, concurrency=1, task_id="same-task")
    assert result["captures"] == 1
    assert calls == [("same-task", "before_change", "wooden tool")]
    assert not (tmp_path / module.OUTPUT_DIR / "same-task--after_change.json").exists()
