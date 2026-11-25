import json
from collections.abc import Mapping
from email.message import Message
from pathlib import Path
from urllib.error import HTTPError

import pytest

from tools.catalog import capture_amazon_product_details as details


@pytest.mark.parametrize("key_tag", ["th", "td"])
def test_detail_reads_product_table_cells(key_tag: str) -> None:
    body = f"""<span id="productTitle">Kadai with lid</span>
    <table class="a-keyvalue prodDetTable">
      <tr><{key_tag}>Number of Items</{key_tag}><td>1</td></tr>
      <tr><{key_tag}>Included Components</{key_tag}>
        <td>KADAI WITH LID - 1.8 QUART</td></tr>
    </table>""".encode()

    result = details._parse_detail(
        "B0DJHD9ZC7", body, "https://www.amazon.ca/dp/B0DJHD9ZC7"
    )

    assert result["product_details"] == {
        "Number of Items": "1",
        "Included Components": "KADAI WITH LID - 1.8 QUART",
    }


def test_detail_keeps_product_overview_and_description_separate_from_other_tables() -> (
    None
):
    body = b"""<span id="productTitle">Kadai with lid</span>
    <div id="productOverview_feature_div"><table>
      <tr><td>Material</td><td>Stainless steel</td></tr>
    </table></div>
    <table id="productDetails_techSpec_section_1">
      <tr><th>Number of Items</th><td>2</td></tr>
    </table>
    <div id="productDescription"><p>1-Piece Kadai + 1-Piece Glass lid</p></div>
    <table><tr><td>Number of Items</td><td>12</td></tr></table>"""

    result = details._parse_detail(
        "B089G1K3Q8", body, "https://www.amazon.ca/dp/B089G1K3Q8"
    )

    assert result["product_details"] == {
        "Material": "Stainless steel",
        "Number of Items": "2",
        "Product description": "1-Piece Kadai + 1-Piece Glass lid",
    }


def test_reader_markdown_enriches_sparse_amazon_detail() -> None:
    markdown = """Title: Steel Mixing Bowl : Amazon.ca: Home
URL Source: https://www.amazon.ca/dp/B012345678

[Brand: Example Kitchen](https://www.amazon.ca/example)

# About this item

* Durable stainless steel construction for everyday mixing.
* Dishwasher safe with a nonslip base.

# Product information

| Brand | Example Kitchen |
| Material | Stainless Steel |
"""

    result = details._parse_reader_markdown(
        "B012345678",
        markdown,
        {"asin": "B012345678", "title": "Old title"},
        "https://r.jina.ai/http://www.amazon.ca/dp/B012345678",
    )

    assert result["title"] == "Steel Mixing Bowl"
    assert result["brand"] == "Example Kitchen"
    assert result["feature_bullets"] == [
        "Durable stainless steel construction for everyday mixing.",
        "Dishwasher safe with a nonslip base.",
    ]
    assert result["product_details"]["Material"] == "Stainless Steel"
    assert result["provider"].endswith("jina-reader-v1")


@pytest.mark.parametrize(
    "identity",
    [
        '<input id="ASIN" value="B079NWHSCS">',
        '<link rel="canonical" href="https://www.amazon.ca/dp/B079NWHSCS">',
        '<table class="prodDetTable"><tr><th>ASIN &#x200f;</th>'
        "<td>&#x200e; B079NWHSCS</td></tr></table>",
        '<div id="detailBullets_feature_div"><li>ASIN: B079NWHSCS</li></div>',
    ],
)
def test_html_rejects_another_products_identity(identity: str) -> None:
    body = ('<span id="productTitle">Pool Trowel</span>' + identity).encode()

    with pytest.raises(ValueError, match="ASIN mismatch: B079NWHSCS"):
        details._parse_detail("B007H4QCUE", body, "https://www.amazon.ca/dp/B007H4QCUE")


def test_html_rejects_redirect_to_another_product() -> None:
    with pytest.raises(ValueError, match="response_url ASIN mismatch"):
        details._parse_detail(
            "B007H4QCUE",
            b'<span id="productTitle">Battery</span>',
            "https://www.amazon.ca/dp/B079NWHSCS",
        )


def test_html_checks_all_asin_rows_before_duplicate_keys_are_overwritten() -> None:
    body = b"""<span id="productTitle">Pool Trowel</span>
    <table class="prodDetTable">
      <tr><th>ASIN</th><td>B079NWHSCS</td></tr>
      <tr><th>ASIN</th><td>B007H4QCUE</td></tr>
    </table>"""

    with pytest.raises(ValueError, match="ASIN mismatch: B079NWHSCS"):
        details._parse_detail("B007H4QCUE", body, "https://www.amazon.ca/dp/B007H4QCUE")


@pytest.mark.parametrize("asin", ["B007H4QCUE", "\u200e B007H4QCUE\u200f"])
def test_identity_accepts_same_asin_with_amazon_direction_marks(asin: str) -> None:
    details.validate_product_identity(
        "B007H4QCUE",
        {"product_details": {"ASIN \u200f": asin}},
    )


@pytest.mark.parametrize(
    "markdown",
    [
        "# Pool Trowel\n## About this item\n* Blue steel blade\n",
        "URL Source: https://www.amazon.ca/dp/B079NWHSCS\n"
        "# Pool Trowel\n## About this item\n* Blue steel blade\n",
        "URL Source: https://www.amazon.ca/dp/B007H4QCUE\n"
        "# Pool Trowel\n## Product information\n| ASIN | B079NWHSCS |\n"
        "| ASIN | B007H4QCUE |\n| Blade | Blue steel |\n",
    ],
)
def test_reader_rejects_missing_or_conflicting_product_identity(markdown: str) -> None:
    with pytest.raises(ValueError, match="source URL|ASIN mismatch"):
        details._parse_reader_markdown(
            "B007H4QCUE",
            markdown,
            {},
            "https://r.jina.ai/http://www.amazon.ca/dp/B007H4QCUE",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"title": "Steel Mixing Bowl", "brand": "Example Kitchen"},
        {"title": "Amazon.ca", "feature_bullets": ["Stainless steel construction"]},
        {
            "title": "Amazon.ca: Home & Kitchen",
            "product_details": {"Material": "Steel"},
        },
        {
            "title": "Example Kitchen",
            "brand": "Example Kitchen",
            "product_details": {"Material": "Steel"},
        },
        {
            "title": "Steel Mixing Bowl",
            "product_details": {"Brand": "Example", "ASIN": "B012345678"},
        },
        {
            "title": "Steel Mixing Bowl",
            "product_details": {"Customer reviews": "4.5", "Best Sellers Rank": "1"},
        },
    ],
)
def test_rich_rejects_generic_identity_only_metadata(
    payload: dict[str, object],
) -> None:
    assert details._rich(payload) is False


@pytest.mark.parametrize(
    "metadata",
    [
        {"product_details": {"Material": "Steel"}},
        {"feature_bullets": ["Hand wash only"]},
    ],
)
def test_rich_requires_product_decision_evidence(metadata: dict[str, object]) -> None:
    assert details._rich({"title": "Steel Mixing Bowl", **metadata}) is True


def test_single_word_product_title_is_not_a_service_placeholder() -> None:
    assert details._product_title("Colander") is True
    assert details._product_title(" ") is False


def _parse(markdown: str) -> dict[str, object]:
    return details._parse_reader_markdown(
        "B012345678",
        "URL Source: https://www.amazon.ca/dp/B012345678\n" + markdown,
        {"title": "Amazon.ca"},
        "https://r.jina.ai/http://www.amazon.ca/dp/B012345678",
    )


@pytest.mark.parametrize(
    "heading",
    [
        "# Steel Mixing Bowl",
        "Steel Mixing Bowl\n=================",
        "## Product Summary: Steel Mixing Bowl",
    ],
)
def test_reader_extracts_actual_product_heading_when_page_title_is_generic(
    heading: str,
) -> None:
    result = _parse(
        f"Title: Amazon.ca\n\n{heading}\n\n## About this item\n\n* Hand wash only\n"
    )

    assert result["title"] == "Steel Mixing Bowl"


def test_reader_ignores_amazon_product_summary_navigation_heading() -> None:
    result = _parse("""Title: Amazon.ca
# [Product summary presents key product information](http://www.amazon.ca/dp/B012345678#)
# Steel Mixing Bowl
## About this item
* Hand wash only
# Product Summary: Steel Mixing Bowl
""")

    assert result["title"] == "Steel Mixing Bowl"


def test_reader_repeated_identical_product_headings_are_not_ambiguous() -> None:
    result = _parse(
        "# Steel Mixing Bowl\n# Steel Mixing Bowl\n## About this item\n* Hand wash only\n"
    )

    assert result["title"] == "Steel Mixing Bowl"


def test_reader_preserves_failure_for_two_distinct_product_headings() -> None:
    with pytest.raises(ValueError, match="ambiguous product headings"):
        _parse(
            "# Steel Mixing Bowl\n# Silicone Mixing Bowl\n## About this item\n* Hand wash only\n"
        )


def test_reader_stops_product_table_before_recommendations() -> None:
    result = _parse("""# Steel Mixing Bowl
## Product information
### Technical details
| Material | Steel |
## Frequently bought together
| Material | Silicone |
| Brand | Other seller |
""")

    assert result["product_details"] == {"Material": "Steel"}


def test_reader_does_not_count_an_unscoped_table_as_product_evidence() -> None:
    with pytest.raises(ValueError, match="remains sparse"):
        _parse("""# Steel Mixing Bowl
| Material | Steel |
## Customer reviews
| Material | Silicone |
""")


def test_reader_does_not_inherit_stale_features_into_a_sparse_new_response() -> None:
    with pytest.raises(ValueError, match="remains sparse"):
        details._parse_reader_markdown(
            "B012345678",
            "URL Source: https://www.amazon.ca/dp/B012345678\n# Steel Mixing Bowl\n",
            {"feature_bullets": ["Old unverified dishwasher safe claim"]},
            "https://r.jina.ai/http://www.amazon.ca/dp/B012345678",
        )


def test_reader_does_not_inherit_an_unobserved_brand() -> None:
    result = details._parse_reader_markdown(
        "B012345678",
        "URL Source: https://www.amazon.ca/dp/B012345678\n"
        "# Steel Mixing Bowl\n## About this item\n* Hand wash only\n",
        {"brand": "Old brand"},
        "https://r.jina.ai/http://www.amazon.ca/dp/B012345678",
    )

    assert result["brand"] is None


def test_reader_generic_title_cannot_be_recovered_from_product_url() -> None:
    with pytest.raises(ValueError, match="no descriptive product title"):
        _parse("""Title: Amazon.ca
URL Source: http://www.amazon.ca/Steel-Mixing-Bowl/dp/B012345678
## About this item
* Stainless steel construction
""")


def test_capture_reports_sparse_detail_without_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        details,
        "_direct_capture",
        lambda asin: {"asin": asin, "title": "Amazon.ca", "brand": "Example"},
    )

    with pytest.raises(ValueError, match="remains sparse"):
        details._capture("B012345678")


def test_enrichment_can_recover_from_a_failed_direct_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_direct(asin: str) -> None:
        raise RuntimeError("upstream shell")

    monkeypatch.setattr(details, "_direct_capture", fail_direct)
    monkeypatch.setattr(
        details,
        "_reader_capture",
        lambda asin, base: {
            **base,
            "title": "Steel Mixing Bowl",
            "product_details": {"Material": "Steel"},
        },
    )

    assert details._capture("B012345678", enrich_sparse=True)[1][
        "direct_capture_error"
    ] == ("RuntimeError: upstream shell")


def test_reader_does_not_retry_rate_limited_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = 0

    def rate_limited(*args: object, **kwargs: object) -> None:
        nonlocal requests
        requests += 1
        raise HTTPError("https://r.jina.ai/", 429, "Too Many Requests", Message(), None)

    monkeypatch.setattr(details, "urlopen", rate_limited)

    with pytest.raises(HTTPError, match="429"):
        details._reader_capture("B012345678", {"title": "Amazon.ca"})

    assert requests == 1


def _snapshot_root(root: Path, snapshot: Mapping[str, object]) -> Path:
    catalog = root / details.CATALOG
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "item_id": "product-one",
                        "text_provenance": {
                            "source_url": "https://www.amazon.ca/dp/B012345678"
                        },
                    }
                ]
            }
        )
    )
    path = root / details.OUTPUT_DIR / "B012345678.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(snapshot))
    return path


def test_failed_forced_capture_preserves_existing_valid_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {"title": "Steel Mixing Bowl", "product_details": {"Material": "Steel"}}
    path = _snapshot_root(tmp_path, snapshot)
    original = path.read_bytes()
    monkeypatch.setattr(
        details,
        "_capture",
        lambda *args, **kwargs: (
            "B012345678",
            {"asin": "B012345678", "title": "Amazon.ca"},
        ),
    )

    with pytest.raises(RuntimeError, match="remains sparse"):
        details.capture(
            tmp_path,
            entity_ids={"product-one"},
            task_ids=None,
            concurrency=1,
            force=True,
            enrich_sparse=True,
        )

    assert path.read_bytes() == original


def test_repaired_sparse_snapshot_keeps_lightweight_capture_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = {
        "asin": "B012345678",
        "title": "Amazon.ca",
        "provider": "direct",
        "response_sha256": "old-hash",
    }
    path = _snapshot_root(tmp_path, snapshot)
    monkeypatch.setattr(
        details,
        "_capture",
        lambda *args, **kwargs: (
            "B012345678",
            {
                "asin": "B012345678",
                "title": "Steel Mixing Bowl",
                "product_details": {"Material": "Steel"},
            },
        ),
    )

    details.capture(
        tmp_path,
        entity_ids={"product-one"},
        task_ids=None,
        concurrency=1,
        force=False,
        enrich_sparse=True,
    )

    assert json.loads(path.read_text())["capture_history"] == [
        {key: snapshot[key] for key in ("provider", "response_sha256", "title")}
    ]


def test_cached_conflicting_detail_stops_before_capture_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _snapshot_root(
        tmp_path,
        {
            "asin": "B012345678",
            "title": "Bowl",
            "product_details": {"ASIN": "B079NWHSCS"},
        },
    )
    original = path.read_bytes()
    calls = []
    monkeypatch.setattr(details, "_capture", lambda *args, **kwargs: calls.append(args))

    with pytest.raises(ValueError, match="ASIN mismatch"):
        details.capture(
            tmp_path,
            entity_ids={"product-one"},
            task_ids=None,
            concurrency=1,
            force=False,
            enrich_sparse=True,
        )

    assert (path.read_bytes(), calls) == (original, [])


@pytest.mark.parametrize(
    "captured",
    [
        ("B079NWHSCS", {"asin": "B079NWHSCS", "title": "Battery"}),
        (
            "B012345678",
            {
                "asin": "B012345678",
                "title": "Bowl",
                "product_details": {"ASIN": "B079NWHSCS"},
            },
        ),
    ],
)
def test_conflicting_capture_result_preserves_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    captured: tuple[str, dict[str, object]],
) -> None:
    path = _snapshot_root(tmp_path, {"asin": "B012345678", "title": "Original bowl"})
    original = path.read_bytes()
    monkeypatch.setattr(details, "_capture", lambda *args, **kwargs: captured)

    with pytest.raises(RuntimeError, match="ASIN mismatch"):
        details.capture(
            tmp_path,
            entity_ids={"product-one"},
            task_ids=None,
            concurrency=1,
            force=True,
            enrich_sparse=True,
        )

    assert path.read_bytes() == original
