from tools.catalog.capture_abo_search_windows import _rank


def test_rank_prefers_title_and_color_match() -> None:
    records = [
        {
            "asin": "B000000001",
            "title": "Blue dining chair",
            "brand": "A",
            "product_types": ["CHAIR"],
            "keywords": ["furniture"],
            "color": "Blue",
        },
        {
            "asin": "B000000002",
            "title": "Red folding rocking chair",
            "brand": "B",
            "product_types": ["CHAIR"],
            "keywords": ["outdoor"],
            "color": "Red",
        },
    ]

    ranked = _rank(records, "red folding chair", limit=2)

    assert [row["asin"] for row in ranked] == ["B000000002", "B000000001"]
