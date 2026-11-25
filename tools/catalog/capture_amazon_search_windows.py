"""Capture complete real Amazon first-page result windows for EgoMart tasks.

The capture preserves upstream order and stable ASINs. It is evidence only: the
task landing and core-neighbor labels still require Coding-Agent review.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup  # type: ignore[import-untyped]

from egocua.schema import load_tasks
from egocua.workspace import write_json_atomic


OUTPUT_DIR = Path("data/sources/upstream_search_snapshots/amazon")
PROVIDER = "amazon-ca-search-first-page-v1"
SEARCH_ORIGIN = "https://www.amazon.ca"
READER_ORIGIN = "https://r.jina.ai/http://www.amazon.ca"
READER_PROVIDER = "amazon-ca-search-first-page-v1+jina-reader-v1"
MAX_CONCURRENCY = 4
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0 Safari/537.36"
)


def _text(node: Any) -> str:
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _fetch(query: str) -> tuple[str, bytes]:
    url = f"{SEARCH_ORIGIN}/s?k={quote_plus(query)}"
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-CA,en;q=0.9",
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=45) as response:
        body = response.read()
        return response.geturl(), body


def _parse(body: bytes) -> list[dict[str, Any]]:
    soup = BeautifulSoup(body, "html.parser")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    cards = soup.select('[data-component-type="s-search-result"][data-asin]')
    for card in cards:
        asin = str(card.get("data-asin") or "").strip()
        if len(asin) != 10 or asin in seen:
            continue
        title_node = card.select_one("h2")
        link = None
        for heading in card.select("h2"):
            heading_link = heading.select_one("a[href]") or heading.find_parent(
                "a", href=True
            )
            if heading_link and f"/dp/{asin}" in str(heading_link.get("href", "")):
                title_node, link = heading, heading_link
                break
        title = _text(title_node)
        image = card.select_one("img.s-image")
        link = (
            link
            or card.select_one("h2 a[href]")
            or card.select_one("a.a-link-normal[href]")
        )
        if not title or not image or not link:
            continue
        image_url = str(image.get("src") or "").strip()
        page_url = urljoin(SEARCH_ORIGIN, str(link.get("href") or ""))
        if not image_url or "/dp/" not in page_url:
            page_url = f"{SEARCH_ORIGIN}/dp/{asin}"
        rows.append(
            {
                "rank": len(rows) + 1,
                "asin": asin,
                "title": title,
                "price": _text(card.select_one(".a-price .a-offscreen")) or None,
                "rating": _text(card.select_one(".a-icon-alt")) or None,
                "review_count": _text(
                    card.select_one('a[href*="customerReviews"] span')
                )
                or None,
                "sponsored": card.select_one(".s-sponsored-label-text") is not None,
                "page_url": page_url.split("?")[0],
                "image_url": image_url,
            }
        )
        seen.add(asin)
    return rows


def _primary_identity(url: str) -> tuple[int, str, bool] | None:
    parsed = urlparse(url)
    if parsed.hostname not in {"amazon.ca", "www.amazon.ca"}:
        return None
    sponsored = parsed.path == "/sspa/click"
    if sponsored:
        destination = parse_qs(parsed.query).get("url", [])
        if len(destination) != 1:
            raise ValueError("reader sponsored link has no unambiguous destination")
        parsed = urlparse(urljoin(SEARCH_ORIGIN, destination[0]))
        if parsed.hostname not in {"amazon.ca", "www.amazon.ca"}:
            raise ValueError("reader sponsored destination is not Amazon.ca")
    reference = re.search(r"/ref=sr_(\d+)_(\d+)([^/?]*)", parsed.path)
    if reference is None:
        return None
    if reference.group(1) != "1" or reference.group(3) not in {
        "",
        "_mod_primary_new",
        "_sspa",
    }:
        raise ValueError("reader result has an unsupported primary-search reference")
    asin = re.search(r"/dp/([A-Z0-9]{10})(?:/|$)", parsed.path)
    if asin is None:
        raise ValueError("reader primary result has no ASIN")
    return (
        int(reference.group(2)),
        asin.group(1),
        sponsored or reference.group(3) == "_sspa",
    )


def _reader_card(match: re.Match[str], block: str) -> dict[str, Any]:
    identity = _primary_identity(match.group("url"))
    if identity is None:
        raise ValueError("reader card is not a primary result")
    rank, asin, sponsored = identity
    title = re.search(
        r"^(?:Sponsored\s*)?##\s+\[(?P<title>.+?)\]\((?P<url>https?://[^\s)]+)\)\s*$",
        block,
        re.MULTILINE,
    )
    if title is None or _primary_identity(title.group("url")) != identity:
        raise ValueError(f"reader result {rank}: missing or mismatched public title")
    metadata = re.split(
        r"^#{1,6}\s+|^Sponsored##", block[title.end() :], maxsplit=1, flags=re.MULTILINE
    )[0]
    price_label = re.search(r"Price, product page\[([^\]]+)\]", metadata)
    price = (
        re.search(r"(?:CA\s*)?\$\s*\d[\d,]*\.\d{2}", price_label.group(1))
        if price_label
        else None
    )
    rating = re.search(r"\d(?:\.\d)? out of 5 stars", metadata)
    reviews = re.search(r"\[\(([^)]+)\)\]\([^\s)]*#customerReviews\)", metadata)
    return {
        "rank": rank,
        "asin": asin,
        "title": title.group("title").strip(),
        "price": price.group(0) if price else None,
        "rating": rating.group(0) if rating else None,
        "review_count": reviews.group(1) if reviews else None,
        "sponsored": sponsored,
        "page_url": f"{SEARCH_ORIGIN}/dp/{asin}",
        "observed_page_url": match.group("url"),
        "image_url": match.group("image"),
    }


def _parse_reader_markdown(markdown: str) -> list[dict[str, Any]]:
    declared = re.search(
        r"\b1\s*[-–]\s*(\d+)\s+of\s+(?:over\s+)?[\d,]+\s+results\b", markdown
    )
    start = re.search(r"^## Results\s*$", markdown, re.MULTILINE)
    if declared is None or start is None:
        raise ValueError("reader page has no explicit first-page result range")
    cards = list(
        re.finditer(
            r"^\[!\[(?P<alt>.*?)\]\((?P<image>https?://[^\s)]+)\)\]"
            r"\((?P<url>https?://[^\s)]+)\)[ \t]*$",
            markdown,
            re.MULTILINE,
        )
    )
    rows: list[dict[str, Any]] = []
    last_card_end = 0
    for index, card in enumerate(cards):
        if card.start() < start.end() or _primary_identity(card.group("url")) is None:
            continue
        end = cards[index + 1].start() if index + 1 < len(cards) else len(markdown)
        rows.append(_reader_card(card, markdown[card.end() : end]))
        last_card_end = card.end()
    expected_ranks = list(range(1, int(declared.group(1)) + 1))
    if [row["rank"] for row in rows] != expected_ranks:
        raise ValueError(
            "reader primary-result ranks are incomplete, duplicated, or out of order"
        )
    if len({row["asin"] for row in rows}) != len(rows):
        raise ValueError(
            "reader primary-result ASINs are duplicated; capture needs browser review"
        )
    if (
        re.search(r"^## Need help\?\s*$", markdown[last_card_end:], re.MULTILINE)
        is None
    ):
        raise ValueError("reader page has no complete result-window ending")
    return rows


def _reader_capture(
    task_id: str, stage: str, query: str
) -> tuple[str, str, dict[str, Any]]:
    reader_url = f"{READER_ORIGIN}/s?k={quote_plus(query)}"
    request = Request(
        reader_url, headers={"User-Agent": "Mozilla/5.0", "Accept": "text/plain"}
    )
    with urlopen(request, timeout=55) as response:
        body = response.read()
        reader_response_url = response.geturl()
    markdown = body.decode("utf-8")
    source = re.search(r"^URL Source:\s*(\S+)\s*$", markdown, re.MULTILINE)
    if source is None:
        raise ValueError("reader response has no canonical source URL")
    parsed = urlparse(source.group(1))
    if (
        parsed.hostname not in {"amazon.ca", "www.amazon.ca"}
        or parsed.path != "/s"
        or parse_qs(parsed.query).get("k") != [query]
    ):
        raise ValueError(
            "reader response source does not match the requested Amazon query"
        )
    results = _parse_reader_markdown(markdown)
    if not results:
        raise RuntimeError(
            f"{task_id}/{stage}: complete reader window contains no results"
        )
    return (
        task_id,
        stage,
        {
            "schema_version": 1,
            "task_id": task_id,
            "site": "egomart",
            "stage": stage,
            "query": query,
            "provider": READER_PROVIDER,
            "page_url": f"{SEARCH_ORIGIN}/s?k={quote_plus(query)}",
            "response_url": reader_response_url,
            "reader_url": reader_url,
            "capture_method": "public_reader_markdown_primary_results",
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "result_count": len(results),
            "collection_anomalies": [],
            "completeness": {
                "method": "declared_range_and_primary_search_reference_sequence",
                "first_rank": results[0]["rank"],
                "last_rank": results[-1]["rank"],
                "excluded": "recommendation and advertising links without primary sr_1_N references",
            },
            "results": results,
        },
    )


def _capture_with_reader(
    task_id: str, stage: str, query: str
) -> tuple[str, str, dict[str, Any]]:
    try:
        return _capture(task_id, stage, query)
    except Exception as exc:
        captured_id, captured_stage, payload = _reader_capture(task_id, stage, query)
        payload["direct_capture_error"] = f"{type(exc).__name__}: {exc}"
        return captured_id, captured_stage, payload


def _capture(task_id: str, stage: str, query: str) -> tuple[str, str, dict[str, Any]]:
    response_url, body = _fetch(query)
    results = _parse(body)
    if not results:
        raise RuntimeError(
            f"{task_id}/{stage}: Amazon returned only {len(results)} usable results"
        )
    anomalies: list[str] = []
    if len(results) < 24:
        anomalies.append("fewer_than_24_results")
    elif len(results) > 64:
        anomalies.append("more_than_64_results")
    return (
        task_id,
        stage,
        {
            "schema_version": 1,
            "task_id": task_id,
            "site": "egomart",
            "stage": stage,
            "query": query,
            "provider": PROVIDER,
            "response_url": response_url,
            "capture_method": "first_party_http_search_cards",
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "result_count": len(results),
            "collection_anomalies": anomalies,
            "results": results,
        },
    )


def capture(
    root: Path,
    *,
    concurrency: int,
    task_id: str | None = None,
    reader_fallback: bool = False,
) -> dict[str, int]:
    tasks = {
        str(task["id"]): task
        for task in load_tasks(root / "data/tasks")
        if task["site"] == "egomart"
    }
    reviews = json.loads((root / "data/sources/search_query_reviews.json").read_text())[
        "tasks"
    ]
    if task_id is not None:
        if task_id not in tasks and task_id not in reviews:
            raise ValueError(f"unknown EgoMart task: {task_id}")
        # A replacement task is captured before it can be compiled because its
        # real entity IDs do not exist until this evidence step finishes.
        tasks = {task_id: tasks.get(task_id, {"id": task_id, "site": "egomart"})}
    jobs: list[tuple[str, str, str]] = []
    for current_id in sorted(tasks):
        review = reviews.get(current_id) or {}
        for stage, field in (
            ("before_change", "before_query"),
            ("after_change", "after_query"),
        ):
            query = str(review.get(field) or "").strip()
            if (
                stage == "after_change"
                and query == str(review.get("before_query") or "").strip()
            ):
                continue
            if not query:
                raise ValueError(f"{current_id}: missing reviewed {field}")
            path = root / OUTPUT_DIR / f"{current_id}--{stage}.json"
            if path.is_file():
                existing = json.loads(path.read_text())
                if (
                    existing.get("provider") in {PROVIDER, READER_PROVIDER}
                    and existing.get("query") == query
                    and bool(existing.get("results"))
                ):
                    continue
            jobs.append((current_id, stage, query))

    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    completed = 0
    failures: list[str] = []
    capture_window = _capture_with_reader if reader_fallback else _capture
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {
            pool.submit(capture_window, current_id, stage, query): (current_id, stage)
            for current_id, stage, query in jobs
        }
        for future in as_completed(pending):
            job_id, job_stage = pending[future]
            try:
                current_id, stage, payload = future.result()
            except Exception as exc:
                failure = f"{job_id}/{job_stage}: {exc}"
                failures.append(failure)
                print(f"FAILED {failure}", flush=True)
                continue
            write_json_atomic(output / f"{current_id}--{stage}.json", payload)
            completed += 1
            print(
                f"captured {completed}/{len(jobs)} {current_id} {stage} "
                f"({payload['result_count']} results)",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"Amazon capture failed for {len(failures)} windows:\n"
            + "\n".join(sorted(failures))
        )
    return {"tasks": len(tasks), "captures": completed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--concurrency", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--task-id")
    parser.add_argument(
        "--reader-fallback",
        action="store_true",
        help="On direct HTTP failure, require a complete primary-result window from the public reader.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            capture(
                Path(args.root).resolve(),
                concurrency=args.concurrency,
                task_id=args.task_id,
                reader_fallback=args.reader_fallback,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
