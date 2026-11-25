"""Freeze real Amazon detail metadata for task answers and comparison neighbors.

Search cards are intentionally insufficient for hard tasks.  This capture keeps
the public product title, brand, feature bullets, detail table, price, and primary
image from the real listing so EgoMart can expose the same decision evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup  # type: ignore[import-untyped]

from egocua.schema import load_tasks
from egocua.sources import task_detail_entity_ids
from egocua.workspace import write_json_atomic


OUTPUT_DIR = Path("data/sources/upstream_detail_snapshots/amazon")
CATALOG = Path("sites/egomart/products.json")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0 Safari/537.36"
)
READER_BASE = "https://r.jina.ai/http://www.amazon.ca/dp/"
READER_USER_AGENT = "Mozilla/5.0"


def _text(node: Any) -> str:
    return " ".join(node.get_text(" ", strip=True).split()) if node else ""


def _identity_text(value: Any) -> str:
    return "".join(
        character
        for character in str(value)
        if not character.isspace() and unicodedata.category(character) != "Cf"
    ).upper()


def validate_product_identity(asin: str, payload: Mapping[str, Any]) -> None:
    identifiers: dict[str, Any] = {}
    if payload.get("asin"):
        identifiers["asin"] = payload["asin"]
    if payload.get("observed_page_asin"):
        identifiers["observed_page_asin"] = payload["observed_page_asin"]
    for field in (
        "page_url",
        "response_url",
        "source_url",
        "canonical_url",
        "reader_source_url",
    ):
        url = payload.get(field)
        if not url:
            continue
        match = re.search(r"/dp/([A-Z0-9]{10})(?:[/#?]|$)", str(url))
        if match is None:
            raise ValueError(f"{asin}: {field} has no stable product ASIN")
        identifiers[field] = match.group(1)
    for key, value in (payload.get("product_details") or {}).items():
        if _identity_text(key) == "ASIN":
            identifiers[f"product_details.{key}"] = value
    if not identifiers:
        raise ValueError(f"{asin}: product detail has no stable product ASIN")
    for field, value in identifiers.items():
        observed = _identity_text(value)
        if observed != asin:
            raise ValueError(f"{asin}: {field} ASIN mismatch: {observed}")


def _asin(product: Mapping[str, Any]) -> str | None:
    provenance = product.get("text_provenance")
    url = (
        str(provenance.get("source_url") or "")
        if isinstance(provenance, Mapping)
        else ""
    )
    match = re.search(r"/dp/([A-Z0-9]{10})", url)
    return match.group(1) if match else None


def _required_entity_ids(root: Path, task_ids: set[str] | None) -> set[str]:
    tasks = [
        task for task in load_tasks(root / "data/tasks") if task["site"] == "egomart"
    ]
    if task_ids is not None:
        tasks = [task for task in tasks if str(task["id"]) in task_ids]
    required = {
        entity_id
        for task in tasks
        for entity_id in task_detail_entity_ids(
            task,
            root=root,
            include_related=True,
        )
    }
    return required


def _parse_detail(asin: str, body: bytes, response_url: str) -> dict[str, Any]:
    soup = BeautifulSoup(body, "html.parser")
    title = _text(soup.select_one("#productTitle"))
    if not title:
        page_title = _text(soup.select_one("title"))
        title = page_title.split(" : Amazon.ca", 1)[0].strip()
    if not title:
        raise ValueError(f"{asin}: product detail page has no title")

    byline = _text(soup.select_one("#bylineInfo"))
    brand = re.sub(r"^(Visit the|Brand:)\s+", "", byline, flags=re.IGNORECASE)
    brand = re.sub(r"\s+Store$", "", brand, flags=re.IGNORECASE).strip()
    price = (
        _text(
            soup.select_one("#corePrice_feature_div .a-offscreen")
            or soup.select_one("#priceblock_ourprice")
            or soup.select_one("#priceblock_dealprice")
        )
        or None
    )
    features = [
        text
        for node in soup.select("#feature-bullets li span.a-list-item")
        if (text := _text(node))
    ]
    details: dict[str, str] = {}
    for row in soup.select(
        "#productDetails_techSpec_section_1 tr, #productDetails_detailBullets_sections1 tr, "
        "table.prodDetTable tr, #productOverview_feature_div tr"
    ):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) != 2:
            continue
        key, value = (_text(cell) for cell in cells)
        if key and value:
            if _identity_text(key) == "ASIN":
                validate_product_identity(asin, {"product_details": {key: value}})
            details[key] = value
    for item in soup.select("#detailBullets_feature_div li"):
        text = _text(item)
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        key = key.replace("\u200e", "").strip()
        value = value.strip()
        if key and value:
            if _identity_text(key) == "ASIN":
                validate_product_identity(asin, {"product_details": {key: value}})
            details.setdefault(key, value)
    description = _text(soup.select_one("#productDescription"))
    if description:
        details["Product description"] = description
    image = soup.select_one("#landingImage")
    image_url = ""
    if image:
        image_url = str(image.get("data-old-hires") or image.get("src") or "").strip()
    payload = {
        "schema_version": 1,
        "provider": "amazon-ca-product-detail-v1",
        "asin": asin,
        "page_url": f"https://www.amazon.ca/dp/{asin}",
        "response_url": response_url,
        "retrieved_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "response_sha256": hashlib.sha256(body).hexdigest(),
        "title": title,
        "brand": brand or None,
        "price": price,
        "feature_bullets": features,
        "product_details": details,
        "image_url": image_url or None,
    }
    for page_asin in soup.select("input#ASIN[value]"):
        observed_asin = str(page_asin["value"])
        validate_product_identity(asin, {"observed_page_asin": observed_asin})
        payload["observed_page_asin"] = observed_asin
    canonical = soup.select_one('link[rel="canonical"][href]')
    if canonical:
        payload["canonical_url"] = str(canonical["href"])
    validate_product_identity(asin, payload)
    return payload


def _rich(payload: Mapping[str, Any]) -> bool:
    title = str(payload.get("title") or "").strip()
    brand = str(payload.get("brand") or "").strip()
    if not _product_title(title) or title.casefold() == brand.casefold():
        return False
    features = payload.get("feature_bullets") or []
    details = payload.get("product_details") or {}
    return any(_product_feature(value) for value in features) or any(
        _product_detail(key, value) for key, value in details.items()
    )


def _product_title(title: str) -> bool:
    return bool(
        title.strip()
        and not re.match(r"^Amazon\.(?:ca|com|co\.uk)(?:\b|:)", title, re.IGNORECASE)
        and title.casefold()
        not in {
            "robot check",
            "sorry, there was a problem.",
            "product summary presents key product information",
        }
    )


def _product_feature(value: Any) -> bool:
    if not isinstance(value, str) or len(value.split()) < 2:
        return False
    return not re.match(
        r"^(?:brand:|manufacturer:|asin:|see more\b|report\b|free delivery\b|"
        r"make sure this fits\b|about this item\b|amazon\.(?:ca|com)\b)",
        value.strip(),
        re.IGNORECASE,
    )


def _product_detail(key: Any, value: Any) -> bool:
    identity_only = {
        "brand",
        "brand name",
        "manufacturer",
        "asin",
        "customer reviews",
        "best sellers rank",
        "date first available",
        "department",
    }
    return bool(
        isinstance(key, str)
        and isinstance(value, str)
        and key.strip().casefold() not in identity_only
        and key.strip()
        and value.strip()
    )


def _clean_markdown(text: str) -> str:
    text = re.sub(r"!\[[^]]*]\([^)]*\)", "", text)
    text = re.sub(r"\[([^]]+)]\([^)]*\)", r"\1", text)
    text = text.replace("**", "").replace("__", "")
    return " ".join(text.split()).strip(" -*")


def _reader_section(lines: list[str], heading: str) -> list[str]:
    wanted = heading.lstrip("# ").casefold()
    headings = [
        (index, match)
        for index, line in enumerate(lines)
        if (match := re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line))
    ]
    found = next(
        (
            (index, match)
            for index, match in headings
            if match.group(2).casefold() == wanted
        ),
        None,
    )
    if found is None:
        return []
    index, match = found
    start, level = index + 1, len(match.group(1))
    end = next(
        (
            index
            for index, match in headings
            if index >= start and len(match.group(1)) <= level
        ),
        len(lines),
    )
    return lines[start:end]


def _reader_title(lines: list[str]) -> str:
    candidates: list[str] = []
    for index, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+About this item\s*$", line, re.IGNORECASE):
            break
        if line.startswith("# "):
            candidates.append(_clean_markdown(line[2:]))
        elif index + 1 < len(lines) and re.fullmatch(r"={3,}\s*", lines[index + 1]):
            candidates.append(_clean_markdown(line))
    titles = list(dict.fromkeys(title for title in candidates if _product_title(title)))
    if len(titles) == 1:
        return titles[0]
    if len(titles) > 1:
        raise ValueError("reader has ambiguous product headings")
    for line in lines:
        summary = re.match(r"^#{1,6}\s+Product Summary:\s*(.+)$", line, re.IGNORECASE)
        if summary and _product_title(summary.group(1).strip()):
            return summary.group(1).strip()
    for line in lines:
        match = re.match(r"^Title:\s*(.+?)\s*:\s*Amazon\.ca(?::|$)", line)
        if match and _product_title(match.group(1).strip()):
            return match.group(1).strip()
    raise ValueError("reader has no descriptive product title")


def _parse_reader_markdown(
    asin: str,
    markdown: str,
    base: Mapping[str, Any],
    reader_url: str,
) -> dict[str, Any]:
    lines = markdown.splitlines()
    payload = dict(base)
    payload["title"] = _reader_title(lines)
    source = re.search(r"^URL Source:\s*(\S+)\s*$", markdown, re.MULTILINE)
    if source is None:
        raise ValueError(f"{asin}: reader response has no canonical source URL")
    payload["reader_source_url"] = source.group(1)
    payload["asin"] = asin
    payload["page_url"] = f"https://www.amazon.ca/dp/{asin}"
    payload["brand"] = None

    brand_patterns = (
        r"^\[Brand:\s*([^]]+)]\(",
        r"^\|\s*Brand\s*\|\s*([^|]+?)\s*\|\s*$",
        r"^Brand\s+([^\n]+)$",
    )
    first_section = next(
        (
            index
            for index, line in enumerate(lines)
            if re.match(r"^#{1,6}\s+About this item\s*$", line, re.IGNORECASE)
        ),
        len(lines),
    )
    brand_context = "\n".join(lines[:first_section])
    for pattern in brand_patterns:
        match = re.search(pattern, brand_context, re.MULTILINE | re.IGNORECASE)
        if match:
            brand = _clean_markdown(match.group(1))
            if brand and len(brand) <= 120:
                payload["brand"] = brand
                break

    about = _reader_section(lines, "About this item")
    features = [
        cleaned
        for line in about
        if line.lstrip().startswith("*")
        and _product_feature(cleaned := _clean_markdown(line.lstrip()[1:]))
    ]
    payload["feature_bullets"] = list(dict.fromkeys(features))

    information = _reader_section(lines, "Product information")
    details: dict[str, str] = {}
    for line in information:
        match = re.match(r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*$", line)
        if not match:
            continue
        key = _clean_markdown(match.group(1))
        value = _clean_markdown(match.group(2))
        if (
            key
            and value
            and set(key) != {"-"}
            and set(value) != {"-"}
            and len(key) <= 120
        ):
            if _identity_text(key) == "ASIN":
                validate_product_identity(asin, {"product_details": {key: value}})
            details[key] = value
    payload["product_details"] = details
    validate_product_identity(asin, payload)

    payload["provider"] = "amazon-ca-product-detail-v1+jina-reader-v1"
    payload["reader_url"] = reader_url
    payload["reader_response_sha256"] = hashlib.sha256(markdown.encode()).hexdigest()
    payload["reader_retrieved_at"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["reader_capture_method"] = "public_reader_markdown_product_sections"
    payload["reader_response_hash_scope"] = "reader_response_body"
    if not _rich(payload):
        raise ValueError(f"{asin}: reader detail remains sparse")
    return payload


def _reader_capture(asin: str, base: Mapping[str, Any]) -> dict[str, Any]:
    reader_url = READER_BASE + quote(asin, safe="")
    request = Request(
        reader_url,
        headers={"User-Agent": READER_USER_AGENT, "Accept": "text/plain"},
    )
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=90) as response:
                markdown = response.read().decode("utf-8")
            break
        except Exception as exc:
            if isinstance(exc, HTTPError) and exc.code == 429:
                raise
            error = exc
            time.sleep(0.5 * (attempt + 1))
    else:
        raise RuntimeError(f"{asin}: reader request failed: {error}")
    return _parse_reader_markdown(asin, markdown, base, reader_url)


def _capture(
    asin: str,
    *,
    enrich_sparse: bool = False,
    existing: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if existing is not None:
        payload = dict(existing)
    else:
        try:
            payload = _direct_capture(asin)
        except Exception as exc:
            if not enrich_sparse:
                raise
            payload = {
                "schema_version": 1,
                "asin": asin,
                "page_url": f"https://www.amazon.ca/dp/{asin}",
                "direct_capture_error": f"{type(exc).__name__}: {exc}",
            }
    if enrich_sparse and not _rich(payload):
        payload = _reader_capture(asin, payload)
    validate_product_identity(asin, payload)
    if not _rich(payload):
        raise ValueError(
            f"{asin}: product detail remains sparse; no decision-bearing metadata"
        )
    return asin, payload


def _direct_capture(asin: str) -> dict[str, Any]:
    url = f"https://www.amazon.ca/dp/{asin}"
    request = Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept-Language": "en-CA,en;q=0.9",
            "Accept-Encoding": "identity",
        },
    )
    with urlopen(request, timeout=60) as response:
        body = response.read()
        response_url = response.geturl()
    return _parse_detail(asin, body, response_url)


def capture(
    root: Path,
    *,
    entity_ids: set[str] | None,
    task_ids: set[str] | None,
    concurrency: int,
    force: bool,
    enrich_sparse: bool,
) -> dict[str, int]:
    maximum_concurrency = 8 if enrich_sparse else 4
    if not 1 <= concurrency <= maximum_concurrency:
        raise ValueError(f"concurrency must be between 1 and {maximum_concurrency}")
    catalog = json.loads((root / CATALOG).read_text())["products"]
    by_entity = {str(product["item_id"]): product for product in catalog}
    requested = entity_ids or _required_entity_ids(root, task_ids)
    unknown = sorted(requested - set(by_entity))
    if unknown:
        raise ValueError(f"unknown EgoMart entity IDs: {unknown}")
    pairs = {
        asin: entity_id
        for entity_id in sorted(requested)
        if (asin := _asin(by_entity[entity_id])) is not None
    }
    output = root / OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    for asin in pairs:
        path = output / f"{asin}.json"
        if path.is_file():
            existing[asin] = json.loads(path.read_text())
            if not force:
                validate_product_identity(asin, existing[asin])
    pending = [
        asin
        for asin in pairs
        if force or asin not in existing or not _rich(existing[asin])
    ]
    completed = 0
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _capture,
                asin,
                enrich_sparse=enrich_sparse,
                existing=(None if force else existing.get(asin)),
            ): asin
            for asin in pending
        }
        for future in as_completed(futures):
            asin = futures[future]
            try:
                got_asin, payload = future.result()
                if got_asin != asin:
                    raise ValueError(f"capture ASIN mismatch: {got_asin}")
                validate_product_identity(asin, payload)
                if not _rich(payload):
                    raise ValueError("captured detail remains sparse")
            except Exception as exc:
                failed.append(f"{asin}: {type(exc).__name__}: {exc}")
                continue
            previous = existing.get(got_asin)
            if previous is not None:
                history = list(previous.get("capture_history") or [])
                history.append(
                    {
                        field: previous[field]
                        for field in (
                            "provider",
                            "response_url",
                            "response_sha256",
                            "retrieved_at",
                            "title",
                            "reader_url",
                            "reader_response_sha256",
                            "reader_retrieved_at",
                        )
                        if field in previous
                    }
                )
                payload["capture_history"] = history
            write_json_atomic(output / f"{got_asin}.json", payload)
            completed += 1
            print(f"captured {completed}/{len(pending)} {got_asin}", flush=True)
    if failed:
        raise RuntimeError(
            f"Amazon detail capture failed for {len(failed)} listings:\n"
            + "\n".join(failed[:30])
        )
    return {
        "requested": len(pairs),
        "reused": len(pairs) - len(pending),
        "captured": completed,
        "failed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--entity-id", action="append")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--enrich-sparse",
        action="store_true",
        help="Use a public text mirror when the direct page contains only a title.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            capture(
                Path(args.root).resolve(),
                entity_ids=set(args.entity_id) if args.entity_id else None,
                task_ids=set(args.task_id) if args.task_id else None,
                concurrency=args.concurrency,
                force=args.force,
                enrich_sparse=args.enrich_sparse,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
