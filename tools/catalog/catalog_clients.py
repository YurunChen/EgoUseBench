"""Shared upstream clients for catalog tooling (Wikimedia, YouTube via yt-dlp).

Every script that fetches upstream listings or photos imports from here so the
retry policy, headers, and query shapes live in one place.
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_USER_AGENT = (
    "egoCUAbench-catalog/1.0 (local research benchmark; "
    "https://en.wikipedia.org/wiki/User:EgoCUAbench)"
)


def http_get(
    url: str,
    *,
    timeout: int = 20,
    attempts: int = 3,
    user_agent: str = DEFAULT_USER_AGENT,
    backoff: float = 0.3,
) -> bytes:
    """GET with retries; Wikimedia hosts get the referer and query-less URL they require."""

    parsed = urllib.parse.urlsplit(url)
    wikimedia = parsed.netloc.endswith("wikimedia.org")
    if wikimedia:
        url = urllib.parse.urlunsplit(parsed._replace(query="", fragment=""))
    headers = {
        "User-Agent": user_agent,
        "Accept": "image/jpeg,image/png,image/webp,application/json;q=0.9,*/*;q=0.1",
    }
    if wikimedia:
        headers["Referer"] = "https://en.wikipedia.org/"
    request = urllib.request.Request(url, headers=headers)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"download failed: {url}") from last


def wiki_geosearch(
    lat: float,
    lon: float,
    *,
    radius_m: int = 10000,
    limit: int = 50,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[dict[str, Any]]:
    """Wikipedia pages with coordinates near a point, with thumbnails and intros."""

    query = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "geosearch",
            "ggscoord": f"{lat}|{lon}",
            "ggsradius": radius_m,
            "ggslimit": limit,
            "prop": "coordinates|pageimages|extracts|info",
            "piprop": "thumbnail",
            "pithumbsize": "1000",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": limit,
            "inprop": "url",
        }
    )
    payload = json.loads(
        http_get(
            f"https://en.wikipedia.org/w/api.php?{query}", user_agent=user_agent
        ).decode()
    )
    if payload.get("error"):
        raise RuntimeError(payload["error"].get("info") or payload["error"])
    return list(payload.get("query", {}).get("pages") or [])


def yt_search_json(
    query: str,
    *,
    limit: int,
    flat: bool,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """Run ``yt-dlp ytsearch`` and return one JSON object per result line."""

    command = [
        "yt-dlp",
        "--dump-json",
        "--no-download",
        "--no-warnings",
    ]
    if flat:
        command.insert(1, "--flat-playlist")
    else:
        command.append("--skip-download")
    command.append(f"ytsearch{limit}:{query}")
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"YouTube search failed ({result.returncode}): {result.stderr.strip()}"
        )
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(result.stdout.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"YouTube search returned invalid JSON on line {number}"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"YouTube search returned a non-object on line {number}")
        rows.append(row)
    return rows


def yt_video_json(video_id: str, *, timeout: int = 120) -> dict[str, Any]:
    """Fetch one public YouTube listing by its stable video id."""

    command = [
        "yt-dlp",
        "--dump-single-json",
        "--skip-download",
        "--no-warnings",
        "--ignore-no-formats-error",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or payload.get("id") != video_id:
        raise RuntimeError(f"unexpected YouTube listing for {video_id}")
    return payload
