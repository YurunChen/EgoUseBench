"""Fetch CaptainCook4D recordings selectively, resolution by resolution.

The upstream downloader fetches every recording at once; the full 4K set is
about 1.7 TB, which no screening pass needs. Screening runs on 360P and only
the recordings that become tasks are refetched at 4K, so this script takes a
recording filter and reports the byte cost before transferring anything.

Usage:
    uv run python tools/mine/fetch_captaincook4d.py --root . --errors-only --limit 24
    uv run python tools/mine/fetch_captaincook4d.py --root . --resolution 4k --recordings 1_10,2_7
"""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egocua.workspace import read_json_strict

_SOURCE_ROOT = "reference/sources/CaptainCook4D"
_DOWNLOADER = f"{_SOURCE_ROOT}/downloader/metadata/download_links.json"
_ERRORS = f"{_SOURCE_ROOT}/annotations/annotation_json/error_annotations.json"
_OUT_ROOT = f"{_SOURCE_ROOT}/videos"
_URL_KEY = {"360p": "gopro_360p", "4k": "gopro_4k"}
_USER_AGENT = "egoCUAbench-fetch/1.0 (local research benchmark)"
_CHUNK = 1 << 20
_ATTEMPTS = 5
_BACKOFF = 3.0


class FetchError(RuntimeError):
    """Raised when a recording cannot be resolved or transferred."""


def _open(request: urllib.request.Request, *, timeout: int):
    """Retry through the flaky egress proxy, which returns sporadic 503 tunnels."""

    last: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            last = exc
            time.sleep(_BACKOFF * (attempt + 1))
    raise FetchError(f"{request.full_url}: {last}")


def remote_size(url: str, *, timeout: int = 30) -> int:
    """Box rejects HEAD on shared links, so ask for one byte and read the range."""

    request = urllib.request.Request(
        url, headers={"Range": "bytes=0-0", "User-Agent": _USER_AGENT}
    )
    with _open(request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range", "")
        status = response.status
    if status != 206 or not content_range.startswith("bytes 0-0/"):
        raise FetchError(f"no size in Content-Range for {url}")
    total = content_range.rsplit("/", 1)[1]
    if not total.isdigit() or int(total) <= 0:
        raise FetchError(f"invalid remote size for {url}: {content_range}")
    return int(total)


def error_recording_ids(error_annotations: list[dict[str, Any]]) -> set[str]:
    """The recordings CaptainCook4D annotates as containing an execution error."""

    return {
        str(record["recording_id"])
        for record in error_annotations
        if record.get("is_error")
    }


def named_recordings(links: dict[str, Any], recordings: list[str]) -> list[str]:
    if len(recordings) != len(set(recordings)):
        raise FetchError("recording ids must be unique")
    missing = [rid for rid in recordings if rid not in links]
    if missing:
        raise FetchError(f"unknown recording ids: {', '.join(sorted(missing))}")
    return recordings


@dataclass(frozen=True)
class Transfer:
    """One recording to fetch: where it comes from and where it lands."""

    recording_id: str
    url: str
    destination: Path
    size: int = 0

    @property
    def complete(self) -> bool:
        return (
            self.destination.is_file() and self.destination.stat().st_size == self.size
        )


def plan_downloads(
    links: dict[str, Any], recording_ids: list[str], resolution: str, out_dir: Path
) -> list[Transfer]:
    key = _URL_KEY[resolution]
    missing = [rid for rid in recording_ids if not links[rid].get(key)]
    if missing:
        raise FetchError(f"no {resolution} URL for: {', '.join(missing)}")
    return [
        Transfer(
            recording_id=recording_id,
            url=links[recording_id][key],
            destination=out_dir / f"{recording_id}_{resolution}.mp4",
        )
        for recording_id in recording_ids
    ]


def resolve_sizes(plan: list[Transfer]) -> list[Transfer]:
    """Ask the host how big each file is; Box only answers a ranged GET."""

    return [
        Transfer(
            transfer.recording_id,
            transfer.url,
            transfer.destination,
            remote_size(transfer.url),
        )
        for transfer in plan
    ]


def download(url: str, destination: Path, *, expected: int) -> int:
    """Resume until the file is whole; return the bytes transferred by this call."""

    if expected <= 0:
        raise FetchError("expected byte length must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == expected:
        return 0
    partial = destination.with_suffix(destination.suffix + ".part")
    transferred = 0
    for attempt in range(_ATTEMPTS):
        have = partial.stat().st_size if partial.exists() else 0
        if have >= expected:
            break
        headers = {"User-Agent": _USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with _open(request, timeout=120) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {have}-"):
                        raise FetchError(f"unexpected resume range: {content_range}")
                mode = "ab" if have and response.status == 206 else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(_CHUNK):
                        handle.write(chunk)
                        transferred += len(chunk)
        except (FetchError, urllib.error.URLError, TimeoutError):
            # A dropped tunnel leaves a shorter .part; the next attempt resumes it.
            time.sleep(_BACKOFF * (attempt + 1))
            continue
    final = partial.stat().st_size if partial.exists() else 0
    if final != expected:
        raise FetchError(f"{destination.name}: got {final} bytes, expected {expected}")
    partial.rename(destination)
    return transferred


BYTES_PER_GB = 1e9
BYTES_PER_MB = 1e6


def _selected_ids(
    args: argparse.Namespace, links: dict[str, Any], root: Path
) -> list[str]:
    if args.recordings:
        selected = named_recordings(
            links, [value.strip() for value in args.recordings.split(",")]
        )
    elif args.errors_only:
        with_errors = error_recording_ids(read_json_strict(root / _ERRORS))
        selected = [
            recording_id for recording_id in links if recording_id in with_errors
        ]
    else:
        selected = list(links)
    return selected[: args.limit] if args.limit else selected


def _transfer(plan: list[Transfer], pending: list[Transfer]) -> None:
    done = 0
    for index, transfer in enumerate(pending, start=1):
        download(transfer.url, transfer.destination, expected=transfer.size)
        done += transfer.size
        print(
            f"[{index}/{len(pending)}] {transfer.recording_id} "
            f"{transfer.size / BYTES_PER_MB:.0f} MB ({done / BYTES_PER_GB:.1f} GB done)",
            flush=True,
        )
    print(f"complete: {len(plan)} recordings")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="workspace root")
    parser.add_argument("--resolution", choices=sorted(_URL_KEY), default="360p")
    parser.add_argument(
        "--errors-only", action="store_true", help="only recordings annotated is_error"
    )
    parser.add_argument("--recordings", help="comma-separated recording ids")
    parser.add_argument("--limit", type=int, help="stop after this many recordings")
    parser.add_argument(
        "--max-gb", type=float, default=25.0, help="refuse a plan larger than this"
    )
    parser.add_argument("--dry-run", action="store_true", help="report the plan only")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    links = read_json_strict(root / _DOWNLOADER)
    out_dir = root / _OUT_ROOT / args.resolution
    plan = plan_downloads(
        links, _selected_ids(args, links, root), args.resolution, out_dir
    )

    print(
        f"resolving sizes for {len(plan)} recordings at {args.resolution}", flush=True
    )
    plan = resolve_sizes(plan)
    pending = [transfer for transfer in plan if not transfer.complete]
    total = sum(transfer.size for transfer in plan)
    remaining = sum(transfer.size for transfer in pending)
    print(
        f"plan {total / BYTES_PER_GB:.1f} GB total, {remaining / BYTES_PER_GB:.1f} GB to "
        f"transfer ({len(pending)} of {len(plan)} recordings missing)",
        flush=True,
    )
    print(f"limit: {args.max_gb} GB; official checksum: not published in link metadata")
    print("resume: sibling .mp4.part; completed files require the expected byte length")
    for transfer in plan:
        print(
            f"{transfer.recording_id}: {transfer.size} bytes -> {transfer.destination}"
        )
    if args.dry_run:
        return 0
    if remaining / BYTES_PER_GB > args.max_gb:
        raise FetchError(
            f"plan needs {remaining / BYTES_PER_GB:.1f} GB, over --max-gb {args.max_gb}"
        )
    _transfer(plan, pending)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
