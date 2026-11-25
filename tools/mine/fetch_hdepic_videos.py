"""Fetch only HD-EPIC videos selected by the expansion candidate pool.

Unlike the upstream downloader, this tool selects only MP4 files, reports the
remaining byte cost, enforces a hard capacity limit, resumes partial transfers,
and verifies the official MD5 before admitting a file as local source media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

BASE_URL = "https://data.bris.ac.uk/datasets/3cqb5b81wk2dc2379fx1mrxh47/Videos"
MD5_FILE = "reference/sources/HD-EPIC/downloader/data/md5.txt"
OUT_ROOT = "reference/sources/HD-EPIC/videos/HD-EPIC/Videos"
DEFAULT_POOL = "data/sources/audits/expansion_candidate_pool.json"
CONTENT_RANGE = re.compile(r"^content-range:\s*bytes\s+\d+-\d+/(\d+)\s*$", re.I | re.M)
BYTES_PER_GB = 1_000_000_000


class FetchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transfer:
    recording: str
    url: str
    destination: Path
    md5: str
    size: int


def _run_curl(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["curl", "--silent", "--show-error", "--location", "--fail", *arguments],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise FetchError(result.stderr.strip() or f"curl exited {result.returncode}")
    return result


def remote_size(url: str) -> int:
    result = _run_curl(
        [
            "--retry",
            "4",
            "--retry-all-errors",
            "--max-time",
            "60",
            "--range",
            "0-0",
            "--dump-header",
            "-",
            "--output",
            "/dev/null",
            url,
        ]
    )
    match = CONTENT_RANGE.search(result.stdout.replace("\r", ""))
    if not match:
        raise FetchError(f"server did not return Content-Range for {url}")
    return int(match.group(1))


def official_md5(path: Path) -> dict[str, str]:
    checksums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if name.endswith(".mp4"):
            checksums[Path(name).stem] = digest
    return checksums


def file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(4 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def selected_recordings(
    root: Path, recordings: str | None, pool_path: str
) -> list[str]:
    if recordings:
        return sorted(set(filter(None, recordings.split(","))))
    pool = json.loads((root / pool_path).read_text(encoding="utf-8"))
    return sorted(
        {
            row["media_id"]
            for row in pool["datasets"]["HD-EPIC"]
            if not row.get("media_available")
        }
    )


def plan(root: Path, recordings: list[str]) -> list[Transfer]:
    checksums = official_md5(root / MD5_FILE)
    transfers = []
    for recording in recordings:
        if recording not in checksums:
            raise FetchError(f"no official MP4 checksum for {recording}")
        participant = recording.split("-", 1)[0]
        url = f"{BASE_URL}/{participant}/{recording}.mp4"
        transfers.append(
            Transfer(
                recording=recording,
                url=url,
                destination=root / OUT_ROOT / participant / f"{recording}.mp4",
                md5=checksums[recording],
                size=remote_size(url),
            )
        )
    return transfers


def complete(transfer: Transfer) -> bool:
    return (
        transfer.destination.is_file()
        and transfer.destination.stat().st_size == transfer.size
        and file_md5(transfer.destination) == transfer.md5
    )


def download(transfer: Transfer) -> None:
    transfer.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = transfer.destination.with_suffix(".mp4.part")
    _run_curl(
        [
            "--retry",
            "8",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            transfer.url,
        ]
    )
    if partial.stat().st_size != transfer.size:
        raise FetchError(
            f"{transfer.recording}: got {partial.stat().st_size}, expected {transfer.size} bytes"
        )
    if file_md5(partial) != transfer.md5:
        raise FetchError(f"{transfer.recording}: official MD5 mismatch")
    os.replace(partial, transfer.destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--recordings", help="comma-separated recording IDs")
    parser.add_argument("--candidate-pool", default=DEFAULT_POOL)
    parser.add_argument("--max-gb", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    transfers = plan(
        root, selected_recordings(root, args.recordings, args.candidate_pool)
    )
    pending = [transfer for transfer in transfers if not complete(transfer)]
    remaining = sum(transfer.size for transfer in pending)
    print(
        f"selected={len(transfers)} pending={len(pending)} "
        f"remaining={remaining / BYTES_PER_GB:.2f} GB"
    )
    if args.dry_run:
        for transfer in pending:
            print(f"{transfer.recording}\t{transfer.size / BYTES_PER_GB:.2f} GB")
        return 0
    if remaining > args.max_gb * BYTES_PER_GB:
        raise FetchError(
            f"plan needs {remaining / BYTES_PER_GB:.2f} GB, over --max-gb {args.max_gb:g}"
        )
    for index, transfer in enumerate(pending, start=1):
        download(transfer)
        print(f"[{index}/{len(pending)}] verified {transfer.recording}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
