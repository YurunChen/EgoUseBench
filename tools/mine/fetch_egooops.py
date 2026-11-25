"""Fetch explicitly selected EgoOops recordings from the official ZIP by HTTP range.

Raw compressed members are retained for resumable acquisition. ZIP CRC verifies
transfer integrity; the publisher provides no cryptographic checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import stat
import struct
import subprocess
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import requests

VIDEO_URL = "https://www.lsta.media.kyoto-u.ac.jp/resource/data/EgoOops/videos-processed-720p.zip"
OUT_ROOT = "reference/sources/EgoOops"
MEMBER_PREFIX = "videos-processed-720p"
CHUNK_BYTES = 1 << 20
BYTES_PER_GB = 1_000_000_000
LOCAL_HEADER = struct.Struct("<4s5H3I2H")
RECORDING_ID = re.compile(r"(?:blacklight|cardboard|electronics|ion|tsumiki)/S\d{7}$")


class FetchError(RuntimeError):
    """Acquisition failed; partial data must not be treated as source media."""


@dataclass(frozen=True)
class RemoteArchive:
    url: str
    size: int
    etag: str

    def open_range(self, start: int, stop: int) -> requests.Response:
        if not 0 <= start < stop <= self.size:
            raise FetchError(f"invalid archive range: {start}:{stop}")
        response = requests.get(
            self.url,
            headers={
                "Range": f"bytes={start}-{stop - 1}",
                "If-Match": self.etag,
                "Accept-Encoding": "identity",
            },
            stream=True,
            timeout=(30, 60),
        )
        expected = f"bytes {start}-{stop - 1}/{self.size}"
        if (
            response.status_code != 206
            or response.headers.get("Content-Range") != expected
            or response.headers.get("ETag") != self.etag
            or response.headers.get("Content-Encoding", "identity") != "identity"
        ):
            response.close()
            raise FetchError("server ignored the range or the archive changed")
        return response


class ArchiveReader(io.RawIOBase):
    """A small, bounded seekable reader for ZIP headers and the central directory."""

    def __init__(self, archive: RemoteArchive):
        self.archive = archive
        self.position = 0

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        bases = {
            io.SEEK_SET: 0,
            io.SEEK_CUR: self.position,
            io.SEEK_END: self.archive.size,
        }
        position = bases[whence] + offset
        if not 0 <= position <= self.archive.size:
            raise FetchError("ZIP seek outside the archive")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        remaining = self.archive.size - self.position
        count = remaining if size < 0 else min(size, remaining)
        if count > CHUNK_BYTES:
            raise FetchError("unexpectedly large ZIP metadata read")
        if not count:
            return b""
        with self.archive.open_range(self.position, self.position + count) as response:
            data = response.raw.read(count + 1)
        if len(data) != count:
            raise FetchError("incomplete ZIP metadata range")
        self.position += count
        return data


@dataclass(frozen=True)
class Transfer:
    recording_id: str
    member: zipfile.ZipInfo
    offset: int
    compressed_path: Path
    destination: Path

    @property
    def partial(self) -> Path:
        return self.compressed_path.with_suffix(self.compressed_path.suffix + ".part")

    def pending_bytes(self) -> int:
        for path in (self.compressed_path, self.partial):
            if path.exists():
                if path.is_symlink() or not path.is_file():
                    raise FetchError(f"unsafe local cache path: {path}")
                size = path.stat().st_size
                if size > self.member.compress_size:
                    raise FetchError(f"oversized compressed cache: {path}")
                if path == self.compressed_path and size != self.member.compress_size:
                    raise FetchError(f"incomplete finalized compressed cache: {path}")
                return self.member.compress_size - size
        return self.member.compress_size


def probe_archive() -> RemoteArchive:
    with requests.head(
        VIDEO_URL,
        headers={"Accept-Encoding": "identity"},
        allow_redirects=True,
        timeout=30,
    ) as response:
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
        etag = response.headers.get("ETag", "")
    if size <= 0 or not etag or etag.startswith("W/"):
        raise FetchError("archive needs a positive size and a strong ETag for resume")
    return RemoteArchive(VIDEO_URL, size, etag)


def selected_recordings(text: str, metadata: dict) -> list[str]:
    ids = text.split(",")
    if any(not RECORDING_ID.fullmatch(recording) for recording in ids):
        raise FetchError("use explicit domain/S1234567 recording IDs")
    if len(ids) != len(set(ids)):
        raise FetchError("duplicate recording IDs")
    known = {f"{row['task_id']}/{row['video_id']}" for row in metadata["videos"]}
    if missing := set(ids) - known:
        raise FetchError(f"recordings absent from official metadata: {sorted(missing)}")
    return ids


def safe_destination(base: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise FetchError(f"unsafe archive path: {relative}")
    destination = base.joinpath(*path.parts)
    for parent in (destination, *destination.parents):
        if parent.is_symlink():
            raise FetchError(f"symlink in destination: {parent}")
    if not destination.resolve().is_relative_to(base.resolve()):
        raise FetchError(f"destination escaped output directory: {relative}")
    return destination


def member_offset(reader: ArchiveReader, member: zipfile.ZipInfo) -> int:
    if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
        raise FetchError("encrypted or symlink ZIP members are not supported")
    if member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise FetchError("only stored and deflated video members are supported")
    reader.seek(member.header_offset)
    header = LOCAL_HEADER.unpack(reader.read(LOCAL_HEADER.size))
    if (
        header[0] != b"PK\x03\x04"
        or header[2] != member.flag_bits
        or header[3] != member.compress_type
    ):
        raise FetchError("ZIP local header does not match its central directory")
    filename = reader.read(header[9]).decode("utf-8" if header[2] & 0x800 else "cp437")
    if filename != member.filename:
        raise FetchError("ZIP local filename does not match its central directory")
    offset = reader.tell() + header[10]
    if (
        member.file_size <= 0
        or member.compress_size <= 0
        or offset + member.compress_size > reader.archive.size
    ):
        raise FetchError("invalid ZIP member size or offset")
    return offset


def plan_transfers(
    archive: RemoteArchive, ids: list[str], output: Path
) -> list[Transfer]:
    fingerprint = hashlib.sha256(archive.etag.encode()).hexdigest()[:16]
    transfers = []
    reader = ArchiveReader(archive)
    with zipfile.ZipFile(reader) as directory:
        for recording in ids:
            member_name = f"{MEMBER_PREFIX}/{recording}.MP4"
            if directory.namelist().count(member_name) != 1:
                raise FetchError(f"ZIP must contain exactly one member for {recording}")
            member = directory.getinfo(member_name)
            transfers.append(
                Transfer(
                    recording,
                    member,
                    member_offset(reader, member),
                    safe_destination(
                        output / "archive-parts", f"{recording}.{fingerprint}.deflate"
                    ),
                    safe_destination(output / "videos", f"{recording}.MP4"),
                )
            )
    return transfers


def fetch_compressed(archive: RemoteArchive, transfer: Transfer) -> None:
    pending = transfer.pending_bytes()
    if pending:
        transfer.partial.parent.mkdir(parents=True, exist_ok=True)
        have = transfer.member.compress_size - pending
        start = transfer.offset + have
        with archive.open_range(
            start, transfer.offset + transfer.member.compress_size
        ) as response:
            with transfer.partial.open("ab") as handle:
                for chunk in response.iter_content(CHUNK_BYTES):
                    if len(chunk) > pending:
                        raise FetchError(
                            "server sent bytes beyond the selected ZIP member"
                        )
                    handle.write(chunk)
                    pending -= len(chunk)
        if pending:
            raise FetchError(
                f"incomplete download: {transfer.recording_id}; rerun to resume"
            )
    if transfer.partial.exists() and not transfer.compressed_path.exists():
        transfer.partial.rename(transfer.compressed_path)


def checksum_video(path: Path) -> tuple[int, int, str]:
    size, crc = 0, 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            size += len(chunk)
            crc = zlib.crc32(chunk, crc)
            digest.update(chunk)
    return size, crc, digest.hexdigest()


def extract_video(transfer: Transfer) -> tuple[str, dict]:
    if transfer.destination.exists():
        size, crc, digest = checksum_video(transfer.destination)
        if (size, crc) != (transfer.member.file_size, transfer.member.CRC):
            raise FetchError(f"existing video failed size/CRC: {transfer.destination}")
        return digest, probe_video(transfer.destination)
    transfer.destination.parent.mkdir(parents=True, exist_ok=True)
    partial = transfer.destination.with_suffix(".MP4.part")
    safe_destination(transfer.destination.parent, partial.name)
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    total = 0
    with transfer.compressed_path.open("rb") as source, partial.open("wb") as target:
        while chunk := source.read(CHUNK_BYTES):
            if transfer.member.compress_type == zipfile.ZIP_DEFLATED:
                chunk = decompressor.decompress(
                    chunk, transfer.member.file_size - total + 1
                )
            total += len(chunk)
            if total > transfer.member.file_size:
                raise FetchError("extracted data exceeds the advertised video size")
            target.write(chunk)
    if transfer.member.compress_type == zipfile.ZIP_DEFLATED:
        if (
            not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
        ):
            raise FetchError("incomplete or oversized deflate stream")
    size, crc, digest = checksum_video(partial)
    if (size, crc) != (transfer.member.file_size, transfer.member.CRC):
        raise FetchError(f"ZIP size/CRC mismatch: {transfer.recording_id}")
    video = probe_video(partial)
    partial.rename(transfer.destination)
    return digest, video


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    video = json.loads(result.stdout)
    if not video.get("streams") or float(video["format"]["duration"]) <= 0:
        raise FetchError(f"ffprobe found no valid video stream: {path}")
    return video


def fetch_verified(archive: RemoteArchive, transfer: Transfer) -> dict:
    fetch_compressed(archive, transfer)
    digest, video = extract_video(transfer)
    return {
        "local_sha256": digest,
        "compressed_path": str(transfer.compressed_path),
        "ffprobe": video,
    }


def save_receipt(path: Path, receipt: dict) -> None:
    partial = safe_destination(path.parent, path.name + ".part")
    partial.write_text(json.dumps(receipt, indent=2) + "\n")
    partial.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--recordings", required=True, help="comma-separated domain/recording IDs"
    )
    parser.add_argument("--max-gb", type=float, default=6.0)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.max_gb <= 6:
        raise FetchError("--max-gb must be positive and at most 6 GB")
    root = Path(args.root).resolve()
    output = root / OUT_ROOT
    metadata_path = output / "annotations/meta/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    ids = selected_recordings(args.recordings, metadata)
    receipt_path = safe_destination(output, "acquisition.json")
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    archive = probe_archive()
    if receipt and (receipt["etag"], receipt["archive_size_bytes"]) != (
        archive.etag,
        archive.size,
    ):
        raise FetchError(
            "archive changed since the saved acquisition; inspect before mixing versions"
        )
    transfers = plan_transfers(archive, ids, output)
    pending = sum(transfer.pending_bytes() for transfer in transfers)
    extraction = sum(
        transfer.member.file_size
        for transfer in transfers
        if not transfer.destination.exists()
    )
    available = shutil.disk_usage(root).free
    plan: dict[str, Any] = {
        "archive_url": archive.url,
        "archive_size_bytes": archive.size,
        "etag": archive.etag,
        "official_cryptographic_checksum": None,
        "integrity": "publisher ZIP CRC32; local SHA256",
        "pending_bytes": pending,
        "max_download_bytes": int(args.max_gb * BYTES_PER_GB),
        "additional_disk_bytes": pending + extraction,
        "disk_free_bytes": available,
        "resume": "append missing compressed member bytes; strong ETag must match; retain raw compressed members",
        "recordings": [
            {
                "recording_id": transfer.recording_id,
                "destination": str(transfer.destination),
                "compressed_bytes": transfer.member.compress_size,
                "video_bytes": transfer.member.file_size,
                "zip_crc32": f"{transfer.member.CRC:08x}",
                "offset": transfer.offset,
                "member": transfer.member.filename,
                "compression_method": transfer.member.compress_type,
            }
            for transfer in transfers
        ],
    }
    print(json.dumps(plan, indent=2), flush=True)
    if pending > plan["max_download_bytes"] or pending + extraction > available:
        raise FetchError("download budget or free disk capacity exceeded")
    if args.dry_run:
        return 0
    receipt.update({key: value for key, value in plan.items() if key != "recordings"})
    receipt["annotations_revision"] = subprocess.check_output(
        ["git", "-C", str(output / "annotations"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    receipt["license"] = "CC-BY-SA-4.0"
    receipt["project_url"] = "https://y-haneji.github.io/EgoOops-project-page/"
    receipt["annotations_url"] = "https://github.com/Y-Haneji/EgoOops-annotations/"
    receipt["license_path"] = str(output / "annotations/LICENSE")
    recordings = receipt.setdefault("recordings", {})
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending_jobs = {
            pool.submit(fetch_verified, archive, transfer): row
            for transfer, row in zip(transfers, plan["recordings"], strict=True)
        }
        for future in as_completed(pending_jobs):
            row = pending_jobs[future]
            recording = row["recording_id"]
            recordings[recording] = {**row, **future.result()}
            save_receipt(receipt_path, receipt)
            print(f"verified {recording}: ZIP CRC, local SHA256, ffprobe", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
