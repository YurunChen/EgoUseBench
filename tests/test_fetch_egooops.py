from __future__ import annotations

import io
import json
import re
import stat
import subprocess
import zipfile

import pytest
import requests

from tools.mine import fetch_egooops as fetch


RECORDING = "electronics/S1790010"
PAYLOAD = b"unaltered source video bytes" * 80


def response_for(data: bytes, start: int, stop: int) -> requests.Response:
    response = requests.Response()
    response.status_code = 206
    response.headers.update(
        {"Content-Range": f"bytes {start}-{stop - 1}/{len(data)}", "ETag": '"stable"'}
    )
    response.raw = io.BytesIO(data[start:stop])
    return response


@pytest.fixture
def archive(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as writer:
        with writer.open(
            f"{fetch.MEMBER_PREFIX}/{RECORDING}.MP4", "w", force_zip64=True
        ) as video:
            video.write(PAYLOAD)
    data = buffer.getvalue()
    calls = []

    def get(url, *, headers, stream, timeout):
        start, last = map(
            int, re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"]).groups()
        )
        calls.append((start, last + 1))
        return response_for(data, start, last + 1)

    monkeypatch.setattr(fetch.requests, "get", get)
    return fetch.RemoteArchive(
        "https://official.example/source.zip", len(data), '"stable"'
    ), calls


@pytest.fixture
def transfer(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch, "probe_video", lambda path: {"test_fixture": "not real media"}
    )
    remote, _ = archive
    return fetch.plan_transfers(remote, [RECORDING], tmp_path)[0]


@pytest.mark.parametrize(
    "recordings",
    ["", "electronics/../S1790010", "electronics/S1", "electronics/S1790010,"],
)
def test_selected_recordings_rejects_malformed_ids(recordings):
    with pytest.raises(fetch.FetchError, match="explicit"):
        fetch.selected_recordings(recordings, {"videos": []})


def test_selected_recordings_rejects_duplicates():
    with pytest.raises(fetch.FetchError, match="duplicate"):
        fetch.selected_recordings(f"{RECORDING},{RECORDING}", {"videos": []})


def test_selected_recordings_rejects_ids_without_annotations():
    with pytest.raises(fetch.FetchError, match="absent"):
        fetch.selected_recordings(RECORDING, {"videos": []})


def test_selected_recordings_accepts_annotated_ids():
    metadata = {"videos": [{"task_id": "electronics", "video_id": "S1790010"}]}
    assert fetch.selected_recordings(RECORDING, metadata) == [RECORDING]


@pytest.mark.parametrize(
    "name", ["../escape.mp4", "/absolute.mp4", "folder\\escape.mp4"]
)
def test_safe_destination_rejects_unsafe_names(tmp_path, name):
    with pytest.raises(fetch.FetchError, match="unsafe"):
        fetch.safe_destination(tmp_path, name)


def test_safe_destination_rejects_symlinked_parent(tmp_path):
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(fetch.FetchError, match="symlink"):
        fetch.safe_destination(tmp_path, "link/video.MP4")


@pytest.mark.parametrize(
    "header,value",
    [
        ("status", 200),
        ("ETag", '"changed"'),
        ("Content-Range", "bytes 0-3/8"),
        ("Content-Encoding", "gzip"),
    ],
)
def test_range_request_rejects_ignored_or_changed_range(monkeypatch, header, value):
    response = response_for(b"data", 0, 4)
    if header == "status":
        response.status_code = value
    else:
        response.headers[header] = value
    monkeypatch.setattr(fetch.requests, "get", lambda *args, **kwargs: response)
    with pytest.raises(fetch.FetchError, match="ignored.*changed"):
        fetch.RemoteArchive("https://official.example/a.zip", 4, '"stable"').open_range(
            0, 4
        )


def test_archive_reader_refuses_unbounded_large_read():
    reader = fetch.ArchiveReader(
        fetch.RemoteArchive("https://official.example/a.zip", 2 << 20, '"stable"')
    )
    with pytest.raises(fetch.FetchError, match="large ZIP metadata"):
        reader.read()


def test_archive_reader_rejects_short_range(monkeypatch):
    response = response_for(b"data", 0, 4)
    response.raw = io.BytesIO(b"da")
    monkeypatch.setattr(fetch.requests, "get", lambda *args, **kwargs: response)
    reader = fetch.ArchiveReader(
        fetch.RemoteArchive("https://official.example/a.zip", 4, '"stable"')
    )
    with pytest.raises(fetch.FetchError, match="incomplete"):
        reader.read(4)


def test_plan_uses_only_explicit_member(transfer):
    assert transfer.member.filename == f"{fetch.MEMBER_PREFIX}/{RECORDING}.MP4"


def test_plan_rejects_recording_absent_from_zip(archive, tmp_path):
    with pytest.raises(fetch.FetchError, match="exactly one member"):
        fetch.plan_transfers(archive[0], ["electronics/S1790011"], tmp_path)


def test_zip_member_rejects_symlink(archive, transfer):
    transfer.member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(fetch.FetchError, match="symlink"):
        fetch.member_offset(fetch.ArchiveReader(archive[0]), transfer.member)


def test_pending_bytes_accounts_for_partial(transfer):
    transfer.partial.parent.mkdir(parents=True)
    transfer.partial.write_bytes(b"123")
    assert transfer.pending_bytes() == transfer.member.compress_size - 3


def test_pending_bytes_rejects_oversized_partial(transfer):
    transfer.partial.parent.mkdir(parents=True)
    transfer.partial.write_bytes(b"x" * (transfer.member.compress_size + 1))
    with pytest.raises(fetch.FetchError, match="oversized"):
        transfer.pending_bytes()


def test_pending_bytes_rejects_incomplete_final_cache(transfer):
    transfer.compressed_path.parent.mkdir(parents=True)
    transfer.compressed_path.write_bytes(b"123")
    with pytest.raises(fetch.FetchError, match="incomplete finalized"):
        transfer.pending_bytes()


def test_fetch_resumes_only_missing_member_bytes(archive, transfer):
    remote, calls = archive
    transfer.partial.parent.mkdir(parents=True)
    with remote.open_range(transfer.offset, transfer.offset + 3) as response:
        transfer.partial.write_bytes(response.raw.read())
    fetch.fetch_compressed(remote, transfer)
    assert calls[-1] == (
        transfer.offset + 3,
        transfer.offset + transfer.member.compress_size,
    )


def test_download_and_extract_preserves_source_bytes(archive, transfer):
    fetch.fetch_compressed(archive[0], transfer)
    fetch.extract_video(transfer)
    assert transfer.destination.read_bytes() == PAYLOAD


def test_second_download_needs_no_video_transfer(archive, transfer):
    remote, calls = archive
    fetch.fetch_compressed(remote, transfer)
    before = len(calls)
    fetch.fetch_compressed(remote, transfer)
    assert len(calls) == before


def test_extract_rejects_crc_mismatch(archive, transfer):
    fetch.fetch_compressed(archive[0], transfer)
    transfer.member.CRC ^= 1
    with pytest.raises(fetch.FetchError, match="CRC mismatch"):
        fetch.extract_video(transfer)


def test_extract_refuses_to_overwrite_existing_corrupt_video(transfer):
    transfer.destination.parent.mkdir(parents=True)
    transfer.destination.write_bytes(b"unrelated existing file")
    with pytest.raises(fetch.FetchError, match="existing video failed"):
        fetch.extract_video(transfer)


def test_failed_video_probe_does_not_promote_partial(archive, transfer, monkeypatch):
    fetch.fetch_compressed(archive[0], transfer)

    def fail_probe(path):
        raise fetch.FetchError("not a valid video")

    monkeypatch.setattr(fetch, "probe_video", fail_probe)
    with pytest.raises(fetch.FetchError, match="not a valid video"):
        fetch.extract_video(transfer)
    assert not transfer.destination.exists()


@pytest.mark.parametrize("limit", ["0", "-1", "6.01", "nan", "inf"])
def test_main_rejects_invalid_budget_before_network(limit):
    with pytest.raises(fetch.FetchError, match="max-gb"):
        fetch.main(["--recordings", RECORDING, "--max-gb", limit])


@pytest.fixture
def prepared_root(tmp_path, archive, monkeypatch):
    output = tmp_path / fetch.OUT_ROOT
    metadata = output / "annotations/meta/metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"videos": [{"task_id": "electronics", "video_id": "S1790010"}]})
    )
    monkeypatch.setattr(fetch, "probe_archive", lambda: archive[0])
    monkeypatch.setattr(
        fetch, "probe_video", lambda path: {"test_fixture": "not real media"}
    )
    monkeypatch.setattr(
        fetch.subprocess, "check_output", lambda *args, **kwargs: "test_revision\n"
    )
    return tmp_path


def test_dry_run_does_not_create_video_files(prepared_root):
    fetch.main(["--root", str(prepared_root), "--recordings", RECORDING, "--dry-run"])
    assert not (prepared_root / fetch.OUT_ROOT / "videos").exists()


def test_dry_run_reports_remote_checksum_and_resume(prepared_root, capsys):
    fetch.main(["--root", str(prepared_root), "--recordings", RECORDING, "--dry-run"])
    assert (
        json.loads(capsys.readouterr().out)["official_cryptographic_checksum"] is None
    )


def test_main_rejects_plan_over_download_budget(prepared_root):
    with pytest.raises(fetch.FetchError, match="budget"):
        fetch.main(
            [
                "--root",
                str(prepared_root),
                "--recordings",
                RECORDING,
                "--max-gb",
                "0.000000001",
            ]
        )


def test_main_rejects_insufficient_free_disk(prepared_root, monkeypatch):
    monkeypatch.setattr(
        fetch.shutil,
        "disk_usage",
        lambda path: fetch.shutil._ntuple_diskusage(100, 99, 1),
    )
    with pytest.raises(fetch.FetchError, match="capacity"):
        fetch.main(["--root", str(prepared_root), "--recordings", RECORDING])


def test_main_rejects_changed_archive_before_resuming(prepared_root):
    receipt = prepared_root / fetch.OUT_ROOT / "acquisition.json"
    receipt.write_text(json.dumps({"etag": '"old"', "archive_size_bytes": 10}))
    with pytest.raises(fetch.FetchError, match="archive changed"):
        fetch.main(["--root", str(prepared_root), "--recordings", RECORDING])


def test_main_saves_verified_receipt(prepared_root):
    fetch.main(["--root", str(prepared_root), "--recordings", RECORDING])
    receipt = json.loads(
        (prepared_root / fetch.OUT_ROOT / "acquisition.json").read_text()
    )
    assert (
        receipt["recordings"][RECORDING]["local_sha256"]
        == fetch.hashlib.sha256(PAYLOAD).hexdigest()
    )


def test_main_repeat_needs_no_new_video_bytes(prepared_root, capsys):
    fetch.main(["--root", str(prepared_root), "--recordings", RECORDING])
    capsys.readouterr()
    fetch.main(["--root", str(prepared_root), "--recordings", RECORDING, "--dry-run"])
    assert json.loads(capsys.readouterr().out)["pending_bytes"] == 0


def test_probe_video_rejects_file_without_stream(tmp_path, monkeypatch):
    result = subprocess.CompletedProcess(
        [], 0, '{"streams": [], "format": {"duration": "5"}}', ""
    )
    monkeypatch.setattr(fetch.subprocess, "run", lambda *args, **kwargs: result)
    with pytest.raises(fetch.FetchError, match="no valid video stream"):
        fetch.probe_video(tmp_path / "no-video.mp4")
