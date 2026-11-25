from __future__ import annotations

import io
import json

import pytest

from tools.mine import fetch_captaincook4d as fetch


class Response(io.BytesIO):
    def __init__(self, body: bytes = b"", *, status: int = 206, content_range: str):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range}


def test_remote_size_requires_exact_one_byte_response(monkeypatch):
    monkeypatch.setattr(
        fetch, "_open", lambda *args, **kwargs: Response(content_range="bytes 0-0/12")
    )
    assert fetch.remote_size("https://example.test/video") == 12


@pytest.mark.parametrize(
    "content_range", ["", "bytes 0-1/12", "bytes 0-0/*", "bytes 0-0/0"]
)
def test_remote_size_rejects_invalid_range(monkeypatch, content_range):
    monkeypatch.setattr(
        fetch, "_open", lambda *args, **kwargs: Response(content_range=content_range)
    )
    with pytest.raises(fetch.FetchError):
        fetch.remote_size("https://example.test/video")


def test_plan_rejects_missing_resolution_instead_of_silently_omitting(tmp_path):
    with pytest.raises(fetch.FetchError, match="no 4k URL for: 1_1"):
        fetch.plan_downloads(
            {"1_1": {"gopro_360p": "https://example.test/video"}},
            ["1_1"],
            "4k",
            tmp_path,
        )


def test_duplicate_recording_is_rejected():
    with pytest.raises(fetch.FetchError, match="unique"):
        fetch.named_recordings({"1_1": {}}, ["1_1", "1_1"])


def test_unknown_recording_is_rejected():
    with pytest.raises(fetch.FetchError, match="unknown recording"):
        fetch.named_recordings({"1_1": {}}, ["2_2"])


def test_dry_run_reads_current_source_root(tmp_path, monkeypatch, capsys):
    metadata = (
        tmp_path
        / "reference/sources/CaptainCook4D/downloader/metadata/download_links.json"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"1_1": {"gopro_360p": "https://example.test/video"}})
    )
    monkeypatch.setattr(fetch, "remote_size", lambda url: 12)
    monkeypatch.setattr(
        fetch, "_transfer", lambda *args: pytest.fail("dry-run transferred bytes")
    )
    fetch.main(["--root", str(tmp_path), "--recordings", "1_1", "--dry-run"])
    assert (
        f"12 bytes -> {tmp_path}/reference/sources/CaptainCook4D/videos/360p/1_1_360p.mp4"
        in capsys.readouterr().out
    )


def test_resume_keeps_existing_prefix(tmp_path, monkeypatch):
    destination = tmp_path / "video.mp4"
    destination.with_suffix(".mp4.part").write_bytes(b"abc")
    monkeypatch.setattr(
        fetch,
        "_open",
        lambda *args, **kwargs: Response(b"def", content_range="bytes 3-5/6"),
    )
    fetch.download("https://example.test/video", destination, expected=6)
    assert destination.read_bytes() == b"abcdef"


def test_bad_resume_range_does_not_modify_partial(tmp_path, monkeypatch):
    destination = tmp_path / "video.mp4"
    partial = destination.with_suffix(".mp4.part")
    partial.write_bytes(b"abc")
    monkeypatch.setattr(fetch, "_ATTEMPTS", 1)
    monkeypatch.setattr(fetch.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        fetch,
        "_open",
        lambda *args, **kwargs: Response(b"xxx", content_range="bytes 0-2/6"),
    )
    with pytest.raises(fetch.FetchError):
        fetch.download("https://example.test/video", destination, expected=6)
    assert partial.read_bytes() == b"abc"


def test_server_ignoring_resume_restarts_partial(tmp_path, monkeypatch):
    destination = tmp_path / "video.mp4"
    destination.with_suffix(".mp4.part").write_bytes(b"abc")
    monkeypatch.setattr(
        fetch,
        "_open",
        lambda *args, **kwargs: Response(b"abcdef", status=200, content_range=""),
    )
    fetch.download("https://example.test/video", destination, expected=6)
    assert destination.read_bytes() == b"abcdef"


def test_oversized_partial_is_not_promoted(tmp_path):
    destination = tmp_path / "video.mp4"
    destination.with_suffix(".mp4.part").write_bytes(b"abcdefg")
    with pytest.raises(fetch.FetchError, match="got 7 bytes, expected 6"):
        fetch.download("https://example.test/video", destination, expected=6)
    assert not destination.exists()


def test_complete_download_makes_no_network_request(tmp_path, monkeypatch):
    destination = tmp_path / "video.mp4"
    destination.write_bytes(b"abcdef")
    monkeypatch.setattr(
        fetch, "_open", lambda *args, **kwargs: pytest.fail("already complete")
    )
    assert fetch.download("https://example.test/video", destination, expected=6) == 0
