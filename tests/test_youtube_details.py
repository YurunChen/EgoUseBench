import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.catalog.freeze_youtube_details import (
    DETAIL_DIR,
    _chapters,
    _normalize_snapshot,
    _projection,
    _public_details,
    _summary,
    _youtube_id,
    freeze,
    import_captions,
)
from tools.catalog.sanitize_catalog_metadata import (
    _captured_chapters,
    _upstream_chapters,
)
from egocua.workspace import write_json_atomic


def test_native_comments_are_projected_without_rewriting_their_text(tmp_path) -> None:
    raw = {
        "id": "public-video",
        "title": "Original tutorial",
        "comment_count": 12,
        "comments": [
            {
                "id": "native-comment",
                "author": "@OriginalAuthor",
                "text": "Keep runoff off nearby surfaces.\nUse a catch pan.",
                "like_count": 0,
                "_time_text": "2 years ago",
                "is_pinned": False,
                "author_is_uploader": True,
            }
        ],
    }
    projected = _projection(raw)
    projected["public_comments"][0]["text"] = "Outdated display copy"
    write_json_atomic(tmp_path / DETAIL_DIR / "public-video.json", projected)
    write_json_atomic(
        tmp_path / "sites/egomedia/videos.json",
        {
            "videos": [
                {
                    "video_id": "one",
                    "source": {
                        "page_url": "https://www.youtube.com/watch?v=public-video"
                    },
                },
                {"video_id": "other", "comments": [{"text": "Unchanged"}]},
            ]
        },
    )

    freeze(tmp_path, {"one"}, fetch_missing=False)

    videos = json.loads((tmp_path / "sites/egomedia/videos.json").read_text())["videos"]
    assert videos[0]["comments"] == [
        {
            "id": "native-comment",
            "author": "@OriginalAuthor",
            "text": raw["comments"][0]["text"],
            "like_count": 0,
            "published": "2 years ago",
            "pinned": False,
            "author_is_uploader": True,
        }
    ]
    assert videos[0]["comment_count"] == 12
    assert videos[1] == {"video_id": "other", "comments": [{"text": "Unchanged"}]}


def test_empty_explicit_scope_never_expands_to_all_tasks(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit entity_ids is empty"):
        freeze(tmp_path, set(), fetch_missing=False)
    assert not list(tmp_path.iterdir())


def test_import_captions_preserves_capture_and_only_updates_owning_entity(
    tmp_path,
) -> None:
    videos = [
        {
            "video_id": "one",
            "source": {"page_url": "https://www.youtube.com/watch?v=public-video"},
        },
        {"video_id": "other", "title": "Unchanged"},
    ]
    snapshot = {"id": "public-video", "title": "Original tutorial"}
    write_json_atomic(tmp_path / "sites/egomedia/videos.json", {"videos": videos})
    write_json_atomic(tmp_path / DETAIL_DIR / "public-video.json", snapshot)
    caption_path = tmp_path / "egocua-public-public-video.en.vtt"
    raw = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nClamp before the glue sets.\n"
    caption_path.write_text(raw)

    result = import_captions(tmp_path, "one", caption_path)

    captured = json.loads((tmp_path / DETAIL_DIR / "public-video.json").read_text())
    actual = json.loads((tmp_path / "sites/egomedia/videos.json").read_text())["videos"]
    assert result["updated"] == 1
    assert captured["public_captions"]["raw_vtt"] == raw
    assert actual[0]["chapters"][-1]["text"] == "Clamp before the glue sets."
    assert actual[1] == videos[1]


@pytest.mark.parametrize(
    "filename,raw",
    [
        ("egocua-public-other-video.en.vtt", "WEBVTT\n\n"),
        (
            "egocua-public-public-video.en.vtt",
            "An authored summary, not native captions",
        ),
    ],
)
def test_import_captions_rejects_wrong_owner_or_invalid_capture_before_writing(
    tmp_path, filename, raw
) -> None:
    snapshot = {"id": "public-video", "title": "Original tutorial"}
    write_json_atomic(tmp_path / DETAIL_DIR / "public-video.json", snapshot)
    catalog = {
        "videos": [
            {
                "video_id": "one",
                "source": {"page_url": "https://www.youtube.com/watch?v=public-video"},
            }
        ]
    }
    write_json_atomic(tmp_path / "sites/egomedia/videos.json", catalog)
    caption_path = tmp_path / filename
    caption_path.write_text(raw)

    with pytest.raises(ValueError):
        import_captions(tmp_path, "one", caption_path)

    assert (
        json.loads((tmp_path / DETAIL_DIR / "public-video.json").read_text())
        == snapshot
    )
    assert json.loads((tmp_path / "sites/egomedia/videos.json").read_text()) == catalog


def test_freeze_does_not_import_a_noncanonical_scratch_cache(tmp_path) -> None:
    video = {
        "video_id": "one",
        "source": {"page_url": "https://www.youtube.com/watch?v=public-video"},
    }
    write_json_atomic(tmp_path / "sites/egomedia/videos.json", {"videos": [video]})
    write_json_atomic(
        tmp_path / ".scratch/upstream-catalog/youtube-details.json",
        {"public-video": {"id": "public-video", "title": "Unregistered cache"}},
    )

    result = freeze(tmp_path, {"one"}, fetch_missing=False)

    assert result["updated"] == 0
    assert result["missing"] == 1
    assert not (tmp_path / DETAIL_DIR / "public-video.json").exists()
    assert json.loads((tmp_path / "sites/egomedia/videos.json").read_text())[
        "videos"
    ] == [video]


@pytest.mark.parametrize("previews", [[{"image_path": "public-preview.jpg"}], []])
def test_freeze_projects_previews_from_the_canonical_snapshot(
    tmp_path, previews
) -> None:
    video = {
        "video_id": "one",
        "source": {"page_url": "https://www.youtube.com/watch?v=public-video"},
        "public_previews": [{"image_path": "stale-preview.jpg"}],
    }
    snapshot = {"id": "public-video", "title": "Public video"}
    if previews:
        snapshot["public_previews"] = previews
    write_json_atomic(tmp_path / "sites/egomedia/videos.json", {"videos": [video]})
    write_json_atomic(tmp_path / DETAIL_DIR / "public-video.json", snapshot)

    freeze(tmp_path, {"one"}, fetch_missing=False)

    actual = json.loads((tmp_path / "sites/egomedia/videos.json").read_text())[
        "videos"
    ][0]
    assert actual.get("public_previews", []) == previews


def test_summary_preserves_upstream_words_and_bounds_card_length() -> None:
    text = "  How   to use a glue gun.\n\n" + "step " * 400

    result = _summary(text, "Fallback")

    assert result.startswith("How to use a glue gun.")
    assert result.endswith("…")
    assert len(result) <= 1201


def test_freeze_joins_chapters_by_video_identity_not_catalog_order(tmp_path) -> None:
    videos = [
        {
            "video_id": entity_id,
            "title": entity_id,
            "chapters": [],
            "source": {"page_url": f"https://www.youtube.com/watch?v={youtube_id}"},
        }
        for entity_id, youtube_id in [
            ("second", "video-b"),
            ("missing", "video-without-detail"),
            ("first", "video-a"),
        ]
    ]
    write_json_atomic(tmp_path / "sites/egomedia/videos.json", {"videos": videos})
    for youtube_id, title in [("video-a", "Clay pot"), ("video-b", "Minecraft")]:
        write_json_atomic(
            tmp_path / DETAIL_DIR / f"{youtube_id}.json",
            {
                "id": youtube_id,
                "title": title,
                "chapters": [{"title": title, "start_time": 0, "end_time": 30}],
            },
        )

    result = freeze(tmp_path, {"first", "second", "missing"}, fetch_missing=False)

    actual = {
        row["video_id"]: row
        for row in json.loads((tmp_path / "sites/egomedia/videos.json").read_text())[
            "videos"
        ]
    }
    assert result["updated"] == 2
    assert result["missing"] == 1
    assert actual["first"]["chapters"] == [{"heading": "Clay pot", "text": "0:00–0:30"}]
    assert actual["second"]["chapters"] == [
        {"heading": "Minecraft", "text": "0:00–0:30"}
    ]
    assert actual["missing"] == videos[1]


def test_chapters_are_only_derived_from_upstream_timestamps() -> None:
    result = _chapters(
        {
            "chapters": [
                {"title": "Fastening Boards", "start_time": 75, "end_time": 120},
                {"title": "Missing timestamp"},
            ]
        }
    )

    assert result == [{"heading": "Fastening Boards", "text": "1:15–2:00"}]


def test_projection_persists_public_detail_fields() -> None:
    result = _projection(
        {
            "id": "video-id",
            "title": "Build a Shelf",
            "description": "Measure first, then fasten the shelf.",
            "uploader": "Workshop",
            "upload_date": "20260829",
            "duration": 90,
            "view_count": 42,
            "chapters": [{"title": "Measure", "start_time": 5}],
        }
    )

    assert result["provider"] == "youtube-video-detail-v1"
    assert result["published_at"] == "2026-08-29"
    assert result["public_summary"] == "Measure first, then fasten the shelf."
    assert result["public_chapters"] == [
        {
            "heading": "Measure",
            "text": "Starts at 0:05",
        }
    ]


def test_public_details_drop_task_authored_procedure_claims() -> None:
    entity = {
        "details": {
            "covered_steps": "unsupported task-authored method claim",
            "language": "English",
        }
    }

    assert _public_details(entity) == {"language": "English", "format": "video"}


def _caption_snapshot(
    raw: str, *, browser: bool = False, kind: str = "automatic"
) -> dict:
    raw_key = "raw_transcript" if browser else "raw_vtt"
    return {
        "id": "public-video",
        "title": "A public tutorial",
        "description": "Public description.",
        "chapters": [{"title": "Introduction", "start_time": 0, "end_time": 10}],
        "public_chapters": [
            {"heading": "Unverified summary", "text": "private task conclusion"}
        ],
        "public_captions": {
            "kind": kind,
            "language": "en",
            "source_url": "https://www.youtube.com/watch?v=public-video",
            "retrieved_at": "2026-09-05T00:00:00Z",
            raw_key: raw,
            f"{raw_key}_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "segments": [
                {"start": "00:00:00.000", "text": "Do not trust this authored summary"}
            ],
        },
    }


def test_normalization_projects_complete_original_vtt_not_authored_segments() -> None:
    snapshot = _caption_snapshot("""WEBVTT

00:00:00.000 --> 00:00:02.000
 
Hello<c> from</c> the kitchen.

00:00:02.000 --> 00:00:04.000
Hello from the kitchen. Use a bowl.

00:00:32.000 --> 00:00:34.000
Mix well.
""")

    result = _normalize_snapshot(snapshot)

    assert result["public_chapters"] == [
        {"heading": "Introduction", "text": "0:00–0:10"},
        {
            "heading": "Auto-generated public transcript — 0:00",
            "text": "Hello from the kitchen. Use a bowl.",
        },
        {"heading": "Auto-generated public transcript — 0:32", "text": "Mix well."},
    ]


def test_caption_normalization_preserves_original_capture_and_is_idempotent() -> None:
    snapshot = _caption_snapshot(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nUse a bowl.\n"
    )
    snapshot["public_captions"]["response_url"] = (
        "https://www.youtube.com/api/timedtext?v=public-video&expire=1"
    )
    original = copy.deepcopy(snapshot)

    normalized = _normalize_snapshot(snapshot)

    assert snapshot == original
    assert normalized["public_captions"] == original["public_captions"]
    assert _normalize_snapshot(normalized) == normalized


def test_unknown_caption_kind_does_not_claim_automatic_transcription() -> None:
    snapshot = _caption_snapshot(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nStir.\n\n"
        "00:00:02.000 --> 00:00:04.000\nStir.\n",
        kind="unknown",
    )

    assert _normalize_snapshot(snapshot)["public_chapters"][-1] == {
        "heading": "Public transcript — 0:00",
        "text": "Stir. Stir.",
    }


@pytest.mark.parametrize("raw", ["", "WEBVTT\n\n"])
def test_caption_normalization_rejects_empty_original_text(raw: str) -> None:
    with pytest.raises(ValueError):
        _normalize_snapshot(_caption_snapshot(raw))


def test_browser_captions_without_end_times_are_rebuilt_from_the_original_export() -> (
    None
):
    snapshot = _caption_snapshot(
        "YouTube transcript\nVideo ID: public-video\n\n[0:00] Hello from the kitchen.\n[0:31] Use a bowl.\n",
        browser=True,
        kind="auto-generated",
    )

    result = _normalize_snapshot(snapshot)

    assert result["public_chapters"][1:] == [
        {
            "heading": "Auto-generated public transcript — 0:00",
            "text": "Hello from the kitchen.",
        },
        {"heading": "Auto-generated public transcript — 0:31", "text": "Use a bowl."},
    ]


def test_manual_captions_preserve_intentionally_repeated_words() -> None:
    snapshot = _caption_snapshot(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nStir.\n\n00:00:01.000 --> 00:00:02.000\nStir.\n",
        kind="manual",
    )

    assert _normalize_snapshot(snapshot)["public_chapters"][-1] == {
        "heading": "Public transcript — 0:00",
        "text": "Stir. Stir.",
    }


@pytest.mark.parametrize(
    "change",
    [
        {"raw_vtt_sha256": "wrong-hash"},
        {"source_url": "https://www.youtube.com/watch?v=another-video"},
        {"source_url": "https://not-youtube.example/watch?v=public-video"},
        {"source_url": "https://www.youtube.com/api/timedtext?v=public-video"},
        {"kind": "task-summary"},
        {"retrieved_at": ""},
    ],
)
def test_caption_normalization_rejects_unverifiable_capture_metadata(
    change: dict,
) -> None:
    snapshot = _caption_snapshot(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nUse a bowl.\n"
    )
    snapshot["public_captions"].update(change)

    with pytest.raises(ValueError):
        _normalize_snapshot(snapshot)


def test_caption_normalization_requires_original_text_not_just_segments() -> None:
    snapshot = _caption_snapshot(
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nUse a bowl.\n"
    )
    snapshot["public_captions"].pop("raw_vtt")

    with pytest.raises(ValueError, match="original caption response"):
        _normalize_snapshot(snapshot)


def test_normalization_without_captions_keeps_the_previous_projection() -> None:
    original = _projection(
        {
            "id": "old-video",
            "title": "An existing tutorial",
            "description": "Original words.",
            "chapters": [{"title": "Original chapter", "start_time": 5}],
        }
    )

    assert _normalize_snapshot(original) == original


@pytest.mark.requires_local_data(
    "data/sources/upstream_detail_snapshots/youtube/g8KOpFCoAMw.json",
)
def test_released_media_catalog_contains_only_public_procedure_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    videos = json.loads((root / "sites/egomedia/videos.json").read_text())["videos"]

    for video in videos:
        assert "covered_steps" not in video.get("details", {})
        assert "covered_steps" not in video.get("facts", {})
        chapters = video.get("chapters") or []
        if (
            chapters
            and not _upstream_chapters(chapters)
            and not _captured_chapters(video, root)
        ):
            snapshot = json.loads(
                (root / DETAIL_DIR / f"{_youtube_id(video)}.json").read_text()
            )
            assert snapshot.get("public_captions")
            assert chapters == _normalize_snapshot(snapshot)["public_chapters"]


def test_linked_manufacturer_instructions_keep_their_separate_source(tmp_path):
    linked = {
        "publisher": "Manufacturer",
        "source_url": "https://manufacturer.example/application",
        "text": "Cover adjacent parts before application.",
    }
    snapshot = _projection(
        {
            "id": "public-video",
            "title": "Tutorial",
            "description": "Original description.",
        }
    )
    snapshot["linked_manufacturer_instructions"] = linked
    write_json_atomic(tmp_path / DETAIL_DIR / "public-video.json", snapshot)
    write_json_atomic(
        tmp_path / "sites/egomedia/videos.json",
        {
            "videos": [
                {
                    "video_id": "one",
                    "source": {
                        "page_url": "https://www.youtube.com/watch?v=public-video"
                    },
                }
            ]
        },
    )

    freeze(tmp_path, {"one"}, fetch_missing=False)

    entity = json.loads((tmp_path / "sites/egomedia/videos.json").read_text())[
        "videos"
    ][0]
    assert entity["details"]["linked_manufacturer_instructions"] == linked
    assert entity["description"] == "Original description."
    assert entity["chapters"] == []
    snapshot.pop("linked_manufacturer_instructions")
    write_json_atomic(tmp_path / DETAIL_DIR / "public-video.json", snapshot)
    freeze(tmp_path, {"one"}, fetch_missing=False)
    entity = json.loads((tmp_path / "sites/egomedia/videos.json").read_text())[
        "videos"
    ][0]
    assert "linked_manufacturer_instructions" not in entity["details"]


@pytest.mark.parametrize("change", ["none", "wrong_owner", "wrong_title", "wrong_time"])
def test_native_chapter_capture_must_match_owner_and_actual_content(tmp_path, change):
    capture = {
        "url": "https://www.youtube.com/watch?v=source",
        "chapters": [{"title": "Start", "start_time": 0, "end_time": 10}],
    }
    entity = {
        "source": {
            "page_url": capture["url"],
            "detail_snapshot_paths": ["capture.json"],
        },
        "chapters": [{"title": "Start", "start_seconds": 0}],
    }
    (tmp_path / "capture.json").write_text(json.dumps(capture))
    if change == "wrong_owner":
        entity["source"]["page_url"] += "-other"
    elif change == "wrong_title":
        entity["chapters"][0]["title"] = "Authored claim"
    elif change == "wrong_time":
        entity["chapters"][0]["start_seconds"] = 99
    assert _captured_chapters(entity, tmp_path) is (change == "none")
