import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.mine import mine_hdepic_refresh as refresh
from tools.mine.mine_hdepic_refresh import has_refresh_cue


def test_detects_direct_replacement_cues() -> None:
    assert has_refresh_cue("Change my mind and grab a different knife.")
    assert has_refresh_cue("With my right hand, pick up different sponge.")
    assert has_refresh_cue("Returning the napkin to make a different selection.")
    assert has_refresh_cue("I use this cloth instead of the other one.")


def test_ignores_plain_repetition() -> None:
    assert not has_refresh_cue("Pick up another spoonful of sugar.")
    assert not has_refresh_cue("Wash both hands under running water.")


def _mock_narrations(monkeypatch: pytest.MonkeyPatch, records: list[dict]) -> None:
    frame = SimpleNamespace(
        columns=set(records[0]),
        to_dict=lambda orient: records,
    )
    monkeypatch.setattr(
        refresh.importlib,
        "import_module",
        lambda name: SimpleNamespace(read_pickle=lambda path: frame),
    )


def test_find_hits_reads_sorted_cues_without_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_narrations(
        monkeypatch,
        [
            {
                "video_id": "second-video",
                "unique_narration_id": "second-video-1",
                "start_timestamp": 15,
                "end_timestamp": 16,
                "narration": "Change my mind and grab a different knife.",
            },
            {
                "video_id": "first-video",
                "unique_narration_id": "first-video-1",
                "start_timestamp": 10,
                "end_timestamp": 11,
                "narration": "Pick up a different spoon.",
            },
            {
                "video_id": "third-video",
                "unique_narration_id": "third-video-1",
                "start_timestamp": 20,
                "end_timestamp": 21,
                "narration": "Wash both hands under running water.",
            },
        ],
    )
    monkeypatch.setattr(
        refresh,
        "write_json_atomic",
        lambda *args: pytest.fail("find_hits must be read-only"),
    )

    assert refresh.find_hits(Path("narrations.pkl")) == [
        {
            "record_id": "first-video-1",
            "video_id": "first-video",
            "start_seconds": 10.0,
            "end_seconds": 11.0,
            "narration": "Pick up a different spoon.",
        },
        {
            "record_id": "second-video-1",
            "video_id": "second-video",
            "start_seconds": 15.0,
            "end_seconds": 16.0,
            "narration": "Change my mind and grab a different knife.",
        },
    ]


def test_find_hits_fails_for_missing_narration_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_narrations(monkeypatch, [{"video_id": "video"}])

    with pytest.raises(ValueError, match="missing"):
        refresh.find_hits(Path("narrations.pkl"))


@pytest.mark.parametrize(
    "start,end", [(1.0, 1.0), (2.0, 1.0), (-1.0, 1.0), (float("nan"), 1.0)]
)
def test_find_hits_fails_for_invalid_cue_interval(
    monkeypatch: pytest.MonkeyPatch,
    start: float,
    end: float,
) -> None:
    _mock_narrations(
        monkeypatch,
        [
            {
                "video_id": "video",
                "unique_narration_id": "video-1",
                "start_timestamp": start,
                "end_timestamp": end,
                "narration": "Pick up a different spoon.",
            }
        ],
    )

    with pytest.raises(ValueError, match="Invalid narration interval"):
        refresh.find_hits(Path("narrations.pkl"))


def test_mine_keeps_writing_the_cue_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    narrations = tmp_path / "narrations.pkl"
    narrations.write_bytes(b"annotation fixture")
    output = tmp_path / "report.json"
    hit = {
        "record_id": "video-1",
        "video_id": "video",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "narration": "Pick up a different spoon.",
    }
    monkeypatch.setattr(refresh, "find_hits", lambda path: [hit])

    result = refresh.mine(narrations, output, source_name="official/narrations.pkl")

    assert json.loads(output.read_text()) == result
