from __future__ import annotations

from types import SimpleNamespace

import pytest

from egocua.sampling_media import SourceMedia
from tools.audit.stream_windows import TAIL_SECONDS, proposal, stream_block


def _source(times: dict[str, list[float]], stream: dict | None = None) -> SimpleNamespace:
    source = SimpleNamespace(
        packet_timestamps=lambda packet: times[packet["id"]],
        stream=stream,
    )
    source.refresh_wait = lambda task, onset: SourceMedia.refresh_wait(source, task, onset)
    return source


def _task(track: str, packets: tuple[str, ...] = ("V0",)) -> dict:
    task = {
        "id": f"t-{track}",
        "track": track,
        "instructions": {},
        "visual_packets": [{"id": name} for name in packets],
    }
    if track == "refresh":
        task["refresh"] = {"policy": "retain"}
    if track == "clarify":
        task["clarification"] = {"slot": "position"}
    return task


def test_refresh_proposal_starts_early_enough_for_a_close_change() -> None:
    brief = proposal(
        _task("refresh", ("V0", "V1")),
        _source({"V0": [100.0, 102.0], "V1": [103.0, 106.0]}),
        media_end=200.0,
    )

    assert brief["session_seconds"] == [87.0, 106.0]
    assert brief["change_seconds"] == 103.0


def test_refresh_proposal_keeps_the_opening_when_the_change_is_already_late() -> None:
    brief = proposal(
        _task("refresh", ("V0", "V1")),
        _source({"V0": [10.0, 12.0], "V1": [60.0, 64.0]}),
        media_end=200.0,
    )

    assert brief["session_seconds"] == [10.0, 64.0]
    assert brief["skipped_seconds"] == [12.0, 44.0]


def test_graft_proposal_hints_the_widest_gap_and_stops_before_media_end() -> None:
    brief = proposal(
        _task("graft"),
        _source({"V0": [0.0, 2.0, 40.0, 42.0]}),
        media_end=60.0,
    )

    assert brief["proposed_update_onset_seconds"] == 40.0
    assert brief["session_seconds"] == [0.0, 59.5]


def test_clarify_proposal_extends_a_fixed_tail_after_the_opening() -> None:
    brief = proposal(_task("clarify"), _source({"V0": [5.0, 9.0]}), media_end=500.0)

    assert brief["session_seconds"] == [5.0, 9.0 + TAIL_SECONDS]
    assert brief["clarification"] == {"slot": "position"}


def test_proposal_reuses_an_existing_window() -> None:
    brief = proposal(
        _task("graft"),
        _source(
            {"V0": [0.0, 30.0]},
            {"session_seconds": [0.0, 90.0], "update_onset_seconds": 25.0},
        ),
        media_end=500.0,
    )

    assert brief["session_seconds"] == [0.0, 90.0]
    assert brief["proposed_update_onset_seconds"] == 25.0


@pytest.mark.parametrize(
    ("track", "expected"),
    [
        ("graft", {"session_seconds": [1.0, 9.0], "update_onset_seconds": 4.0, "review_evidence": "r.json"}),
        ("clarify", {"session_seconds": [1.0, 9.0], "review_evidence": "r.json"}),
        ("refresh", {"session_seconds": [1.0, 9.0], "review_evidence": "r.json"}),
    ],
)
def test_stream_block_records_only_fields_the_track_owns(track: str, expected: dict) -> None:
    review = {"session_seconds": [1, 9], "update_onset_seconds": 4, "verdict": "accept"}

    assert stream_block(track, review, "r.json") == expected
