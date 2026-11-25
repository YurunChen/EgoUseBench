from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from tools.mine import mine_expansion_candidates as expansion
from tools.mine.mine_hdepic_decisions import ASSOC_INFO


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _ego_annotation(
    annotation_id: int, media_id: str, instruction: str, start: int = 0
) -> dict:
    return {
        "id": annotation_id,
        "vid": media_id,
        "domain": "E-Commerce",
        "instruction": instruction,
        "visual_clues": [f"00:{start:02d}-00:{start + 5:02d}"],
    }


MEDIA = {
    "Ego2Web": ("ego-video", f"{expansion.EGO2WEB_VIDEOS}/ego-video.mp4"),
    "HD-EPIC": ("P01-recording", f"{expansion.HDEPIC_VIDEOS}/P01/P01-recording.mp4"),
    "CaptainCook4D": (
        "captain-video",
        f"{expansion.CAPTAIN_VIDEOS}/captain-video_360p.mp4",
    ),
}


@pytest.fixture
def source_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_json(
        tmp_path / "data/selection.json", {"task_set": {}, "source_manifests": []}
    )
    _write_json(
        tmp_path / expansion.EGO2WEB_ANNOTATIONS,
        [
            _ego_annotation(1, "ego-video", "Add both items to the cart."),
        ],
    )
    _write_json(
        tmp_path / ASSOC_INFO,
        {
            "P01-recording": {
                "first": {"name": "first cup", "tracks": [{"time_segment": [0, 5]}]},
                "second": {
                    "name": "second cup",
                    "tracks": [{"time_segment": [10, 15]}],
                },
            },
        },
    )
    _write_json(
        tmp_path / expansion.CAPTAIN_ANNOTATIONS,
        [
            {
                "recording_id": "captain-video",
                "activity_id": "cook",
                "is_error": True,
                "step_annotations": [
                    {
                        "step_id": 1,
                        "description": "Prepare a bowl.",
                        "start_time": 0,
                        "end_time": 5,
                        "errors": [
                            {
                                "tag": "Preparation Error",
                                "description": "Use the other bowl.",
                            }
                        ],
                    }
                ],
            }
        ],
    )
    narrations = tmp_path / expansion.HDEPIC_NARRATIONS
    narrations.parent.mkdir(parents=True, exist_ok=True)
    narrations.write_bytes(b"official narration fixture")
    monkeypatch.setattr(expansion, "find_hdepic_refresh_hits", lambda path: [])
    for _, relative_path in MEDIA.values():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local video fixture")
    return tmp_path


def test_ego2web_collects_each_nested_augment_group(source_root: Path) -> None:
    annotations = [
        _ego_annotation(1, "bundled", "Add both items to the cart."),
        _ego_annotation(
            2, "related", "Find a compatible filter for the kettle shown in the video."
        ),
        _ego_annotation(3, "pair", "Open a kettle product page."),
        _ego_annotation(4, "pair", "Buy a cup.", start=10),
    ]
    _write_json(source_root / expansion.EGO2WEB_ANNOTATIONS, annotations)
    for media_id in ("bundled", "related", "pair"):
        (source_root / expansion.EGO2WEB_VIDEOS / f"{media_id}.mp4").write_bytes(
            b"video"
        )

    candidates = expansion._ego2web_candidates(source_root, set())

    assert {
        row["media_id"]: (
            row["signals"],
            [
                annotation.get("annotation_ids", [annotation.get("annotation_id")])
                for annotation in row["annotations"]
            ],
        )
        for row in candidates
    } == {
        "bundled": (["graft.augment"], [[1]]),
        "pair": (["graft.augment"], [[3, 4]]),
        "related": (["graft.augment"], [[2]]),
    }


@pytest.mark.parametrize(
    "candidate_kind", ["single", "graft_pair", "refresh_pair", "new_event"]
)
def test_ego2web_does_not_remine_rejected_events_or_pairs_with_a_rejected_endpoint(
    source_root: Path,
    candidate_kind: str,
) -> None:
    rejected_ids = ["295"] if candidate_kind == "single" else [376, 377]
    _write_json(
        source_root / "data/sources/manifests/previous_ego2web_screening_manifest.json",
        {
            "dataset": {"name": "Ego2Web"},
            "tasks": [
                {
                    "task_id": "rejected-task",
                    "status": "rejected",
                    "source": {"media_id": "ego-video", "record_ids": rejected_ids},
                    "rejection_reason": "Released frames do not establish the claimed source identity.",
                }
            ],
        },
    )
    if candidate_kind == "single":
        annotations = [
            _ego_annotation(
                295, "ego-video", "Find the same table saw model shown in the video."
            ),
        ]
    elif candidate_kind == "graft_pair":
        annotations = [
            _ego_annotation(376, "ego-video", "Open a kettle product page."),
            _ego_annotation(10, "ego-video", "Buy a cup.", start=10),
        ]
    elif candidate_kind == "refresh_pair":
        annotations = [
            dict(
                _ego_annotation(
                    record_id,
                    "ego-video",
                    "Find the historic district.",
                    start=index * 10,
                ),
                domain="Local",
            )
            for index, record_id in enumerate((377, 10, 376))
        ]
    else:
        annotations = [_ego_annotation(10, "ego-video", "Add both items to the cart.")]
    _write_json(source_root / expansion.EGO2WEB_ANNOTATIONS, annotations)

    candidates = expansion._ego2web_candidates(source_root, set())

    assert [row["media_id"] for row in candidates] == (
        ["ego-video"] if candidate_kind == "new_event" else []
    )


@pytest.mark.parametrize("dataset", MEDIA)
@pytest.mark.parametrize("media_state", ["missing", "empty", "present"])
def test_only_nonempty_local_media_counts_as_available(
    source_root: Path,
    dataset: str,
    media_state: str,
) -> None:
    path = source_root / MEDIA[dataset][1]
    if media_state == "missing":
        path.unlink()
    elif media_state == "empty":
        path.write_bytes(b"")

    result = expansion.build(source_root)

    available = int(media_state == "present")
    candidate_count = 1 if dataset == "HD-EPIC" else available
    assert result["summary"][dataset] == {
        "candidates": candidate_count,
        "media": candidate_count,
        "media_available": available,
    }


@pytest.mark.parametrize("dataset", MEDIA)
@pytest.mark.parametrize(
    "exclusion", ["selected", "screened_v2", "screened_v1", "unselected"]
)
def test_exclusion_preserves_pool_v2_media_level_rules(
    source_root: Path,
    dataset: str,
    exclusion: str,
) -> None:
    media_id = MEDIA[dataset][0]
    if dataset == "HD-EPIC":
        media_id += "__0-15"
    if exclusion in {"selected", "unselected"}:
        _write_json(
            source_root / "data/selection.json",
            {
                "task_set": {"graft": ["chosen-task"]},
                "source_manifests": ["data/source.json"],
            },
        )
        _write_json(
            source_root / "data/source.json",
            {
                "tasks": [
                    {
                        "task_id": "chosen-task"
                        if exclusion == "selected"
                        else "unselected-task",
                        "source": {
                            "original_video_id": media_id,
                            "media_id": "derived-id",
                        },
                    }
                ]
            },
        )
    else:
        version = 2 if exclusion == "screened_v2" else 1
        _write_json(
            source_root
            / f"data/sources/audits/expansion_screening_decisions_v{version}.json",
            {
                "candidate_pool_schema_version": version,
                "datasets": {
                    dataset: {"media_ids": [media_id]},
                },
            },
        )

    result = expansion.build(source_root)

    assert bool(result["datasets"][dataset]) == (
        exclusion in {"screened_v1", "unselected"}
    )


@pytest.mark.parametrize("pool_exists", [False, True])
def test_check_does_not_write_pool_or_decisions(
    source_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    pool_exists: bool,
) -> None:
    decision = source_root / "data/sources/audits/expansion_screening_decisions_v2.json"
    _write_json(decision, {"candidate_pool_schema_version": 2, "datasets": {}})
    if pool_exists:
        _write_json(source_root / expansion.DEFAULT_OUTPUT, {"existing_pool": True})
    before = {
        path: path.read_bytes() for path in source_root.rglob("*") if path.is_file()
    }

    def unexpected_write(*args: object) -> None:
        pytest.fail("--check must not call the pool writer")

    monkeypatch.setattr(expansion, "write_json_atomic", unexpected_write)
    expansion.main(["--root", str(source_root), "--check"])

    assert {
        path: path.read_bytes() for path in source_root.rglob("*") if path.is_file()
    } == before


def test_check_prints_current_build_summary(
    source_root: Path, capsys: pytest.CaptureFixture
) -> None:
    expansion.main(["--root", str(source_root), "--check"])

    assert (
        json.loads(capsys.readouterr().out) == expansion.build(source_root)["summary"]
    )


def test_normal_build_preserves_reviewed_decisions(source_root: Path) -> None:
    decision = source_root / "data/sources/audits/expansion_screening_decisions_v2.json"
    _write_json(
        decision,
        {
            "candidate_pool_schema_version": 2,
            "datasets": {
                "Ego2Web": {"media_ids": ["ego-video"], "decision": "reviewed"},
            },
        },
    )
    before = decision.read_bytes()

    expansion.main(["--root", str(source_root)])

    assert decision.read_bytes() == before


@pytest.fixture
def narrated_source(source_root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    clips = source_root / expansion.HDEPIC_CLIP_ROOTS[0]
    clips.mkdir(parents=True)
    (clips / "P01-recording__10-30.mp4").write_bytes(b"local narrated clip")
    monkeypatch.setattr(expansion, "_video_duration", lambda path: 20.0)
    monkeypatch.setattr(
        expansion,
        "find_hdepic_refresh_hits",
        lambda path: [
            {
                "record_id": "P01-recording-1",
                "video_id": "P01-recording",
                "start_seconds": 15.25,
                "end_seconds": 17.5,
                "narration": "Change my mind and grab a different knife.",
            }
        ],
    )
    return source_root


def test_narration_candidate_preserves_provenance_and_both_time_ranges(
    narrated_source: Path,
) -> None:
    (candidate,) = expansion._hdepic_narrated_candidates(narrated_source, set())

    assert candidate == {
        "dataset": "HD-EPIC",
        "media_id": "P01-recording",
        "media_path": f"{expansion.HDEPIC_CLIP_ROOTS[0]}/P01-recording__10-30.mp4",
        "media_available": True,
        "media_sha256": hashlib.sha256(b"local narrated clip").hexdigest(),
        "media_duration_seconds": 20.0,
        "source_start_seconds": 10.0,
        "source_clip_window_seconds": [10.0, 30.0],
        "window_seconds": [15.25, 17.5],
        "media_window_seconds": [5.25, 7.5],
        "signals": [expansion.NARRATED_CHOICE_SIGNAL],
        "signal_schema_version": 1,
        "annotation_id": "P01-recording-1",
        "narration": "Change my mind and grab a different knife.",
        "annotation_source": {
            "path": expansion.HDEPIC_NARRATIONS,
            "sha256": hashlib.sha256(b"official narration fixture").hexdigest(),
        },
        "prior_rejections": [],
        "stage": "narration_mined_local_clip",
    }


@pytest.mark.parametrize("selected", [{"P01-recording"}, {"P01-recording__40-60"}])
def test_narration_excludes_selected_source_across_windows(
    narrated_source: Path,
    selected: set[str],
) -> None:
    assert expansion._hdepic_narrated_candidates(narrated_source, selected) == []


@pytest.mark.parametrize("start,end", [(9.0, 12.0), (29.0, 31.0), (31.0, 33.0)])
def test_narration_requires_entire_event_in_local_clip(
    narrated_source: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: float,
    end: float,
) -> None:
    (hit,) = expansion.find_hdepic_refresh_hits(
        narrated_source / expansion.HDEPIC_NARRATIONS
    )
    monkeypatch.setattr(
        expansion,
        "find_hdepic_refresh_hits",
        lambda path: [
            dict(hit, start_seconds=start, end_seconds=end),
        ],
    )

    assert expansion._hdepic_narrated_candidates(narrated_source, set()) == []


def test_narration_rejects_clip_shorter_than_event_end(
    narrated_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(expansion, "_video_duration", lambda path: 6.0)

    assert expansion._hdepic_narrated_candidates(narrated_source, set()) == []


def test_narration_rejects_empty_clip(narrated_source: Path) -> None:
    (
        narrated_source / expansion.HDEPIC_CLIP_ROOTS[0] / "P01-recording__10-30.mp4"
    ).write_bytes(b"")

    assert expansion._hdepic_narrated_candidates(narrated_source, set()) == []


def test_narration_new_signal_survives_old_v2_media_rejection(
    narrated_source: Path,
) -> None:
    relative_path = "data/sources/audits/expansion_screening_decisions_v2.json"
    _write_json(
        narrated_source / relative_path,
        {
            "candidate_pool_schema_version": 2,
            "datasets": {
                "HD-EPIC": {
                    "media_ids": ["P01-recording"],
                    "rejected_media_ids": ["P01-recording"],
                    "rejection_reason": "The old sequence signal was not visually separable.",
                }
            },
        },
    )

    (candidate,) = expansion.build(narrated_source)["datasets"]["HD-EPIC"]

    assert (candidate["signals"], candidate["prior_rejections"][0]["path"]) == (
        [expansion.NARRATED_CHOICE_SIGNAL],
        relative_path,
    )


@pytest.mark.parametrize("same_event", ["record_id", "clip_window", "different_event"])
def test_narration_respects_rejected_events_and_keeps_other_event_history(
    narrated_source: Path,
    same_event: str,
) -> None:
    relative_path = "data/sources/manifests/previous_hdepic_screening_manifest.json"
    _write_json(
        narrated_source / relative_path,
        {
            "dataset": {"name": "HD-EPIC"},
            "tasks": [
                {
                    "task_id": "rejected-task",
                    "status": "rejected",
                    "source": {
                        "original_video_id": "P01-recording",
                        "media_id": "P01-recording__10-30"
                        if same_event == "clip_window"
                        else "P01-recording__40-60",
                        "record_ids": [
                            "P01-recording-1"
                            if same_event == "record_id"
                            else "P01-recording-old"
                        ],
                    },
                    "rejection_reason": "The decisive attribute was unreadable.",
                }
            ],
        },
    )

    candidates = expansion._hdepic_narrated_candidates(narrated_source, set())

    assert [row["prior_rejections"][0]["path"] for row in candidates] == (
        [relative_path] if same_event == "different_event" else []
    )


def test_narration_counts_source_once_for_multiple_events_and_duplicate_clips(
    narrated_source: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (hit,) = expansion.find_hdepic_refresh_hits(
        narrated_source / expansion.HDEPIC_NARRATIONS
    )
    monkeypatch.setattr(
        expansion,
        "find_hdepic_refresh_hits",
        lambda path: [
            hit,
            dict(
                hit, record_id="P01-recording-2", start_seconds=20.0, end_seconds=22.0
            ),
        ],
    )
    duplicate = (
        narrated_source / expansion.HDEPIC_CLIP_ROOTS[1] / "P01-recording__10-30.mp4"
    )
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"duplicate clip")

    summary = expansion.build(narrated_source)["summary"]["HD-EPIC"]

    assert summary == {"candidates": 3, "media": 1, "media_available": 1}


def test_check_is_read_only_with_narration_candidates(narrated_source: Path) -> None:
    before = {
        path: path.read_bytes() for path in narrated_source.rglob("*") if path.is_file()
    }

    expansion.main(["--root", str(narrated_source), "--check"])

    assert {
        path: path.read_bytes() for path in narrated_source.rglob("*") if path.is_file()
    } == before
