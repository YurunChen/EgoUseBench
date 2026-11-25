from __future__ import annotations

import subprocess
from dataclasses import replace
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from egocua.checker import check_run
from egocua.execution import RunConfig, run_experiment
from egocua.metrics import _condition
from egocua.run_ledger import asset_hashes, load_run_record
from egocua.sampling_media import prepare_sampling_tasks, materialize_sampling_frames
from egocua.timeline import STANDARD_CONDITIONS, Condition, Timeline
from egocua.visual_sampling import sample_packet, sampling_times
from tests.helpers import (
    item_id,
    materialize_site,
    products,
    task,
)


def sampled_task(track):
    definition = task(track)
    for index, packet in enumerate(definition["visual_packets"]):
        start = 10.25 + index * 10
        packet["frequency_samples"] = {
            "window_seconds": [start, start + 6.5],
            "timestamps_seconds": [start + offset for offset in range(7)],
            "frame_paths": [
                f"samples/{packet['id']}/{offset}.jpg" for offset in range(7)
            ],
        }
    return definition


def condition(interval):
    return replace(
        STANDARD_CONDITIONS["online"],
        name=f"interval_{interval:g}s",
        frame_interval_seconds=interval,
    )


@pytest.mark.parametrize(
    "interval, expected",
    [
        (1, [10.25, 11.25, 12.25, 13.25, 14.25, 15.25, 16.25]),
        (2, [10.25, 12.25, 14.25, 16.25]),
        (5, [10.25, 15.25]),
        (20, [10.25]),
        (0.1, [10.25, 10.35, 10.45]),
    ],
)
def test_grid_is_anchored_to_source_start_without_forced_last_frame(interval, expected):
    end = 10.45 if interval == 0.1 else 16.75
    assert sampling_times([10.25, end], interval) == expected


@pytest.mark.parametrize("interval", [0, -1, True, "2", float("nan"), float("inf")])
def test_invalid_frequency_is_rejected(interval):
    with pytest.raises(ValueError, match="finite positive"):
        replace(STANDARD_CONDITIONS["online"], frame_interval_seconds=interval)


@pytest.mark.parametrize("window", [[2, 1], [-1, 1], [float("nan"), 1], [0]])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="window"):
        sampling_times(window, 1)


def test_sparse_keyframes_cannot_masquerade_as_fixed_frequency():
    with pytest.raises(ValueError, match="prepared source-video"):
        sample_packet(task("graft")["visual_packets"][0], 1)


def test_missing_sampling_frame_does_not_fall_back_to_nearest_keyframe():
    packet = sampled_task("graft")["visual_packets"][0]
    packet["frequency_samples"]["timestamps_seconds"].pop()
    packet["frequency_samples"]["frame_paths"].pop()
    with pytest.raises(ValueError, match="missing requested"):
        sample_packet(packet, 2)


@pytest.mark.parametrize("scope", ["first_frame", "none", "initial_only"])
def test_frequency_does_not_silently_combine_with_scope_ablation(scope):
    with pytest.raises(ValueError, match="scope=full"):
        replace(condition(1), scope=scope)


def test_condition_serialization_preserves_frequency():
    assert Condition.from_dict(condition(2).as_dict()) == condition(2)


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_frequency_preserves_instruction_answer_and_delivery_schedule(track):
    definition = sampled_task(track)
    timelines = [
        Timeline.compile(
            definition,
            condition(interval),
            default_release_step=3,
            reference_steps=None,
        )
        for interval in (1, 2, 5)
    ]
    assert (
        len(
            {
                (
                    timeline.instruction,
                    timeline.offer,
                    tuple((r.ref, r.step, r.kind) for r in timeline.releases),
                )
                for timeline in timelines
            }
        )
        == 1
    )


def test_refresh_resamples_both_packets_independently():
    timeline = Timeline.compile(
        sampled_task("refresh"),
        condition(5),
        default_release_step=3,
        reference_steps=None,
    )
    assert [
        timeline.initial_observations[0]["timestamps_seconds"],
        timeline.releases[0].payload["timestamps_seconds"],
    ] == [[10.25, 15.25], [20.25, 25.25]]


def test_sampling_does_not_inherit_full_packet_sufficiency():
    timeline = Timeline.compile(
        sampled_task("graft"),
        condition(1),
        default_release_step=3,
        reference_steps=None,
    )
    assert timeline.sufficiency_step() is None


def test_frequency_results_do_not_pollute_original_full_frame_contrasts():
    rows = [
        {"condition": item.as_dict()}
        for item in (STANDARD_CONDITIONS["online"], condition(1))
    ]
    assert _condition(rows, scope="full", frame_interval_seconds=None) == rows[:1]


def test_asset_hashes_include_frequency_frames(tmp_path):
    definition = sampled_task("graft")
    path = tmp_path / "samples/V0/0.jpg"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"fixture-image")
    assert "samples/V0/0.jpg" in asset_hashes(definition, tmp_path)


def test_frequency_runs_preserve_terminal_scoring_and_record_actual_frames(tmp_path):
    definition = sampled_task("graft")
    materialize_site(products("graft"), tmp_path)

    def execute(context):
        context.advance()
        context._state.select(item_id("graft", 0))
        context.advance()
        return []

    results = []
    for interval in (1, 2, 5):
        record = run_experiment(
            definition,
            condition(interval),
            RunConfig(
                agent="fixture",
                model="scripted-fixture",
                default_release_step=2,
                reference_steps=None,
                asset_root=tmp_path,
                output_root=tmp_path / "artifacts",
                experiment_id="sampling",
            ),
            execute,
        )
        restored = load_run_record(record["meta"]["artifact_directory"])
        assert restored["initial_context"]["observations"][0] == sample_packet(
            definition["visual_packets"][0], interval
        )
        results.append(check_run(definition, restored)["e2e_success"])
    assert results == [True, True, True]


@pytest.fixture
def source_sampling(tmp_path, monkeypatch):
    video = tmp_path / "source/clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"test video source identity")
    paths = ["a.jpg", "b.jpg"]
    for path in paths:
        Image.new("RGB", (16, 12)).save(tmp_path / path)
    definition = {
        "id": "test",
        "source": {"task_id": "test"},
        "visual_packets": [{"id": "V0", "frame_paths": paths}],
    }
    record = {
        "task_id": "test",
        "source": {"media_id": "clip", "media_start_seconds": 100},
        "packets": [
            {"id": "V0", "frame_paths": paths, "timestamps_seconds": [100.5, 102.7]}
        ],
    }
    dossier = {"dataset": {"source_root": "source"}, "tasks": [record]}
    monkeypatch.setattr(
        "egocua.sampling_media.load_dossier", lambda *args, **kwargs: (None, dossier)
    )
    prepared = prepare_sampling_tasks([definition], tmp_path, (1, 2, 5))
    return definition, prepared


def refresh_session_source(tmp_path, monkeypatch, stream=None):
    video = tmp_path / "source/clip.mp4"
    video.parent.mkdir()
    video.write_bytes(b"test video source identity")
    for path in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        Image.new("RGB", (16, 12)).save(tmp_path / path)
    packets = [
        {"id": "V0", "frame_paths": ["a.jpg", "b.jpg"]},
        {"id": "V1", "frame_paths": ["c.jpg", "d.jpg"]},
    ]
    record = {
        "task_id": "test",
        "source": {"media_id": "clip", "media_start_seconds": 100},
        "packets": [
            {**packets[0], "timestamps_seconds": [100.0, 101.0]},
            {**packets[1], "timestamps_seconds": [104.0, 105.0]},
        ],
    }
    if stream is not None:
        record["stream"] = stream
    dossier = {"dataset": {"source_root": "source"}, "tasks": [record]}
    monkeypatch.setattr(
        "egocua.sampling_media.load_dossier", lambda *args, **kwargs: (None, dossier)
    )
    return {
        "id": "test",
        "track": "refresh",
        "source": {"task_id": "test"},
        "visual_packets": packets,
    }


def test_refresh_session_spans_the_gap_between_reviewed_windows(
    tmp_path, monkeypatch
):
    from egocua.sampling_media import prepare_session_tasks

    definition = refresh_session_source(tmp_path, monkeypatch)
    prepared = prepare_session_tasks([definition], tmp_path, 1)
    pool = prepared[0]["visual_session"]["frequency_samples"]

    assert "visual_session" not in definition
    assert prepared[0]["visual_session"]["update_onset_seconds"] == 104.0
    assert prepared[0]["visual_session"]["reviewed_seconds"] == [
        100.0,
        101.0,
        104.0,
        105.0,
    ]
    assert pool["window_seconds"] == [100.0, 105.0]
    assert pool["timestamps_seconds"] == [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    assert all(
        path.startswith("data/experiments/assets/sampling/")
        for path in pool["frame_paths"]
    )


@pytest.mark.parametrize(("lead_in", "skipped"), [(16, None), (1, [101.0, 103.0])])
def test_refresh_cuts_the_wait_between_the_opening_and_the_change(
    tmp_path, monkeypatch, lead_in, skipped
):
    from egocua.sampling_media import prepare_session_tasks

    monkeypatch.setattr("egocua.sampling_media.REFRESH_LEAD_IN_SECONDS", lead_in)
    definition = refresh_session_source(tmp_path, monkeypatch)
    session = prepare_session_tasks([definition], tmp_path, 1)[0]["visual_session"]

    assert session.get("skipped_seconds") == skipped


def test_session_frames_after_the_opening_need_a_reviewed_window(
    tmp_path, monkeypatch
):
    from egocua.sampling_media import prepare_session_tasks

    definition = {**refresh_session_source(tmp_path, monkeypatch), "track": "graft"}
    with pytest.raises(ValueError, match="reviewed stream window"):
        prepare_session_tasks([definition], tmp_path, 1)


def test_graft_session_carries_the_reviewed_follow_up_onset(tmp_path, monkeypatch):
    from egocua.sampling_media import prepare_session_tasks

    stream = {"session_seconds": [100, 105], "update_onset_seconds": 103.5}
    definition = {
        **refresh_session_source(tmp_path, monkeypatch, stream),
        "track": "graft",
    }
    session = prepare_session_tasks([definition], tmp_path, 1)[0]["visual_session"]

    assert session["update_onset_seconds"] == 103.5
    assert session["frequency_samples"]["window_seconds"] == [100.0, 105.0]


def test_session_image_limit_comes_only_from_its_source_record(tmp_path, monkeypatch):
    from egocua.sampling_media import prepare_session_tasks

    definition = refresh_session_source(
        tmp_path, monkeypatch,
        stream={"session_seconds": [100, 105], "wearer_max_image_edge": 1280},
    )
    prepared = prepare_session_tasks([definition], tmp_path, 1)[0]
    assert prepared["visual_session"]["wearer_max_image_edge"] == 1280
    assert "visual_session" not in definition


def test_session_without_image_limit_keeps_legacy_rendering(tmp_path, monkeypatch):
    from egocua.sampling_media import prepare_session_tasks

    definition = refresh_session_source(tmp_path, monkeypatch)
    prepared = prepare_session_tasks([definition], tmp_path, 1)[0]
    assert "wearer_max_image_edge" not in prepared["visual_session"]


@pytest.mark.parametrize("packing", [1, 4])
def test_session_packing_comes_from_its_source_record(tmp_path, monkeypatch, packing):
    from egocua.sampling_media import prepare_session_tasks

    definition = refresh_session_source(
        tmp_path, monkeypatch,
        stream={"session_seconds": [100, 105], "wearer_frames_per_image": packing},
    )
    prepared = prepare_session_tasks([definition], tmp_path, 1)[0]
    assert prepared["visual_session"]["wearer_frames_per_image"] == packing
    assert "visual_session" not in definition


def test_session_without_packing_keeps_condition_rendering(tmp_path, monkeypatch):
    from egocua.sampling_media import prepare_session_tasks

    definition = refresh_session_source(tmp_path, monkeypatch)
    prepared = prepare_session_tasks([definition], tmp_path, 1)[0]
    assert "wearer_frames_per_image" not in prepared["visual_session"]


@pytest.mark.parametrize("packing", [2, True])
def test_session_rejects_invalid_source_packing(tmp_path, monkeypatch, packing):
    from egocua.sampling_media import prepare_session_tasks

    definition = refresh_session_source(
        tmp_path, monkeypatch,
        stream={"session_seconds": [100, 105], "wearer_frames_per_image": packing},
    )
    with pytest.raises(ValueError, match="wearer_frames_per_image"):
        prepare_session_tasks([definition], tmp_path, 1)


@pytest.mark.parametrize(
    ("track", "stream", "message"),
    [
        ("refresh", {"update_onset_seconds": 102}, "changes at V1's first frame"),
        ("graft", {"session_seconds": [100, 105]}, "only Graft records"),
        (
            "clarify",
            {"session_seconds": [100, 105], "update_onset_seconds": 103},
            "only Graft records",
        ),
        ("refresh", {"session_seconds": [100.5, 105]}, "cover reviewed frames"),
        ("refresh", {"session_seconds": [100, 104.5]}, "cover reviewed frames"),
        (
            "graft",
            {"session_seconds": [100, 105], "update_onset_seconds": 106},
            "outside the stream",
        ),
    ],
)
def test_stream_windows_keep_one_owner_for_each_change(
    tmp_path, monkeypatch, track, stream, message
):
    from egocua.sampling_media import prepare_session_tasks

    definition = {
        **refresh_session_source(tmp_path, monkeypatch, stream),
        "track": track,
    }
    with pytest.raises(ValueError, match=message):
        prepare_session_tasks([definition], tmp_path, 1)


def test_source_preparation_leaves_canonical_task_unchanged(source_sampling):
    original, prepared = source_sampling
    assert "frequency_samples" not in original["visual_packets"][0]


def test_source_preparation_keeps_absolute_source_time_and_image_size(source_sampling):
    _, prepared = source_sampling
    pool = prepared[0]["visual_packets"][0]["frequency_samples"]
    assert (
        pool["timestamps_seconds"],
        pool["image_size"],
        pool["sources"][0]["media_start_seconds"],
    ) == (
        [100.5, 101.5, 102.5],
        [16, 12],
        100,
    )


def test_source_preparation_does_not_write_images(source_sampling, tmp_path):
    assert not (tmp_path / "data").exists()


def test_sampling_cache_is_separate_from_formal_observations(source_sampling):
    _, prepared = source_sampling
    paths = prepared[0]["visual_packets"][0]["frequency_samples"]["frame_paths"]
    assert all(path.startswith("data/experiments/assets/sampling/") for path in paths)


def test_materialization_uses_local_video_time_and_reuses_frames(
    source_sampling, tmp_path, monkeypatch
):
    _, prepared = source_sampling
    commands = []

    def decode(command, **kwargs):
        commands.append(command)
        for index in range(1, int(command[command.index("-frames:v") + 1]) + 1):
            Image.new("RGB", (16, 12)).save(Path(command[-1] % index))

    monkeypatch.setattr("egocua.sampling_media.subprocess.run", decode)
    materialize_sampling_frames(prepared, tmp_path)
    materialize_sampling_frames(prepared, tmp_path)
    assert [command[command.index("-ss") + 1] for command in commands] == [
        "0.500000000"
    ]
    assert "source_sha256" not in prepared[0]["visual_packets"][0]["frequency_samples"]


def test_missing_source_stops_extraction(source_sampling, tmp_path):
    _, prepared = source_sampling
    (tmp_path / "source/clip.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        materialize_sampling_frames(prepared, tmp_path)


def test_batch_extraction_preserves_source_time_order(source_sampling, tmp_path):
    _, prepared = source_sampling
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=red:size=16x12:rate=10:duration=2",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=16x12:rate=10:duration=2",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            str(tmp_path / "source/clip.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    materialize_sampling_frames(prepared, tmp_path)
    pool = prepared[0]["visual_packets"][0]["frequency_samples"]
    colors = []
    for path in pool["frame_paths"]:
        with Image.open(tmp_path / path) as image:
            red, _, blue = image.convert("RGB").getpixel((0, 0))
            colors.append("red" if red > blue else "blue")
    assert colors == ["red", "red", "blue"]


def test_missing_frequency_asset_is_rejected_by_task_validation(tmp_path):
    from egocua.schema import _validate_packets

    definition = sampled_task("graft")
    for path in definition["visual_packets"][0]["frame_paths"]:
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"original frame")
    with pytest.raises(ValueError, match="samples"):
        _validate_packets(deepcopy(definition), tmp_path)


@pytest.mark.parametrize("interval", [0, -1, True, float("nan"), float("inf"), 5, 10])
def test_repetition_requires_a_valid_shorter_target_interval(interval):
    with pytest.raises(ValueError):
        replace(condition(5), frame_repeat_to_interval_seconds=interval)


def test_repetition_requires_source_time_sampling():
    with pytest.raises(ValueError, match="shorter target"):
        replace(STANDARD_CONDITIONS["online"], frame_repeat_to_interval_seconds=1)


@pytest.mark.parametrize("window", [[10.25, 16.75], [10.25, 10.25], [10.25, 12.75]])
def test_repetition_matches_dense_count_without_new_frames(window):
    packet = sampled_task("graft")["visual_packets"][0]
    packet["frequency_samples"]["window_seconds"] = window
    sparse, repeated, dense = (
        sample_packet(packet, 5),
        sample_packet(packet, 5, 1),
        sample_packet(packet, 1),
    )
    assert (len(repeated["frame_paths"]), set(repeated["frame_paths"])) == (
        len(dense["frame_paths"]),
        set(sparse["frame_paths"]),
    )


def test_repeated_frames_keep_real_source_timestamps_and_order():
    repeated = sample_packet(sampled_task("graft")["visual_packets"][0], 5, 1)
    assert repeated["timestamps_seconds"] == [10.25] * 4 + [15.25] * 3

