"""Multiple acquired files retain one recording timeline without inventing its gaps."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from PIL import Image

from egocua.confidence_packets import (
    CONTINUOUS_VIDEO_REQUIREMENT,
    ConfidencePacketError,
    _continuous_video_ref,
    build_review_packet,
)
from egocua.lineage import (
    LineageError,
    _validate_frame_records,
    media_segment_at,
    source_media_segments,
    source_segment_references,
)
from egocua.sampling_media import prepare_sampling_tasks, prepare_session_tasks
from tools.audit.audit_task_acceptance import _video_check
from tools.audit.validate_frame_provenance import validate_provenance
from tools.build import extract_video_evidence as extractor
from tests.helpers import PLAN, task


@pytest.fixture
def segmented_source(tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    rows = []
    for media_id, start in (("clip-a", 10), ("clip-b", 600)):
        content = media_id.encode()
        (videos / f"{media_id}.mp4").write_bytes(content)
        rows.append(
            {
                "media_id": media_id,
                "original_video_id": "original",
                "source_start_seconds": start,
                "source_end_seconds": start + 10,
                "media_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return {
        "media_id": "clip-a",
        "original_video_id": "original",
        "media_start_seconds": 10,
        "media_segments": rows,
    }


def test_old_single_source_does_not_infer_a_contract_from_acquisition_logs():
    source = {"media_id": "old", "acquisition": {"intervals": [{"requested": 7}, {}]}}
    segments = source_media_segments(source, context="old")
    assert len(segments) == 1
    assert segments[0].source_start_seconds == 0
    assert segments[0].source_end_seconds is None
    assert segments[0].media_sha256 is None


@pytest.mark.parametrize(
    "change",
    [
        "wrong_recording",
        "duplicate_id",
        "overlap",
        "unordered",
        "missing_digest",
        "invalid_digest",
        "nonfinite",
        "boolean",
        "empty_interval",
        "single_segment",
        "missing_parent",
        "primary_offset",
        "primary_digest",
        "primary_id",
    ],
)
def test_invalid_explicit_segments_fail_before_extraction(segmented_source, change):
    source = segmented_source
    rows = source["media_segments"]
    if change == "wrong_recording":
        rows[1]["original_video_id"] = "other-recording"
    elif change == "duplicate_id":
        rows[1]["media_id"] = rows[0]["media_id"]
    elif change == "overlap":
        rows[1]["source_start_seconds"] = 19
    elif change == "unordered":
        rows.reverse()
    elif change == "missing_digest":
        rows[1].pop("media_sha256")
    elif change == "invalid_digest":
        rows[1]["media_sha256"] = "not-a-digest"
    elif change == "nonfinite":
        rows[1]["source_end_seconds"] = float("inf")
    elif change == "boolean":
        rows[1]["source_start_seconds"] = True
    elif change == "empty_interval":
        rows[1]["source_end_seconds"] = rows[1]["source_start_seconds"]
    elif change == "single_segment":
        rows.pop()
    elif change == "missing_parent":
        source.pop("original_video_id")
    elif change == "primary_offset":
        source["media_start_seconds"] = 0
    elif change == "primary_digest":
        source["media_sha256"] = "a" * 64
    else:
        source["media_id"] = "different"
    with pytest.raises(LineageError):
        source_media_segments(source, context="invalid")


@pytest.mark.parametrize("timestamp", [9, 20, 400, 610, float("nan")])
def test_uncovered_source_seconds_are_not_silently_assigned(
    segmented_source, timestamp
):
    segments = source_media_segments(segmented_source, context="test")
    with pytest.raises(LineageError):
        media_segment_at(segments, timestamp, context="test")


@pytest.fixture
def extracted_source(tmp_path, monkeypatch, segmented_source):
    task_id = "candidate-media-refresh-replace-2026110101"
    record = {
        "task_id": task_id,
        "status": "video_screened",
        "family": "refresh",
        "source": segmented_source,
        "construct": {},
        "packets": [
            {
                "id": packet_id,
                "timestamps_seconds": [start + index for index in range(4)],
                "frame_paths": [
                    f"frames/{packet_id}-{index}.jpg" for index in range(4)
                ],
            }
            for packet_id, start in (("V0", 11), ("V1", 601))
        ],
    }
    dossier = {
        "schema_version": 2,
        "dataset": {"name": "HD-EPIC", "source_root": "videos"},
        "tasks": [record],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dossier))
    monkeypatch.setattr(
        extractor,
        "_probe",
        lambda video: {"duration_seconds": 10.0, "width": 32, "height": 24},
    )
    calls = []

    def extract(video, timestamp, destination):
        calls.append((video.name, timestamp))
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (round(timestamp) * 20, 0, 0)).save(destination)

    monkeypatch.setattr(extractor, "_extract_frame", extract)
    sidecar = tmp_path / "provenance.json"
    provenance = extractor.extract(path, tmp_path, sidecar)
    definition = {
        "id": task_id,
        "track": "refresh",
        "source": {"task_id": task_id},
        "visual_packets": deepcopy(record["packets"]),
    }
    return definition, dossier, record, sidecar, provenance, calls


def test_extractor_selects_each_file_by_real_source_time(extracted_source, tmp_path):
    definition, _, record, sidecar, provenance, calls = extracted_source
    assert calls == [
        (name, time) for name in ("clip-a.mp4", "clip-b.mp4") for time in (1, 2, 3, 4)
    ]
    assert [frame["media_id"] for frame in provenance["frames"]] == ["clip-a"] * 4 + [
        "clip-b"
    ] * 4
    _validate_frame_records(definition, record, provenance, root=tmp_path)
    assert (
        validate_provenance([sidecar.relative_to(tmp_path)], root=tmp_path)["status"]
        == "valid"
    )


@pytest.mark.parametrize(
    "change",
    [
        "media_id",
        "source_time",
        "local_time",
        "offset",
        "end",
        "duration",
        "parent",
        "digest",
    ],
)
def test_lineage_rejects_wrong_segment_or_time_mapping(
    extracted_source, tmp_path, change
):
    definition, _, record, _, provenance, _ = extracted_source
    frame = provenance["frames"][-1]
    metadata = provenance["videos"]["clip-b"]
    if change == "media_id":
        frame["media_id"] = "clip-a"
    elif change == "source_time":
        frame["timestamp_seconds"] += 1
    elif change == "local_time":
        frame["media_timestamp_seconds"] += 1
    else:
        field = {
            "offset": "source_start_seconds",
            "end": "source_end_seconds",
            "duration": "duration_seconds",
            "parent": "original_video_id",
            "digest": "media_sha256",
        }[change]
        metadata[field] = (
            "wrong" if change in {"parent", "digest"} else metadata[field] + 1
        )
    with pytest.raises(LineageError):
        _validate_frame_records(definition, record, provenance, root=tmp_path)


def _probe_response(command, **kwargs):
    return SimpleNamespace(
        stdout=json.dumps(
            {"format": {"duration": "10"}, "streams": [{"width": 32, "height": 24}]}
        )
    )


@pytest.mark.parametrize(
    "change",
    [
        None,
        "second_digest",
        "offset",
        "duration",
        "local_time",
        "wrong_owner",
        "continuous_gap",
    ],
)
def test_video_acceptance_checks_per_file_integrity_and_real_offsets(
    extracted_source, tmp_path, monkeypatch, change
):
    definition, dossier, record, sidecar, provenance, _ = extracted_source
    if change == "second_digest":
        record["source"]["media_segments"][1]["media_sha256"] = "a" * 64
    elif change == "offset":
        provenance["videos"]["clip-b"]["source_start_seconds"] += 1
    elif change == "duration":
        record["source"]["media_segments"][1]["source_end_seconds"] += 1
    elif change == "local_time":
        provenance["frames"][-1]["media_timestamp_seconds"] += 1
    elif change == "wrong_owner":
        provenance["frames"][-1]["media_id"] = "clip-a"
    elif change == "continuous_gap":
        record["construct"]["evidence_requirements"] = [CONTINUOUS_VIDEO_REQUIREMENT]
    sidecar.write_text(json.dumps(provenance))
    monkeypatch.setattr(
        "tools.audit.audit_task_acceptance.subprocess.run", _probe_response
    )
    result = _video_check(
        definition, dossier, record, sidecar, tmp_path, verify_source_frames=False
    )
    assert (result["status"] == "pass") is (change is None)
    if change is None:
        assert len(set(result["details"]["video_sha256"].values())) == 2


def test_source_refs_disclose_gap_and_do_not_relax_continuous_requirement(
    extracted_source, tmp_path
):
    definition, dossier, record, _, _, _ = extracted_source
    refs = source_segment_references(record, dossier["dataset"], tmp_path)
    assert refs["unavailable_intervals_seconds"] == [[20, 600]]
    assert len(refs["segments"]) == 2
    assert (
        _continuous_video_ref(definition, record, dossier["dataset"], tmp_path) is None
    )
    record["construct"]["evidence_requirements"] = [CONTINUOUS_VIDEO_REQUIREMENT]
    with pytest.raises(ConfidencePacketError, match="continuous interval"):
        _continuous_video_ref(definition, record, dossier["dataset"], tmp_path)


def test_blind_packet_does_not_gain_unreleased_source_video(
    extracted_source, tmp_path, monkeypatch
):
    definition, dossier, record, _, _, _ = extracted_source
    playable = task("refresh")
    playable["id"] = definition["id"]
    playable["visual_packets"] = definition["visual_packets"]
    monkeypatch.setattr(
        "egocua.confidence_packets._source_material",
        lambda *args: (record, dossier["dataset"]),
    )
    packet = build_review_packet(
        playable,
        root=tmp_path,
        plan=PLAN,
        public_site_path="public.json",
        public_site_sha256="a" * 64,
    )
    assert "source_video_segments" not in packet["materials"]
    assert "continuous_source_video" not in packet["materials"]
    assert "clip-a.mp4" not in json.dumps(packet)


def test_sampling_uses_each_available_window_but_continuous_stream_rejects_gap(
    extracted_source, tmp_path, monkeypatch
):
    definition, dossier, _, _, _, _ = extracted_source
    monkeypatch.setattr(
        "egocua.sampling_media.load_dossier", lambda *args, **kwargs: (None, dossier)
    )
    prepared = prepare_sampling_tasks([definition], tmp_path, (1,))
    assert [
        source["source_video"]
        for packet in prepared[0]["visual_packets"]
        for source in packet["frequency_samples"]["sources"]
    ] == ["videos/clip-a.mp4", "videos/clip-b.mp4"]
    with pytest.raises(LineageError, match="source timestamp 585 has no unique media"):
        prepare_session_tasks([definition], tmp_path, 1)


def test_refresh_cut_lets_each_delivered_span_use_its_own_file(
    extracted_source, tmp_path, monkeypatch
):
    definition, dossier, _, _, _, _ = extracted_source
    monkeypatch.setattr(
        "egocua.sampling_media.load_dossier", lambda *args, **kwargs: (None, dossier)
    )
    monkeypatch.setattr("egocua.sampling_media.REFRESH_LEAD_IN_SECONDS", 1)
    session = prepare_session_tasks([definition], tmp_path, 1)[0]["visual_session"]
    pool = session["frequency_samples"]

    assert session["skipped_seconds"] == [14, 600]
    assert [
        (source["source_video"], source["window_seconds"]) for source in pool["sources"]
    ] == [("videos/clip-a.mp4", [11.0, 14]), ("videos/clip-b.mp4", [600, 604.0])]
    assert pool["timestamps_seconds"] == [11.0, 12.0, 13.0, 14.0, 600.0, 601.0, 602.0, 603.0, 604.0]
    assert [path.split("/")[-3] for path in pool["frame_paths"]] == ["clip-a"] * 4 + [
        "clip-b"
    ] * 5


def test_packet_index_registers_diagnostic_pointer_without_blind_source_frames(
    extracted_source, tmp_path, monkeypatch
):
    from egocua import confidence_packets as builder

    definition, _, record, _, _, _ = extracted_source
    playable = task("refresh")
    playable.update(id=definition["id"], source={"dossier": "manifest.json"})
    playable["visual_packets"] = definition["visual_packets"]
    monkeypatch.setattr(
        builder, "screened_records", lambda root: [(tmp_path / "manifest.json", record)]
    )
    monkeypatch.setattr(
        builder, "load_construction_tasks", lambda *args, **kwargs: [playable]
    )
    monkeypatch.setattr(builder, "load_experiment_plan", lambda path: PLAN)
    monkeypatch.setattr(builder, "load_site_data", lambda *args: {"entities": []})
    monkeypatch.setattr(builder, "task_public_site_data", lambda task, site: site)
    (tmp_path / "pilot.json").write_text("{}")

    index = builder.build_packets(
        tmp_path, output_dir="packets", plan_path="pilot.json"
    )
    row = index["tasks"][playable["id"]]
    assert row["source_segments_dossier"] == "manifest.json"
    saved = json.loads((tmp_path / row["path"]).read_text())
    assert "source_segments_dossier" not in saved
    assert "clip-a.mp4" not in json.dumps(saved)



def test_legacy_recording_group_does_not_read_the_media_offset():
    from egocua.lineage import _recording_identity

    identity = _recording_identity(
        {"task_id": "old", "source": {"media_id": "recording", "media_start_seconds": True}},
        "HD-EPIC",
        {},
    )
    assert identity["recording_id"] == "recording"
