from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.extract_video_evidence import _read_manifest, _validate_packet_sampling
from tools.validate_frame_provenance import ProvenanceError, validate_provenance


def _sidecar(
    root: Path, *, digest: str, frame_path: str = "artifacts/frames/one.jpg"
) -> Path:
    destination = root / "data" / "provenance.json"
    destination.parent.mkdir(parents=True)
    destination.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "videos": {
                    "fixture-video": {
                        "duration_seconds": 10.0,
                        "source_start_seconds": 0.0,
                        "source_end_seconds": 10.0,
                    }
                },
                "packet_sampling": [
                    {
                        "task_id": "fixture-1",
                        "packet_id": "V0",
                        "sampling_method": "uniform_inclusive_endpoints",
                        "frame_count": 1,
                        "start_seconds": 1.5,
                        "end_seconds": 1.5,
                        "inter_frame_seconds": None,
                        "effective_hz": None,
                    }
                ],
                "frames": [
                    {
                        "task_id": "fixture-1",
                        "packet_id": "V0",
                        "frame_index": 0,
                        "media_id": "fixture-video",
                        "timestamp_seconds": 1.5,
                        "media_timestamp_seconds": 1.5,
                        "frame_path": frame_path,
                        "sha256": digest,
                    }
                ],
            }
        )
    )
    return destination


def _task(family: str, packets: list[tuple[str, list[float]]]) -> dict[str, object]:
    return {
        "task_id": f"fixture-{family}",
        "family": family,
        "packets": [
            {
                "id": packet_id,
                "timestamps_seconds": timestamps,
                "frame_paths": [
                    f"frames/{packet_id}-{index}.jpg"
                    for index in range(len(timestamps))
                ],
            }
            for packet_id, timestamps in packets
        ],
    }


class FrameProvenanceTest(unittest.TestCase):
    def test_rejected_screening_record_requires_known_reason_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "dataset": {"name": "fixture", "source_root": "videos"},
                        "tasks": [
                            {
                                "task_id": "rejected-1",
                                "status": "rejected",
                                "rejection_code": "not-a-taxonomy-code",
                                "rejection_reason": "No supported event.",
                            }
                        ],
                    }
                )
            )

            with self.assertRaisesRegex(ValueError, "invalid rejection_code"):
                _read_manifest(manifest)

    def test_uniform_refresh_packets_report_derived_frequency(self) -> None:
        summaries = _validate_packet_sampling(
            _task(
                "refresh",
                [
                    ("V0", [10.0, 11.666667, 13.333333, 15.0]),
                    ("V1", [17.0, 17.666667, 18.333333, 19.0]),
                ],
            )
        )

        self.assertEqual(
            summaries[0],
            {
                "task_id": "fixture-refresh",
                "packet_id": "V0",
                "sampling_method": "uniform_inclusive_endpoints",
                "frame_count": 4,
                "start_seconds": 10.0,
                "end_seconds": 15.0,
                "inter_frame_seconds": 1.666667,
                "effective_hz": 0.6,
            },
        )
        self.assertEqual(summaries[1]["effective_hz"], 1.5)

    def test_nonuniform_packet_is_registered_as_event_aligned(self) -> None:
        summaries = _validate_packet_sampling(
            _task(
                "graft",
                [("V0", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])],
            )
        )

        self.assertEqual(
            summaries[0],
            {
                "task_id": "fixture-graft",
                "packet_id": "V0",
                "sampling_method": "event_aligned_timestamps",
                "frame_count": 8,
                "start_seconds": 0.0,
                "end_seconds": 8.0,
            },
        )

    def test_track_packet_layout_is_rejected_when_frame_count_is_wrong(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 8 timestamps"):
            _validate_packet_sampling(
                _task(
                    "clarify",
                    [("V0", [0.0, 1.0, 2.0, 3.0])],
                )
            )

    def test_refresh_requires_two_time_ordered_packets(self) -> None:
        with self.assertRaisesRegex(ValueError, "packets must be ordered"):
            _validate_packet_sampling(
                _task(
                    "refresh",
                    [("V0", [0.0, 1.0, 2.0, 3.0])],
                )
            )

    def test_hash_and_directory_coverage_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "artifacts" / "frames" / "one.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"frame")
            sidecar = _sidecar(root, digest=hashlib.sha256(b"frame").hexdigest())

            result = validate_provenance(
                [sidecar.relative_to(root)],
                root=root,
                coverage_root="artifacts/frames",
            )

        self.assertEqual(
            result,
            {
                "status": "valid",
                "sidecars": 1,
                "frames": 1,
                "coverage_files": 1,
            },
        )

    def test_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "artifacts" / "frames" / "one.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"frame")
            sidecar = _sidecar(root, digest="0" * 64)

            with self.assertRaisesRegex(ProvenanceError, "SHA-256 mismatch"):
                validate_provenance([sidecar.relative_to(root)], root=root)

    def test_clipped_media_preserves_original_source_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "artifacts" / "frames" / "one.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"frame")
            sidecar = _sidecar(root, digest=hashlib.sha256(b"frame").hexdigest())
            payload = json.loads(sidecar.read_text())
            payload["videos"]["fixture-video"].update(
                {
                    "duration_seconds": 40.0,
                    "source_start_seconds": 890.0,
                    "source_end_seconds": 930.0,
                }
            )
            payload["packet_sampling"][0].update(
                {"start_seconds": 898.4, "end_seconds": 898.4}
            )
            payload["frames"][0].update(
                {"timestamp_seconds": 898.4, "media_timestamp_seconds": 8.4}
            )
            sidecar.write_text(json.dumps(payload))

            result = validate_provenance([sidecar.relative_to(root)], root=root)

        self.assertEqual(result["status"], "valid")

    def test_sampling_summary_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "artifacts" / "frames" / "one.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_bytes(b"frame")
            sidecar = _sidecar(root, digest=hashlib.sha256(b"frame").hexdigest())
            payload = json.loads(sidecar.read_text())
            payload["packet_sampling"][0]["frame_count"] = 2
            sidecar.write_text(json.dumps(payload))

            with self.assertRaisesRegex(ProvenanceError, "frame_count does not match"):
                validate_provenance([sidecar.relative_to(root)], root=root)

    def test_untracked_frame_is_rejected_by_coverage_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "artifacts" / "frames"
            frames.mkdir(parents=True)
            (frames / "one.jpg").write_bytes(b"frame")
            (frames / "untracked.jpg").write_bytes(b"extra")
            sidecar = _sidecar(root, digest=hashlib.sha256(b"frame").hexdigest())

            with self.assertRaisesRegex(ProvenanceError, "untracked"):
                validate_provenance(
                    [sidecar.relative_to(root)],
                    root=root,
                    coverage_root="artifacts/frames",
                )


if __name__ == "__main__":
    unittest.main()
