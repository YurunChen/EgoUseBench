from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from egocua.schema import ValidationError, validate_task
from egocua.workspace import resolve_within, write_bytes_atomic
from tests.helpers import site_data, task


class WorkspacePathTest(unittest.TestCase):
    def test_write_bytes_atomic_replaces_the_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "assets" / "image.bin"
            write_bytes_atomic(destination, b"first")
            write_bytes_atomic(destination, b"second")

            self.assertEqual(destination.read_bytes(), b"second")
            self.assertEqual(list(destination.parent.glob(".*.tmp")), [])

    def test_resolve_within_rejects_absolute_and_parent_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inside = root / "assets/frame.jpg"
            inside.parent.mkdir()
            inside.write_bytes(b"frame")

            self.assertEqual(
                resolve_within(root, "assets/frame.jpg", context="frame"),
                inside.resolve(),
            )
            for unsafe in (inside.as_posix(), "../frame.jpg"):
                with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                    resolve_within(root, unsafe, context="frame")

    def test_strict_validation_uses_the_same_containment_rule(self) -> None:
        benchmark_task = task("clarify")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root.parent / "outside-frame.jpg"
            outside.write_bytes(b"frame")
            benchmark_task["visual_packets"][0]["frame_paths"] = [
                outside.as_posix() for _ in range(8)
            ]

            with self.assertRaisesRegex(ValidationError, "must be relative"):
                validate_task(
                    benchmark_task, site_data("clarify"), root=root, require_assets=True
                )
