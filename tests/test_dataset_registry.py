"""configs/datasets.json is the only statement of what reference/ must contain.

reference/ is untracked, so a fresh checkout has an empty one and nothing in the
repository would otherwise say which third-party data the benchmark depends on.
The registry says it; these tests keep the registry honest about the tree that
exists, and keep the pipeline from reading anything the registry calls literature.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from egocua.sources import iter_manifests

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs/datasets.json"


def _registry() -> dict:
    return json.loads(REGISTRY.read_text())


class DatasetRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = _registry()
        self.entries = self.registry["datasets"]

    def test_every_entry_declares_a_known_class(self) -> None:
        """An entry lives under its class, or is the whole class in one row."""

        classes = set(self.registry["classes"])
        for entry in self.entries:
            class_dir = f"{self.registry['root']}/{entry['class']}"
            with self.subTest(dataset=entry["id"]):
                self.assertIn(entry["class"], classes)
                self.assertTrue(
                    entry["path"] == class_dir
                    or entry["path"].startswith(f"{class_dir}/")
                )

    def test_paths_are_unique(self) -> None:
        paths = [entry["path"] for entry in self.entries]
        self.assertEqual(len(paths), len(set(paths)))

    def test_a_present_directory_is_registered(self) -> None:
        """A dataset on disk that nothing declares is a dataset nobody can account for."""

        registered = {entry["path"] for entry in self.entries}
        for class_name in self.registry["classes"]:
            class_dir = ROOT / self.registry["root"] / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.name.startswith("."):
                    continue
                relative = path.relative_to(ROOT).as_posix()
                with self.subTest(path=relative):
                    self.assertTrue(
                        relative in registered
                        or f"{self.registry['root']}/{class_name}" in registered,
                        f"{relative} is on disk but not in configs/datasets.json",
                    )

    def test_screening_manifests_only_read_registered_sources(self) -> None:
        source_paths = tuple(
            entry["path"] for entry in self.entries if entry["class"] == "sources"
        )
        for manifest_path, manifest in iter_manifests(ROOT):
            source_root = (manifest.get("dataset") or {}).get("source_root")
            if not source_root:
                continue
            with self.subTest(manifest=manifest_path.name):
                self.assertTrue(
                    source_root.startswith(source_paths),
                    f"{manifest_path.name} reads {source_root}, which no registered "
                    f"source dataset covers",
                )

    def test_registry_does_not_duplicate_derived_task_counts(self) -> None:
        for entry in self.entries:
            with self.subTest(dataset=entry["id"]):
                self.assertNotIn("tasks_built", entry)

    def test_a_candidate_says_what_blocks_it(self) -> None:
        for entry in self.entries:
            if entry["class"] != "candidates":
                continue
            with self.subTest(dataset=entry["id"]):
                self.assertIn(entry["state"], {"partial", "unusable"})
                self.assertTrue(entry.get("blocked_on"))


if __name__ == "__main__":
    unittest.main()
