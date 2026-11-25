from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import pytest

from egocua.environment import EnvironmentError as SiteEnvironmentError
from egocua.environment import load_site_data
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty
from tools.audit.audit_role_recoverability import audit
from tools.audit import audit_role_recoverability

ROOT = Path(__file__).resolve().parents[1]


class RoleNonRecoverabilityTest(unittest.TestCase):
    @unittest.skipIf(live_corpus_empty(), ARCHIVED_CORPUS)
    def test_no_public_site_exposes_explicit_answer_fields(self) -> None:
        report = audit(ROOT)
        for task_id, record in report["tasks"].items():
            with self.subTest(task=task_id):
                self.assertEqual(record["blocking_paths"], [])
                self.assertEqual(record["status"], "pass")
        self.assertEqual(report["total_leaks"], 0)
        self.assertGreaterEqual(report["advisory_selection_fingerprints"], 0)


@pytest.fixture
def role_audit_catalog(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    task = {
        "id": "brand-gap",
        "site": "egomart",
        "track": "refresh",
        "refresh": {"initial_entity_id": "target"},
        "goal": {
            "acceptable_states": [{"selected_entity_ids": ["target"], "fields": {}}]
        },
        "truth": {"site_scope_entity_ids": ["target", "neighbor"]},
    }
    catalog = {
        "site": "egomart",
        "supported_fields": [],
        "entities": [
            {"entity_id": "target", "name": "Cup", "attributes": {"Brand": "Acme"}},
            {
                "entity_id": "neighbor",
                "name": "Cup",
                "attributes": {"Material": "Glass"},
            },
        ],
    }
    monkeypatch.setattr(audit_role_recoverability, "load_construction_tasks", lambda _, task_ids=None: [task])
    monkeypatch.setattr(audit_role_recoverability, "load_site_data", lambda *_: catalog)
    return catalog


def test_brand_presence_gap_is_visible_but_advisory(
    role_audit_catalog: dict[str, Any],
) -> None:
    report = audit(ROOT)
    site = report["sites"]["egomart"]
    expected = [{"field": "brand", "present_in_group": 1.0, "present_in_rest": 0.0}]

    assert site["contrasts"][1]["field_presence"] == expected
    assert (
        report["tasks"]["brand-gap"]["advisory_contrast"]["field_presence"] == expected
    )
    assert site["advisory_selection_fingerprints"] == 1
    assert (site["blocking_leaks"], site["leaks"], report["total_leaks"]) == (0, 0, 0)
    assert report["tasks"]["brand-gap"]["status"] == "pass"
    assert "brand" not in role_audit_catalog["entities"][0]


@pytest.mark.parametrize("field", ["answer", "benchmark_role", "is_target"])
def test_explicit_answer_fields_block_task_site_and_total(
    role_audit_catalog: dict[str, Any], field: str
) -> None:
    role_audit_catalog["entities"][0]["attributes"][field] = "target"

    report = audit(ROOT)
    site = report["sites"]["egomart"]
    task = report["tasks"]["brand-gap"]

    assert task["blocking_paths"] == [f"$.entities[0].attributes.{field}"]
    assert task["status"] == "fail"
    assert (
        task["blocking_leaks"],
        site["blocking_leaks"],
        site["leaks"],
        report["total_leaks"],
    ) == (1, 1, 1, 1)


class SiteLoadGateTest(unittest.TestCase):
    def _load(self, entities: list[dict[str, object]]) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sites" / "egomaps").mkdir(parents=True)
            (root / "sites" / "egomaps" / "places.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "site": "egomaps",
                        "supported_fields": [],
                        "places": entities,
                    }
                )
            )
            load_site_data(root, "egomaps")

    def _place(self, place_id: str, **overrides: object) -> dict[str, object]:
        return {
            "place_id": place_id,
            "name": f"Place {place_id}",
            "kind": "cafe",
            "summary": "Corner cafe with pastries.",
            "address": "1 Main Street",
            **overrides,
        }

    def test_a_listing_may_not_narrate_its_benchmark_role(self) -> None:
        with self.assertRaisesRegex(SiteEnvironmentError, "narrate its benchmark role"):
            self._load(
                [
                    self._place("l_1"),
                    self._place(
                        "l_2", summary="A similarly named cafe two blocks away."
                    ),
                ]
            )

    def test_a_listing_may_not_quote_evaluator_vocabulary(self) -> None:
        with self.assertRaisesRegex(SiteEnvironmentError, "evaluator language"):
            self._load(
                [
                    self._place("l_1", summary="The cafe shown in the source video."),
                ]
            )

    def test_one_upstream_item_may_not_be_listed_twice(self) -> None:
        page = {"record": "listing", "page_url": "https://example.invalid/watch?v=abc"}
        with self.assertRaisesRegex(
            SiteEnvironmentError, "lists one upstream item twice"
        ):
            self._load(
                [
                    self._place("l_1", source=page),
                    self._place("l_2", source=dict(page)),
                ]
            )


if __name__ == "__main__":
    unittest.main()
