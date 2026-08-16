from __future__ import annotations

import hashlib
import json
from typing import Any

from egocua.construction import opaque_item_id
from tools.model_preaudit_selection import _manifest_digests, _proposed_entities, _summary


def test_preaudit_summary_marks_incomplete_batches_partial() -> None:
    results: list[dict[str, Any]] = [
        {"status": "completed", "verdict": {"pass": True}},
        {"status": "completed", "verdict": {"pass": False}},
        {"status": "error", "error": "quota exhausted"},
    ]

    assert _summary(results) == {
        "status": "partial",
        "requested": 3,
        "completed": 2,
        "passed": 1,
        "failed": 1,
        "errors": 1,
    }


def test_manifest_digests_cover_every_declared_input(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"tasks": []}\n')
    second.write_text('{"tasks": [{"task_id": "example"}]}\n')

    records = _manifest_digests(
        {"source_manifests": ["first.json", "second.json"]}, tmp_path
    )

    assert records == [
        {
            "path": "first.json",
            "sha256": hashlib.sha256(first.read_bytes()).hexdigest(),
        },
        {
            "path": "second.json",
            "sha256": hashlib.sha256(second.read_bytes()).hexdigest(),
        },
    ]


def test_refresh_preaudit_includes_later_distractor(tmp_path) -> None:
    task_id = "refresh-example"
    initial_id = opaque_item_id(task_id, "first_plate")
    later_id = opaque_item_id(task_id, "second_plate")
    filler_ids = [f"p_{index:012x}" for index in range(14)]
    visible_ids = [initial_id, later_id, *filler_ids]
    image_paths = {}
    for item_id in (initial_id, later_id):
        image_path = tmp_path / "sites" / "egomart" / "static" / f"{item_id}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(item_id.encode())
        image_paths[item_id] = image_path.relative_to(tmp_path).as_posix()
    product_store = {
        "schema_version": 1,
        "site": "egomart",
        "products": [
            {
                "item_id": item_id,
                "name": item_id,
                    "image_path": image_paths.get(
                        item_id, f"sites/egomart/static/{item_id}.jpg"
                    ),
            }
            for item_id in visible_ids
        ],
    }
    (tmp_path / "sites" / "egomart" / "products.json").write_text(
        json.dumps(product_store)
    )
    source_task = {
        "construct": {
            "task_blueprint": {
                "initial_target_role": "first_plate",
                "final_target_role": "first_plate",
                "later_distractor_role": "second_plate",
            }
        }
    }
    task = {
        "id": task_id,
        "track": "refresh",
        "site": "egomart",
        "goal": {"acceptable_states": [{
            "selected_entity_ids": [initial_id],
            "fields": {},
            "submission_completed": True,
        }]},
        "refresh": {"initial_entity_id": initial_id},
    }

    entities = _proposed_entities(source_task, task, tmp_path)

    roles_by_id = {entity["entity_id"]: entity["roles"] for entity in entities}
    assert roles_by_id[later_id] == ["later_distractor"]
