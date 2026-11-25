import json
from pathlib import Path

from tools.catalog.propose_amazon_landings import CAPTURES, _parse, _payload


def test_parse_rejects_sponsored_standard_answer() -> None:
    payload = {
        "role_requirements": [{"role": "target", "required_stage": "after_change"}],
        "before_results": [],
        "after_results": [{"asin": "B012345678", "sponsored": True}],
    }
    text = """{"task_assessment":{"useful":true,
    "scene_update_website_consistent":true,"continuous_change":true,
    "verifiable_from_public_metadata":true,"issue":""},
    "role_landings":[{"role":"target","asin":"B012345678","reason":"match"}]}"""

    try:
        _parse(text, payload)
    except ValueError as exc:
        assert "sponsored" in str(exc)
    else:
        raise AssertionError("sponsored landing was accepted")


def test_payload_honors_explicit_role_stage(tmp_path: Path) -> None:
    task = {"id": "refresh-stage", "track": "refresh"}
    record = {
        "task_id": "refresh-stage",
        "construct": {
            "task_blueprint": {
                "initial_target_role": "target",
                "final_target_role": "target",
                "later_distractor_role": "distractor",
                "catalog_role_stages": {
                    "target": "after_change",
                    "distractor": "after_change",
                },
                "instruction": "Add the book I keep holding.",
                "environment_role_ids": {
                    "target": "old-target",
                    "distractor": "old-distractor",
                },
            }
        },
    }
    authored_path = tmp_path / "data/instructions/authored.json"
    authored_path.parent.mkdir(parents=True)
    authored_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "instructions": {
                    "refresh-stage": {"base": "Add the book I keep holding."}
                },
            }
        )
    )
    capture_root = tmp_path / CAPTURES
    capture_root.mkdir(parents=True)
    (capture_root / "refresh-stage--before_change.json").write_text(
        json.dumps(
            {
                "query": "book",
                "results": [{"asin": "BEFORE", "title": "Before"}],
            }
        )
    )
    (capture_root / "refresh-stage--after_change.json").write_text(
        json.dumps(
            {
                "query": "named book",
                "results": [
                    {"asin": "TARGET", "title": "Target"},
                    {"asin": "DISTRACTOR", "title": "Distractor"},
                ],
            }
        )
    )

    payload = _payload(tmp_path, task, record)

    requirements = {row["role"]: row for row in payload["role_requirements"]}
    assert requirements["target"]["required_stage"] == "after_change"
    assert requirements["distractor"]["required_stage"] == "after_change"
