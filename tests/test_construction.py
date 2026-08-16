from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from egocua.construction import ConstructionError, compile_task


def _spec(root: Path, task_id: str, roles: list[str]) -> dict[str, Any]:
    products = []
    for index, role in enumerate([*roles, *[f"distractor_{i}" for i in range(16 - len(roles))]]):
        path = Path("product-images") / f"image-{index}.jpg"
        image = root / path
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"image-{index}".encode())
        products.append({
            "role": role,
            "name": f"Product {index}",
            "price": f"${index + 1}.00",
            "image_path": path.as_posix(),
            "image_provenance": {
                "source_url": f"https://example.invalid/image-{index}.jpg",
                "retrieved_at": "2026-08-10T00:00:00Z",
                "license_or_access_note": "Unit-test fixture.",
            },
            "tier": "target" if index < len(roles) else (
                "hard" if index < len(roles) + 3 else "medium" if index < len(roles) + 7 else "easy"
            ),
            "attributes": {"variant": f"Variant {index}"},
        })
    return {"schema_version": 1, "task_id": task_id, "products": products}


def _packets(root: Path, track: str) -> list[dict[str, Any]]:
    counts = (("V0", 4), ("V1", 4)) if track == "refresh" else (("V0", 8),)
    packets = []
    for packet_id, count in counts:
        paths = []
        for index in range(count):
            path = Path("frames") / f"{track}-{packet_id}-{index}.jpg"
            frame = root / path
            frame.parent.mkdir(parents=True, exist_ok=True)
            frame.write_bytes(f"frame-{track}-{packet_id}-{index}".encode())
            paths.append(path.as_posix())
        packets.append({"id": packet_id, "frame_paths": paths})
    return packets


def _record(root: Path, track: str) -> tuple[dict[str, Any], dict[str, Any]]:
    task_id = f"seed-{track}"
    common: dict[str, Any] = {
        "task_id": task_id,
        "status": "video_screened",
        "family": track,
        "packets": _packets(root, track),
    }
    if track == "graft":
        common.update({
            "subtype": "P",
            "construct": {"task_blueprint": {
                "instructions": {"a": "Find A", "b": "Also add B", "bundled": "Find A and B"},
                "initial_goal_roles": [["base_product"]],
                "final_goal_roles": [["base_product", "second_product"]],
                "target_granularity": "variant",
            }},
        })
        roles = ["base_product", "second_product"]
    elif track == "refresh":
        common.update({
            "subtype": "replace",
            "construct": {"task_blueprint": {
                "instruction": "Find the product I switched to",
                "initial_target_role": "first_product",
                "final_target_role": "second_product",
                "target_granularity": "variant",
            }},
        })
        roles = ["first_product", "second_product"]
    else:
        common.update({
            "subtype": "variant",
            "construct": {"task_blueprint": {
                "ambiguous_instruction": "Find this product",
                "oracle_instruction": "Find the blue product",
                "candidate_roles": ["red_product", "blue_product"],
                "hidden_target_role": "blue_product",
                "target_granularity": "variant",
                "valid_slot": "variant",
                "canonical_question": "Do you mean the red or blue product?",
                "canonical_answer": "The blue one.",
            }},
        })
        roles = ["red_product", "blue_product"]
    return common, _spec(root, task_id, roles)


def _source_identity(record: dict[str, Any]) -> dict[str, str]:
    return {
        "dossier": "data/source.json",
        "dossier_sha256": "1" * 64,
        "task_id": str(record["task_id"]),
        "record_sha256": "2" * 64,
        "frame_provenance": "data/source_frame_provenance.json",
        "frame_provenance_sha256": "3" * 64,
    }


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_compile_task_produces_a_valid_executable_task(track: str, tmp_path: Path) -> None:
    record, spec = _record(tmp_path, track)

    task, products = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )

    assert task["track"] == track
    assert len(products) == 16
    assert all(set(item).isdisjoint({"role", "tier"}) for item in products)
    assert "environment" not in task
    assert all(item["item_id"].startswith("p_") for item in products)
    assert task["source"] == _source_identity(record)


def test_product_ids_and_order_do_not_depend_on_spec_order(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "graft")
    first_task, first_products = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )
    spec["products"].reverse()

    second_task, second_products = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )

    assert first_task == second_task
    assert first_products == second_products


def test_compile_task_refuses_to_guess_a_missing_role(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "refresh")
    spec["products"][0]["role"] = "unknown_role"

    with pytest.raises(ConstructionError, match="initial_target_role"):
        compile_task(
            record, spec, source_identity=_source_identity(record), root=tmp_path,
        )


def test_compile_task_requires_provenance_for_declared_ratings(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "graft")
    spec["products"][0]["rating"] = 4.8
    spec["products"][0]["ratings_count"] = 143

    with pytest.raises(ConstructionError, match="rating_provenance"):
        compile_task(
            record, spec, source_identity=_source_identity(record), root=tmp_path,
        )

    spec["products"][0]["rating_provenance"] = {
        "source_url": "https://example.invalid/product",
        "retrieved_at": "2026-08-10T00:00:00Z",
        "access_note": "Unit-test fixture.",
    }
    _, products = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )

    first = next(item for item in products if item["name"] == "Product 0")
    assert first["rating"] == 4.8
    assert first["ratings_count"] == 143
    assert first["rating_provenance"]["source_url"] == "https://example.invalid/product"


def test_compile_task_accepts_explicit_mock_rating_provenance(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "graft")
    spec["products"][0].update({
        "rating": 4.8,
        "ratings_count": 143,
        "rating_provenance": {
            "method": "benchmark_authored_mock_product",
            "created_at": "2026-08-10T00:00:00Z",
            "access_note": "Frozen synthetic experimental metadata.",
        },
    })

    _, compiled = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )

    first = next(item for item in compiled if item["name"] == "Product 0")
    assert first["rating_provenance"]["method"] == "benchmark_authored_mock_product"


def test_compile_task_requires_provenance_for_compatibility_claims(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "graft")
    spec["products"][0]["attributes"]["compatible_with"] = "Example cap system"

    with pytest.raises(ConstructionError, match="compatibility_provenance"):
        compile_task(
            record, spec, source_identity=_source_identity(record), root=tmp_path,
        )

    spec["products"][0]["compatibility_provenance"] = {
        "source_url": "https://example.invalid/manual.pdf",
        "retrieved_at": "2026-08-10T00:00:00Z",
        "access_note": "Unit-test fixture.",
    }
    _, products = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )

    first = next(item for item in products if item["name"] == "Product 0")
    assert first["attributes"]["compatible_with"] == "Example cap system"


def test_refresh_goal_is_derived_from_its_single_final_target_role(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "refresh")

    task, products = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )

    second_id = next(item["item_id"] for item in products if item["name"] == "Product 1")
    assert task["goal"]["acceptable_states"][0]["selected_entity_ids"] == [second_id]


def test_refresh_may_register_a_later_distractor_without_changing_the_goal(
    tmp_path: Path,
) -> None:
    record, spec = _record(tmp_path, "refresh")
    record["construct"]["task_blueprint"]["later_distractor_role"] = "distractor_0"

    task, products = compile_task(
        record, spec, source_identity=_source_identity(record), root=tmp_path,
    )

    assert task["goal"]["acceptable_states"][0]["selected_entity_ids"] == [
        next(
            product["item_id"]
            for product in products
            if product["name"] == "Product 1"
        )
    ]


def test_clarify_hidden_target_must_be_one_of_exactly_two_candidates(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "clarify")
    record["construct"]["task_blueprint"]["hidden_target_role"] = "distractor_0"

    with pytest.raises(ConstructionError, match="must be one of the candidates"):
        compile_task(
            record, spec, source_identity=_source_identity(record), root=tmp_path,
        )


def test_clarify_compiler_rejects_the_obsolete_nested_blueprint(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "clarify")
    blueprint = record["construct"]["task_blueprint"]
    blueprint["instructions"] = {
        "ambiguous": blueprint.pop("ambiguous_instruction"),
        "oracle": blueprint.pop("oracle_instruction"),
    }

    with pytest.raises(ConstructionError, match="obsolete fields"):
        compile_task(
            record, spec, source_identity=_source_identity(record), root=tmp_path,
        )


def test_graft_compiler_rejects_obsolete_b_essential_witness(tmp_path: Path) -> None:
    record, spec = _record(tmp_path, "graft")
    record["construct"]["task_blueprint"]["b_essential_witness"] = ["base_product"]

    with pytest.raises(ConstructionError, match="obsolete field 'b_essential_witness'"):
        compile_task(
            record, spec, source_identity=_source_identity(record), root=tmp_path,
        )
