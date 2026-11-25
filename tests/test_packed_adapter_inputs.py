from __future__ import annotations

import base64
from dataclasses import replace
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from egocua.agents import minimal_loop
from egocua import execution
from egocua.frame_inputs import observation_images
from egocua.timeline import OBSERVATION
from tests.conditions import CONDITIONS


@pytest.fixture
def context(tmp_path):
    paths = []
    for index in range(11):
        path = tmp_path / f"frame-{index}.png"
        Image.new("RGB", (100, 70), (index * 20, 30, 40)).save(path)
        paths.append(path.name)
    initial = {
        "frame_paths": paths[:7],
        "timestamps_seconds": list(range(7)),
        "wearer_frames_per_image": 4,
        "wearer_jpeg_quality": 85,
    }
    newer = {
        "frame_paths": paths[7:],
        "timestamps_seconds": list(range(7, 11)),
        "wearer_frames_per_image": 4,
        "wearer_jpeg_quality": 85,
    }
    context = SimpleNamespace(
        initial_payload={
            "instruction": "Save the item I use.",
            "observations": [initial],
            "clarification_available": False,
        },
        base_url="http://127.0.0.1:12345",
        current_step=0,
        site_data={"entities": []},
        capture_partial_trace=Mock(),
        _asset_root=tmp_path,
        _visual_asset_root=tmp_path / "visual-inputs",
        _session=SimpleNamespace(
            timeline=SimpleNamespace(
                initial_observations=[initial],
                releases=[SimpleNamespace(kind=OBSERVATION, payload=newer)],
            )
        ),
    )
    context.observation_images = MethodType(
        execution.RunContext.observation_images, context
    )

    def advance():
        context.current_step += 1
        return (
            [{"type": "visual_context_update", "step": 2, "payload": newer}]
            if context.current_step == 2
            else []
        )

    context.advance = advance
    context.standby = lambda: False
    return context


def test_minimal_loop_packs_new_releases_once_and_records_layout_without_changing_product_images(
    context, tmp_path, monkeypatch
):
    captured = []
    build_images = Mock(wraps=execution.observation_images)
    monkeypatch.setattr(execution, "observation_images", build_images)
    monkeypatch.setattr(
        minimal_loop,
        "run_experiment",
        lambda task, condition, config, execute: execute(context),
    )

    def transport(body):
        captured.append(body)
        if len(captured) == 1:
            return {
                "content": [
                    {"type": "tool_use", "name": "wait", "id": "wait-1", "input": {}}
                ]
            }
        return {
            "content": [
                {
                    "type": "tool_use",
                    "name": "finish",
                    "id": "finish-1",
                    "input": {"reason": "done"},
                }
            ]
        }

    context.wait = Mock()
    trace = minimal_loop.run_task(
        {"site": "egomart", "track": "graft", "visual_session": {}},
        replace(
            CONDITIONS["online"],
            frame_interval_seconds=1,
            frame_delivery_steps=1,
            batch_seconds=8,
        ),
        transport=transport,
        asset_root=tmp_path,
        default_release_step=2,
        reference_steps=None,
    )
    assert build_images.call_count == 2
    assert [len(row["wearer_image_layouts"]) for row in trace] == [2, 3]
    assert [
        entry["frame_index"]
        for entry in trace[-1]["wearer_image_layouts"][-1]["layout"]
    ] == [0, 1, 2, 3]
    first_images = [
        block
        for block in captured[0]["messages"][0]["content"]
        if block["type"] == "image"
    ]
    assert len(first_images) == 2
    assert all(block["source"]["media_type"] == "image/jpeg" for block in first_images)
    expected = observation_images(context.initial_payload["observations"][0], tmp_path)
    assert [block["source"]["data"] for block in first_images] == [
        image.data for image in expected
    ]
    assert all(
        call.args[1] == context._visual_asset_root
        for call in build_images.call_args_list
    )
    product_path = tmp_path / "product.jpg"
    Image.new("RGB", (90, 40), "yellow").save(product_path)
    blocks = minimal_loop._product_blocks(
        "Product",
        [{"item_id": "product-1", "name": "Product", "image_path": product_path.name}],
        tmp_path,
    )
    product_image = next(block for block in blocks if block["type"] == "image")
    assert (
        base64.b64decode(product_image["source"]["data"]) == product_path.read_bytes()
    )


