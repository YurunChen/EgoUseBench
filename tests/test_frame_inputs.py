from __future__ import annotations

import base64
from copy import deepcopy
from io import BytesIO

import pytest
from PIL import Image

from egocua import frame_inputs
from egocua.frame_inputs import (
    JPEG_QUALITY,
    LABEL_HEIGHT,
    observation_images,
    packed_image_size,
)


@pytest.fixture
def released_frames(tmp_path):
    paths = []
    for index in range(7):
        path = tmp_path / f"frame-{index}.png"
        image = Image.new("RGBA", (80 + index, 60 + index), (index * 30, 50, 100, 255))
        image.putpixel((3, 4), (1, 2, 3, index * 20))
        image.save(path)
        paths.append(path.name)
    return {
        "frame_paths": paths,
        "timestamps_seconds": [index * 1.5 for index in range(7)],
        "wearer_frames_per_image": 4,
    }


def _decoded(image):
    return Image.open(BytesIO(base64.b64decode(image.data)))


@pytest.fixture
def packed_canvases(monkeypatch):
    canvases = []
    save = Image.Image.save

    def capture(image, fp, format=None, **kwargs):
        if format == "JPEG":
            canvases.append(image.copy())
        return save(image, fp, format=format, **kwargs)

    monkeypatch.setattr(Image.Image, "save", capture)
    return canvases


def test_seven_frames_produce_two_fixed_four_slot_images_without_changing_pixels(
    released_frames, tmp_path, packed_canvases
):
    original = deepcopy(released_frames)
    images = observation_images(released_frames, tmp_path)
    assert len(images) == 2
    assert released_frames == original
    assert [len(image.layout) for image in images] == [4, 3]
    for image, canvas in zip(images, packed_canvases, strict=True):
        assert image.media_type == "image/jpeg"
        assert _decoded(image).size == canvas.size
        assert image.encoded_bytes == base64.b64decode(image.data)
        sizes = [frame["original_size"] for frame in image.layout]
        assert canvas.size == packed_image_size(
            max(width for width, _ in sizes), max(height for _, height in sizes), 4
        )
        for frame in image.layout:
            with Image.open(tmp_path / frame["frame_path"]) as source:
                assert (
                    canvas.crop(frame["bounds"]).tobytes()
                    == source.convert("RGB").tobytes()
                )
    assert [frame["frame_index"] for image in images for frame in image.layout] == list(
        range(7)
    )
    assert [
        frame["timestamp_seconds"] for image in images for frame in image.layout
    ] == original["timestamps_seconds"]
    assert images[1].label.endswith("bottom_left: Frame 7 | source t=9 s")
    last = packed_canvases[1]
    width, height = last.size
    blank = last.crop((width // 2, height // 2, width, height))
    assert blank.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_one_frame_tail_keeps_three_empty_black_cells(
    released_frames, tmp_path, packed_canvases
):
    released_frames["frame_paths"] = released_frames["frame_paths"][:5]
    released_frames["timestamps_seconds"] = released_frames["timestamps_seconds"][:5]
    tail = observation_images(released_frames, tmp_path)[1]
    canvas = packed_canvases[1]
    width, height = canvas.size
    assert len(tail.layout) == 1
    for bounds in (
        (width // 2, 0, width, height // 2),
        (0, height // 2, width, height),
    ):
        assert canvas.crop(bounds).getextrema() == ((0, 0), (0, 0), (0, 0))


@pytest.mark.parametrize("quality", [None, 85])
def test_legacy_single_frame_representation_keeps_original_file_bytes(
    released_frames, tmp_path, quality
):
    del released_frames["wearer_frames_per_image"]
    if quality is not None:
        released_frames["wearer_jpeg_quality"] = quality
    images = observation_images(released_frames, tmp_path)
    assert len(images) == 7
    for image, path in zip(images, released_frames["frame_paths"], strict=True):
        assert base64.b64decode(image.data) == (tmp_path / path).read_bytes()
        assert image.layout[0]["position"] == "single"


def test_lower_quality_changes_only_jpeg_encoding(
    released_frames, tmp_path, packed_canvases
):
    defaults = observation_images(released_frames, tmp_path)
    compressed = observation_images(
        {**released_frames, "wearer_jpeg_quality": 85}, tmp_path
    )
    assert len(defaults) == len(compressed) == 2
    for original, changed in zip(defaults, compressed, strict=True):
        assert original.encoded_bytes != changed.encoded_bytes
        assert _decoded(original).size == _decoded(changed).size
        assert (original.layout, original.label) == (changed.layout, changed.label)
    assert [canvas.tobytes() for canvas in packed_canvases[:2]] == [
        canvas.tobytes() for canvas in packed_canvases[2:]
    ]
    assert sum(len(image.encoded_bytes) for image in compressed) < sum(
        len(image.encoded_bytes) for image in defaults
    )


def test_explicit_default_quality_preserves_old_bytes(released_frames, tmp_path):
    original = observation_images(released_frames, tmp_path)
    explicit = observation_images(
        {**released_frames, "wearer_jpeg_quality": JPEG_QUALITY}, tmp_path
    )
    assert [image.encoded_bytes for image in original] == [
        image.encoded_bytes for image in explicit
    ]


@pytest.mark.parametrize("quality", [0, 96, True, False, 85.0, "85", None])
def test_invalid_jpeg_quality_fails(quality):
    with pytest.raises(
        ValueError, match="wearer_jpeg_quality must be an integer from 1 to 95"
    ):
        observation_images(
            {
                "frame_paths": [],
                "wearer_frames_per_image": 4,
                "wearer_jpeg_quality": quality,
            },
            ".",
        )


@pytest.mark.parametrize("packing", [0, 2, 3, 5, True, 4.0, "4"])
def test_invalid_packing_fails_instead_of_dropping_frames(packing):
    with pytest.raises(ValueError, match="must be 1 or 4"):
        observation_images({"frame_paths": [], "wearer_frames_per_image": packing}, ".")


def test_mismatched_timestamps_fail(released_frames, tmp_path):
    released_frames["timestamps_seconds"] = [0]
    with pytest.raises(ValueError, match="timestamps must align"):
        observation_images(released_frames, tmp_path)


def test_empty_release_sends_no_image(tmp_path):
    assert (
        observation_images({"frame_paths": [], "wearer_frames_per_image": 4}, tmp_path)
        == []
    )


def test_size_preflight_includes_labels_but_legacy_size_is_unchanged():
    assert LABEL_HEIGHT == 24
    assert packed_image_size(640, 360, 4) == (1280, 768)
    assert packed_image_size(640, 360, 1) == (640, 360)


def test_task_image_limit_preserves_frames_labels_and_source_geometry(
    released_frames, tmp_path
):
    before = deepcopy(released_frames)
    original = observation_images(released_frames, tmp_path)
    fitted = observation_images(
        {**released_frames, "wearer_max_image_edge": 80}, tmp_path
    )

    assert released_frames == before
    assert len(original) == len(fitted) == 2
    for old, new in zip(original, fitted, strict=True):
        old_size, new_size = _decoded(old).size, _decoded(new).size
        assert max(new_size) == 80
        assert abs(new_size[0] / new_size[1] - old_size[0] / old_size[1]) < 0.02
        assert new.label == old.label
        for source, frame in zip(old.layout, new.layout, strict=True):
            assert {k: v for k, v in frame.items() if k != "bounds"} == {
                k: v for k, v in source.items() if k != "bounds"
            }
            assert frame["bounds"] == [
                round(value * new_size[index % 2] / old_size[index % 2])
                for index, value in enumerate(source["bounds"])
            ]


@pytest.mark.parametrize("packing", [1, 4])
def test_image_limit_never_upscales_or_reencodes_small_images(
    released_frames, tmp_path, packing
):
    released_frames["wearer_frames_per_image"] = packing
    original = observation_images(released_frames, tmp_path)
    fitted = observation_images(
        {**released_frames, "wearer_max_image_edge": 1024}, tmp_path
    )
    assert fitted == original


@pytest.mark.parametrize("limit", [0, -1, True, False, 1280.0, "1280"])
def test_invalid_image_limit_fails_before_rendering(limit):
    with pytest.raises(ValueError, match="wearer_max_image_edge"):
        observation_images({"frame_paths": [], "wearer_max_image_edge": limit}, ".")


def test_single_frame_downscale_preserves_source_identity(released_frames, tmp_path):
    released_frames.update(wearer_frames_per_image=1, wearer_max_image_edge=40)
    images = observation_images(released_frames, tmp_path)
    assert len(images) == len(released_frames["frame_paths"])
    assert all(max(_decoded(image).size) == 40 for image in images)
    assert [image.layout[0]["frame_index"] for image in images] == list(range(7))
    assert images[0].layout[0]["original_size"] == [80, 60]
    assert images[0].layout[0]["bounds"] == [0, 0, 40, 30]


@pytest.mark.parametrize("quality", [85, JPEG_QUALITY])
@pytest.mark.parametrize("max_edge", [None, 80])
@pytest.mark.parametrize("selected", [range(1, 2), range(3, 5), range(5, 7)])
def test_selected_frame_range_reproduces_complete_input_grids(
    released_frames, tmp_path, quality, selected, max_edge
):
    released_frames["wearer_jpeg_quality"] = quality
    if max_edge is not None:
        released_frames["wearer_max_image_edge"] = max_edge
    complete = observation_images(released_frames, tmp_path)
    partial = observation_images(released_frames, tmp_path, selected)
    expected = [
        complete[index]
        for index in sorted({frame_index // 4 for frame_index in selected})
    ]
    assert [image.encoded_bytes for image in partial] == [
        image.encoded_bytes for image in expected
    ]
    assert [(image.label, image.layout) for image in partial] == [
        (image.label, image.layout) for image in expected
    ]


def test_selected_frame_encodes_only_its_original_group(
    released_frames, tmp_path, monkeypatch
):
    released_frames["frame_paths"] += ["unselected-missing.png"] * 620
    released_frames["timestamps_seconds"] = list(
        range(len(released_frames["frame_paths"]))
    )
    grid = frame_inputs._grid
    calls = []

    def record_grid(frames, root, quality):
        calls.append([frame["frame_index"] for frame in frames])
        return grid(frames, root, quality)

    monkeypatch.setattr(frame_inputs, "_grid", record_grid)
    images = observation_images(released_frames, tmp_path, range(1, 2))
    assert len(images) == 1
    assert calls == [[0, 1, 2, 3]]
    assert [frame["frame_index"] for frame in images[0].layout] == [0, 1, 2, 3]


def test_selected_single_frame_preserves_original_index_and_bytes(
    released_frames, tmp_path
):
    released_frames["wearer_frames_per_image"] = 1
    image = observation_images(released_frames, tmp_path, range(5, 6))[0]
    assert (
        image.encoded_bytes
        == (tmp_path / released_frames["frame_paths"][5]).read_bytes()
    )
    assert image.label == "Frame 6 | source t=7.5 s"
    assert image.layout == [
        {
            "frame_path": released_frames["frame_paths"][5],
            "frame_index": 5,
            "timestamp_seconds": 7.5,
            "position": "single",
        }
    ]


@pytest.mark.parametrize("packing", [1, 4])
def test_empty_selected_range_builds_no_images(
    released_frames, tmp_path, monkeypatch, packing
):
    released_frames["wearer_frames_per_image"] = packing

    def unexpected_grid(*args):
        pytest.fail("empty range must not encode a grid")

    monkeypatch.setattr(frame_inputs, "_grid", unexpected_grid)
    assert observation_images(released_frames, tmp_path, range(3, 3)) == []
