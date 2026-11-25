from copy import deepcopy
from dataclasses import replace
import hashlib

import pytest

from egocua.run_ledger import SITE_RUNTIME_CONTRACTS
from egocua.run_materials import (
    execution_inputs,
    public_site_materials,
    same_public_site,
    site_evidence_mismatch,
    site_images_match,
    visual_mismatch,
)
from egocua.workspace import write_json_atomic
from tests.helpers import PLAN, task


def test_execution_projection_keeps_public_scope_but_not_hidden_labels():
    definition = task("graft")
    definition["truth"] = {
        "site_scope_entity_ids": ["one", "two"],
        "tempting_referent_ids": ["one"],
    }
    original = execution_inputs(definition, PLAN, "online")
    definition["truth"]["tempting_referent_ids"] = ["two"]
    definition["truth"]["site_scope_entity_ids"].reverse()
    assert execution_inputs(definition, PLAN, "online") == original
    definition["truth"]["site_scope_entity_ids"] = ["one"]
    assert execution_inputs(definition, PLAN, "online") != original


def test_unset_protocol_fields_stay_out_of_the_execution_projection():
    projected = execution_inputs(task("clarify"), PLAN, "online")
    for field in ("strategy_hint", "frame_delivery_steps", "request", "streamed"):
        assert f'"{field}"' not in projected
    hinted = replace(PLAN.condition("online"), name="online", strategy_hint=True)
    plan = replace(PLAN, conditions={**PLAN.conditions, "online": hinted})
    assert execution_inputs(task("clarify"), plan, "online") != projected


def test_empty_truth_does_not_change_execution_projection():
    definition = task("graft")
    definition.pop("truth", None)
    original = execution_inputs(definition, PLAN, "online")
    definition["truth"] = {"tempting_referent_ids": ["one"]}
    assert execution_inputs(definition, PLAN, "online") == original


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_private_answer_identity_does_not_change_execution_inputs(track):
    definition = task(track)
    original = execution_inputs(definition, PLAN, "online")
    if track == "graft":
        definition["graft"]["integration_type"] = "C"
    elif track == "refresh":
        definition["refresh"]["initial_entity_id"] = "corrected-initial-answer"
    else:
        definition["clarification"]["candidate_ids"] = ["different", "answer-ids"]

    assert execution_inputs(definition, PLAN, "online") == original


def test_clarification_answer_stays_in_execution_inputs():
    definition = task("clarify")
    original = execution_inputs(definition, PLAN, "online")
    definition["clarification"]["canonical_answer"] = "The other one."

    assert execution_inputs(definition, PLAN, "online") != original


def _sampled_definition(track):
    definition = task(track)
    for index, packet in enumerate(definition["visual_packets"]):
        start = index * 8
        packet["frequency_samples"] = {
            "window_seconds": [start, start + 1],
            "timestamps_seconds": [start, start + 1],
            "frame_paths": [f"samples/{index}/0.jpg", f"samples/{index}/1.jpg"],
        }
    return definition


@pytest.mark.parametrize("track", ["graft", "refresh", "clarify"])
def test_source_seconds_have_one_identity_for_integer_and_float_notation(track):
    definition = _sampled_definition(track)
    floating = deepcopy(definition)
    for packet in floating["visual_packets"]:
        for field in ("window_seconds", "timestamps_seconds"):
            packet["frequency_samples"][field] = list(
                map(float, packet["frequency_samples"][field])
            )
    plan = replace(PLAN, conditions={
        **PLAN.conditions,
        "online": replace(PLAN.condition("online"), frame_interval_seconds=1),
    })

    assert execution_inputs(definition, plan, "online") == execution_inputs(
        floating, plan, "online"
    )


def test_timestamp_identity_keeps_fractional_time_changes_and_literal_requests():
    definition = _sampled_definition("graft")
    plan = replace(PLAN, conditions={
        **PLAN.conditions,
        "online": replace(PLAN.condition("online"), frame_interval_seconds=1),
    })
    original = execution_inputs(definition, plan, "online")
    changed = deepcopy(definition)
    for field in ("window_seconds", "timestamps_seconds"):
        changed["visual_packets"][0]["frequency_samples"][field] = [0.25, 1.25]
    assert execution_inputs(changed, plan, "online") != original

    definition["instructions"]["a"] += " Version 1.0."
    literal = execution_inputs(definition, plan, "online")
    definition["instructions"]["a"] = definition["instructions"]["a"].replace("1.0", "1")
    assert execution_inputs(definition, plan, "online") != literal


def test_task_render_limit_changes_visual_inputs_but_not_computer_use():
    from egocua.interaction_settings import setting_conditions

    definition = task("graft")
    definition["instructions"]["computer_use"] = "Select the requested items."
    definition["visual_session"] = {
        "frequency_samples": {
            "window_seconds": [0, 23],
            "timestamps_seconds": list(range(24)),
            "frame_paths": [f"samples/{index}.jpg" for index in range(24)],
        },
        "update_onset_seconds": 16,
    }
    online = replace(
        PLAN.condition("online"), frame_interval_seconds=1,
        frame_delivery_steps=1, batch_seconds=8, ask="open",
        replay_timing="action_aligned",
    )
    plan = replace(PLAN, conditions={c.name: c for c in setting_conditions(online)})
    old = {name: execution_inputs(definition, plan, name) for name in plan.conditions}
    definition["visual_session"]["wearer_max_image_edge"] = 1024

    assert execution_inputs(definition, plan, "computer_use") == old["computer_use"]
    for name in ("offline_ego_use", "online"):
        changed = execution_inputs(definition, plan, name)
        assert changed != old[name]
        assert '"wearer_max_image_edge": 1024' in changed


@pytest.mark.parametrize("site", ["egomart", "egomedia", "egomaps", "preview"])
def test_frozen_site_comparison_includes_visible_image_content(tmp_path, site):
    path = "image.jpg"
    content = b"original image"
    digest = hashlib.sha256(content).hexdigest()
    entity = {"entity_id": "one"}
    if site == "egomart":
        entity.update(image_path=path, image_provenance={"sha256": digest})
        evidence_path = ["image_provenance"]
        digest_key = "sha256"
    elif site == "egomaps":
        entity.update(photo_path=path, source={"sha256": digest})
        evidence_path = ["source"]
        digest_key = "sha256"
    elif site == "egomedia":
        entity.update(thumbnail_path=path, source={"thumbnail_sha256": digest})
        evidence_path = ["source"]
        digest_key = "thumbnail_sha256"
    else:
        site = "egomedia"
        entity["public_previews"] = [
            {
                "image_path": path,
                "heading": "Video storyboard",
                "source_url": "https://i.ytimg.com/sb/one/storyboard3_L2/M0.jpg",
                "start_sec": 0,
                "end_sec": 30,
                "provenance": {"kind": "youtube_storyboard", "sha256": digest},
            }
        ]
        evidence_path = ["public_previews", 0, "provenance"]
        digest_key = "sha256"
    definition = {"id": "example", "site": site}
    data = {"site": site, "entities": [entity]}
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    write_json_atomic(old, data)
    write_json_atomic(new, data)
    (tmp_path / path).write_bytes(content)
    assert same_public_site((definition, definition), (str(old), str(new)))
    materials = public_site_materials(definition, data)
    assert materials["images"] == {path: digest}
    assert site_images_match(materials["images"], tmp_path)

    updated = deepcopy(data)
    evidence = updated["entities"][0]
    for key in evidence_path:
        evidence = evidence[key]
    evidence["retrieved_at"] = "later"
    write_json_atomic(new, updated)
    assert same_public_site((definition, definition), (str(old), str(new)))

    replacement = b"corrected image"
    (tmp_path / path).write_bytes(replacement)
    assert not site_images_match(materials["images"], tmp_path)
    # Historical comparison reads frozen evidence, not the editable picture.
    assert same_public_site((definition, definition), (str(old), str(new)))
    evidence[digest_key] = hashlib.sha256(replacement).hexdigest()
    write_json_atomic(new, updated)
    assert not same_public_site((definition, definition), (str(old), str(new)))
    assert site_images_match(
        public_site_materials(definition, updated)["images"], tmp_path
    )


def test_missing_image_evidence_cannot_certify_material_equality():
    definition = {"id": "example", "site": "egomart"}
    site = {
        "site": "egomart",
        "entities": [{"entity_id": "one", "image_path": "one.jpg"}],
    }
    with pytest.raises(ValueError, match="missing historical site image evidence"):
        public_site_materials(definition, site)


@pytest.mark.parametrize(
    "evidence", ["absent", "missing_file", "no_contract", "current"]
)
def test_site_reuse_requires_snapshot_and_contract_not_legacy_hash(tmp_path, evidence):
    data = {"site": "egomart", "entities": []}
    snapshot = tmp_path / "site.json"
    write_json_atomic(snapshot, data)
    runtime = {}
    record = {
        "meta": {
            "site": "egomart",
            "runtime": runtime,
            "site_data_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        }
    }
    if evidence != "absent":
        runtime["site_snapshot"] = (
            "missing.json" if evidence == "missing_file" else "site.json"
        )
    if evidence != "no_contract":
        runtime["site_contract"] = SITE_RUNTIME_CONTRACTS["egomart"]
    expected = {
        "absent": "missing_current_site_evidence",
        "missing_file": "missing_current_site_evidence",
        "no_contract": "missing_site_runtime_contract",
        "current": None,
    }
    assert site_evidence_mismatch(record, tmp_path) == expected[evidence]


@pytest.mark.parametrize(
    "evidence", ["absent", "missing_directory", "missing_frame", "changed", "current"]
)
def test_visual_reuse_requires_saved_pixels_not_legacy_hash(tmp_path, evidence):
    pixels = b"original frame"
    (tmp_path / "frame.jpg").write_bytes(pixels)
    archive = tmp_path / "visual-inputs"
    runtime = {}
    record = {
        "meta": {
            "runtime": runtime,
            "asset_sha256": {"frame.jpg": hashlib.sha256(pixels).hexdigest()},
        }
    }
    if evidence != "absent":
        runtime["visual_asset_root"] = "visual-inputs"
    if evidence not in {"absent", "missing_directory"}:
        archive.mkdir()
    if evidence in {"changed", "current"}:
        (archive / "frame.jpg").write_bytes(
            b"different pixels" if evidence == "changed" else pixels
        )
    expected = (
        "changed_visual_inputs"
        if evidence == "changed"
        else None
        if evidence == "current"
        else "missing_visual_evidence"
    )
    assert visual_mismatch(record, (["frame.jpg"], ["frame.jpg"]), tmp_path) == expected


def test_text_only_condition_needs_no_visual_archive(tmp_path):
    assert visual_mismatch({"meta": {"runtime": {}}}, ([], []), tmp_path) is None


def test_fixed_replay_compares_only_delivered_images(tmp_path):
    archive = tmp_path / "visual-inputs"
    archive.mkdir()
    (archive / "delivered.jpg").write_bytes(b"frame")
    frame = tmp_path / "current.jpg"
    frame.write_bytes(b"frame")
    record = {
        "condition": {"frame_delivery_steps": 1},
        "meta": {
            "runtime": {"visual_asset_root": str(archive)},
            "released_visual_inputs": [{"frame_paths": ["delivered.jpg"]}],
        },
    }
    paths = (["delivered.jpg", "undelivered.jpg"], ["current.jpg", "future.jpg"])
    assert visual_mismatch(record, paths, tmp_path) is None
    frame.write_bytes(b"changed")
    assert visual_mismatch(record, paths, tmp_path) == "changed_visual_inputs"


@pytest.mark.parametrize("changed", [False, True])
def test_known_public_change_precedes_missing_historical_evidence(
    tmp_path, monkeypatch, changed
):
    from egocua import run_materials

    definition = task("graft")
    old_site = {
        "site": definition["site"],
        "entities": [{"entity_id": "one", "name": "Original"}],
    }
    current_site = deepcopy(old_site)
    if changed:
        current_site["entities"][0]["name"] = "New public listing"
    write_json_atomic(tmp_path / "site.json", old_site)
    monkeypatch.setattr(run_materials, "load_site_data", lambda *_: current_site)
    record = {
        "task_id": definition["id"],
        "task_snapshot": deepcopy(definition),
        "condition": {
            "name": "online",
            "schedule": "online",
            "scope": "full",
            "ask": "allowed",
            "release_at": None,
        },
        "meta": {"site": definition["site"], "runtime": {"site_snapshot": "site.json"}},
    }

    result = run_materials.material_mismatch(definition, record, PLAN, root=tmp_path)

    assert result == ("changed_public_site" if changed else "missing_visual_evidence")
