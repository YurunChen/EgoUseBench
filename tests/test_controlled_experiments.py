from __future__ import annotations

import json
import hashlib
import asyncio
import signal
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from egocua.controlled_experiments import (
    ControlledArm,
    build_variable_arm,
    execution_schedule,
    export_study,
    information_probe_arm,
    read_study,
    load_formal_study,
    load_frozen_study,
    online_arm,
    repeated_controlled_study,
    study_matrix,
)
from egocua.study_execution import (
    run_exported_evaluation,
    EvaluationSettings,
)
from egocua.study_cli import main as study_cli
from egocua.study_summary import (
    summarize_records, unresolved_request_failures, _request_failure_summary,
    repetition_statistics,
)
from egocua.cli import main as run_cli
from egocua.checker import check_run
from egocua.environment import site_snapshot
from egocua.execution import RunConfig, run_experiment
from egocua.experiment import expand_runs, load_experiment_plan
from egocua.run_ledger import load_run_record
from egocua.schema import load_tasks
from egocua.timeline import Timeline
from egocua.timeline import Condition
from tests.helpers import PLAN, item_id, materialize_assets, materialize_site, products, task
from egocua.workspace import read_json_strict, write_json_atomic


def _write_study_fixture(directory, definition):
    progress_keys = {"execution_results", "execution_stop", "run_purpose"}
    write_json_atomic(
        directory / "plan.json",
        {key: value for key, value in definition.items() if key not in progress_keys},
    )
    write_json_atomic(
        directory / "results.json",
        {key: value for key, value in definition.items() if key in progress_keys},
    )


@pytest.fixture
def reusable_study(tmp_path, source_groups):
    definitions = [task("graft"), deepcopy(task("graft"))]
    definitions[1]["id"] += "-peer"
    for index, definition in enumerate(definitions):
        ids = [item_id("graft", index * 2 + offset) for offset in (0, 1)]
        definition["truth"] = {"site_scope_entity_ids": ids}
        definition["goal"]["acceptable_states"][0]["selected_entity_ids"] = ids[:1]
        for state, entity_id in zip(
            definition["graft"]["initial_goal"]["acceptable_states"], ids, strict=True
        ):
            state["selected_entity_ids"] = [entity_id]
    materialize_site(products("graft"), tmp_path)
    for path in [
        *(item["image_path"] for item in products("graft")),
        *definitions[0]["visual_packets"][0]["frame_paths"],
    ]:
        asset = tmp_path / path
        asset.parent.mkdir(parents=True, exist_ok=True)
        content = f"asset:{asset.stem}" if path.startswith("products/") else path
        asset.write_bytes(content.encode())
    plan = replace(
        PLAN,
        experiment_id="old-study",
        replicate_ids=(0,),
        track_conditions={"graft": ("online",)},
        track_probe_conditions={},
        conditions={"online": PLAN.condition("online")},
    )
    old = tmp_path / "old"
    export_study(
        [ControlledArm("formal", definitions, plan, "formal")], old, root=tmp_path
    )
    arms, _, _ = load_frozen_study(old)
    arm = arms[0]
    study = read_study(old)
    outcomes = []
    for entry in study["execution_schedule"]:
        run = entry["run"]
        definition = next(task for task in arm.tasks if task["id"] == run["task_id"])
        config = RunConfig(
            agent=plan.adapter,
            model=plan.model,
            model_version=plan.model_version,
            max_steps=plan.max_steps,
            default_release_step=plan.default_release_step,
            reference_steps=plan.reference_steps,
            provider_seed=run["provider_seed"],
            replicate_id=run["replicate_id"],
            schedule_index=run["schedule_index"],
            asset_root=tmp_path,
            output_root=tmp_path / "artifacts",
            site_snapshot=arm.plan.site_snapshots["egomart"],
            experiment_id=plan.experiment_id,
        )

        def execute(context, definition=definition):
            for observation in context.initial_payload["observations"]:
                context.observation_images(observation)
            context.advance()
            context.advance()
            context._state.select(
                definition["truth"]["site_scope_entity_ids"][
                    int(definition["id"].endswith("peer"))
                ]
            )
            return []

        record = run_experiment(definition, plan.condition("online"), config, execute)
        outcomes.append(
            {
                "execution_index": entry["execution_index"],
                "run": run,
                "valid": True,
                "artifact_directory": record["meta"]["artifact_directory"],
                "run_id": record["meta"]["run_id"],
            }
        )
    study["execution_results"] = outcomes
    _write_study_fixture(old, study)
    return old, [
        replace(
            arm,
            tasks=sorted(arm.tasks, key=lambda task: task["id"]),
            plan=replace(
                arm.plan, experiment_id="new-study", frozen_asset_root=None,
                site_snapshots={},
            ),
        )
    ]


def test_export_reuses_success_and_failure_without_changing_old_archives(
    reusable_study,
    tmp_path,
):
    old, arms = reusable_study
    original = (old / "plan.json").read_bytes()
    new = tmp_path / "new"
    export_study(arms, new, root=tmp_path, reuse_from=old)
    outcomes = read_study(new)["execution_results"]
    assert len(outcomes) == 2
    assert {outcome["success"] for outcome in outcomes} == {True, False}
    assert (old / "plan.json").read_bytes() == original
    frozen, _, _ = load_frozen_study(new)
    snapshot = Path(frozen[0].plan.site_snapshots["egomart"])
    assert snapshot.parent == new / "inputs/site-snapshots"
    assert snapshot.is_file()
    assert not (new / "sites").exists()
    assert not list(tmp_path.glob(".new-*"))
    records = [
        (
            outcome["run"]["experiment_id"],
            load_run_record(outcome["artifact_directory"]),
        )
        for outcome in outcomes
    ]
    report = summarize_records(
        frozen, records, root=tmp_path, execution_results=outcomes
    )
    assert (report["physical_runs"], report["observed_runs"]) == (2, 2)


def test_retained_site_reuses_exact_task_input_without_rolling_back_catalog(reusable_study, tmp_path):
    old, arms = reusable_study
    task, peer = arms[0].tasks
    path = tmp_path / "sites/egomart/products.json"
    site = read_json_strict(path)
    site["products"][1]["attributes"]["capacity"] = "5 L"
    write_json_atomic(path, site)

    destination = tmp_path / "mixed"
    export_study(
        arms, destination, root=tmp_path, reuse_from=old, retain_site_for=(task["id"],)
    )
    [arm], _, _ = load_frozen_study(destination)
    [old_arm], _, _ = load_frozen_study(old)
    assert read_json_strict(Path(arm.plan.site_snapshot_for(task))) == read_json_strict(
        Path(old_arm.plan.site_snapshot_for(task))
    )
    assert arm.plan.site_snapshot_for(peer) != arm.plan.site_snapshot_for(task)
    outcomes = read_study(destination)["execution_results"]
    assert {row["run"]["task_id"] for row in outcomes} == {task["id"], peer["id"]}
    report = summarize_records(
        [arm],
        [(row["run"]["experiment_id"], load_run_record(row["artifact_directory"])) for row in outcomes],
        root=tmp_path, execution_results=outcomes,
    )
    assert report["observed_runs"] == 2
    assert read_json_strict(path)["products"][1]["attributes"]["capacity"] == "5 L"


def test_retaining_site_cannot_reuse_a_changed_task_request(reusable_study, tmp_path):
    old, arms = reusable_study
    task = arms[0].tasks[0]
    task["instructions"]["b"] += " extra request"
    destination = tmp_path / "rejected"
    with pytest.raises(ValueError, match="retained site did not restore"):
        export_study(
            arms, destination, root=tmp_path, reuse_from=old, retain_site_for=(task["id"],)
        )
    assert not destination.exists()


@pytest.mark.parametrize(
    "change",
    [
        "unrelated_entity",
        "provenance",
        "source",
        "frame_path",
        "unused_text",
        "hidden_labels",
        "scope_order",
    ],
)
def test_non_input_changes_keep_valid_runs(reusable_study, tmp_path, change):
    old, arms = reusable_study
    if change in {"unrelated_entity", "provenance"}:
        site = read_json_strict(tmp_path / "sites/egomart/products.json")
        site["products"][8]["name"] = "Unrelated changed listing"
        if change == "provenance":
            site["products"][0]["image_provenance"]["retrieved_at"] = "later"
        write_json_atomic(tmp_path / "sites/egomart/products.json", site)
    elif change == "source":
        arms[0].tasks[0]["source"]["new_audit_note"] = "metadata only"
    elif change == "unused_text":
        arms[0].tasks[0]["instructions"]["bundled"] = "not used in online input"
    elif change == "hidden_labels":
        definition = arms[0].tasks[0]
        definition["truth"]["tempting_referent_ids"] = [
            definition["truth"]["site_scope_entity_ids"][1]
        ]
    elif change == "scope_order":
        arms[0].tasks[0]["truth"]["site_scope_entity_ids"].reverse()
    else:
        packet = arms[0].tasks[0]["visual_packets"][0]
        original = packet["frame_paths"][0]
        (tmp_path / "moved.jpg").write_bytes((tmp_path / original).read_bytes())
        packet["frame_paths"][0] = "moved.jpg"
    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    assert len(read_study(tmp_path / "new")["execution_results"]) == 2


@pytest.mark.parametrize(
    "change", ["candidate", "instruction", "variant", "image", "scope"]
)
def test_changed_task_only_retests_its_own_execution(reusable_study, tmp_path, change):
    old, arms = reusable_study
    changed = arms[0].tasks[0]
    if change == "candidate":
        site = read_json_strict(tmp_path / "sites/egomart/products.json")
        site["products"][1]["attributes"]["variant"] = "different fit"
        write_json_atomic(tmp_path / "sites/egomart/products.json", site)
    elif change == "instruction":
        changed["instructions"]["b"] += " extra request"
    elif change == "image":
        (tmp_path / "different.jpg").write_bytes(b"different image content")
        changed["visual_packets"][0]["frame_paths"][0] = "different.jpg"
    elif change == "scope":
        changed["truth"]["site_scope_entity_ids"].append(item_id("graft", 4))
    else:
        changed["id"] = "test-graft-variant"
    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    outcomes = read_study(tmp_path / "new")["execution_results"]
    assert len(outcomes) == 1
    assert outcomes[0]["run"]["task_id"] == "test-graft-peer"


def test_invalid_outcome_is_not_reused(reusable_study, tmp_path):
    old, arms = reusable_study
    study = read_study(old)
    study["execution_results"][0]["valid"] = False
    _write_study_fixture(old, study)
    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    assert len(read_study(tmp_path / "new")["execution_results"]) == 1


def test_overwritten_frames_do_not_reuse_results(reusable_study, tmp_path):
    old, arms = reusable_study
    original = (old / "plan.json").read_bytes()
    frame = arms[0].tasks[0]["visual_packets"][0]["frame_paths"][0]
    (tmp_path / frame).write_bytes(b"replacement pixels at the same path")
    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    assert read_study(tmp_path / "new")["execution_results"] == []
    assert (old / "plan.json").read_bytes() == original


@pytest.mark.parametrize("product_index", [0, 8])
def test_image_replacement_only_invalidates_visible_tasks(
    reusable_study, tmp_path, product_index
):
    old, arms = reusable_study
    catalog_path = tmp_path / "sites/egomart/products.json"
    catalog = read_json_strict(catalog_path)
    product = catalog["products"][product_index]
    content = b"corrected product image at the same path"
    (tmp_path / product["image_path"]).write_bytes(content)
    product["image_provenance"]["sha256"] = hashlib.sha256(content).hexdigest()
    write_json_atomic(catalog_path, catalog)

    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    outcomes = read_study(tmp_path / "new")["execution_results"]
    expected = (
        {"test-graft-peer"} if product_index == 0 else {"test-graft", "test-graft-peer"}
    )
    assert {row["run"]["task_id"] for row in outcomes} == expected


def test_missing_historical_pixels_do_not_certify_a_new_study(reusable_study, tmp_path):
    old, arms = reusable_study
    study = read_study(old)
    for outcome in study["execution_results"]:
        path = Path(outcome["artifact_directory"]) / "run.json"
        record = read_json_strict(path)
        record["meta"]["runtime"].pop("visual_asset_root")
        write_json_atomic(path, record)
    with pytest.raises(ValueError, match="missing historical visual evidence"):
        export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".new-*"))
    assert len(read_study(old)["execution_results"]) == 2
    records = [
        (
            outcome["run"]["experiment_id"],
            load_run_record(outcome["artifact_directory"]),
        )
        for outcome in study["execution_results"]
    ]
    frozen, _, _ = load_frozen_study(old)
    assert (
        summarize_records(
            frozen, records, root=tmp_path, execution_results=study["execution_results"]
        )["observed_runs"]
        == 2
    )


def test_missing_site_evidence_blocks_new_reuse_not_historical_results(
    reusable_study, tmp_path
):
    old, arms = reusable_study
    study = read_study(old)
    outcome = study["execution_results"][0]
    path = Path(outcome["artifact_directory"]) / "run.json"
    record = read_json_strict(path)
    snapshot = load_run_record(path.parent)["task_snapshot"]
    historical_check = check_run(snapshot, record)
    record["meta"]["runtime"].pop("site_snapshot")
    write_json_atomic(path, record)
    original = path.read_bytes()

    with pytest.raises(ValueError, match="missing historical website evidence"):
        export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    assert not (tmp_path / "new").exists()
    assert not list(tmp_path.glob(".new-*"))
    assert path.read_bytes() == original
    assert len(read_study(old)["execution_results"]) == 2
    checked = check_run(snapshot, load_run_record(path.parent))
    assert checked["run_valid"]
    assert checked == historical_check


def test_unrecorded_site_contract_does_not_block_other_reuse(reusable_study, tmp_path):
    old, arms = reusable_study
    study = read_study(old)
    outcome = study["execution_results"][0]
    path = Path(outcome["artifact_directory"]) / "run.json"
    record = read_json_strict(path)
    record["meta"]["runtime"].pop("site_contract")
    write_json_atomic(path, record)
    original = path.read_bytes()

    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)

    reused = read_study(tmp_path / "new")["execution_results"]
    assert [row["run"]["task_id"] for row in reused] == ["test-graft-peer"]
    assert path.read_bytes() == original
    assert len(read_study(old)["execution_results"]) == 2
    assert check_run(
        load_run_record(path.parent)["task_snapshot"], load_run_record(path.parent)
    )["run_valid"]


def test_failed_export_leaves_no_partial_bundle_and_can_retry(
    reusable_study, tmp_path, monkeypatch
):
    old, arms = reusable_study
    destination = tmp_path / "new"
    original = (old / "plan.json").read_bytes()

    def refuse_reuse(*args):
        assert not destination.exists()
        raise ValueError("reuse preflight failed")

    with monkeypatch.context() as patcher:
        patcher.setattr("egocua.controlled_experiments._reusable_results", refuse_reuse)
        with pytest.raises(ValueError, match="reuse preflight failed"):
            export_study(arms, destination, root=tmp_path, reuse_from=old)
    assert not destination.exists()
    assert not list(tmp_path.glob(".new-*"))
    assert (old / "plan.json").read_bytes() == original

    export_study(arms, destination, root=tmp_path, reuse_from=old)
    frozen, _, _ = load_frozen_study(destination)
    assert Path(frozen[0].plan.site_snapshots["egomart"]).parent == (
        destination / "inputs/site-snapshots"
    )
    assert len(read_study(destination)["execution_results"]) == 2


def test_different_packed_quality_requires_a_separate_execution():
    arm = online_arm(
        "packed",
        [task("graft")],
        replace(
            PLAN,
            conditions={
                name: replace(condition, wearer_frames_per_image=4)
                for name, condition in PLAN.conditions.items()
            },
        ),
    )
    compressed = replace(
        arm,
        name="compressed",
        plan=replace(
            arm.plan,
            conditions={
                name: replace(condition, wearer_jpeg_quality=85)
                for name, condition in arm.plan.conditions.items()
            },
        ),
    )
    assert len(execution_schedule([arm, compressed])) == 2


def test_key_change_that_exposes_new_site_controls_requires_new_execution(
    reusable_study, tmp_path, monkeypatch
):
    old, arms = reusable_study
    frozen, _, _ = load_frozen_study(old)
    snapshot_path = Path(frozen[0].plan.site_snapshots["egomart"])
    site = read_json_strict(snapshot_path)
    site["supported_fields"] = ["mode"]
    monkeypatch.setattr(
        "egocua.controlled_experiments.load_site_data",
        lambda *args, **kwargs: deepcopy(site),
    )
    arms[0].tasks[0]["goal"]["acceptable_states"][0]["fields"] = {"mode": "walking"}
    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    outcomes = read_study(tmp_path / "new")["execution_results"]
    assert len(outcomes) == 1
    assert outcomes[0]["run"]["task_id"] == "test-graft-peer"


@pytest.mark.parametrize("source_change", ["none", "replace", "delete"])
def test_reused_study_runner_does_not_call_model(
    reusable_study, tmp_path, monkeypatch, source_change
):
    old, arms = reusable_study
    destination = tmp_path / "new"
    export_study(arms, destination, root=tmp_path, reuse_from=old)
    frame = tmp_path / arms[0].tasks[0]["visual_packets"][0]["frame_paths"][0]
    if source_change == "replace":
        frame.write_bytes(b"current material differs from historical run")
    elif source_change == "delete":
        frame.unlink()

    async def unexpected_model(*args, **kwargs):
        pytest.fail("unchanged completed executions must not call the model")

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", unexpected_model)
    summary = asyncio.run(
        run_exported_evaluation(destination, EvaluationSettings(), tmp_path)
    )
    assert summary["remaining_executions"] == 0
    assert summary["reused_valid_executions"] == 2
    assert summary["execution_stop"] is None


@pytest.mark.parametrize("key_field", ["goal", "initial_goal"])
def test_hidden_key_changes_reuse_traces_and_preserve_historical_scores(
    reusable_study, tmp_path, monkeypatch, key_field
):
    old, arms = reusable_study
    old_arms, _, _ = load_frozen_study(old)
    old_outcomes = read_study(old)["execution_results"]
    records = [
        (row["run"]["experiment_id"], load_run_record(row["artifact_directory"]))
        for row in old_outcomes
    ]
    old_report = summarize_records(old_arms, records, root=tmp_path)
    old_report["execution_results"] = old_outcomes
    write_json_atomic(old / "results.json", old_report)
    old_files = [old / "plan.json", old / "results.json"]
    for outcome in old_outcomes:
        old_files.extend(Path(outcome["artifact_directory"]).glob("*"))
    original = {path: path.read_bytes() for path in old_files if path.is_file()}
    peer = next(task for task in arms[0].tasks if task["id"].endswith("peer"))
    peer_record = next(
        record for _, record in records if record["task_id"] == peer["id"]
    )
    assert check_run(peer, peer_record)["e2e_success"] is False
    if key_field == "goal":
        peer["goal"]["acceptable_states"][0]["selected_entity_ids"] = peer_record[
            "live_state"
        ]["selected_entity_ids"]
    else:
        peer["graft"]["initial_goal"]["acceptable_states"] = [
            deepcopy(peer_record["live_state"])
        ]
    plan = arms[0].plan
    plan = replace(
        plan,
        conditions={
            "online": plan.condition("online"),
            "same_input": replace(plan.condition("online"), name="same_input"),
        },
        track_conditions={"graft": ("online", "same_input")},
    )
    arms = [replace(arms[0], name="visual", variable="visual", plan=plan)]
    destination = tmp_path / "rescored"
    export_study(arms, destination, root=tmp_path, reuse_from=old)
    outcomes = read_study(destination)["execution_results"]
    assert len(outcomes) == 2

    async def unexpected_model(*args, **kwargs):
        pytest.fail("hidden-key changes must only rescore the saved trace")

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", unexpected_model)
    summary = asyncio.run(
        run_exported_evaluation(destination, EvaluationSettings(), tmp_path)
    )
    assert summary["reused_valid_executions"] == 2
    assert summary["remaining_executions"] == 0
    report = read_json_strict(destination / "results.json")
    peer_rows = [row for row in report["runs"] if row["task_id"] == peer["id"]]
    assert len(peer_rows) == 2
    assert all(row["e2e_success"] is (key_field == "goal") for row in peer_rows)
    assert all(row["scoring"]["historical_success"] is False for row in peer_rows)
    assert all(
        row["scoring"]["task_source"] == "current_frozen_task" for row in peer_rows
    )
    assert peer_rows[0]["evaluator"] == peer_rows[1]["evaluator"]
    assert set(peer_rows[0]["evaluator"]) == {"version", "predicate"}
    frozen, _, _ = load_frozen_study(destination)
    frozen_peer = next(t for t in frozen[0].tasks if t["id"] == peer["id"])
    assert frozen_peer["goal"]["acceptable_states"] == peer["goal"]["acceptable_states"]
    assert report["comparisons"][0]["difference"] == 0
    assert check_run(peer, peer_record)["e2e_success"] is False
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("change", ["model", "budget", "seed", "condition"])
def test_different_execution_protocol_does_not_reuse_runs(
    reusable_study, tmp_path, change
):
    old, arms = reusable_study
    plan = arms[0].plan
    if change == "model":
        plan = replace(plan, model="another-model", model_version="another-model")
    elif change == "budget":
        plan = replace(plan, max_steps=plan.max_steps + 1)
    elif change == "seed":
        plan = replace(plan, scheduler_seed=plan.scheduler_seed + 1)
    else:
        plan = replace(
            plan,
            conditions={
                "online": replace(plan.condition("online"), scope="first_frame")
            },
        )
    export_study(
        [replace(arms[0], plan=plan)], tmp_path / "new", root=tmp_path, reuse_from=old
    )
    assert read_study(tmp_path / "new")["execution_results"] == []


def test_diagnostic_only_archive_cannot_become_a_formal_score(
    reusable_study, tmp_path
):
    old, arms = reusable_study
    plan = read_json_strict(old / "plan.json")
    plan["review_gate"] = {**(plan.get("review_gate") or {}), "diagnostic_only": True}
    write_json_atomic(old / "plan.json", plan)
    progress = (old / "results.json").read_bytes()

    export_study(arms, tmp_path / "formal", root=tmp_path, reuse_from=old)

    assert read_study(tmp_path / "formal")["execution_results"] == []
    assert (old / "results.json").read_bytes() == progress


def test_invalid_archive_marked_valid_is_not_reused(
    reusable_study, tmp_path, monkeypatch
):
    old, arms = reusable_study
    from egocua.controlled_experiments import check_run as original_check

    def reject_invalid_archive(definition, record):
        checked = original_check(definition, record)
        if definition["id"] == "test-graft":
            checked["run_valid"] = False
        return checked

    monkeypatch.setattr(
        "egocua.controlled_experiments.check_run", reject_invalid_archive
    )
    export_study(arms, tmp_path / "new", root=tmp_path, reuse_from=old)
    outcomes = read_study(tmp_path / "new")["execution_results"]
    assert len(outcomes) == 1
    assert outcomes[0]["run"]["task_id"] == "test-graft-peer"


def test_changed_inputs_cannot_be_attached_with_old_reuse_reference(
    reusable_study, tmp_path
):
    old, arms = reusable_study
    destination = tmp_path / "new"
    export_study(arms, destination, root=tmp_path, reuse_from=old)
    outcomes = read_study(destination)["execution_results"]
    frozen, _, _ = load_frozen_study(destination)
    records = [
        (row["run"]["experiment_id"], load_run_record(row["artifact_directory"]))
        for row in outcomes
    ]
    frozen[0].tasks[0]["instructions"]["b"] = "changed after reuse"
    with pytest.raises(ValueError, match="content differs from registered inputs"):
        summarize_records(frozen, records, root=tmp_path, execution_results=outcomes)


def test_public_site_change_after_reuse_is_rejected(reusable_study, tmp_path):
    old, arms = reusable_study
    destination = tmp_path / "new"
    export_study(arms, destination, root=tmp_path, reuse_from=old)
    outcomes = read_study(destination)["execution_results"]
    frozen, _, _ = load_frozen_study(destination)
    records = [
        (row["run"]["experiment_id"], load_run_record(row["artifact_directory"]))
        for row in outcomes
    ]
    summarize_records(frozen, records, root=tmp_path, execution_results=outcomes)
    site = read_json_strict(Path(frozen[0].plan.site_snapshots["egomart"]))
    site["entities"][0]["name"] = "new public specification"
    changed = str(site_snapshot(tmp_path, "egomart", site))
    frozen = [
        replace(arm, plan=replace(arm.plan, site_snapshots={"egomart": changed}))
        for arm in frozen
    ]
    with pytest.raises(ValueError, match="public site content differs"):
        summarize_records(frozen, records, root=tmp_path, execution_results=outcomes)


def test_formal_study_uses_only_members_and_existing_status(monkeypatch, tmp_path):
    definitions = [task(track) for track in ("graft", "refresh", "clarify")]
    members = [definition["id"] for definition in definitions[:2]]
    calls = []
    module = "egocua.controlled_experiments"
    monkeypatch.setattr(f"{module}.load_selection", lambda root: {"task_set": {}})
    monkeypatch.setattr(f"{module}.task_ids", lambda selection: members)
    monkeypatch.setattr(f"{module}.load_tasks", lambda path: definitions)
    monkeypatch.setattr(
        f"{module}.compute_task_statuses",
        lambda root, ids: calls.append(ids) or dict.fromkeys(ids, "ready"),
    )
    monkeypatch.setattr(f"{module}.load_experiment_plan", lambda path: PLAN)
    (tmp_path / "plan.json").write_text("{}")

    arms, evidence, gate = load_formal_study(tmp_path, "plan.json")

    assert calls == [members]
    assert evidence is None
    assert gate["formal_task_statuses"] == dict.fromkeys(members, "ready")
    assert [definition["id"] for definition in arms[0].tasks] == members
    assert len(execution_schedule(arms)) == len(members)
    assert arms[0].plan.track_conditions == {
        "graft": ("online",),
        "refresh": ("online",),
    }
    assert arms[0].plan.track_probe_conditions == {}
    assert arms[0].plan.condition("online") == PLAN.condition("online")


def test_registered_probes_pair_with_online_without_repeating_it(monkeypatch, tmp_path):
    from egocua.study_summary import _comparison_pairs

    definitions = [task(track) for track in ("graft", "refresh", "clarify")]
    members = [definition["id"] for definition in definitions]
    plan = replace(
        PLAN,
        conditions={
            "online": PLAN.condition("online"),
            "text_only": Condition("text_only", "online", "none", "open", None),
        },
        track_conditions={track: ("online",) for track in ("graft", "refresh", "clarify")},
        track_probe_conditions={
            track: ("text_only",) for track in ("graft", "refresh", "clarify")
        },
    )
    module = "egocua.controlled_experiments"
    monkeypatch.setattr(f"{module}.load_selection", lambda root: {"task_set": {}})
    monkeypatch.setattr(f"{module}.task_ids", lambda selection: members)
    monkeypatch.setattr(f"{module}.load_tasks", lambda path: definitions)
    monkeypatch.setattr(
        f"{module}.compute_task_statuses", lambda root, ids: dict.fromkeys(ids, "ready")
    )
    monkeypatch.setattr(f"{module}.load_experiment_plan", lambda path: plan)

    (tmp_path / "plan.json").write_text("{}")
    assert [arm.name for arm in load_formal_study(tmp_path, "plan.json")[0]] == [
        "formal"
    ]
    arms, _, _ = load_formal_study(tmp_path, "plan.json", with_probes=True)

    assert [arm.name for arm in arms] == ["formal", "visual"]
    schedule = execution_schedule(arms)
    assert len(schedule) == 2 * len(members)
    assert sum(len(entry["analysis_references"]) == 2 for entry in schedule) == len(
        members
    )
    assert [pair[:2] for pair in _comparison_pairs(arms)] == [
        ("visual::text_only", "visual::online")
    ]


def test_formal_summary_counts_failed_physical_runs_in_both_references(
    source_groups, tmp_path
):
    tasks = [task(track) for track in ("graft", "refresh", "clarify")]
    plan = replace(
        PLAN,
        track_probe_conditions={
            "graft": (), "refresh": (), "clarify": ("no_ask",)
        },
    )
    arms = [online_arm("formal", tasks, plan), information_probe_arm(tasks, plan)]
    entry = next(
        row for row in execution_schedule(arms)
        if row["run"]["track"] == "clarify" and row["run"]["condition"] == "online"
    )
    report = summarize_records(
        arms, [], root=tmp_path,
        execution_results=[{"execution_index": entry["execution_index"], "valid": False}],
    )
    assert report["formal_success"]["clarify"]["invalid_tasks"] == 1
    assert report["formal_success"]["overall"]["success_bounds"] == [0, 1]
    assert report["formal_success"]["overall"]["unobserved_tasks"] == 2
    assert report["rule_baselines"]["clarify"]["keep_initial"]["not_applicable_tasks"] == 1
    comparison, = report["comparisons"]
    assert (comparison["left"], comparison["right"]) == ("visual::no_ask", "visual::online")
    assert comparison["coverage"]["expected_tasks"] == 1
    assert comparison["coverage"]["invalid_runs"] == 1
    assert comparison["coverage"]["paired_replicates"] == 0


@pytest.mark.parametrize("status", ["needs_repair", "needs_review"])
def test_formal_study_does_not_silently_drop_unready_members(monkeypatch, status):
    module = "egocua.controlled_experiments"
    monkeypatch.setattr(f"{module}.load_selection", lambda root: {})
    monkeypatch.setattr(f"{module}.task_ids", lambda selection: ["pending"])
    monkeypatch.setattr(f"{module}.load_tasks", lambda path: [])
    monkeypatch.setattr(
        f"{module}.compute_task_statuses", lambda root, ids: {"pending": status}
    )

    with pytest.raises(ValueError, match=f"pending.*{status}"):
        load_formal_study(Path.cwd(), "unused.json")


def test_main_benchmark_config_has_only_one_regular_trial_per_member():
    plan = load_experiment_plan(Path(__file__).parents[1] / "configs/benchmark.json")
    definitions = [task(track) for track in ("graft", "refresh", "clarify")]
    assert len(expand_runs(definitions, plan)) == 3
    assert plan.adapter == "browsergym"
    assert plan.model == "qwen3.5-flash"
    assert plan.max_steps == 20
    assert plan.release_trigger == "timeline_step"
    assert plan.condition("online").schedule == "online"
    assert plan.condition("online").scope == "full"
    assert plan.condition("online").ask == "open"
    assert plan.condition("online").wearer_frames_per_image == 4
    assert plan.condition("online").wearer_jpeg_quality == 85
    assert plan.condition("online").frame_delivery_steps == 1
    assert plan.condition("online").batch_seconds == 8


def test_benchmark_probes_share_the_online_replay_protocol():
    plan = load_experiment_plan(Path(__file__).parents[1] / "configs/benchmark.json")
    online = plan.condition("online")
    assert plan.probe_conditions_for("graft") == ("text_only", "upfront", "initial_only")
    assert plan.probe_conditions_for("refresh") == plan.probe_conditions_for("graft")
    assert plan.probe_conditions_for("clarify") == ("text_only", "upfront", "no_ask")
    assert plan.condition("upfront") == replace(online, name="upfront", schedule="upfront")
    assert plan.condition("initial_only") == replace(
        online, name="initial_only", scope="initial_only"
    )
    assert plan.condition("no_ask") == replace(online, name="no_ask", ask="denied")


def test_formal_cli_preparation_does_not_load_research_or_call_models(
    monkeypatch, tmp_path, capsys
):
    formal = [online_arm("formal", [task("refresh")], PLAN)]
    module = "egocua.study_cli"
    monkeypatch.setattr(
        f"{module}.load_formal_study", lambda root, path, **kwargs: (formal, None, {})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("formal preparation must not run a model")

    monkeypatch.setattr(f"{module}.run_exported_evaluation", forbidden)
    monkeypatch.setattr(f"{module}.study_matrix", lambda *a, **kw: [])
    monkeypatch.setattr(
        sys, "argv", ["study", "--formal-plan", "configs/benchmark.json"]
    )

    study_cli()

    report = json.loads(capsys.readouterr().out)
    assert report["paid_runs_started"] == 0
    assert report["unique_executions"] == 1
    assert set(report["variables"]) == {"formal"}


def test_identical_inputs_share_one_physical_execution():
    definition = task("refresh")
    first = ControlledArm("first", [definition], PLAN)
    second = ControlledArm("second", [definition], replace(PLAN, experiment_id="other"))
    schedule = execution_schedule([first, second])
    assert len(schedule) == len(first.schedule())
    assert all(len(run["analysis_references"]) == 2 for run in schedule)


@pytest.mark.parametrize(
    "change", ["trigger", "budget", "frame", "instruction", "model", "packing"]
)
def test_shared_task_id_does_not_merge_different_execution_inputs(change):
    definition = task("refresh")
    changed = deepcopy(definition)
    plan = PLAN
    if change == "trigger":
        plan = replace(plan, release_trigger="first_selection")
    elif change == "budget":
        plan = replace(plan, max_steps=plan.max_steps + 1)
    elif change == "frame":
        changed["visual_packets"][0]["frame_paths"][0] = "other.png"
    elif change == "instruction":
        changed["instructions"]["base"] += " with another requirement"
    elif change == "model":
        plan = replace(plan, model="another-model")
    else:
        plan = replace(
            plan,
            conditions={
                name: replace(condition, wearer_frames_per_image=4)
                for name, condition in plan.conditions.items()
            },
        )
    first = ControlledArm("first", [definition], PLAN)
    second = ControlledArm("second", [changed], plan)
    assert len(execution_schedule([first, second])) == len(first.schedule()) + len(
        second.schedule()
    )


def test_same_task_id_requires_one_frozen_scoring_key():
    definition = task("refresh")
    changed = deepcopy(definition)
    changed["goal"]["acceptable_states"][0]["selected_entity_ids"] = [
        "different-answer"
    ]
    first = ControlledArm("first", [definition], PLAN)
    second = ControlledArm("second", [changed], replace(PLAN, experiment_id="other"))
    with pytest.raises(ValueError, match="inconsistent frozen scoring key"):
        execution_schedule([first, second])


def test_shared_result_is_referenced_without_rewriting_record(
    arms, source_groups, tmp_path
):
    first = arms[0]
    second = replace(
        first, name="other", plan=replace(first.plan, experiment_id="other")
    )
    records = _visual_records([first], tmp_path)
    original = deepcopy(records)
    report = summarize_records([first, second], records, root=tmp_path)
    assert report["observed_runs"] == 2 * report["physical_runs"]
    assert records == original
    assert {row["analysis_reference"]["arm"] for row in report["runs"]} == {
        "visual",
        "other",
    }


def test_duplicate_physical_result_is_not_counted_twice(arms, source_groups, tmp_path):
    records = _visual_records(arms, tmp_path)
    with pytest.raises(ValueError, match="duplicate physical execution"):
        summarize_records(arms, [*records, records[0]], root=tmp_path)


@pytest.fixture
def evaluation_export(tmp_path, monkeypatch):
    from egocua.controlled_experiments import online_arm
    from egocua.experiment import load_experiment_plan

    definitions = [task(track) for track in ("graft", "refresh", "clarify")]
    for definition in definitions:
        materialize_assets(definition, products(definition["track"]), tmp_path)
        frame = definition["visual_packets"][0]["frame_paths"][0]
        definition["visual_session"] = {
            "update_onset_seconds": 8,
            "frequency_samples": {
                "window_seconds": [0, 15],
                "timestamps_seconds": list(range(16)),
                "frame_paths": [frame] * 16,
                "sources": [],
            },
        }
    plan = load_experiment_plan(Path("configs/benchmark.json"))
    destination = tmp_path / "export"
    export_study([online_arm("formal", definitions, plan)], destination, root=tmp_path)
    monkeypatch.setattr(
        "egocua.study_execution.check_run",
        lambda *args: {"run_valid": True, "e2e_success": False},
    )
    monkeypatch.setattr(
        "egocua.study_execution.summarize_records",
        lambda arms, records, **kwargs: {"physical_runs": len(records)},
    )
    return destination


@pytest.mark.parametrize("concurrency", [10, 20, 24])
def test_full_runner_continues_after_valid_model_failure(
    evaluation_export, tmp_path, monkeypatch, concurrency
):
    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        await asyncio.sleep(0)
        return {
            "meta": {
                "artifact_directory": str(tmp_path / definition["id"]),
                "run_id": definition["id"],
            }
        }

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(
            evaluation_export, EvaluationSettings(concurrency=concurrency), tmp_path
        )
    )
    assert summary["completed_valid_executions"] == 3
    assert len(calls) == len(set(calls)) == 3
    assert summary["execution_stop"] is None


@pytest.mark.parametrize("concurrency", [0, 25])
def test_runner_rejects_out_of_range_concurrency(tmp_path, concurrency):
    write_json_atomic(tmp_path / "plan.json", {})
    with pytest.raises(ValueError, match="between 1 and 24"):
        asyncio.run(run_exported_evaluation(
            tmp_path, EvaluationSettings(concurrency=concurrency), tmp_path
        ))


def test_runner_default_concurrency_stays_ten():
    assert EvaluationSettings().concurrency == 10


@pytest.mark.parametrize("concurrency", [1, 2, 6])
@pytest.mark.parametrize("route_count", [2, 3])
def test_dual_service_queues_preserve_every_execution(
    evaluation_export, tmp_path, monkeypatch, concurrency, route_count
):
    if route_count == 3:
        monkeypatch.setenv("OPENAI_BASE_URL_TERTIARY", "https://third.example/v1")
        monkeypatch.setenv("OPENAI_API_KEY_TERTIARY", "third-key")
    monkeypatch.setenv("OPENAI_BASE_URL_SECONDARY", "https://second.example/v1")
    monkeypatch.setenv("OPENAI_API_KEY_SECONDARY", "test-key")
    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(kwargs["schedule_index"])
        await asyncio.sleep(0)
        return {"meta": {"artifact_directory": "fixture", "run_id": definition["id"]}}

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    result = asyncio.run(run_exported_evaluation(
        evaluation_export, EvaluationSettings(concurrency=concurrency), tmp_path
    ))
    assert sorted(calls) == [0, 1, 2]
    assert result["completed_valid_executions"] == 3
    assert result["remaining_executions"] == 0


@pytest.mark.parametrize("reason", ["data_inspection_failed", "maximum context length"])
def test_input_rejection_does_not_stop_unrelated_tasks(
    evaluation_export, tmp_path, monkeypatch, reason,
):
    from egocua.execution import RemoteRunError

    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        if len(calls) == 1:
            raise RemoteRunError({"type": "BadRequestError", "message": reason})
        return {"meta": {"artifact_directory": "fixture", "run_id": definition["id"]}}

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(run_exported_evaluation(
        evaluation_export, EvaluationSettings(concurrency=1), tmp_path,
    ))
    assert len(calls) == 3
    assert summary["completed_valid_executions"] == 2
    assert summary["remaining_executions"] == summary["failed_request_executions"] == 1
    assert summary["execution_stop"] is None
    outcomes = read_study(evaluation_export)["execution_results"]
    assert outcomes[0]["valid"] is False and "success" not in outcomes[0]
    assert outcomes[0]["input_rejection"] in ("moderation", "context_limit")


@pytest.mark.parametrize("selected", [None, (0,)])
def test_resume_skips_known_rejection_unless_explicitly_selected(
    evaluation_export, tmp_path, monkeypatch, selected,
):
    saved = {}
    schedule = read_study(evaluation_export)["execution_schedule"]
    old = {"execution_index": 0, "run": schedule[0]["run"], "valid": False,
           "error": {"type": "BadRequestError", "message": "PROHIBITED_CONTENT"}}
    saved["execution_results"] = [old]
    write_json_atomic(evaluation_export / "results.json", saved)
    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        return {"meta": {"artifact_directory": "fixture", "run_id": definition["id"]}}

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(run_exported_evaluation(
        evaluation_export, EvaluationSettings(concurrency=1, execution_indexes=selected),
        tmp_path,
    ))
    assert (schedule[0]["run"]["task_id"] in calls) == (selected is not None)
    assert len(calls) == (1 if selected else 2)
    assert summary["remaining_executions"] == (2 if selected else 1)
    assert read_study(evaluation_export)["execution_results"][0] == old


def test_request_failure_summary_deduplicates_and_preserves_valid_results():
    def failed(index, message):
        return {"execution_index": index, "valid": False, "error": {
            "type": "BadRequestError", "message": message,
        }}

    outcomes = [failed(0, "data_inspection_failed"), failed(0, "PROHIBITED_CONTENT"),
                failed(1, "maximum context length"), failed(2, "Connection error"),
                failed(3, "PROHIBITED_CONTENT"), {"execution_index": 3, "valid": True},
                failed(4, "PROHIBITED_CONTENT"), {"execution_index": 4}]
    summary = _request_failure_summary(outcomes)
    assert {kind: group["count"] for kind, group in summary.items()} == {
        "moderation": 1, "context_limit": 1, "other": 1,
    }
    assert [row["execution_index"] for row in unresolved_request_failures(outcomes)] == [0, 1, 2]


def test_external_running_execution_is_deferred_not_completed(
    evaluation_export, tmp_path, monkeypatch
):
    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        return {
            "meta": {
                "artifact_directory": str(tmp_path / definition["id"]),
                "run_id": definition["id"],
            }
        }

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(
            evaluation_export,
            EvaluationSettings(deferred_execution_indexes=(0,)),
            tmp_path,
        )
    )
    first = read_study(evaluation_export)["execution_schedule"][0]
    assert first["run"]["task_id"] not in calls
    assert summary["completed_valid_executions"] == 2
    assert summary["remaining_executions"] == summary["deferred_executions"] == 1


def test_unknown_deferred_execution_fails_before_model_calls(
    evaluation_export, tmp_path
):
    with pytest.raises(IndexError, match="unknown deferred execution"):
        asyncio.run(
            run_exported_evaluation(
                evaluation_export,
                EvaluationSettings(deferred_execution_indexes=(999,)),
                tmp_path,
            )
        )


def test_failed_service_stops_batch_without_retry(
    evaluation_export, tmp_path, monkeypatch
):
    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        if len(calls) == 2:
            raise RuntimeError("provider service unavailable")
        return {
            "meta": {
                "artifact_directory": str(tmp_path / definition["id"]),
                "run_id": definition["id"],
            }
        }

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(
            evaluation_export, EvaluationSettings(concurrency=1), tmp_path
        )
    )
    assert (summary["attempted_executions"], summary["completed_valid_executions"]) == (
        2,
        1,
    )
    assert "provider service unavailable" in summary["execution_stop"]["message"]


@pytest.mark.parametrize("failed_index", [0, 1])
def test_transport_failure_keeps_workers_consuming_without_retry(
    evaluation_export, tmp_path, monkeypatch, failed_index
):
    import httpx
    from openai import APIConnectionError

    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        if len(calls) - 1 == failed_index:
            transport = httpx.ConnectError("socket failed")
            api_error = APIConnectionError(
                request=httpx.Request("POST", "https://offline.invalid")
            )
            api_error.__cause__ = transport
            error = RuntimeError("connection failed")
            error.__cause__ = api_error
            error.run_artifact = {
                "artifact_directory": "failed-archive",
                "run_id": "failed-run",
            }
            raise error
        return {
            "meta": {"artifact_directory": definition["id"], "run_id": definition["id"]}
        }

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(
            evaluation_export, EvaluationSettings(concurrency=1), tmp_path
        )
    )
    outcomes = read_study(evaluation_export)["execution_results"]
    assert len(calls) == len(set(calls)) == 3
    assert summary["completed_valid_executions"] == 2
    assert summary["remaining_executions"] == summary["failed_request_executions"] == 1
    assert summary["execution_stop"] is None
    failed = outcomes[failed_index]
    assert failed["valid"] is False
    assert "success" not in failed
    assert failed["artifact_directory"] == "failed-archive"
    assert failed["run_id"] == "failed-run"
    assert failed["error"]["cause"]["cause"]["type"] == "ConnectError"


def test_incomplete_requests_exit_cli_nonzero_without_global_stop(
    evaluation_export, tmp_path, monkeypatch, capsys
):
    async def incomplete(*args):
        return {
            "execution_stop": None,
            "failed_request_executions": 1,
            "remaining_executions": 1,
        }

    monkeypatch.setattr("egocua.study_cli.run_exported_evaluation", incomplete)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "study",
            "--study-from",
            str(evaluation_export),
            "--root",
            str(tmp_path),
            "--run-all",
        ],
    )
    with pytest.raises(SystemExit) as caught:
        study_cli()
    assert caught.value.code == 1
    assert json.loads(capsys.readouterr().out)["failed_request_executions"] == 1


def test_transport_failure_cli_finishes_queue_and_exits_nonzero(evaluation_export):
    script = """
import asyncio
import sys
from pathlib import Path
import httpx
from openai import APIConnectionError
from egocua import study_cli
from egocua.adapters import browsergym

directory = Path(sys.argv[1])
calls = 0

async def model_task(task, condition, **kwargs):
    global calls
    calls += 1
    if calls == 2:
        error = RuntimeError('connection failed')
        error.__cause__ = APIConnectionError(request=httpx.Request('POST', 'https://offline.invalid'))
        error.__cause__.__cause__ = httpx.RemoteProtocolError('peer closed')
        raise error
    return {'meta': {'artifact_directory': task['id'], 'run_id': task['id']}}

browsergym.run_task = model_task
from egocua import study_execution as execution
execution.check_run = lambda *args: {'run_valid': True, 'e2e_success': False}
execution.summarize_records = lambda arms, records, **kwargs: {'physical_runs': len(records)}
sys.argv = ['study', '--study-from', str(directory), '--root', str(directory.parent), '--run-all', '--concurrency', '1']
study_cli.main()
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(evaluation_export)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert process.returncode == 1, (process.stdout, process.stderr)
    definition = read_study(evaluation_export)
    assert definition["execution_stop"] is None
    assert len(definition["execution_results"]) == 3
    assert sum(row.get("valid") is True for row in definition["execution_results"]) == 2
    assert read_json_strict(evaluation_export / "results.json")[
        "failed_request_execution_indexes"
    ] == [1]
    assert '"remaining_executions": 1' in process.stdout


def test_transport_failure_preserves_prior_valid_row_and_other_active_worker(
    evaluation_export, tmp_path, monkeypatch
):
    import httpx

    definition = read_study(evaluation_export)
    first = definition["execution_schedule"][0]
    old = {
        "execution_index": first["execution_index"],
        "run": first["run"],
        "valid": True,
        "success": False,
        "artifact_directory": "old-valid",
        "run_id": "old-valid",
    }
    definition["execution_results"] = [old]
    write_json_atomic(evaluation_export / "plan.json", definition)
    monkeypatch.setattr(
        "egocua.study_execution.load_run_record",
        lambda path: {"task_snapshot": {}, "meta": {"run_id": "old-valid"}},
    )
    sibling_started = asyncio.Event()
    failed = asyncio.Event()
    sibling_completed = asyncio.Event()
    calls = []

    async def model_task(task, condition, **kwargs):
        calls.append(task["id"])
        if len(calls) == 1:
            await sibling_started.wait()
            failed.set()
            raise httpx.ReadTimeout("request timed out")
        sibling_started.set()
        await failed.wait()
        await asyncio.sleep(0)
        sibling_completed.set()
        return {"meta": {"artifact_directory": "new-valid", "run_id": "new-valid"}}

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(
            evaluation_export, EvaluationSettings(concurrency=2), tmp_path
        )
    )
    outcomes = read_study(evaluation_export)["execution_results"]
    assert sibling_completed.is_set()
    assert len(calls) == len(set(calls)) == 2
    assert first["run"]["task_id"] not in calls
    assert outcomes[0] == old
    assert summary["reused_valid_executions"] == 1
    assert summary["completed_valid_executions"] == 2
    assert summary["failed_request_executions"] == 1
    assert summary["execution_stop"] is None


@pytest.mark.parametrize("status_code", [401, 403, 429, 500, 503])
def test_http_service_response_still_stops_entire_evaluation(
    evaluation_export, tmp_path, monkeypatch, status_code
):
    import httpx
    from openai import APIStatusError

    calls = []

    async def model_task(task, condition, **kwargs):
        calls.append(task["id"])
        request = httpx.Request("POST", "https://offline.invalid")
        error = RuntimeError("HTTP service error")
        error.__cause__ = APIStatusError(
            "HTTP service error",
            response=httpx.Response(status_code, request=request),
            body=None,
        )
        raise error

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(evaluation_export, EvaluationSettings(), tmp_path)
    )
    assert len(calls) == 1
    assert summary["remaining_executions"] == 3
    assert summary["failed_request_executions"] == 0
    assert summary["execution_stop"]["type"] == "RuntimeError"



@pytest.mark.parametrize("concurrency", [1, 2])
def test_proxy_gateway_cause_stops_canary_and_concurrent_workers(
    evaluation_export, tmp_path, monkeypatch, concurrency
):
    import httpx
    from openai import APIConnectionError

    calls = []
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def model_task(task, condition, **kwargs):
        calls.append(task["id"])
        if concurrency == 2 and len(calls) == 1:
            return {"meta": {"artifact_directory": "canary", "run_id": "canary"}}
        if len(calls) == concurrency:
            if concurrency == 2:
                await sibling_started.wait()
            proxy = httpx.ProxyError("503 Service Unavailable")
            error = APIConnectionError(
                request=httpx.Request("POST", "https://offline.invalid")
            )
            error.__cause__ = proxy
            raise error
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(run_exported_evaluation(
        evaluation_export, EvaluationSettings(concurrency=concurrency), tmp_path
    ))
    assert summary["execution_stop"]["type"] == "APIConnectionError"
    assert summary["failed_request_executions"] == 0
    assert summary["completed_valid_executions"] == concurrency - 1
    assert len(calls) == (1 if concurrency == 1 else 3)
    assert sibling_cancelled.is_set() == (concurrency == 2)


def test_service_stop_cancels_workers_before_returning(
    evaluation_export, tmp_path, monkeypatch
):
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    calls = []

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        if len(calls) == 1:
            return {"meta": {"artifact_directory": "completed", "run_id": "completed"}}
        if len(calls) == 2:
            await sibling_started.wait()
            raise RuntimeError("provider connection error")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(
            evaluation_export, EvaluationSettings(concurrency=2), tmp_path
        )
    )
    outcomes = read_study(evaluation_export)["execution_results"]
    assert sibling_cancelled.is_set()
    assert summary["completed_valid_executions"] == 1
    assert len(outcomes) == 3
    assert sum(row.get("valid") is True for row in outcomes) == 1
    assert {row["error"]["type"] for row in outcomes if "error" in row} == {
        "RuntimeError",
        "CancelledError",
    }


def test_service_stop_exits_cli_with_uncooperative_background_task(
    evaluation_export,
):
    script = """
import asyncio
import sys
from pathlib import Path
from egocua import study_cli
from egocua.adapters import browsergym

directory = Path(sys.argv[1])
calls = 0
background_tasks = []

async def browser_background():
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        print('BACKGROUND_CANCELLED', flush=True)
        await asyncio.Event().wait()

async def model_task(task, condition, **kwargs):
    global calls
    calls += 1
    if calls == 1:
        return {'meta': {'artifact_directory': 'completed', 'run_id': 'completed'}}
    background_tasks.append(asyncio.create_task(browser_background()))
    await asyncio.sleep(0)
    raise RuntimeError('provider connection error')

from egocua import study_execution as execution
original_evaluation = execution.run_exported_evaluation
async def observed_evaluation(*args):
    summary = await original_evaluation(*args)
    print('EVALUATION_RETURNED', flush=True)
    return summary

browsergym.run_task = model_task
study_cli.run_exported_evaluation = observed_evaluation
execution.check_run = lambda *args: {'run_valid': True, 'e2e_success': False}
execution.summarize_records = lambda arms, records, **kwargs: {'physical_runs': len(records)}
sys.argv = ['study', '--study-from', str(directory), '--root', str(directory.parent), '--run-all', '--concurrency', '1']
study_cli.main()
"""
    try:
        process = subprocess.run(
            [sys.executable, "-c", script, str(evaluation_export)],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"CLI hung after worker completion: {error.stdout!r}")
    assert process.returncode == 1, (process.stdout, process.stderr)
    assert "EVALUATION_RETURNED" in process.stdout
    assert "BACKGROUND_CANCELLED" in process.stdout
    definition = read_study(evaluation_export)
    assert definition["execution_stop"] == {
        "type": "RuntimeError",
        "message": "provider connection error",
    }
    assert len(definition["execution_results"]) == 2
    assert definition["execution_results"][0]["valid"] is True
    assert definition["execution_results"][1]["error"]["type"] == "RuntimeError"
    assert read_json_strict(evaluation_export / "results.json")["physical_runs"] == 1


@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGHUP])
def test_process_signal_saves_completed_runs_and_interruption(
    evaluation_export, tmp_path, stop_signal
):
    ready = tmp_path / "waiting"
    script = """
import asyncio
import json
import sys
from pathlib import Path
from egocua import controlled_experiments as studies
from egocua.adapters import browsergym

directory, ready = map(Path, sys.argv[1:])

async def model_task(task, condition, **kwargs):
    if task['track'] != 'graft':
        ready.touch()
        await asyncio.Event().wait()
    return {'meta': {'artifact_directory': str(directory / task['id']), 'run_id': task['id']}}

browsergym.run_task = model_task
from egocua import study_execution as execution
execution.check_run = lambda *args: {'run_valid': True, 'e2e_success': False}
execution.summarize_records = lambda arms, records, **kwargs: {'physical_runs': len(records)}
definition = studies.read_study(directory)
canary = next(row['execution_index'] for row in definition['execution_schedule'] if row['run']['track'] == 'graft')
summary = asyncio.run(execution.run_exported_evaluation(directory, execution.EvaluationSettings(canary_index=canary), directory.parent))
print(json.dumps(summary), flush=True)
sys.exit(1 if summary['execution_stop'] else 0)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(evaluation_export), str(ready)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                pytest.fail("fixture did not reach its pending execution")
            time.sleep(0.02)
        process.send_signal(stop_signal)
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert process.returncode == 1, (stdout, stderr)
    definition = read_study(evaluation_export)
    assert definition["execution_stop"] == {
        "type": "InterruptedError",
        "message": f"received {stop_signal.name}",
    }
    assert sum(row.get("valid") is True for row in definition["execution_results"]) == 1
    assert read_json_strict(evaluation_export / "results.json")["physical_runs"] == 1
    assert all(
        row.get("valid") is True or row.get("error", {}).get("type") == "CancelledError"
        for row in definition["execution_results"]
    )


@pytest.mark.parametrize("blocked_count", [1, 3])
def test_task_repairs_block_only_affected_executions(
    evaluation_export, tmp_path, monkeypatch, blocked_count
):
    path = evaluation_export / "plan.json"
    definition = json.loads(path.read_text())
    blocked_ids = {
        entry["run"]["task_id"]
        for entry in definition["execution_schedule"][:blocked_count]
    }
    definition["review_gate"] = {
        "formal_task_statuses": dict.fromkeys(blocked_ids, "needs_repair")
    }
    path.write_text(json.dumps(definition))
    calls = []

    async def model_task(task, condition, **kwargs):
        calls.append(task["id"])
        return {
            "meta": {
                "artifact_directory": str(tmp_path / task["id"]),
                "run_id": task["id"],
            }
        }

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(evaluation_export, EvaluationSettings(), tmp_path)
    )
    assert not blocked_ids.intersection(calls)
    assert summary["completed_valid_executions"] == 3 - blocked_count
    assert summary["remaining_executions"] == blocked_count
    assert summary["blocked_executions"] == blocked_count
    assert summary["execution_stop"] is None
    results = json.loads((evaluation_export / "results.json").read_text())
    assert {
        entry["run"]["task_id"] for entry in results["blocked_executions"]
    } == blocked_ids


def test_completed_valid_attempts_are_reused_and_only_pending_are_run(
    evaluation_export, tmp_path, monkeypatch
):
    definition = read_study(evaluation_export)
    original_run = definition["execution_schedule"][0]["run"]
    definition["execution_results"] = [
        {
            "execution_index": 0,
            "run": original_run,
            "valid": True,
            "artifact_directory": "original-run",
        }
    ]
    (evaluation_export / "plan.json").write_text(json.dumps(definition))
    monkeypatch.setattr(
        "egocua.study_execution.load_run_record",
        lambda path: {"task_snapshot": {"id": original_run["task_id"]}},
    )
    calls = []
    both_pending_started = asyncio.Event()

    async def model_task(task, condition, **kwargs):
        calls.append(task["id"])
        if len(calls) == 2:
            both_pending_started.set()
        await asyncio.wait_for(both_pending_started.wait(), timeout=1)
        return {
            "meta": {
                "artifact_directory": str(tmp_path / task["id"]),
                "run_id": task["id"],
            }
        }

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    summary = asyncio.run(
        run_exported_evaluation(
            evaluation_export, EvaluationSettings(concurrency=2), tmp_path
        )
    )
    assert original_run["task_id"] not in calls
    assert len(calls) == 2
    assert summary["reused_valid_executions"] == 1
    assert summary["completed_valid_executions"] == 3


def test_task_validation_batch_explicitly_passes_and_records_purpose(
    evaluation_export, tmp_path, monkeypatch
):
    purposes = []

    async def model_task(definition, condition, **kwargs):
        purposes.append(kwargs["run_purpose"])
        return {
            "meta": {
                "artifact_directory": str(tmp_path / definition["id"]),
                "run_id": definition["id"],
            }
        }

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    asyncio.run(
        run_exported_evaluation(
            evaluation_export,
            EvaluationSettings(run_purpose="task_validation"),
            tmp_path,
        )
    )
    assert purposes == ["task_validation"] * 3
    assert read_study(evaluation_export)["run_purpose"] == "task_validation"


@pytest.fixture
def arms():
    definitions = [task(track) for track in ("graft", "refresh", "clarify")]
    base = replace(PLAN, release_trigger="first_native_commitment")
    historical_visual = replace(
        base,
        experiment_id="fixture-study-visual",
        conditions={
            "online": Condition("online", "online", "full", "allowed", None),
            "first_frame": Condition(
                "first_frame", "online", "first_frame", "allowed", None
            ),
        },
        track_conditions={
            track: ("online", "first_frame")
            for track in ("graft", "refresh", "clarify")
        },
        replicate_ids=(0,),
        track_probe_conditions={},
    )
    return [
        ControlledArm("visual", definitions, historical_visual, "visual"),
        online_arm("formal", definitions[:2], base),
    ]


def _replay_task(track):
    definition = task(track)
    frame = definition["visual_packets"][0]["frame_paths"][0]
    timestamps = [index / 2 for index in range(64)]
    definition["visual_session"] = {
        **({"update_onset_seconds": 16} if track != "clarify" else {}),
        "reviewed_seconds": [4] if track == "clarify" else [4, 16],
        "frequency_samples": {
            "window_seconds": [0, timestamps[-1]],
            "timestamps_seconds": timestamps,
            "frame_paths": [frame] * len(timestamps),
            "sources": [],
        },
    }
    return definition


@pytest.fixture
def replay_arms(tmp_path):
    definitions = [_replay_task(track) for track in ("graft", "refresh", "clarify")]
    for definition in definitions:
        materialize_assets(definition, products(definition["track"]), tmp_path)
    plan = load_experiment_plan(Path(__file__).parents[1] / "configs/benchmark.json")
    return [online_arm("formal", definitions, plan), information_probe_arm(definitions, plan)]


@pytest.fixture
def source_groups(monkeypatch):
    def group(snapshot, *, root):
        return f"fixture:{snapshot['source']['id']}"

    monkeypatch.setattr(
        "egocua.controlled_experiments.task_recording_group",
        group
    )
    monkeypatch.setattr(
        "egocua.study_summary.task_recording_group",
        group
    )


def test_visual_arms_change_only_scope(arms):
    visual = arms[0]
    assert visual.plan.release_trigger == "first_native_commitment"
    full, sparse = (
        visual.plan.condition(name).as_dict() for name in ("online", "first_frame")
    )
    assert {
        key: value for key, value in full.items() if key not in {"name", "scope"}
    } == {key: value for key, value in sparse.items() if key not in {"name", "scope"}}
    for definition in visual.tasks:
        timelines = [
            Timeline.compile(
                definition,
                visual.plan.condition(name),
                default_release_step=2,
                reference_steps=None,
            )
            for name in ("online", "first_frame")
        ]
        assert timelines[0].instruction == timelines[1].instruction
        assert timelines[0].offer == timelines[1].offer
        assert [release.step for release in timelines[0].releases] == [
            release.step for release in timelines[1].releases
        ]
        assert len(timelines[0].releases) == len(timelines[1].releases)


def test_frozen_study_keeps_historical_gate_without_claiming_current_status(
    arms, tmp_path, monkeypatch, capsys
):
    evidence = {"task_reviews": [], "content_groups": []}
    historical_gate = {"sampled_task_phases": {"test-graft": "release_ready"}}
    monkeypatch.setattr(
        "egocua.study_cli.load_frozen_study",
        lambda path: (arms, evidence, historical_gate),
    )
    monkeypatch.setattr("egocua.study_cli.study_matrix", lambda *args, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["study", "--study-from", str(tmp_path)])

    study_cli()

    report = json.loads(capsys.readouterr().out)
    assert report["review_gate"] == historical_gate
    assert "research_readiness" not in report


def test_sampling_visibility_comes_from_required_facts_not_a_duplicate_label(
    tmp_path, source_groups, replay_arms
):
    definition = _replay_task("graft")
    arm = build_variable_arm("pace", [definition], replay_arms[0].plan)
    evidence = {
        "task_reviews": [
            {
                "task_id": definition["id"],
                "review_status": "review_required",
                "visibility": {"interval_1s": "visible"},
                "sampling_review": {
                    "status": "reviewed",
                    "conditions": {
                        "interval_1s": {
                            "facts": [
                                {"required_for_task": True, "visibility": "absent"},
                                {"required_for_task": False, "visibility": "uncertain"},
                            ]
                        }
                    },
                },
            }
        ]
    }
    rows = study_matrix([arm], root=tmp_path, evidence=evidence)
    row = next(
        row
        for row in rows
        if row["task_id"] == definition["id"] and row["condition"] == "pace_8s"
    )
    assert row["key_evidence_visibility"] == "absent"
    assert row["visibility_review_status"] == "reviewed"


def test_unreviewed_sampling_does_not_inherit_original_frame_visibility(
    tmp_path, source_groups, replay_arms
):
    definition = _replay_task("graft")
    arm = build_variable_arm("pace", [definition], replay_arms[0].plan)
    evidence = {
        "task_reviews": [
            {
                "task_id": definition["id"],
                "review_status": "reviewed",
                "visibility": {"full": "visible", "interval_1s": "visible"},
            }
        ]
    }
    rows = study_matrix([arm], root=tmp_path, evidence=evidence)
    assert len(rows) == 3
    assert {row["key_evidence_visibility"] for row in rows} == {"unreviewed"}


def test_observational_groups_do_not_create_paired_comparisons(
    arms, source_groups, tmp_path
):
    records = _visual_records(arms, tmp_path)
    evidence = {
        "analysis_groups": [
            {
                "id": "descriptive",
                "variable": "span",
                "status": "verified",
                "members": [{"task_id": records[0][1]["task_id"], "level": "current"}],
            }
        ]
    }
    report = summarize_records(arms, records, root=tmp_path, evidence=evidence)
    assert report["observational_analysis"][0]["inference"] == "descriptive_only"
    assert not any(
        comparison["left"].startswith("span::") for comparison in report["comparisons"]
    )


def test_matrix_retains_refresh_second_packet(arms, source_groups, tmp_path):
    matrix = study_matrix(arms, root=tmp_path)
    assert len(matrix) == 8
    refresh = [
        row for row in matrix if row["arm"] == "visual" and row["track"] == "refresh"
    ]
    counts = {row["condition"]: row["frame_counts_if_delivered"] for row in refresh}
    assert counts == {"online": [4, 4], "first_frame": [1, 1]}
    assert all(row["source_group"] == "fixture:1" for row in matrix)


def test_export_roundtrips_through_existing_plan_and_cli(replay_arms, tmp_path):
    arms = replay_arms
    destination = tmp_path / "export"
    export_study(arms, destination, root=tmp_path)
    for arm in arms:
        directory = destination / "tasks" / arm.name
        loaded_plan = load_experiment_plan(destination / "plan.json", arm.name)
        loaded_tasks = load_tasks(directory)
        assert replace(
            loaded_plan, site_snapshots={}, frozen_asset_root=arm.plan.frozen_asset_root
        ) == arm.plan
        assert Path(loaded_plan.site_snapshots["egomart"]).is_file()
        assert expand_runs(loaded_tasks, loaded_plan) == arm.schedule()
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "egocua.cli",
                "expand",
                str(directory),
                "--plan",
                str(directory.parent.parent / "plan.json"),
                "--arm",
                arm.name,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(process.stdout) == arm.schedule()
    with pytest.raises(ValueError, match="overwrite"):
        export_study(arms, destination, root=tmp_path)


def test_frozen_study_preserves_inputs_and_evidence_without_current_selection(
    replay_arms, tmp_path
):
    arms = replay_arms
    evidence = {"schema_version": 2, "task_reviews": [], "content_groups": []}
    export_study(arms, tmp_path / "frozen", root=tmp_path, evidence=evidence)
    restored, restored_evidence, _gate = load_frozen_study(tmp_path / "frozen")
    for actual, expected in zip(restored, arms, strict=True):
        assert {task["id"]: task for task in actual.tasks} == {
            task["id"]: task for task in expected.tasks
        }
        assert replace(
            actual,
            tasks=expected.tasks,
            plan=replace(actual.plan, site_snapshots={}, frozen_asset_root=expected.plan.frozen_asset_root),
        ) == expected
    assert restored_evidence == evidence


def test_every_export_owns_its_frames_images_and_site_snapshot(replay_arms, tmp_path):
    from egocua.run_materials import require_frozen_assets, wearer_input_paths
    from egocua.run_materials import public_site_materials

    destination = tmp_path / "frozen"
    export_study(replay_arms, destination, root=tmp_path)
    frozen, _, _ = load_frozen_study(destination)
    for arm in frozen:
        assert Path(arm.plan.frozen_asset_root) == destination / "inputs"
        for snapshot in arm.plan.site_snapshots.values():
            assert Path(snapshot).is_relative_to(destination / "inputs")
        for definition in arm.tasks:
            data = read_json_strict(Path(arm.plan.site_snapshot_for(definition)))
            for relative in public_site_materials(definition, data)["images"]:
                (tmp_path / relative).write_bytes(b"changed live website image")
        for relative in wearer_input_paths(arm.tasks, arm.plan):
            (tmp_path / relative).write_bytes(b"changed live frame")
        require_frozen_assets(arm.tasks, arm.plan)
    assert not (tmp_path / "site-snapshots").exists()


def test_export_requires_an_explicit_project_root(replay_arms, tmp_path):
    with pytest.raises(TypeError, match="root"):
        export_study(replay_arms, tmp_path / "missing-root")
    with pytest.raises(ValueError, match="project root"):
        export_study(replay_arms, tmp_path / "null-root", root=None)
    assert not (tmp_path / "missing-root").exists()
    assert not (tmp_path / "null-root").exists()


def test_reexporting_frozen_inputs_ignores_later_live_changes(reusable_study, tmp_path):
    old, _current_arms = reusable_study
    frozen, _, _ = load_frozen_study(old)
    frame = frozen[0].tasks[0]["visual_packets"][0]["frame_paths"][0]
    (tmp_path / frame).write_bytes(b"later live change")
    destination = tmp_path / "same-frozen-version"
    export_study(frozen, destination, root=tmp_path, reuse_from=old)
    assert len(read_study(destination)["execution_results"]) == 2


def test_subset_export_drops_unused_historical_snapshot_overrides(reusable_study, tmp_path):
    old, _current_arms = reusable_study
    [arm], _, _ = load_frozen_study(old)
    arm = replace(arm, plan=replace(arm.plan, site_snapshots={
        **arm.plan.site_snapshots,
        "unused-task": arm.plan.site_snapshots["egomart"],
    }))
    destination = tmp_path / "subset"
    export_study([arm], destination, root=tmp_path, reuse_from=old)
    [restored], _, _ = load_frozen_study(destination)
    assert set(restored.plan.site_snapshots) == {"egomart"}
    assert len(read_study(destination)["execution_results"]) == 2


def test_frozen_study_cli_can_select_one_variable_without_current_tasks(
    arms, tmp_path, monkeypatch, capsys
):
    selected = replace(arms[0], variable="pace")
    other = replace(arms[1], variable="cue")
    monkeypatch.setattr(
        "egocua.study_cli.load_frozen_study", lambda path: ([selected, other], None, {})
    )
    monkeypatch.setattr("egocua.study_cli.study_matrix", lambda *args, **kwargs: [])
    monkeypatch.setattr("egocua.study_cli.execution_schedule", lambda arms: [])
    monkeypatch.setattr(sys, "argv", [
        "study", "--study-from", str(tmp_path), "--variable", "pace",
    ])
    study_cli()
    result = json.loads(capsys.readouterr().out)
    assert set(result["variables"]) == {"pace"}
    assert set(result["task_sample"]) == {"pace"}


def test_frozen_study_cli_rejects_missing_variable(arms, tmp_path, monkeypatch):
    monkeypatch.setattr(
        "egocua.study_cli.load_frozen_study", lambda path: (arms, None, {})
    )
    monkeypatch.setattr(sys, "argv", [
        "study", "--study-from", str(tmp_path), "--variable", "pace",
    ])
    with pytest.raises(SystemExit, match="2"):
        study_cli()


def test_frozen_study_cli_selects_registered_replicates(arms, tmp_path, monkeypatch, capsys):
    selected = replace(arms[0], plan=replace(arms[0].plan, replicate_ids=(0, 1, 2)))
    monkeypatch.setattr("egocua.study_cli.load_frozen_study", lambda path: ([selected], None, {}))
    monkeypatch.setattr("egocua.study_cli.study_matrix", lambda *args, **kwargs: [])
    captured = []
    monkeypatch.setattr("egocua.study_cli.execution_schedule", lambda arms: captured.extend(arms) or [])
    monkeypatch.setattr(sys, "argv", [
        "study", "--study-from", str(tmp_path), "--replicate-id", "0",
    ])
    study_cli()
    assert captured[0].plan.replicate_ids == (0,)


@pytest.mark.parametrize("replicates", [["4"], ["0", "0"]])
def test_frozen_study_cli_rejects_unregistered_or_duplicate_replicates(
    arms, tmp_path, monkeypatch, replicates
):
    monkeypatch.setattr("egocua.study_cli.load_frozen_study", lambda path: (arms, None, {}))
    options = [argument for value in replicates for argument in ("--replicate-id", value)]
    monkeypatch.setattr(sys, "argv", ["study", "--study-from", str(tmp_path), *options])
    with pytest.raises(SystemExit, match="2"):
        study_cli()


def test_frozen_variable_filter_does_not_rescore_mixed_schedule(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "study", "--study-from", str(tmp_path), "--variable", "pace",
        "--records", str(tmp_path / "run"),
    ])
    with pytest.raises(SystemExit, match="2"):
        study_cli()


def test_exported_plan_reaches_browser_cli_without_paid_call(
    replay_arms, tmp_path, monkeypatch, capsys
):
    arm = replay_arms[0]
    export_study([arm], tmp_path / "export", root=tmp_path)
    directory = tmp_path / "export" / "tasks" / arm.name
    scheduled = arm.schedule()[0]
    captured = {}

    async def fake_evaluation(directory, settings, root):
        captured.update(
            directory=directory,
            settings=settings,
            plan=load_frozen_study(directory)[0][0].plan,
        )
        return {"completed_valid_executions": 0}

    monkeypatch.setattr(
        "egocua.study_execution.run_exported_evaluation", fake_evaluation
    )
    run_cli(
        [
            "run-browser",
            scheduled["task_id"],
            scheduled["condition"],
            "--plan",
            str(directory.parent.parent / "plan.json"),
            "--arm",
            arm.name,
            "--root",
            str(tmp_path),
        ]
    )
    assert replace(captured["plan"], site_snapshots={}, frozen_asset_root=arm.plan.frozen_asset_root) == arm.plan
    assert captured["settings"].execution_indexes == (0,)
    assert captured["directory"] == tmp_path / "export"
    assert captured["plan"].experiment_id == arm.plan.experiment_id
    assert json.loads(capsys.readouterr().out)["completed_valid_executions"] == 0


def test_prepared_study_uses_snapshot_and_plan_without_review_replay(
    replay_arms, tmp_path, monkeypatch
):
    definition = replay_arms[0].tasks[0]
    arm = online_arm("formal", [definition], replay_arms[0].plan)
    export_study([arm], tmp_path / "export", root=tmp_path)
    directory = tmp_path / "export" / "tasks" / arm.name
    plan = load_experiment_plan(directory.parent.parent / "plan.json", arm.name)
    scheduled = expand_runs([definition], plan)[0]
    captured = {}

    async def fake_evaluation(directory, settings, root):
        captured.update(
            directory=directory,
            settings=settings,
            plan=load_frozen_study(directory)[0][0].plan,
        )
        return {"completed_valid_executions": 0}

    def unexpected_review(*args, **kwargs):
        raise AssertionError("prepared experiment must not replay publication reviews")

    monkeypatch.setattr(
        "egocua.study_execution.run_exported_evaluation", fake_evaluation
    )
    monkeypatch.setattr("egocua.cli.validate_task_for_model_run", unexpected_review)
    run_cli(
        [
            "run-browser",
            definition["id"],
            scheduled["condition"],
            "--plan",
            str(directory.parent.parent / "plan.json"),
            "--arm",
            arm.name,
            "--root",
            str(tmp_path),
            "--replicate-id",
            str(scheduled["replicate_id"]),
        ]
    )
    assert captured["plan"].model == plan.model
    assert captured["settings"].execution_indexes == (0,)
    assert captured["settings"].concurrency == 1
    assert captured["plan"].site_snapshots["egomart"] == plan.site_snapshots["egomart"]


def _visual_records(arms, root):
    arm = arms[0]
    definition = next(
        definition for definition in arm.tasks if definition["track"] == "graft"
    )
    materialize_site(products("graft"), root)
    records = []
    for row in arm.schedule():
        if row["task_id"] != definition["id"]:
            continue
        config = RunConfig(
            agent=arm.plan.adapter,
            model=arm.plan.model,
            model_version=arm.plan.model_version,
            default_release_step=arm.plan.default_release_step,
            reference_steps=arm.plan.reference_steps,
            release_trigger=arm.plan.release_trigger,
            max_steps=arm.plan.max_steps,
            replicate_id=row["replicate_id"],
            schedule_index=row["schedule_index"],
            asset_root=root,
            output_root=root / "artifacts",
            experiment_id=arm.plan.experiment_id,
        )

        def execute(context):
            context.advance()
            context._state.select(item_id("graft", 0))
            context.advance()
            return []

        record = run_experiment(
            definition, arm.plan.condition(row["condition"]), config, execute
        )
        # The reporter's input really comes back through the canonical artifact reader.
        records.append(
            (
                arm.plan.experiment_id,
                load_run_record(record["meta"]["artifact_directory"]),
            )
        )
    return records


def test_report_rejects_changed_current_goal(arms, source_groups, tmp_path):
    records = _visual_records(arms, tmp_path)
    arms[0].tasks[0]["goal"]["acceptable_states"][0]["selected_entity_ids"] = [
        "new-answer"
    ]
    with pytest.raises(ValueError, match="content differs from registered inputs"):
        summarize_records(arms, records, root=tmp_path)


@pytest.mark.parametrize(
    "mutation, error",
    [
        ("duplicate", "duplicate"),
        ("snapshot", "snapshot"),
        ("protocol", "wrong_release_trigger"),
        ("materials", "content differs"),
    ],
)
def test_report_rejects_uncomparable_records(
    arms, source_groups, tmp_path, mutation, error
):
    records = _visual_records(arms, tmp_path)
    if mutation == "duplicate":
        records.append(records[0])
    elif mutation == "snapshot":
        records[0][1].pop("task_snapshot")
    elif mutation == "protocol":
        records[0][1]["meta"]["runtime"]["release_trigger"] = "timeline_step"
    else:
        records[1][1]["task_snapshot"]["instructions"]["a"] = (
            "A different initial request"
        )
    with pytest.raises(ValueError, match=error):
        summarize_records(arms, records, root=tmp_path)


def test_invalid_delivery_does_not_emit_valid_process_metrics(
    arms, source_groups, tmp_path
):
    records = _visual_records(arms, tmp_path)
    record = records[0][1]
    record["actions"][-1]["trigger_reason"] = "invented-trigger"
    report = summarize_records(arms, records, root=tmp_path)
    invalid = next(row for row in report["runs"] if not row["run_valid"])
    assert invalid["process"]["recovery_status"] == "invalid_run"


def test_visibility_is_reported_without_changing_outcome(arms, source_groups, tmp_path):
    records = _visual_records(arms, tmp_path)
    evidence = {
        "task_reviews": [
            {
                "task_id": definition["id"],
                "review_status": "review_required",
                "visibility": {"full": "visible", "first_frame": "absent"},
            }
            for definition in arms[0].tasks
        ]
    }
    matrix = study_matrix(arms, root=tmp_path, evidence=evidence)
    assert all(
        row["key_evidence_visibility"] == "absent"
        for row in matrix
        if row["condition"] == "first_frame"
    )
    report = summarize_records(arms, records, root=tmp_path, evidence=evidence)
    assert all(row["e2e_success"] for row in report["runs"])
    sparse = report["conditions"]["visual::first_frame"]
    assert sparse["success_by_evidence_visibility"]["absent"]["rate"] == 1
    assert sparse["actual_frame_counts"] == [1]


def test_report_keeps_unknown_and_invalid_separate_from_failure(
    arms, source_groups, tmp_path
):
    records = _visual_records(arms, tmp_path)
    records[0][1]["actions"][-1]["trigger_reason"] = "invented-trigger"
    report = summarize_records(arms, records, root=tmp_path)
    condition = "visual::" + records[0][1]["condition"]["name"]
    summary = report["conditions"][condition]
    assert summary["invalid"] == 1
    assert summary["success"]["rate"] is None
    assert summary["obsolete_residual"]["denominator"] == 0


def test_run_inputs_use_one_seed_per_replicate_across_conditions(arms):
    rows = [row for arm in arms for row in arm.schedule()]
    assert {row["provider_seed"] for row in rows} == {arms[0].plan.scheduler_seed}


def test_legacy_hash_metadata_does_not_gate_experiment_summary(
    arms, source_groups, tmp_path
):
    records = _visual_records(arms, tmp_path)
    records[0][1]["meta"]["site_data_sha256"] = "old-unrelated-value"
    report = summarize_records(arms, records, root=tmp_path)
    assert report["physical_runs"] == 2
    assert report["observed_runs"] == 3


def test_report_summarizes_real_request_inputs_and_separates_groups(
    arms, source_groups, tmp_path
):
    records = _visual_records(arms, tmp_path)
    records[0][1]["trace"] = {
        "llm_requests": [
            {
                "wearer_image_count": 8,
                "website_screenshot_count": 1,
                "success": True,
                "usage": {"input_tokens": 100},
            },
            {
                "wearer_image_count": 8,
                "website_screenshot_count": 1,
                "success": False,
                "usage": None,
            },
        ]
    }
    report = summarize_records(arms, records, root=tmp_path)
    assert report["runs"][0]["request_inputs"] == {
        "requests": 2,
        "successful_requests": 1,
        "wearer_image_count": 16,
        "website_screenshot_count": 2,
        "requests_with_usage": 1,
    }
    assert set(report["by_track"]) == {"graft"}
    assert set(report["by_site"]) == {"egomart"}


def test_trajectory_summary_uses_recorded_states_without_changing_scores(
    arms, source_groups, tmp_path
):
    records = _visual_records(arms, tmp_path)
    prior = {
        record["meta"]["run_id"]: check_run(record["task_snapshot"], record)
        for _, record in records
    }
    report = summarize_records(arms, records, root=tmp_path)
    assert {row["run_id"] for row in report["runs"]} == prior.keys()
    for row in report["runs"]:
        score = prior[row["run_id"]]
        assert row["e2e_success"] == score["e2e_success"]
        assert row["evaluator"] == {
            key: score["evaluator"][key] for key in ("version", "predicate")
        }
        assert row["trajectory"]["reason"] == "valid_selection_retained"
        assert row["trajectory"]["final_goal_satisfied_before_update"] is True
        assert row["trajectory"]["native_state_changed_after_update"] is False
    summary = report["by_variable"]["visual"]
    assert summary["planned"] == sum(
        len(arm.schedule()) for arm in arms if arm.variable == "visual"
    )
    assert summary["valid"] == 2
    assert summary["success"]["count"] == 2
    assert summary["trajectory"]["by_reason"]["valid_selection_retained"]["count"] == 2
    assert summary["analysis_modes"]["within_task"]["valid"] == 2
    assert report["by_variable"]["formal"]["valid"] == 1


def test_span_summaries_keep_content_pairs_separate_from_observational_runs(
    arms, source_groups, tmp_path
):
    paired = replace(arms[0], variable="span", level="near", pair_id="source-one")
    observed = replace(
        arms[0],
        variable="span",
        name="span-observed",
        plan=replace(arms[0].plan, experiment_id="span-observed-fixture"),
    )
    records = _visual_records([paired], tmp_path)
    report = summarize_records([paired, observed], records, root=tmp_path)
    summary = report["by_variable"]["span"]
    assert summary["planned"] == 12
    assert summary["valid"] == 4
    assert summary["analysis_modes"]["paired"]["valid"] == 2
    assert summary["analysis_modes"]["observational"]["valid"] == 2
    assert summary["analysis_modes"]["paired"]["trajectory"]["known"] == 2
    assert summary["analysis_modes"]["observational"]["success"]["count"] == 2
    assert [row["analysis_mode"] for row in report["runs"]] == [
        "paired",
        "observational",
        "paired",
        "observational",
    ]


def test_single_registered_execution_saves_progress_and_does_not_repeat(
    evaluation_export, tmp_path, monkeypatch
):
    from egocua.controlled_experiments import select_execution

    entry = read_study(evaluation_export)["execution_schedule"][1]
    reference = entry["analysis_references"][0]
    index = select_execution(
        evaluation_export,
        {
            key: reference[key]
            for key in ("arm", "task_id", "condition", "replicate_id")
        },
    )
    calls = []
    record = {
        "task_snapshot": {"id": reference["task_id"]},
        "meta": {"run_id": "single", "artifact_directory": str(tmp_path / "single")},
    }

    async def model_task(definition, condition, **kwargs):
        calls.append(definition["id"])
        return record

    monkeypatch.setattr("egocua.adapters.browsergym.run_task", model_task)
    monkeypatch.setattr(
        "egocua.study_execution.load_run_record", lambda path: record
    )
    settings = EvaluationSettings(
        canary_index=index, concurrency=1, execution_indexes=(index,)
    )
    before = (evaluation_export / "plan.json").read_bytes()
    first = asyncio.run(run_exported_evaluation(evaluation_export, settings, tmp_path))
    second = asyncio.run(run_exported_evaluation(evaluation_export, settings, tmp_path))
    assert calls == [reference["task_id"]]
    assert (
        first["completed_valid_executions"] == second["completed_valid_executions"] == 1
    )
    assert second["remaining_executions"] == 2
    assert len(read_study(evaluation_export)["execution_results"]) == 1
    assert (evaluation_export / "plan.json").read_bytes() == before


@pytest.mark.parametrize(
    ("variable", "tracks", "levels", "changed"),
    [
        (
            "pace",
            ("graft", "refresh", "clarify"),
            ("pace_8s", "pace_4s", "pace_16s"),
            {"batch_seconds", "frame_interval_seconds"},
        ),
        ("retention", ("graft", "refresh"), ("compressed", "full", "summary"), {"visual_history"}),
        ("cue", ("refresh",), ("silent", "cued", "cued_facts"), {"change_cue", "change_facts"}),
        ("arrival", ("graft",), ("concurrent", "after_finish"), {"follow_up_after_finish"}),
    ],
)
def test_controlled_variables_change_one_online_setting(variable, tracks, levels, changed):
    from egocua.controlled_experiments import sampled_task_ids
    from egocua.study_summary import _comparison_pairs

    plan = load_experiment_plan(Path("configs/benchmark.json"))
    online = plan.condition("online").as_dict()
    definitions = [task(track) for track in tracks]
    arm = build_variable_arm(variable, definitions, plan)

    assert tuple(arm.plan.conditions) == levels
    assert arm.plan.reference_condition_name == levels[0]
    assert replace(arm.plan.condition(levels[0]), name="online") == plan.condition("online")
    assert arm.plan.replicate_ids == (0,)
    assert sampled_task_ids([arm]) == {variable: [item["id"] for item in definitions]}
    for condition in arm.plan.conditions.values():
        payload = condition.as_dict()
        differs = {
            key for key in {*payload, *online} - {"name"} if payload.get(key) != online.get(key)
        }
        assert differs <= changed
        assert condition.batch_seconds / condition.frame_interval_seconds == 8
    base = "full" if variable == "retention" else levels[0]
    expected_pairs = (
        [("cue::silent", "cue::cued"), ("cue::cued", "cue::cued_facts")]
        if variable == "cue" else [
            (f"{variable}::{base}", f"{variable}::{level}") for level in levels if level != base
        ]
    )
    assert [pair[:2] for pair in _comparison_pairs([arm])] == expected_pairs
    assert arm.plan.as_dict()["conditions"] == [
        condition.as_dict() for condition in arm.plan.conditions.values()
    ]
    excluded = {"graft", "refresh", "clarify"} - set(tracks)
    for track in excluded:
        with pytest.raises(ValueError, match="requires a sample"):
            build_variable_arm(variable, [task(track)], plan)


def test_twenty_step_results_cannot_be_reused_at_thirty_steps(reusable_study, tmp_path):
    old, arms = reusable_study
    changed = [replace(arm, plan=replace(arm.plan, max_steps=30)) for arm in arms]
    export_study(changed, tmp_path / "new-budget", root=tmp_path, reuse_from=old)
    assert read_study(tmp_path / "new-budget")["execution_results"] == []


def test_retention_breakdowns_keep_the_same_complete_triplets():
    from egocua.study_summary import _comparison_results

    definitions = [task('graft'), task('refresh')]
    plan = load_experiment_plan(Path('configs/benchmark.json'))
    arm = build_variable_arm('retention', definitions, plan)
    rows = [
        {'task_id': definition['id'], 'replicate_id': 0,
         'condition': {'name': f'retention::{condition}'},
         'run_valid': not (definition['track'] == 'graft' and condition == 'summary'),
         'e2e_success': True}
        for definition in definitions for condition in arm.plan.conditions
    ]
    comparisons = _comparison_results([arm], rows, {t['id']: t['id'] for t in definitions})
    for comparison in comparisons:
        assert comparison['by_track']['graft']['coverage']['paired_pairs'] == 0
        assert comparison['by_track']['refresh']['coverage']['paired_pairs'] == 1
        subtype = definitions[1]['refresh']['policy']
        assert comparison['by_subtype'][subtype]['left_success'] == 1


@pytest.mark.parametrize("variable", ["pace", "retention", "cue", "arrival"])
def test_frozen_controlled_model_variant_preserves_inputs(variable):
    from egocua.controlled_experiments import model_variant
    plan = replace(PLAN, frozen_asset_root="/frozen/inputs")
    arm = ControlledArm(variable, [task("graft")], plan, variable=variable)
    changed = model_variant([arm], "deepseek-flash")[0]
    assert changed.tasks == arm.tasks
    assert replace(changed.plan, model=plan.model, model_version=plan.model_version) == plan
    assert changed.plan.model == changed.plan.model_version == "deepseek-flash"
    with pytest.raises(ValueError, match="frozen"):
        model_variant([replace(arm, plan=PLAN)], "deepseek-flash")


def test_repeated_variables_share_inputs_and_reuse_first_round(reusable_study, tmp_path):
    old, original = reusable_study
    sources = []
    for variable in ('pace', 'arrival'):
        source = tmp_path / variable
        arm = replace(original[0], name=variable, variable=variable,
                      plan=replace(original[0].plan, experiment_id='repeat-' + variable))
        export_study([arm], source, root=tmp_path, reuse_from=old)
        sources.append(source)
    originals = {p: (p / 'plan.json').read_bytes() for p in sources}
    arms, evidence, gate = repeated_controlled_study(sources, 3, tmp_path)
    assert all(arm.plan.replicate_ids == (0, 1, 2) for arm in arms)
    schedule = execution_schedule(arms)
    assert len(schedule) == 6 and all(len(row['analysis_references']) == 2 for row in schedule)
    assert {row['run']['provider_seed'] for row in schedule} == {
        original[0].plan.scheduler_seed + repeat for repeat in range(3)
    }
    destination = tmp_path / 'repeated'
    export_study(arms, destination, root=tmp_path, evidence=evidence, gate=gate,
                 reuse_from=tuple(sources))
    saved = read_study(destination)
    assert len(saved['execution_results']) == 2
    assert {row['run']['replicate_id'] for row in saved['execution_results']} == {0}
    assert len({row['artifact_directory'] for row in saved['execution_results']}) == 2
    assert all((p / 'plan.json').read_bytes() == data for p, data in originals.items())
    restored, _, _ = load_frozen_study(destination)
    assert [arm.first_run_ids for arm in restored] == [arm.first_run_ids for arm in arms]
    assert execution_schedule(restored) == saved['execution_schedule']


def test_repeated_variables_reject_noncontrolled_sources(reusable_study, tmp_path):
    old, _ = reusable_study
    with pytest.raises(ValueError, match='online input variables'):
        repeated_controlled_study([old], 3, tmp_path)


@pytest.mark.parametrize('count', [0, 1])
def test_repetition_extension_requires_additional_rounds(count, tmp_path):
    with pytest.raises(ValueError, match='at least two'):
        repeated_controlled_study([tmp_path], count, tmp_path)


def test_repetition_statistics_wait_for_complete_rounds(arms):
    arm = replace(arms[0], plan=replace(arms[0].plan, replicate_ids=(0, 1, 2)))
    tasks = arm.tasks[:2]
    arm = replace(arm, tasks=tasks)
    condition = next(iter(arm.plan.conditions))
    rows = []
    for replicate, outcomes in enumerate(((True, False), (True, True), (False, False))):
        for definition, success in zip(tasks, outcomes, strict=True):
            rows.append({'task_id': definition['id'], 'replicate_id': replicate,
                         'run_valid': True, 'e2e_success': success,
                         'analysis_reference': {'arm': arm.name, 'condition': condition}})
    cell = repetition_statistics([arm], rows)[arm.name][condition]['overall']
    assert cell['complete']
    assert cell['mean_success_rate'] == 0.5
    assert cell['round_standard_deviation'] == 0.5
    rows[-1]['run_valid'] = False
    cell = repetition_statistics([arm], rows)[arm.name][condition]['overall']
    assert not cell['complete'] and cell['mean_success_rate'] is None
    assert cell['round_standard_deviation'] is None


def test_update_state_partition_keeps_unfinished_initial_goals_separate():
    from egocua.study_summary import update_state_summary

    def row(held, initial, final, success=True, delivered=True):
        process = {'release_step': 2, 'selection_held_at_update': held}
        for name in ('obsolete_residual', 'valid_item_loss'):
            process.update({name + '_opportunity': False, name + '_count': 0})
        return {'run_valid': True, 'track': 'graft', 'e2e_success': success,
                'process': process, 'trajectory': {'status': 'known', 'update_delivered': delivered,
                'initial_goal_satisfied': initial, 'final_goal_satisfied_before_update': final}}

    rows = [row(False, False, False), row(True, True, False), row(True, False, False),
            row(True, True, True), row(None, None, None), row(False, False, False, delivered=False)]
    report = update_state_summary(rows)['groups']
    assert set(report) == {'no_prior_selection', 'revision_required', 'initial_goal_incomplete',
                           'final_state_already_valid', 'unknown', 'undelivered'}
    assert all(cell['runs'] == 1 for cell in report.values())
    assert sum(cell['runs'] for cell in report.values()) == len(rows)


def test_repetition_count_cannot_silently_modify_an_unrelated_cli_mode(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['study', '--repetitions', '3'])
    with pytest.raises(SystemExit) as result:
        study_cli()
    assert result.value.code == 2


def test_independent_existing_first_rounds_are_not_merged(reusable_study):
    _, originals = reusable_study
    source = originals[0]
    tasks = source.tasks
    a = {task['id']: {'online': 'original-' + task['id']} for task in tasks}
    b = deepcopy(a)
    b[tasks[0]['id']]['online'] = 'another-independent-first-run'
    arms = [replace(source, name=name, variable=name, first_run_ids=ids,
                    plan=replace(source.plan, replicate_ids=(0, 1, 2)))
            for name, ids in [('pace', a), ('arrival', b)]]
    schedule = execution_schedule(arms)
    first = [entry for entry in schedule if entry['run']['replicate_id'] == 0]
    newer = [entry for entry in schedule if entry['run']['replicate_id'] != 0]
    assert len(first) == 3  # One original observation is distinct, the other really is shared.
    assert len(newer) == 4 and all(len(entry['analysis_references']) == 2 for entry in newer)


def test_repetition_export_refuses_to_rerun_a_missing_original(reusable_study, tmp_path):
    old, originals = reusable_study
    arm = originals[0]
    planned = replace(arm, variable='pace', first_run_ids={
        task['id']: {'online': 'missing-original-id'} for task in arm.tasks},
        plan=replace(arm.plan, replicate_ids=(0, 1, 2)))
    destination = tmp_path / 'must-not-be-runnable'
    with pytest.raises(ValueError, match='original first-round reuse is incomplete'):
        export_study([planned], destination, root=tmp_path, reuse_from=old,
                     gate={'repetition_extension': {'original_round_reused': True}})
    assert not destination.exists()


def test_superseded_study_cannot_restart_its_old_schedule(evaluation_export, tmp_path, monkeypatch):
    write_json_atomic(evaluation_export / 'results.json', {'superseded_by': 'replacement', 'execution_results': []})
    monkeypatch.setattr('egocua.adapters.browsergym.run_task', lambda *a, **k: pytest.fail('model called'))
    with pytest.raises(ValueError, match='superseded'):
        asyncio.run(run_exported_evaluation(evaluation_export, EvaluationSettings(), tmp_path))
    assert read_study(evaluation_export)['superseded_by'] == 'replacement'


def test_execution_supplement_resumes_reused_runs_without_missing_companions(
    reusable_study, tmp_path, monkeypatch,
):
    from egocua.controlled_experiments import EXECUTION_SUPPLEMENT

    old, arms = reusable_study
    scoped = replace(arms[0], variable=EXECUTION_SUPPLEMENT, tasks=arms[0].tasks[:1])
    destination = tmp_path / 'supplement'
    export_study([scoped], destination, root=tmp_path, reuse_from=old)
    monkeypatch.setattr('egocua.adapters.browsergym.run_task',
                        lambda *a, **k: pytest.fail('valid run repeated'))
    report = asyncio.run(run_exported_evaluation(destination, EvaluationSettings(), tmp_path))
    assert report['completed_valid_executions'] == report['reused_valid_executions'] == 1
    saved = read_json_strict(destination / 'results.json')
    assert saved['comparisons'] == []
    assert saved['runs'][0]['analysis_mode'] == 'observational'


def test_main_interaction_comparison_still_requires_all_settings(reusable_study, tmp_path):
    from egocua.interaction_settings import INTERACTION_SETTINGS
    from egocua.study_summary import _comparison_pairs

    _, arms = reusable_study
    incomplete = replace(arms[0], name='online', variable=INTERACTION_SETTINGS)
    with pytest.raises(ValueError, match='all three settings'):
        _comparison_pairs([incomplete])
