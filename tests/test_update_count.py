from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from egocua.controlled_experiments import (
    ControlledArm, execution_schedule, export_study, load_update_count_study, load_frozen_study,
    model_variant, sampled_task_ids,
    repeated_controlled_study, read_study,
)
from egocua.development_gate import DevelopmentDecision, DevelopmentEvidence
from egocua.review_web import create_review_app
from egocua.run_materials import material_mismatch
from egocua.study_summary import _comparison_results, _composite_statistics
from egocua.synthesis_workflow import SynthesisWorkflow, load_composite_drafts
from egocua.timeline import composite_input
from egocua.update_count import (
    REGISTRATION, VARIANTS, audit_case, compile_variant,
)
from egocua.workspace import write_json_atomic
from tests.helpers import PLAN
from tests.test_composite_timeline import ONLINE


@pytest.fixture
def pair_root(tmp_path):
    raw = {
        "schema_version": 2, "id": "RG-01", "track": "composite", "site": "egomedia",
        "source": {"dossier": "source.json", "task_id": "parent", "window_seconds": [0, 47]},
        "visual_session": {}, "instructions": {"base": "Save a guide matching what I use."},
        "composite_events": [
            {"kind": "refresh", "onset_seconds": 16},
            {"kind": "graft", "onset_seconds": 24, "request": "Under ten minutes."},
            {"kind": "graft", "onset_seconds": 32, "request": "Make it under five minutes."},
        ],
        "goal": {"target_granularity": "category", "acceptable_states": [
            {"selected_entity_ids": ["short"], "fields": {}},
        ]},
        "truth": {"site_scope_entity_ids": ["short", "long"]},
    }
    frozen = deepcopy(raw)
    frames = [f"frames/{index}.jpg" for index in range(48)]
    frozen["visual_session"]["frequency_samples"] = {
        "timestamps_seconds": list(range(48)), "frame_paths": frames, "window_seconds": [0, 47],
    }
    frozen["visual_packets"] = [{"id": "V0", "frame_paths": frames[:8]}]
    snapshot = tmp_path / "snapshot.json"
    write_json_atomic(snapshot, {"site": "egomedia", "entities": [
        {"entity_id": "short", "duration_seconds": 120},
        {"entity_id": "long", "duration_seconds": 720},
    ]})
    plan = replace(
        PLAN, experiment_id="original", max_steps=30, composite_max_steps=30,
        conditions={"online": ONLINE}, reference_condition_name="online",
        track_conditions={"composite": ("online",)}, track_probe_conditions={},
        replicate_ids=(0,), site_snapshots={"egomedia": str(snapshot)},
        frozen_asset_root=str(tmp_path / "inputs"),
    )
    case = {
        "schema_version": 1, "id": "RG-01-update-count", "pair_id": "RG-01",
        "source_task_path": "data/experiments/composite/RG-01.json", "included": True,
        "direction": "merge", "subtype": "merged_requests",
        "composite_events": [raw["composite_events"][0], {
            "kind": "graft", "onset_seconds": 32, "request": "Keep the guide under five minutes.",
        }],
    }
    write_json_atomic(tmp_path / case["source_task_path"], raw)
    write_json_atomic(tmp_path / VARIANTS / "RG-01/variant.json", case)
    write_json_atomic(tmp_path / REGISTRATION, {
        "schema_version": 1, "reference_model": plan.model,
        "source_studies": {plan.model: "original"},
    })
    write_json_atomic(tmp_path / "original/tasks/composite-formal/composite/RG-01.json", frozen)
    write_json_atomic(tmp_path / "original/plan.json", {
        "arms": [{"name": "composite-formal", "variable": "composite-formal",
                  "level": None, "pair_id": None, "plan": plan.as_dict()}],
        "evidence": None, "execution_schedule": [],
    })
    return tmp_path, raw, frozen, case, plan


def test_merge_inherits_scene_key_and_source_without_mutating_original(pair_root):
    root, raw, _, case, _ = pair_root
    before = deepcopy(raw)
    variant = compile_variant(raw, case)
    assert raw == before
    assert variant["goal"] == raw["goal"]
    assert variant["source"] == raw["source"]
    assert variant["instructions"] == raw["instructions"]
    assert len(composite_input(variant).updates) == len(composite_input(raw).updates) - 1
    assert audit_case(root, case)["errors"] == []


def test_gap_audit_includes_visual_updates_and_preserves_original_request_spacing(pair_root):
    root, _, _, case, _ = pair_root
    audit = audit_case(root, case)
    assert audit["more_request_gap"] == {
        "from_event": "G1", "to_event": "G2", "source_seconds": 8, "replay_steps": 1,
    }
    assert [gap["replay_steps"] for gap in audit["adjacent_update_gaps"]["original"]] == [1, 1]
    assert audit["adjacent_update_gaps"]["variant"] == [{
        "from_event": "R1", "to_event": "G1", "source_seconds": 16, "replay_steps": 2,
    }]


@pytest.mark.parametrize("onset", [0, 16, 24])
def test_merge_rejects_opening_collision_or_early_final_information(pair_root, onset):
    root, _, _, case, _ = pair_root
    case["composite_events"][-1]["onset_seconds"] = onset
    case["composite_events"].sort(key=lambda event: event["onset_seconds"])
    if onset == 16:
        with pytest.raises(ValueError, match="strictly increasing"):
            audit_case(root, case)
    else:
        assert audit_case(root, case)["errors"]


@pytest.mark.parametrize("legacy", [False, True])
def test_insert_witness_checks_frozen_public_values_and_actual_exclusion(pair_root, legacy):
    root, raw, frozen, case, _ = pair_root
    for task in (raw, frozen):
        task["composite_events"].pop(1)
    case.update(direction="insert", composite_events=[
        {"kind": "graft", "onset_seconds": 8, "request": "Keep it under ten minutes."},
        *raw["composite_events"],
    ], exclusion_witness={
        "excluded_state": {"selected_entity_ids": ["long"], "fields": {}},
        "retained_state": {"selected_entity_ids": ["short"], "fields": {}},
        "prior_legality_evidence": "Both guides match the reviewed opening activity.",
        "original_request_effect": "A six-minute guide remains valid before the original five-minute limit.",
        "public_facts": [{"entity_id": "long", "path": ["duration_seconds"], "equals": 720}],
        "constraint": {"kind": "field", "field": "duration_seconds", "operator": "<", "value": 600,
                       "entity_id_by_state": {"excluded_state": "long", "retained_state": "short"}},
    })
    if legacy:
        for task in (raw, frozen):
            refresh, request = task.pop("composite_events")
            task["visual_session"].update(
                refresh_onset_seconds=refresh["onset_seconds"],
                update_onset_seconds=request["onset_seconds"],
            )
            task["instructions"]["b"] = request["request"]
    write_json_atomic(root / case["source_task_path"], raw)
    write_json_atomic(root / "original/tasks/composite-formal/composite/RG-01.json", frozen)
    audit = audit_case(root, case)
    assert audit["errors"] == []
    assert audit["more_request_gap"]["source_seconds"] == 24
    assert audit["more_request_gap"]["replay_steps"] == 3
    case["composite_events"][0]["onset_seconds"] = 9
    audit = audit_case(root, case)
    assert audit["errors"] == []
    assert audit["more_request_gap"]["source_seconds"] == 23
    assert audit["more_request_gap"]["replay_steps"] == 3
    case["exclusion_witness"]["constraint"]["value"] = 900
    assert "exclude the old state" in " ".join(audit_case(root, case)["errors"])
    case["exclusion_witness"]["public_facts"][0]["equals"] = 721
    assert "public fact differs" in " ".join(audit_case(root, case)["errors"])


def test_variants_are_explicit_opt_in_and_never_enter_original_composite_list(pair_root):
    root, _, _, case, _ = pair_root
    assert [task["id"] for task in load_composite_drafts(root)] == ["RG-01"]
    assert load_composite_drafts(root, [case["id"]])[0]["id"] == case["id"]
    case.update(included=False, exclusion_reason="No free replay batch.")
    write_json_atomic(root / VARIANTS / "RG-01/variant.json", case)
    with pytest.raises(ValueError, match="excluded"):
        load_composite_drafts(root, [case["id"]])


def test_update_count_trial_uses_thirty_without_changing_existing_budget(pair_root):
    root, raw, _, case, plan = pair_root
    write_json_atomic(root / "configs/benchmark.json", replace(plan, max_steps=20).as_dict())
    development = DevelopmentEvidence(root)
    assert development.trial_plan(raw).max_steps == 20
    assert development.trial_plan(compile_variant(raw, case)).max_steps == 30
    assert development.formal_plan.max_steps == 20


def test_prepare_pairs_reuses_frozen_protocol_and_requires_delta_review(pair_root, monkeypatch):
    root, _, frozen, case, plan = pair_root
    status = {"tasks": {case["id"]: {"next_action": "run_development_trial"}},
              "review_evidence": {case["id"]: {"verdict": "accepted"}},
              "update_count_audits": {case["pair_id"]: audit_case(root, case)}}
    monkeypatch.setattr(SynthesisWorkflow, "status", lambda *args, **kwargs: status)
    arms, _, gate = load_update_count_study(root, model=plan.model)
    assert [(arm.level, arm.direction) for arm in arms] == [("more", "merge"), ("fewer", "merge")]
    assert arms[0].tasks == [frozen]
    assert all(arm.plan.max_steps == 30 and arm.plan.condition("online") == ONLINE for arm in arms)
    assert gate["required_original_reuse"] == ["RG-01"]
    assert sampled_task_ids(arms) == {"update-count": ["RG-01", case["id"]]}
    assert model_variant(arms, "peer-model")[0].plan.model == "peer-model"
    status["tasks"][case["id"]]["next_action"] = "coding_agent_assess"
    with pytest.raises(ValueError, match="delta review"):
        load_update_count_study(root, model=plan.model)


def test_status_and_export_share_one_audit_and_fail_on_a_changed_event(pair_root, monkeypatch):
    root, raw, frozen, case, plan = pair_root
    variant = compile_variant(raw, case)
    write_json_atomic(root / "data/reviews/composite" / f"{case['id']}.json", {
        "reviewer": "fixture", "verdict": "accepted", "reviewed_task": variant,
        "evidence": {"naturalness": "A concise merged request.", "answers": "Inherited.",
                     "operations": "Unchanged native saving.",
                     "counterexamples": {event: "Original reviewed evidence." for event in composite_input(variant).event_ids}},
    })
    monkeypatch.setattr("egocua.synthesis_workflow.prepare_composite_candidate", lambda root, task, plan: compile_variant(frozen, case))
    monkeypatch.setattr(DevelopmentEvidence, "decision", lambda *args: DevelopmentDecision("run_development_trial", "Awaiting an authorized run."))
    calls = []
    def audit(workspace, candidate):
        calls.append(candidate["pair_id"])
        return audit_case(workspace, candidate)
    monkeypatch.setattr("egocua.synthesis_workflow.audit_case", audit)
    arms, _, _ = load_update_count_study(root, model=plan.model)
    assert len(arms) == 2 and calls == ["RG-01"]
    case["composite_events"][-1]["onset_seconds"] = 24
    write_json_atomic(root / VARIANTS / "RG-01/variant.json", case)
    row = SynthesisWorkflow(root).status(update_count=True)["tasks"][case["id"]]
    assert row["status"] == "needs_repair"


def test_export_refuses_to_schedule_any_missing_original_again(pair_root, monkeypatch):
    root, _, frozen, _, plan = pair_root
    arm = ControlledArm("pair", [frozen], plan, "update-count", "more", "RG-01", direction="merge")
    monkeypatch.setattr("egocua.controlled_experiments._reusable_results", lambda *args: [])
    monkeypatch.setattr("egocua.controlled_experiments.wearer_input_paths", lambda *args: set())
    monkeypatch.setattr("egocua.controlled_experiments.public_site_materials", lambda *args: {"images": []})
    with pytest.raises(ValueError, match="original composite reuse is incomplete"):
        export_study([arm], root / "missing", root=root)
    assert not (root / "missing").exists()


def test_export_round_trip_keeps_pairs_and_references_original_only(pair_root, monkeypatch):
    root, _, frozen, case, plan = pair_root
    arms = [
        ControlledArm("original", [frozen], plan, "update-count", "more", "RG-01", direction="merge"),
        ControlledArm("variant", [compile_variant(frozen, case)], replace(plan, experiment_id="variant"), "update-count", "fewer", "RG-01", direction="merge"),
    ]
    def reuse(prepared, source, workspace):
        entry = next(row for row in execution_schedule(prepared) if row["run"]["task_id"] == "RG-01")
        return [{"execution_index": entry["execution_index"], "run": entry["run"],
                 "run_id": "existing-run", "artifact_directory": "unchanged-original", "valid": True}]
    monkeypatch.setattr("egocua.controlled_experiments._reusable_results", reuse)
    monkeypatch.setattr("egocua.controlled_experiments.wearer_input_paths", lambda *args: set())
    monkeypatch.setattr("egocua.controlled_experiments.public_site_materials", lambda *args: {"images": []})
    export_study(arms, root / "export", root=root)
    frozen_arms, _, _ = load_frozen_study(root / "export")
    assert [(arm.pair_id, arm.level, arm.direction) for arm in frozen_arms] == [
        ("RG-01", "more", "merge"), ("RG-01", "fewer", "merge"),
    ]
    from egocua.controlled_experiments import read_study
    reused = read_study(root / "export")["execution_results"]
    assert len(reused) == 1 and reused[0]["run"]["task_id"] == "RG-01"


@pytest.fixture
def frozen_update_count_study(pair_root, monkeypatch):
    root, _, frozen, case, plan = pair_root
    arms = [
        ControlledArm("original", [frozen], plan, "update-count", "more", "RG-01", direction="merge"),
        ControlledArm("variant", [compile_variant(frozen, case)], plan, "update-count", "fewer", "RG-01", direction="merge"),
    ]
    def reuse(prepared, source, workspace):
        return [{
            "execution_index": row["execution_index"], "run": row["run"],
            "run_id": "existing-" + row["run"]["task_id"],
            "artifact_directory": "archived-" + row["run"]["task_id"], "valid": True,
            "success": False,
        } for row in execution_schedule(prepared) if row["run"]["replicate_id"] == 0]
    monkeypatch.setattr("egocua.controlled_experiments._reusable_results", reuse)
    monkeypatch.setattr("egocua.controlled_experiments.wearer_input_paths", lambda *args: set())
    monkeypatch.setattr("egocua.controlled_experiments.public_site_materials", lambda *args: {"images": []})
    source = root / "completed-pair"
    evidence = {"update_count_audits": {"RG-01": audit_case(root, case)}}
    export_study(arms, source, root=root, evidence=evidence, reuse_from=root / "archive")
    return root, source, evidence


def test_update_count_repetitions_preserve_first_round_and_only_add_two_rounds(frozen_update_count_study):
    root, source, evidence = frozen_update_count_study
    before = (source / "plan.json").read_text()
    arms, copied_evidence, gate = repeated_controlled_study([source], 3, root)
    assert copied_evidence == evidence
    assert all(arm.plan.replicate_ids == (0, 1, 2) for arm in arms)
    destination = root / "stability"
    export_study(arms, destination, root=root, evidence=copied_evidence, gate=gate, reuse_from=source)
    saved = read_study(destination)
    assert saved["category"] == "稳定性复测"
    assert len(saved["execution_schedule"]) == 6
    assert len(saved["execution_results"]) == 2
    done = {row["execution_index"] for row in saved["execution_results"]}
    pending = [row["run"] for row in saved["execution_schedule"] if row["execution_index"] not in done]
    assert {(row["task_id"], row["replicate_id"]) for row in pending} == {
        (task_id, replicate) for task_id in ("RG-01", "RG-01-update-count") for replicate in (1, 2)
    }
    assert (source / "plan.json").read_text() == before


def test_original_repeats_require_the_explicit_repetition_export(frozen_update_count_study):
    root, source, _ = frozen_update_count_study
    arms, evidence, _ = repeated_controlled_study([source], 3, root)
    with pytest.raises(ValueError, match="original composite reuse is incomplete"):
        export_study(arms, root / "unregistered", root=root, evidence=evidence, reuse_from=source)
    assert not (root / "unregistered").exists()


@pytest.mark.parametrize("missing_last", [False, True])
def test_repeated_pair_summary_keeps_source_count_and_reports_outcome_flips(pair_root, missing_last):
    _, _, frozen, _, plan = pair_root
    plan = replace(plan, replicate_ids=(0, 1, 2))
    arms, rows, sources = [], [], {}
    for pair_id, outcomes in [
        ("CG-01", {"fewer": [True, False, True], "more": [True, True, True]}),
        ("CG-02", {"fewer": [False, False, False], "more": [False, True, False]}),
    ]:
        for level, values in outcomes.items():
            task = {**deepcopy(frozen), "id": f"{pair_id}-{level}"}
            arms.append(ControlledArm(task["id"], [task], plan, "update-count", level, pair_id, direction="insert"))
            sources[task["id"]] = pair_id
            for replicate, success in enumerate(values):
                rows.append({"task_id": task["id"], "pair_id": pair_id,
                             "condition": {"name": f"update-count::{level}"},
                             "replicate_id": replicate, "run_valid": True, "e2e_success": success})
    if missing_last:
        rows.pop()
    [comparison] = _comparison_results(arms, rows, sources)
    stability = comparison["repetition_stability"]
    assert comparison["coverage"]["paired_sources"] == stability["independent_sources"] == 2
    assert stability["planned_replicates"] == [0, 1, 2]
    assert comparison["discordant_pair_unit"] == "task_pair_replicate"
    assert stability["outcome_stability"]["update-count::fewer"]["changed_outcome"] == 1
    if missing_last:
        assert not stability["complete"]
        assert stability["mean_difference"] is None
        assert stability["round_difference_standard_deviation"] is None
        assert stability["outcome_stability"]["update-count::more"]["complete_tasks"] == 1
    else:
        assert stability["complete"]
        assert comparison["coverage"]["paired_replicates"] == 6
        assert stability["mean_difference"] == pytest.approx(1 / 3)
        assert [row["difference"] for row in stability["rounds"].values()] == [0, 1, 0]
        assert stability["outcome_stability"]["update-count::more"]["changed_outcome"] == 1


def test_variant_development_checks_frozen_not_changed_live_website(pair_root, monkeypatch):
    root, _, frozen, case, plan = pair_root
    variant = compile_variant(frozen, case)
    record = {"task_id": variant["id"], "task_snapshot": variant, "condition": ONLINE.as_dict(),
              "meta": {"runtime": {"site_snapshot": plan.site_snapshot_for(frozen)}}}
    monkeypatch.setattr("egocua.run_materials.site_evidence_mismatch", lambda *args: None)
    def public_data(task, data):
        assert isinstance(task, dict), "the task snapshot must not be replaced by a website path"
        assert task["id"] == variant["id"]
        return data
    monkeypatch.setattr("egocua.run_materials.task_public_site_data", public_data)
    monkeypatch.setattr("egocua.run_materials.public_site_materials", lambda *args: {"images": []})
    monkeypatch.setattr("egocua.run_materials.site_images_match", lambda *args: True)
    roots = []
    def verify_frames(record, paths, workspace, *, current_root=None):
        roots.append(current_root)
    monkeypatch.setattr("egocua.run_materials.visual_mismatch", verify_frames)
    def reject_live_catalog(*args, **kwargs):
        raise AssertionError("variant validation must use the original frozen website")
    monkeypatch.setattr("egocua.run_materials.load_site_data", reject_live_catalog)
    assert material_mismatch(variant, record, plan, root=root) is None
    assert roots == [Path(plan.frozen_asset_root)]


def test_composite_summary_counts_every_repeat_instead_of_only_the_last(pair_root):
    _, _, frozen, _, plan = pair_root
    arm = ControlledArm("repeat", [frozen], replace(plan, replicate_ids=(0, 1, 2)), "update-count")
    events = {e.event_id: {"kind": e.mechanism, "status": "delivered"}
              for e in composite_input(frozen).updates}
    rows = []
    for replicate, success in enumerate([True, False, True]):
        details = deepcopy(events)
        details["R1"]["status"] = "not_delivered" if replicate == 1 else "delivered"
        rows.append({
            "task_id": frozen["id"], "replicate_id": replicate, "run_valid": True,
            "e2e_success": success, "analysis_reference": {"arm": arm.name},
            "composite": {"step_budget_reached": False, "events": details,
                          "clarification_offered": False, "clarification_received": None,
                          "delivery": {"R": replicate != 1, "G": True}},
        })
    report = _composite_statistics([arm], rows, None)["composite_statistics"][arm.name]
    assert report["overall"]["planned"] == report["overall"]["valid"] == 3
    assert report["overall"]["planned_tasks"] == 1
    assert report["overall"]["success"]["rate"] == pytest.approx(2 / 3)
    assert report["actual_update_events"]["G"]["denominator"] == 6
    assert report["actual_update_events"]["R"]["rate"] == pytest.approx(2 / 3)


def test_summary_keeps_direction_combination_and_missing_pairs_separate(pair_root):
    _, _, frozen, case, plan = pair_root
    arms, rows, sources = [], [], {}
    for pair_id, direction, outcomes in [
        ("RG-01", "merge", (False, True)),
        ("RG-02", "merge", (True, False)),
        ("RG-03", "insert", (True, True)),
        ("CG-01", "insert", (False, True)),
        ("CG-02", "insert", (False, None)),
    ]:
        for level, success in zip(("fewer", "more"), outcomes, strict=True):
            task = {**deepcopy(frozen), "id": f"{pair_id}-{level}"}
            arms.append(ControlledArm(f"{pair_id}-{level}", [task], plan, "update-count", level, pair_id, direction=direction))
            sources[task["id"]] = pair_id
            if success is not None:
                rows.append({"task_id": task["id"], "pair_id": pair_id,
                             "condition": {"name": f"update-count::{level}"},
                             "replicate_id": 0, "run_valid": True, "e2e_success": success})
    results = _comparison_results(arms, rows, sources)
    assert len(results) == 3
    by_group = {(row["combination"], row["direction"]): row for row in results}
    merged = by_group["RG", "merge"]
    assert merged["difference"] == 0
    assert merged["discordant_pairs"] == {"more_only": 1, "fewer_only": 1}
    cg = by_group["CG", "insert"]
    assert cg["difference"] == 1
    assert cg["expected_pairs"] == 2
    assert cg["coverage"]["paired_pairs"] == 1
    assert cg["difference_ci95"] is None
    assert "adaptability" not in cg


def test_review_page_shows_only_delta_materials_and_unified_status(pair_root, monkeypatch):
    root, _, _, case, _ = pair_root
    monkeypatch.setattr(SynthesisWorkflow, "status", lambda *args, **kwargs: {
        "summary": {"tasks": 1, "status_counts": {"needs_review": 1}},
        "tasks": {case["id"]: {"status": "needs_review", "next_action": "run_development_trial", "reason": "Awaiting an authorized run."}},
        "review_evidence": {},
    })
    workspace = SimpleNamespace(root=root, reviewer_id="fixture", scope="all")
    response = create_review_app(workspace).test_client().get("/review/update-count")
    assert response.status_code == 200
    assert "RG-01" in response.text and "Under ten minutes." in response.text
    assert "待授权开发试跑与轨迹核查" in response.text
    assert "原93题及成绩不变" in response.text
