import json
import hashlib
import shutil
from copy import deepcopy
from dataclasses import replace

import pytest

from tests.helpers import unchanged_task_owners as unchanged_task_owners

from egocua.checker import check_run
from egocua.development_gate import (
    POLICY,
    REVIEWS,
    REPORT,
    DevelopmentEvidence,
    material_identity,
    record_mismatch,
)
from egocua.sources import source_record
from egocua.environment import load_site_data
from egocua.execution import RELEASE_SEMANTICS_VERSION
from egocua.run_ledger import (
    SITE_RUNTIME_CONTRACTS,
    asset_hashes,
    task_site_execution_sha256,
    task_sha256,
    load_run_record,
)
from egocua.synthesis_workflow import SynthesisWorkflow
from egocua.synthesis_reports import REPORTS
from egocua.timeline import ACTION_ALIGNED_REPLAY
from tests.helpers import task, products, materialize_assets, _write_run_artifact

pytestmark = pytest.mark.usefixtures("unchanged_task_owners")


@pytest.fixture
def prepared(tmp_path):
    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    return definition, tmp_path


def _write_reviews(root, payload):
    (root / REVIEWS).write_text(json.dumps(payload))


def _new_task(root):
    _write_reviews(root, {"tasks": {}})


def _trial(definition, root, *, success=True, covered=True):
    source = root / "artifacts/runs/human/graft/online/human-1/run.json"
    record = json.loads(source.read_text())
    record["meta"].update(
        agent="browsergym",
        model="test-model",
        model_version="test-model",
        transport="browsergym-openai-vision-v1",
        task_sha256=task_sha256(definition),
    )
    record["trace"] = {
        "history": [
            {
                "metadata": {"step_number": 1},
                "model_output": {"action": [{"wait": {"seconds": 1}}]},
            }
        ]
    }
    if not success:
        record["final_state"]["selected_entity_ids"] = []
        record["live_state"]["selected_entity_ids"] = []
    if not covered:
        plan_path = root / "configs/pilot.json"
        plan = json.loads(plan_path.read_text())
        plan["runtime"]["release_trigger"] = "first_native_commitment"
        plan_path.write_text(json.dumps(plan))
        record["meta"]["runtime"]["release_trigger"] = "first_native_commitment"
        record["meta"]["runtime"]["release_semantics"] = RELEASE_SEMANTICS_VERSION
        record["actions"] = []
        record["steps"] = 1
    artifact = "artifacts/runs/development/test"
    directory = root / artifact
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "site.json"
    snapshot.write_text(json.dumps(load_site_data(root, "egomart")))
    archive = directory / "visual-inputs"
    for packet in definition["visual_packets"]:
        for path in packet["frame_paths"]:
            archived = archive / path
            archived.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / path, archived)
    record["meta"]["runtime"].update(
        site_snapshot=str(snapshot),
        site_contract=SITE_RUNTIME_CONTRACTS[definition["site"]],
        visual_asset_root=str(archive),
    )
    _write_run_artifact(root / artifact, "online", record)
    manifest_path = root / artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["experiment_id"] = "development"
    manifest_path.write_text(json.dumps(manifest))
    report = {
        "tasks": {definition["id"]: {"development_screen": {"run_artifact": artifact}}}
    }
    (root / REPORT).write_text(json.dumps(report))
    return artifact, load_run_record(root / artifact)


def _review(definition, root, artifact, *, success=True, covered=True):
    review = {
        "task_sha256": task_sha256(definition),
        "run_artifact": artifact,
        "reviewer": "coding-agent-trajectory-review",
        "evidence": ["Checked released inputs, actions and final state."],
        "verdict": "valid_model_success" if success else "valid_model_failure",
        "mechanism_coverage": {"observed": covered},
    }
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    return review


def test_composite_development_requires_actual_or_directed_coverage_per_mechanism(prepared):
    definition, root = prepared
    definition["track"] = "composite"
    checked = {
        "run_valid": True,
        "e2e_success": False,
        "composite_delivery": {"R": True, "G": False, "C": False},
    }
    review = {
        "task_sha256": task_sha256(definition),
        "reviewer": "reviewer",
        "verdict": "valid_model_failure",
        "evidence": ["The model did not wait for the revision or ask which item."],
        "mechanism_coverage": {"observed": True},
    }
    assert DevelopmentEvidence(root).trajectory_decision(
        definition, checked, review
    ).action == "review_development_trial"

    smoke = REPORTS["smoke"]
    review["mechanism_coverage"] = {
        "observed": False,
        "by_mechanism": checked["composite_delivery"],
        "directed_check": smoke.path.as_posix(),
    }
    report = root / smoke.path
    report.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "task_sha256": task_sha256(definition),
        "task_site_execution_sha256": task_site_execution_sha256(
            definition, load_site_data(root, "egomart")
        ),
        "status": "pass",
        "protocol_checks": 3,
        "composite_delivery": {"R": True, "G": True, "C": False},
    }
    report.write_text(json.dumps({
        "schema_version": smoke.schema_version,
        "composite_tasks": {definition["id"]: row},
    }))
    assert DevelopmentEvidence(root).trajectory_decision(
        definition, checked, review
    ).action == "review_development_trial"
    row["composite_delivery"]["C"] = True
    report.write_text(json.dumps({
        "schema_version": smoke.schema_version,
        "composite_tasks": {definition["id"]: row},
    }))
    assert DevelopmentEvidence(root).trajectory_decision(
        definition, checked, review
    ).action is None

    checked["composite_event_delivery"] = {
        "R1": {"status": "delivered"},
        "G1": {"status": "delivered"},
        "G2": {"status": "not_delivered"},
        "C1": {"status": "not_delivered"},
    }
    missing_review = DevelopmentEvidence(root).trajectory_decision(definition, checked, review)
    assert missing_review.action == "review_development_trial"
    assert "scheduled composite event" in missing_review.reason
    review["mechanism_coverage"]["by_event"] = {
        "R1": True, "G1": True, "G2": False, "C1": False,
    }
    assert DevelopmentEvidence(root).trajectory_decision(
        definition, checked, review
    ).action == "review_development_trial"
    row["composite_event_coverage"] = {"G2": True, "C1": True}
    report.write_text(json.dumps({
        "schema_version": smoke.schema_version,
        "composite_tasks": {definition["id"]: row},
    }))
    assert DevelopmentEvidence(root).trajectory_decision(
        definition, checked, review
    ).action is None


def test_superseded_fixed_replay_supports_existing_members_only(prepared, monkeypatch):
    definition, root = prepared
    evidence = DevelopmentEvidence(root)
    base = evidence.formal_plan
    current = replace(
        base.condition("online"),
        frame_interval_seconds=1,
        frame_delivery_steps=1,
        batch_seconds=8,
        replay_timing=ACTION_ALIGNED_REPLAY,
    )
    monkeypatch.setattr(evidence, "formal_plan", replace(base, conditions={"online": current}))
    monkeypatch.setattr(evidence, "formal_members", frozenset({definition["id"]}))
    archived = replace(current, replay_timing=None)
    old_record = {"condition": archived.as_dict()}
    inherited_plan = evidence._historical_admission_plan(definition["id"], old_record)
    assert inherited_plan.condition("online") == archived
    assert evidence._historical_admission_plan("unadmitted", old_record) is None
    assert evidence._historical_admission_plan(
        definition["id"], {"condition": current.as_dict()}
    ) is None


def test_explicit_legacy_admission_preserves_ready(prepared):
    definition, root = prepared
    assert DevelopmentEvidence(root).decision(definition).action is None
    assert (
        SynthesisWorkflow(root).status()["tasks"][definition["id"]]["status"] == "ready"
    )


@pytest.mark.parametrize(("source_step", "inherits"), [(20, True), (21, False)])
def test_fixed_replay_inherits_admission_unless_update_exceeds_budget(
    prepared, monkeypatch, source_step, inherits
):
    definition, root = prepared
    evidence = DevelopmentEvidence(root)
    condition = replace(
        evidence.formal_plan.condition("online"),
        frame_interval_seconds=1, frame_delivery_steps=1, batch_seconds=8,
    )
    evidence.formal_plan = replace(
        evidence.formal_plan, max_steps=20, conditions={"online": condition}
    )
    end = source_step * 8
    replay = {
        **definition,
        "visual_session": {
            "update_onset_seconds": end,
            "frequency_samples": {
                "window_seconds": [0, end],
                "timestamps_seconds": list(range(end + 1)),
                "frame_paths": [f"frame-{t}.jpg" for t in range(end + 1)],
            },
        },
    }
    source = source_record(definition, root=root)
    monkeypatch.setattr(
        "egocua.development_gate.source_record",
        lambda *_a, **_kw: {
            **source,
            "stream": {"session_seconds": [0, end], "review_evidence": "review.json"},
        },
    )
    monkeypatch.setattr("egocua.development_gate.prepare_session_tasks", lambda *_: [replay])

    assert evidence.inherited_admission(definition) is inherits
    assert evidence.decision(definition).action == (
        None if inherits else "run_development_trial"
    )


@pytest.mark.parametrize("reviewed", [False, True])
def test_missing_snapshot_is_checked_once_before_new_trial(prepared, reviewed):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    path = root / artifact / "run.json"
    record = json.loads(path.read_text())
    del record["task_snapshot"]
    path.write_text(json.dumps(record))
    if reviewed:
        review = _review(definition, root, artifact)
        review["verdict"] = "invalid_run"
        review["evidence"] = [
            "The archived task snapshot is missing; no other applicable run was found."
        ]
        _write_reviews(root, {"tasks": {definition["id"]: review}})
    decision = DevelopmentEvidence(root).decision(definition)
    expected = "run_development_trial" if reviewed else "verify_development_evidence"
    assert decision.action == expected


@pytest.mark.parametrize("change", ["frame", "site_image", "unrecorded_site_image"])
def test_real_execution_archive_satisfies_material_check_without_hash_bindings(
    prepared, change
):
    from egocua.execution import RunConfig, run_experiment
    from egocua.experiment import load_experiment_plan

    definition, root = prepared
    plan = load_experiment_plan(root / "configs/pilot.json")
    config = RunConfig(
        agent="browsergym",
        model="test-model",
        model_version="test-model",
        transport="browsergym-openai-vision-v1",
        default_release_step=plan.default_release_step,
        reference_steps=plan.reference_steps,
        max_steps=plan.max_steps,
        release_trigger=plan.release_trigger,
        asset_root=root,
        output_root=root / "experiments",
    )

    def execute(context):
        for observation in context.initial_payload["observations"]:
            context.observation_images(observation)
        for _ in range(plan.default_release_step):
            context.advance()
        return {"history": [{"model_output": {"action": [{"wait": {"seconds": 1}}]}}]}

    record = run_experiment(definition, plan.condition("online"), config, execute)
    assert record_mismatch(definition, record, root) is None
    assert (
        not {"asset_sha256", "site_data_sha256", "task_sha256"} & record["meta"].keys()
    )
    content = b"replacement at the original path"
    if change == "frame":
        path = definition["visual_packets"][0]["frame_paths"][0]
        (root / path).write_bytes(content)
        assert record_mismatch(definition, record, root) == "changed_visual_inputs"
    else:
        path = root / "sites/egomart/products.json"
        site = json.loads(path.read_text())
        product = site["products"][0]
        (root / product["image_path"]).write_bytes(content)
        if change == "site_image":
            product["image_provenance"]["sha256"] = hashlib.sha256(content).hexdigest()
            path.write_text(json.dumps(site))
        assert record_mismatch(definition, record, root) == "changed_public_site"


def test_no_baseline_is_not_an_implicit_legacy_exemption(prepared):
    definition, root = prepared
    _new_task(root)
    row = SynthesisWorkflow(root).status()["tasks"][definition["id"]]
    assert row["status"] == "needs_review"
    assert row["next_action"] == "run_development_trial"


@pytest.mark.parametrize("field", ["instructions", "source", "pixels", "site"])
def test_actual_input_change_ends_legacy_exemption(prepared, field):
    definition, root = prepared
    if field == "instructions":
        definition["instructions"]["b"] += " Also avoid plastic."
    elif field == "source":
        path = root / definition["source"]["dossier"]
        payload = json.loads(path.read_text())
        payload["tasks"][0]["source"]["media_id"] = "different-recording"
        path.write_text(json.dumps(payload))
    elif field == "pixels":
        (root / definition["visual_packets"][0]["frame_paths"][0]).write_bytes(
            b"changed"
        )
    else:
        path = root / "sites/egomart/products.json"
        payload = json.loads(path.read_text())
        payload["products"][0]["name"] += " changed"
        path.write_text(json.dumps(payload))
    assert (
        DevelopmentEvidence(root).decision(definition).action == "run_development_trial"
    )


def test_hidden_answer_change_retains_legacy_input_identity(prepared):
    definition, root = prepared
    previous = material_identity(definition, root)
    definition["goal"]["acceptable_states"] = (
        definition["goal"]["acceptable_states"] * 2
    )
    definition.setdefault("truth", {})["audit_note"] = (
        "Only an internal answer annotation."
    )
    assert material_identity(definition, root) == previous
    assert DevelopmentEvidence(root).decision(definition).action is None


@pytest.mark.parametrize("success", [True, False])
def test_success_and_failure_require_review_then_allow_admission(prepared, success):
    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root, success=success)
    assert record_mismatch(definition, record, root) is None
    assert check_run(definition, record)["e2e_success"] is success
    assert (
        DevelopmentEvidence(root).decision(definition).action
        == "review_development_trial"
    )
    _review(definition, root, artifact, success=success)
    assert DevelopmentEvidence(root).decision(definition).action is None


def test_hidden_answer_repair_rescores_without_new_model_run(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root)
    definition["goal"]["acceptable_states"] = [
        {"selected_entity_ids": [], "fields": {}}
    ]
    assert record_mismatch(definition, record, root) is None
    assert (
        DevelopmentEvidence(root).decision(definition).action
        == "review_development_trial"
    )
    _review(definition, root, artifact, success=False)
    assert DevelopmentEvidence(root).decision(definition).action is None
    assert (
        check_run(record["task_snapshot"], load_run_record(root / artifact))[
            "e2e_success"
        ]
        is True
    )


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("input", "changed_execution_inputs"),
        ("provider", "invalid_model_trace"),
        ("empty_trace", "invalid_model_trace"),
        ("old_site", "missing_current_site_evidence"),
        ("visual", "changed_visual_inputs"),
    ],
)
def test_unusable_runs_do_not_satisfy_the_gate(prepared, mutation, expected):
    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root)
    if mutation == "input":
        definition["instructions"]["b"] += " New requirement."
    elif mutation == "provider":
        record["trace"]["error"] = "insufficient_balance"
    elif mutation == "empty_trace":
        record["trace"] = {}
    elif mutation == "old_site":
        record["meta"]["runtime"].pop("site_snapshot")
    else:
        path = definition["visual_packets"][0]["frame_paths"][0]
        (root / artifact / "visual-inputs" / path).write_bytes(b"old pixels")
    assert record_mismatch(definition, record, root) == expected


def test_uncovered_mechanism_requires_current_directed_check(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root, success=False, covered=False)
    review = _review(definition, root, artifact, success=False, covered=False)
    assert (
        DevelopmentEvidence(root).decision(definition).action
        == "review_development_trial"
    )
    path = "artifacts/reports/task-smoke.json"
    report = json.loads((root / path).read_text())
    report["tasks"][definition["id"]]["protocol_checks"] = 1
    (root / path).write_text(json.dumps(report))
    review["mechanism_coverage"]["directed_check"] = path
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    assert DevelopmentEvidence(root).decision(definition).action is None


def test_unchanged_historical_trace_uses_directed_check_for_current_wording(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root, success=False, covered=False)
    review = _review(definition, root, artifact, success=False, covered=False)
    before = deepcopy(definition)
    definition["instructions"]["b"] += " Add the products to the cart."
    review["instruction_clarification"] = {
        "before_task": before,
        "after_instructions": deepcopy(definition["instructions"]),
        "reviewer": "coding-agent-wording-review",
        "reason": "Only restates the existing cart action.",
    }
    path = "artifacts/reports/task-smoke.json"
    review["mechanism_coverage"]["directed_check"] = path
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    report = json.loads((root / path).read_text())
    report["tasks"][definition["id"]].update(
        task_sha256=task_sha256(definition),
        task_site_execution_sha256=task_site_execution_sha256(
            definition, load_site_data(root, definition["site"])
        ),
        protocol_checks=1,
    )
    (root / path).write_text(json.dumps(report))

    assert DevelopmentEvidence(root).inherited_admission(definition)


def test_confirmed_defect_cannot_be_hidden_by_legacy_exemption(prepared):
    definition, root = prepared
    payload = json.loads((root / REVIEWS).read_text())
    payload["tasks"][definition["id"]] = {
        "task_sha256": task_sha256(definition),
        "verdict": "task_or_site_defect",
        "evidence": ["A concrete unresolved task defect."],
    }
    _write_reviews(root, payload)
    assert DevelopmentEvidence(root).decision(definition).action == "repair_task"


@pytest.mark.parametrize("inputs_changed", [False, True])
def test_targeted_repair_review_retains_only_unchanged_legacy_inputs(
    prepared, inputs_changed
):
    definition, root = prepared
    payload = json.loads((root / REVIEWS).read_text())
    payload["tasks"][definition["id"]] = {
        "task_sha256": task_sha256(definition),
        "reviewer": "repair-reviewer",
        "verdict": "repair_verified",
        "evidence": ["The omitted legal alternative now passes native state checks."],
        "history": [{"verdict": "task_or_site_defect"}],
    }
    _write_reviews(root, payload)
    definition["goal"]["acceptable_states"] *= 2
    if inputs_changed:
        definition["instructions"]["a"] += " Also check the new requirement."

    expected = "run_development_trial" if inputs_changed else None
    assert DevelopmentEvidence(root).decision(definition).action == expected


def test_malformed_baseline_fails_loudly(prepared):
    definition, root = prepared
    _write_reviews(root, {"legacy_baseline": {"policy": POLICY, "tasks": []}})
    with pytest.raises(ValueError, match="must be an object"):
        DevelopmentEvidence(root).decision(definition)


def test_missing_archive_requests_recovery_not_another_paid_run(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    (root / artifact / "run.json").unlink()
    assert (
        DevelopmentEvidence(root).decision(definition).action
        == "verify_development_evidence"
    )


@pytest.mark.parametrize("finished", [True, False])
def test_other_budgets_stay_historical_even_when_agent_finishes(prepared, finished):
    definition, root = prepared
    _, record = _trial(definition, root)
    record["meta"]["max_steps"] -= 1
    record["steps"] = 2
    record["trace"]["history"][-1]["model_output"]["action"] = [
        {"done": {"success": True, "text": "Finished"}}
        if finished
        else {"wait": {"seconds": 1}}
    ]
    assert record_mismatch(definition, record, root) == "wrong_execution_budget"


def test_exhausted_development_run_cannot_replace_larger_budget(prepared):
    definition, root = prepared
    _, record = _trial(definition, root)
    record["meta"]["max_steps"] -= 1
    record["steps"] = record["meta"]["max_steps"]
    record["trace"]["history"][-1]["model_output"]["action"] = [
        {"done": {"success": False, "text": "Step limit"}}
    ]
    assert record_mismatch(definition, record, root) == "wrong_execution_budget"


def test_short_historical_run_can_support_admission_but_not_current_score(prepared):
    definition, root = prepared
    artifact, record = _trial(definition, root)
    record["meta"]["max_steps"] -= 1
    record["trace"]["history"][-1]["model_output"]["action"] = [
        {"done": {"text": "Finished"}}
    ]
    _write_run_artifact(root / artifact, "online", record)
    _review(definition, root, artifact)

    assert record_mismatch(definition, record, root) == "wrong_execution_budget"
    assert DevelopmentEvidence(root).inherited_admission(definition)


def test_unknown_site_version_requires_evidence_recovery(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root)
    record["meta"]["runtime"]["site_contract"] = "unknown-runtime-version"
    _write_run_artifact(root / artifact, "online", record)
    assert (
        DevelopmentEvidence(root).decision(definition).action
        == "verify_development_evidence"
    )


def test_qualification_report_uses_same_trajectory_review_gate(prepared):
    from egocua.qualification_report import build_report

    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    report = build_report(root, experiment_ids=("development",))
    assert (
        report["tasks"][definition["id"]]["development_screen"]["status"]
        == "trajectory_review_required"
    )
    _review(definition, root, artifact)
    report = build_report(root, experiment_ids=())
    assert (
        report["tasks"][definition["id"]]["development_screen"]["status"]
        == "qualified_after_triage"
    )


def test_path_only_change_reuses_the_existing_trajectory_review(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    _review(definition, root, artifact)
    packet = definition["visual_packets"][0]
    old_path = packet["frame_paths"][0]
    new_path = "renamed-frame.jpg"
    (root / new_path).write_bytes((root / old_path).read_bytes())
    packet["frame_paths"][0] = new_path
    assert DevelopmentEvidence(root).decision(definition).action is None


def test_frozen_site_data_alone_does_not_prove_runtime_equivalence(prepared):
    definition, root = prepared
    _, record = _trial(definition, root)
    snapshot = root / "site-snapshot.json"
    snapshot.write_text(json.dumps(load_site_data(root, "egomart")))
    record["meta"]["runtime"]["site_snapshot"] = str(snapshot)
    record["meta"]["runtime"].pop("site_contract")
    assert record_mismatch(definition, record, root) == "missing_site_runtime_contract"


@pytest.mark.parametrize(
    "missing, reason",
    [
        ("site_snapshot", "missing_current_site_evidence"),
        ("site_contract", "missing_site_runtime_contract"),
        ("visual_asset_root", "missing_visual_evidence"),
    ],
)
def test_matching_legacy_hashes_do_not_replace_missing_run_evidence(
    prepared, missing, reason
):
    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root)
    record["meta"]["site_data_sha256"] = task_site_execution_sha256(
        definition, load_site_data(root, definition["site"])
    )
    record["meta"]["asset_sha256"] = asset_hashes(definition, root)
    record["meta"]["runtime"].pop(missing)
    _write_run_artifact(root / artifact, "online", record)
    _review(definition, root, artifact)

    assert record_mismatch(definition, record, root) == reason
    original = (root / artifact / "run.json").read_bytes()
    decision = DevelopmentEvidence(root).decision(definition)
    assert decision.action == "verify_development_evidence"
    assert reason in decision.reason
    assert (root / artifact / "run.json").read_bytes() == original
    assert check_run(record["task_snapshot"], record)["run_valid"]


def test_stale_review_pointer_does_not_hide_available_current_run(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    _review(definition, root, "artifacts/runs/deleted/old")
    assert (
        DevelopmentEvidence(root).decision(definition).action
        == "review_development_trial"
    )


def test_default_batch_does_not_reschedule_grandfathered_tasks(prepared):
    from egocua.qualification_report import build_report
    from tools.run_task_qualification import _select_tasks

    definition, root = prepared
    report = build_report(root, experiment_ids=())
    assert report["tasks"][definition["id"]]["development_screen"]["required"] is False
    assert _select_tasks([definition], report, None, concurrency=1) == []
    with pytest.raises(ValueError, match="not runnable"):
        _select_tasks([definition], report, definition["id"], concurrency=1)


@pytest.mark.parametrize("change", ["answer", "path"])
def test_unresolved_defect_survives_hidden_key_or_path_change(prepared, change):
    definition, root = prepared
    payload = json.loads((root / REVIEWS).read_text())
    payload["tasks"][definition["id"]] = {
        "task_sha256": task_sha256(definition),
        "verdict": "task_or_site_defect",
        "evidence": ["An unresolved material defect."],
    }
    _write_reviews(root, payload)
    if change == "answer":
        definition["goal"]["acceptable_states"] *= 2
    else:
        packet = definition["visual_packets"][0]
        old_path = packet["frame_paths"][0]
        (root / "renamed.jpg").write_bytes((root / old_path).read_bytes())
        packet["frame_paths"][0] = "renamed.jpg"
    assert (
        DevelopmentEvidence(root).decision(definition).action
        == "verify_development_evidence"
    )


@pytest.mark.parametrize(
    "request_evidence, valid",
    [
        ({"success": True}, True),
        ({"success": False, "model_output_error": True}, True),
        ({"success": False, "error": {"type": "TimeoutError"}}, False),
        ({"success": True, "provider_service_error": True}, False),
    ],
)
def test_received_empty_or_malformed_output_is_not_a_service_failure(
    prepared, request_evidence, valid
):
    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root, success=False)
    record["trace"] = {"history": [], "llm_requests": [request_evidence]}
    record["actions"] = [a for a in record["actions"] if a["type"] != "select"]
    _write_run_artifact(root / artifact, "online", record)
    _review(definition, root, artifact, success=False)
    expected = None if valid else "run_development_trial"
    assert DevelopmentEvidence(root).decision(definition).action == expected


def test_aborted_run_is_invalid_even_after_a_model_response(prepared):
    definition, root = prepared
    _, record = _trial(definition, root, success=False)
    record["trace"] = {"history": [], "llm_requests": [{"success": True}]}
    record["run_status"] = {"valid": False, "reason": "browser failed"}
    assert record_mismatch(definition, record, root) == "invalid_run"


def test_same_run_review_survives_task_identity_format_change(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    review = _review(definition, root, artifact)
    review["task_sha256"] = "superseded-task-identity-format"
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    assert DevelopmentEvidence(root).decision(definition).action is None


@pytest.mark.parametrize(
    "change", [None, "digests", "before", "after", "goal", "frames"]
)
def test_explicit_instruction_clarification_retains_only_unchanged_task_evidence(
    prepared, change
):
    definition, root = prepared
    _new_task(root)
    artifact, original_run = _trial(definition, root)
    review = _review(definition, root, artifact)
    before = deepcopy(definition)
    definition["instructions"]["b"] += " Add the products to the cart."
    review["instruction_clarification"] = {
        "before_task": before,
        "after_instructions": deepcopy(definition["instructions"]),
        "reviewer": "coding-agent-wording-review",
        "reason": "Clarify the existing cart outcome without changing the request.",
    }
    if change == "digests":
        before["source"]["dossier_sha256"] = "a" * 64
    elif change == "before":
        before["instructions"]["a"] += " Not the reviewed request."
    elif change == "after":
        definition["instructions"]["b"] += " Another change."
    elif change == "goal":
        definition["goal"]["acceptable_states"] = []
    elif change == "frames":
        definition["visual_packets"][0]["description"] = "Changed evidence."
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    decision = DevelopmentEvidence(root).decision(definition)
    assert (decision.action is None) == (change in {None, "digests"})
    assert load_run_record(root / artifact) == original_run


def test_instruction_clarification_preserves_legacy_admission_without_new_score(
    prepared,
):
    definition, root = prepared
    before = deepcopy(definition)
    definition["instructions"]["b"] += " Add the products to the cart."
    reviews = json.loads((root / REVIEWS).read_text())
    reviews["tasks"][definition["id"]] = {
        "instruction_clarification": {
            "before_task": before,
            "after_instructions": deepcopy(definition["instructions"]),
            "reviewer": "coding-agent-wording-review",
            "reason": "The requested website outcome is still the same cart.",
        }
    }
    _write_reviews(root, reviews)
    assert DevelopmentEvidence(root).decision(definition).action is None
    assert "run_artifact" not in reviews["tasks"][definition["id"]]


def test_review_packet_uses_the_same_named_instruction_record(prepared):
    from egocua.confidence_packets import CurrentReviewMaterials

    definition, root = prepared
    before = deepcopy(definition)
    original_packet = CurrentReviewMaterials(root).packet(before)
    definition["instructions"]["b"] += " Add the products to the cart."
    reviews = json.loads((root / REVIEWS).read_text())
    reviews["tasks"][definition["id"]] = {
        "instruction_clarification": {
            "before_task": before,
            "after_instructions": deepcopy(definition["instructions"]),
            "reviewer": "coding-agent-wording-review",
            "reason": "The cart outcome remains unchanged.",
        }
    }
    _write_reviews(root, reviews)
    current = CurrentReviewMaterials(root).packet(definition)
    assert current["retained_instruction_packet"] == original_packet
    assert current["materials_sha256"] != original_packet["materials_sha256"]
    definition["goal"]["acceptable_states"] = []
    assert "retained_instruction_packet" not in CurrentReviewMaterials(root).packet(
        definition
    )


def test_qualification_discovers_real_runs_outside_named_batches(prepared):
    from egocua.qualification_report import build_report

    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    (root / REPORT).write_text(json.dumps({"tasks": {}}))
    report = build_report(root)
    assert (
        report["tasks"][definition["id"]]["development_screen"]["run_artifact"]
        == artifact
    )


def test_named_impact_review_preserves_task_evidence_not_model_score(prepared):
    from egocua.checker import rescore_run

    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root)
    review = _review(definition, root, artifact)
    review["evidence_reuse"] = {
        "scope": "task_validation_only",
        "changes": ["Only another candidate received additional teaching metadata."],
        "retained_evidence": [
            "Existing native selection and terminal state were checked."
        ],
    }
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    path = root / "sites/egomart/products.json"
    catalog = json.loads(path.read_text())
    catalog["products"][1]["name"] += " additional metadata"
    path.write_text(json.dumps(catalog))
    assert record_mismatch(definition, record, root) == "changed_public_site"
    assert DevelopmentEvidence(root).decision(definition).action is None
    assert rescore_run(definition, record)["e2e_success"] is True
    assert load_run_record(root / artifact)["task_snapshot"] == record["task_snapshot"]
    definition["instructions"]["b"] += " A new user purpose."
    assert DevelopmentEvidence(root).decision(definition).action is not None


def test_retained_evidence_uses_corrected_hidden_answer_without_changing_run(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, record = _trial(definition, root)
    record["task_snapshot"]["goal"]["acceptable_states"] = deepcopy(
        definition["graft"]["initial_goal"]["acceptable_states"][1:]
    )
    _write_run_artifact(root / artifact, "online", record)
    original = (root / artifact / "run.json").read_bytes()
    assert check_run(record["task_snapshot"], record)["e2e_success"] is False
    review = _review(definition, root, artifact)
    review["evidence_reuse"] = {
        "scope": "task_validation_only",
        "changes": [
            "Corrected an omitted final answer; performer inputs are unchanged."
        ],
        "retained_evidence": [
            "Archived live state satisfies the corrected complete goal."
        ],
    }
    _write_reviews(root, {"tasks": {definition["id"]: review}})

    assert DevelopmentEvidence(root).decision(definition).action is None
    assert (root / artifact / "run.json").read_bytes() == original


def test_report_and_lifecycle_reuse_the_same_reviewed_wording_clarification(prepared):
    from egocua.qualification_report import build_report

    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    review = _review(definition, root, artifact)
    before = deepcopy(definition)
    definition["instructions"]["a"] += " Add it to the cart."
    review["instruction_clarification"] = {
        "before_task": before,
        "after_instructions": deepcopy(definition["instructions"]),
        "reviewer": "coding-agent-wording-review",
        "reason": "Clarify the supported cart action without changing the need.",
    }
    review["evidence_reuse"] = {
        "scope": "task_validation_only",
        "changes": ["The native cart outcome is unchanged."],
        "retained_evidence": ["The actual original trial and state were checked."],
    }
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    (root / "data/tasks/graft" / f"{definition['id']}.json").write_text(
        json.dumps(definition)
    )

    assert DevelopmentEvidence(root).decision(definition).action is None
    report = build_report(root, experiment_ids=("no-new-trial",))
    screen = report["tasks"][definition["id"]]["development_screen"]
    assert screen["status"] == "executable_demonstrated_once"
    assert screen["required"] is False
    assert screen["e2e_success"] is None


def test_retained_evidence_does_not_override_a_current_task_defect(prepared):
    definition, root = prepared
    _new_task(root)
    artifact, _ = _trial(definition, root)
    review = _review(definition, root, artifact)
    review["verdict"] = "task_or_site_defect"
    review["evidence_reuse"] = {
        "scope": "task_validation_only",
        "changes": ["Website metadata changed."],
        "retained_evidence": ["Old trial exists."],
    }
    _write_reviews(root, {"tasks": {definition["id"]: review}})
    assert DevelopmentEvidence(root).decision(definition).action == "repair_task"


def test_evidence_is_loaded_once_per_request_and_reread_next_request(
    prepared, monkeypatch
):
    from egocua import development_gate

    definition, root = prepared
    counts = {}
    read = development_gate._read_optional

    def counted(path):
        counts[path] = counts.get(path, 0) + 1
        return read(path)

    monkeypatch.setattr(development_gate, "_read_optional", counted)
    evidence = DevelopmentEvidence(root)
    first = evidence.decision(definition)
    assert evidence.decision(definition) == first
    assert counts[root / REVIEWS] == 1
    assert evidence.report is evidence.report
    assert counts[root / REPORT] == 1
    payload = read(root / REVIEWS)
    payload["tasks"][definition["id"]] = {
        "task_sha256": task_sha256(definition),
        "verdict": "task_or_site_defect",
        "evidence": ["Current public material has an unresolved contradiction."],
    }
    _write_reviews(root, payload)
    assert DevelopmentEvidence(root).decision(definition).action == "repair_task"
    assert counts[root / REVIEWS] == 2


@pytest.mark.parametrize("output", [None, {"action": [{"wait": {"seconds": 1}}]}])
def test_incomplete_old_budget_run_is_not_reused(prepared, output):
    definition, root = prepared
    _new_task(root)
    _, record = _trial(definition, root)
    record["meta"]["max_steps"] = 2
    record["trace"]["history"].append({"model_output": output})
    assert record_mismatch(definition, record, root) == "wrong_execution_budget"


def test_secondary_segment_mapping_is_part_of_new_source_identity(prepared):
    definition, root = prepared
    dossier_path = root / definition["source"]["dossier"]
    dossier = json.loads(dossier_path.read_text())
    source = dossier["tasks"][0]["source"]
    source.update(media_id="part-a", original_video_id="original", media_start_seconds=10)
    source.pop("media_sha256", None)
    source["media_segments"] = [
        {
            "media_id": name,
            "original_video_id": "original",
            "source_start_seconds": start,
            "source_end_seconds": start + 10,
            "media_sha256": digest * 64,
        }
        for name, start, digest in (("part-a", 10, "a"), ("part-b", 600, "b"))
    ]
    dossier_path.write_text(json.dumps(dossier))
    before = material_identity(definition, root)
    source["media_segments"][1]["source_start_seconds"] += 1
    source["media_segments"][1]["source_end_seconds"] += 1
    dossier_path.write_text(json.dumps(dossier))
    after = material_identity(definition, root)
    assert before["source_identity"] != after["source_identity"]
    assert before["execution_inputs"] == after["execution_inputs"]
    assert before["visual_contents"] == after["visual_contents"]
