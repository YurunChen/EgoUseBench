from copy import deepcopy
from subprocess import run

import pytest
from PIL import Image

from egocua.controlled_experiments import (
    execution_schedule,
    export_study,
    load_composite_validation_study,
    load_composite_formal_study,
    model_variant,
    load_frozen_study,
    output_labels,
    read_study,
)
from egocua.development_gate import DevelopmentDecision, DevelopmentEvidence
from egocua.experiment import FORMAL_PLAN
from egocua.synthesis_workflow import (
    COMPOSITE_DRAFTS,
    COMPOSITE_REVIEWS,
    SynthesisWorkflow,
    SynthesisWorkflowError,
)
from egocua.workspace import write_json_atomic
from tests.helpers import materialize_site, plan_payload, products
from tests.test_composite_timeline import ONLINE


@pytest.fixture
def composite_candidate(tmp_path):
    plan = plan_payload()
    plan.update(
        conditions=[ONLINE.as_dict()],
        tracks={"graft": ["online"]},
        runtime={**plan["runtime"], "max_steps": 20, "composite_max_steps": 30},
    )
    del plan["probes"]
    write_json_atomic(tmp_path / FORMAL_PLAN, plan)
    items = products("graft")
    materialize_site(items, tmp_path)
    for product in items:
        image = tmp_path / product["image_path"]
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"asset:{product['item_id']}".encode())

    videos = tmp_path / "videos"
    videos.mkdir()
    run(
        [
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x24:rate=2",
            "-t", "26", "-c:v", "mpeg4", str(videos / "recording.mp4"),
        ],
        check=True,
    )
    Image.new("RGB", (32, 24)).save(tmp_path / "reviewed-opening.jpg")
    write_json_atomic(tmp_path / "source.json", {
        "dataset": {"name": "fixture", "source_root": "videos"},
        "tasks": [{
            "task_id": "parent",
            "source": {"media_id": "recording"},
            "packets": [{"id": "V0", "frame_paths": ["reviewed-opening.jpg"]}],
        }],
    })
    first, second = (product["item_id"] for product in items[:2])
    draft = {
        "id": "RG-01",
        "schema_version": 2,
        "track": "composite",
        "site": "egomart",
        "source": {"dossier": "source.json", "task_id": "parent", "window_seconds": [0, 24]},
        "instructions": {"base": "Match the one I finish with.", "b": "Make it two."},
        "visual_session": {"refresh_onset_seconds": 8, "update_onset_seconds": 16},
        "goal": {
            "target_granularity": "variant",
            "acceptable_states": [{"selected_entity_ids": [second, second], "fields": {}}],
        },
        "truth": {"site_scope_entity_ids": [first, second]},
    }
    review = {
        "reviewer": "fixture reviewer",
        "verdict": "accepted",
        "reviewed_task": deepcopy(draft),
        "evidence": {
            "naturalness": "The task changes the tool and then the requested quantity.",
            "answers": "Both in-scope variants were classified.",
            "operations": "Select two through the real cart.",
            "counterexamples": {"R": "The first tool remains plausible.", "G": "Only one is needed."},
        },
    }
    return draft, review


def test_export_composite_keeps_its_own_sample_and_formal_online_protocol(
    tmp_path, composite_candidate
):
    draft, review = composite_candidate
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    write_json_atomic(tmp_path / COMPOSITE_REVIEWS / "RG-01.json", review)

    arms, evidence, gate = load_composite_validation_study(
        tmp_path, ["RG-01"], model="gemini-3.8-flash"
    )
    path = tmp_path / "experiments/composite-fixture"
    export_study(arms, path, root=tmp_path, evidence=evidence, gate=gate)
    assert read_study(path)["run_purpose"] == "task_validation"
    frozen, _, frozen_gate = load_frozen_study(path)
    assert frozen_gate == gate
    assert frozen[0].tasks == arms[0].tasks
    assert len(frozen[0].tasks[0]["visual_packets"][0]["frame_paths"]) == 8
    assert frozen[0].plan.condition("online") == ONLINE
    assert frozen[0].plan.model == "gemini-3.8-flash"
    assert frozen[0].plan.max_steps == 20
    assert len(execution_schedule(frozen)) == 1
    assert output_labels(arms)["category"] == "任务试跑"


def test_changed_definition_and_incomplete_counterexamples_cannot_export(
    tmp_path, composite_candidate
):
    draft, review = composite_candidate
    draft_path = tmp_path / COMPOSITE_DRAFTS / "RG-01.json"
    review_path = tmp_path / COMPOSITE_REVIEWS / "RG-01.json"
    write_json_atomic(draft_path, draft)
    review["evidence"]["counterexamples"].pop("G")
    write_json_atomic(review_path, review)
    with pytest.raises(SynthesisWorkflowError, match="each removed mechanism"):
        load_composite_validation_study(tmp_path, ["RG-01"], model="gemini-3.8-flash")

    draft["instructions"]["b"] = "Make it three."
    write_json_atomic(draft_path, draft)
    assert SynthesisWorkflow(tmp_path).status(["RG-01"])["tasks"]["RG-01"][
        "next_action"
    ] == "coding_agent_assess"
    with pytest.raises(SynthesisWorkflowError, match="no matching semantic review"):
        load_composite_validation_study(tmp_path, ["RG-01"], model="gemini-3.8-flash")


def test_repeated_composite_needs_a_counterexample_for_every_event(tmp_path, composite_candidate):
    draft, review = composite_candidate
    draft["visual_session"] = {}
    draft["instructions"].pop("b")
    draft["composite_events"] = [
        {"kind": "refresh", "onset_seconds": 8},
        {"kind": "graft", "onset_seconds": 16, "request": "Make it two."},
        {"kind": "refresh", "onset_seconds": 24},
    ]
    review["reviewed_task"] = deepcopy(draft)
    review["evidence"]["counterexamples"] = {
        "R1": "Without the first change the first variant fits.",
        "G1": "Without the request one item is sufficient.",
    }
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    review_path = tmp_path / COMPOSITE_REVIEWS / "RG-01.json"
    write_json_atomic(review_path, review)
    with pytest.raises(SynthesisWorkflowError, match="each removed event"):
        load_composite_validation_study(tmp_path, ["RG-01"], model="gemini-3.8-flash")

    review["evidence"]["counterexamples"]["R2"] = "Without the second change the size is wrong."
    write_json_atomic(review_path, review)
    [arm], _, _ = load_composite_validation_study(tmp_path, ["RG-01"], model="gemini-3.8-flash")
    assert arm.tasks[0]["composite_events"] == draft["composite_events"]


def test_composite_readiness_requires_development_review_and_is_not_formal_membership(
    tmp_path, composite_candidate, monkeypatch
):
    draft, review = composite_candidate
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    workflow = SynthesisWorkflow(tmp_path)
    assert workflow.decide("RG-01", review)["recorded_pass"] == "composite_semantic_review"
    row = workflow.status(composite=True)["tasks"]["RG-01"]
    assert row["status"] == "needs_review"
    assert row["next_action"] == "run_development_trial"

    monkeypatch.setattr(
        DevelopmentEvidence,
        "decision",
        lambda self, task: DevelopmentDecision(None, "current valid trial and trace reviewed"),
    )
    report = workflow.status(["RG-01"])
    assert report["summary"]["ready"] == 1
    assert report["summary"]["in_progress"] == 0
    assert report["tasks"]["RG-01"]["remaining_actions"] == []
    with pytest.raises(SynthesisWorkflowError, match="not formal members"):
        workflow.admit(["RG-01"])


def test_formal_composite_requires_development_qualification(tmp_path, composite_candidate):
    draft, review = composite_candidate
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    write_json_atomic(tmp_path / COMPOSITE_REVIEWS / "RG-01.json", review)
    with pytest.raises(ValueError, match="requires ready tasks"):
        load_composite_formal_study(tmp_path, model="gemini-3.8-flash")


def test_formal_composite_freezes_same_inputs_for_second_model(
    tmp_path, composite_candidate, monkeypatch
):
    from egocua.workspace import read_json_strict
    from egocua.run_materials import wearer_input_paths

    draft, review = composite_candidate
    semantics = {
        "events": {
            "R": {"types": ["invalidate", "reveal"], "reason": "The final tool differs."},
            "G": {"types": ["augment"], "reason": "Two are required."},
        },
        "clarify": None,
        "choice_relation": {"type": "same_choice", "reason": "Both updates affect the tool."},
    }
    review["evidence"]["change_semantics"] = semantics
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    write_json_atomic(tmp_path / COMPOSITE_REVIEWS / "RG-01.json", review)
    monkeypatch.setattr(DevelopmentEvidence, "decision", lambda self, task: DevelopmentDecision(None, "reviewed old 20-step trial"))
    arms, evidence, gate = load_composite_formal_study(tmp_path, model="gemini-3.8-flash")
    assert arms[0].plan.max_steps == 30
    assert arms[0].tasks[0]["source"]["window_seconds"] == [0, 24]
    assert arms[0].tasks[0]["visual_session"]["refresh_onset_seconds"] == 8
    assert arms[0].tasks[0]["visual_session"]["update_onset_seconds"] == 16
    assert evidence == {"change_semantics": {"RG-01": semantics}}
    assert "change_semantics" not in str(arms[0].tasks)
    first = tmp_path / "experiments/first"
    export_study(arms, first, root=tmp_path, evidence=evidence, gate=gate)
    frozen, _, _ = load_frozen_study(first)
    assert "run_purpose" not in read_study(first)
    assert output_labels(frozen)["category"] == "正式评测"
    paths = sorted(wearer_input_paths(frozen[0].tasks, frozen[0].plan))
    assert len(paths) == 25
    original = {path: (first / "inputs" / path).read_bytes() for path in paths}
    for path in paths:
        (tmp_path / path).write_bytes(b"changed live source")
    live_site = tmp_path / "sites/egomart/products.json"
    live_site.write_text('{"broken": true}')
    second = tmp_path / "experiments/second"
    export_study(model_variant(frozen, "deepseek-flash"), second, root=tmp_path, evidence=evidence, gate=gate)
    paired, paired_evidence, _ = load_frozen_study(second)
    assert paired[0].tasks == frozen[0].tasks
    assert paired_evidence == evidence
    assert paired[0].plan.max_steps == 30
    assert paired[0].plan.model == "deepseek-flash"
    assert {
        key: read_json_strict(path) for key, path in paired[0].plan.site_snapshots.items()
    } == {
        key: read_json_strict(path) for key, path in frozen[0].plan.site_snapshots.items()
    }
    assert {path: (second / "inputs" / path).read_bytes() for path in paths} == original
    assert read_json_strict(second / "plan.json")["input_source_assets"] == [str(first / "inputs")]
    from egocua.execution import RunConfig, run_experiment
    from egocua.run_materials import require_frozen_assets

    require_frozen_assets(paired[0].tasks, paired[0].plan)
    entry = execution_schedule(paired)[0]
    config = RunConfig(agent="scripted", asset_root=tmp_path,
                       **paired[0].plan.run_parameters(paired[0].tasks[0], entry["run"]))

    rendered_paths = set()

    def read_frozen_images(context):
        for observation in context.initial_payload["observations"]:
            assert context.observation_images(observation)
            rendered_paths.update(observation["frame_paths"])
        for _ in range(4):
            for event in context.advance():
                if event["type"] == "visual_context_update":
                    assert context.observation_images(event["payload"])
                    rendered_paths.update(event["payload"]["frame_paths"])
        return []

    record = run_experiment(paired[0].tasks[0], paired[0].plan.condition("online"), config, read_frozen_images)
    assert record["meta"]["runtime"]["visual_asset_root"] == str(second / "inputs")
    assert rendered_paths == set(paths)
    later_path = next(path for path in paths if path not in paired[0].tasks[0]["visual_packets"][0]["frame_paths"])
    (second / "inputs" / later_path).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        require_frozen_assets(paired[0].tasks, paired[0].plan)


def test_formal_composite_requires_reviewed_semantics(tmp_path, composite_candidate, monkeypatch):
    draft, review = composite_candidate
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    write_json_atomic(tmp_path / COMPOSITE_REVIEWS / "RG-01.json", review)
    monkeypatch.setattr(DevelopmentEvidence, "decision", lambda self, task: DevelopmentDecision(None, "reviewed trial"))
    with pytest.raises(ValueError, match="missing reviewed change_semantics"):
        load_composite_formal_study(tmp_path, model="gemini-3.8-flash")


def test_model_variant_cannot_change_development_study(tmp_path, composite_candidate):
    draft, review = composite_candidate
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    write_json_atomic(tmp_path / COMPOSITE_REVIEWS / "RG-01.json", review)
    arms, _, _ = load_composite_validation_study(tmp_path, model="gemini-3.8-flash")
    with pytest.raises(ValueError, match="frozen formal composite"):
        model_variant(arms, "deepseek-flash")


def test_composite_statistics_keep_pending_invalid_and_valid_denominators(
    tmp_path, composite_candidate, monkeypatch
):
    from dataclasses import replace
    from egocua.execution import RunConfig, run_experiment
    from egocua.study_summary import summarize_records
    from tests.test_composite_execution import cart_action

    draft, review = composite_candidate
    write_json_atomic(tmp_path / COMPOSITE_DRAFTS / "RG-01.json", draft)
    write_json_atomic(tmp_path / COMPOSITE_REVIEWS / "RG-01.json", review)
    [arm], _, _ = load_composite_validation_study(tmp_path, model="gemini-3.8-flash")
    tasks = [deepcopy(arm.tasks[0]) for _ in range(3)]
    for index, task in enumerate(tasks, 1):
        task["id"] = f"RG-{index:02d}"
    arm = replace(arm, name="composite-formal", variable="composite-formal", tasks=tasks,
                  plan=replace(arm.plan, max_steps=30))
    other = replace(
        arm, name="other-composite", plan=replace(
            arm.plan, experiment_id="other-composite", max_steps=20
        )
    )
    schedule = execution_schedule([arm, other])
    entry = next(entry for entry in schedule if entry["run"]["task_id"] == "RG-01"
                 and entry["run"]["arm"] == arm.name)
    config = RunConfig(agent=arm.plan.adapter, asset_root=tmp_path,
                       **arm.plan.run_parameters(tasks[0], entry["run"]))
    final = tasks[0]["goal"]["acceptable_states"][0]["selected_entity_ids"][0]

    def perform(context):
        context.advance()
        cart_action(context, "add", final)
        context.advance()
        context.advance()
        cart_action(context, "add", final)
        return []

    record = run_experiment(tasks[0], arm.plan.condition("online"), config, perform)
    failed = next(entry for entry in schedule if entry["run"]["task_id"] == "RG-02")
    annotation = {
        "events": {"R": {"types": ["reveal", "invalidate"]}, "G": {"types": ["augment"]}},
        "clarify": None, "choice_relation": {"type": "same_choice"},
    }
    summary = summarize_records([arm, other], [(arm.plan.experiment_id, record)], root=tmp_path,
        evidence={"change_semantics": {task["id"]: annotation for task in tasks}},
        execution_results=[{"execution_index": failed["execution_index"], "run": failed["run"], "valid": False}])
    stats = summary["composite_statistics"][arm.name]
    assert stats["overall"] == {
        "planned": 3, "observed": 2, "valid": 1, "invalid": 1, "pending": 1,
        "success": {"count": 1, "denominator": 1, "unknown": 0, "rate": 1.0},
    }
    other_stats = summary["composite_statistics"][other.name]["overall"]
    assert other_stats["observed"] == other_stats["valid"] == 0
    assert other_stats["pending"] == 3
    assert stats["by_combination_update_count"]["RG"]["2"] == stats["overall"]
    assert stats["by_update_count"]["2"] == stats["overall"]
    assert stats["by_semantic_type"]["R:reveal"] == stats["overall"]
    assert stats["step_budget_reached"]["count"] == 0
    assert stats["actual_update_events"]["R"]["count"] == 1
    assert summary["runs"][0]["composite"]["events"]["R"]["selected_before_update"] is True
