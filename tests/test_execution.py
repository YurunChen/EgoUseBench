from __future__ import annotations

import asyncio
import errno
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from egocua.checker import check_run
from egocua.failure_attribution import attribute_run
from egocua.execution import (
    RunConfig,
    RunContext,
    exception_details,
    request_rejection_kind,
    run_experiment,
    run_experiment_async,
)
from egocua.run_ledger import read_run_identity, inspect_run_artifact, load_run_record
from egocua.timeline import Condition
from tests.conditions import CONDITIONS
from tests.helpers import products, item_id, materialize_assets, materialize_site, task


@pytest.mark.parametrize(
    "error_number", [errno.ECONNREFUSED, errno.EADDRNOTAVAIL, errno.EMFILE]
)
def test_exception_details_preserves_socket_context_and_errno(error_number):
    import httpcore
    import httpx

    socket_error = OSError(error_number, "offline socket failure")
    connection = OSError("All connection attempts failed")
    connection.__cause__ = socket_error
    core_error = httpcore.ConnectError("All connection attempts failed")
    core_error.__context__ = connection
    error = httpx.ConnectError("All connection attempts failed")
    error.__cause__ = core_error
    details = exception_details(error)
    assert details["cause"]["type"] == "ConnectError"
    assert details["cause"]["context"]["message"] == "All connection attempts failed"
    assert details["cause"]["context"]["cause"]["errno"] == error_number


def test_exception_details_prefers_cause_and_stops_context_cycles():
    error = RuntimeError("outer")
    cause = OSError(errno.EMFILE, "too many files")
    error.__cause__ = cause
    error.__context__ = RuntimeError("unused context")
    cause.__context__ = error
    assert exception_details(error) == {
        "type": "RuntimeError",
        "message": "outer",
        "cause": {"type": "OSError", "message": str(cause), "errno": errno.EMFILE},
    }


@pytest.mark.parametrize(
    "message, expected",
    [
        ("InternalError.Algo.DataInspectionFailed: Input image data", "moderation"),
        ("data_inspection_failed", "moderation"),
        ("PROHIBITED_CONTENT", "moderation"),
        ("content_filter", "moderation"),
        ("maximum context length is 262144 tokens", "context_limit"),
        ("The decoder prompt is longer than the maximum model length", "context_limit"),
        ("Range of input length should be [1, 262144]", "context_limit"),
        ("Connection error", None),
        ("Invalid image", None),
        ("Quota exceeded", None),
    ],
)
def test_request_rejection_classification_uses_explicit_provider_reason(
    message, expected
):
    details = {
        "type": "RemoteRunError",
        "message": "worker failed",
        "cause": {
            "type": "BadRequestError",
            "message": message,
        },
    }
    assert request_rejection_kind(details) == expected


def test_delivered_frames_are_archived_once_per_experiment(tmp_path):
    definition = task("refresh")
    materialize_assets(definition, products("refresh"), tmp_path)
    original = {
        path: (tmp_path / path).read_bytes()
        for packet in definition["visual_packets"]
        for path in packet["frame_paths"]
    }
    config = RunConfig(
        agent="fixture-adapter",
        model="fixture-model",
        default_release_step=2,
        reference_steps=None,
        asset_root=tmp_path,
        output_root=tmp_path / "experiments",
    )
    first_frames = definition["visual_packets"][0]["frame_paths"]

    def execute(context):
        first = context.initial_payload["observations"][0]
        initial_images = context.observation_images(first)
        for path in first_frames:
            (tmp_path / path).write_bytes(b"changed after initial image delivery")
        assert context.observation_images(first) == initial_images
        return []

    record = run_experiment(definition, CONDITIONS["online"], config, execute)
    archive = Path(record["meta"]["runtime"]["visual_asset_root"])
    archived = {
        path.relative_to(archive).as_posix()
        for path in archive.rglob("*")
        if path.is_file()
    }
    assert archived == set(first_frames)
    assert all((archive / path).read_bytes() == original[path] for path in first_frames)
    assert (
        archive
        == Path(record["meta"]["artifact_directory"]).parent.parent / "images/wearer"
    )
    assert "asset_sha256" not in record["meta"]
    assert record["task_snapshot"] == definition

    def reread(context):
        first = context.initial_payload["observations"][0]
        assert [image.encoded_bytes for image in context.observation_images(first)] == [
            original[path] for path in first_frames
        ]
        return []

    same_experiment = replace(config, experiment_directory=archive.parent.parent)
    run_experiment(definition, CONDITIONS["online"], same_experiment, reread)


def _post(base_url: str, path: str, form: dict[str, str] | None = None) -> None:
    payload = urlencode(form or {}).encode()
    urlopen(Request(base_url + path, data=payload, method="POST")).read()


class ExperimentExecutionTest(unittest.TestCase):
    def test_human_solvability_explicitly_uses_full_materials(self) -> None:
        from egocua.timeline import complete_materials_condition

        condition = replace(
            CONDITIONS["online"],
            frame_interval_seconds=1,
            frame_delivery_steps=1,
            batch_seconds=8,
            wearer_frames_per_image=4,
            visual_history="summary",
        )
        human_condition = complete_materials_condition(condition)
        self.assertEqual(human_condition.schedule, "human_solvability")
        self.assertEqual(human_condition.visual_history, "full")
        self.assertEqual(condition.visual_history, "summary")

    def test_context_delivers_events_and_lifecycle_records_the_outcome(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                default_release_step=2,
                asset_root=root,
                output_root=root / "experiments",
                experiment_id="execution-interface",
                reference_steps=4,
            )

            def adapter(context: RunContext) -> list[dict]:
                self.assertEqual(context.current_step, 0)
                self.assertEqual(context.advance(), [])
                urlopen(context.base_url + "/search?q=Product").read()
                events = context.advance()
                self.assertEqual(
                    [event["type"] for event in events], ["instruction_injection"]
                )
                _post(context.base_url, "/cart/add", {"item_id": item_id("graft", 0)})
                _post(context.base_url, "/submit")
                return [{"adapter": "fixture", "status": "complete"}]

            record = run_experiment(
                benchmark_task,
                Condition("release_half", "online", "full", "allowed", 0.5),
                config,
                adapter,
            )
            run_directory = Path(record["meta"]["artifact_directory"])
            manifest = read_run_identity(run_directory)

        self.assertEqual(record["meta"]["event_delivery_steps"], [2])
        self.assertEqual(record["meta"]["reference_steps"], 4)
        self.assertFalse(
            {"task_sha256", "site_data_sha256", "asset_sha256"} & record["meta"].keys()
        )
        self.assertIn("python", record["meta"]["runtime"])
        self.assertEqual(
            record["final_state"]["selected_entity_ids"], [item_id("graft", 0)]
        )
        self.assertTrue(record["final_state"]["submission_completed"])
        self.assertEqual(
            record["trace"], [{"adapter": "fixture", "status": "complete"}]
        )
        self.assertEqual(manifest["status"], "completed")

    def test_context_catalog_matches_the_task_scoped_public_website(self) -> None:
        benchmark_task = task("graft")
        visible_ids = [
            item_id("graft", 0),
            item_id("graft", 1),
            item_id("graft", 2),
        ]
        benchmark_task["truth"] = {"site_scope_entity_ids": visible_ids}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                default_release_step=2,
                reference_steps=None,
                asset_root=root,
            )

            def adapter(context: RunContext) -> list[dict]:
                self.assertEqual(
                    [entity["entity_id"] for entity in context.site_data["entities"]],
                    visible_ids,
                )
                self.assertTrue(
                    all(
                        "image_provenance" not in entity
                        for entity in context.site_data["entities"]
                    )
                )
                return []

            run_experiment(
                benchmark_task,
                CONDITIONS["online"],
                config,
                adapter,
            )

    def test_context_releases_update_after_first_unsubmitted_native_commitment(
        self,
    ) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                default_release_step=12,
                reference_steps=None,
                release_trigger="first_native_commitment",
                asset_root=root,
            )

            def adapter(context: RunContext) -> list[dict]:
                self.assertEqual(context.advance(), [])
                _post(context.base_url, "/cart/add", {"item_id": item_id("graft", 0)})
                events = context.advance()
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["trigger_reason"], "first_native_commitment")
                _post(context.base_url, "/submit")
                return []

            record = run_experiment(
                benchmark_task,
                CONDITIONS["online"],
                config,
                adapter,
            )

        self.assertEqual(record["meta"]["event_delivery_steps"], [2])
        self.assertEqual(
            record["meta"]["runtime"]["release_trigger"],
            "first_native_commitment",
        )

    def test_commitment_trigger_does_not_fall_back_to_timeline_step(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                default_release_step=2,
                reference_steps=None,
                release_trigger="first_native_commitment",
                asset_root=root,
            )

            def adapter(context: RunContext) -> list[dict]:
                self.assertEqual(context.advance(), [])
                self.assertEqual(context.advance(), [])
                self.assertEqual(context.advance(), [])
                _post(context.base_url, "/cart/add", {"item_id": item_id("graft", 0)})
                events = context.advance()
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["step"], 4)
                self.assertEqual(events[0]["trigger_reason"], "first_native_commitment")
                return []

            record = run_experiment(
                benchmark_task,
                CONDITIONS["online"],
                config,
                adapter,
            )

        self.assertEqual(record["meta"]["event_delivery_steps"], [4])

    def test_adapter_failure_preserves_partial_trace_through_the_same_interface(
        self,
    ) -> None:
        benchmark_task = task("clarify")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("clarify"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                default_release_step=2,
                reference_steps=None,
                asset_root=root,
                output_root=root / "experiments",
                experiment_id="execution-failure",
            )

            def adapter(context: RunContext) -> list[dict]:
                context.capture_partial_trace(
                    [{"step": 0, "status": "request_started"}]
                )
                raise RuntimeError("provider unavailable")

            with self.assertRaisesRegex(RuntimeError, "provider unavailable") as caught:
                run_experiment(benchmark_task, CONDITIONS["upfront"], config, adapter)

            run_directory = next(
                (root / "experiments").glob("*/runs/*/run.json")
            ).parent
            manifest = read_run_identity(run_directory)
            record = json.loads((run_directory / "run.json").read_text())
            error = json.loads((run_directory / "run.json").read_text())["archive"][
                "error"
            ]
            self.assertEqual(
                caught.exception.run_artifact,
                {
                    "artifact_directory": str(run_directory.resolve()),
                    "run_id": manifest["run_id"],
                },
            )

        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(record["trace"], [{"step": 0, "status": "request_started"}])
        self.assertFalse(record["run_status"]["valid"])
        self.assertEqual(error["type"], "RuntimeError")

    def test_metadata_preparation_failure_preserves_original_error_and_artifact(
        self,
    ) -> None:
        for asynchronous in (False, True):
            with (
                self.subTest(asynchronous=asynchronous),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                materialize_site(products("graft"), root)
                config = RunConfig(
                    agent="fixture-adapter",
                    model="fixture-model",
                    default_release_step=2,
                    reference_steps=None,
                    release_trigger="first_native_commitment",
                    operator_id="fixture-operator",
                    asset_root=root,
                    output_root=root / "experiments",
                )
                failure = OSError("provenance read failed")
                adapter = Mock()
                with patch("egocua.execution.build_run_meta", side_effect=failure):
                    with self.assertRaises(OSError) as caught:
                        if asynchronous:
                            asyncio.run(
                                run_experiment_async(
                                    task("graft"),
                                    CONDITIONS["online"],
                                    config,
                                    adapter,
                                )
                            )
                        else:
                            run_experiment(
                                task("graft"),
                                CONDITIONS["online"],
                                config,
                                adapter,
                            )

                self.assertIs(caught.exception, failure)
                adapter.assert_not_called()
                manifest_path = next((root / "experiments").glob("*/runs/*/run.json"))
                summary = inspect_run_artifact(manifest_path.parent)
                record = load_run_record(manifest_path.parent)
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(
                    summary["error"],
                    {
                        "type": "OSError",
                        "message": "provenance read failed",
                    },
                )
                self.assertFalse(record["run_status"]["valid"])
                self.assertEqual(record["steps"], 0)
                self.assertEqual(record["actions"], [])
                self.assertEqual(record["meta"]["operator_id"], "fixture-operator")
                self.assertEqual(record["meta"]["site"], "egomart")
                self.assertEqual(
                    record["meta"]["runtime"]["release_trigger"],
                    "first_native_commitment",
                )
                self.assertFalse(
                    {"task_sha256", "site_data_sha256", "asset_sha256"}
                    & record["meta"].keys()
                )

    def test_step_budget_is_enforced_by_the_shared_run_context(self) -> None:
        benchmark_task = task("graft")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            materialize_site(products("graft"), root)
            config = RunConfig(
                agent="fixture-adapter",
                model="fixture-model",
                default_release_step=2,
                reference_steps=None,
                max_steps=1,
                asset_root=root,
            )

            def adapter(context: RunContext) -> list[dict]:
                context.advance()
                context.advance()
                return []

            record = run_experiment(
                benchmark_task,
                CONDITIONS["online"],
                config,
                adapter,
            )
            self.assertTrue(record["run_status"]["valid"])
            self.assertEqual(record["steps"], 1)
            self.assertEqual(record["meta"]["termination_reason"], "step_limit")
            self.assertEqual(
                record["live_state"], {"selected_entity_ids": [], "fields": {}}
            )


@pytest.mark.parametrize("mutation", ["submit", "remove", "field"])
def test_commitment_release_survives_same_step_state_changes(
    tmp_path: Path, mutation: str
) -> None:
    definition = task("graft")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="test",
        model="test",
        default_release_step=12,
        reference_steps=None,
        release_trigger="first_native_commitment",
        asset_root=tmp_path,
    )

    def adapter(context: RunContext) -> list:
        assert context.advance() == []
        if mutation == "field":
            context._state.set_field("quantity", "2")
        else:
            context._state.select(item_id("graft", 0))
            if mutation == "submit":
                context._state.commit()
            else:
                context._state.deselect(item_id("graft", 0))
        events = context.advance()
        assert len(events) == 1
        assert events[0]["trigger_reason"] == "first_native_commitment"
        assert context.advance() == []
        return []

    record = run_experiment(definition, CONDITIONS["online"], config, adapter)

    assert check_run(definition, record)["run_valid"] is True


def test_commitment_on_final_step_remains_a_valid_model_outcome(tmp_path: Path) -> None:
    definition = task("graft")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="test",
        model="test",
        default_release_step=2,
        reference_steps=None,
        release_trigger="first_native_commitment",
        max_steps=4,
        asset_root=tmp_path,
    )

    def adapter(context: RunContext) -> list:
        for _ in range(4):
            context.advance()
        context._state.select(item_id("graft", 2))
        context._state.commit()
        return []

    checked = check_run(
        definition, run_experiment(definition, CONDITIONS["online"], config, adapter)
    )

    assert checked["run_valid"] is True
    assert checked["e2e_success"] is False
    assert checked["premature_submission"] is True


def test_correct_live_selection_without_submission_succeeds_without_failure_candidate(
    tmp_path: Path,
) -> None:
    definition = task("graft")
    expected = definition["goal"]["acceptable_states"][0]
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="test",
        model="test",
        default_release_step=2,
        reference_steps=None,
        asset_root=tmp_path,
        output_root=tmp_path / "experiments",
    )

    def adapter(context: RunContext) -> list:
        context.advance()
        for entity_id in expected["selected_entity_ids"]:
            context._state.select(entity_id)
        for name, value in expected["fields"].items():
            context._state.set_field(name, value)
        return []

    record = run_experiment(definition, CONDITIONS["upfront"], config, adapter)
    persisted = load_run_record(record["meta"]["artifact_directory"])
    checked = check_run(definition, persisted)
    candidate = attribute_run(checked, persisted)

    assert (
        persisted["live_state"]["selected_entity_ids"]
        == expected["selected_entity_ids"]
    )
    assert persisted["final_state"]["selected_entity_ids"] == []
    assert checked["goal_state_correct"] is True
    assert checked["e2e_success"] is True
    assert candidate["candidate_stage"] is None


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["complete", "step_limit", "error", "cancelled"])
def test_sync_and_async_runs_preserve_the_same_terminal_trace(
    tmp_path: Path,
    asynchronous: bool,
    outcome: str,
) -> None:
    definition = task("graft")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="fixture",
        model="fixture",
        default_release_step=2,
        reference_steps=None,
        max_steps=1,
        asset_root=tmp_path,
        output_root=tmp_path / "experiments",
    )
    trace = [{"step": 1, "status": "completed observation"}]

    def adapter(context: RunContext) -> list[dict]:
        context.advance()
        context.capture_partial_trace(trace)
        if outcome == "step_limit":
            context.advance()
        if outcome == "error":
            raise RuntimeError("provider unavailable")
        if outcome == "cancelled":
            raise asyncio.CancelledError("run cancelled")
        return trace

    async def async_adapter(context: RunContext) -> list[dict]:
        return adapter(context)

    def execute() -> dict:
        if asynchronous:
            return asyncio.run(
                run_experiment_async(
                    definition,
                    CONDITIONS["upfront"],
                    config,
                    async_adapter,
                )
            )
        return run_experiment(definition, CONDITIONS["upfront"], config, adapter)

    if outcome in {"error", "cancelled"}:
        error_type = RuntimeError if outcome == "error" else asyncio.CancelledError
        with pytest.raises(error_type):
            execute()
        directory = next((tmp_path / "experiments").glob("*/runs/*/run.json")).parent
        record = load_run_record(directory)
    else:
        record = execute()

    assert (
        record["trace"],
        record["steps"],
        record["run_status"]["valid"],
        record["meta"].get("termination_reason"),
    ) == (
        trace,
        1,
        outcome in {"complete", "step_limit"},
        "step_limit" if outcome == "step_limit" else None,
    )


def test_task_validation_purpose_is_preserved_before_a_provider_failure(tmp_path):
    definition = task("graft")
    materialize_site(products("graft"), tmp_path)
    config = RunConfig(
        agent="fixture-adapter",
        model="fixture-model",
        default_release_step=4,
        reference_steps=None,
        asset_root=tmp_path,
        output_root=tmp_path / "experiments",
        experiment_id="task-validation",
        run_purpose="task_validation",
    )

    def adapter(context):
        context.advance()
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError, match="provider unavailable"):
        run_experiment(definition, CONDITIONS["online"], config, adapter)

    directory = next((tmp_path / "experiments").glob("*/runs/*/run.json")).parent
    record = load_run_record(directory)
    assert (
        record["run_status"],
        record["meta"]["run_purpose"],
        record["meta"]["used_for_task_repair"],
        record["meta"]["valid_as_held_out_difficulty_evidence"],
    ) == (
        {"valid": False, "reason": "infrastructure_error"},
        "task_validation",
        True,
        False,
    )


if __name__ == "__main__":
    unittest.main()
