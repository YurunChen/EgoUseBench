from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .checker import check_run
from .environment import EnvironmentError, create_site_runtime, load_site_data
from .experiment import (
    ExperimentPlanError,
    expand_runs,
    load_experiment_plan,
    validate_run_records,
)
from .gates import (
    load_validation_evidence,
    validate_release,
    validate_task_for_human_run,
    validate_task_for_model_run,
)
from .metrics import aggregate, bootstrap_intervals
from .schema import (
    ValidationError,
    load_tasks,
    validate_dataset,
    validate_pilot_distribution,
)
from .run_ledger import inspect_run_artifact
from .review_web import create_review_app
from .review_workspace import ReviewWorkspace, ReviewWorkspaceError
from .workspace import resolve_within, write_json_atomic


def _task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    try:
        return next(task for task in tasks if task["id"] == task_id)
    except StopIteration as exc:
        raise SystemExit(f"unknown task id: {task_id}") from exc


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except (
        ValidationError,
        ExperimentPlanError,
        EnvironmentError,
        ReviewWorkspaceError,
    ) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="egocua")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a track-grouped task directory")
    validate.add_argument("tasks")
    validate.add_argument("--root", default=".")
    validate.add_argument("--release-evidence")
    validate.add_argument("--plan", default="configs/pilot.json")
    validate.add_argument(
        "--pilot",
        action="store_true",
        help="require the registered candidate-pilot distribution",
    )

    expand = subparsers.add_parser("expand", help="expand tasks into experiment arms")
    expand.add_argument("tasks")
    expand.add_argument("--plan", default="configs/pilot.json")

    score = subparsers.add_parser("score", help="check run records and aggregate metrics")
    score.add_argument("tasks")
    score.add_argument("records", nargs="+")
    score.add_argument("--output")
    score.add_argument("--artifacts-root", default="artifacts")
    score.add_argument("--experiment-id", default="manual")
    score.add_argument("--plan", default="configs/pilot.json")

    serve = subparsers.add_parser("serve", help="serve one task website for manual testing")
    serve.add_argument("tasks")
    serve.add_argument("task_id")
    serve.add_argument("arm")
    serve.add_argument("--plan", default="configs/pilot.json")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--root", default=".")

    run = subparsers.add_parser("run-browser", help="run the registered Browser-Use production adapter")
    run.add_argument("tasks")
    run.add_argument("task_id")
    run.add_argument("arm")
    run.add_argument("--plan", default="configs/pilot.json")
    run.add_argument("--evidence", required=True)
    run.add_argument("--model", default="gpt-5.4-mini")
    run.add_argument("--replicate-id", type=int, default=0)
    run.add_argument("--schedule-index", type=int)
    run.add_argument("--provider-seed", type=int)
    run.add_argument("--output")
    run.add_argument("--root", default=".")
    run.add_argument("--artifacts-root", default="artifacts")
    run.add_argument("--experiment-id", default="manual")

    minimal = subparsers.add_parser("run-minimal", help="run the diagnostic minimal chat-loop agent")
    minimal.add_argument("tasks")
    minimal.add_argument("task_id")
    minimal.add_argument("arm")
    minimal.add_argument("--plan", default="configs/pilot.json")
    minimal.add_argument("--evidence", required=True)
    minimal.add_argument("--model", default="gpt-5.4-mini")
    minimal.add_argument("--env-file", default=".env")
    minimal.add_argument("--replicate-id", type=int, default=0)
    minimal.add_argument("--schedule-index", type=int)
    minimal.add_argument("--output")
    minimal.add_argument("--root", default=".")
    minimal.add_argument("--artifacts-root", default="artifacts")
    minimal.add_argument("--experiment-id", default="manual")

    human = subparsers.add_parser(
        "run-human", help="collect one exact-harness human validation Run Artifact"
    )
    human.add_argument("tasks")
    human.add_argument("task_id")
    human.add_argument("arm")
    human.add_argument("--plan", default="configs/pilot.json")
    human.add_argument("--evidence", required=True)
    human.add_argument("--evaluator", required=True)
    human.add_argument("--replicate-id", type=int, default=0)
    human.add_argument("--output")
    human.add_argument("--root", default=".")
    human.add_argument("--artifacts-root", default="artifacts")
    human.add_argument("--experiment-id", default="human-validation")

    inspect = subparsers.add_parser("inspect-run", help="summarize one Run Artifact for debugging")
    inspect.add_argument("path", help="Run Artifact directory or a file inside it")

    review = subparsers.add_parser("review", help="open the local construction-review website")
    review.add_argument("--reviewer", required=True)
    review.add_argument("--root", default=".")
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8000)
    review.add_argument("--no-open", action="store_true")
    review.add_argument(
        "--all-tasks",
        action="store_true",
        help="review every constructed task instead of only the primary selection",
    )

    args = parser.parse_args(argv)
    if args.command == "inspect-run":
        print(json.dumps(inspect_run_artifact(args.path), ensure_ascii=False, indent=2))
        return
    if args.command == "review":
        if args.host not in {"127.0.0.1", "localhost"}:
            raise ReviewWorkspaceError("人工审核网站只能绑定到本机")
        workspace = ReviewWorkspace.open(
            args.root,
            args.reviewer,
            include_all_tasks=args.all_tasks,
        )
        app = create_review_app(workspace)
        address = f"http://127.0.0.1:{args.port}/review"
        print(f"人工审核网站：{address}")
        if not args.no_open:
            threading.Timer(0.5, webbrowser.open, args=(address,)).start()
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
        return
    tasks = load_tasks(args.tasks)
    plan = load_experiment_plan(args.plan)
    if args.command == "validate":
        counts = validate_dataset(tasks, root=args.root)
        if args.release_evidence:
            counts = validate_release(
                tasks, load_validation_evidence(args.release_evidence), plan, root=args.root
            )
        if args.pilot:
            validate_pilot_distribution(tasks)
        print(json.dumps({"tasks": sum(counts.values()), "tracks": counts}, indent=2))
    elif args.command == "expand":
        print(json.dumps(expand_runs(tasks, plan), indent=2))
    elif args.command == "score":
        task_by_id = {task["id"]: task for task in tasks}
        records = [json.loads(Path(path).read_text()) for path in args.records]
        validate_run_records(tasks, records, plan)
        checked = [check_run(task_by_id[record["task_id"]], record) for record in records]
        report = {
            "runs": checked,
            "metrics": aggregate(checked),
            "bootstrap_95": bootstrap_intervals(checked),
        }
        output = Path(args.output) if args.output else resolve_within(
            Path(args.artifacts_root).resolve(),
            Path("reports") / args.experiment_id
            / f"score-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json",
            context="score report output", must_exist=False,
        )
        write_json_atomic(output, report)
        print(json.dumps({**report, "report_path": output.as_posix()}, ensure_ascii=False, indent=2))
    elif args.command == "serve":
        task = _task(tasks, args.task_id)
        plan.validate_arm(task["track"], args.arm)
        site_data = load_site_data(args.root, task["site"], require_assets=True)
        runtime = create_site_runtime(
            task, args.arm, site_data, asset_root=args.root, port=args.port,
        )
        print(f"Serving {task['id']} ({task['site']}) at {runtime.server.base_url}")
        try:
            runtime.server.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            runtime.server.httpd.server_close()
    elif args.command == "run-minimal":
        from .adapters.openai_requests import OpenAIRequestsTransport
        from .agents.minimal_loop import run_task as run_minimal
        task = _task(tasks, args.task_id)
        plan.validate_arm(task["track"], args.arm)
        evidence = load_validation_evidence(args.evidence)
        validate_task_for_model_run(
            task, evidence.get(task["id"], {}), plan, root=args.root
        )
        transport = OpenAIRequestsTransport.from_env(args.env_file)
        record = run_minimal(
            task, args.arm, model=args.model, replicate_id=args.replicate_id,
            schedule_index=args.schedule_index, max_steps=plan.max_steps,
            output_path=args.output, asset_root=args.root,
            artifacts_root=args.artifacts_root, experiment_id=args.experiment_id,
            transport=transport,
        )
        print(json.dumps({
            "final_state": record["final_state"],
            "artifact_directory": record["meta"].get("artifact_directory"),
        }, ensure_ascii=False))
    elif args.command == "run-human":
        from .adapters.human import run_task as run_human
        task = _task(tasks, args.task_id)
        plan.validate_arm(task["track"], args.arm)
        evidence = load_validation_evidence(args.evidence)
        validate_task_for_human_run(
            task, evidence.get(task["id"], {}), root=args.root
        )
        record = run_human(
            task,
            args.arm,
            evaluator_id=args.evaluator,
            max_steps=plan.max_steps,
            replicate_id=args.replicate_id,
            output_path=args.output,
            asset_root=args.root,
            artifacts_root=args.artifacts_root,
            experiment_id=args.experiment_id,
        )
        print(json.dumps({
            "final_state": record["final_state"],
            "artifact_directory": record["meta"].get("artifact_directory"),
        }, ensure_ascii=False))
    else:
        from .adapters.browser_use import run_task
        task = _task(tasks, args.task_id)
        plan.validate_arm(task["track"], args.arm)
        if plan.adapter != "browser-use":
            raise ExperimentPlanError(
                f"experiment plan requires adapter {plan.adapter!r}, not 'browser-use'"
            )
        if args.model != plan.model:
            raise ExperimentPlanError(
                f"experiment plan requires model {plan.model!r}, not {args.model!r}"
            )
        scheduled_runs = {
            (run["task_id"], run["arm"], run["replicate_id"]): run["schedule_index"]
            for run in expand_runs(tasks, plan)
        }
        run_key = (task["id"], args.arm, args.replicate_id)
        if run_key not in scheduled_runs or args.schedule_index != scheduled_runs[run_key]:
            raise ExperimentPlanError(
                f"run must use registered schedule_index {scheduled_runs.get(run_key)!r}"
            )
        evidence = load_validation_evidence(args.evidence)
        validate_task_for_model_run(
            task, evidence.get(task["id"], {}), plan, root=args.root
        )
        record = asyncio.run(run_task(
            task, args.arm, model=args.model, replicate_id=args.replicate_id,
            schedule_index=args.schedule_index, max_steps=plan.max_steps,
            provider_seed=args.provider_seed,
            output_path=args.output, asset_root=args.root,
            artifacts_root=args.artifacts_root, experiment_id=args.experiment_id,
        ))
        print(json.dumps({
            "final_state": record["final_state"],
            "artifact_directory": record["meta"].get("artifact_directory"),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
