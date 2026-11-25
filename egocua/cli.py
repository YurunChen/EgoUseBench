from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

from .frozen_run import FrozenRun
from .checker import check_run
from .environment import EnvironmentError, create_site_runtime, load_site_data
from .experiment import (
    TASK_REVIEW_PLAN,
    ExperimentPlanError,
    expand_probe_runs,
    expand_runs,
    load_experiment_plan,
    validate_run_records,
)
from .failure_attribution import attribute_run
from .gates import validate_task_for_human_run, validate_task_for_model_run
from .lineage import task_recording_group
from .metrics import aggregate, source_intervals
from .schema import (
    ValidationError,
    load_tasks,
    validate_dataset,
)
from .synthesis_workflow import SynthesisWorkflow, SynthesisWorkflowError
from .run_ledger import load_run_record
from .run_inspection import RUN_VIEWS, read_run_view
from .experiment_output import create_experiment_directory
from .experiment_results import score_summary
from .review_web import create_review_app
from .review_workspace import ReviewWorkspace, ReviewWorkspaceError
from .workspace import write_json_atomic


def _review_runtime_paths(port: int) -> tuple[Path, Path]:
    return Path(f"/tmp/egocua-review-{port}.pid"), Path(
        f"/tmp/egocua-review-{port}.log"
    )


def _process_identity(pid: int) -> str | None:
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat=", "-o", "lstart=", "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1 and not result.stdout.strip():
        return None
    if result.returncode:
        raise ReviewWorkspaceError(f"无法核查进程 {pid}：{result.stderr.strip()}")
    state, _, identity = result.stdout.strip().partition(" ")
    if state.startswith("Z"):
        return None
    if not identity.strip():
        raise ReviewWorkspaceError(f"进程 {pid} 缺少启动信息")
    return identity.strip()


def _read_review_process(pidfile: Path) -> dict[str, Any] | None:
    try:
        text = pidfile.read_text().strip()
    except FileNotFoundError:
        return None
    try:
        record = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReviewWorkspaceError(f"审核进程记录损坏：{pidfile}") from exc
    if (
        not isinstance(record, dict)
        or type(record.get("pid")) is not int
        or record["pid"] <= 0
        or not isinstance(record.get("identity"), str)
        or not record["identity"].strip()
    ):
        raise ReviewWorkspaceError(
            f"审核进程记录缺少身份信息，请核查后重新启动：{pidfile}"
        )
    return record


def _review_process_running(record: dict[str, Any]) -> bool:
    identity = _process_identity(record["pid"])
    if identity is None:
        return False
    if identity != record["identity"]:
        raise ReviewWorkspaceError(f"进程 {record['pid']} 已被其他程序使用，不执行停止")
    return True


def _forget_review_process(pidfile: Path, record: dict[str, Any]) -> None:
    if _read_review_process(pidfile) == record:
        pidfile.unlink(missing_ok=True)


def _port_is_listening(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _wait_until_listening(host: str, port: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_is_listening(host, port):
            return True
        time.sleep(0.1)
    return False


def _stop_review_process(port: int) -> str:
    pidfile, _ = _review_runtime_paths(port)
    record = _read_review_process(pidfile)
    if record is None:
        return f"端口 {port} 没有记录中的审核网站"
    pid = record["pid"]
    if not _review_process_running(record):
        _forget_review_process(pidfile, record)
        return f"端口 {port} 的审核网站已不在运行"
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        _forget_review_process(pidfile, record)
        return f"端口 {port} 的审核网站已不在运行"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if _process_identity(pid) != record["identity"]:
            _forget_review_process(pidfile, record)
            return f"已停止端口 {port} 的审核网站 (pid {pid})"
        time.sleep(0.1)
    if _review_process_running(record):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    _forget_review_process(pidfile, record)
    return f"已停止端口 {port} 的审核网站 (pid {pid})"


def _redirect_stdio(logfile: Path) -> None:
    log_fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    if log_fd > 2:
        os.close(log_fd)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def _daemonize_review(pidfile: Path, logfile: Path) -> bool:
    if os.fork() > 0:
        return False
    os.setsid()
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    if os.fork() > 0:
        os._exit(0)
    _redirect_stdio(logfile)
    identity = _process_identity(os.getpid())
    if identity is None:
        raise ReviewWorkspaceError("审核进程未能记录启动身份")
    write_json_atomic(pidfile, {"pid": os.getpid(), "identity": identity})
    return True


def _serve_review(args: argparse.Namespace) -> None:
    if args.stop:
        print(_stop_review_process(args.port))
        return
    if not args.reviewer:
        raise ReviewWorkspaceError("人工审核网站需要 --reviewer")
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ReviewWorkspaceError("人工审核网站只能绑定到本机")
    address = f"http://127.0.0.1:{args.port}/review"
    pidfile, logfile = _review_runtime_paths(args.port)
    existing = _read_review_process(pidfile)
    if (
        existing
        and _review_process_running(existing)
        and _port_is_listening(args.host, args.port)
    ):
        print(f"人工审核网站已在运行：{address}")
        if not args.no_open:
            webbrowser.open(address)
        return
    if _port_is_listening(args.host, args.port):
        raise ReviewWorkspaceError(f"端口 {args.port} 已被占用")
    workspace = ReviewWorkspace.open(
        args.root,
        args.reviewer,
        include_all_tasks=True,
    )
    app = create_review_app(
        workspace, default_view="all" if args.all_tasks else "pending"
    )
    if args.daemon:
        print(f"人工审核网站：{address}")
        if not _daemonize_review(pidfile, logfile):
            if not _wait_until_listening(args.host, args.port):
                raise SystemExit(f"审核网站启动失败，日志：{logfile}")
            if not args.no_open:
                webbrowser.open(address)
            return
    else:
        print(f"人工审核网站：{address}")
        if not args.no_open:
            threading.Timer(0.5, webbrowser.open, args=(address,)).start()
    threading.Thread(
        target=app.extensions["review_session"].current,
        name="review-warmup",
        daemon=True,
    ).start()
    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        if args.daemon:
            recorded = _read_review_process(pidfile)
            if recorded and recorded["pid"] == os.getpid():
                pidfile.unlink(missing_ok=True)


def _task(tasks: list[dict[str, Any]], task_id: str) -> dict[str, Any]:
    try:
        return next(task for task in tasks if task["id"] == task_id)
    except StopIteration as exc:
        raise SystemExit(f"unknown task id: {task_id}") from exc


def _recording_groups(tasks: list[dict[str, Any]], root: str) -> dict[str, str]:
    """Map each task to its recording, falling back to the task itself.

    A scored run may carry only a task snapshot, whose source dossier is no
    longer in the workspace. Such a task counts as its own source; any other
    lineage error still fails, so a broken source cannot inflate independence.
    """
    groups = {}
    for task in tasks:
        try:
            groups[task["id"]] = task_recording_group(task, root=root)
        except FileNotFoundError:
            groups[task["id"]] = task["id"]
    return groups




def _print_run(record: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "final_state": record["final_state"],
                "artifact_directory": record["meta"].get("artifact_directory"),
            },
            ensure_ascii=False,
        )
    )


def _run_browser(args: argparse.Namespace) -> None:
    from .controlled_experiments import select_execution
    from .execution import run_isolated
    from .study_execution import EvaluationSettings, run_exported_evaluation

    run = FrozenRun.resolve(args)
    index = select_execution(
        run.directory,
        {
            "arm": args.arm,
            "task_id": run.task["id"],
            "condition": args.condition,
            "replicate_id": args.replicate_id,
        },
    )
    settings = EvaluationSettings(
        canary_index=index, concurrency=1, execution_indexes=(index,)
    )
    result = run_isolated(
        run_exported_evaluation(run.directory, settings, Path(args.root).resolve())
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_minimal(args: argparse.Namespace) -> None:
    from .adapters.openai_requests import OpenAIRequestsTransport
    from .agents.minimal_loop import run_task

    run = FrozenRun.resolve(args)
    validate_task_for_model_run(run.task, root=args.root)
    _print_run(
        run_task(
            run.task,
            run.plan.condition(args.condition),
            model=run.plan.model,
            schedule_index=run.schedule_index,
            transport=OpenAIRequestsTransport.from_env(args.env_file),
            **run.adapter_settings(),
        )
    )


def _run_human(args: argparse.Namespace) -> None:
    from .adapters.human import run_task

    run = FrozenRun.resolve(args)
    validate_task_for_human_run(run.task, root=args.root)
    _print_run(
        run_task(
            run.task,
            run.plan.condition(args.condition),
            evaluator_id=args.evaluator,
            **run.solvability_settings(),
        )
    )


FROZEN_RUNS = {
    "run-browser": _run_browser,
    "run-minimal": _run_minimal,
    "run-human": _run_human,
}


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except (
        ValidationError,
        ExperimentPlanError,
        EnvironmentError,
        ReviewWorkspaceError,
        SynthesisWorkflowError,
    ) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="egocua")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser(
        "validate", help="validate a track-grouped task directory"
    )
    validate.add_argument("tasks")
    validate.add_argument("--root", default=".")
    validate.add_argument("--plan", default=TASK_REVIEW_PLAN)

    expand = subparsers.add_parser(
        "expand", help="expand tasks into experiment conditions"
    )
    expand.add_argument("tasks")
    expand.add_argument("--plan", default=TASK_REVIEW_PLAN)

    expand_probes = subparsers.add_parser(
        "expand-probes", help="expand tasks into the necessity-probe conditions"
    )
    expand_probes.add_argument("tasks")
    expand_probes.add_argument("--plan", default=TASK_REVIEW_PLAN)

    score = subparsers.add_parser(
        "score", help="check run records and aggregate metrics"
    )
    score.add_argument("tasks")
    score.add_argument("records", nargs="+")
    score.add_argument("--output")
    score.add_argument("--output-root", default="experiments")
    score.add_argument("--experiment-id", default="manual")
    score.add_argument("--plan", default=TASK_REVIEW_PLAN)
    score.add_argument("--root", default=".")

    serve = subparsers.add_parser(
        "serve", help="serve one task website for manual testing"
    )
    serve.add_argument("tasks")
    serve.add_argument("task_id")
    serve.add_argument("condition")
    serve.add_argument("--plan", default=TASK_REVIEW_PLAN)
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--root", default=".")

    run = subparsers.add_parser(
        "run-browser", help="run one registered execution of a frozen experiment"
    )
    minimal = subparsers.add_parser(
        "run-minimal", help="run the diagnostic minimal chat-loop agent"
    )
    minimal.add_argument("--env-file", default=".env")
    human = subparsers.add_parser(
        "run-human", help="diagnostic human execution on a native website"
    )
    human.add_argument("--evaluator", required=True)
    for frozen_run in (run, minimal, human):
        frozen_run.add_argument("task_id")
        frozen_run.add_argument("condition")
        frozen_run.add_argument(
            "--plan", required=True, help="plan.json of an exported experiment"
        )
        frozen_run.add_argument(
            "--arm", required=True, help="condition name in that experiment"
        )
        frozen_run.add_argument("--replicate-id", type=int, default=0)
        frozen_run.add_argument("--root", default=".")

    inspect = subparsers.add_parser(
        "inspect-run", help="summarize one Run Artifact for debugging"
    )
    inspect.add_argument("path", help="Run Artifact directory")
    inspect.add_argument("--view", choices=RUN_VIEWS, default="summary")
    inspect.add_argument(
        "--results", type=Path, help="read an explicitly selected historical score file"
    )

    review = subparsers.add_parser(
        "review", help="open the local construction-review website"
    )
    review.add_argument("--reviewer")
    review.add_argument("--root", default=".")
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8000)
    review.add_argument("--no-open", action="store_true")
    review.add_argument(
        "--daemon",
        action="store_true",
        help="detach so the site keeps running after this command exits",
    )
    review.add_argument(
        "--stop",
        action="store_true",
        help="stop the detached review site on this port",
    )
    review.add_argument(
        "--all-tasks",
        action="store_true",
        help="open the unified review site on the all-task preview by default",
    )

    synthesize = subparsers.add_parser(
        "synthesize", help="inspect and advance the task-synthesis lifecycle"
    )
    synthesis_commands = synthesize.add_subparsers(
        dest="synthesis_command", required=True
    )

    synthesis_status = synthesis_commands.add_parser(
        "status", help="show each task's current status and next action"
    )
    synthesis_status.add_argument("--root", default=".")
    synthesis_status.add_argument("--task-id", action="append", default=[])
    synthesis_status.add_argument(
        "--composite", action="store_true", help="show experimental composite tasks"
    )
    synthesis_status.add_argument(
        "--update-count",
        action="store_true",
        help="show independent update-count variants",
    )
    synthesis_status.add_argument(
        "--summary", action="store_true", help="show counts and next work only"
    )

    synthesis_next = synthesis_commands.add_parser(
        "next", help="return the next required work item without starting model trials"
    )
    synthesis_next.add_argument("--root", default=".")
    synthesis_next.add_argument("--task-id")

    synthesis_admit = synthesis_commands.add_parser(
        "admit", help="install ready candidates without publishing a release"
    )
    synthesis_admit.add_argument("task_id", nargs="+")
    synthesis_admit.add_argument("--root", default=".")

    synthesis_decide = synthesis_commands.add_parser(
        "decide", help="record the current Coding-Agent confidence decision"
    )
    synthesis_decide.add_argument("task_id")
    synthesis_decide.add_argument("verdict", help="one JSON object")
    synthesis_decide.add_argument("--root", default=".")

    for command in subparsers.choices.values():
        destinations = {action.dest for action in command._actions}
        if "plan" in destinations and "arm" not in destinations:
            command.add_argument(
                "--arm", help="condition name in an exported experiment plan"
            )
    args = parser.parse_args(argv)
    if args.command == "inspect-run":
        print(
            json.dumps(
                read_run_view(args.path, args.view, args.results),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.command == "review":
        _serve_review(args)
        return
    if args.command == "synthesize":
        workflow = SynthesisWorkflow(args.root)
        if args.synthesis_command == "status":
            result = workflow.status(
                args.task_id or None,
                composite=args.composite,
                update_count=args.update_count,
            )
            if args.summary:
                result = result["summary"]
        elif args.synthesis_command == "next":
            result = workflow.next_work_item(args.task_id)
        elif args.synthesis_command == "admit":
            result = workflow.admit(args.task_id)
        else:
            try:
                verdict = json.loads(args.verdict)
            except json.JSONDecodeError as exc:
                raise SynthesisWorkflowError(
                    f"confidence verdict is not valid JSON: {exc.msg}"
                ) from exc
            if not isinstance(verdict, dict):
                raise SynthesisWorkflowError(
                    "confidence verdict must be one JSON object"
                )
            result = workflow.decide(args.task_id, verdict)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command in FROZEN_RUNS:
        FROZEN_RUNS[args.command](args)
        return
    if args.command == "score":
        records = [load_run_record(path) for path in args.records]
        if all("task_snapshot" in record for record in records):
            tasks = list(
                {
                    record["task_id"]: record["task_snapshot"] for record in records
                }.values()
            )
        else:
            tasks = load_tasks(args.tasks)
    else:
        tasks = load_tasks(args.tasks)
    plan = load_experiment_plan(args.plan, args.arm)
    if args.arm is not None and args.command == "score":
        args.experiment_id = plan.experiment_id
    if args.command == "validate":
        counts = validate_dataset(tasks, root=args.root)
        print(json.dumps({"tasks": sum(counts.values()), "tracks": counts}, indent=2))
    elif args.command == "expand":
        print(json.dumps(expand_runs(tasks, plan), indent=2))
    elif args.command == "expand-probes":
        print(json.dumps(expand_probe_runs(tasks, plan), indent=2))
    elif args.command == "score":
        task_by_id = {task["id"]: task for task in tasks}
        validate_run_records(tasks, records, plan)
        checked = [
            check_run(task_by_id[record["task_id"]], record) for record in records
        ]
        attributions = []
        for result, record_path, record in zip(
            checked, args.records, records, strict=True
        ):
            attribution = attribute_run(result, record)
            attribution["run_artifact"] = str(record_path)
            attributions.append(attribution)
        report = {
            "runs": [
                {**score_summary(result), "artifact_directory": str(path)}
                for result, path in zip(checked, args.records, strict=True)
            ],
            "metrics": aggregate(checked),
            "source_intervals": source_intervals(
                checked, _recording_groups(tasks, args.root)
            ),
            "failure_attributions": attributions,
        }
        output = (
            Path(args.output)
            if args.output
            else create_experiment_directory(
                args.output_root,
                {
                    "category": "离线评分",
                    "variable": args.experiment_id,
                    "model": "、".join(
                        sorted({record["meta"]["model"] for record in records})
                    ),
                    "run_references": [
                        str(Path(path).resolve()) for path in args.records
                    ],
                    "evaluation_plan": str(Path(args.plan).resolve()),
                },
            )
            / "results.json"
        )
        write_json_atomic(output, report)
        print(
            json.dumps(
                {**report, "report_path": output.as_posix()},
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "serve":
        task = _task(tasks, args.task_id)
        plan.validate_condition(task["track"], args.condition)
        site_data = load_site_data(args.root, task["site"], require_assets=True)
        runtime = create_site_runtime(
            task,
            site_data,
            asset_root=args.root,
            port=args.port,
        )
        print(f"Serving {task['id']} ({task['site']}) at {runtime.server.base_url}")
        try:
            runtime.server.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            runtime.server.httpd.server_close()


if __name__ == "__main__":
    main()
