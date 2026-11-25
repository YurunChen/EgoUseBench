"""Human solvability checks with complete materials and native website interaction."""

from __future__ import annotations

import secrets
import threading
import time
import webbrowser
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from queue import Queue
from typing import Any

from flask import Flask, jsonify, render_template_string, request
from werkzeug.serving import make_server

from sites.runtime import QuietRequestHandler

from ..execution import RunConfig, RunContext, prepare_visual_task, run_experiment
from ..timeline import Condition, complete_materials_condition


MessageWriter = Callable[[str], Any]
BrowserOpener = Callable[[str], Any]


class _WearerPanel:
    """Complete public materials, without answers or step-wise delivery."""

    def __init__(self, context: RunContext) -> None:
        self.context = context
        self.messages = [context.initial_payload["instruction"]]
        images = []
        for observation in context.initial_payload["observations"]:
            for image in context.observation_images(observation):
                images.append(
                    f'<img src="data:{image.media_type};base64,{image.data}" '
                    f'alt="{escape(image.label, quote=True)}">'
                )
        self.images = "".join(images)

    def add_answer(self, answer: dict[str, Any]) -> None:
        self.messages.append("佩戴者回答：" + answer["payload"])

    def render(self) -> str:
        messages = "".join(f"<p>{escape(message)}</p>" for message in self.messages)
        return (
            messages
            + "<section><h2>完整画面（按源时间排列）</h2>"
            + self.images
            + "</section>"
        )


@dataclass
class _Command:
    action: str
    question: str
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] = field(default_factory=dict)


class _HumanRequestHandler(QuietRequestHandler):
    timeout = 10


class _HumanPage:
    """Local controls transport; only the execution thread changes the protocol."""

    def __init__(self, context: RunContext, session_label: str) -> None:
        self.session_label = session_label
        self.panel = _WearerPanel(context)
        self.commands: Queue[_Command] = Queue()
        self._lock = threading.Lock()
        self._pending: _Command | None = None
        self._state: dict[str, Any] = {}
        self.started_at: float | None = None
        self._token = secrets.token_urlsafe(24)
        app = Flask(__name__)
        template = Path(__file__).with_name("human.html").read_text()

        @app.after_request
        def local_only(response):
            response.headers["Cache-Control"] = "no-store"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                f"frame-src {context.base_url}; frame-ancestors 'none'"
            )
            return response

        @app.get("/")
        def index():
            if self.started_at is None:
                self.started_at = time.time()
            return render_template_string(
                template, site_url=context.base_url, session_token=self._token
            )

        @app.get("/state")
        def state():
            with self._lock:
                return jsonify(self._state)

        @app.post("/command")
        def command():
            data = request.get_json(silent=True)
            if request.headers.get("X-Human-Session") != self._token:
                return jsonify(error="无效的页面会话，请刷新页面。"), 403
            if not isinstance(data, dict):
                return jsonify(error="操作格式不正确。"), 400
            action = data.get("action")
            question = data.get("question", "")
            if not isinstance(action, str) or action not in {
                "ask",
                "finish",
            }:
                return jsonify(error="未知操作。"), 400
            if not isinstance(question, str) or (
                action == "ask" and not question.strip()
            ):
                return jsonify(error="请先填写问题。"), 400
            with self._lock:
                if self._state.get("status") != "active":
                    return jsonify(error="任务已结束。"), 409
                if (
                    self._pending is not None
                    or type(data.get("revision")) is not int
                    or data["revision"] != self._state["revision"]
                ):
                    return jsonify(error="操作已提交，请刷新当前状态。"), 409
                if action == "ask" and not self._state["can_ask"]:
                    return jsonify(error="当前任务不能提问。"), 400
                pending = _Command(action, question.strip())
                self._pending = pending
                self.commands.put(pending)
            pending.done.wait()
            return jsonify(pending.response), (
                500 if "error" in pending.response else 200
            )

        self.server = make_server(
            "127.0.0.1", 0, app, threaded=True, request_handler=_HumanRequestHandler
        )
        self.server.daemon_threads = False
        self.url = f"http://127.0.0.1:{self.server.server_port}/"
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )

    def __enter__(self) -> _HumanPage:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def show(self) -> None:
        context = self.panel.context
        state = {
            "status": "active",
            "session_label": self.session_label,
            "revision": context.current_step,
            "can_ask": context.initial_payload["clarification_available"],
            "evidence": self.panel.render(),
            "notice": "按正常方式操作右侧网站，完成后点击完成任务。",
        }
        context.save_progress()
        self._publish(state)

    def complete(self, record: dict[str, Any]) -> None:
        notice = "已完成并保存。"
        participant_record = {
            "run_id": record["meta"]["run_id"],
            "evaluator_id": record["trace"]["evaluator_id"],
            "steps": record["steps"],
            "actions": record["actions"],
            "live_state": record["live_state"],
            "trace": record["trace"],
        }
        self._publish(
            {
                **self._state,
                "status": "finished",
                "notice": notice,
                "evidence": self.panel.render(),
                "record": participant_record,
            }
        )

    def fail(self, error: BaseException) -> None:
        self._publish(
            {
                **self._state,
                "status": "error",
                "error": str(error),
                "notice": "运行出错，请保留页面并联系实验组织者。",
            }
        )

    def _publish(self, state: dict[str, Any]) -> None:
        with self._lock:
            self._state = state
            if self._pending is not None:
                self._pending.response = state
                self._pending.done.set()
                self._pending = None


def run_task(
    task: dict[str, Any],
    condition: Condition,
    *,
    evaluator_id: str,
    source_max_steps: int = 20,
    replicate_id: int = 0,
    asset_root: str | Path = ".",
    output_root: str | Path | None = None,
    experiment_id: str = "human-validation",
    experiment_directory: str | Path | None = None,
    message_writer: MessageWriter = print,
    browser_opener: BrowserOpener = webbrowser.open,
    session_label: str = "",
    default_release_step: int,
    reference_steps: int | None,
    release_trigger: str = "timeline_step",
    pre_release_max_steps: int | None = None,
    post_release_max_steps: int | None = None,
    site_snapshot: str | Path | None = None,
    frozen_asset_root: str | Path | None = None,
    initial_site_state: dict[str, Any] | None = None,
    source_study: str | None = None,
) -> dict[str, Any]:
    """Collect one human Run Artifact without bypassing task events or the website."""

    evaluator = evaluator_id.strip()
    if not evaluator:
        raise ValueError("evaluator_id must be non-empty")
    config = RunConfig(
        agent="human",
        model="human",
        provider="human",
        model_version="human",
        transport="egocua-human-solvability-v1",
        max_steps=None,
        replicate_id=replicate_id,
        asset_root=asset_root,
        output_root=output_root,
        experiment_id=experiment_id,
        experiment_directory=experiment_directory,
        operator_id=evaluator,
        default_release_step=default_release_step,
        reference_steps=reference_steps,
        release_trigger="timeline_step",
        site_snapshot=site_snapshot,
        frozen_asset_root=frozen_asset_root,
        initial_site_state=initial_site_state,
    )

    page: _HumanPage | None = None
    resources = ExitStack()

    def execute(context: RunContext) -> dict[str, Any]:
        nonlocal page
        page = resources.enter_context(_HumanPage(context, session_label))
        trace: dict[str, Any] = {
            "evaluator_id": evaluator,
            "commands": [],
            "stop_reason": None,
            "protocol": "complete-materials-native-website",
            "source_study": source_study,
            "source_protocol": {
                "condition": condition.as_dict(),
                "max_steps": source_max_steps,
                "release_trigger": release_trigger,
                "pre_release_max_steps": pre_release_max_steps,
                "post_release_max_steps": post_release_max_steps,
            },
            "step_meaning": "control interactions, not website action steps",
        }
        context.capture_partial_trace(trace)
        context.track_website_progress()
        page.show()
        message_writer(f"人工可解性页面：{page.url}")
        browser_opener(page.url)
        while True:
            command = page.commands.get()
            context.advance()
            entry: dict[str, Any] = {
                "step": context.current_step,
                "command": f"ask {command.question}"
                if command.action == "ask"
                else "finish",
                "received_at": time.time(),
            }
            trace["commands"].append(entry)
            if command.action == "finish":
                trace["stop_reason"] = "operator_finished"
                trace["elapsed_seconds"] = (
                    round(time.time() - page.started_at, 1)
                    if page.started_at is not None
                    else None
                )
                context.save_progress()
                return trace
            answer = context.ask_user(command.question)
            entry["answer_event"] = answer
            page.panel.add_answer(answer)
            page.show()

    try:
        record = run_experiment(
            prepare_visual_task(task, condition, config.asset_root),
            complete_materials_condition(condition),
            config,
            execute,
        )
        if page is not None:
            page.complete(record)
        return record
    except BaseException as error:
        if page is not None:
            page.fail(error)
        raise
    finally:
        resources.close()
