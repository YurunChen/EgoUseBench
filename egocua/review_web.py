"""Local Flask adapter for the construction-review workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import Flask, redirect, render_template, request, send_file, url_for

from .environment import create_site_runtime, load_site_data
from .experiment import PRIMARY_CONDITION
from .review_workspace import ReviewWorkspace, ReviewWorkspaceError
from .workspace import resolve_within


def create_review_app(workspace: ReviewWorkspace) -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    runtimes: dict[str, Any] = {}
    app.extensions["review_site_runtimes"] = runtimes

    @app.get("/")
    def root() -> Any:
        return redirect(url_for("review_index"))

    @app.get("/review")
    def review_index() -> Any:
        filters = {
            "track": request.args.get("track") or None,
            "site": request.args.get("site") or None,
            "status": request.args.get("status") or None,
        }
        progress = workspace.view(**filters)
        return render_template(
            "review/index.html",
            progress=progress,
            filters=filters,
            message=request.args.get("message"),
        )

    @app.get("/review/task/<task_id>")
    def review_case(task_id: str) -> Any:
        return _render_case(workspace, task_id)

    @app.post("/review/task/<task_id>/submit")
    def submit_case(task_id: str) -> Any:
        response = {
            "decision": request.form.get("decision", ""),
            "note": request.form.get("note", ""),
        }
        try:
            receipt = workspace.submit(
                task_id,
                response,
                expected_revision=request.form.get("revision", ""),
            )
        except ReviewWorkspaceError as exc:
            return _render_case(
                workspace,
                task_id,
                answers=response,
                error=str(exc),
                status=400,
            )
        message = "已保留" if receipt["decision"] == "admit" else "已淘汰"
        return redirect(url_for("review_index", message=f"{task_id}：{message}"))

    @app.get("/review/asset")
    def review_asset() -> Any:
        relative = request.args.get("path", "")
        try:
            path = resolve_within(workspace.root, relative, context="review asset")
        except (ValueError, FileNotFoundError):
            return "未找到审核材料", 404
        return send_file(path, conditional=True)

    @app.get("/review/task/<task_id>/site")
    def open_site(task_id: str) -> Any:
        case = workspace.view(task_id)
        task = case["task"]
        previous = runtimes.pop(task_id, None)
        if previous is not None:
            previous.server.close()
        site_data = load_site_data(workspace.root, task["site"], require_assets=True)
        runtime = create_site_runtime(
            task, PRIMARY_CONDITION[task["track"]],
            site_data,
            asset_root=workspace.root,
        )
        runtime.server.start()
        runtimes[task_id] = runtime
        return redirect(f"{runtime.server.base_url}/search")

    return app


def _render_case(
    workspace: ReviewWorkspace,
    task_id: str,
    *,
    answers: dict[str, Any] | None = None,
    error: str | None = None,
    status: int = 200,
) -> Any:
    case = workspace.view(task_id)
    return render_template(
        "review/case.html",
        case=case,
        answers=answers or {},
        error=error,
    ), status
