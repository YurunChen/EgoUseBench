"""Local Flask adapter for the construction-review workspace."""

from __future__ import annotations

from collections import Counter
from typing import Any

from flask import Flask, g, jsonify, redirect, render_template, request, send_file, url_for

from .answer_validation import AnswerValidationWorkspace
from .confidence import high_confidence_sample
from .environment import create_site_runtime, load_site_data, site_data_request
from .failure_review import FailureReviewError, FailureReviewWorkspace
from .review_workspace import ReviewWorkspace, ReviewWorkspaceError
from .sample_review import SampleReviewWorkspace
from .sample_review_web import register_sample_review
from .review_session import ReviewSession
from .synthesis_workflow import SynthesisWorkflow, TASK_STATUS_LABELS, TaskStatus
from .timeline import composite_input
from .update_count import compile_variant, load_cases
from .workspace import read_json_strict, resolve_within


def create_review_app(
    workspace: ReviewWorkspace, *, default_view: str | None = None
) -> Flask:
    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    runtimes: dict[str, Any] = {}
    app.extensions["review_site_runtimes"] = runtimes
    default_view = default_view or ("all" if workspace.scope == "all" else "pending")
    register_sample_review(app, SampleReviewWorkspace(workspace.root, workspace.reviewer_id))
    session = ReviewSession(workspace)
    app.extensions["review_session"] = session

    @app.before_request
    def begin_site_data_request() -> None:
        scope = site_data_request(workspace.root)
        scope.__enter__()
        g.site_data_scope = scope

    @app.teardown_request
    def end_site_data_request(error: BaseException | None) -> None:
        scope = g.pop("site_data_scope", None)
        if scope is not None:
            scope.__exit__(
                type(error) if error is not None else None,
                error,
                error.__traceback__ if error is not None else None,
            )

    def current_workspaces() -> tuple[dict[str, ReviewWorkspace], dict[str, Any]]:
        return session.current()

    def current_portal(
        workspaces: dict[str, ReviewWorkspace],
        lifecycle: dict[str, Any],
    ) -> tuple[dict[str, Any], frozenset[str]]:
        all_workspace = workspaces["all"]
        pending_ids = frozenset(workspaces["pending"].ordered_ids)
        failure_progress = FailureReviewWorkspace.open(
            workspace.root, workspace.reviewer_id
        ).overview()
        confidence_rows = lifecycle["review_evidence"]
        quality_control = high_confidence_sample(
            {
                task_id: all_workspace.tasks[task_id]
                for task_id in all_workspace.ordered_ids
            },
            {
                task_id: row
                for task_id, row in confidence_rows.items()
                if task_id in lifecycle["tasks"]
                and (
                    lifecycle["tasks"][task_id]["status"] == TaskStatus.READY
                    or lifecycle["tasks"][task_id]["next_action"]
                    == "resolve_source_review"
                )
            },
        )
        return {
            "all_total": len(all_workspace.ordered_ids),
            "pending_total": len(pending_ids),
            "track_counts": dict(
                Counter(
                    all_workspace.tasks[task_id]["track"]
                    for task_id in all_workspace.ordered_ids
                )
            ),
            "status_counts": lifecycle["summary"]["status_counts"],
            "status_labels": TASK_STATUS_LABELS,
            "task_statuses": lifecycle["tasks"],
            "pulse_total": failure_progress["total"],
            "pulse_todo": failure_progress["todo"],
            "pulse_blocked": failure_progress["source_blocked"],
            "quality_control": quality_control,
        }, pending_ids

    def requested_view() -> str:
        view = request.args.get("view", default_view)
        if view not in {"all", "pending", "quality"}:
            raise ReviewWorkspaceError(f"unknown review view: {view}")
        return view

    @app.errorhandler(ReviewWorkspaceError)
    def review_workspace_error(exc: ReviewWorkspaceError) -> tuple[str, int]:
        message = str(exc)
        status = 404 if message.startswith("unknown review task:") else 409
        return message, status

    @app.errorhandler(FailureReviewError)
    def failure_review_error(exc: FailureReviewError) -> tuple[str, int]:
        message = str(exc)
        status = 404 if message.startswith("unknown PULSE review run:") else 409
        return message, status

    @app.context_processor
    def inject_reviewer() -> dict[str, str]:
        return {
            "workspace_reviewer": workspace.reviewer_id,
            "current_review_view": requested_view(),
        }

    @app.get("/")
    def root() -> Any:
        return redirect(url_for("review_index"))

    @app.get("/review")
    def review_index() -> Any:
        view = requested_view()
        if view == "quality":
            return redirect(
                url_for("review_index", view="all", _anchor="quality-control")
            )
        workspaces, lifecycle = current_workspaces()
        portal, pending_ids = current_portal(workspaces, lifecycle)
        filters = {
            "track": request.args.get("track") or None,
            "site": request.args.get("site") or None,
            "status": request.args.get("status") or None,
        }
        progress = workspaces[view].view(**filters, lifecycle=lifecycle)
        return render_template(
            "review/index.html",
            progress=progress,
            filters=filters,
            review_view=view,
            portal=portal,
            pending_ids=pending_ids,
            message=request.args.get("message"),
        )

    @app.get("/review/task/<task_id>")
    def review_case(task_id: str) -> Any:
        view = requested_view()
        workspaces, lifecycle = current_workspaces()
        portal, pending_ids = current_portal(workspaces, lifecycle)
        if view == "quality" and task_id not in portal["quality_control"]["task_ids"]:
            return "这道任务不在当前高置信抽查清单中", 403
        return _render_case(
            workspaces[view],
            task_id,
            lifecycle=lifecycle,
            review_view=view,
            portal=portal,
            is_pending_task=task_id in pending_ids,
        )

    @app.get("/review/update-count")
    def update_count_index() -> Any:
        cases = load_cases(workspace.root)
        lifecycle = SynthesisWorkflow(workspace.root).status(update_count=True)
        rows = []
        for case in cases:
            original = read_json_strict(workspace.root / case["source_task_path"])
            variant = compile_variant(original, case)
            rows.append({
                **case, "opening": original["instructions"]["base"],
                "original_events": composite_input(original).updates,
                "variant_events": composite_input(variant).updates,
                "lifecycle": lifecycle["tasks"].get(case["id"]),
                "review": lifecycle["review_evidence"].get(case["id"], {}),
                "final_state_count": len(original["goal"]["acceptable_states"]),
            })
        return render_template(
            "review/update_count.html", cases=rows,
            summary=lifecycle["summary"], status_labels=TASK_STATUS_LABELS,
        )

    @app.post("/review/task/<task_id>/submit")
    def submit_case(task_id: str) -> Any:
        view = requested_view()
        if view not in {"pending", "quality"}:
            return "预览不能提交。请切换到人工审核或高置信抽查后提交意见。", 403
        workspaces, lifecycle = current_workspaces()
        portal, _ = current_portal(workspaces, lifecycle)
        if view == "quality" and task_id not in portal["quality_control"]["task_ids"]:
            return "这道任务不在当前高置信抽查清单中", 403
        active_workspace = workspaces[view]
        response = {
            "decision": request.form.get("decision", ""),
            "note": request.form.get("note", ""),
            **({"purpose": "quality_control"} if view == "quality" else {}),
        }
        try:
            receipt = active_workspace.submit(
                task_id,
                response,
                expected_revision=request.form.get("revision", ""),
                lifecycle=lifecycle,
            )
        except ReviewWorkspaceError as exc:
            return _render_case(
                active_workspace,
                task_id,
                lifecycle=lifecycle,
                review_view=view,
                portal=portal,
                is_pending_task=view == "pending",
                answers=response,
                error=str(exc),
                status=400,
            )
        message = (
            "抽查已记录，录取状态不变"
            if view == "quality" and receipt["decision"] == "admit"
            else "已保留"
            if receipt["decision"] == "admit"
            else "已进入修复"
        )
        return redirect(
            url_for(
                "review_index",
                view="all" if view == "quality" else "pending",
                message=f"{task_id}：{message}",
            )
        )

    @app.get("/review/asset")
    def review_asset() -> Any:
        relative = request.args.get("path", "")
        try:
            path = resolve_within(workspace.root, relative, context="review asset")
        except (ValueError, FileNotFoundError):
            return "未找到审核材料", 404
        media_types = {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".gif",
            ".mp4",
            ".webm",
            ".mov",
            ".mkv",
            ".avi",
        }
        archive = any(
            path.is_relative_to(workspace.root / folder)
            for folder in ("experiments", "artifacts")
        ) and path.suffix in {
            ".json",
            ".jsonl",
        }
        if not path.is_file() or (
            path.suffix.lower() not in media_types and not archive
        ):
            return "未找到审核材料", 404
        response = send_file(path, conditional=True)
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/review/task/<task_id>/site")
    def open_site(task_id: str) -> Any:
        workspaces, lifecycle = current_workspaces()
        case = workspaces[requested_view()].view(task_id, lifecycle=lifecycle)
        task = case["task"]
        for previous in runtimes.values():
            previous.server.close()
        runtimes.clear()
        site_data = load_site_data(workspace.root, task["site"], require_assets=True)
        runtime = create_site_runtime(
            task,
            site_data,
            asset_root=workspace.root,
        )
        runtime.server.start()
        runtimes[task_id] = runtime
        return redirect(f"{runtime.server.base_url}/search")

    @app.get("/review/failures")
    def failure_index() -> Any:
        failure_workspace = FailureReviewWorkspace.open(
            workspace.root, workspace.reviewer_id, scope=request.args.get("scope", "all"),
        )
        return render_template(
            "review/failure_index.html",
            progress=failure_workspace.overview(),
        )

    @app.get("/review/failure/<run_id>")
    def failure_case(run_id: str) -> Any:
        failure_workspace = FailureReviewWorkspace.open(
            workspace.root, workspace.reviewer_id, scope=request.args.get("scope", "all"),
        )
        return render_template(
            "review/failure_case.html",
            case=failure_workspace.case(run_id),
            error=None,
        )

    @app.post("/review/failure/<run_id>/submit")
    def submit_failure(run_id: str) -> Any:
        failure_workspace = FailureReviewWorkspace.open(
            workspace.root, workspace.reviewer_id, scope=request.args.get("scope", "all"),
        )
        raw_stage = request.form.get("pulse_stage", "")
        raw_step = request.form.get("first_decisive_step", "").strip()
        try:
            if raw_step and (not raw_step.isdecimal()):
                raise FailureReviewError("关键步骤必须是非负整数")
            failure_workspace.submit(
                run_id,
                raw_stage or None,
                request.form.get("evidence", ""),
                first_decisive_step=int(raw_step) if raw_step else None,
                run_verdict=request.form.get("run_verdict", "valid_model_failure"),
            )
        except FailureReviewError as exc:
            return render_template(
                "review/failure_case.html",
                case=failure_workspace.case(run_id),
                error=str(exc),
            ), 400
        return redirect(url_for("failure_index", scope=request.args.get("scope", "all")))

    return app


def _render_case(
    workspace: ReviewWorkspace,
    task_id: str,
    *,
    lifecycle: dict[str, Any],
    review_view: str,
    portal: dict[str, Any],
    is_pending_task: bool,
    answers: dict[str, Any] | None = None,
    error: str | None = None,
    status: int = 200,
) -> Any:
    case = workspace.view(task_id, lifecycle=lifecycle)
    uncertainty_reasons = (
        lifecycle["review_evidence"][task_id]["uncertainty_reasons"]
        if lifecycle["tasks"][task_id]["next_action"] == "human_review"
        else []
    )
    return render_template(
        "review/case.html",
        case=case,
        review_view=review_view,
        portal=portal,
        is_pending_task=is_pending_task,
        uncertainty_reasons=uncertainty_reasons,
        answers=answers or {},
        error=error,
    ), status


def create_answer_validation_app(
    workspace: AnswerValidationWorkspace, *, preview: bool = False
) -> Flask:
    """Serve participant-only choices, without the answer-bearing review routes."""
    import base64
    import secrets

    from .frame_inputs import observation_images

    app = Flask(__name__)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    token = secrets.token_urlsafe(24)

    @app.after_request
    def private_materials(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; frame-ancestors 'none'; form-action 'self'"
        )
        return response

    @app.errorhandler(ValueError)
    def invalid_input(error):
        return {"error": str(error)}, 400

    @app.get("/")
    def validation_index():
        return render_template(
            "review/answer_validation_index.html", rows=workspace.progress(),
            participant=workspace.participant, preview=preview,
        )

    @app.get("/task/<int:item>")
    def validation_case(item):
        case = workspace.case(item)
        workspace.open(item)
        paths = case.assets()
        observations = case.session.initial_payload()["observations"]
        images = []
        for batch, observation in enumerate(observations):
            packing = observation.get("wearer_frames_per_image", 1)
            for start in range(0, len(observation["frame_paths"]), packing):
                timestamp = observation["timestamps_seconds"][start]
                images.append({
                    "batch": batch, "image": start // packing,
                    "label": f"源时间 {timestamp:g} 秒起",
                })
        return render_template(
            "review/answer_validation.html", item=item, total=len(workspace.rows),
            participant=workspace.participant, preview=preview,
            instruction=case.session.initial_payload()["instruction"], images=images,
            entities=case.public_site["entities"], fields=case.public_site["supported_fields"],
            asset_indexes={path: index for index, path in enumerate(paths)},
            can_ask=case.session.timeline.offer is not None,
            record=workspace.public_record(item), session_token=token,
        )

    @app.get("/task/<int:item>/frame/<int:batch>/<int:image>")
    def validation_frame(item, batch, image):
        case = workspace.case(item)
        observations = case.session.initial_payload()["observations"]
        if not 0 <= batch < len(observations):
            return "未找到画面", 404
        rendered = observation_images(observations[batch], case.asset_root)
        if not 0 <= image < len(rendered):
            return "未找到画面", 404
        result = rendered[image]
        return app.response_class(base64.b64decode(result.data), mimetype=result.media_type)

    @app.get("/task/<int:item>/asset/<int:index>")
    def validation_asset(item, index):
        return send_file(workspace.candidate_asset(item, index))

    @app.post("/task/<int:item>")
    def validation_update(item):
        if request.headers.get("X-Validation-Session") != token:
            return {"error": "无效的页面会话，请刷新页面"}, 403
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return {"error": "操作格式不正确"}, 400
        return workspace.update(item, data)

    @app.get("/export")
    def validation_export():
        records = [workspace.public_record(row["item"]) for row in workspace.progress() if row["started"]]
        response = jsonify({"participant": workspace.participant, "preview": preview, "records": records})
        response.headers["Content-Disposition"] = 'attachment; filename="human-answer-validation.json"'
        return response

    return app
