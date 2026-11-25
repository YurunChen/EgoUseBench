"""Expose sampled quality review inside the existing local review website."""

from __future__ import annotations

import base64
import secrets
from collections import Counter

from flask import jsonify, redirect, render_template, request, send_file, url_for

from .frame_inputs import observation_images
from .final_state import goal_entity_ids
from .sample_review import DECISIONS, SampleReviewWorkspace
from .workspace import resolve_within


def register_sample_review(app, workspace: SampleReviewWorkspace) -> None:
    token = secrets.token_urlsafe(24)

    @app.get("/review/sample")
    def sample_review_index():
        return render_template(
            "review/sample_index.html",
            rows=workspace.progress(),
            decisions=DECISIONS,
            reviewer=workspace.reviewer,
        )

    @app.route("/review/sample/<int:item>", methods=["GET", "POST"])
    def sample_review_case(item):
        try:
            case = workspace.materials.case(item)
            error = None
            if request.method == "POST":
                if not secrets.compare_digest(request.form.get("token", ""), token):
                    return "页面已过期，请刷新后继续", 403
                try:
                    workspace.submit(
                        item,
                        request.form.get("decision", ""),
                        request.form.get("note", ""),
                    )
                except ValueError as exc:
                    error = str(exc)
                else:
                    return redirect(
                        url_for("sample_review_case", item=item + 1)
                        if item < len(workspace.materials.rows)
                        else url_for("sample_review_index")
                    )
            payload = case.session.initial_payload()
            images = [
                {
                    "batch": batch,
                    "image": start // observation.get("wearer_frames_per_image", 1),
                    "label": f"源时间 {observation['timestamps_seconds'][start]:g} 秒起",
                }
                for batch, observation in enumerate(payload["observations"])
                for start in range(
                    0,
                    len(observation["frame_paths"]),
                    observation.get("wearer_frames_per_image", 1),
                )
            ]
            entities = {e["entity_id"]: e for e in case.public_site["entities"]}
            states = [
                {
                    "items": [
                        (entities[eid], count)
                        for eid, count in Counter(state["selected_entity_ids"]).items()
                    ],
                    "fields": state["fields"],
                }
                for state in case.task["goal"]["acceptable_states"]
            ]
            goal_ids = goal_entity_ids(case.task["goal"])
            offer = case.session.timeline.offer
            return render_template(
                "review/sample_case.html",
                item=item,
                total=len(workspace.materials.rows),
                case=case,
                states=states,
                images=images,
                instruction=payload["instruction"],
                answer_entities=[e for eid, e in entities.items() if eid in goal_ids],
                other_entities=[
                    e for eid, e in entities.items() if eid not in goal_ids
                ],
                answer=offer.canonical_answer if offer else None,
                assets={path: index for index, path in enumerate(case.assets())},
                decisions=DECISIONS,
                reviewer=workspace.reviewer,
                token=token,
                record=workspace.record(item),
                error=error,
                selected=request.form.get("decision", ""),
                note=request.form.get("note", ""),
            ), 400 if error else 200
        except ValueError as exc:
            return str(exc), 400

    @app.get("/review/sample/<int:item>/frame/<int:batch>/<int:image>")
    def sample_review_frame(item, batch, image):
        try:
            case = workspace.materials.case(item)
        except ValueError as exc:
            return str(exc), 404
        observations = case.session.initial_payload()["observations"]
        if not 0 <= batch < len(observations):
            return "未找到画面", 404
        rendered = observation_images(observations[batch], case.asset_root)
        if not 0 <= image < len(rendered):
            return "未找到画面", 404
        frame = rendered[image]
        return app.response_class(
            base64.b64decode(frame.data), mimetype=frame.media_type
        )

    @app.get("/review/sample/<int:item>/asset/<int:index>")
    def sample_review_asset(item, index):
        try:
            case = workspace.materials.case(item)
        except ValueError as exc:
            return str(exc), 404
        paths = case.assets()
        if not 0 <= index < len(paths):
            return "未找到图片", 404
        return send_file(
            resolve_within(case.asset_root, paths[index], context="核查图片")
        )

    @app.get("/review/sample/export")
    def sample_review_export():
        records = [row["record"] for row in workspace.progress() if row["record"]]
        response = jsonify({"reviewer": workspace.reviewer, "records": records})
        response.headers["Content-Disposition"] = (
            'attachment; filename="task-quality-review.json"'
        )
        return response
