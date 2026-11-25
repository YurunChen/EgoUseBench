"""Flask routes for the project site."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from flask import Flask, render_template

from website.content import SITE
from website.replay import validate_replay_bundle

ROOT = Path(__file__).resolve().parent
SAMPLE_ROOT = ROOT / "static" / "samples"
REQUIRED_KEYS = (
    "title",
    "paper_title",
    "abstract",
    "authors",
    "lab",
    "lab_url",
    "affiliation",
    "code_url",
    "dataset_url",
    "run_command",
    "stats",
    "means",
    "settings",
    "task_types",
    "domains",
    "main_eval",
    "leaderboard",
    "citation",
    "source_note",
    "player",
    "data",
)
LEADERBOARD_SIZE = 6


def validate_content(content: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_KEYS if key not in content]
    if missing:
        raise ValueError("website content is missing " + ", ".join(missing))
    if not str(content["dataset_url"]).strip():
        raise ValueError("website content dataset_url is empty")
    if not content["leaderboard"] and not content["means"]:
        return
    if len(content["leaderboard"]) != LEADERBOARD_SIZE or len(content["means"]) != 3:
        raise ValueError(
            f"website results require {LEADERBOARD_SIZE} models and three setting means"
        )


def create_app(content: Mapping[str, Any] | None = None) -> Flask:
    page = SITE if content is None else content
    validate_content(page)
    samples = page["data"]["samples"]
    has_replays = bool(samples) and SAMPLE_ROOT.exists()
    if has_replays:
        validate_replay_bundle(samples, SAMPLE_ROOT)
    app = Flask(
        __name__,
        template_folder=str(ROOT / "templates"),
        static_folder=str(ROOT / "static"),
    )
    app.config["WEBSITE_CONTENT"] = page

    def show(template: str, **extra: Any):
        return render_template(template, content=page, has_replays=has_replays, **extra)

    @app.get("/")
    def home():
        return show("home.html")

    @app.get("/benchmark")
    def benchmark():
        return show("benchmark.html")

    @app.get("/leaderboard")
    def leaderboard():
        return show("leaderboard.html")

    @app.get("/docs")
    def docs():
        return show("docs.html")

    @app.get("/data")
    def data():
        return show("data.html")

    @app.errorhandler(404)
    def page_not_found(_error: Exception):
        return render_template("not_found.html", content=page), 404

    return app


def main() -> None:
    port = int(os.environ.get("WEBSITE_PORT", "5050"))
    create_app().run(host="127.0.0.1", port=port)
