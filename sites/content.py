"""Reusable Flask implementation for frozen search-and-submit websites."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from flask import Flask, request, send_file

from .runtime import SiteRuntime, SiteServer, SiteState, register_observation_routes
from egocua.workspace import resolve_within


@dataclass(frozen=True)
class ContentSiteConfig:
    site: str
    title: str
    entity_key: str
    id_key: str
    detail_path: str
    selection_label: str


def load_content_data(
    root: str | Path,
    config: ContentSiteConfig,
    filename: str,
) -> dict[str, Any]:
    path = Path(root).resolve() / "sites" / config.site / filename
    payload = json.loads(path.read_text())
    entities = payload.get(config.entity_key)
    if (
        payload.get("schema_version") != 1
        or payload.get("site") != config.site
        or not isinstance(entities, list)
        or not entities
    ):
        raise ValueError(f"{path}: invalid {config.site} data")
    seen: set[str] = set()
    for index, entity in enumerate(entities):
        entity_id = entity.get(config.id_key) if isinstance(entity, dict) else None
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError(f"{path}: entity {index} needs {config.id_key}")
        if entity_id in seen:
            raise ValueError(f"{path}: duplicate entity id {entity_id}")
        seen.add(entity_id)
    fields = payload.get("supported_fields", [])
    if not isinstance(fields, list) or any(not isinstance(item, str) for item in fields):
        raise ValueError(f"{path}: supported_fields must be strings")
    return {
        **{key: value for key, value in payload.items() if key != config.entity_key},
        "entities": [
            {
                **{key: value for key, value in entity.items() if key != config.id_key},
                "entity_id": entity[config.id_key],
            }
            for entity in entities
        ],
    }


def create_content_runtime(
    task: dict[str, Any],
    arm: str,
    data: dict[str, Any],
    asset_root: str | Path,
    config: ContentSiteConfig,
    *,
    port: int = 0,
) -> SiteRuntime:
    state = SiteState(task, arm, data, asset_root=Path(asset_root))
    app = create_content_app(state, config)
    return SiteRuntime(state=state, server=SiteServer(state, app, port=port))


def create_content_app(state: SiteState, config: ContentSiteConfig) -> Flask:
    app = Flask(config.site)
    entities = state.data["entities"]
    by_id = {entity["entity_id"]: entity for entity in entities}

    def render(body: str, status: int = 200) -> tuple[str, int, dict[str, str]]:
        return _shell(config, body, len(state.selected_entity_ids)), status, {
            "Content-Type": "text/html; charset=utf-8"
        }

    register_observation_routes(
        app,
        state,
        render,
        _context_page,
        _frame_page,
    )

    @app.get("/thumbnail/<entity_id>")
    def thumbnail(entity_id: str) -> Any:
        entity = by_id.get(entity_id)
        relative_path = _visual_path(entity) if entity else None
        if not isinstance(relative_path, str) or not relative_path:
            return "", 404
        path = resolve_within(
            state.asset_root,
            relative_path,
            context=f"{config.site} thumbnail",
        )
        if not path.is_file():
            return "", 404
        return send_file(path)

    @app.get("/search")
    def search() -> Any:
        query = request.args.get("q", "").strip()
        terms = query.casefold().split()
        matched = [
            entity for entity in entities
            if not terms or all(term in _searchable_text(entity) for term in terms)
        ]
        if terms and matched:
            state.successful_search_seen = True
        state.log("search", query=query, results=len(matched))
        return render(_search_page(config, query, matched))

    def detail(entity_id: str) -> Any:
        entity = by_id.get(entity_id)
        if entity is None:
            return render("<h1>Unknown record</h1>", 404)
        state.log("open_entity", entity_id=entity_id)
        return render(_detail_page(config, entity, entity_id in state.selected_entity_ids))

    app.add_url_rule(
        f"/{config.detail_path}/<entity_id>",
        "detail",
        detail,
        methods=["GET"],
    )

    @app.get("/selection")
    def selection() -> Any:
        selected = [by_id[item] for item in state.selected_entity_ids if item in by_id]
        return render(_selection_page(config, selected, state.fields, state.data))

    @app.post("/select")
    def select() -> Any:
        entity_id = request.form.get("entity_id", "")
        if entity_id not in by_id:
            return render("<h1>Unknown record</h1>", 404)
        state.select(entity_id)
        return render(_selection_page(
            config,
            [by_id[item] for item in state.selected_entity_ids if item in by_id],
            state.fields,
            state.data,
        ))

    @app.post("/deselect")
    def deselect() -> Any:
        entity_id = request.form.get("entity_id", "")
        if entity_id in by_id:
            state.deselect(entity_id)
        return render(_selection_page(
            config,
            [by_id[item] for item in state.selected_entity_ids if item in by_id],
            state.fields,
            state.data,
        ))

    @app.post("/field")
    def set_field() -> Any:
        name = request.form.get("name", "").strip()
        value = request.form.get("value", "").strip()
        if name not in state.data.get("supported_fields", []):
            return render("<h1>Unknown report field</h1>", 400)
        state.set_field(name, value)
        return render(_selection_page(
            config,
            [by_id[item] for item in state.selected_entity_ids if item in by_id],
            state.fields,
            state.data,
        ))

    @app.post("/submit")
    def submit() -> Any:
        if state.event_gate is not None and state.event_gate():
            state.log("submit_blocked")
            return render(_blocked_page(), 409)
        if not state.selected_entity_ids:
            state.log("submit_empty")
            return render("<h1>Nothing selected</h1><p>Select a record before submitting.</p>", 400)
        state.commit()
        return render(_confirmation_page(config, state.submitted_entity_ids, state.submitted_fields))

    return app


def _searchable_text(entity: dict[str, Any]) -> str:
    visible = {key: value for key, value in entity.items() if key != "source"}
    return json.dumps(visible, ensure_ascii=False).casefold()


def _entity_title(entity: dict[str, Any]) -> str:
    return str(entity.get("title") or entity.get("name") or "Untitled record")


def _entity_summary(entity: dict[str, Any]) -> str:
    return str(entity.get("summary") or entity.get("description") or "")


def _visual_path(entity: dict[str, Any]) -> str | None:
    value = entity.get("thumbnail_path") or entity.get("photo_path")
    return value if isinstance(value, str) and value else None


def _media_preview(entity: dict[str, Any], *, detail: bool = False) -> str:
    if not _visual_path(entity):
        return ""
    entity_id = quote(str(entity["entity_id"]))
    css_class = "thumbnail detail-thumbnail" if detail else "thumbnail"
    overlay = '<span class="play" aria-hidden="true">&#9654;</span>' if entity.get(
        "thumbnail_path"
    ) else ""
    return (
        '<div class="thumbnail-wrap">'
        f'<img class="{css_class}" src="/thumbnail/{entity_id}" '
        f'alt="Visual for {html.escape(_entity_title(entity))}">'
        f"{overlay}"
        "</div>"
    )


def _media_metadata(entity: dict[str, Any]) -> str:
    values: list[str] = []
    channel = entity.get("channel_title")
    if channel:
        values.append(html.escape(str(channel)))
    views = entity.get("view_count")
    if isinstance(views, int):
        values.append(f"{views:,} views")
    published = entity.get("published_at")
    if published:
        values.append(html.escape(str(published)))
    duration = entity.get("duration_seconds")
    if isinstance(duration, int) and duration >= 0:
        minutes, seconds = divmod(duration, 60)
        values.append(f"{minutes}:{seconds:02d}")
    return f'<p class="media-meta">{" · ".join(values)}</p>' if values else ""


def _search_page(
    config: ContentSiteConfig, query: str, entities: Iterable[dict[str, Any]],
) -> str:
    cards = "".join(
        '<article class="card">'
        f'{_media_preview(entity)}'
        f"<h2>{html.escape(_entity_title(entity))}</h2>"
        f'{_media_metadata(entity)}'
        f"<p>{html.escape(_entity_summary(entity))}</p>"
        f'<a class="button" href="/{config.detail_path}/{quote(str(entity["entity_id"]))}">Open record</a>'
        "</article>"
        for entity in entities
    )
    return (
        f"<h1>Search {html.escape(config.title)}</h1>"
        f'<p>Results for <b>{html.escape(query or "all records")}</b></p>'
        f'<div class="grid">{cards or "<p>No matching records.</p>"}</div>'
    )


def _detail_page(
    config: ContentSiteConfig, entity: dict[str, Any], selected: bool,
) -> str:
    ignored = {
        "entity_id", "title", "name", "summary", "description", "source", "facts",
        "thumbnail_path", "photo_path", "channel_title", "published_at", "duration_seconds",
        "view_count",
    }
    sections = "".join(
        _render_value(str(key).replace("_", " ").title(), value)
        for key, value in entity.items()
        if key not in ignored
    )
    entity_id = str(entity["entity_id"])
    selection = (
        '<form method="post" action="/deselect">'
        f'<input type="hidden" name="entity_id" value="{html.escape(entity_id)}">'
        '<button type="submit">Remove selection</button></form>'
        if selected else
        '<form method="post" action="/select">'
        f'<input type="hidden" name="entity_id" value="{html.escape(entity_id)}">'
        f'<button type="submit">{html.escape(config.selection_label)}</button></form>'
    )
    return (
        f"{_media_preview(entity, detail=True)}"
        f"<h1>{html.escape(_entity_title(entity))}</h1>"
        f'<p class="muted">Record ID: {html.escape(entity_id)}</p>'
        f"{_media_metadata(entity)}"
        f"<p>{html.escape(_entity_summary(entity))}</p>{sections}{selection}"
    )


def _render_value(label: str, value: Any) -> str:
    if isinstance(value, list):
        rows = "".join(
            _render_value(
                str(
                    item.get("heading") or item.get("label") or label
                    if isinstance(item, dict) else label
                ),
                item.get("text") if isinstance(item, dict) and "text" in item else item,
            )
            for item in value
        )
        return rows
    if isinstance(value, dict):
        body = "".join(
            f"<dt>{html.escape(str(key).replace('_', ' ').title())}</dt>"
            f"<dd>{html.escape(str(child))}</dd>"
            for key, child in value.items()
        )
        return f"<section><h2>{html.escape(label)}</h2><dl>{body}</dl></section>"
    return f"<section><h2>{html.escape(label)}</h2><p>{html.escape(str(value))}</p></section>"


def _selection_page(
    config: ContentSiteConfig,
    entities: list[dict[str, Any]],
    fields: dict[str, str],
    data: dict[str, Any],
) -> str:
    rows = "".join(
        '<div class="selected">'
        f"<b>{html.escape(_entity_title(entity))}</b>"
        '<form method="post" action="/deselect">'
        f'<input type="hidden" name="entity_id" value="{html.escape(str(entity["entity_id"]))}">'
        '<button type="submit">Remove</button></form></div>'
        for entity in entities
    ) or "<p>No records selected.</p>"
    options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name.replace("_", " "))}</option>'
        for name in data.get("supported_fields", [])
    )
    recorded = "".join(
        f"<li><b>{html.escape(name)}</b>: {html.escape(value)}</li>"
        for name, value in fields.items()
    ) or "<li>No report fields entered.</li>"
    return (
        f"<h1>{html.escape(config.selection_label)}</h1>{rows}"
        "<h2>Report fields</h2>"
        f"<ul>{recorded}</ul>"
        '<form method="post" action="/field"><label>Field <select name="name">'
        f"{options}</select></label>"
        '<label>Exact value <input name="value" required></label>'
        '<button type="submit">Set field</button></form>'
        '<form method="post" action="/submit"><button class="primary" type="submit">Submit</button></form>'
    )


def _context_page(observations: list[tuple[int, dict[str, Any]]]) -> str:
    galleries = ""
    for display_index, (packet_index, packet) in enumerate(observations):
        heading = "Wearer's view" if len(observations) == 1 else (
            "Earlier view" if display_index == 0 else "More recent view"
        )
        images = "".join(
            f'<a href="/observation/{packet_index}/image/{image_index}">'
            f'<img src="/observations/{packet_index}/images/{image_index}" alt="wearer view {image_index + 1}"></a>'
            for image_index in range(len(packet["frame_paths"]))
        )
        galleries += f"<section><h2>{heading}</h2><div class=\"frames\">{images}</div></section>"
    return "<h1>What the wearer saw</h1><p>Images are chronological.</p>" + galleries


def _frame_page(observation_index: int, image_index: int) -> str:
    return (
        f"<h1>Wearer view · image {image_index + 1}</h1>"
        f'<img class="full-frame" src="/observations/{observation_index}/images/{image_index}">'
        '<p><a href="/context">Back to context</a></p>'
    )


def _blocked_page() -> str:
    return (
        "<h1>Not submitted yet</h1><p>A new wearer update is arriving. "
        "Review it, then submit again.</p>"
    )


def _confirmation_page(
    config: ContentSiteConfig, entity_ids: list[str], fields: dict[str, str],
) -> str:
    return (
        "<h1>✓ Submission complete</h1>"
        f"<p>{html.escape(config.selection_label)}: {html.escape(', '.join(entity_ids))}</p>"
        f"<p>Fields: {html.escape(json.dumps(fields, ensure_ascii=False, sort_keys=True))}</p>"
    )


def _shell(config: ContentSiteConfig, body: str, selected_count: int) -> str:
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(config.title)}</title>
<style>{_CSS}</style></head><body><header><a href="/context">{html.escape(config.title)}</a>
<form action="/search"><input name="q" placeholder="Search"><button>Search</button></form>
<a href="/selection">Selection ({selected_count})</a></header><nav><a href="/context">Wearer context</a>
<a href="/search">All records</a><a href="/selection">Current selection</a></nav><main>{body}</main>
<footer>Frozen deterministic benchmark environment.</footer></body></html>'''


_CSS = """
*{box-sizing:border-box}body{font-family:Arial,sans-serif;color:#17202a;margin:0;background:#f5f7f9}
a{color:#145da0;text-decoration:none}header{display:flex;align-items:center;gap:24px;background:#17324d;color:white;padding:14px 24px}
header>a{color:white;font-size:20px;font-weight:bold}header form{display:flex;flex:1;max-width:720px}header input{flex:1;padding:9px}
button{padding:8px 14px;border:1px solid #b8c4ce;border-radius:5px;background:white;cursor:pointer}.primary{background:#1769aa;color:white;margin-top:18px}
nav{background:#e7edf2;padding:9px 24px;display:flex;gap:24px}main{max-width:1100px;margin:auto;padding:24px;min-height:75vh;background:white}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}.card{border:1px solid #d9e0e6;padding:16px;border-radius:8px}
.thumbnail-wrap{position:relative;background:#111;aspect-ratio:16/9;overflow:hidden;border-radius:6px}.thumbnail{width:100%;height:100%;object-fit:cover;display:block}.detail-thumbnail{max-height:520px}.play{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,.68);color:white;border-radius:50%;width:52px;height:52px;display:grid;place-items:center;padding-left:3px}.media-meta{color:#657786;font-size:13px}
.button{display:inline-block;border:1px solid #8ca4b8;padding:7px 12px;border-radius:5px}.muted{color:#657786;font-size:12px}
section{border-top:1px solid #e2e8ed;padding-top:8px;margin-top:18px}dt{font-weight:bold}dd{margin:2px 0 10px}.selected{display:flex;gap:16px;align-items:center;margin:8px 0}
label{display:block;margin:8px 0}select,input{padding:7px;margin-left:6px}.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.frames img{width:100%}.full-frame{max-width:100%;width:1100px}
footer{text-align:center;padding:18px;color:#657786}
"""
