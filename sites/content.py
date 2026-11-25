"""Reusable Flask implementation for frozen search-and-submit websites."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from flask import Flask, request, send_file

from . import content_chrome
from ._paths import resolve_within
from .runtime import ActionTrace, SiteRuntime, SiteServer, SiteState
from .search import flatten_public_text, rank_records


@dataclass(frozen=True)
class ContentSiteConfig:
    site: str
    title: str
    entity_key: str
    id_key: str
    detail_path: str
    selection_label: str
    searchable_keys: tuple[str, ...] = ("title", "channel_title", "kind", "name")
    search_mode: str = "all_terms"


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
    if not isinstance(fields, list) or any(
        not isinstance(item, str) for item in fields
    ):
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
    data: dict[str, Any],
    asset_root: str | Path,
    config: ContentSiteConfig,
    *,
    trace: ActionTrace | None = None,
    port: int = 0,
) -> SiteRuntime:
    state = SiteState(data, asset_root=Path(asset_root), trace=trace or ActionTrace())
    app = create_content_app(state, config)
    return SiteRuntime(state=state, server=SiteServer(state, app, port=port))


def create_content_app(state: SiteState, config: ContentSiteConfig) -> Flask:
    app = Flask(config.site)
    entities = state.data["entities"]
    by_id = {entity["entity_id"]: entity for entity in entities}
    search_cache: dict[str, list[dict[str, Any]]] = {}

    def render(body: str, status: int = 200) -> tuple[str, int, dict[str, str]]:
        query = request.args.get("q", "") if request.method == "GET" else ""
        return (
            _shell(config, body, len(state.selected_entity_ids), query),
            status,
            {"Content-Type": "text/html; charset=utf-8"},
        )

    @app.get("/fonts/roboto.ttf")
    def font() -> Any:
        return send_file(Path(__file__).with_name("fonts") / "Roboto.ttf")

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
        return send_file(path)

    @app.get("/public-preview/<entity_id>/<int:index>")
    def public_preview(entity_id: str, index: int) -> Any:
        entity = by_id.get(entity_id)
        previews = entity.get("public_previews", []) if entity else []
        if (
            config.site != "egomedia"
            or not isinstance(previews, list)
            or index >= len(previews)
        ):
            return "", 404
        preview = previews[index]
        relative_path = preview.get("image_path") if isinstance(preview, dict) else None
        if not isinstance(relative_path, str) or not relative_path:
            return "", 404
        try:
            path = resolve_within(
                state.asset_root,
                relative_path,
                context="egomedia public preview",
            )
        except (ValueError, FileNotFoundError):
            return "", 404
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            return "", 404
        return send_file(path)

    @app.get("/")
    def home() -> Any:
        return render(_home_page(config, entities))

    @app.get("/search")
    def search() -> Any:
        query = request.args.get("q", "").strip()
        terms = query.casefold().split()
        if config.site == "egomaps" and not terms:
            matched = []
        elif config.search_mode == "ranked_terms" and terms:
            cache_key = query.casefold()
            if cache_key not in search_cache:
                search_cache[cache_key] = rank_records(
                    entities,
                    query,
                    lambda entity: _searchable_fields(entity, config),
                )
            matched = search_cache[cache_key]
        else:
            matched = [
                entity
                for entity in entities
                if not terms
                or all(term in _searchable_text(entity, config) for term in terms)
            ]
        page_size = (
            20 if config.site == "egomedia" else 15 if config.site == "egomaps" else 24
        )
        try:
            requested_page = max(1, int(request.args.get("page", "1")))
        except ValueError:
            requested_page = 1
        page_count = max(1, (len(matched) + page_size - 1) // page_size)
        current_page = min(requested_page, page_count)
        start = (current_page - 1) * page_size
        shown = matched[start : start + page_size]
        state.log("search", query=query, results=len(matched))
        return render(
            _search_page(
                config,
                query,
                shown,
                current_page=current_page,
                page_count=page_count,
            )
        )

    def detail(entity_id: str) -> Any:
        entity = by_id.get(entity_id)
        if entity is None:
            return render("<h1>Unknown record</h1>", 404)
        state.log("open_entity", entity_id=entity_id)
        return render(
            _detail_page(
                config,
                entity,
                entity_id in state.selected_entity_ids,
                [entity] if config.site == "egomaps" else entities,
            )
        )

    app.add_url_rule(
        f"/{config.detail_path}/<entity_id>",
        "detail",
        detail,
        methods=["GET"],
    )
    if config.detail_path == "video":
        app.add_url_rule("/watch/<entity_id>", "watch", detail, methods=["GET"])

    @app.get("/selection")
    def selection() -> Any:
        selected = [by_id[item] for item in state.selected_entity_ids if item in by_id]
        return render(_selection_page(config, selected, state.fields, state.data))

    @app.post("/select")
    def select() -> Any:
        entity_id = request.form.get("entity_id", "")
        if entity_id not in by_id:
            return render("<h1>Unknown record</h1>", 404)
        state.save(entity_id)
        if config.site == "egomaps":
            return render(
                _detail_page(config, by_id[entity_id], True, [by_id[entity_id]])
            )
        return selection()

    @app.post("/deselect")
    def deselect() -> Any:
        entity_id = request.form.get("entity_id", "")
        if entity_id in by_id:
            state.deselect(entity_id)
        if config.site == "egomaps" and entity_id in by_id:
            return render(
                _detail_page(config, by_id[entity_id], False, [by_id[entity_id]])
            )
        return selection()

    @app.post("/field")
    def set_field() -> Any:
        name = request.form.get("name", "").strip()
        value = request.form.get("value", "").strip()
        if name not in state.data.get("supported_fields", []):
            return render("<h1>Unknown report field</h1>", 400)
        state.set_field(name, value)
        return selection()

    @app.post("/submit")
    def submit() -> Any:
        if not state.selected_entity_ids:
            state.log("submit_empty")
            return render(
                "<h1>Nothing selected</h1><p>Select a record before submitting.</p>",
                400,
            )
        state.commit()
        return render(
            _confirmation_page(
                config, state.submitted_entity_ids, state.submitted_fields
            )
        )

    return app


def _searchable_fields(
    entity: dict[str, Any],
    config: ContentSiteConfig,
) -> list[tuple[str, float]]:
    """Index only site-declared public fields; never IDs or task metadata."""

    fields: list[tuple[str, float]] = []
    for key in config.searchable_keys:
        value = entity.get(key)
        text = flatten_public_text(value)
        if not text.strip():
            continue
        weight = (
            3.0
            if key in {"title", "name"}
            else 2.0
            if key in {"brand", "channel_title", "address", "summary"}
            else 1.0
        )
        fields.append((text, weight))
    return fields


def _searchable_text(entity: dict[str, Any], config: ContentSiteConfig) -> str:
    """Compatibility search text for simple fixture sites."""

    return " ".join(value for value, _ in _searchable_fields(entity, config)).casefold()


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
    overlay = (
        '<span class="play" aria-hidden="true">&#9654;</span>'
        if entity.get("thumbnail_path")
        else ""
    )
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
    published = str(entity.get("published_at") or "").strip()
    if published and not published.casefold().startswith("not shown"):
        values.append(html.escape(published))
    return f'<p class="media-meta">{" · ".join(values)}</p>' if values else ""


def _home_page(config: ContentSiteConfig, entities: Iterable[dict[str, Any]]) -> str:
    shown = list(entities)
    if config.site == "egomedia":
        return content_chrome.media_home_page(config, shown)
    if config.site == "egomaps":
        return content_chrome.maps_home_page(config, shown)
    return _search_page(config, "", shown)


def _search_page(
    config: ContentSiteConfig,
    query: str,
    entities: Iterable[dict[str, Any]],
    *,
    current_page: int = 1,
    page_count: int = 1,
) -> str:
    shown = list(entities)
    if config.site == "egomedia":
        return content_chrome.media_search_page(
            config,
            query,
            shown,
            current_page=current_page,
            page_count=page_count,
        )
    if config.site == "egomaps":
        return content_chrome.maps_search_page(
            config,
            query,
            shown,
            current_page=current_page,
            page_count=page_count,
        )
    cards = "".join(
        '<article class="card">'
        f"{_media_preview(entity)}"
        f"<h2>{html.escape(_entity_title(entity))}</h2>"
        f"{_media_metadata(entity)}"
        f"<p>{html.escape(_entity_summary(entity))}</p>"
        f'<a class="button" href="/{config.detail_path}/{quote(str(entity["entity_id"]))}">Open record</a>'
        "</article>"
        for entity in shown
    )
    return (
        f"<h1>Search {html.escape(config.title)}</h1>"
        f"<p>Results for <b>{html.escape(query or 'all records')}</b></p>"
        f'<div class="grid">{cards or "<p>No matching records.</p>"}</div>'
    )


def _detail_page(
    config: ContentSiteConfig,
    entity: dict[str, Any],
    selected: bool,
    all_entities: list[dict[str, Any]] | None = None,
) -> str:
    if config.site == "egomaps":
        facts = entity.get("facts")
        fact_rows = ""
        if isinstance(facts, dict):
            fact_rows = "".join(
                _render_value(str(key).replace("_", " ").title(), value)
                for key, value in facts.items()
            )
        return content_chrome.maps_detail_page(
            config,
            entity,
            selected,
            fact_rows,
            all_entities or [entity],
        )
    ignored = {
        "entity_id",
        "title",
        "name",
        "summary",
        "description",
        "source",
        "facts",
        "thumbnail_path",
        "photo_path",
        "channel_title",
        "published_at",
        "duration_seconds",
        "view_count",
        "comments",
        "comment_count",
        "sections",
    }
    if config.site == "egomedia":
        ignored.update(
            {"kind", "details", "public_previews"}
        )
    extra = "".join(
        _render_value(str(key).replace("_", " ").title(), value)
        for key, value in entity.items()
        if key not in ignored
    )
    if config.site == "egomedia":
        instructions = entity.get("details", {}).get("linked_manufacturer_instructions")
        if instructions:
            extra += _render_value("Linked manufacturer instructions", instructions)
        current_id = entity["entity_id"]
        channel = entity.get("channel_title")
        related = [
            item for item in (all_entities or []) if item["entity_id"] != current_id
        ]
        related.sort(key=lambda item: item.get("channel_title") != channel)
        return content_chrome.media_detail_page(
            config,
            entity,
            selected,
            extra,
            related[:8],
        )
    entity_id = str(entity["entity_id"])
    selection = (
        '<form method="post" action="/deselect">'
        f'<input type="hidden" name="entity_id" value="{html.escape(entity_id)}">'
        '<button type="submit">Remove selection</button></form>'
        if selected
        else '<form method="post" action="/select">'
        f'<input type="hidden" name="entity_id" value="{html.escape(entity_id)}">'
        f'<button type="submit">{html.escape(config.selection_label)}</button></form>'
    )
    return (
        f"{_media_preview(entity, detail=True)}"
        f"<h1>{html.escape(_entity_title(entity))}</h1>"
        f'<p class="muted">Record ID: {html.escape(entity_id)}</p>'
        f"{_media_metadata(entity)}"
        f"<p>{html.escape(_entity_summary(entity))}</p>"
        f"{extra}{selection}"
    )


def _render_value(label: str, value: Any) -> str:
    if isinstance(value, list):
        rows = "".join(
            _render_value(
                str(
                    item.get("heading") or item.get("label") or label
                    if isinstance(item, dict)
                    else label
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
    supported_fields = data.get("supported_fields", [])
    options = "".join(
        f'<option value="{html.escape(name)}">{html.escape(name.replace("_", " "))}</option>'
        for name in supported_fields
    )
    recorded = (
        "".join(
            f"<li><b>{html.escape(name)}</b>: {html.escape(value)}</li>"
            for name, value in fields.items()
        )
        or "<li>No report fields entered.</li>"
    )
    fields_html = ""
    if supported_fields:
        fields_html = f"<h2>Report fields</h2><ul>{recorded}</ul>"
        fields_html += (
            '<form method="post" action="/field"><label>Field <select name="name">'
            f"{options}</select></label>"
            '<label>Exact value <input name="value" required></label>'
            '<button type="submit">Set field</button></form>'
        )
    if config.site == "egomedia":
        return content_chrome.media_selection_page(config, entities, fields_html)
    if config.site == "egomaps":
        return content_chrome.maps_selection_page(config, entities, fields_html, "")
    rows = (
        "".join(
            '<div class="selected">'
            f"<b>{html.escape(_entity_title(entity))}</b>"
            '<form method="post" action="/deselect">'
            f'<input type="hidden" name="entity_id" value="{html.escape(str(entity["entity_id"]))}">'
            '<button type="submit">Remove</button></form></div>'
            for entity in entities
        )
        or "<p>No records selected.</p>"
    )
    return f"<h1>{html.escape(config.selection_label)}</h1>{rows}{fields_html}"


def _confirmation_page(
    config: ContentSiteConfig,
    entity_ids: list[str],
    fields: dict[str, str],
) -> str:
    return (
        "<h1>✓ Submission complete</h1>"
        f"<p>{html.escape(config.selection_label)}: {html.escape(', '.join(entity_ids))}</p>"
        f"<p>Fields: {html.escape(json.dumps(fields, ensure_ascii=False, sort_keys=True))}</p>"
    )


def _shell(
    config: ContentSiteConfig,
    body: str,
    selected_count: int,
    query: str = "",
) -> str:
    if config.site == "egomedia":
        return content_chrome.media_shell(config, body, selected_count, query)
    if config.site == "egomaps":
        return content_chrome.maps_shell(config, body, selected_count, query)
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(config.title)}</title>
<style>{_CSS}</style></head><body><header><a href="/">{html.escape(config.title)}</a>
<form action="/search"><input name="q" placeholder="Search" value="{html.escape(query)}"><button>Search</button></form>
<a href="/selection">Selection ({selected_count})</a></header><nav>
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
