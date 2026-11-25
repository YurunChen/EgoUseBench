"""Chrome and page layouts for YouTube and Google Maps mocks.

Visual tokens come from `.mock-clone/{youtube,maps}/research/design-tokens.json`.
Routes and state changes stay on the shared content-site contract. Sign-in is omitted.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from datetime import date, datetime
from typing import Any
from urllib.parse import quote


_FONT_CSS = """
@font-face{font-family:Roboto;font-style:normal;font-weight:100 900;font-display:swap;src:url(/fonts/roboto.ttf) format('truetype')}
"""


def media_shell(config: Any, body: str, selected_count: int, query: str) -> str:
    del selected_count
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(config.title)}</title>
<link rel="icon" href="{_YT_FAVICON}">
<style>{_FONT_CSS}{_MEDIA_CSS}</style></head>
<body class="media">
<header class="yt-top">
  <div class="yt-start">
    <button class="yt-menu" type="button" aria-label="Guide" aria-controls="yt-guide" aria-expanded="true">{_MENU_ICON}</button>
    <a class="yt-logo" href="/" aria-label="{html.escape(config.title)}">{_YT_LOGO}<span>{html.escape(config.title)}</span></a>
  </div>
  <form class="yt-search" action="/search">
    <input name="q" value="{html.escape(query)}" placeholder="Search" aria-label="Search">
    <button type="submit" aria-label="Search">{_SEARCH_ICON}</button>
  </form>
  <div class="yt-end">
    <a class="yt-more" href="/selection" aria-label="Library">{_MORE_ICON}</a>
  </div>
</header>
<div class="yt-scrim" hidden></div>
<div class="yt-app">
  <nav class="yt-rail" id="yt-guide">
    <a class="yt-nav-home" href="/">{_HOME_ICON}<span>Home</span></a>
    <a class="yt-nav-shorts" href="/">{_SHORTS_ICON}<span>Shorts</span></a>
    <a class="yt-nav-subs" href="/">{_SUBS_ICON}<span>Subscriptions</span></a>
    <hr class="yt-rail-rule">
    <a class="yt-nav-you" href="/selection">{_LIBRARY_ICON}<span>You</span></a>
    <a class="yt-nav-library" href="/selection">{_HISTORY_ICON}<span>Library</span></a>
  </nav>
  <main>{body}</main>
</div>
<script>{_MEDIA_JS}</script>
</body></html>"""


def maps_shell(config: Any, body: str, selected_count: int, query: str) -> str:
    del query
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>{html.escape(config.title)}</title>
<style>{_FONT_CSS}{_MAPS_CSS}</style></head>
<body class="maps">
<nav class="maps-rail">
  <a href="/" title="Menu">{_MAPS_MENU_ICON}</a>
  <a class="maps-rail-active" href="/search" title="Search">{_MAPS_SEARCH_ICON}<small>Search</small></a>
  <a href="/selection" title="Saved">{_MAPS_SAVE_ICON}<small>Saved</small></a>
</nav>
<div class="maps-stage">{body}</div>
</body></html>"""


def media_home_page(config: Any, entities: Iterable[dict[str, Any]]) -> str:
    del config, entities
    return (
        '<div class="yt-chips"><a class="chip chip-on" href="/">All</a>'
        '<a class="chip" href="/search?q=tutorial">Tutorials</a>'
        '<a class="chip" href="/search?q=repair">Repair</a></div>'
        '<div class="yt-home"><section class="yt-home-start">'
        "<h1>Try searching to get started</h1>"
        "<p>Start watching videos to help us build a feed of videos you'll love.</p>"
        "</section></div>"
    )


def media_search_page(
    config: Any,
    query: str,
    entities: Iterable[dict[str, Any]],
    *,
    current_page: int = 1,
    page_count: int = 1,
) -> str:
    rows = "".join(_media_row(config, entity) for entity in entities)
    heading = html.escape(query) if query else "videos"
    navigation = _result_navigation(query, current_page, page_count, "yt-page-nav")
    return (
        f'<h1 class="sr-only">Search {heading}</h1>'
        '<div class="yt-chips"><span class="chip chip-on">All</span>'
        '<span class="chip">Videos</span></div>'
        f'<div class="yt-results">{rows or _MEDIA_NO_RESULTS}{navigation}</div>'
    )


_MEDIA_NO_RESULTS = (
    '<section class="yt-no-results"><h2>No results found</h2>'
    "<p>Try different keywords or remove search filters</p></section>"
)


def _maps_search_form(config: Any, query: str, css: str = "maps-search") -> str:
    return (
        f'<form class="{css}" action="/search">'
        f'<input name="q" value="{html.escape(query)}" placeholder="Search {html.escape(config.title)}" '
        f'aria-label="Search {html.escape(config.title)}">'
        f'<button type="submit" aria-label="Search">{_MAPS_SEARCH_ICON}</button></form>'
    )


def maps_home_page(config: Any, entities: list[dict[str, Any]]) -> str:
    del entities
    return (
        '<div class="maps-home">'
        f'{_maps_search_form(config, "", "maps-search maps-search-float")}'
        f"{_maps_canvas(())}"
        "</div>"
    )


def maps_search_page(
    config: Any,
    query: str,
    entities: list[dict[str, Any]],
    *,
    current_page: int = 1,
    page_count: int = 1,
) -> str:
    cards = "".join(_maps_card(config, entity) for entity in entities)
    navigation = _result_navigation(query, current_page, page_count, "maps-page-nav")
    return (
        f'<div class="maps-panel">'
        f'<form class="maps-search" action="/search">'
        f'<span class="maps-search-icon" aria-hidden="true">{_MAPS_SEARCH_ICON}</span>'
        f'<input name="q" value="{html.escape(query)}" placeholder="Search {html.escape(config.title)}">'
        f'<button type="submit">Search</button></form>'
        f'<p class="maps-status">Results</p>'
        f'<div class="maps-list">{cards or _MAPS_NO_RESULTS}{navigation}</div></div>'
        f"{_maps_canvas(entities)}"
    )


_MAPS_NO_RESULTS = (
    '<section class="maps-no-results"><h2>No results found</h2>'
    "<p>Try a different place, address, or keyword.</p></section>"
)


def _result_navigation(query: str, current_page: int, page_count: int, css: str) -> str:
    if page_count <= 1:
        return ""
    links: list[str] = []
    if current_page > 1:
        links.append(
            f'<a href="/search?q={quote(query)}&amp;page={current_page - 1}">Previous</a>'
        )
    links.append(f"<span>Page {current_page} of {page_count}</span>")
    if current_page < page_count:
        links.append(
            f'<a href="/search?q={quote(query)}&amp;page={current_page + 1}">Next</a>'
        )
    return f'<nav class="{css}">{"".join(links)}</nav>'


def media_detail_page(
    config: Any,
    entity: dict[str, Any],
    selected: bool,
    extra_html: str,
    related: Iterable[dict[str, Any]] = (),
) -> str:
    entity_id = str(entity["entity_id"])
    action = "/deselect" if selected else "/select"
    label = "Saved" if selected else "Save"
    duration = _duration(entity)
    badge = f'<span class="duration">{html.escape(duration)}</span>' if duration else ""
    channel = str(entity.get("channel_title") or "Channel")
    related_html = "".join(_media_related(config, item) for item in related)
    del badge
    player = (
        f'<img src="/thumbnail/{quote(entity_id)}" alt="{html.escape(_title(entity))}">'
        f'<div class="yt-player-chrome" aria-label="Video preview image; playback is not available">'
        f'<span class="yt-big-play" aria-hidden="true">{_PLAY_ICON}</span>'
        '<div class="yt-controls" aria-hidden="true"><div class="yt-progress"></div>'
        f'<div class="yt-controls-row">{_PLAY_ICON}<span>0:00 / {html.escape(duration or "0:00")}</span></div>'
        "</div></div>"
    )
    return (
        '<div class="watch-layout">'
        '<article class="watch">'
        '<div class="player">'
        f"{player}"
        "</div>"
        f"<h1>{html.escape(_title(entity))}</h1>"
        '<div class="watch-owner">'
        f'<span class="yt-avatar" aria-hidden="true">{html.escape(channel[:1])}</span>'
        f'<p class="yt-channel">{html.escape(channel)}</p>'
        f'<form method="post" action="{action}">'
        f'<input type="hidden" name="entity_id" value="{html.escape(entity_id)}">'
        f'<button type="submit">{label}</button></form></div>'
        '<div class="yt-desc-box">'
        f"{_media_description_header(entity)}"
        f"<p>{html.escape(str(entity.get('description') or _summary(entity)))}</p>"
        f"</div>{extra_html}{_media_public_previews(entity)}{_media_comments(entity)}</article>"
        f'<aside class="yt-related">{related_html}</aside></div>'
    )


def _media_description_header(entity: dict[str, Any]) -> str:
    parts: list[str] = []
    views = entity.get("view_count")
    if isinstance(views, int):
        parts.append(f"{_count_label(views)} views")
    age = _published_age(entity)
    if age:
        parts.append(age)
    return f'<p class="yt-desc-stats">{"&nbsp;&nbsp;".join(parts)}</p>' if parts else ""


def _media_public_previews(entity: dict[str, Any]) -> str:
    previews = entity.get("public_previews")
    if not previews:
        return ""
    entity_id = quote(str(entity["entity_id"]), safe="")
    figures = []
    for index, preview in enumerate(previews):
        heading = html.escape(str(preview["heading"]))
        start, end = preview["start_sec"], preview["end_sec"]
        time_range = (
            _clock(start) if start == end else f"{_clock(start)}–{_clock(end)}"
        )
        source = _preview_source_label(preview)
        figures.append(
            '<figure style="margin:12px 0">'
            f"<figcaption><h3>{heading}</h3><p>{time_range}</p>"
            f'<p class="muted">{html.escape(source)}</p></figcaption>'
            f'<a href="/public-preview/{entity_id}/{index}" target="_blank" rel="noopener">'
            f'<img class="full-frame" src="/public-preview/{entity_id}/{index}" '
            f'alt="{heading}" style="height:auto;display:block"></a></figure>'
        )
    return (
        '<section class="yt-public-previews"><h2>Video previews</h2>'
        + "".join(figures)
        + "</section>"
    )


def _preview_source_label(preview: dict[str, Any]) -> str:
    kind = str(
        preview.get("kind") or (preview.get("provenance") or {}).get("kind") or ""
    )
    if kind == "canonical_watch_frame":
        return "YouTube player screenshot source"
    if kind == "youtube_storyboard":
        return "YouTube storyboard source"
    return "Public preview source"


def maps_detail_page(
    config: Any,
    entity: dict[str, Any],
    selected: bool,
    extra_html: str,
    all_entities: list[dict[str, Any]],
) -> str:
    entity_id = str(entity["entity_id"])
    action = "/deselect" if selected else "/select"
    label = "Remove from saved" if selected else "Save"
    photograph = (
        f'<img class="maps-place-photo" src="/thumbnail/{quote(entity_id)}" '
        f'alt="{html.escape(_title(entity))} — place image">'
        if entity.get("photo_path")
        else ""
    )
    availability = (
        '<p class="maps-address">Exact map pin unavailable.</p>'
        if _map_coordinates(entity) is None
        else ""
    )
    if not entity.get("route_options"):
        availability += '<p class="maps-address">Route information unavailable.</p>'
    return (
        '<div class="maps-panel">'
        f'<p class="maps-back"><a href="/search">← Results</a></p>'
        f"{photograph}"
        f"{_maps_location_preview(entity, hero=True)}"
        f"<h1>{html.escape(_title(entity))}</h1>"
        f"{_maps_meta(entity)}"
        f"{availability}"
        f"<p>{html.escape(_summary(entity))}</p>{extra_html}"
        f'<form method="post" action="{action}">'
        f'<input type="hidden" name="entity_id" value="{html.escape(entity_id)}">'
        f'<button class="maps-save" type="submit">{label}</button></form></div>'
        f"{_maps_canvas(all_entities, active_id=entity_id)}"
    )


def media_selection_page(
    config: Any,
    entities: list[dict[str, Any]],
    fields_html: str,
) -> str:
    rows = (
        "".join(
            _media_playlist_row(config, index, entity)
            for index, entity in enumerate(entities, start=1)
        )
        or '<p class="yt-playlist-empty">No videos in this playlist yet</p>'
    )
    return (
        '<div class="yt-library-page yt-playlist">'
        f"{_media_playlist_header(config, entities)}"
        f'<section class="yt-playlist-list">{rows}{fields_html}</section></div>'
        '<div class="yt-snackbar" role="status">Changes are saved.'
        '<button type="button" aria-label="Dismiss">×</button></div>'
    )


def _media_playlist_header(config: Any, entities: list[dict[str, Any]]) -> str:
    label = html.escape(config.selection_label)
    count = f"{len(entities)} video{'' if len(entities) == 1 else 's'}"
    if entities:
        first_id = quote(str(entities[0]["entity_id"]))
        cover = f'<img src="/thumbnail/{first_id}" alt="">'
        backdrop = f' style="--yt-cover:url(/thumbnail/{first_id})"'
        play_all = (
            f'<a class="yt-play-all" href="/{config.detail_path}/{first_id}">'
            f"{_PLAY_ICON}<span>Play all</span></a>"
        )
    else:
        cover, backdrop, play_all = _LIBRARY_ICON, "", ""
    return (
        f'<aside class="yt-playlist-header"{backdrop}>'
        f'<div class="yt-playlist-cover">{cover}</div>'
        f"<h1>{label}</h1>"
        '<p class="yt-playlist-owner">You</p>'
        f'<p class="yt-playlist-meta">Playlist · Private · {count}</p>'
        f'<div class="yt-playlist-actions">{play_all}</div></aside>'
    )


def _media_playlist_row(config: Any, index: int, entity: dict[str, Any]) -> str:
    entity_id = str(entity["entity_id"])
    duration = _duration(entity)
    badge = f'<span class="duration">{html.escape(duration)}</span>' if duration else ""
    return (
        '<div class="yt-playlist-row">'
        f'<span class="yt-playlist-index">{index}</span>'
        f'<a class="yt-playlist-item" href="/{config.detail_path}/{quote(entity_id)}">'
        '<div class="yt-thumb">'
        f'<img src="/thumbnail/{quote(entity_id)}" alt="{html.escape(_title(entity))}">'
        f"{badge}</div>"
        f"<div><h2>{html.escape(_title(entity))}</h2>"
        f"{_media_channel(entity, avatar=False)}{_media_stats(entity)}</div></a>"
        '<form method="post" action="/deselect">'
        f'<input type="hidden" name="entity_id" value="{html.escape(entity_id)}">'
        f'<button class="yt-playlist-remove" type="submit">{_TRASH_ICON}<span>Remove</span></button>'
        "</form></div>"
    )


def maps_selection_page(
    config: Any,
    entities: list[dict[str, Any]],
    fields_html: str,
    submit_html: str,
) -> str:
    rows = (
        "".join(
            '<div class="library-row">'
            f'<a href="/{config.detail_path}/{quote(str(entity["entity_id"]))}">{html.escape(_title(entity))}</a>'
            '<form method="post" action="/deselect">'
            f'<input type="hidden" name="entity_id" value="{html.escape(str(entity["entity_id"]))}">'
            '<button type="submit">Remove</button></form></div>'
            for entity in entities
        )
        or "<p>No records selected.</p>"
    )
    return (
        '<div class="maps-panel">'
        f"<h1>{html.escape(config.selection_label)}</h1>{rows}{fields_html}{submit_html}"
        "</div>"
        f"{_maps_canvas(entities)}"
    )


def _title(entity: dict[str, Any]) -> str:
    return str(entity.get("title") or entity.get("name") or "Untitled record")


def _summary(entity: dict[str, Any]) -> str:
    return str(entity.get("summary") or entity.get("description") or "")


def _search_snippet(entity: dict[str, Any]) -> str:
    text = " ".join(_summary(entity).split())
    return text if len(text) <= 160 else text[:157].rstrip() + "..."


def _duration(entity: dict[str, Any]) -> str:
    duration = entity.get("duration_seconds")
    if not isinstance(duration, int) or duration < 0:
        return ""
    minutes, seconds = divmod(duration, 60)
    return f"{minutes}:{seconds:02d}"


def _clock(seconds: float) -> str:
    minutes, rest = divmod(int(seconds), 60)
    return f"{minutes}:{rest:02d}"


def _published_date(entity: dict[str, Any]) -> date | None:
    match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", str(entity.get("published_at") or "").strip())
    return date(int(match[1]), int(match[2]), int(match[3])) if match else None


def _published_age(entity: dict[str, Any]) -> str:
    """YouTube's relative age, measured from when the listing was captured."""

    published = str(entity.get("published_at") or "").strip()
    if not published or published.casefold().startswith("not shown"):
        return ""
    stated = re.match(
        r"(\d+ (?:second|minute|hour|day|week|month|year)s? ago)", published
    )
    if stated:
        return stated.group(1)
    day = _published_date(entity)
    retrieved = str((entity.get("source") or {}).get("retrieved_at") or "")
    if day is None or not retrieved:
        return ""
    days = (datetime.fromisoformat(retrieved.replace("Z", "+00:00")).date() - day).days
    if days < 0:
        return ""
    for unit, length in (("year", 365), ("month", 30), ("week", 7), ("day", 1)):
        if days >= length:
            count = days // length
            return f"{count} {unit}{'' if count == 1 else 's'} ago"
    return "today"


def _media_stats(entity: dict[str, Any]) -> str:
    values: list[str] = []
    views = entity.get("view_count")
    if isinstance(views, int):
        values.append(f"{_count_label(views)} views")
    age = _published_age(entity)
    if age:
        values.append(age)
    return f'<p class="media-meta">{" • ".join(values)}</p>' if values else ""


def _media_channel(entity: dict[str, Any], *, avatar: bool) -> str:
    channel = str(entity.get("channel_title") or "")
    if not channel:
        return ""
    icon = (
        f'<span class="yt-avatar yt-avatar-sm" aria-hidden="true">{html.escape(channel[:1])}</span>'
        if avatar
        else ""
    )
    return f'<p class="media-meta yt-channel-line">{icon}{html.escape(channel)}</p>'


def _media_row(config: Any, entity: dict[str, Any]) -> str:
    entity_id = quote(str(entity["entity_id"]))
    duration = _duration(entity)
    badge = f'<span class="duration">{html.escape(duration)}</span>' if duration else ""
    return (
        f'<a class="yt-row" href="/{config.detail_path}/{entity_id}">'
        '<div class="yt-thumb">'
        f'<img src="/thumbnail/{entity_id}" alt="{html.escape(_title(entity))}">'
        f"{badge}</div>"
        "<div>"
        f"<h2>{html.escape(_title(entity))}</h2>"
        f"{_media_stats(entity)}"
        f"{_media_channel(entity, avatar=True)}"
        f'<p class="yt-desc">{html.escape(_search_snippet(entity))}</p>'
        "</div></a>"
    )


def _media_related(config: Any, entity: dict[str, Any]) -> str:
    entity_id = quote(str(entity["entity_id"]))
    duration = _duration(entity)
    badge = f'<span class="duration">{html.escape(duration)}</span>' if duration else ""
    return (
        f'<a class="yt-related-card" href="/{config.detail_path}/{entity_id}">'
        '<div class="yt-thumb">'
        f'<img src="/thumbnail/{entity_id}" alt="{html.escape(_title(entity))}">'
        f"{badge}</div>"
        "<div>"
        f"<h2>{html.escape(_title(entity))}</h2>"
        f"{_media_channel(entity, avatar=False)}{_media_stats(entity)}"
        "</div></a>"
    )


def _media_comments(entity: dict[str, Any]) -> str:
    comments = entity.get("comments")
    rows = comments if isinstance(comments, list) else []
    count = entity.get("comment_count")
    if not isinstance(count, int) or count < len(rows):
        count = len(rows)
    items = "".join(_media_comment(item) for item in rows if isinstance(item, dict))
    if not count:
        return ""
    return (
        '<section class="yt-comments" id="comments">'
        f"<h2>{_count_label(count)} Comments</h2>"
        f"{items}</section>"
    )


def _media_comment(item: dict[str, Any]) -> str:
    author = str(item.get("author") or "Viewer")
    letter = next((char for char in author.lstrip("@") if char.isalnum()), "?")
    published = str(item.get("published") or "")
    likes = item.get("like_count")
    like_text = _count_label(int(likes)) if isinstance(likes, int) and likes > 0 else ""
    pinned = '<p class="muted">Pinned</p>' if item.get("pinned") else ""
    meta = " ".join(
        part for part in (html.escape(author), html.escape(published)) if part
    )
    return (
        '<article class="yt-comment">'
        f'<span class="yt-avatar" aria-hidden="true">{html.escape(letter.upper())}</span>'
        "<div>"
        f'{pinned}<p class="yt-comment-meta">{meta}</p>'
        f"<p>{html.escape(str(item.get('text') or ''))}</p>"
        f'<p class="yt-comment-likes">{like_text}</p>'
        "</div></article>"
    )


def _count_label(value: int) -> str:
    """YouTube's compact counts: one decimal below ten units, whole numbers above."""

    for size, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if value >= size:
            scaled = value / size
            text = f"{int(scaled * 10) / 10:.1f}" if scaled < 10 else f"{int(scaled)}"
            return text.removesuffix(".0") + suffix
    return f"{value:,}"


_MAP_COORDINATE_DECIMAL_PLACES = 6


def maps_rating(entity: dict[str, Any]) -> tuple[float, int] | None:
    """Return only listing metadata explicitly stored in the frozen catalog."""

    declared_rating = entity.get("rating")
    declared_count = entity.get("ratings_count")
    if declared_rating is None or declared_count is None:
        return None
    return float(declared_rating), int(declared_count)


def maps_hours(entity: dict[str, Any]) -> tuple[str, str] | None:
    declared = entity.get("hours")
    if isinstance(declared, dict) and declared.get("state"):
        return str(declared["state"]), str(declared.get("detail") or "")
    return None


def _maps_category(entity: dict[str, Any]) -> str:
    return str(entity.get("kind") or "").replace("_", " ").title()


def _maps_stars(rating: float) -> str:
    filled = round(rating)
    return "".join("★" if index < filled else "☆" for index in range(5))


def _maps_meta(entity: dict[str, Any]) -> str:
    rating = maps_rating(entity)
    hours = maps_hours(entity)
    category = _maps_category(entity)
    address = str(entity.get("address") or "")
    line_two = " · ".join(part for part in (category, address) if part)
    rating_html = ""
    if rating is not None:
        score, count = rating
        rating_html = (
            f'<p class="maps-rating">{score:.1f}'
            f'<span class="maps-stars" aria-hidden="true">{_maps_stars(score)}</span>'
            f'<span class="maps-reviews">({count:,})</span></p>'
        )
    hours_html = ""
    if hours is not None:
        state, detail = hours
        state_class = "maps-open" if state.startswith("Open") else "maps-closed"
        rendered = f'<span class="{state_class}">{html.escape(state)}</span>' + (
            f" · {html.escape(detail)}" if detail else ""
        )
        hours_html = f'<p class="maps-hours">{rendered}</p>'
    return (
        rating_html
        + (f'<p class="maps-address">{html.escape(line_two)}</p>')
        + hours_html
    )


def _maps_card(config: Any, entity: dict[str, Any]) -> str:
    entity_id = quote(str(entity["entity_id"]))
    return (
        f'<a class="maps-card" href="/{config.detail_path}/{entity_id}">'
        "<div>"
        f"<h2>{html.escape(_title(entity))}</h2>"
        f"{_maps_meta(entity)}"
        f'<p class="yt-desc">{html.escape(_summary(entity))}</p>'
        f"</div>{_maps_location_preview(entity)}</a>"
    )


def _maps_location_preview(
    entity: dict[str, Any],
    *,
    hero: bool = False,
) -> str:
    """Render address and coordinate context alongside the independent place image."""

    coordinates = _map_coordinates(entity)
    if coordinates is None:
        return ""
    latitude, longitude = coordinates
    title = _title(entity)
    address = str(entity.get("address") or title)
    css = (
        "maps-location-preview maps-location-hero" if hero else "maps-location-preview"
    )
    coordinate_label = (
        f"{latitude:.{_MAP_COORDINATE_DECIMAL_PLACES}f}, "
        f"{longitude:.{_MAP_COORDINATE_DECIMAL_PLACES}f}"
    )
    return (
        f'<div class="{css}" role="img" '
        f'aria-label="Map location for {html.escape(title)} at {html.escape(address)}">'
        '<span class="maps-location-pin" aria-hidden="true"></span>'
        '<span class="maps-location-caption">'
        f"<strong>{html.escape(address)}</strong>"
        f"<small>{coordinate_label}</small></span></div>"
    )


def _map_coordinates(entity: dict[str, Any]) -> tuple[float, float] | None:
    location = entity.get("location")
    if not isinstance(location, dict):
        return None
    latitude = location.get("latitude")
    longitude = location.get("longitude")
    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, (int, float))
        or isinstance(longitude, bool)
        or not isinstance(longitude, (int, float))
    ):
        return None
    return float(latitude), float(longitude)


def _maps_canvas(
    entities: Iterable[dict[str, Any]],
    *,
    active_id: str | None = None,
) -> str:
    placed = [
        entity
        for entity in entities
        if isinstance(entity.get("location"), dict)
        and entity["location"].get("latitude") is not None
        and entity["location"].get("longitude") is not None
    ]
    lats = [float(entity["location"]["latitude"]) for entity in placed]
    lngs = [float(entity["location"]["longitude"]) for entity in placed]
    pins = []
    for entity in placed:
        top, left = _pin_position(entity, lats, lngs)
        css = " pin-active" if str(entity["entity_id"]) == active_id else ""
        pins.append(
            f'<a class="pin{css}" style="top:{top:.1f}%;left:{left:.1f}%" '
            f'href="/place/{quote(str(entity["entity_id"]))}" '
            f'title="{html.escape(_title(entity))}">'
            f'<span class="pin-label">{html.escape(_title(entity))}</span></a>'
        )
    return f'<div class="maps-canvas" aria-hidden="true">{"".join(pins)}</div>'


def _pin_position(
    entity: dict[str, Any],
    lats: list[float],
    lngs: list[float],
) -> tuple[float, float]:
    lat = float(entity["location"]["latitude"])
    lng = float(entity["location"]["longitude"])
    if len(lats) == 1:
        return 48.0, 52.0
    lat_span = max(max(lats) - min(lats), 0.01)
    lng_span = max(max(lngs) - min(lngs), 0.01)
    top = 12 + (max(lats) - lat) / lat_span * 76
    left = 12 + (lng - min(lngs)) / lng_span * 76
    return top, left


_YT_FAVICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E"
    "%3Crect width='24' height='24' rx='6' fill='%23ff0000'/%3E"
    "%3Cpath d='M10 8v8l6-4z' fill='%23fff'/%3E%3C/svg%3E"
)
_YT_LOGO = (
    '<svg class="yt-mark" viewBox="0 0 28 20" aria-hidden="true">'
    '<rect width="28" height="20" rx="5" fill="#ff0000"/>'
    '<path d="M11 5.5v9l8-4.5z" fill="#fff"/></svg>'
)
_MENU_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/>'
    "</svg>"
)
_MORE_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/>'
    "</svg>"
)
_SEARCH_ICON = (
    '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">'
    '<path fill="currentColor" d="M20.87 20.17l-3.68-3.68a7.5 7.5 0 10-1.06 1.06l3.68 3.68.96-.96zM11 17a6 6 0 110-12 6 6 0 010 12z"/>'
    "</svg>"
)
_HOME_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z"/>'
    "</svg>"
)
_SHORTS_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M10 14.65v-5.3L15 12l-5 2.65zm7.77-4.33c-.77-.32-1.2-.5-1.2-.5L18 9.06c1.84-.96 2.53-3.23 1.56-5.06s-3.24-2.53-5.07-1.56L6 6.94c-1.29.68-2.07 2.04-2 3.49.07 1.42.93 2.67 2.22 3.25.03.01 1.2.5 1.2.5L6 14.93c-1.83.97-2.53 3.24-1.56 5.07.97 1.83 3.24 2.53 5.07 1.56l8.3-4.37c1.28-.68 2.07-2.05 2-3.5-.07-1.42-.94-2.68-2.23-3.26z"/>'
    "</svg>"
)
_SUBS_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M10 18v-6l5 3-5 3zm9-13H5v1h14V5zm3 3H2v1h20V8zM2 9h20v11H2V9z"/>'
    "</svg>"
)
_LIBRARY_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M4 6H2v14c0 1.1.9 2 2 2h14v-2H4V6zm16-4H8c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H8V4h12v12z"/>'
    "</svg>"
)
_HISTORY_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M13 3a9 9 0 109 9h-2a7 7 0 11-7-7V3l4 3-4 3V7a5 5 0 105 5h2A7 7 0 1113 5V3z"/>'
    "</svg>"
)
_PLAY_ICON = (
    '<svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">'
    '<path fill="currentColor" d="M6 4l15 8-15 8z"/>'
    "</svg>"
)
_TRASH_ICON = (
    '<svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">'
    '<path fill="currentColor" d="M11 17H9V8h2v9zm4-9h-2v9h2V8zm4-4v1h-1v16H6V5H5V4h4V3h6v1h4zm-2 1H7v15h10V5z"/>'
    "</svg>"
)
_MAPS_MENU_ICON = (
    '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
    '<path fill="currentColor" d="M3 6h18v2H3V6zm0 5h18v2H3v-2zm0 5h18v2H3v-2z"/>'
    "</svg>"
)
_MAPS_SEARCH_ICON = (
    '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
    '<path fill="currentColor" d="M15.5 14h-.79l-.28-.27A6.471 6.471 0 0016 9.5 6.5 6.5 0 109.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C8.01 14 6 11.99 6 9.5S8.01 5 10.5 5 15 7.01 15 9.5 12.99 14 10.5 14z"/>'
    "</svg>"
)
_MAPS_SAVE_ICON = (
    '<svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">'
    '<path fill="currentColor" d="M17 3H7c-1.1 0-2 .9-2 2v16l7-3 7 3V5c0-1.1-.9-2-2-2z"/>'
    "</svg>"
)

_MEDIA_CSS = """
*{box-sizing:border-box}
body.media{margin:0;background:#fff;color:#0f0f0f;font:14px/20px Roboto,Arial,sans-serif}
a{color:#0f0f0f;text-decoration:none}
.yt-menu,.yt-more{border:0;background:transparent;color:#0f0f0f;cursor:pointer;width:40px;height:40px;border-radius:50%;display:grid;place-items:center;padding:0}
.yt-menu:hover,.yt-more:hover{background:rgba(0,0,0,.05)}
.yt-top{position:sticky;top:0;z-index:4;display:flex;align-items:center;justify-content:space-between;height:56px;padding:0 16px;background:#fff}
.yt-start,.yt-end{flex:1;display:flex;align-items:center;min-width:0}
.yt-end{justify-content:flex-end}
.yt-logo{display:flex;align-items:center;gap:4px;padding:0 14px 0 8px;font:700 20px/28px "YouTube Sans",Roboto,Arial,sans-serif;letter-spacing:-.4px;color:#0f0f0f}
.yt-mark{width:28px;height:20px;flex-shrink:0}
.yt-search{flex:0 1 640px;display:flex;height:40px;margin:0 40px}
.yt-search input{flex:1;border:1px solid #ccc;border-right:0;border-radius:40px 0 0 40px;padding:0 16px;background:#fff;font:16px Roboto,Arial,sans-serif;color:#0f0f0f;box-shadow:inset 0 1px 2px rgba(0,0,0,.05)}
.yt-search button{width:64px;border:1px solid #d3d3d3;border-left:0;border-radius:0 40px 40px 0;background:#f8f8f8;color:#0f0f0f;cursor:pointer;display:grid;place-items:center}
.yt-app{display:grid;grid-template-columns:240px minmax(0,1fr);min-height:calc(100vh - 56px)}
.yt-rail{display:flex;flex-direction:column;gap:0;padding:12px 12px 24px;background:#fff}
.yt-rail a{display:flex;align-items:center;gap:24px;padding:10px 12px;border-radius:10px;font-size:14px;font-weight:400;color:#0f0f0f}
.yt-rail a:hover{background:#f2f2f2}
.yt-rail-rule{border:0;border-top:1px solid rgba(0,0,0,.1);margin:12px 12px}
.yt-app:has(.yt-home) .yt-nav-home,.yt-app:has(.yt-library-page) .yt-nav-you{background:#f2f2f2}
.yt-app:has(.watch-layout){grid-template-columns:minmax(0,1fr)}
.yt-app:has(.watch-layout) .yt-rail{display:none}
.yt-rail svg{display:block;flex-shrink:0}
.yt-scrim{position:fixed;inset:56px 0 0 0;background:rgba(0,0,0,.5);z-index:5}
.yt-scrim[hidden]{display:none}
body.yt-guide-mini .yt-app{grid-template-columns:72px minmax(0,1fr)}
body.yt-guide-mini .yt-rail{padding:8px 4px}
body.yt-guide-mini .yt-rail a{flex-direction:column;gap:6px;padding:16px 4px;font-size:10px;line-height:14px;text-align:center;border-radius:10px}
body.yt-guide-mini .yt-rail-rule,body.yt-guide-mini .yt-nav-library{display:none}
body.yt-guide-open .yt-rail{display:flex;position:fixed;left:0;top:56px;bottom:0;width:240px;z-index:6;overflow:auto;box-shadow:2px 0 8px rgba(0,0,0,.2)}
body.yt-guide-open .yt-rail a{flex-direction:row;gap:24px;padding:10px 12px;font-size:14px;line-height:20px;text-align:left}
body.yt-guide-open .yt-rail-rule,body.yt-guide-open .yt-nav-library{display:block}
.yt-chips{display:flex;gap:8px;margin:0 0 16px;flex-wrap:wrap}
.chip{background:#f2f2f2;border-radius:8px;padding:8px 12px;font-size:14px;font-weight:500;color:#0f0f0f}
.chip-on{background:#0f0f0f;color:#fff}
.yt-home{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:40px 16px}
.yt-home-start{grid-column:1/-1;border:1px solid #e5e5e5;border-radius:14px;padding:42px;text-align:center;background:#fafafa}.yt-home-start h1{font-size:28px;margin:0 0 10px}.yt-home-start p{color:#606060;margin:0}
.yt-home-card h2{margin:8px 0 4px;font-size:16px;line-height:22px;font-weight:500}
.yt-home-card .media-meta{font-size:14px;line-height:20px}
main{padding:24px 24px 48px}
.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}
.yt-results{display:flex;flex-direction:column;gap:16px;max-width:1096px}
.yt-page-nav{display:flex;justify-content:center;align-items:center;gap:18px;padding:24px}.yt-page-nav a{padding:9px 16px;border:1px solid #d3d3d3;border-radius:18px;font-weight:500}.yt-page-nav a:hover{background:#f2f2f2}
.yt-row{display:grid;grid-template-columns:360px minmax(0,1fr);gap:16px;align-items:start}
.yt-thumb{position:relative;background:#000;border-radius:12px;overflow:hidden;aspect-ratio:16/9}
.yt-thumb img,.player img{width:100%;height:100%;object-fit:cover;display:block}
.duration{position:absolute;right:8px;bottom:8px;background:rgba(0,0,0,.8);color:#fff;border-radius:4px;padding:1px 4px;font-size:12px;font-weight:500}
.yt-row h2{margin:2px 0 6px;font-size:18px;line-height:26px;font-weight:400}
.media-meta,.yt-desc,.muted{color:#606060;font-size:12px;margin:0}
.yt-desc{margin-top:6px}
.watch-layout{display:grid;grid-template-columns:minmax(0,1fr) 402px;gap:24px;align-items:start;max-width:1750px;margin:0 auto}
.player{position:relative;background:#000;border-radius:12px;overflow:hidden;aspect-ratio:16/9}
.yt-player-chrome{position:absolute;inset:0}
.yt-big-play{position:absolute;left:50%;top:50%;width:68px;height:48px;margin:-24px 0 0 -34px;border-radius:14px;background:rgba(33,33,33,.8);color:#fff;display:grid;place-items:center}
.yt-big-play svg{width:28px;height:28px}
.yt-controls{position:absolute;left:0;right:0;bottom:0;padding:0 12px 6px;color:#fff;font-size:13px;background:linear-gradient(transparent,rgba(0,0,0,.55))}
.yt-progress{height:3px;background:rgba(255,255,255,.3);margin:24px 0 6px}
.yt-controls-row{display:flex;align-items:center;gap:14px;height:28px}
.yt-controls-row svg{width:22px;height:22px}
.yt-desc-stats{font-weight:500;margin:0 0 4px!important}
.yt-channel-line{display:flex;align-items:center;gap:8px;margin:10px 0 0}
.yt-avatar-sm{width:24px;height:24px;font-size:11px}
.yt-related-card .yt-channel-line{margin:0}
.yt-no-results{text-align:center;padding:64px 0}.yt-no-results h2{font-size:20px;font-weight:400;margin:0 0 8px}.yt-no-results p{color:#606060;margin:0}
.watch h1{font-size:20px;line-height:28px;font-weight:700;margin:12px 0 8px}
.watch-owner{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.watch-owner form{margin-left:auto}
.yt-avatar{width:40px;height:40px;border-radius:50%;background:#0f0f0f;color:#fff;display:grid;place-items:center;font-weight:500;flex-shrink:0}
.yt-channel{margin:0;font-size:16px;font-weight:500;line-height:22px}
.yt-desc-box{background:#f2f2f2;border-radius:12px;padding:12px;margin-top:4px;font-size:14px;line-height:20px}
.yt-desc-box p{margin:0}
.yt-related{display:flex;flex-direction:column;gap:8px}
.yt-related-card{display:grid;grid-template-columns:168px minmax(0,1fr);gap:8px}
.yt-related-card .yt-thumb{border-radius:8px}
.yt-related-card h2{margin:0 0 4px;font-size:14px;line-height:20px;font-weight:500}
.yt-comments{margin-top:24px;padding-top:8px}
.yt-comments h2{font-size:20px;line-height:28px;font-weight:700;margin:0 0 16px}
.yt-comment{display:grid;grid-template-columns:40px minmax(0,1fr);gap:12px;margin:0 0 16px}
.yt-comment-meta{color:#606060;font-size:13px;margin:0 0 4px}
.yt-comment p{margin:0;font-size:14px;line-height:20px;white-space:pre-wrap}
.yt-comment-likes{color:#606060;font-size:12px;margin:8px 0 0}
.watch-owner button,form button{border:0;background:#f2f2f2;border-radius:18px;padding:8px 16px;font:500 14px Roboto,Arial,sans-serif;cursor:pointer}
.primary{background:#0f0f0f;color:#fff}
.yt-playlist{display:grid;grid-template-columns:360px minmax(0,1fr);gap:24px;align-items:start;max-width:1284px}
.yt-playlist-header{position:sticky;top:80px;overflow:hidden;isolation:isolate;box-sizing:border-box;min-height:calc(100vh - 104px);padding:24px;border-radius:16px;background:#5f5f5f;color:#fff}
.yt-playlist-header::before{content:"";position:absolute;inset:-60px;z-index:-2;background:var(--yt-cover,none) center/cover;filter:blur(56px) saturate(1.5)}
.yt-playlist-header::after{content:"";position:absolute;inset:0;z-index:-1;background:linear-gradient(rgba(0,0,0,.2),rgba(0,0,0,.5))}
.yt-playlist-cover{display:grid;place-items:center;aspect-ratio:16/9;overflow:hidden;border-radius:12px;background:rgba(255,255,255,.1);color:rgba(255,255,255,.7)}
.yt-playlist-cover img{width:100%;height:100%;object-fit:cover;display:block}
.yt-playlist-cover svg{width:56px;height:56px}
.yt-playlist-header h1{margin:16px 0 0;font:700 28px/38px "YouTube Sans",Roboto,Arial,sans-serif}
.yt-playlist-owner{margin:16px 0 0;font-size:14px;font-weight:500;line-height:20px}
.yt-playlist-meta{margin:2px 0 0;font-size:12px;line-height:18px;color:rgba(255,255,255,.7)}
.yt-playlist-actions{display:flex;margin:12px 0 8px}
.yt-play-all{display:flex;align-items:center;justify-content:center;gap:6px;width:168px;height:40px;border-radius:20px;background:#fff;color:#0f0f0f;font-size:14px;font-weight:500}
.yt-play-all:hover{background:#e6e6e6}
.yt-playlist-list{display:flex;flex-direction:column;min-width:0}
.yt-playlist-row{display:grid;grid-template-columns:24px minmax(0,1fr) auto;align-items:center;gap:4px;padding:0 8px 0 0;border-radius:12px}
.yt-playlist-row:hover{background:#f2f2f2}
.yt-playlist-index{color:#606060;font-size:14px;text-align:center}
.yt-playlist-item{display:grid;grid-template-columns:160px minmax(0,1fr);gap:8px;align-items:start;padding:4px 0}
.yt-playlist-item .yt-thumb{border-radius:8px}
.yt-playlist-item h2{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;margin:0 0 4px;font-size:16px;line-height:22px;font-weight:500}
.yt-playlist-row .yt-playlist-remove{display:flex;align-items:center;gap:6px;background:transparent;color:#0f0f0f;padding:8px 12px}
.yt-playlist-row .yt-playlist-remove:hover{background:rgba(0,0,0,.1)}
.yt-playlist-empty{margin:24px 0;color:#606060;font-size:14px}
.yt-snackbar{position:fixed;left:24px;bottom:24px;z-index:8;display:flex;align-items:center;gap:16px;min-width:280px;padding:14px 16px;border-radius:12px;background:#0f0f0f;color:#fff;font-size:14px;box-shadow:0 4px 12px rgba(0,0,0,.3)}
.yt-snackbar button{border:0;background:transparent;color:#fff;font-size:18px;line-height:1;cursor:pointer;padding:0 4px}
.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.frames img{width:100%;border-radius:8px}
.full-frame{max-width:100%}
section{margin-top:18px}
@media (max-width:1015px){
  .yt-app{grid-template-columns:1fr}
  .yt-rail{display:none}
  .watch-layout{grid-template-columns:1fr}
  .yt-playlist{grid-template-columns:1fr}
  .yt-playlist-header{position:static;min-height:0}
  .yt-search{margin:0 16px}
}
body.yt-guide-open .yt-rail{display:flex}
@media (max-width:800px){
  .yt-row{grid-template-columns:minmax(160px,240px) 1fr;gap:12px}
  .yt-row h2{font-size:16px;line-height:22px}
}
"""

_MEDIA_JS = """
(function(){
  var body=document.body;
  var button=document.querySelector('.yt-menu');
  var scrim=document.querySelector('.yt-scrim');
  if(!button)return;
  function overlayMode(){
    return Boolean(document.querySelector('.watch-layout'))||window.matchMedia('(max-width:1015px)').matches;
  }
  function sync(){
    var overlay=overlayMode();
    var expanded=overlay?body.classList.contains('yt-guide-open'):!body.classList.contains('yt-guide-mini');
    button.setAttribute('aria-expanded', expanded?'true':'false');
    if(scrim) scrim.hidden=!body.classList.contains('yt-guide-open');
  }
  button.addEventListener('click', function(){
    if(overlayMode()){
      body.classList.remove('yt-guide-mini');
      body.classList.toggle('yt-guide-open');
    }else{
      body.classList.remove('yt-guide-open');
      body.classList.toggle('yt-guide-mini');
    }
    sync();
  });
  if(scrim) scrim.addEventListener('click', function(){
    body.classList.remove('yt-guide-open');
    sync();
  });
  document.addEventListener('keydown', function(event){
    if(event.key==='Escape'){
      body.classList.remove('yt-guide-open');
      sync();
    }
  });
  sync();
})();
"""

_MAPS_CSS = """
*{box-sizing:border-box}
body.maps{margin:0;height:100vh;display:flex;background:#fff;color:#1f1f1f;font:14px/20px Roboto,Arial,sans-serif}
a{color:#1a73e8;text-decoration:none}
.maps-rail{width:72px;flex-shrink:0;display:flex;flex-direction:column;align-items:center;gap:8px;padding:12px 0;border-right:1px solid #e8eaed;background:#fff}
.maps-rail a{display:flex;flex-direction:column;align-items:center;gap:2px;color:#5e5e5e;text-align:center;font-size:10px;padding:10px 0;width:100%;border-radius:0 24px 24px 0}
.maps-rail a:hover,.maps-rail-active{background:#e8f0fe;color:#1a73e8}
.maps-rail small{display:block;font-size:10px;font-weight:500}
.maps-rail svg{display:block}
.maps-stage{flex:1;min-width:0;display:grid;grid-template-columns:400px minmax(0,1fr)}
.maps-stage:has(.maps-home){grid-template-columns:1fr}
.maps-panel{background:#fff;border-right:1px solid #e8eaed;overflow:auto;padding:12px 16px 32px}
.maps-home{position:relative;min-height:100%}
.maps-home .maps-canvas{min-height:100vh}
.maps-search-float{position:absolute;left:12px;top:12px;z-index:3;width:392px;margin:0}
.maps-search{display:flex;align-items:center;gap:0;height:48px;border-radius:24px;box-shadow:0 1px 4px rgba(0,0,0,.2);overflow:hidden;margin:4px 0 12px;background:#fff}
.maps-search-icon{display:grid;place-items:center;width:48px;color:#5e5e5e;flex-shrink:0}
.maps-search input{flex:1;border:0;padding:0 8px 0 0;font:16px Roboto,Arial,sans-serif}
.maps-search button{border:0;background:#fff;padding:0 16px;color:#5e5e5e;cursor:pointer}
.maps-status{color:#5e5e5e;margin:0 0 8px}
.maps-list{display:flex;flex-direction:column}
.maps-no-results{padding:24px 8px 32px}.maps-no-results h2{margin:0 0 6px;font:400 16px/24px Roboto,Arial,sans-serif;color:#202124}.maps-no-results p{margin:0;color:#70757a;font-size:14px}
.maps-page-nav{display:flex;justify-content:space-between;align-items:center;padding:16px 18px;border-top:1px solid #e0e0e0}.maps-page-nav a{color:#1a73e8;font-weight:500}
.maps-card{display:grid;grid-template-columns:minmax(0,1fr) 88px;gap:12px;padding:14px 0;border-bottom:1px solid #e8eaed;color:#1f1f1f}
.maps-card h2{margin:0 0 4px;font-size:16px;font-weight:500}
.maps-location-preview{position:relative;width:104px;min-height:104px;overflow:hidden;border-radius:8px;background-color:#e6f0d8;background-image:linear-gradient(26deg,transparent 46%,#fff 47%,#fff 53%,transparent 54%),linear-gradient(90deg,transparent 47%,#c5d0c0 48%,#c5d0c0 50%,transparent 51%);background-size:80px 80px;color:#3c4043}
.maps-location-hero{width:100%;height:180px;margin-bottom:10px}
.maps-place-photo{display:block;width:100%;height:220px;object-fit:contain;background:#f1f3f4;border-radius:8px;margin-bottom:10px}
.maps-location-pin{position:absolute;left:50%;top:42%;width:28px;height:28px;margin:-28px 0 0 -14px;background:#ea4335;border-radius:50% 50% 50% 0;transform:rotate(-45deg);box-shadow:0 2px 6px rgba(0,0,0,.25)}
.maps-location-pin:after{content:"";position:absolute;inset:8px;background:#fff;border-radius:50%}
.maps-location-caption{position:absolute;left:6px;right:6px;bottom:6px;padding:5px 6px;border-radius:5px;background:rgba(255,255,255,.92);font-size:10px;line-height:13px}
.maps-location-caption strong,.maps-location-caption small{display:block}.maps-location-caption strong{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.maps-location-caption small{color:#5f6368}
.maps-address,.yt-desc{color:#5e5e5e;margin:0;font-size:13px}
.maps-rating{margin:2px 0 0;font-size:13px;color:#3c4043;display:flex;align-items:center;gap:4px}
.maps-stars{color:#e7711b;letter-spacing:-1px}
.maps-reviews{color:#5e5e5e}
.maps-hours{margin:2px 0 0;font-size:13px}
.maps-open{color:#188038}
.maps-closed{color:#d93025}
.maps-canvas{position:relative;background:#e8eaed url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80'%3E%3Crect fill='%23e6f0d8' width='80' height='80'/%3E%3Cpath fill='%23d5e8c6' d='M0 40h80v40H0z'/%3E%3Cpath stroke='%23c5d0c0' stroke-width='1' fill='none' d='M0 20h80M0 60h80M20 0v80M60 0v80'/%3E%3C/svg%3E");overflow:hidden}
.pin{position:absolute;width:28px;height:28px;margin:-28px 0 0 -14px;background:#ea4335;border-radius:50% 50% 50% 0;transform:rotate(-45deg);box-shadow:0 2px 6px rgba(0,0,0,.3)}
.pin:after{content:"";position:absolute;inset:8px;background:#fff;border-radius:50%}
.pin-active{background:#1a73e8;z-index:2}
.pin-label{position:absolute;left:28px;top:4px;transform:rotate(45deg);white-space:nowrap;color:#1f1f1f;font:500 12px Roboto,Arial,sans-serif;text-shadow:0 0 4px #fff}
.maps-save,.primary,form button{border:0;background:#1a73e8;color:#fff;border-radius:20px;padding:10px 16px;font:500 14px Roboto,Arial,sans-serif;cursor:pointer}
.library-row{display:flex;justify-content:space-between;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid #e8eaed}
.library-row button{background:#f1f3f4;color:#1f1f1f}
.maps-back{margin:0 0 10px}
.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.frames img{width:100%;border-radius:8px}
.full-frame{max-width:100%}
section{margin-top:16px}
h1{font:400 20px/24px "Google Sans",Roboto,Arial,sans-serif;margin:8px 0;color:#1f1f1f}
"""
