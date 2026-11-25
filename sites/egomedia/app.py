"""YouTube site adapter. Internal site id remains `egomedia`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from sites.content import ContentSiteConfig, create_content_runtime, load_content_data
from sites.runtime import ActionTrace, SiteRuntime


CONFIG = ContentSiteConfig(
    site="egomedia",
    title="YouTube",
    entity_key="videos",
    id_key="video_id",
    detail_path="video",
    selection_label="Saved media",
    searchable_keys=(
        "title",
        "channel_title",
        "kind",
        "summary",
        "details",
        "chapters",
    ),
    search_mode="ranked_terms",
)


def load_data(
    root: str | Path,
    *,
    require_assets: bool = False,
    released_frame_paths: Iterable[str] = (),
) -> dict[str, Any]:
    del require_assets, released_frame_paths
    return load_content_data(root, CONFIG, "videos.json")


def create_runtime(
    data: dict[str, Any],
    asset_root: str | Path,
    *,
    trace: ActionTrace | None = None,
    port: int = 0,
) -> SiteRuntime:
    return create_content_runtime(data, asset_root, CONFIG, trace=trace, port=port)
