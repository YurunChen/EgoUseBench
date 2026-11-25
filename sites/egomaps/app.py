"""Google Maps site adapter. Internal site id remains `egomaps`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from sites.content import ContentSiteConfig, create_content_runtime, load_content_data
from sites.runtime import ActionTrace, SiteRuntime


CONFIG = ContentSiteConfig(
    site="egomaps",
    title="Google Maps",
    entity_key="places",
    id_key="place_id",
    detail_path="place",
    selection_label="Saved places",
    searchable_keys=("title", "name", "kind", "address", "summary", "facts"),
    search_mode="ranked_terms",
)


def load_data(
    root: str | Path,
    *,
    require_assets: bool = False,
    released_frame_paths: Iterable[str] = (),
) -> dict[str, Any]:
    del require_assets, released_frame_paths
    return load_content_data(root, CONFIG, "places.json")


def create_runtime(
    data: dict[str, Any],
    asset_root: str | Path,
    *,
    trace: ActionTrace | None = None,
    port: int = 0,
) -> SiteRuntime:
    return create_content_runtime(data, asset_root, CONFIG, trace=trace, port=port)
