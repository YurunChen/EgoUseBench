"""EgoMaps site adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from sites.content import ContentSiteConfig, create_content_runtime, load_content_data
from sites.runtime import SiteRuntime


CONFIG = ContentSiteConfig(
    site="egomaps",
    title="EgoMaps",
    entity_key="places",
    id_key="place_id",
    detail_path="place",
    selection_label="Saved destinations",
)


def load_data(
    root: str | Path, *, require_assets: bool = False,
    released_frame_paths: Iterable[str] = (),
) -> dict[str, Any]:
    del require_assets, released_frame_paths
    return load_content_data(root, CONFIG, "places.json")


def create_runtime(
    task: dict[str, Any], arm: str, data: dict[str, Any], asset_root: str | Path,
    *, port: int = 0,
) -> SiteRuntime:
    return create_content_runtime(task, arm, data, asset_root, CONFIG, port=port)
