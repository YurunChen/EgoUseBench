"""EgoDocs site adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from sites.content import ContentSiteConfig, create_content_runtime, load_content_data
from sites.runtime import SiteRuntime


CONFIG = ContentSiteConfig(
    site="egodocs",
    title="EgoDocs",
    entity_key="documents",
    id_key="document_id",
    detail_path="document",
    selection_label="Saved sources",
)


def load_data(
    root: str | Path, *, require_assets: bool = False,
    released_frame_paths: Iterable[str] = (),
) -> dict[str, Any]:
    del require_assets, released_frame_paths
    payload = load_content_data(root, CONFIG, "documents.json")
    payload["supported_fields"] = sorted({
        name
        for document in payload["entities"]
        for name in document.get("facts", {})
    })
    return payload


def create_runtime(
    task: dict[str, Any], arm: str, data: dict[str, Any], asset_root: str | Path,
    *, port: int = 0,
) -> SiteRuntime:
    return create_content_runtime(task, arm, data, asset_root, CONFIG, port=port)
