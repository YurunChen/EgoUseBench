from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .workspace import resolve_within


SITES = {"egomart", "egodocs", "egomedia", "egomaps"}
ENTITY_VISUAL_KEYS = {
    "egomart": "image_path",
    "egomedia": "thumbnail_path",
    "egomaps": "photo_path",
}
HIDDEN_ROLE_KEYS = {"role in scene", "benchmark role", "target role"}
LEAKING_PUBLIC_PHRASES = {"search distractor", "correct answer", "wrong answer"}


class EnvironmentError(ValueError):
    """Raised when a task site cannot be loaded through the site seam."""


def _site_app(site: str) -> Any:
    if site not in SITES:
        raise EnvironmentError(f"unsupported site {site!r}")
    try:
        return importlib.import_module(f"sites.{site}.app")
    except (ImportError, AttributeError) as exc:
        raise EnvironmentError(f"site {site!r} is not installed correctly: {exc}") from exc


def load_site_data(
    root: str | Path,
    site: str,
    *,
    require_assets: bool = False,
    released_frame_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Load one site's frozen data through its own implementation."""

    app = _site_app(site)
    loader = getattr(app, "load_data", None)
    if not callable(loader):
        raise EnvironmentError(f"site {site!r} does not expose load_data")
    try:
        data = loader(
            Path(root).resolve(),
            require_assets=require_assets,
            released_frame_paths=released_frame_paths,
        )
    except (OSError, ValueError) as exc:
        raise EnvironmentError(str(exc)) from exc
    if not isinstance(data, dict) or data.get("site") != site:
        raise EnvironmentError(f"site {site!r} returned invalid frozen data")
    entities = data.get("entities")
    if not isinstance(entities, list) or not entities:
        raise EnvironmentError(f"site {site!r} must expose a non-empty entities list")
    entity_ids = [entity.get("entity_id") for entity in entities if isinstance(entity, dict)]
    if len(entity_ids) != len(entities) or any(
        not isinstance(entity_id, str) or not entity_id for entity_id in entity_ids
    ):
        raise EnvironmentError(f"site {site!r} contains an invalid entity_id")
    if len(entity_ids) != len(set(entity_ids)):
        raise EnvironmentError(f"site {site!r} contains duplicate entity IDs")
    for entity in entities:
        attributes = entity.get("attributes", {})
        if isinstance(attributes, dict) and any(
            str(key).strip().casefold() in HIDDEN_ROLE_KEYS for key in attributes
        ):
            raise EnvironmentError(f"site {site!r} exposes a hidden benchmark role")
        public_text = " ".join(
            str(entity.get(key, "")) for key in ("title", "name", "summary", "description")
        ).casefold()
        if any(phrase in public_text for phrase in LEAKING_PUBLIC_PHRASES):
            raise EnvironmentError(f"site {site!r} exposes evaluator language")
    return data


def site_entity_ids(site_data: dict[str, Any]) -> set[str]:
    """Return the identifiers exposed by one already-validated site dataset."""

    return {str(entity["entity_id"]) for entity in site_data["entities"]}


def entity_visual_path(site: str, entity: dict[str, Any]) -> str | None:
    """Return the single public visual asset used by a site entity, if any."""

    key = ENTITY_VISUAL_KEYS.get(site)
    value = entity.get(key) if key else None
    return value if isinstance(value, str) and value else None


def validate_referenced_site_assets(
    site: str,
    entities: Iterable[dict[str, Any]],
    *,
    root: str | Path,
    released_frame_paths: Iterable[str] = (),
) -> None:
    """Validate release-critical assets for entities used by one task.

    The frozen sites are shared by all tasks, so incomplete unused catalog
    records must not invalidate an otherwise complete task.  Callers pass only
    the entities that the task can select or compare.
    """

    released = set(released_frame_paths)
    for entity in entities:
        entity_id = str(entity.get("entity_id", "<unknown>"))
        if site == "egomart":
            _validate_visual_asset(
                entity,
                "image_path",
                root=root,
                released_frame_paths=released,
                context=f"{entity_id} product image",
            )
        elif site == "egomedia":
            _validate_media_entity(
                entity,
                root=root,
                released_frame_paths=released,
                context=f"{entity_id} media thumbnail",
            )
        elif site == "egomaps":
            _validate_map_entity(
                entity,
                root=root,
                released_frame_paths=released,
                context=entity_id,
            )


def _validate_visual_asset(
    entity: dict[str, Any],
    key: str,
    *,
    root: str | Path,
    released_frame_paths: set[str],
    context: str,
) -> None:
    relative_path = entity.get(key)
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise EnvironmentError(f"{context}: missing {key}")
    if relative_path in released_frame_paths:
        raise EnvironmentError(f"{context}: site visual reuses a released observation frame")
    try:
        path = resolve_within(root, relative_path, context=context)
    except (ValueError, FileNotFoundError) as exc:
        raise EnvironmentError(str(exc)) from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise EnvironmentError(f"{context}: visual asset is empty or not a file")


def _validate_media_entity(
    entity: dict[str, Any],
    *,
    root: str | Path,
    released_frame_paths: set[str],
    context: str,
) -> None:
    """Require the metadata a user would normally see on a video result."""

    _validate_visual_asset(
        entity,
        "thumbnail_path",
        root=root,
        released_frame_paths=released_frame_paths,
        context=context,
    )
    for key in ("title", "summary", "channel_title", "published_at"):
        value = entity.get(key)
        if not isinstance(value, str) or not value.strip():
            raise EnvironmentError(f"{context}: missing {key}")
    for key in ("duration_seconds", "view_count"):
        value = entity.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise EnvironmentError(f"{context}: missing usable {key}")


def _validate_map_entity(
    entity: dict[str, Any],
    *,
    root: str | Path,
    released_frame_paths: set[str],
    context: str,
) -> None:
    _validate_visual_asset(
        entity,
        "photo_path",
        root=root,
        released_frame_paths=released_frame_paths,
        context=f"{context} place photo",
    )
    address = entity.get("address")
    if not isinstance(address, str) or not address.strip():
        raise EnvironmentError(f"{context}: missing address")

    location = entity.get("location")
    if not isinstance(location, dict):
        raise EnvironmentError(f"{context}: missing location coordinates")
    latitude = location.get("latitude", location.get("lat"))
    longitude = location.get(
        "longitude", location.get("lng", location.get("lon"))
    )
    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, (int, float))
        or not -90 <= latitude <= 90
        or isinstance(longitude, bool)
        or not isinstance(longitude, (int, float))
        or not -180 <= longitude <= 180
    ):
        raise EnvironmentError(f"{context}: invalid location coordinates")

    routes = entity.get("route_options")
    if (
        not isinstance(routes, dict)
        or not routes
        or any(
            not isinstance(mode, str)
            or not mode.strip()
            or not isinstance(description, str)
            or not description.strip()
            for mode, description in routes.items()
        )
    ):
        raise EnvironmentError(f"{context}: missing usable route_options")


def create_site_runtime(
    task: dict[str, Any], arm: str, site_data: dict[str, Any], *,
    asset_root: str | Path, port: int = 0,
) -> Any:
    """Create the selected site's state and local server behind one interface."""

    site = str(task["site"])
    app = _site_app(site)
    factory = getattr(app, "create_runtime", None)
    if not callable(factory):
        raise EnvironmentError(f"site {site!r} does not expose create_runtime")
    runtime = factory(task, arm, site_data, Path(asset_root).resolve(), port=port)
    if not hasattr(runtime, "state") or not hasattr(runtime, "server"):
        raise EnvironmentError(f"site {site!r} returned an invalid runtime")
    return runtime
