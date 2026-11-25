from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from sites.runtime import ActionTrace, SiteRuntime

from .workspace import resolve_within, write_text_atomic


@dataclass(frozen=True)
class _SiteDefinition:
    catalog: str
    image_field: str


_SITE_DEFINITIONS = {
    "egomart": _SiteDefinition("products.json", "image_path"),
    "egomedia": _SiteDefinition("videos.json", "thumbnail_path"),
    "egomaps": _SiteDefinition("places.json", "photo_path"),
}
SITES = frozenset(_SITE_DEFINITIONS)
SITE_DATA_FILES = {
    site: f"sites/{site}/{definition.catalog}"
    for site, definition in _SITE_DEFINITIONS.items()
}
ENTITY_VISUAL_KEYS = {
    site: definition.image_field for site, definition in _SITE_DEFINITIONS.items()
}
HIDDEN_ROLE_KEYS = {
    "role in scene",
    "benchmark role",
    "target role",
    "benchmark_role",
    "catalog role",
    "filler",
    "rival",
}
# A listing that explains its relation to another listing ("a similarly named X",
# "the plaza containing Y") hands the Agent the distractor structure for free.
ROLE_NARRATING_PHRASES = {
    "similarly named",
    "same name as",
    "look-alike",
    "lookalike",
    "distractor",
    "decoy",
    "the correct one",
    "not the one",
}

LEAKING_PUBLIC_PHRASES = {
    "search distractor",
    "correct answer",
    "wrong answer",
    # Benchmark-internal vocabulary must never reach a public listing field: a site
    # entity that narrates its own role in the task tells the Agent which one to pick.
    "source observation",
    "source video",
    "source frame",
    "source clip",
    "released observation",
    "benchmark",
    "task target",
    "the wearer",
    "first-person",
    "egocentric",
    "frozen map",
    "frozen catalog",
    "frozen site",
}

AUDIT_ONLY_ROOT_KEYS = frozenset(
    {
        "catalog_source",
        "content_origin",
        "provenance",
        "source",
    }
)
AUDIT_ONLY_ENTITY_KEYS = frozenset(
    {
        "source",
        # Legacy test fixtures use this as checker grouping metadata. It is never a
        # product attribute and therefore must not enter the storefront runtime.
        "tier",
    }
)
PUBLIC_SITE_FIELDS = frozenset({"schema_version", "site", "supported_fields"})
PUBLIC_ENTITY_FIELDS = {
    "egomart": frozenset(
        {
            "entity_id",
            "item_id",
            "name",
            "brand",
            "price",
            "attributes",
            "image_path",
            "rating",
            "ratings_count",
        }
    ),
    "egomedia": frozenset(
        {
            "entity_id",
            "title",
            "summary",
            "description",
            "kind",
            "channel_title",
            "published_at",
            "duration_seconds",
            "view_count",
            "thumbnail_path",
            "details",
            "facts",
            "chapters",
            "comment_count",
            "comments",
            "public_previews",
        }
    ),
    "egomaps": frozenset(
        {
            "entity_id",
            "name",
            "summary",
            "kind",
            "address",
            "location",
            "route_options",
            "facts",
            "entrances",
            "photo_path",
            "rating",
            "ratings_count",
            "hours",
        }
    ),
}


def _is_https_url(value: Any) -> bool:
    parsed = urlparse(str(value).strip())
    return parsed.scheme == "https" and bool(parsed.netloc)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def site_data_path(site: str) -> str:
    """Return the frozen catalog path for one allowed site."""

    if site not in SITES:
        raise EnvironmentError(f"unsupported site {site!r}")
    return SITE_DATA_FILES[site]


SITE_SNAPSHOT_DIRECTORY = "site-snapshots"
"""Beside the experiment directories; one file per distinct website version."""
SNAPSHOT_DIGEST_LENGTH = 16


def site_snapshot(experiments_root: Path, site: str, data: Mapping[str, Any]) -> Path:
    """Store one website version once; every run that used it shares the file."""

    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    digest = hashlib.sha256(text.encode()).hexdigest()[:SNAPSHOT_DIGEST_LENGTH]
    path = experiments_root / SITE_SNAPSHOT_DIRECTORY / f"{site}-{digest}.json"
    if not path.is_file():
        write_text_atomic(path, text)
    return path


class EnvironmentError(ValueError):
    """Raised when a task site cannot be loaded through the site seam."""


@dataclass
class _SiteDataRequest:
    root: Path
    loaded: dict[tuple[str, bool, frozenset[str]], dict[str, Any]] = field(
        default_factory=dict
    )
    public: dict[int, dict[str, Any]] = field(default_factory=dict)


_SITE_DATA_REQUEST: ContextVar[_SiteDataRequest | None] = ContextVar(
    "site_data_request", default=None
)


@contextmanager
def site_data_request(root: str | Path) -> Iterator[None]:
    """Share validated, read-only catalogs until this explicit request ends."""

    workspace = Path(root).resolve()
    current = _SITE_DATA_REQUEST.get()
    if current is not None and current.root == workspace:
        yield
        return
    token = _SITE_DATA_REQUEST.set(_SiteDataRequest(workspace))
    try:
        yield
    finally:
        _SITE_DATA_REQUEST.reset(token)


def _site_app(site: str) -> Any:
    if site not in SITES:
        raise EnvironmentError(f"unsupported site {site!r}")
    return importlib.import_module(f"sites.{site}.app")


def _verified_publisher_description(
    root: str | Path, entity: Mapping[str, Any]
) -> bool:
    source = entity.get("source")
    if (
        not isinstance(source, dict)
        or source.get("metadata_snapshot_basis") != "frozen_youtube_detail"
    ):
        return False
    path = source.get("detail_snapshot_path")
    if not isinstance(path, str):
        return False
    try:
        payload = json.loads(
            resolve_within(Path(root), path, context="publisher snapshot").read_text()
        )
    except (OSError, ValueError):
        return False
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (
        payload.get("provider") == "youtube-video-detail-v1"
        and source.get("page_url")
        == f"https://www.youtube.com/watch?v={payload.get('id')}"
        and hashlib.sha256(serialized.encode()).hexdigest()
        == source.get("metadata_sha256")
        and entity.get("description") == payload.get("description")
        and entity.get("summary") == payload.get("public_summary")
        and entity.get("title") == payload.get("title")
    )


def load_site_data(
    root: str | Path,
    site: str,
    *,
    require_assets: bool = False,
    released_frame_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Load one site's frozen data through its own implementation."""

    workspace = Path(root).resolve()
    frames = tuple(released_frame_paths) if require_assets else ()
    current = _SITE_DATA_REQUEST.get()
    key = (site, require_assets, frozenset(frames))
    if current is not None and current.root == workspace and key in current.loaded:
        return current.loaded[key]
    data = _load_site_data(
        workspace, site, require_assets=require_assets, released_frame_paths=frames
    )
    if current is not None and current.root == workspace:
        current.loaded[key] = data
    return data


def _load_site_data(
    root: Path,
    site: str,
    *,
    require_assets: bool,
    released_frame_paths: Iterable[str],
) -> dict[str, Any]:

    app = _site_app(site)
    try:
        data = app.load_data(
            root,
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
    entity_ids = [
        entity.get("entity_id") for entity in entities if isinstance(entity, dict)
    ]
    if len(entity_ids) != len(entities) or any(
        not isinstance(entity_id, str) or not entity_id for entity_id in entity_ids
    ):
        raise EnvironmentError(f"site {site!r} contains an invalid entity_id")
    if len(entity_ids) != len(set(entity_ids)):
        raise EnvironmentError(f"site {site!r} contains duplicate entity IDs")
    # One real-world entity, one catalog record. Two listings for the same upstream
    # item make the task ambiguous: the Agent can pick the twin the checker rejects.
    # This applies where the upstream URL *is* the item, as a video watch URL is. A
    # retail product page legitimately covers several variants, so a shared
    # `text_provenance.source_url` is not a duplicate and is not checked here.
    upstream_urls: dict[str, str] = {}
    for entity in entities:
        source = entity.get("source")
        page_url = source.get("page_url") if isinstance(source, dict) else None
        if not isinstance(page_url, str) or not page_url:
            continue
        previous = upstream_urls.get(page_url)
        if previous is not None:
            raise EnvironmentError(
                f"site {site!r} lists one upstream item twice: "
                f"{previous} and {entity.get('entity_id')} share {page_url}"
            )
        upstream_urls[page_url] = str(entity.get("entity_id"))

    for entity in entities:
        attributes = entity.get("attributes", {})
        if isinstance(attributes, dict) and any(
            str(key).strip().casefold() in HIDDEN_ROLE_KEYS for key in attributes
        ):
            raise EnvironmentError(f"site {site!r} exposes a hidden benchmark role")
        for key in entity:
            if str(key).strip().casefold() in HIDDEN_ROLE_KEYS:
                raise EnvironmentError(f"site {site!r} exposes a hidden benchmark role")
        public_text = " ".join(
            json.dumps(entity.get(key), ensure_ascii=False)
            if not isinstance(entity.get(key), str)
            else entity.get(key, "")
            for key in (
                "title",
                "name",
                "summary",
                "description",
                "nearby",
                "facts",
                "details",
            )
        ).casefold()
        suspect_phrases = {
            phrase for phrase in LEAKING_PUBLIC_PHRASES if phrase in public_text
        }
        if (
            suspect_phrases <= {"correct answer", "wrong answer", "benchmark"}
            and suspect_phrases
        ):
            # These phrases also occur in rules and curriculum descriptions. Only exact,
            # hash-bound publisher text can disambiguate them from authored leakage.
            other_text = " ".join(
                str(entity.get(key, ""))
                for key in ("name", "nearby", "facts", "details")
            ).casefold()
            if not any(
                phrase in other_text for phrase in suspect_phrases
            ) and _verified_publisher_description(root, entity):
                suspect_phrases = set()
        if suspect_phrases:
            raise EnvironmentError(f"site {site!r} exposes evaluator language")
        role_phrases = {
            phrase for phrase in ROLE_NARRATING_PHRASES if phrase in public_text
        }
        if role_phrases and role_phrases <= {"not the one", "decoy", "the correct one"}:
            other_text = " ".join(
                str(entity.get(key, ""))
                for key in ("name", "nearby", "facts", "details")
            ).casefold()
            if not any(
                phrase in other_text for phrase in role_phrases
            ) and _verified_publisher_description(root, entity):
                role_phrases = set()
        if role_phrases:
            raise EnvironmentError(
                f"site {site!r} lets a listing narrate its benchmark role"
            )
    return data


def site_entity_ids(site_data: dict[str, Any]) -> set[str]:
    """Return the identifiers exposed by one already-validated site dataset."""

    return {str(entity["entity_id"]) for entity in site_data["entities"]}


def entity_display_title(entity: Mapping[str, Any]) -> str:
    """Return the title rendered for one product, video, or place."""

    return str(entity.get("name") or entity.get("title") or "")


def entity_source_url(entity: Mapping[str, Any]) -> str:
    """Return the canonical public page supporting one frozen entity."""

    for key in ("source", "text_provenance", "image_provenance"):
        block = entity.get(key)
        if not isinstance(block, Mapping):
            continue
        for url_field in ("page_url", "source_url", "wikipedia_url"):
            value = block.get(url_field)
            if isinstance(value, str) and _is_https_url(value):
                return value
    return ""


def entity_visual_path(site: str, entity: dict[str, Any]) -> str | None:
    """Return the single public visual asset used by a site entity, if any."""

    key = ENTITY_VISUAL_KEYS.get(site)
    value = entity.get(key) if key else None
    return value if isinstance(value, str) and value else None


def entity_visual_sha256(
    site: str,
    entity: dict[str, Any],
    *,
    root: str | Path,
) -> str | None:
    """Return the sha256 digest of one entity's public visual asset, if any."""

    relative = entity_visual_path(site, entity)
    if not relative:
        return None
    try:
        path = resolve_within(
            root, relative, context=f"{entity.get('entity_id')} visual"
        )
    except (ValueError, FileNotFoundError):
        return None
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_entity_visual_distinctness(
    site: str,
    entities: Iterable[dict[str, Any]],
    *,
    root: str | Path,
    context: str,
) -> None:
    """Reject reused images unless real retail variants share the published image."""

    seen: dict[str, dict[str, Any]] = {}
    for entity in entities:
        entity_id = str(entity.get("entity_id", "<unknown>"))
        digest = entity_visual_sha256(site, entity, root=root)
        if digest is None:
            continue
        previous = seen.get(digest)
        if previous is not None:
            image_url = entity.get("image_provenance", {}).get("image_url")
            if (
                site == "egomart"
                and isinstance(image_url, str)
                and _is_https_url(image_url)
                and image_url == previous.get("image_provenance", {}).get("image_url")
                and (entity_display_title(entity), entity.get("attributes"))
                != (entity_display_title(previous), previous.get("attributes"))
            ):
                continue
            raise EnvironmentError(
                f"{context}: entities {previous['entity_id']} and {entity_id} share the same visual asset"
            )
        seen[digest] = entity


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
        else:
            raise EnvironmentError(f"unsupported site {site!r}")


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
        raise EnvironmentError(
            f"{context}: site visual reuses a released observation frame"
        )
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
    source = entity.get("source")
    if not isinstance(source, dict):
        raise EnvironmentError(f"{context}: independent web source missing")
    page_url = str(source.get("page_url", "")).strip()
    thumbnail_url = str(source.get("thumbnail_url", "")).strip()
    parsed_page = urlparse(page_url)
    parsed_thumbnail = urlparse(thumbnail_url)
    video_id = parse_qs(parsed_page.query).get("v", [""])[0]
    if (
        parsed_page.scheme != "https"
        or parsed_page.netloc not in {"youtube.com", "www.youtube.com"}
        or parsed_page.path != "/watch"
        or not video_id
        or parsed_thumbnail.scheme != "https"
        or parsed_thumbnail.netloc not in {"i.ytimg.com", "img.youtube.com"}
    ):
        raise EnvironmentError(f"{context}: canonical YouTube source missing")
    if any(
        key in source
        for key in (
            "crop_method",
            "retrieval_method",
            "source_recording",
            "timestamp_seconds",
        )
    ):
        raise EnvironmentError(f"{context}: source-video thumbnail forbidden")
    expected_hash = str(
        source.get("thumbnail_sha256") or source.get("sha256") or ""
    ).strip()
    actual_hash = entity_visual_sha256("egomedia", entity, root=root)
    if not _is_sha256(expected_hash) or actual_hash != expected_hash:
        raise EnvironmentError(f"{context}: thumbnail provenance hash mismatch")
    retrieved_at = source.get("metadata_retrieved_at")
    metadata_hash = str(source.get("metadata_sha256") or "").strip()
    if (
        not isinstance(retrieved_at, str)
        or not retrieved_at.strip()
        or not _is_sha256(metadata_hash)
    ):
        raise EnvironmentError(
            f"{context}: frozen upstream metadata provenance missing"
        )
    _validate_media_preview_assets(entity, root, released_frame_paths)


def _validate_media_preview_assets(
    entity: dict[str, Any],
    root: str | Path,
    released_frame_paths: set[str],
) -> None:
    if "public_previews" not in entity:
        return
    _public_media_previews(entity)
    for index, preview in enumerate(entity["public_previews"]):
        context = f"{entity.get('entity_id')} public preview {index}"
        _validate_visual_asset(
            preview,
            "image_path",
            root=root,
            released_frame_paths=released_frame_paths,
            context=context,
        )
        provenance = preview.get("provenance") or {}
        expected = str(provenance.get("sha256") or "")
        path = resolve_within(root, preview["image_path"], context=context)
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise EnvironmentError(
                f"{context}: unsupported public preview image format"
            )
        if (
            not _is_sha256(expected)
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise EnvironmentError(
                f"{context}: public preview provenance hash mismatch"
            )
        if not (provenance.get("retrieved_at") or provenance.get("captured_at")):
            raise EnvironmentError(f"{context}: public preview retrieval time missing")


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
    source = entity.get("source")
    if not isinstance(source, dict):
        raise EnvironmentError(f"{context}: independent place source missing")
    if not _is_https_url(source.get("page_url")) or not _is_https_url(
        source.get("image_url")
    ):
        raise EnvironmentError(f"{context}: independent place page or image missing")
    expected_hash = str(source.get("sha256", "")).strip()
    actual_hash = entity_visual_sha256("egomaps", entity, root=root)
    if not _is_sha256(expected_hash) or actual_hash != expected_hash:
        raise EnvironmentError(f"{context}: place photo provenance hash mismatch")
    address = entity.get("address")
    if not isinstance(address, str) or not address.strip():
        raise EnvironmentError(f"{context}: missing address")

    location = entity.get("location")
    if location is not None:
        if not isinstance(location, dict):
            raise EnvironmentError(f"{context}: invalid location coordinates")
        latitude = location.get("latitude")
        longitude = location.get("longitude")
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
    if routes is not None and (
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
    task: dict[str, Any],
    site_data: dict[str, Any],
    *,
    asset_root: str | Path,
    port: int = 0,
    trace: ActionTrace | None = None,
    initial_site_state: dict[str, Any] | None = None,
) -> SiteRuntime:
    """Create the task's site state and local server behind one interface."""

    site = str(task["site"])
    app = _site_app(site)
    public_data = task_public_site_data(task, site_data)
    runtime = app.create_runtime(
        public_data,
        Path(asset_root).resolve(),
        trace=trace,
        port=port,
    )
    if initial_site_state is not None:
        runtime.state.initialize(initial_site_state)
    return runtime


def task_public_site_data(
    task: dict[str, Any],
    site_data: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the public site data after applying task-local form capabilities."""

    site = str(task["site"])
    public_data = dict(_public_site_projection(site, site_data))
    supported = public_data.get("supported_fields", [])
    required = _required_task_fields(task)
    unknown = required - set(supported)
    if unknown:
        raise EnvironmentError(
            f"{task['id']}: task requires unsupported site fields {sorted(unknown)}"
        )
    public_data["supported_fields"] = [
        field for field in supported if field in required
    ]
    scope = task.get("truth", {}).get("site_scope_entity_ids")
    if isinstance(scope, list) and scope:
        visible_ids = set(scope)
        public_data["entities"] = [
            entity
            for entity in public_data["entities"]
            if str(entity["entity_id"]) in visible_ids
        ]
    public_data["entities"] = [dict(entity) for entity in public_data["entities"]]
    return public_data


def _required_task_fields(task: dict[str, Any]) -> set[str]:
    """Expose only form fields that one task can legitimately ask the Agent to set."""

    goals = [task.get("goal")]
    graft = task.get("graft")
    if isinstance(graft, dict):
        goals.append(graft.get("initial_goal"))
    return {
        str(field)
        for goal in goals
        if isinstance(goal, dict)
        for state in goal.get("acceptable_states", [])
        if isinstance(state, dict) and isinstance(state.get("fields"), dict)
        for field in state["fields"]
    }


def public_site_data(site: str, site_data: Mapping[str, Any]) -> dict[str, Any]:
    """Remove audit metadata before a catalog enters the public mock runtime.

    Provenance remains beside the frozen catalog for reproducible auditing, but it is
    not website content and must never become searchable, renderable, or agent-visible.
    """

    public = dict(_public_site_projection(site, site_data))
    public["entities"] = [dict(entity) for entity in public["entities"]]
    return public


def _public_site_projection(site: str, site_data: Mapping[str, Any]) -> dict[str, Any]:
    current = _SITE_DATA_REQUEST.get()
    scoped = current is not None and any(
        data is site_data for data in current.loaded.values()
    )
    if scoped and id(site_data) in current.public:
        return current.public[id(site_data)]
    public = _public_root_data(site, site_data)
    public["entities"] = [
        _public_entity_data(site, entity) for entity in site_data["entities"]
    ]
    if scoped:
        current.public[id(site_data)] = public
    return public


def _public_root_data(site: str, site_data: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in site_data.items()
        if key not in AUDIT_ONLY_ROOT_KEYS and key != "entities"
    }
    unexpected = sorted(set(public) - PUBLIC_SITE_FIELDS)
    if unexpected:
        raise EnvironmentError(
            f"site {site!r} exposes non-native public root fields {unexpected}"
        )
    return public


def _public_entity_data(site: str, entity: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in entity.items()
        if key not in AUDIT_ONLY_ENTITY_KEYS and not key.endswith("_provenance")
    }
    unexpected = sorted(set(public) - PUBLIC_ENTITY_FIELDS[site])
    if unexpected:
        raise EnvironmentError(
            f"site {site!r} exposes non-native public fields {unexpected}"
        )
    if site == "egomedia" and "public_previews" in public:
        public["public_previews"] = _public_media_previews(entity)
    return public


def _public_media_previews(entity: dict[str, Any]) -> list[dict[str, Any]]:
    previews = entity["public_previews"]
    if not isinstance(previews, list):
        raise EnvironmentError("egomedia public_previews must be a list")
    public_fields = {"image_path", "heading", "source_url", "start_sec", "end_sec"}
    public = []
    for preview in previews:
        if not isinstance(preview, dict) or not public_fields <= preview.keys():
            raise EnvironmentError(
                "egomedia public preview is missing display metadata"
            )
        if any(
            not isinstance(preview[key], str) or not preview[key].strip()
            for key in ("image_path", "heading", "source_url")
        ):
            raise EnvironmentError("egomedia public preview has invalid text metadata")
        source = urlparse(preview["source_url"])
        provenance = preview.get("provenance") or {}
        if not isinstance(provenance, dict):
            raise EnvironmentError(
                "egomedia public preview provenance must be an object"
            )
        kind = preview.get("kind") or provenance.get("kind")
        storyboard = (
            kind == "youtube_storyboard"
            and source.hostname == "i.ytimg.com"
            and source.path.startswith("/sb/")
        )
        watch_frame = (
            kind == "canonical_watch_frame"
            and source.hostname in {"youtube.com", "www.youtube.com"}
            and source.path == "/watch"
            and bool(parse_qs(source.query).get("v"))
        )
        if source.scheme != "https" or not (storyboard or watch_frame):
            raise EnvironmentError(
                "egomedia public preview requires a matching official YouTube source kind"
            )
        parent_source = entity.get("source")
        if not isinstance(parent_source, dict):
            parent_source = {}
        expected_id = parse_qs(
            urlparse(str(parent_source.get("page_url") or "")).query
        ).get("v")
        source_id = (
            [source.path.split("/")[2]]
            if storyboard
            else parse_qs(source.query).get("v")
        )
        if expected_id and source_id != expected_id:
            raise EnvironmentError(
                "egomedia public preview belongs to a different YouTube video"
            )
        times = [preview["start_sec"], preview["end_sec"]]
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in times
            )
            or not 0 <= times[0] <= times[1]
        ):
            raise EnvironmentError("egomedia public preview has an invalid time range")
        if storyboard and times[0] == times[1]:
            raise EnvironmentError(
                "a YouTube storyboard preview requires a nonempty time range"
            )
        heading = preview["heading"].casefold()
        if any(
            phrase in heading
            for phrase in LEAKING_PUBLIC_PHRASES | ROLE_NARRATING_PHRASES
        ):
            raise EnvironmentError("egomedia public preview exposes evaluator language")
        public.append(
            {
                **{key: preview[key] for key in preview if key in public_fields},
                "kind": kind,
            }
        )
    return public
