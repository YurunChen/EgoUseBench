"""Build task-local search windows from the shared real-source catalogs.

This script does not invent listings.  It searches the same public fields used by
the mock sites, keeps the complete first result window for the before/after query,
and records every upstream source URL in an auditable snapshot.  Ring assignments
are candidate proposals; Coding-Agent semantic review remains a separate gate.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping

from egocua.environment import (
    EnvironmentError,
    entity_display_title,
    entity_source_url,
    entity_visual_path,
    load_site_data,
    validate_referenced_site_assets,
)
from egocua.neighborhood import (
    NEIGHBORHOOD_POLICY_VERSION,
    TARGET_RELATED_RESULTS,
)
from egocua.schema import load_tasks
from egocua.selection import load_selection, task_ids
from egocua.sources import iter_manifests
from egocua.truth import Truth
from egocua.workspace import write_json_atomic
from sites.search import flatten_public_text, rank_records, tokenize

SNAPSHOT_DIR = Path("data/sources/search_snapshots")
PROVIDER = "frozen-real-source-catalog-bm25-v2"
UPSTREAM_CAPTURES = {
    "egomart": (
        (
            Path("data/sources/upstream_search_snapshots/amazon"),
            Path("data/sources/upstream_search_snapshots/abo"),
        ),
        "asin",
    ),
    "egomedia": (Path("data/sources/upstream_search_snapshots/youtube"), "youtube_id"),
}
PAGE_SIZE = {"egomart": 24, "egomedia": 20, "egomaps": 15}
MAX_PAGE_SIZE = {"egomart": 48, "egomedia": 40, "egomaps": 30}
ID_KEY = {"egomart": "item_id", "egomedia": "video_id", "egomaps": "place_id"}
SEARCH_KEYS = {
    "egomart": ("name", "brand", "attributes"),
    "egomedia": ("title", "channel_title", "kind", "summary", "details", "chapters"),
    "egomaps": ("name", "kind", "address", "summary", "facts"),
}
DETAIL_FIELDS = {
    "egomart": ["name", "brand", "attributes", "price"],
    "egomedia": ["title", "channel_title", "summary", "details", "chapters"],
    "egomaps": ["name", "address", "location", "summary", "facts"],
}
STOPWORDS = {
    "a",
    "absolute",
    "an",
    "and",
    "at",
    "beginner",
    "beginners",
    "by",
    "complete",
    "easy",
    "for",
    "from",
    "guide",
    "how",
    "in",
    "into",
    "make",
    "making",
    "my",
    "of",
    "on",
    "one",
    "or",
    "steps",
    "the",
    "this",
    "to",
    "tutorial",
    "ultimate",
    "using",
    "video",
    "with",
    "your",
}


def _usable_entity(root: Path, site: str, entity: dict[str, Any]) -> bool:
    if not entity_source_url(entity) or not entity_visual_path(site, entity):
        return False
    try:
        validate_referenced_site_assets(site, [entity], root=root)
    except EnvironmentError:
        return False
    return True


def _entity_id(entity: Mapping[str, Any]) -> str:
    return str(entity["entity_id"])


def _unique_ids_for_refresh(task_id: str, value: Any, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"{task_id}: {field} must contain unique entity IDs")
    return list(value)


def _title(entity: Mapping[str, Any]) -> str:
    return entity_display_title(entity)


def _search_fields(site: str, entity: Mapping[str, Any]) -> list[tuple[str, float]]:
    fields: list[tuple[str, float]] = []
    for key in SEARCH_KEYS[site]:
        text = flatten_public_text(entity.get(key))
        if not text:
            continue
        weight = (
            3.0
            if key in {"name", "title"}
            else 2.0
            if key in {"brand", "channel_title", "address", "summary"}
            else 1.0
        )
        fields.append((text, weight))
    return fields


def _map_search_scope(
    entities: list[dict[str, Any]],
    family_query: str,
    round_query: str,
) -> list[dict[str, Any]]:
    """Keep only place families a real local-search query could plausibly return.

    A site-wide BM25 ranker is useful only after query admission. Otherwise a
    single common location word can pull arbitrary global landmarks into a local
    result window. Admission uses public query and listing text only: an explicit
    category token admits that category; otherwise the leading business token or
    two non-leading round terms must match.
    """

    family_tokens = list(dict.fromkeys(tokenize(family_query)))
    round_tokens = list(dict.fromkeys(tokenize(round_query)))
    if not family_tokens:
        return []
    public_tokens = {
        _entity_id(entity): set(
            token
            for value, _ in _search_fields("egomaps", entity)
            for token in tokenize(value)
        )
        for entity in entities
    }
    kind_tokens = {
        _entity_id(entity): set(tokenize(str(entity.get("kind") or "")))
        for entity in entities
    }
    category_terms = {
        token
        for token in family_tokens
        if any(token in values for values in kind_tokens.values())
    }
    if category_terms:
        return [
            entity
            for entity in entities
            if kind_tokens[_entity_id(entity)] & category_terms
        ]

    anchor = family_tokens[0]
    detail_terms = set(round_tokens) - {anchor}
    return [
        entity
        for entity in entities
        if anchor in public_tokens[_entity_id(entity)]
        or len(public_tokens[_entity_id(entity)] & detail_terms) >= 2
    ]


def _query_for_entities(
    ids: Iterable[str], by_id: Mapping[str, Mapping[str, Any]]
) -> str:
    titles = [_title(by_id[entity_id]) for entity_id in ids]
    token_rows = [
        [
            token
            for token in tokenize(title)
            if token not in STOPWORDS and len(token) > 1
        ]
        for title in titles
    ]
    frequency = Counter(token for row in token_rows for token in set(row))
    common = [
        token for token, count in frequency.most_common() if count == len(token_rows)
    ]
    chosen = common[:5]
    # Interleave distinct title terms so a multi-entity query does not become an
    # exact-title query for whichever entity happened to be listed first.
    positions = [0] * len(token_rows)
    while len(chosen) < 32:
        added = False
        for index, row in enumerate(token_rows):
            while positions[index] < len(row) and row[positions[index]] in chosen:
                positions[index] += 1
            if positions[index] >= len(row):
                continue
            chosen.append(row[positions[index]])
            positions[index] += 1
            added = True
            if len(chosen) == 32:
                break
        if not added:
            break
    return " ".join(chosen) or " ".join(titles)


def _selected_options(states: Iterable[Mapping[str, Any]]) -> list[set[str]]:
    return [
        {str(entity_id) for entity_id in state["selected_entity_ids"]}
        for state in states
    ]


def _stage_requirements(
    task: Mapping[str, Any],
) -> tuple[list[set[str]], list[set[str]], set[str], set[str]]:
    """Return alternative complete result sets for each search stage.

    Category-level tasks enumerate several equally valid website states. A natural
    query only has to expose one complete valid state, not every interchangeable
    listing in the global catalog. The compiled task is later restricted to the
    valid states that are actually present in the frozen two-query window.
    """

    truth = Truth.of(task)
    before_options: list[set[str]]
    after_options: list[set[str]]
    if truth.track == "graft":
        before_options = _selected_options(truth.initial_acceptable_states)
        invalidated = set(truth.invalidated_referent_ids)
        if invalidated:
            before_options = [
                option for option in before_options if option & invalidated
            ]
        after_options = []
        for option in _selected_options(truth.final_acceptable_states):
            introduced = option - set(truth.initial_referent_ids)
            after_options.append(introduced or option)
    elif truth.track == "refresh":
        if task["refresh"]["policy"] == "retain" and truth.tempting_referent_ids:
            # A retain task may deliberately begin with several plausible web
            # mappings and reveal the exact identity only in V1. The first query
            # must expose a real provisional alternative; the final target must
            # be reachable after the later visual evidence.
            before_options = [
                set(truth.initial_referent_ids),
                *({entity_id} for entity_id in truth.tempting_referent_ids),
            ]
        else:
            before_options = [set(truth.initial_referent_ids)]
        after_options = _selected_options(truth.final_acceptable_states)
    else:
        # Both physical referents must remain plausible before the single allowed
        # clarification question; the clarified answer then drives the second query.
        before_options = [set(truth.candidate_ids)]
        after_options = _selected_options(truth.final_acceptable_states)
    semantic = (
        set(truth.initial_referent_ids)
        | set(truth.final_referent_ids)
        | set(truth.tempting_referent_ids)
        | set(truth.candidate_ids)
    )
    return before_options, after_options, semantic, set(truth.final_referent_ids)


def _rank_window(
    site: str,
    entities: list[dict[str, Any]],
    query: str,
    required_options: Iterable[set[str]],
    by_id: Mapping[str, Mapping[str, Any]],
    *,
    allow_derived_query: bool = True,
) -> tuple[str, list[str]]:
    size = PAGE_SIZE[site]
    options = [set(option) for option in required_options if option]
    if not options:
        raise ValueError("search stage has no non-empty acceptable result state")

    def reaches_any(ids: Iterable[str]) -> bool:
        reached = set(ids)
        return any(option <= reached for option in options)

    def ranked(value: str, limit: int = size) -> list[str]:
        return [
            _entity_id(entity)
            for entity in rank_records(
                entities,
                value,
                lambda row: _search_fields(site, row),
            )[:limit]
        ]

    result = ranked(query)
    if reaches_any(result):
        return query, result
    result = ranked(query, MAX_PAGE_SIZE[site])
    if reaches_any(result):
        return query, result
    if not allow_derived_query:
        missing = [sorted(option - set(result)) for option in options]
        raise ValueError(
            f"natural query cannot reach any acceptable real-listing state {missing}; "
            "repair the query, task, or source"
        )
    required = min(options, key=lambda option: (len(option), sorted(option)))
    derived = _query_for_entities(required, by_id)
    result = ranked(derived)
    if reaches_any(result):
        return derived, result
    result = ranked(derived, MAX_PAGE_SIZE[site])
    if reaches_any(result):
        return derived, result
    # Some tasks intentionally accept several real variants.  Build the shortest
    # discriminative category query we can from public titles; this changes the
    # query, never the result ordering or membership.
    document_frequency: Counter[str] = Counter()
    for entity in entities:
        document_frequency.update(
            set(
                token
                for value, _ in _search_fields(site, entity)
                for token in tokenize(value)
                if token not in STOPWORDS and len(token) > 1
            )
        )
    token_rows = [
        [
            token
            for token in tokenize(_title(by_id[entity_id]))
            if token not in STOPWORDS and len(token) > 1
        ]
        for entity_id in required
    ]
    shared_frequency = Counter(token for row in token_rows for token in set(row))
    shared = [
        token
        for token, count in shared_frequency.most_common()
        if count >= max(2, len(token_rows) // 2)
    ]
    for terms_per_entity in (1, 2, 3):
        discriminative = shared[:3]
        for row in token_rows:
            ordered = sorted(
                set(row),
                key=lambda value: (document_frequency[value], -len(value), value),
            )
            for token in ordered[:terms_per_entity]:
                if token not in discriminative:
                    discriminative.append(token)
        fitted = " ".join(discriminative[:24])
        result = ranked(fitted, MAX_PAGE_SIZE[site])
        if reaches_any(result):
            return fitted, result
    missing = [sorted(option - set(result)) for option in options]
    raise ValueError(
        f"query cannot reach any acceptable real-listing state {missing}; "
        "change the task or source"
    )


def _previous_rejections(record: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = (
        (record.get("construct") or {})
        .get("catalog_resolution", {})
        .get("rejected", [])
    )
    return {
        str(row["entity_id"]): dict(row)
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("entity_id"), str)
    }


def _rings(
    *,
    window: list[str],
    final_answer_ids: set[str],
    before_results: set[str],
    after_results: set[str],
    previous: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
    # A core neighbor is wrong for the *final* state, not necessarily absent from
    # the task's causal history. Earlier targets, tempting Refresh choices, and
    # the unchosen Clarify branch are the most natural hard negatives because a
    # shallow Agent can genuinely commit to them before the update. Only entities
    # accepted by a final state are excluded from the negative pool.
    candidates = [
        entity_id for entity_id in window if entity_id not in final_answer_ids
    ]
    preferred = [entity_id for entity_id in previous if entity_id in candidates]
    buckets = [
        [value for value in candidates if value in before_results - after_results],
        [value for value in candidates if value in after_results - before_results],
        [value for value in candidates if value in before_results & after_results],
    ]
    core = list(preferred)
    for values in buckets:
        # Propose only from an observed bucket, never pad an empty bucket.
        if values and not set(values) & set(core):
            core.append(values[0])
    related = [value for value in candidates if value not in core][
        :TARGET_RELATED_RESULTS
    ]
    ambient = [value for value in candidates if value not in set(core) | set(related)]
    rejected: list[dict[str, Any]] = []
    for entity_id in core:
        prior = previous.get(entity_id)
        reason = str((prior or {}).get("reason") or "").strip()
        rejected.append(
            {
                "entity_id": entity_id,
                "ring": "core",
                "reason": reason,
                "failure_predicates": list((prior or {}).get("failure_predicates", [])),
                "semantic_review": "pending_coding_agent",
            }
        )
    return core, related, ambient, rejected


def _snapshot_rows(
    ids: list[str],
    by_id: Mapping[str, Mapping[str, Any]],
    site: str,
    *,
    observed_rows: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "rank": observed_rows[entity_id]["rank"]
            if observed_rows is not None
            else rank,
            "entity_id": entity_id,
            "title": (
                observed_rows[entity_id]["title"]
                if observed_rows is not None
                else _title(by_id[entity_id])
            ),
            "source_url": entity_source_url(by_id[entity_id]),
            "visual_path": entity_visual_path(site, dict(by_id[entity_id])),
            "mock_detail_path": f"/{'product' if site == 'egomart' else 'watch' if site == 'egomedia' else 'place'}/{entity_id}",
        }
        for rank, entity_id in enumerate(ids, start=1)
    ]


def _blueprint_final_options(record: Mapping[str, Any]) -> list[set[str]]:
    construct = record.get("construct") or {}
    blueprint = construct.get("task_blueprint") or {}
    role_ids = blueprint.get("environment_role_ids") or {}
    track = str(record.get("family") or "")
    if track == "graft":
        role_options = [
            {str(role) for role in state}
            for state in blueprint.get("final_goal_roles", [])
        ]
    elif track == "refresh":
        role_options = [{str(blueprint.get("final_target_role") or "")}]
    else:
        role_options = [{str(blueprint.get("hidden_target_role") or "")}]
    roles = set().union(*role_options) if role_options else set()
    if not roles or "" in roles or not roles <= set(role_ids):
        raise ValueError(f"{record.get('task_id')}: final role bindings are incomplete")
    equivalent_ids = blueprint.get("equivalent_role_ids") or {}
    options: list[set[str]] = []
    for role_option in role_options:
        entity_options = [
            [
                str(role_ids[role]),
                *[str(value) for value in equivalent_ids.get(role, [])],
            ]
            for role in sorted(role_option)
        ]
        for chosen in product(*entity_options):
            state = set(chosen)
            if state not in options:
                options.append(state)
    return options


def _unbuilt_map_graft_requirements(
    record: Mapping[str, Any],
) -> tuple[list[set[str]], list[set[str]], set[str]]:
    """Resolve first-build map requirements directly from a frozen Graft blueprint."""

    task_id = str(record.get("task_id") or "")
    if record.get("family") != "graft":
        raise ValueError(f"{task_id}: an unbuilt map task must be a Graft task")
    blueprint = (record.get("construct") or {}).get("task_blueprint") or {}
    role_ids = blueprint.get("environment_role_ids") or {}
    equivalents = blueprint.get("equivalent_role_ids") or {}

    def expand(field: str) -> list[set[str]]:
        states = blueprint.get(field)
        if not isinstance(states, list) or not states:
            raise ValueError(f"{task_id}: {field} is missing")
        options: list[set[str]] = []
        for state in states:
            if (
                not isinstance(state, list)
                or not state
                or any(role not in role_ids for role in state)
            ):
                raise ValueError(f"{task_id}: {field} contains an unbound role")
            choices = [
                [
                    str(role_ids[role]),
                    *[str(value) for value in equivalents.get(role, [])],
                ]
                for role in state
            ]
            for chosen in product(*choices):
                option = set(chosen)
                if option not in options:
                    options.append(option)
        return options

    before_options = expand("initial_goal_roles")
    final_options = expand("final_goal_roles")
    initial_ids = set().union(*before_options)
    after_options = [option - initial_ids or option for option in final_options]
    return before_options, after_options, set().union(*final_options)


def _captured_round(
    root: Path,
    *,
    task_id: str,
    site: str,
    stage: str,
    upstream_index: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    directories, key = UPSTREAM_CAPTURES[site]
    directories = directories if isinstance(directories, tuple) else (directories,)
    candidates = [
        root / directory / f"{task_id}--{stage}.json" for directory in directories
    ]
    reused_query = None
    reviews_path = root / "data/sources/search_query_reviews.json"
    if (
        stage == "after_change"
        and not any(path.is_file() for path in candidates)
        and reviews_path.is_file()
    ):
        review = json.loads(reviews_path.read_text()).get("tasks", {}).get(task_id, {})
        before = str(review.get("before_query") or "").strip()
        after = str(review.get("after_query") or "").strip()
        if before and before == after:
            reused_query = before
            candidates = [
                root / directory / f"{task_id}--before_change.json"
                for directory in directories
            ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        raise ValueError(f"{task_id}: missing complete upstream {stage} capture")
    if len(existing) != 1:
        raise ValueError(f"{task_id}: multiple upstream providers captured {stage}")
    path = existing[0]
    payload = json.loads(path.read_text())
    if reused_query is not None and payload.get("query") != reused_query:
        raise ValueError(
            f"{task_id}: primary capture does not match the reviewed shared query"
        )
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("results", []):
        upstream_id = str(row.get(key) or "")
        if not upstream_id or upstream_id not in upstream_index:
            raise ValueError(
                f"{task_id}/{stage}: captured entity {upstream_id!r} is not materialized"
            )
        entity_id = str(upstream_index[upstream_id])
        rows.setdefault(entity_id, row)
    if not rows:
        raise ValueError(f"{task_id}/{stage}: real capture has no usable rows")
    return payload, rows


def build(
    root: Path,
    *,
    only_sites: set[str] | None = None,
    only_tasks: set[str] | None = None,
) -> dict[str, int]:
    built_tasks = {str(task["id"]): task for task in load_tasks(root / "data/tasks")}
    selected_ids = set(task_ids(load_selection(root)))
    records: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for path, payload in iter_manifests(root):
        for record in payload.get("tasks", []):
            if isinstance(record, dict) and str(record.get("task_id")) in selected_ids:
                records[str(record["task_id"])] = (path, payload, record)
    if set(records) != selected_ids:
        raise ValueError("selected tasks and source records do not match")
    if only_tasks is not None:
        unknown = only_tasks - selected_ids
        if unknown:
            raise ValueError(f"unknown selected task IDs: {sorted(unknown)}")
    sites_by_task = {
        task_id: str(
            ((record.get("construct") or {}).get("task_blueprint") or {}).get("site")
            or built_tasks.get(task_id, {}).get("site")
            or "egomart"
        )
        for task_id, (_, _, record) in records.items()
    }
    indexes: dict[str, dict[str, dict[str, Any]]] = {}
    for site in set(sites_by_task.values()):
        usable = [
            entity
            for entity in load_site_data(root, site)["entities"]
            if _usable_entity(root, site, entity)
        ]
        indexes[site] = {_entity_id(entity): entity for entity in usable}

    upstream_entity_index = json.loads(
        (root / "data/sources/upstream_entity_index.json").read_text()
    )["sites"]

    touched: dict[Path, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    processed = 0
    retrieved_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    for task_id in sorted(records):
        if only_tasks is not None and task_id not in only_tasks:
            continue
        path, payload, record = records[task_id]
        site = sites_by_task[task_id]
        if only_sites is not None and site not in only_sites:
            continue
        if site == "egomaps":
            # Place entities come from individually audited location pages. Freeze
            # their task-independent BM25 window here so catalog expansion cannot
            # leave Maps snapshots and the runtime search order out of sync.
            task = built_tasks.get(task_id)
            by_id = indexes[site]
            entities = list(by_id.values())
            resolution = (record.get("construct") or {}).get("catalog_resolution")
            if not isinstance(resolution, dict):
                raise ValueError(
                    f"{task_id}: map task lacks a reviewed real-place window"
                )
            rounds = resolution.get("query_rounds")
            if not isinstance(rounds, list) or len(rounds) != 2:
                raise ValueError(f"{task_id}: map task requires two declared queries")
            by_stage = {
                str(row.get("stage")): row for row in rounds if isinstance(row, Mapping)
            }
            before_declared_query = str(by_stage["before_change"]["query"])
            after_declared_query = str(by_stage["after_change"]["query"])
            before_entities = _map_search_scope(
                entities,
                before_declared_query,
                before_declared_query,
            )
            after_entities = _map_search_scope(
                entities,
                before_declared_query,
                after_declared_query,
            )
            if task is None:
                before_options, after_options, final_ids = (
                    _unbuilt_map_graft_requirements(record)
                )
            else:
                before_options, after_options, _, final_ids = _stage_requirements(task)
            before_query, before_results = _rank_window(
                site,
                before_entities,
                before_declared_query,
                before_options,
                by_id,
                allow_derived_query=False,
            )
            after_query, after_results = _rank_window(
                site,
                after_entities,
                after_declared_query,
                after_options,
                by_id,
                allow_derived_query=False,
            )
            window = list(dict.fromkeys([*before_results, *after_results]))
            if task is not None:
                final_ids = set(Truth.of(task).final_referent_ids)
            core = _unique_ids_for_refresh(
                task_id,
                resolution.get("core_neighbor_entity_ids"),
                "core neighbors",
            )
            if not set(core) <= set(window) - final_ids:
                missing = sorted(set(core) - (set(window) - final_ids))
                raise ValueError(
                    f"{task_id}: reviewed place core neighbors are absent from the search window: {missing}"
                )
            remaining = [
                entity_id
                for entity_id in window
                if entity_id not in final_ids and entity_id not in set(core)
            ]
            related = remaining[:TARGET_RELATED_RESULTS]
            ambient = remaining[TARGET_RELATED_RESULTS:]
            snapshot_path = SNAPSHOT_DIR / f"{task_id}.json"
            snapshot = {
                "schema_version": 1,
                "task_id": task_id,
                "site": site,
                "provider": PROVIDER,
                "retrieved_at": retrieved_at,
                "method": (
                    "Exact ordered union of query-admissible BM25 place-result windows over "
                    "audited upstream entities. Admission uses public query/category text; "
                    "neither task ID nor answer role enters admission or ranking."
                ),
                "rounds": [
                    {
                        "stage": "before_change",
                        "query": before_query,
                        "results": _snapshot_rows(before_results, by_id, site),
                    },
                    {
                        "stage": "after_change",
                        "query": after_query,
                        "results": _snapshot_rows(after_results, by_id, site),
                    },
                ],
            }
            write_json_atomic(root / snapshot_path, snapshot)
            resolution["query_rounds"] = [
                {
                    "stage": "before_change",
                    "query": before_query,
                    "provider": PROVIDER,
                    "snapshot_path": snapshot_path.as_posix(),
                    "result_entity_ids": before_results,
                },
                {
                    "stage": "after_change",
                    "query": after_query,
                    "provider": PROVIDER,
                    "snapshot_path": snapshot_path.as_posix(),
                    "result_entity_ids": after_results,
                },
            ]
            resolution["window_entity_ids"] = window
            resolution["related_entity_ids"] = related
            resolution["ambient_entity_ids"] = ambient
            touched[path] = payload
            counts[site] += 1
            processed += 1
            continue
        by_id = indexes[site]
        site_upstream_index = upstream_entity_index.get(site) or {}
        before_capture, before_rows = _captured_round(
            root,
            task_id=task_id,
            site=site,
            stage="before_change",
            upstream_index=site_upstream_index,
        )
        after_capture, after_rows = _captured_round(
            root,
            task_id=task_id,
            site=site,
            stage="after_change",
            upstream_index=site_upstream_index,
        )
        before_query = str(before_capture["query"])
        after_query = str(after_capture["query"])
        before_results = list(before_rows)
        after_results = list(after_rows)
        window = list(dict.fromkeys([*before_results, *after_results]))
        final_options = _blueprint_final_options(record)
        missing_answers = set().union(*final_options) - set(window)
        if missing_answers:
            raise ValueError(
                f"{task_id}: declared acceptable entities are absent from the real captures: "
                f"{sorted(missing_answers)}"
            )
        final_ids = set().union(*final_options)
        role_ids = set(
            str(value)
            for value in ((record.get("construct") or {}).get("task_blueprint") or {})
            .get("environment_role_ids", {})
            .values()
        )
        excluded_semantic = sorted(role_ids - set(window))
        try:
            core, related, ambient, rejected = _rings(
                window=window,
                final_answer_ids=final_ids,
                before_results=set(before_results),
                after_results=set(after_results),
                previous=_previous_rejections(record),
            )
        except ValueError as exc:
            raise ValueError(f"{task_id}: {exc}") from exc
        snapshot_path = SNAPSHOT_DIR / f"{task_id}.json"
        snapshot = {
            "schema_version": 1,
            "task_id": task_id,
            "site": site,
            "provider": str(
                before_capture.get("provider") or "upstream-search-capture"
            ),
            "retrieved_at": retrieved_at,
            "method": (
                "Exact ordered union of complete upstream search-result captures. Every row "
                "is materialized once in the shared Mock catalog. Runtime Mock search remains "
                "task-independent BM25 and receives no task ID or answer role."
            ),
            "rounds": [
                {
                    "stage": "before_change",
                    "query": before_query,
                    "results": _snapshot_rows(
                        before_results,
                        by_id,
                        site,
                        observed_rows=before_rows,
                    ),
                },
                {
                    "stage": "after_change",
                    "query": after_query,
                    "results": _snapshot_rows(
                        after_results,
                        by_id,
                        site,
                        observed_rows=after_rows,
                    ),
                },
            ],
        }
        write_json_atomic(root / snapshot_path, snapshot)
        construct = record.setdefault("construct", {})
        construct["catalog_resolution"] = {
            "policy_version": NEIGHBORHOOD_POLICY_VERSION,
            "query_rounds": [
                {
                    "stage": "before_change",
                    "query": before_query,
                    "provider": str(
                        before_capture.get("provider") or "upstream-search-capture"
                    ),
                    "snapshot_path": snapshot_path.as_posix(),
                    "result_entity_ids": before_results,
                },
                {
                    "stage": "after_change",
                    "query": after_query,
                    "provider": str(
                        after_capture.get("provider") or "upstream-search-capture"
                    ),
                    "snapshot_path": snapshot_path.as_posix(),
                    "result_entity_ids": after_results,
                },
            ],
            "window_entity_ids": window,
            "semantic_scope_policy": "restrict-acceptable-states-to-real-window-v1",
            "excluded_unreachable_semantic_entity_ids": excluded_semantic,
            "core_neighbor_entity_ids": core,
            "related_entity_ids": related,
            "ambient_entity_ids": ambient,
            "detail_decision_fields": DETAIL_FIELDS[site],
            "rejected": rejected,
            "semantic_review": {
                "status": "pending_coding_agent",
                "required_checks": [
                    "usefulness",
                    "scene_update_website_consistency",
                    "continuous_change",
                    "deterministic_verifiability",
                    "source_grounded_core_neighbors",
                    "per_neighbor_rejection_evidence",
                ],
            },
        }
        touched[path] = payload
        counts[site] += 1
        processed += 1
    for path, payload in touched.items():
        write_json_atomic(path, payload)
    counts["tasks"] = processed
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--site", action="append", choices=sorted(PAGE_SIZE))
    parser.add_argument(
        "--task",
        action="append",
        help="Rebuild only this selected task ID; repeat for multiple tasks.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                Path(args.root).resolve(),
                only_sites=set(args.site) if args.site else None,
                only_tasks=set(args.task) if args.task else None,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
