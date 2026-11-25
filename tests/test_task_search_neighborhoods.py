import copy
import json

import pytest

from egocua.neighborhood import NeighborhoodError, validate_search_neighborhood
from egocua.workspace import write_json_atomic
from tools.catalog import build_task_search_neighborhoods as neighborhoods
from tools.catalog.build_task_search_neighborhoods import (
    _blueprint_final_options,
    _map_search_scope,
    _snapshot_rows,
    _unbuilt_map_graft_requirements,
)


def _place(name: str, kind: str, address: str, summary: str = "") -> dict:
    return {
        "entity_id": name.casefold().replace(" ", "-"),
        "name": name,
        "kind": kind,
        "address": address,
        "summary": summary,
        "facts": {},
    }


def test_map_scope_keeps_brand_or_multi_term_location_matches() -> None:
    cocoberry = _place(
        "Cocoberry — GK1",
        "dessert_shop",
        "Greater Kailash, New Delhi",
    )
    same_location = _place(
        "Giani Ice Cream",
        "dessert_shop",
        "Defence Colony, New Delhi",
    )
    one_common_word = _place(
        "New York Marriott",
        "hotel",
        "New York",
    )

    scoped = _map_search_scope(
        [cocoberry, same_location, one_common_word],
        "Cocoberry",
        "Cocoberry Defence Colony New Delhi",
    )

    assert scoped == [cocoberry, same_location]


def test_map_scope_uses_explicit_place_category() -> None:
    winery = _place("Country Moon Winery", "winery", "Noblesville, Indiana")
    water_landmark = _place("AquaDom", "landmark", "Berlin", "Water attraction")

    scoped = _map_search_scope(
        [winery, water_landmark],
        "Peace Water Winery",
        "Peace Water Winery Fishers Indiana",
    )

    assert scoped == [winery]


def test_blueprint_final_options_expand_equivalent_roles() -> None:
    record = {
        "task_id": "graft-example",
        "family": "graft",
        "construct": {
            "task_blueprint": {
                "final_goal_roles": [["kept", "added"]],
                "environment_role_ids": {
                    "kept": "p_kept",
                    "added": "p_added",
                },
                "equivalent_role_ids": {
                    "added": ["p_added_alt_1", "p_added_alt_2"],
                },
            },
        },
    }

    assert _blueprint_final_options(record) == [
        {"p_added", "p_kept"},
        {"p_added_alt_1", "p_kept"},
        {"p_added_alt_2", "p_kept"},
    ]


def test_blueprint_final_options_expand_refresh_target() -> None:
    record = {
        "task_id": "refresh-example",
        "family": "refresh",
        "construct": {
            "task_blueprint": {
                "final_target_role": "final",
                "environment_role_ids": {"final": "p_final"},
                "equivalent_role_ids": {"final": ["p_final_alt"]},
            },
        },
    }

    assert _blueprint_final_options(record) == [
        {"p_final"},
        {"p_final_alt"},
    ]


def test_unbuilt_map_graft_requirements_isolate_added_role() -> None:
    record = {
        "task_id": "candidate-maps-graft-r-99",
        "family": "graft",
        "construct": {
            "task_blueprint": {
                "initial_goal_roles": [["campus"]],
                "final_goal_roles": [["campus", "library"]],
                "environment_role_ids": {
                    "campus": "l_campus",
                    "library": "l_library",
                },
                "equivalent_role_ids": {"library": ["l_library_alt"]},
            },
        },
    }

    before, after, final = _unbuilt_map_graft_requirements(record)

    assert before == [{"l_campus"}]
    assert after == [{"l_library"}, {"l_library_alt"}]
    assert final == {"l_campus", "l_library", "l_library_alt"}


@pytest.fixture(params=["egomart", "egomedia"])
def captured_neighborhood(request, monkeypatch, tmp_path):
    site = request.param
    task_id = "captured-task"
    upstream_index = {f"upstream-{index}": f"entity-{index}" for index in range(24)}
    entities = [
        {
            "entity_id": entity_id,
            "title": f"Canonical detail {entity_id}",
            "source": {"page_url": f"https://example.com/{entity_id}"},
            "image_path": f"images/{entity_id}.jpg",
            "thumbnail_path": f"images/{entity_id}.jpg",
        }
        for entity_id in upstream_index.values()
    ]
    original_entities = copy.deepcopy(entities)
    directories, upstream_key = neighborhoods.UPSTREAM_CAPTURES[site]
    directory = directories[0] if isinstance(directories, tuple) else directories
    captures = []
    for stage, order in (
        ("before_change", list(upstream_index)),
        ("after_change", list(reversed(upstream_index))),
    ):
        capture = {
            "query": f"query {stage}",
            "provider": "upstream-search-capture",
            "results": [
                {
                    "rank": rank,
                    upstream_key: upstream_id,
                    "title": f"{stage} {upstream_id}",
                }
                for rank, upstream_id in enumerate(order, start=1)
            ],
        }
        captures.append(capture)
        write_json_atomic(tmp_path / directory / f"{task_id}--{stage}.json", capture)
    write_json_atomic(
        tmp_path / "data/sources/upstream_entity_index.json",
        {
            "sites": {site: upstream_index},
        },
    )
    payload = {
        "tasks": [
            {
                "task_id": task_id,
                "family": "graft",
                "construct": {
                    "task_blueprint": {
                        "site": site,
                        "final_goal_roles": [["final"]],
                        "environment_role_ids": {"final": "entity-0"},
                    }
                },
            }
        ]
    }
    monkeypatch.setattr(neighborhoods, "load_tasks", lambda _: [])
    monkeypatch.setattr(
        neighborhoods,
        "load_selection",
        lambda _: {
            "task_set": {"graft": [task_id], "refresh": [], "clarify": []},
        },
    )
    monkeypatch.setattr(
        neighborhoods,
        "iter_manifests",
        lambda _: [
            (tmp_path / "manifest.json", payload),
        ],
    )
    monkeypatch.setattr(
        neighborhoods, "load_site_data", lambda *_: {"entities": entities}
    )
    monkeypatch.setattr(neighborhoods, "_usable_entity", lambda *_: True)

    neighborhoods.build(tmp_path)

    snapshot = json.loads(
        (tmp_path / neighborhoods.SNAPSHOT_DIR / f"{task_id}.json").read_text()
    )
    return snapshot, captures, upstream_key, upstream_index, entities, original_entities


def test_build_preserves_each_capture_title_rank_and_entity_order(
    captured_neighborhood,
) -> None:
    snapshot, captures, upstream_key, upstream_index, _, _ = captured_neighborhood

    assert [
        [(row["rank"], row["title"], row["entity_id"]) for row in stage["results"]]
        for stage in snapshot["rounds"]
    ] == [
        [
            (row["rank"], row["title"], upstream_index[row[upstream_key]])
            for row in capture["results"]
        ]
        for capture in captures
    ]


def test_build_preserves_canonical_entity_details(captured_neighborhood) -> None:
    _, _, _, _, entities, original_entities = captured_neighborhood

    assert entities == original_entities


@pytest.mark.parametrize("declare_source", [True, False])
@pytest.mark.parametrize(
    "additional_stage", [None, "later_activity", "before_change", "", 123]
)
def test_replacement_reuses_a_capture_only_with_its_declared_identity(
    captured_neighborhood, tmp_path, declare_source, additional_stage
) -> None:
    snapshot, _, _, _, entities, _ = captured_neighborhood
    snapshot_path = neighborhoods.SNAPSHOT_DIR / "captured-task.json"
    snapshot.update(retrieved_at="2026-10-16", method="fixture capture")
    ids = [entity["entity_id"] for entity in entities]
    if additional_stage is not None:
        extra_entity = copy.deepcopy(entities[-1])
        extra_entity.update(
            entity_id="extra-entity",
            source={"page_url": "https://example.com/extra-entity"},
            image_path="images/extra-entity.jpg",
            thumbnail_path="images/extra-entity.jpg",
        )
        entities.append(extra_entity)
        ids.append("extra-entity")
        snapshot["rounds"].append(
            {
                "stage": "source_extra",
                "query": "utensils from the later activity",
                "provider": snapshot["provider"],
                "results": [
                    {
                        "rank": 1,
                        "entity_id": "extra-entity",
                        "title": extra_entity["title"],
                        "source_url": "https://example.com/extra-entity",
                        "visual_path": "images/extra-entity.jpg",
                        "mock_detail_path": f"/{'product' if snapshot['site'] == 'egomart' else 'watch'}/extra-entity",
                    }
                ],
            }
        )
    write_json_atomic(tmp_path / snapshot_path, snapshot)
    rounds = [
        {
            "stage": row["stage"],
            "query": row["query"],
            "provider": row.get("provider", snapshot["provider"]),
            "snapshot_path": snapshot_path.as_posix(),
            "result_entity_ids": [result["entity_id"] for result in row["results"]],
            **({"snapshot_task_id": "captured-task"} if declare_source else {}),
        }
        for row in snapshot["rounds"]
    ]
    if additional_stage is not None:
        rounds[-1].update(stage=additional_stage, snapshot_stage="source_extra")
    resolution = {
        "policy_version": "real-search-window-v1",
        "query_rounds": rounds,
        "window_entity_ids": ids,
        "core_neighbor_entity_ids": ids[1:5],
        "related_entity_ids": ids[5:13],
        "ambient_entity_ids": ids[13:],
        "detail_decision_fields": ["size"],
        "rejected": [
            {
                "entity_id": entity_id,
                "ring": "core",
                "reason": "Fixture mismatch",
                "why_plausible": "Fixture alternative",
                "semantic_review": "coding_agent_accepted",
                "failure_predicates": ["not_drinking_cup"],
            }
            for entity_id in ids[1:5]
        ],
    }
    kwargs = dict(
        task_id="replacement-task",
        site=snapshot["site"],
        resolution=resolution,
        available_entities={entity["entity_id"]: entity for entity in entities},
        final_ids={ids[0]},
        root=tmp_path,
    )
    if not declare_source:
        with pytest.raises(NeighborhoodError, match="identity mismatch"):
            validate_search_neighborhood(**kwargs)
    elif additional_stage is not None and additional_stage != "later_activity":
        with pytest.raises(NeighborhoodError, match="invalid or duplicate query stage"):
            validate_search_neighborhood(**kwargs)
    else:
        assert validate_search_neighborhood(**kwargs) == ids


def test_snapshot_preserves_observed_rank_without_renumbering() -> None:
    observed = {"entity-a": {"rank": 7, "title": "Observed listing title"}}
    canonical = {"entity-a": {"title": "Canonical detail title"}}

    rows = _snapshot_rows(["entity-a"], canonical, "egomart", observed_rows=observed)

    assert (rows[0]["rank"], rows[0]["title"]) == (7, "Observed listing title")


def test_ring_proposals_preserve_real_reasons_without_inventing_missing_ones():
    ids = [f"entity-{index}" for index in range(14)]
    previous = {
        "entity-1": {
            "reason": "It is a storage jar.",
            "failure_predicates": ["not_drinking_cup"],
        }
    }
    core, related, ambient, rejected = neighborhoods._rings(
        window=ids,
        final_answer_ids={ids[0]},
        before_results=set(ids),
        after_results=set(ids),
        previous=previous,
    )
    assert core == ["entity-1"]
    assert len(related) >= 8
    assert set(core + related + ambient) == set(ids[1:])
    by_id = {row["entity_id"]: row for row in rejected}
    assert by_id["entity-1"]["reason"] == "It is a storage jar."
    assert by_id["entity-1"]["failure_predicates"] == ["not_drinking_cup"]
    assert all(
        not row["reason"] and not row["failure_predicates"]
        for row in rejected
        if row["entity_id"] != "entity-1"
    )
    assert all(row["semantic_review"] == "pending_coding_agent" for row in rejected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ring", "related"),
        ("failure_predicates", None),
        ("failure_predicates", []),
        ("failure_predicates", [" "]),
        ("reason", ""),
        ("why_plausible", ""),
    ],
)
def test_neighborhood_still_requires_evidence_for_every_core_neighbor(
    captured_neighborhood, tmp_path, field, value
):
    snapshot, _, _, _, entities, _ = captured_neighborhood
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    resolution = manifest["tasks"][0]["construct"]["catalog_resolution"]
    for row in resolution["rejected"]:
        row.update(
            why_plausible="Same product category.",
            reason="The listing is a storage jar.",
            semantic_review="coding_agent_accepted",
            failure_predicates=["not_drinking_cup"],
        )
    resolution["rejected"][0][field] = value
    with pytest.raises(
        NeighborhoodError, match="rejection facts and failure predicates"
    ):
        validate_search_neighborhood(
            task_id=snapshot["task_id"],
            site=snapshot["site"],
            resolution=resolution,
            available_entities={entity["entity_id"]: entity for entity in entities},
            final_ids={"entity-0"},
            root=tmp_path,
        )


@pytest.mark.parametrize(
    "semantic_review",
    [None, "constructor_checked", "pending_coding_agent", "coding_agent_accepted"],
)
def test_neighborhood_checks_rejection_facts_without_semantic_admission(
    captured_neighborhood, tmp_path, semantic_review
):
    snapshot, _, _, _, entities, _ = captured_neighborhood
    resolution = json.loads((tmp_path / "manifest.json").read_text())["tasks"][0][
        "construct"
    ]["catalog_resolution"]
    for row in resolution["rejected"]:
        row.update(
            why_plausible="Same product category.",
            reason="The listing is a storage jar.",
            failure_predicates=["not_drinking_cup"],
        )
        if semantic_review is None:
            row.pop("semantic_review", None)
        else:
            row["semantic_review"] = semantic_review

    assert (
        validate_search_neighborhood(
            task_id=snapshot["task_id"],
            site=snapshot["site"],
            resolution=resolution,
            available_entities={entity["entity_id"]: entity for entity in entities},
            final_ids={"entity-0"},
            root=tmp_path,
        )
        == resolution["window_entity_ids"]
    )


def test_single_real_negative_does_not_require_padding():
    core, related, ambient, _ = neighborhoods._rings(
        window=["answer", "near"],
        final_answer_ids={"answer"},
        before_results={"answer", "near"},
        after_results={"answer", "near"},
        previous={},
    )
    assert core == ["near"]
    assert related == ambient == []


@pytest.mark.parametrize("captured_query", ["same", "stale query"])
def test_same_reviewed_query_reuses_primary_capture(tmp_path, captured_query):
    import json

    directory, upstream_key = neighborhoods.UPSTREAM_CAPTURES["egomedia"]
    directory = directory[0] if isinstance(directory, tuple) else directory
    capture_dir = tmp_path / directory
    capture_dir.mkdir(parents=True)
    (capture_dir / "task--before_change.json").write_text(
        json.dumps({"query": captured_query, "results": [{upstream_key: "upstream"}]})
    )
    reviews = tmp_path / "data/sources/search_query_reviews.json"
    reviews.parent.mkdir(parents=True, exist_ok=True)
    reviews.write_text(
        json.dumps({"tasks": {"task": {"before_query": "same", "after_query": "same"}}})
    )
    if captured_query != "same":
        with pytest.raises(ValueError, match="reviewed shared query"):
            neighborhoods._captured_round(
                tmp_path,
                task_id="task",
                site="egomedia",
                stage="after_change",
                upstream_index={"upstream": "entity"},
            )
        return
    _, rows = neighborhoods._captured_round(
        tmp_path,
        task_id="task",
        site="egomedia",
        stage="after_change",
        upstream_index={"upstream": "entity"},
    )
    assert list(rows) == ["entity"]


def test_small_fully_reviewed_real_window_is_valid(captured_neighborhood, tmp_path):
    snapshot, _, _, _, entities, _ = captured_neighborhood
    resolution = json.loads((tmp_path / "manifest.json").read_text())["tasks"][0][
        "construct"
    ]["catalog_resolution"]
    kept = resolution["window_entity_ids"][:2]
    rows = {
        row["entity_id"]: row for rnd in snapshot["rounds"] for row in rnd["results"]
    }
    for rnd in snapshot["rounds"]:
        rnd["results"] = [
            {**rows[eid], "rank": index + 1} for index, eid in enumerate(kept)
        ]
    write_json_atomic(
        tmp_path / resolution["query_rounds"][0]["snapshot_path"], snapshot
    )
    for rnd in resolution["query_rounds"]:
        rnd["result_entity_ids"] = kept
    resolution.update(
        window_entity_ids=kept,
        core_neighbor_entity_ids=[kept[1]],
        related_entity_ids=[],
        ambient_entity_ids=[],
        rejected=[
            {
                "entity_id": kept[1],
                "ring": "core",
                "reason": "Wrong material",
                "why_plausible": "Same kind of tool",
                "failure_predicates": ["wrong_material"],
                "semantic_review": "coding_agent_accepted",
            }
        ],
    )
    assert (
        validate_search_neighborhood(
            task_id=snapshot["task_id"],
            site=snapshot["site"],
            resolution=resolution,
            available_entities={entity["entity_id"]: entity for entity in entities},
            final_ids={kept[0]},
            root=tmp_path,
        )
        == kept
    )
