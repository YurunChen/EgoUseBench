from __future__ import annotations

import argparse
import copy
import json
import re
import urllib.parse
import urllib.request
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from statistics import mean, median
from typing import Any

from egocua.checker import acceptable_states, check_run, composite_event_coverage
from egocua.development_gate import DevelopmentEvidence
from egocua.environment import load_site_data
from egocua.execution import RunConfig, RunContext, run_experiment
from egocua.experiment import TASK_REVIEW_PLAN, ExperimentPlan, load_experiment_plan
from egocua.protocol import ProtocolError, RunSession, first_release_trigger
from egocua.run_ledger import task_sha256, task_site_execution_sha256
from egocua.schema import goal_for_condition, task_subtype
from egocua.construction import load_construction_tasks
from egocua.final_state import goal_entity_ids
from egocua.synthesis_reports import REPORTS
from egocua.timeline import COMPOSITE_TRACK, SCHEDULED_EVENT_TYPES, Condition, Timeline
from egocua.update_count import VARIANT_SUFFIX
from egocua.workspace import read_json_strict, write_json_atomic


_QUERY_STOPWORDS = frozenset(
    "a an and are as at be by for from how i in into is it its of on one or "
    "that the this to up with you your".split()
)


def _searchable_text(site: str, entity: dict[str, Any]) -> str:
    keys = {
        "egomart": ("name", "brand"),
        "egomedia": ("title", "channel_title", "kind"),
        "egomaps": ("title", "name", "kind", "address"),
    }[site]
    parts = [
        str(entity[key])
        for key in keys
        if isinstance(entity.get(key), (str, int, float))
    ]
    if site == "egomart" and isinstance(entity.get("attributes"), dict):
        parts.extend(
            str(value)
            for key, value in entity["attributes"].items()
            if key not in {"listing_source", "listing_type"}
        )
    return " ".join(parts).casefold()


def _requested_scope(
    tasks: list[dict[str, Any]],
    *,
    requested_ids: set[str],
    requested_sites: set[str],
) -> set[str]:
    """Return the exact incremental scope, or every task for a full smoke run."""

    if not requested_ids and not requested_sites:
        return {str(task["id"]) for task in tasks}
    return requested_ids | {
        str(task["id"]) for task in tasks if str(task["site"]) in requested_sites
    }


def _query_matches(site: str, entity: dict[str, Any], query: str) -> bool:
    haystack = _searchable_text(site, entity)
    terms = query.casefold().split()
    for term in terms:
        singular = term[:-1] if site == "egomart" and term.endswith("s") else term
        if term not in haystack and not (len(singular) > 2 and singular in haystack):
            return False
    return bool(terms)


def _broad_queries(
    site: str,
    entity: dict[str, Any],
    entities: list[dict[str, Any]],
    acceptable_ids: set[str],
) -> list[tuple[str, list[str]]]:
    """Rank public tokens best first; the live site decides which one reaches the entity."""
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", _searchable_text(site, entity))
        if len(token) > 2 and token not in _QUERY_STOPWORDS and not token.isdigit()
    }
    candidates: list[tuple[tuple[int, int, int, int], str, list[str]]] = []
    target_id = str(entity["entity_id"])
    for token in tokens:
        matches = [
            str(candidate["entity_id"])
            for candidate in entities
            if _query_matches(site, candidate, token)
        ]
        if target_id not in matches:
            continue
        neighbors = sum(entity_id not in acceptable_ids for entity_id in matches)
        score = (
            int(neighbors > 0),
            -abs(len(matches) - 6),
            min(neighbors, 5),
            len(token),
        )
        candidates.append((score, token, matches))
    if not candidates:
        raise ValueError(f"{target_id}: no public search token reaches the entity")
    return [(query, matches) for _, query, matches in sorted(candidates, reverse=True)]


def _detail_path(site: str, entity_id: str) -> str:
    prefix = {
        "egomart": "product",
        "egomedia": "video",
        "egomaps": "place",
    }[site]
    return f"/{prefix}/{urllib.parse.quote(entity_id)}"


class _SearchNavigation(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.next_href: str | None = None
        self._href: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "a":
            self._href = dict(attrs).get("href")

    def handle_data(self, data: str) -> None:
        if data.strip() in {"Next", "Next ›"}:
            self.next_href = self._href

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self._href = None


def _search_pages(base_url: str, query: str, target_href: str) -> list[str]:
    path: str | None = f"/search?q={urllib.parse.quote(query)}&page=1"
    pages = []
    while path is not None:
        page = _request(base_url, path)
        pages.append(page)
        if target_href in page:
            break
        navigation = _SearchNavigation()
        navigation.feed(page)
        path = navigation.next_href
    return pages


def _request(base_url: str, path: str, form: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(base_url + path)
    if form is not None:
        request = urllib.request.Request(
            base_url + path,
            data=urllib.parse.urlencode(form).encode(),
            method="POST",
        )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def _entity_title(entity: dict[str, Any]) -> str:
    return str(entity.get("title") or entity.get("name") or entity["entity_id"])


def _run_scripted(
    task: dict[str, Any],
    condition: Condition,
    *,
    root: Path,
    plan: ExperimentPlan,
    output_root: str | Path | None,
    duplicate_first: bool = False,
    expected_state: dict[str, Any] | None = None,
    route_checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parameters = plan.run_parameters(
        task,
        {
            "condition": condition.name,
            "replicate_id": plan.replicate_ids[0],
            "provider_seed": None,
            "schedule_index": 0,
        },
    )
    parameters.update(
        model="deterministic-smoke",
        model_version="deterministic-smoke",
        experiment_id="constructed-harness-smoke",
    )
    config = RunConfig(
        agent="deterministic-smoke",
        transport="direct-http",
        asset_root=root,
        output_root=output_root,
        **parameters,
    )

    def execute(context: RunContext) -> list[dict[str, Any]]:
        expected = expected_state or acceptable_states(task, condition)[0]
        entity_by_id = {
            entity["entity_id"]: entity for entity in context.site_data["entities"]
        }
        selected = list(expected["selected_entity_ids"])
        acceptable_ids = goal_entity_ids(goal_for_condition(task, condition))
        asked = False
        composite = task["track"] == COMPOSITE_TRACK

        def advance() -> None:
            events = context.advance()
            if composite:
                for event in events:
                    if event.get("type") == "visual_context_update" and event.get("update"):
                        if not context.observation_images(event["payload"]):
                            raise ValueError("composite update produced no visual input")

        def select(entity_id: str) -> None:
            nonlocal asked
            entity = entity_by_id[entity_id]
            advance()
            site = str(task["site"])
            target_href = _detail_path(site, entity_id)
            for query, matched_ids in _broad_queries(
                site, entity, context.site_data["entities"], acceptable_ids
            ):
                search_pages = _search_pages(context.base_url, query, target_href)
                target_visible = target_href in "\n".join(search_pages)
                neighbor_count = sum(
                    matched_id not in acceptable_ids for matched_id in matched_ids
                )
                if target_visible:
                    break
            advance()
            detail_html = _request(context.base_url, target_href)
            detail_opened = entity_id in detail_html

            if (
                not asked
                and task["track"] == "clarify"
                and condition.schedule == "online"
                and condition.scope == "full"
                and condition.ask in {"open", "allowed"}
            ):
                clarification = task["clarification"]
                context.ask_user(f"Which {clarification['valid_slot']} did you mean?")
                asked = True

            advance()
            if task["site"] == "egomart":
                selected_html = _request(
                    context.base_url,
                    "/cart/add",
                    {"item_id": entity_id},
                )
            else:
                selected_html = _request(
                    context.base_url,
                    "/select",
                    {"entity_id": entity_id},
                )
            if route_checks is not None:
                route_checks.append(
                    {
                        "entity_id": entity_id,
                        "query": query,
                        "search_pages_checked": len(search_pages),
                        "search_target_visible": target_visible,
                        "search_result_count": len(matched_ids),
                        "search_neighbor_count": neighbor_count,
                        "detail_opened": detail_opened,
                        "selection_persisted": entity_id in selected_html,
                    }
                )

        timeline = Timeline.compile(
            task,
            condition,
            default_release_step=plan.default_release_step,
            reference_steps=plan.reference_steps,
        )
        if composite:
            for observation in context.initial_payload["observations"]:
                if not context.observation_images(observation):
                    raise ValueError("composite opening produced no visual input")
            if "clarification" in task:
                advance()
                context.ask_user("Which one did you mean?")
        initial_ids: list[str] = []
        if timeline.releases and plan.release_trigger != "timeline_step":
            if task["track"] == "graft":
                initial_ids = task["graft"]["initial_goal"]["acceptable_states"][0][
                    "selected_entity_ids"
                ]
            elif task["track"] == "refresh":
                initial_ids = [task["refresh"]["initial_entity_id"]]
        for entity_id in initial_ids:
            select(entity_id)
        for entity_id in initial_ids:
            if entity_id in selected:
                continue
            advance()
            path, field = (
                ("/cart/remove", "item_id")
                if task["site"] == "egomart"
                else ("/deselect", "entity_id")
            )
            _request(context.base_url, path, {field: entity_id})
        for entity_id in selected:
            if entity_id not in initial_ids:
                select(entity_id)
        if duplicate_first and selected:
            select(selected[0])
        for name, value in expected["fields"].items():
            advance()
            _request(context.base_url, "/field", {"name": name, "value": value})
        advance()
        selection_path = "/cart" if task["site"] == "egomart" else "/selection"
        confirmation = _request(context.base_url, selection_path)
        if route_checks is not None:
            route_checks.append(
                {
                    "confirmation_visible": all(
                        entity_id in confirmation for entity_id in selected
                    ),
                }
            )
        if composite:
            last_update = max(
                (release.step for release in timeline.releases if release.update), default=0
            )
            while context.current_step < last_update:
                advance()
        elif timeline.releases:
            advance()
        return []

    return run_experiment(task, condition, config, execute)


def _release_protocol_checks(
    task: dict[str, Any], record: dict[str, Any], plan: ExperimentPlan
) -> dict[str, bool]:
    condition = Condition.from_dict(record["condition"])
    timeline = Timeline.compile(
        task,
        condition,
        default_release_step=plan.default_release_step,
        reference_steps=plan.reference_steps,
    )
    if not timeline.releases:
        return {}
    actions = record["actions"]
    delivered = [a for a in actions if a.get("type") in SCHEDULED_EVENT_TYPES]
    checks = {
        "registered_release_trigger": record["meta"]["runtime"]["release_trigger"]
        == plan.release_trigger,
        "registered_events_delivered_once": len(delivered) == len(timeline.releases),
        "release_record_valid": check_run(task, record)["run_valid"] is True,
        "observed_step_after_release": bool(delivered)
        and record["steps"] > max(a["step"] for a in delivered),
    }
    if plan.release_trigger != "timeline_step":
        trigger = first_release_trigger(actions, plan.release_trigger)
        checks["native_trigger_reached"] = trigger is not None
        checks["release_at_next_step_after_native_trigger"] = (
            trigger is not None
            and bool(delivered)
            and all(a["step"] == trigger[1]["step"] + 1 for a in delivered)
        )
    return checks


def _mutated_result(
    task: dict[str, Any],
    record: dict[str, Any],
    mutation: str,
    *,
    selected: list[str] | None = None,
    fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    mutated = copy.deepcopy(record)
    if selected is not None:
        mutated["live_state"]["selected_entity_ids"] = selected
    if fields is not None:
        mutated["live_state"]["fields"] = fields
    checked = check_run(task, mutated)
    return {
        "task_id": task["id"],
        "condition": record["condition"],
        "mutation": mutation,
        "evidence_status": checked["evidence_status"],
        "run_valid": checked["run_valid"],
        "e2e_success": checked["e2e_success"],
        "invalid_reasons": checked["run_invalid_reasons"],
    }


def _negative_checks(
    task: dict[str, Any], record: dict[str, Any], entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    raw_meta = record.get("meta")
    meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
    condition = Condition.from_dict(record.get("condition"))
    states = acceptable_states(task, condition)
    selected = list(states[0]["selected_entity_ids"])
    acceptable_ids = goal_entity_ids(goal_for_condition(task, condition))
    decoy = next(
        entity["entity_id"]
        for entity in entities
        if entity["entity_id"] not in acceptable_ids
    )
    results = [
        _mutated_result(task, record, "decoy_only", selected=[decoy]),
        _mutated_result(task, record, "target_plus_extra", selected=[*selected, decoy]),
        _mutated_result(
            task,
            record,
            "extra_field",
            fields={**states[0]["fields"], "unexpected_field": "unexpected"},
        ),
    ]
    if selected:
        results.append(
            _mutated_result(
                task,
                record,
                "duplicate_target",
                selected=[*selected, selected[0]],
            )
        )
    if states[0]["fields"]:
        field_name = next(iter(states[0]["fields"]))
        missing = dict(states[0]["fields"])
        missing.pop(field_name)
        wrong = {**states[0]["fields"], field_name: "wrong"}
        results.extend(
            [
                _mutated_result(task, record, "missing_field", fields=missing),
                _mutated_result(task, record, "wrong_field", fields=wrong),
            ]
        )

    missing_state = copy.deepcopy(record)
    missing_state.pop("live_state", None)
    checked_missing = check_run(task, missing_state)
    results.append(
        {
            "task_id": task["id"],
            "condition": record["condition"],
            "mutation": "missing_live_state",
            "evidence_status": checked_missing["evidence_status"],
            "run_valid": checked_missing["run_valid"],
            "e2e_success": checked_missing["e2e_success"],
            "invalid_reasons": checked_missing["run_invalid_reasons"],
        }
    )

    timeline = Timeline.compile(
        task,
        condition,
        default_release_step=meta["default_release_step"],
        reference_steps=meta["reference_steps"],
    )
    required_events = {release.as_event()["type"] for release in timeline.releases}
    for required_event in required_events:
        no_event = copy.deepcopy(record)
        no_event["actions"] = [
            action
            for action in no_event["actions"]
            if action.get("type") != required_event
        ]
        checked = check_run(task, no_event)
        results.append(
            {
                "task_id": task["id"],
                "condition": record["condition"],
                "mutation": "missing_required_event",
                "evidence_status": checked["evidence_status"],
                "run_valid": checked["run_valid"],
                "e2e_success": checked["e2e_success"],
                "invalid_reasons": checked["run_invalid_reasons"],
            }
        )
    return results


def _clarification_protocol_checks(
    task: dict[str, Any],
    allowed: Condition,
    *,
    default_release_step: int,
    reference_steps: int | None,
) -> dict[str, bool]:
    session = RunSession(
        task,
        allowed,
        event_step=default_release_step,
        reference_steps=reference_steps,
    )
    first = session.ask_user("Can you clarify?", step=1)
    second = session.ask_user("Which one?", step=2)
    forced_rejected = False
    try:
        RunSession(
            task,
            replace(allowed, name="smoke-ask-denied", ask="denied"),
            event_step=default_release_step,
            reference_steps=reference_steps,
        ).ask_user("Which one?", step=1)
    except ProtocolError:
        forced_rejected = True
    return {
        "first_question_gets_the_answer": first["valid"] is True
        and session.questions_used == 1,
        "second_question_returns_limit_response": second.get("limit_reached") is True,
        "no_ask_rejects_ask_user": forced_rejected,
    }


def _step_stats(runs: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[int]] = {}
    for run in runs:
        grouped.setdefault(f"{run['track']}/{run['condition']['name']}", []).append(
            run["steps"]
        )
    return {
        key: {
            "runs": len(values),
            "min": min(values),
            "median": median(values),
            "mean": round(mean(values), 2),
            "max": max(values),
        }
        for key, values in sorted(grouped.items())
    }


def _covering_states(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the fewest useful states needed to exercise every valid entity."""

    if not states:
        return []
    chosen = [states[0]]
    covered = set(states[0]["selected_entity_ids"])
    remaining = (
        set().union(*(set(state["selected_entity_ids"]) for state in states)) - covered
    )
    available = list(states[1:])
    while remaining:
        state = max(
            available,
            key=lambda candidate: len(
                set(candidate["selected_entity_ids"]) & remaining
            ),
        )
        gained = set(state["selected_entity_ids"]) & remaining
        if not gained:
            raise ValueError("acceptable states do not cover their declared entities")
        chosen.append(state)
        covered.update(gained)
        remaining -= gained
        available.remove(state)
    return chosen


def run_smoke(
    tasks_path: str | Path,
    *,
    root: str | Path,
    plan_path: str | Path,
    output_root: str | Path | None = None,
    task_ids: list[str] | None = None,
    sites: list[str] | None = None,
    previous_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    all_tasks = load_construction_tasks(workspace, task_ids, tasks_path=tasks_path)
    discovered_ids = {str(task["id"]) for task in all_tasks}
    known_sites = {str(task["site"]) for task in all_tasks}
    requested_ids = {str(task_id) for task_id in (task_ids or [])}
    requested_sites = {str(site) for site in (sites or [])}
    unknown_ids = sorted(requested_ids - discovered_ids)
    unknown_sites = sorted(requested_sites - known_sites)
    if unknown_ids:
        raise ValueError(f"unknown task IDs: {unknown_ids}")
    if unknown_sites:
        raise ValueError(f"unknown sites: {unknown_sites}")
    incremental = bool(requested_ids or requested_sites)
    smoke_ids = _requested_scope(
        all_tasks,
        requested_ids=requested_ids,
        requested_sites=requested_sites,
    )
    preserved_ids = (
        discovered_ids | set((previous_report or {}).get("tasks", {}))
    ) - smoke_ids
    previous_hashes = (
        previous_report.get("task_sha256", {})
        if isinstance(previous_report, dict)
        else {}
    )
    previous_run_ids = {
        str(run.get("task_id"))
        for run in (previous_report or {}).get("runs", [])
        if isinstance(run, dict)
    }
    if incremental:
        if not isinstance(previous_report, dict):
            raise ValueError("incremental smoke requires an existing complete report")
        previous_task_rows = previous_report.get("tasks", {})
        previous_hash_ids = (
            set(previous_hashes) if isinstance(previous_hashes, dict) else set()
        )
        previous_task_ids = (
            set(previous_task_rows) if isinstance(previous_task_rows, dict) else set()
        )
        if (
            preserved_ids - previous_run_ids
            or preserved_ids - previous_hash_ids
            or preserved_ids - previous_task_ids
        ):
            raise ValueError(
                "incremental smoke requires a previous report covering every task not being rerun"
            )
    tasks = [task for task in all_tasks if str(task["id"]) in smoke_ids]
    plan = load_experiment_plan(plan_path)
    site_cache = {}
    site_data_by_task = {}
    for task in tasks:
        site = str(task["site"])
        snapshot = plan.site_snapshot_for(task)
        key = (site, snapshot)
        if key not in site_cache:
            site_cache[key] = (
                read_json_strict(Path(snapshot)) if snapshot else load_site_data(workspace, site)
            )
        site_data_by_task[str(task["id"])] = site_cache[key]
    runs: list[dict[str, Any]] = [
        dict(run)
        for run in (previous_report or {}).get("runs", [])
        if incremental
        and isinstance(run, dict)
        and str(run.get("task_id")) in preserved_ids
    ]
    negatives: list[dict[str, Any]] = [
        dict(result)
        for result in (previous_report or {}).get("negative_checks", [])
        if incremental
        and isinstance(result, dict)
        and str(result.get("task_id")) in preserved_ids
    ]
    protocol_checks: dict[str, dict[str, bool]] = {
        str(task_id): dict(checks)
        for task_id, checks in (previous_report or {})
        .get("protocol_checks", {})
        .items()
        if incremental and str(task_id) in preserved_ids and isinstance(checks, dict)
    }

    for task in tasks:
        site_data = site_data_by_task[str(task["id"])]
        reference_condition = plan.reference_condition_for(task["track"])
        for condition_name in plan.conditions_for(task["track"]):
            condition = plan.condition(condition_name)
            all_states = acceptable_states(task, condition)
            states = (
                _covering_states(all_states)
                if condition == reference_condition
                else all_states[:1]
            )
            for state_index, expected in enumerate(states):
                route_checks: list[dict[str, Any]] | None = (
                    [] if condition == reference_condition else None
                )
                record = _run_scripted(
                    task,
                    condition,
                    root=workspace,
                    plan=plan,
                    output_root=output_root,
                    expected_state=expected,
                    route_checks=route_checks,
                )
                checked = check_run(task, record)
                protocol_checks.setdefault(task["id"], {}).update(
                    {
                        f"{condition.name}/{state_index}/{name}": passed
                        for name, passed in _release_protocol_checks(
                            task, record, plan
                        ).items()
                    }
                )
                runs.append(
                    {
                        "task_id": task["id"],
                        "track": task["track"],
                        "site": task["site"],
                        "subtype": task_subtype(task),
                        "condition": condition.as_dict(),
                        "acceptable_state_index": state_index,
                        "expected": expected,
                        "live_state": record["live_state"],
                        "steps": record["steps"],
                        "run_valid": checked["run_valid"],
                        "evidence_status": checked["evidence_status"],
                        "e2e_success": checked["e2e_success"],
                        "invalid_reasons": checked["run_invalid_reasons"],
                        "run_artifact": record.get("meta", {}).get(
                            "artifact_directory"
                        ),
                        "release_trigger": plan.release_trigger,
                        "route_checks": route_checks or [],
                        "route_checked": route_checks is not None,
                        **(
                            {
                                key: checked[key]
                                for key in (
                                    "asked",
                                    "valid_question",
                                    "answer_delivered",
                                    "timely_question",
                                    "answer_used",
                                    "over_ask",
                                    "question_limit_exceeded",
                                )
                            }
                            if task["track"] == "clarify"
                            else {}
                        ),
                    }
                )
                if state_index == 0:
                    negatives.extend(
                        _negative_checks(task, record, site_data["entities"])
                    )

        duplicate_record = _run_scripted(
            task,
            reference_condition,
            root=workspace,
            plan=plan,
            output_root=output_root,
            duplicate_first=True,
        )
        duplicate_checked = check_run(task, duplicate_record)
        protocol_checks.setdefault(task["id"], {}).update(
            {
                f"duplicate_selection/{name}": passed
                for name, passed in _release_protocol_checks(
                    task, duplicate_record, plan
                ).items()
            }
        )
        duplicate_result = {
            "task_id": task["id"],
            "condition": reference_condition.as_dict(),
            "mutation": "duplicate_quantity_through_site",
            "evidence_status": duplicate_checked["evidence_status"],
            "run_valid": duplicate_checked["run_valid"],
            "e2e_success": duplicate_checked["e2e_success"],
            "invalid_reasons": duplicate_checked["run_invalid_reasons"],
        }
        if (
            task["site"] == "egomart"
            and acceptable_states(task, reference_condition)[0]["selected_entity_ids"]
        ):
            negatives.append(duplicate_result)
        else:
            protocol_checks.setdefault(task["id"], {})[
                "repeated_save_is_idempotent"
            ] = duplicate_checked["e2e_success"] is True
        if task["track"] == "clarify":
            protocol_checks.setdefault(task["id"], {}).update(
                _clarification_protocol_checks(
                    task,
                    plan.reference_condition_for("clarify"),
                    default_release_step=plan.default_release_step,
                    reference_steps=plan.reference_steps,
                )
            )

    positive_failures = [
        run
        for run in runs
        if not run["run_valid"]
        or run["evidence_status"] != "PASS"
        or not run["e2e_success"]
    ]

    def route_check_failed(check: dict[str, Any]) -> bool:
        return (
            check.get("search_target_visible") is False
            or check.get("detail_opened") is False
            or check.get("selection_persisted") is False
            or check.get("confirmation_visible") is False
        )

    def route_preflight_failed(
        checks: list[dict[str, Any]],
        *,
        require_neighbor: bool,
    ) -> bool:
        search_checks = [check for check in checks if "search_neighbor_count" in check]
        return (
            any(route_check_failed(check) for check in checks)
            or not search_checks
            or (
                require_neighbor
                and not any(
                    int(check["search_neighbor_count"]) >= 1 for check in search_checks
                )
            )
        )

    preflight_failures = [
        {
            "task_id": run["task_id"],
            "condition": run["condition"],
            "acceptable_state_index": run["acceptable_state_index"],
            "route_checks": run["route_checks"],
        }
        for run in runs
        if run.get("route_checked")
        and route_preflight_failed(
            run["route_checks"],
            require_neighbor=run["condition"]["scope"] == "full",
        )
    ]

    def negative_failed(result: dict[str, Any]) -> bool:
        if result["mutation"] == "missing_live_state":
            return result["evidence_status"] != "UNKNOWN"
        if result["mutation"] == "missing_required_event":
            return result["run_valid"] is not False
        return result["evidence_status"] != "FAIL"

    negative_failures = [result for result in negatives if negative_failed(result)]
    protocol_failures = [
        f"{task_id}:{name}"
        for task_id, checks in protocol_checks.items()
        for name, passed in checks.items()
        if not passed
    ]
    previous_task_rows = (
        previous_report.get("tasks", {}) if isinstance(previous_report, dict) else {}
    )
    task_rows: dict[str, dict[str, Any]] = {
        task_id: dict(previous_task_rows[task_id])
        for task_id in preserved_ids
        if incremental
    }
    task_hashes = {
        task_id: previous_hashes[task_id] for task_id in preserved_ids if incremental
    }
    for task in all_tasks:
        task_id = str(task["id"])
        if incremental and task_id not in smoke_ids:
            previous_task = previous_task_rows.get(task_id)
            if not isinstance(previous_task, dict):
                raise ValueError(
                    f"incremental smoke has no previous task row for {task_id}"
                )
            task_rows[task_id] = dict(previous_task)
            continue
        task_runs = [run for run in runs if run["task_id"] == task_id]
        task_negatives = [check for check in negatives if check["task_id"] == task_id]
        task_protocol = protocol_checks.get(task_id, {})
        failures = [
            *("correct_path" for run in positive_failures if run["task_id"] == task_id),
            *(
                "route_preflight"
                for row in preflight_failures
                if row["task_id"] == task_id
            ),
            *(
                "negative_check"
                for row in negative_failures
                if row["task_id"] == task_id
            ),
            *(
                "protocol_check"
                for value in protocol_failures
                if value.startswith(f"{task_id}:")
            ),
        ]
        if not task_runs:
            failures.append("missing_correct_path")
        if not task_negatives:
            failures.append("missing_negative_check")
        task_rows[task_id] = {
            "task_sha256": task_sha256(task),
            "task_site_execution_sha256": task_site_execution_sha256(
                task, site_data_by_task[task_id]
            ),
            "status": "pass" if not failures else "fail",
            "failures": sorted(set(failures)),
            "correct_paths": len(task_runs),
            "negative_checks": len(task_negatives),
            "protocol_checks": len(task_protocol),
        }
        task_hashes[task_id] = task_sha256(task)
    return {
        "schema_version": REPORTS["smoke"].schema_version,
        "scope": (
            "deterministic search/detail/native-action/persistence/checker preflight; "
            "not human or model validation"
        ),
        "audit_scope": {
            "mode": "incremental" if incremental else "full",
            "requested_task_ids": sorted(requested_ids),
            "requested_sites": sorted(requested_sites),
            "smoked_task_ids": sorted(smoke_ids),
            "preserved_task_count": len(preserved_ids) if incremental else 0,
        },
        "task_sha256": task_hashes,
        "tasks": task_rows,
        "runs": runs,
        "negative_checks": negatives,
        "protocol_checks": protocol_checks,
        "summary": {
            "tasks": len(task_rows),
            "conditions": len(runs),
            "correct_paths": len(runs),
            "successful_correct_runs": len(runs) - len(positive_failures),
            "route_preflights": sum(len(run["route_checks"]) for run in runs),
            "route_preflights_passed": sum(
                len(run["route_checks"])
                for run in runs
                if not route_preflight_failed(
                    run["route_checks"],
                    require_neighbor=run["condition"]["scope"] == "full",
                )
            ),
            "negative_checks": len(negatives),
            "negative_checks_passed": len(negatives) - len(negative_failures),
            "protocol_checks": sum(len(checks) for checks in protocol_checks.values()),
            "protocol_checks_passed": sum(
                sum(checks.values()) for checks in protocol_checks.values()
            ),
            "scripted_harness_turns_by_condition": _step_stats(runs),
            "positive_failures": positive_failures,
            "preflight_failures": preflight_failures,
            "negative_failures": negative_failures,
            "protocol_failures": protocol_failures,
        },
    }


def run_composite_smoke(
    root: Path, task_ids: list[str] | None, *, output_root: str | None = None
) -> dict[str, dict[str, Any]]:
    """Exercise actual composite inputs and one native scoring path per task."""
    from egocua.controlled_experiments import load_composite_validation_study

    [arm], _, _ = load_composite_validation_study(root, task_ids, model="deterministic-smoke")
    rows = {}
    sites = {}
    development = DevelopmentEvidence(root)
    for task in arm.tasks:
        variant = task["id"].endswith(VARIANT_SUFFIX)
        plan = development.trial_plan(task) if variant else arm.plan
        condition = plan.condition("online")
        site_key = plan.site_snapshot_for(task) if variant else task["site"]
        if site_key not in sites:
            sites[site_key] = read_json_strict(Path(site_key)) if variant else load_site_data(root, task["site"])
        routes: list[dict[str, Any]] = []
        record = _run_scripted(
            task, condition, root=root, plan=plan, output_root=output_root, route_checks=routes
        )
        checked = check_run(task, record)
        entities = goal_entity_ids(task["goal"])
        decoy = next(
            entity for entity in task["truth"]["site_scope_entity_ids"]
            if entity not in entities
        )
        expected = record["live_state"]["selected_entity_ids"]
        negatives = [
            _mutated_result(task, record, "empty_selection", selected=[]),
            _mutated_result(task, record, "target_plus_extra", selected=[*expected, decoy]),
        ]
        coverage = checked["composite_delivery"]
        event_coverage = composite_event_coverage(checked)
        checks = {
            "native_path": bool(routes) and all(
                row.get("confirmation_visible") is True
                if "confirmation_visible" in row
                else all(row[key] is True for key in (
                    "search_target_visible", "detail_opened", "selection_persisted"
                ))
                for row in routes
            ),
            "exact_terminal_state": checked["e2e_success"] is True,
            "rejects_negative_states": all(
                row["e2e_success"] is False and row["run_valid"] is True
                for row in negatives
            ),
            **coverage,
            **event_coverage,
        }
        rows[task["id"]] = {
            "task_sha256": task_sha256(task),
            "task_site_execution_sha256": task_site_execution_sha256(task, sites[site_key]),
            "status": "pass" if all(checks.values()) else "fail",
            "failures": [name for name, passed in checks.items() if not passed],
            "protocol_checks": len(checks),
            "composite_delivery": coverage,
            **({"composite_event_coverage": event_coverage} if event_coverage else {}),
            "routes": routes,
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test all tasks and experiment conditions."
    )
    parser.add_argument("tasks", nargs="?", default="data/tasks")
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default=TASK_REVIEW_PLAN)
    parser.add_argument("--output-root")
    parser.add_argument("--output", default=REPORTS["smoke"].path.as_posix())
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--site", action="append", default=[])
    parser.add_argument("--composite", action="store_true", help="check reviewed experimental composites")
    args = parser.parse_args()
    previous_report = None
    output = Path(args.output)
    if args.composite:
        if args.site or args.plan != TASK_REVIEW_PLAN:
            parser.error("--composite uses the registered online plan and optional --task-id")
        root = Path(args.root).resolve()
        rows = run_composite_smoke(root, args.task_id or None, output_root=args.output_root)
        report = read_json_strict(output) if output.is_file() else {
            "schema_version": REPORTS["smoke"].schema_version, "tasks": {}
        }
        report.setdefault("composite_tasks", {}).update(rows)
        write_json_atomic(output, report)
        print(json.dumps({task_id: row["status"] for task_id, row in rows.items()}, indent=2))
        if any(row["status"] != "pass" for row in rows.values()):
            raise SystemExit(1)
        return
    if output.is_file():
        saved = read_json_strict(output)
        if args.task_id or args.site:
            previous_report = saved
    else:
        saved = {}
    report = run_smoke(
        args.tasks,
        root=args.root,
        plan_path=args.plan,
        output_root=args.output_root,
        task_ids=args.task_id,
        sites=args.site,
        previous_report=previous_report,
    )
    if "composite_tasks" in saved:
        report["composite_tasks"] = saved["composite_tasks"]
    write_json_atomic(output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if any(
        report["summary"][key]
        for key in (
            "positive_failures",
            "preflight_failures",
            "negative_failures",
            "protocol_failures",
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
