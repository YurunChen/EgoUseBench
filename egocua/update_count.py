"""Same-task request-count variants, with inherited scene and answer materials."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .experiment import ExperimentPlan, parse_experiment_plan
from .timeline import INSTRUCTION, CompositeUpdate, Timeline, composite_input
from .workspace import read_json_strict, resolve_within


UPDATE_COUNT = "update-count"
VARIANTS = Path("data/experiments/variants/update-count")
REGISTRATION = Path("data/experiments/update-count.json")
VARIANT_SUFFIX = "-update-count"


def variant_id(pair_id: str) -> str:
    return f"{pair_id}{VARIANT_SUFFIX}"


def load_cases(root: Path, pair_ids: list[str] | None = None) -> list[dict[str, Any]]:
    paths = sorted((root / VARIANTS).glob("*/variant.json"))
    cases = [read_json_strict(path) for path in paths]
    if pair_ids is not None:
        requested = set(pair_ids)
        if not pair_ids or len(requested) != len(pair_ids):
            raise ValueError("update-count requires distinct, nonempty pair IDs")
        missing = requested - {case["pair_id"] for case in cases}
        if missing:
            raise ValueError(f"unregistered update-count pairs: {sorted(missing)}")
        cases = [case for case in cases if case["pair_id"] in requested]
    for case in cases:
        expected = f"data/experiments/composite/{case['pair_id']}.json"
        if case["source_task_path"] != expected or case["direction"] not in {"merge", "insert"}:
            raise ValueError(f"{case['pair_id']}: invalid update-count source or direction")
        if case["id"] != variant_id(case["pair_id"]):
            raise ValueError(f"{case['pair_id']}: incorrect variant identity")
    return cases


def compile_variant(original: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Change only requests and identity; inherit all scene and scoring fields."""
    if original["id"] != case["pair_id"]:
        raise ValueError("update-count source identity differs")
    task = deepcopy(original)
    task["id"] = case["id"]
    task["instructions"] = {"base": original["instructions"]["base"]}
    for key in ("refresh_onset_seconds", "update_onset_seconds"):
        task["visual_session"].pop(key, None)
    task["composite_events"] = deepcopy(case["composite_events"])
    composite_input(task)
    return task


def source_study(root: Path, model: str) -> Path:
    registered = read_json_strict(root / REGISTRATION)
    try:
        path = registered["source_studies"][model]
    except KeyError as error:
        raise ValueError(f"no registered original composite study for {model}") from error
    directory = resolve_within(root, path, context="update-count original study", must_exist=False)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return directory


def frozen_original(root: Path, pair_id: str) -> tuple[dict[str, Any], ExperimentPlan]:
    registered = read_json_strict(root / REGISTRATION)
    directory = source_study(root, registered["reference_model"])
    study = read_json_strict(directory / "plan.json")
    matches = []
    for arm in study["arms"]:
        if arm["variable"] != "composite-formal":
            continue
        path = directory / "tasks" / arm["name"] / "composite" / f"{pair_id}.json"
        if path.is_file():
            matches.append((read_json_strict(path), arm["plan"]))
    if len(matches) != 1:
        raise ValueError(f"{pair_id}: expected one frozen formal original")
    task, payload = matches[0]
    return task, parse_experiment_plan(payload, str(directory / "plan.json"))


def _update_steps(task: dict[str, Any], plan: ExperimentPlan) -> dict[str, int]:
    timeline = Timeline.compile(
        task, plan.condition("online"),
        default_release_step=plan.default_release_step, reference_steps=plan.reference_steps,
    )
    return {
        release.composite_event_id or ("G" if release.kind == INSTRUCTION else "R"): release.step
        for release in timeline.releases if release.update
    }


def _event_gaps(
    updates: Sequence[CompositeUpdate], steps: dict[str, int],
) -> list[dict[str, Any]]:
    return [
        {
            "from_event": earlier.event_id, "to_event": later.event_id,
            "source_seconds": round(later.onset_seconds - earlier.onset_seconds, 6),
            "replay_steps": steps[later.event_id] - steps[earlier.event_id],
        }
        for earlier, later in zip(updates, updates[1:], strict=False)
    ]


def audit_case(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Check the real frozen replay grid and registered exclusion witnesses."""
    original, plan = frozen_original(root, case["pair_id"])
    current = read_json_strict(root / case["source_task_path"])
    variant = compile_variant(original, case)
    errors = []
    for key in ("source", "instructions", "clarification", "goal", "truth", "composite_events"):
        if key == "instructions":
            agrees = current[key] == original[key]
        else:
            agrees = current.get(key) == original.get(key)
        if not agrees:
            errors.append(f"current original differs from frozen original: {key}")
    before = composite_input(original)
    if composite_input(current) != before:
        errors.append("current original event contract differs from its frozen version")
    after = composite_input(variant)
    old_r = [event for event in before.updates if event.mechanism == "R"]
    new_r = [event for event in after.updates if event.mechanism == "R"]
    if [(e.onset_seconds, e.request) for e in old_r] != [(e.onset_seconds, e.request) for e in new_r]:
        errors.append("Refresh content or timing changed")
    old_g = [event for event in before.updates if event.mechanism == "G"]
    new_g = [event for event in after.updates if event.mechanism == "G"]
    expected = len(old_g) + (1 if case["direction"] == "insert" else -1)
    if len(new_g) != expected or before.mechanisms != after.mechanisms:
        errors.append("the variant must differ by exactly one Graft event")
    if not new_g or max(e.onset_seconds for e in before.updates) != max(e.onset_seconds for e in after.updates):
        errors.append("final information availability time changed")
    if case["direction"] == "merge":
        if len(old_g) != 2 or len(new_g) != 1 or new_g[0].onset_seconds != old_g[1].onset_seconds:
            errors.append("merge must arrive at the second original request")
    else:
        old_requests = [(e.onset_seconds, e.request) for e in old_g]
        new_requests = [(e.onset_seconds, e.request) for e in new_g]
        if len(old_g) != 1 or any(e not in new_requests for e in old_requests):
            errors.append("insert changed the original request")
        added = [e for e in new_g if (e.onset_seconds, e.request) not in old_requests]
        if len(added) != 1 or added[0].onset_seconds >= old_g[0].onset_seconds:
            errors.append("intermediate request must precede the original request")
    steps = {}
    adjacent_gaps = None
    request_gap = None
    try:
        steps = _update_steps(variant, plan)
        original_steps = _update_steps(original, plan)
        adjacent_gaps = {
            "original": _event_gaps(before.updates, original_steps),
            "variant": _event_gaps(after.updates, steps),
        }
        requests, request_steps = (
            (old_g, original_steps) if case["direction"] == "merge" else (new_g, steps)
        )
        request_gaps = _event_gaps(requests, request_steps)
        if len(request_gaps) == 1:
            request_gap = request_gaps[0]
        if any(step > 30 for step in steps.values()):
            errors.append("update falls outside the 30-step budget")
    except ValueError as error:
        errors.append(str(error))
    if plan.max_steps != 30 or plan.condition("online").batch_seconds != 8:
        errors.append("original protocol is not 30 steps / 8 seconds")
    if variant["goal"] != original["goal"]:
        errors.append("final acceptable states changed")
    if case["direction"] == "insert":
        errors.extend(_witness_errors(case, original, plan))
    return {
        "pair_id": case["pair_id"], "variant_id": case["id"],
        "direction": case["direction"], "subtype": case["subtype"], "included": case["included"],
        "exclusion_reason": case.get("exclusion_reason"),
        "original_updates": len(before.updates), "variant_updates": len(after.updates),
        "event_steps": steps,
        "adjacent_update_gaps": adjacent_gaps, "more_request_gap": request_gap,
        "errors": errors,
        "checks_passed": not errors,
    }


def _witness_errors(case: dict[str, Any], original: dict[str, Any], plan: ExperimentPlan) -> list[str]:
    witness = case.get("exclusion_witness")
    if not witness:
        return ["missing intermediate-request exclusion witness"]
    site = read_json_strict(Path(plan.site_snapshot_for(original)))
    entities = {item["entity_id"]: item for item in site["entities"]}
    scope = set(original["truth"]["site_scope_entity_ids"])
    errors = []
    for state_name in ("excluded_state", "retained_state"):
        state = witness[state_name]
        if not state["selected_entity_ids"] or not set(state["selected_entity_ids"]) <= scope:
            errors.append(f"{state_name}: witness outside the original website scope")
    for fact in witness.get("public_facts", []):
        item = entities[fact["entity_id"]]
        value = item
        for field in fact["path"]:
            value = value[field]
        if value != fact["equals"]:
            errors.append(f"public fact differs: {fact['entity_id']} {fact['path']}")
    constraint = witness["constraint"]
    values = []
    for name in ("excluded_state", "retained_state"):
        state = witness[name]
        if constraint["kind"] == "quantity":
            value = sum(constraint["units"].get(entity, 0) for entity in state["selected_entity_ids"])
        elif constraint["kind"] == "presence":
            value = sum(entity in constraint["entity_ids"] for entity in state["selected_entity_ids"])
        else:
            item = entities[constraint["entity_id_by_state"][name]]
            value = item[constraint["field"]]
            if constraint["field"] == "price":
                value = float(str(value).replace("$", "").replace(",", ""))
        limit = constraint["value"]
        operation = constraint["operator"]
        values.append({"<": value < limit, "<=": value <= limit, "==": value == limit, ">=": value >= limit}[operation])
    if values != [False, True]:
        errors.append("intermediate constraint must exclude the old state and retain another")
    if not witness.get("prior_legality_evidence"):
        errors.append("missing reviewed legality at intermediate delivery time")
    if not witness.get("original_request_effect"):
        errors.append("missing evidence that the original request remains meaningful")
    return errors
