from __future__ import annotations

import argparse
import copy
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from typing import Any

from egocua.checker import acceptable_states, check_run
from egocua.environment import load_site_data
from egocua.execution import RunConfig, RunContext, run_experiment
from egocua.experiment import load_experiment_plan
from egocua.protocol import ProtocolError, RunSession
from egocua.schema import load_tasks, task_subtype
from egocua.workspace import write_json_atomic


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
    task: dict[str, Any], arm: str, *, root: Path, max_steps: int,
    artifacts_root: str | Path | None, duplicate_first: bool = False,
) -> dict[str, Any]:
    config = RunConfig(
        agent="deterministic-smoke",
        model="deterministic-smoke",
        model_version="deterministic-smoke",
        transport="direct-http",
        max_steps=max_steps,
        asset_root=root,
        artifacts_root=artifacts_root,
        experiment_id="constructed-harness-smoke",
    )

    def execute(context: RunContext) -> list[dict[str, Any]]:
        expected = acceptable_states(task, arm)[0]
        entity_by_id = {
            entity["entity_id"]: entity for entity in context.site_data["entities"]
        }
        first_id = expected["selected_entity_ids"][0]
        context.advance()
        query = _entity_title(entity_by_id[first_id]).split()[0]
        _request(context.base_url, "/search?q=" + urllib.parse.quote(query))
        context.advance()

        if task["track"] == "clarify" and arm == "interactive":
            clarification = task["clarification"]
            context.ask_user(
                clarification["valid_slot"],
                f"Which {clarification['valid_slot']} did you mean?",
            )

        selected = list(expected["selected_entity_ids"])
        if duplicate_first:
            selected.insert(1, selected[0])
        for entity_id in selected:
            context.advance()
            if task["site"] == "egomart":
                _request(context.base_url, "/cart/add", {"item_id": entity_id})
            else:
                _request(context.base_url, "/select", {"entity_id": entity_id})
        for name, value in expected["fields"].items():
            context.advance()
            _request(context.base_url, "/field", {"name": name, "value": value})
        context.advance()
        _request(context.base_url, "/submit", {})
        return []

    return run_experiment(task, arm, config, execute)


def _mutated_result(
    task: dict[str, Any], record: dict[str, Any], mutation: str, *,
    selected: list[str] | None = None, fields: dict[str, str] | None = None,
) -> dict[str, Any]:
    mutated = copy.deepcopy(record)
    if selected is not None:
        mutated["final_state"]["selected_entity_ids"] = selected
    if fields is not None:
        mutated["final_state"]["fields"] = fields
    checked = check_run(task, mutated)
    return {
        "task_id": task["id"],
        "arm": record["arm"],
        "mutation": mutation,
        "evidence_status": checked["evidence_status"],
        "run_valid": checked["run_valid"],
        "e2e_success": checked["e2e_success"],
        "invalid_reasons": checked["run_invalid_reasons"],
    }


def _negative_checks(
    task: dict[str, Any], record: dict[str, Any], entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    states = acceptable_states(task, record["arm"])
    selected = list(states[0]["selected_entity_ids"])
    acceptable_ids = {
        entity_id for state in states for entity_id in state["selected_entity_ids"]
    }
    decoy = next(
        entity["entity_id"] for entity in entities
        if entity["entity_id"] not in acceptable_ids
    )
    results = [
        _mutated_result(task, record, "decoy_only", selected=[decoy]),
        _mutated_result(task, record, "target_plus_extra", selected=[*selected, decoy]),
        _mutated_result(task, record, "duplicate_target", selected=[*selected, selected[0]]),
        _mutated_result(
            task,
            record,
            "extra_field",
            fields={**states[0]["fields"], "unexpected_field": "unexpected"},
        ),
    ]
    if states[0]["fields"]:
        field_name = next(iter(states[0]["fields"]))
        missing = dict(states[0]["fields"])
        missing.pop(field_name)
        wrong = {**states[0]["fields"], field_name: "wrong"}
        results.extend([
            _mutated_result(task, record, "missing_field", fields=missing),
            _mutated_result(task, record, "wrong_field", fields=wrong),
        ])

    missing_state = copy.deepcopy(record)
    missing_state.pop("final_state", None)
    checked_missing = check_run(task, missing_state)
    results.append({
        "task_id": task["id"],
        "arm": record["arm"],
        "mutation": "missing_final_state",
        "evidence_status": checked_missing["evidence_status"],
        "run_valid": checked_missing["run_valid"],
        "e2e_success": checked_missing["e2e_success"],
        "invalid_reasons": checked_missing["run_invalid_reasons"],
    })

    required_event = {
        ("graft", "injected"): "instruction_injection",
        ("refresh", "stream"): "visual_context_update",
    }.get((task["track"], record["arm"]))
    if required_event:
        no_event = copy.deepcopy(record)
        no_event["actions"] = [
            action for action in no_event["actions"]
            if action.get("type") != required_event
        ]
        checked = check_run(task, no_event)
        results.append({
            "task_id": task["id"], "arm": record["arm"],
            "mutation": "missing_required_event",
            "evidence_status": checked["evidence_status"],
            "run_valid": checked["run_valid"],
            "e2e_success": checked["e2e_success"],
            "invalid_reasons": checked["run_invalid_reasons"],
        })
    return results


def _clarification_protocol_checks(task: dict[str, Any]) -> dict[str, bool]:
    session = RunSession(task, "interactive")
    invalid_slot = next(
        slot for slot in ("position", "appearance", "brand", "variant", "sequence")
        if slot != task["clarification"]["valid_slot"]
    )
    first = session.ask_user(invalid_slot, "Can you clarify?", step=1)
    second = session.ask_user(task["clarification"]["valid_slot"], "Which one?", step=2)
    forced_rejected = False
    try:
        RunSession(task, "forced_execute").ask_user(
            task["clarification"]["valid_slot"], "Which one?", step=1
        )
    except ProtocolError:
        forced_rejected = True
    return {
        "invalid_slot_consumes_question": first["valid"] is False and session.questions_used == 1,
        "second_question_returns_limit_response": second.get("limit_reached") is True,
        "forced_execute_rejects_ask_user": forced_rejected,
    }


def _step_stats(runs: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[int]] = {}
    for run in runs:
        grouped.setdefault(f"{run['track']}/{run['arm']}", []).append(run["steps"])
    return {
        key: {
            "runs": len(values), "min": min(values), "median": median(values),
            "mean": round(mean(values), 2), "max": max(values),
        }
        for key, values in sorted(grouped.items())
    }


def run_smoke(
    tasks_path: str | Path, *, root: str | Path, plan_path: str | Path,
    artifacts_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    tasks = load_tasks(tasks_path)
    plan = load_experiment_plan(plan_path)
    runs: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    protocol_checks: dict[str, dict[str, bool]] = {}

    for task in tasks:
        site_data = load_site_data(workspace, task["site"])
        for arm in plan.arms[task["track"]]:
            record = _run_scripted(
                task, arm, root=workspace, max_steps=plan.max_steps,
                artifacts_root=artifacts_root,
            )
            checked = check_run(task, record)
            runs.append({
                "task_id": task["id"], "track": task["track"], "site": task["site"],
                "subtype": task_subtype(task), "arm": arm,
                "expected": acceptable_states(task, arm)[0],
                "final_state": record["final_state"], "steps": record["steps"],
                "run_valid": checked["run_valid"],
                "evidence_status": checked["evidence_status"],
                "e2e_success": checked["e2e_success"],
                "invalid_reasons": checked["run_invalid_reasons"],
                **({
                    key: checked[key] for key in (
                        "asked", "valid_question", "timely_question", "answer_used",
                        "unnecessary_ask", "question_limit_exceeded",
                    )
                } if task["track"] == "clarify" else {}),
            })
            negatives.extend(_negative_checks(task, record, site_data["entities"]))

        reference_arm = plan.arms[task["track"]][0]
        duplicate_record = _run_scripted(
            task, reference_arm, root=workspace, max_steps=plan.max_steps,
            artifacts_root=artifacts_root, duplicate_first=True,
        )
        duplicate_checked = check_run(task, duplicate_record)
        negatives.append({
            "task_id": task["id"], "arm": reference_arm,
            "mutation": "duplicate_quantity_through_site",
            "evidence_status": duplicate_checked["evidence_status"],
            "run_valid": duplicate_checked["run_valid"],
            "e2e_success": duplicate_checked["e2e_success"],
            "invalid_reasons": duplicate_checked["run_invalid_reasons"],
        })
        if task["track"] == "clarify":
            protocol_checks[task["id"]] = _clarification_protocol_checks(task)

    positive_failures = [
        run for run in runs
        if not run["run_valid"] or run["evidence_status"] != "PASS" or not run["e2e_success"]
    ]
    def negative_failed(result: dict[str, Any]) -> bool:
        if result["mutation"] == "missing_final_state":
            return result["evidence_status"] != "UNKNOWN"
        if result["mutation"] == "missing_required_event":
            return result["run_valid"] is not False
        return result["evidence_status"] != "FAIL"

    negative_failures = [result for result in negatives if negative_failed(result)]
    protocol_failures = [
        f"{task_id}:{name}"
        for task_id, checks in protocol_checks.items()
        for name, passed in checks.items() if not passed
    ]
    return {
        "schema_version": 3,
        "scope": "deterministic protocol/site/checker smoke; not human or model validation",
        "runs": runs,
        "negative_checks": negatives,
        "protocol_checks": protocol_checks,
        "summary": {
            "tasks": len(tasks), "arms": len(runs),
            "successful_correct_runs": len(runs) - len(positive_failures),
            "negative_checks": len(negatives),
            "negative_checks_passed": len(negatives) - len(negative_failures),
            "protocol_checks": sum(len(checks) for checks in protocol_checks.values()),
            "protocol_checks_passed": sum(sum(checks.values()) for checks in protocol_checks.values()),
            "scripted_harness_turns_by_condition": _step_stats(runs),
            "positive_failures": positive_failures,
            "negative_failures": negative_failures,
            "protocol_failures": protocol_failures,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test all tasks and experiment arms.")
    parser.add_argument("tasks")
    parser.add_argument("--root", default=".")
    parser.add_argument("--plan", default="configs/pilot.json")
    parser.add_argument("--artifacts-root")
    parser.add_argument("--output", default="artifacts/reports/task-smoke.json")
    args = parser.parse_args()
    report = run_smoke(
        args.tasks, root=args.root, plan_path=args.plan,
        artifacts_root=args.artifacts_root,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if any(report["summary"][key] for key in (
        "positive_failures", "negative_failures", "protocol_failures"
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
