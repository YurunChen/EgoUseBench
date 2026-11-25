"""Ask a model for short, natural before/after website search queries.

The model only proposes queries.  It cannot admit a task, assign confidence, choose
the answer, or alter catalog ordering.  Coding-Agent review and deterministic
neighborhood checks remain mandatory after the proposal is applied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import requests
from dotenv import load_dotenv

from egocua.environment import load_site_data
from egocua.run_ledger import canonical_sha256
from egocua.schema import load_tasks
from egocua.truth import Truth
from egocua.workspace import write_json_atomic
from sites.search import rank_records
from tools.catalog.build_task_search_neighborhoods import (
    MAX_PAGE_SIZE,
    _search_fields,
    _stage_requirements,
)

OUTPUT = Path("data/sources/search_query_reviews.json")
MODEL = "qwen3.5-flash"
MAX_ATTEMPTS = 4
BANNED_QUERY_WORDS = {
    "add",
    "find",
    "here",
    "i",
    "save",
    "second",
    "shelf",
    "table",
    "used",
}


def _title(entity: Mapping[str, Any]) -> str:
    return str(entity.get("name") or entity.get("title") or "")


def _task_payload(
    task: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    truth = Truth.of(task)
    before_options, after_options, _, _ = _stage_requirements(task)

    def option_titles(options: list[set[str]]) -> list[list[str]]:
        return [
            [_title(by_id[value]) for value in sorted(option)] for option in options
        ]

    return {
        "task_id": task["id"],
        "site": task["site"],
        "track": task["track"],
        "instructions": task["instructions"],
        "initial_entities": [
            _title(by_id[value]) for value in sorted(truth.initial_referent_ids)
        ],
        "final_entities": [
            _title(by_id[value]) for value in sorted(truth.final_referent_ids)
        ],
        "tempting_entities": [
            _title(by_id[value]) for value in sorted(truth.tempting_referent_ids)
        ],
        "before_search_acceptable_options": option_titles(before_options),
        "after_search_acceptable_options": option_titles(after_options),
    }


def _prompt(payload: Mapping[str, Any], feedback: str = "") -> str:
    return (
        """You design search queries for a realistic multimodal browser benchmark.
Return one JSON object with exactly these keys: before_query, after_query, rationale.

Rules:
- Queries must be short, natural English phrases a real person would type.
- Write noun-phrase website queries, not conversational instructions. Do not use
  words such as find, save, add, used, I, here, first, second, shelf, or table unless
  that word is genuinely part of public listing metadata.
- before_query uses only the initial scene/request; after_query incorporates the later
  change or clarification. The two may be identical when one query covers both stages.
- Preserve the need to use scene evidence. Do not copy a full catalog title, exact
  video title, model number, or hidden answer wording into a query.
- Use a category plus at most the few visible/clarified attributes needed.
- Do not concatenate every candidate title or list alternatives.
- Each query must plausibly retrieve at least one complete group listed in that
  stage's before_search_acceptable_options or after_search_acceptable_options.
  Groups are alternatives, not words to concatenate. Use shared category nouns
  and only useful visible/clarified attributes.
- For Graft, before searches the initial goal and after searches the new/changed goal.
- For Refresh retain, after may add the later distinguishing evidence while keeping
  the original object. For Clarify, after may use the user's physical clarification.
- Each query must contain 2-9 space-separated words. No quotation marks or operators.
- rationale is one concise sentence explaining why the pair is natural.

Previous-attempt feedback (empty on the first attempt):
"""
        + feedback
        + """

Task facts:
"""
        + json.dumps(payload, ensure_ascii=False)
    )


def _parse_json(text: str) -> dict[str, str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("model returned no JSON object")
    row = json.loads(match.group(0))
    if set(row) != {"before_query", "after_query", "rationale"}:
        raise ValueError("model returned an unexpected query schema")
    for key in row:
        if not isinstance(row[key], str) or not row[key].strip():
            raise ValueError(f"model returned an empty {key}")
        row[key] = " ".join(row[key].split())
    for key in ("before_query", "after_query"):
        words = row[key].split()
        if not 2 <= len(words) <= 9:
            raise ValueError(f"{key} must contain 2-9 words")
        if any(character in row[key] for character in ('"', ":")):
            raise ValueError(f"{key} contains title-like punctuation")
        forbidden = BANNED_QUERY_WORDS & {
            word.casefold().strip(".,!?()") for word in words
        }
        if forbidden:
            raise ValueError(f"{key} contains conversational words {sorted(forbidden)}")
    return row


def _reachability_feedback(
    payload: Mapping[str, Any],
    proposal: Mapping[str, str],
    entities: list[dict[str, Any]],
) -> str:
    """Return actionable feedback when a proposed query misses required listings."""

    by_title = {_title(entity): str(entity["entity_id"]) for entity in entities}
    failures: list[str] = []
    stages = (
        ("before", "before_query", "before_search_acceptable_options"),
        ("after", "after_query", "after_search_acceptable_options"),
    )
    for stage, query_key, option_key in stages:
        option_titles = [
            [str(value) for value in option] for option in payload[option_key]
        ]
        option_ids = [{by_title[title] for title in option} for option in option_titles]
        ranked = rank_records(
            entities,
            proposal[query_key],
            lambda row: _search_fields(str(payload["site"]), row),
        )[: MAX_PAGE_SIZE[str(payload["site"])]]
        reached = {str(entity["entity_id"]) for entity in ranked}
        if not any(option <= reached for option in option_ids):
            closest = min(
                option_titles,
                key=lambda option: sum(
                    by_title[title] not in reached for title in option
                ),
            )
            missing = [title for title in closest if by_title[title] not in reached]
            failures.append(
                f"{stage}_query reaches no complete acceptable option; the closest option "
                "still misses these listings within the full result window: "
                + json.dumps(missing, ensure_ascii=False)
            )
    if not failures:
        return ""
    return (
        "The previous pair failed deterministic reachability. Keep both queries natural; "
        "replace category words or remove over-specific modifiers instead of concatenating "
        "listing titles. " + " ".join(failures)
    )


def _propose(
    payload: dict[str, Any],
    endpoint: str,
    api_key: str,
    entities: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    feedback = ""
    last_error = "model produced no valid proposal"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "Return strict JSON only."},
                    {"role": "user", "content": _prompt(payload, feedback)},
                ],
                "temperature": 0,
                "max_tokens": 300,
                "enable_thinking": False,
            },
            timeout=180,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            proposal = _parse_json(str(content))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            feedback = (
                f"The previous response was invalid: {last_error}. Return a new pair."
            )
            continue
        feedback = _reachability_feedback(payload, proposal, entities)
        if feedback:
            last_error = feedback
            continue
        return str(payload["task_id"]), {
            **proposal,
            "model": MODEL,
            "attempts": attempt,
            "input_sha256": canonical_sha256(payload),
            "status": "proposed_pending_coding_agent_review",
        }
    raise ValueError(last_error)


def build(
    root: Path,
    task_ids: set[str] | None,
    concurrency: int,
    *,
    force: bool = False,
) -> dict[str, int]:
    if not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be between 1 and 4")
    load_dotenv(root / ".env", override=False)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"

    tasks = load_tasks(root / "data/tasks")
    if task_ids is not None:
        tasks = [task for task in tasks if str(task["id"]) in task_ids]
        missing = task_ids - {str(task["id"]) for task in tasks}
        if missing:
            raise ValueError(f"unknown task ids: {sorted(missing)}")
    site_entities = {
        site: list(load_site_data(root, site)["entities"])
        for site in {str(task["site"]) for task in tasks}
    }
    site_indexes = {
        site: {str(entity["entity_id"]): entity for entity in entities}
        for site, entities in site_entities.items()
    }
    payloads = [_task_payload(task, site_indexes[str(task["site"])]) for task in tasks]
    output_path = root / OUTPUT
    document: dict[str, Any] = (
        json.loads(output_path.read_text())
        if output_path.is_file()
        else {"schema_version": 1, "model": MODEL, "tasks": {}}
    )
    pending_payloads = [
        payload
        for payload in payloads
        if force
        or document["tasks"].get(str(payload["task_id"]), {}).get("input_sha256")
        != canonical_sha256(payload)
    ]
    completed = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                _propose,
                payload,
                endpoint,
                api_key,
                site_entities[str(payload["site"])],
            ): payload["task_id"]
            for payload in pending_payloads
        }
        for future in as_completed(futures):
            try:
                task_id, row = future.result()
            except Exception as exc:  # let the resumable next pass retry only failures
                failed += 1
                print(
                    f"failed {futures[future]} {type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue
            document["tasks"][task_id] = row
            write_json_atomic(output_path, document)
            completed += 1
            print(f"proposed {completed}/{len(pending_payloads)} {task_id}", flush=True)
    return {
        "requested": len(payloads),
        "reused": len(payloads) - len(pending_payloads),
        "proposed": completed,
        "failed": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build(
        Path(args.root).resolve(),
        set(args.task_id) if args.task_id else None,
        args.concurrency,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
