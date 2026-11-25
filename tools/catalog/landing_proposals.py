"""Shared source roles and incremental landing proposals for the two catalog sites."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from egocua.run_ledger import canonical_sha256
from egocua.sources import iter_manifests
from egocua.workspace import write_json_atomic

MODEL = "qwen3.5-flash"


def source_records(root: Path) -> dict[str, Mapping[str, Any]]:
    rows: dict[str, Mapping[str, Any]] = {}
    for _, payload in iter_manifests(root):
        for row in payload.get("tasks", []):
            if isinstance(row, Mapping) and row.get("task_id"):
                rows[str(row["task_id"])] = row
    return rows


def role_sets(
    task: Mapping[str, Any], blueprint: Mapping[str, Any]
) -> tuple[set[str], set[str]]:
    track = str(task["track"])
    if track == "graft":
        initial_states = [
            {str(role) for role in state}
            for state in blueprint.get("initial_goal_roles", [])
        ]
        final_states = [
            {str(role) for role in state}
            for state in blueprint.get("final_goal_roles", [])
        ]
        if not initial_states or not final_states:
            raise ValueError(f"{task['id']}: Graft role states are empty")
        final = final_states[0]
        all_final_roles = set().union(*final_states)
        transition = str((task.get("graft") or {}).get("transition") or "")
        if transition == "augment":
            candidates = [state for state in initial_states if state < final]
            initial = min(
                candidates or initial_states,
                key=lambda state: (len(state), sorted(state)),
            )
        elif transition == "drop":
            candidates = [state for state in initial_states if final < state]
            initial = max(
                candidates or initial_states,
                key=lambda state: (len(state), sorted(state)),
            )
        else:
            # Prefer a genuinely superseded broad state over the final landing
            # when the old blueprint offered both as A-side alternatives.
            initial = max(
                initial_states,
                key=lambda state: (
                    len(state - all_final_roles),
                    any("broad" in role for role in state),
                    len(state ^ final),
                    sorted(state),
                ),
            )
        return initial, final
    if track == "clarify":
        return (
            {str(value) for value in blueprint["candidate_roles"]},
            {str(blueprint["hidden_target_role"])},
        )
    return (
        {str(blueprint["initial_target_role"])},
        {str(blueprint["final_target_role"])},
    )


@dataclass(frozen=True)
class LandingProposals:
    output: Path
    prompt: Callable[[Mapping[str, Any], str], str]
    parse: Callable[[str, Mapping[str, Any]], dict[str, Any]]

    def build(
        self, root: Path, payloads: list[dict[str, Any]], concurrency: int
    ) -> dict[str, int]:
        load_dotenv(root / ".env", override=False)
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not api_key or not base_url:
            raise ValueError("OPENAI_API_KEY and OPENAI_BASE_URL are required")
        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        path = root / self.output
        document: dict[str, Any] = (
            json.loads(path.read_text())
            if path.is_file()
            else {"schema_version": 1, "model": MODEL, "tasks": {}}
        )
        pending = [
            payload
            for payload in payloads
            if document["tasks"].get(str(payload["task_id"]), {}).get("input_sha256")
            != canonical_sha256(payload)
        ]
        completed = failed = 0
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(self._propose, payload, endpoint, api_key): str(
                    payload["task_id"]
                )
                for payload in pending
            }
            for future in as_completed(futures):
                try:
                    task_id, row = future.result()
                except Exception as exc:
                    failed += 1
                    print(
                        f"failed {futures[future]} {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    continue
                document["tasks"][task_id] = row
                write_json_atomic(path, document)
                completed += 1
                print(f"proposed {completed}/{len(pending)} {task_id}", flush=True)
        return {
            "requested": len(payloads),
            "reused": len(payloads) - len(pending),
            "proposed": completed,
            "failed": failed,
        }

    def _propose(
        self,
        payload: dict[str, Any],
        endpoint: str,
        api_key: str,
    ) -> tuple[str, dict[str, Any]]:
        feedback = ""
        for attempt in range(1, 4):
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
                        {"role": "user", "content": self.prompt(payload, feedback)},
                    ],
                    "temperature": 0,
                    "max_tokens": 2200,
                    "enable_thinking": False,
                },
                timeout=180,
            )
            response.raise_for_status()
            try:
                proposal = self.parse(
                    str(response.json()["choices"][0]["message"]["content"]), payload
                )
            except (ValueError, json.JSONDecodeError) as exc:
                feedback = f"Previous response failed validation: {exc}"
                continue
            return str(payload["task_id"]), {
                **proposal,
                "model": MODEL,
                "attempts": attempt,
                "input_sha256": canonical_sha256(payload),
                "status": "proposed_pending_coding_agent_review",
            }
        raise ValueError(feedback or "model produced no valid landing proposal")
