"""Terminal-guided human execution through the exact benchmark harness."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..execution import RunConfig, RunContext, run_experiment


CommandReader = Callable[[str], str]
MessageWriter = Callable[[str], Any]
BrowserOpener = Callable[[str], Any]


def run_task(
    task: dict[str, Any],
    arm: str,
    *,
    evaluator_id: str,
    max_steps: int = 40,
    replicate_id: int = 0,
    output_path: str | Path | None = None,
    asset_root: str | Path = ".",
    artifacts_root: str | Path | None = None,
    experiment_id: str = "human-validation",
    command_reader: CommandReader = input,
    message_writer: MessageWriter = print,
    browser_opener: BrowserOpener = webbrowser.open,
) -> dict[str, Any]:
    """Collect one human Run Artifact without bypassing task events or the website."""

    evaluator = evaluator_id.strip()
    if not evaluator:
        raise ValueError("evaluator_id must be non-empty")
    config = RunConfig(
        agent="human",
        model="human",
        provider="human",
        model_version="human",
        transport="egocua-browser-terminal-v2",
        max_steps=max_steps,
        replicate_id=replicate_id,
        asset_root=asset_root,
        artifacts_root=artifacts_root,
        experiment_id=experiment_id,
        output_path=output_path,
        operator_id=evaluator,
    )

    def execute(context: RunContext) -> dict[str, Any]:
        trace: dict[str, Any] = {
            "evaluator_id": evaluator,
            "commands": [],
            "stop_reason": None,
        }
        context.capture_partial_trace(trace)
        context_url = f"{context.base_url}/context"
        message_writer(f"Instruction: {context.initial_payload['instruction']}")
        message_writer(f"Task website: {context_url}")
        message_writer(
            "Complete one meaningful browser action per step. Press Enter to continue; "
            "use 'ask <slot> <question>' once when available; use 'finish' after submitting."
        )
        browser_opener(context_url)

        while context.current_step < max_steps:
            events = context.advance()
            for event in events:
                if event["type"] == "visual_context_update":
                    message_writer(
                        f"Step {context.current_step}: a newer glasses view is available; "
                        "reload the task context page."
                    )
                else:
                    message_writer(
                        f"Step {context.current_step}: wearer update — {event['payload']}"
                    )
            command = command_reader(
                f"Step {context.current_step}/{max_steps}> "
            ).strip()
            entry: dict[str, Any] = {
                "step": context.current_step,
                "command": command or "continue",
            }
            trace["commands"].append(entry)
            if command == "finish":
                trace["stop_reason"] = "operator_finished"
                break
            if command.startswith("ask "):
                parts = command.split(maxsplit=2)
                if len(parts) != 3 or not parts[2].strip():
                    message_writer("Invalid ask command; use: ask <slot> <question>")
                    entry["accepted"] = False
                    continue
                answer = context.ask_user(parts[1], parts[2])
                entry["accepted"] = True
                entry["answer_event"] = answer
                message_writer(json.dumps(answer, ensure_ascii=False))
            if context.submission_completed:
                trace["stop_reason"] = "submitted"
                break
        else:
            trace["stop_reason"] = "max_steps"
        context.capture_partial_trace(trace)
        return trace

    return run_experiment(task, arm, config, execute)
