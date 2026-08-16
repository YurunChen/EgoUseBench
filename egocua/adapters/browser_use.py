from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv  # type: ignore[import-untyped]

from ..execution import RunConfig, RunContext, run_experiment_async


class BrowserUseUnavailable(RuntimeError):
    pass


ClarificationSlot = Literal["position", "appearance", "brand", "variant", "sequence"]
EXCLUDED_BROWSER_ACTIONS = (
    "search",
    "evaluate",
    "upload_file",
    "read_file",
    "write_file",
    "replace_file",
    "save_as_pdf",
)


def _imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from browser_use import Agent, Browser, ChatOpenAI, Tools  # type: ignore[import-not-found]
        from browser_use.browser.chrome import find_chrome_executable  # type: ignore[import-not-found]
    except ImportError as exc:
        raise BrowserUseUnavailable(
            'Browser-Use is optional. Install it with: pip install -e ".[browser-use]"'
        ) from exc
    return Agent, Browser, ChatOpenAI, Tools, find_chrome_executable


async def run_task(
    task: dict[str, Any], arm: str, *, model: str = "gpt-5.4-mini",
    max_steps: int = 40, replicate_id: int = 0, provider_seed: int | None = None,
    schedule_index: int | None = None,
    output_path: str | Path | None = None,
    asset_root: str | Path = ".", artifacts_root: str | Path | None = None,
    experiment_id: str = "manual",
) -> dict[str, Any]:
    load_dotenv(Path(asset_root).resolve() / ".env", override=False)
    config = RunConfig(
        agent="browser-use",
        model=model,
        provider="openai-compatible",
        model_version=model,
        transport="browser-use-chat-openai",
        max_steps=max_steps,
        replicate_id=replicate_id,
        schedule_index=schedule_index,
        provider_seed=provider_seed,
        asset_root=asset_root,
        artifacts_root=artifacts_root,
        experiment_id=experiment_id,
        output_path=output_path,
    )

    async def execute(context: RunContext) -> dict[str, Any]:
        initial = context.initial_payload
        empty_trace: dict[str, Any] = {"history": []}
        context.capture_partial_trace(empty_trace)
        Agent, Browser, ChatOpenAI, Tools, find_chrome_executable = _imports()
        tools = Tools()
        for action_name in EXCLUDED_BROWSER_ACTIONS:
            tools.exclude_action(action_name)
        if initial["clarification_available"]:
            @tools.action(description=(
                "Ask the wearer ONE clarification question. `slot` must be one of: "
                "position, appearance, brand, variant, sequence. Choose the slot whose "
                "answer would uniquely identify the intended item."
            ))
            def ask_user(slot: ClarificationSlot, question: str) -> str:
                return context.ask_user(slot, question)["payload"]

        executable_path = find_chrome_executable()
        if not executable_path:
            raise BrowserUseUnavailable("No local Chrome executable was found")
        with tempfile.TemporaryDirectory(prefix="egocua-browser-") as user_data_dir:
            browser = Browser(
                executable_path=executable_path,
                user_data_dir=user_data_dir,
                headless=True,
                keep_alive=False,
                enable_default_extensions=False,
                captcha_solver=False,
            )
            try:
                llm = ChatOpenAI(
                    model=model,
                    seed=provider_seed,
                    add_schema_to_system_prompt=True,
                    dont_force_structured_output=True,
                )
                prompt = _initial_prompt(initial, context.base_url)
                agent = Agent(task=prompt, llm=llm, browser=browser, tools=tools)

                async def on_step_start(active_agent: Any) -> None:
                    for event in context.advance():
                        if event["type"] == "visual_context_update":
                            message = (
                                "The glasses captured a more recent view. It is now available at "
                                f"{context.base_url}/context."
                            )
                        else:
                            message = f"The wearer added this requirement: {event['payload']}"
                        active_agent.add_new_task(message)

                history = await agent.run(on_step_start=on_step_start, max_steps=max_steps)
                trace = history.model_dump(mode="json")
                context.capture_partial_trace(trace)
                return trace
            finally:
                await browser.kill()

    return await run_experiment_async(task, arm, config, execute)


def _initial_prompt(initial: dict[str, Any], base_url: str) -> str:
    clarification = (
        "You may use ask_user once if the visual reference is ambiguous. "
        if initial["clarification_available"] else
        "No clarification tool is available; make the best supported choice. "
    )
    context_line = (
        f"Open {base_url}/context first. The page contains images captured by the wearer's glasses "
        "and the task website. Use only information available in the released glasses views and "
        "the website. "
    )
    return (
        f"{initial['instruction']}\n\n"
        f"{context_line}"
        f"{clarification}Complete the requested work through the website UI and use its Submit action. "
        "Do not claim success without a confirmed submission."
    )
