"""BrowserGym owns browser observations, element IDs and native UI actions."""

from __future__ import annotations

import ast
import asyncio
import base64
import json
import inspect
import os
import signal
import sys
from contextlib import redirect_stdout
from io import BytesIO
from typing import Any

from PIL import Image

from ..execution import RemoteRunError, exception_details
from urllib.parse import urlsplit

from browsergym.core.action import functions
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.env import BrowserEnv
from browsergym.core.task import AbstractBrowserTask
from browsergym.utils.obs import flatten_axtree_to_str


def action_set() -> HighLevelActionSet:
    return HighLevelActionSet(
        subsets=["nav", "tab", "custom"],
        custom_actions=[
            functions.click,
            functions.fill,
            functions.clear,
            functions.press,
            functions.scroll,
            functions.select_option,
            functions.hover,
        ],
        multiaction=False,
        strict=True,
        retry_with_force=False,
    )


def compile_action(name: str, parameters: dict[str, Any]) -> str:
    """Encode one supported browser action; never repair model arguments."""
    if name not in action_set().action_set:
        raise ValueError(f"Unknown browser action: {name}")
    inspect.signature(getattr(functions, name)).bind(**parameters)
    arguments = ", ".join(f"{key}={value!r}" for key, value in parameters.items())
    return f"{name}({arguments})"


def action_definitions() -> list[dict[str, Any]]:
    """Use BrowserGym's action schema, including its two union/list parameters."""
    definitions = action_set().to_tool_description(add_examples=False)
    for definition in definitions:
        properties = definition["parameters"]["properties"]
        if definition["name"] == "click":
            properties["modifiers"] = {"type": "array", "items": {"type": "string"}}
        elif definition["name"] == "select_option":
            properties["options"] = {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ]
            }
    return definitions


class EgoBrowserTask(AbstractBrowserTask):
    """Expose a frozen local site; the shared Checker alone evaluates success."""

    def __init__(self, seed: int, start_url: str, instruction: str) -> None:
        super().__init__(seed)
        self.start_url = start_url
        self.instruction = instruction
        self.viewport = {"width": 1512, "height": 982}
        self.slow_mo = 0

    @classmethod
    def get_task_id(cls) -> str:
        return "egocua"

    def setup(self, page: Any) -> tuple[str, dict[str, Any]]:
        origin = urlsplit(self.start_url)

        def local_site_only(route: Any) -> None:
            target = urlsplit(route.request.url)
            if (target.scheme, target.netloc) == (origin.scheme, origin.netloc):
                route.continue_()
            else:
                route.abort()

        page.context.route("**/*", local_site_only)
        page.goto(self.start_url)
        return self.instruction, {}

    def validate(
        self, page: Any, chat_messages: list[Any]
    ) -> tuple[float, bool, str, dict[str, Any]]:
        return 0.0, False, "", {}


BROWSER_ACTION_TIMEOUT_SECONDS = 30
BROWSER_CLOSE_GRACE_SECONDS = 3
MAX_WAIT_MS = 1000
BROWSER_MESSAGE_LIMIT = 16 * 1024 * 1024


class BrowserActions:
    def __init__(self, start_url: str) -> None:
        self.origin = urlsplit(start_url)
        self.actions = action_set()

    def to_python_code(self, action: str) -> str:
        code = self.actions.to_python_code(action)
        call = ast.parse(action).body[0].value
        if call.func.id == "goto":
            value = (
                call.args[0]
                if call.args
                else next(k.value for k in call.keywords if k.arg == "url")
            )
            target = urlsplit(ast.literal_eval(value))
            if target.scheme not in ("http", "https") or (
                target.scheme,
                target.netloc,
            ) != (self.origin.scheme, self.origin.netloc):
                raise ValueError("Navigation must use the task website's HTTP origin")
        if call.func.id == "noop":
            value = (
                call.args[0]
                if call.args
                else next(
                    (k.value for k in call.keywords if k.arg == "wait_ms"),
                    ast.Constant(1000),
                )
            )
            duration = ast.literal_eval(value)
            if (
                not isinstance(duration, (int, float))
                or not 0 <= duration <= MAX_WAIT_MS
            ):
                raise ValueError(f"Waiting must be between 0 and {MAX_WAIT_MS} ms")
        return code


def create_environment(start_url: str, instruction: str) -> BrowserEnv:
    return BrowserEnv(
        task_entrypoint=EgoBrowserTask,
        task_kwargs={"start_url": start_url, "instruction": instruction},
        action_mapping=BrowserActions(start_url).to_python_code,
        headless=True,
        pw_chromium_kwargs={"channel": "chrome"},
    )


def observation_text(observation: dict[str, Any]) -> str:
    tree = flatten_axtree_to_str(
        observation["axtree_object"],
        extra_properties=observation["extra_element_properties"],
        filter_visible_only=True,
    )
    tabs = "\n".join(
        f"Tab {index}: {url} — {title}"
        for index, (url, title) in enumerate(
            zip(
                observation["open_pages_urls"],
                observation["open_pages_titles"],
                strict=True,
            )
        )
    )
    return (
        f"URL: {observation['url']}\nActive tab: {observation['active_page_index']}\n"
        f"{tabs}\nLast browser action: {observation['last_action']}\n"
        f"Last action error: {observation['last_action_error']}\n"
        f"<browser_state>\n{tree}\n</browser_state>"
    )


class BrowserWorkerError(RemoteRunError):
    """A browser observation could not be produced; not a model action error."""


class BrowserSession:
    """Isolate only Playwright; task state, model calls and records stay in the host."""

    def __init__(self) -> None:
        self.process: asyncio.subprocess.Process | None = None

    async def start(
        self,
        start_url: str,
        instruction: str,
        seed: int,
    ) -> dict[str, Any]:
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            __name__,
            "--worker",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
            limit=BROWSER_MESSAGE_LIMIT,
        )
        return await self._request(
            {
                "start_url": start_url,
                "instruction": instruction,
                "seed": seed,
            }
        )

    async def step(self, action: str) -> dict[str, Any]:
        return await self._request({"action": action})

    async def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        assert (
            self.process is not None
            and self.process.stdin is not None
            and self.process.stdout is not None
        )
        self.process.stdin.write(json.dumps(message).encode() + b"\n")
        await self.process.stdin.drain()
        try:
            line = await asyncio.wait_for(
                self.process.stdout.readline(), BROWSER_ACTION_TIMEOUT_SECONDS
            )
        except TimeoutError as error:
            raise TimeoutError(
                f"BrowserGym did not finish within {BROWSER_ACTION_TIMEOUT_SECONDS} seconds"
            ) from error
        if not line:
            raise RuntimeError(
                "BrowserGym worker exited before returning an observation"
            )
        result = json.loads(line)
        if "error" in result:
            raise BrowserWorkerError(result["error"])
        return result["observation"]

    async def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        cleanup = asyncio.create_task(self.process.communicate())
        try:
            await asyncio.wait_for(asyncio.shield(cleanup), BROWSER_CLOSE_GRACE_SECONDS)
        except TimeoutError:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await cleanup


def _wire_observation(env: BrowserEnv, observation: dict[str, Any]) -> dict[str, Any]:
    stream = BytesIO()
    Image.fromarray(observation["screenshot"]).save(stream, format="PNG")
    result = {
        "url": observation["url"],
        "text": observation_text(observation),
        "screenshot": base64.b64encode(stream.getvalue()).decode(),
        "last_action_error": observation["last_action_error"],
        "browser_version": env.browser.version,
        "viewport": env.task.viewport,
    }
    return result


def _worker() -> None:
    env = None
    try:
        for line in sys.stdin:
            message = json.loads(line)
            with redirect_stdout(sys.stderr):
                if env is None:
                    env = create_environment(
                        message["start_url"], message["instruction"]
                    )
                    observation, _ = env.reset(seed=message["seed"])
                else:
                    observation, _, terminated, truncated, _ = env.step(
                        message["action"]
                    )
                    if terminated or truncated:
                        raise RuntimeError("BrowserGym ended before the host finished")
                result = {"observation": _wire_observation(env, observation)}
            print(json.dumps(result), flush=True)
    except Exception as error:
        print(json.dumps({"error": exception_details(error)}), flush=True)
    finally:
        if env is not None:
            with redirect_stdout(sys.stderr):
                env.close()


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Use the egocua CLI to start a browser session")
    _worker()
