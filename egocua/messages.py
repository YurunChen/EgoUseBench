"""Canonical wording shown to every Agent Adapter.

User instructions remain verbatim. These helpers label observation batches and
released events; execution rules belong to the adapter system prompt.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

MODEL_INPUT_POLICY = "wearer-session-v5"
GRAFT_UPFRONT_INPUT_POLICY = "wearer-session-graft-upfront-v6"
STREAM_INPUT_POLICY = "wearer-session-stream-v2"
INITIAL_STEP = 1

# Visual history: every parameter of the Retention conditions lives here.
HISTORY_RETENTIONS = ("full", "compressed", "summary")
DEFAULT_HISTORY_RETENTION = "compressed"
RECENT_WHOLE_BATCHES = 2
"""Every retention keeps the opening and the latest two delivered batches whole."""
HISTORICAL_RECENT_WHOLE_BATCHES = 4
"""wearer-session-stream-v2 (no visual_history) kept the latest four whole."""
COMPRESSED_FRAME_STRIDE = 2
"""Compressed history keeps every other delivered frame: 0.5 fps at 1 fps replay."""
COMPRESSED_HISTORY_NOTE = "Earlier batch shown at half its frame rate."
SUMMARY_LIMIT = 480
SUMMARY_PROMPT = (
    "When asked for batch summaries, add a top-level visual_summaries object "
    "to your action JSON, keyed by the requested batch IDs. For each ID describe "
    "only what is visible in that batch (objects, distinguishing details, actions), "
    f"in at most {SUMMARY_LIMIT} characters. Do not use the user request, website, "
    "other batches, or guesses to fill in details."
)

KEY_EVIDENCE = "key_evidence"
"""Batch flag: it holds a reviewed frame, so the agent's context keeps it whole."""
SOURCE_PREVIEW = "source_preview"
STRATEGY_HINT = (
    "Later wearer images or requests may change the target; "
    "retain still-valid choices and remove invalid ones."
)


def model_input_policy(track: str, condition: Mapping[str, Any]) -> str:
    """Version only the input that changed; other tracks retain their valid runs."""

    if (
        track == "graft"
        and condition["schedule"] == "upfront"
        and condition["scope"] != "initial_only"
        and condition.get("frame_delivery_steps") is None
    ):
        return GRAFT_UPFRONT_INPUT_POLICY
    retention = condition.get("visual_history")
    if retention is not None:
        return f"wearer-session-stream-v3-{retention}"
    if condition.get("frame_delivery_steps") is not None:
        return STREAM_INPUT_POLICY
    return MODEL_INPUT_POLICY


def older_batch_indices(batch_count: int, policy: str | None) -> range:
    """Batches outside the opening and the recent window, which retention may change."""
    recent = HISTORICAL_RECENT_WHOLE_BATCHES if policy is None else RECENT_WHOLE_BATCHES
    return range(1, batch_count - recent)


def retained_images(
    image_counts: list[int], key_evidence: list[bool], policy: str | None = None
) -> list[range]:
    """Delivered image positions retained per batch; None replays the historical protocol.

    Older compressed batches drop every delivered image and are shown through
    their ``compressed_history`` images instead; older summary batches are text.
    """
    older = older_batch_indices(len(image_counts), policy)
    retained = []
    for index, (count, is_key) in enumerate(
        zip(image_counts, key_evidence, strict=True)
    ):
        images = range(count)
        if index in older and policy != "full":
            if policy is not None:
                images = range(0)
            elif not is_key:
                images = images[-1:]
        retained.append(images)
    return retained


def compressed_history(observation: Mapping[str, Any]) -> dict[str, Any]:
    """The batch as compressed history sees it: evenly spaced, half the frames."""

    thinned = dict(observation)
    for key in ("frame_paths", "timestamps_seconds"):
        if key in observation:
            thinned[key] = list(observation[key])[::COMPRESSED_FRAME_STRIDE]
    return thinned


def wearer_said(text: str, step: int) -> str:
    """Factual label for wearer speech already received before an agent step."""

    _require_step(step)
    return f'The wearer said before step {step}: "{text}"'


def wearer_answered(text: str, step: int) -> str:
    """Factual label for a clarification answer received before an agent step."""

    _require_step(step)
    return f'The wearer answered before step {step}: "{text}"'


def glasses_received(step: int) -> str:
    """Factual label for glasses images received before an agent step."""

    _require_step(step)
    return f"Glasses images received before step {step}:"


def initial_glasses_label(observation: Mapping[str, Any]) -> str:
    """Keep future previews distinct from the opening live batch."""
    if observation.get(SOURCE_PREVIEW) is not True:
        return glasses_received(INITIAL_STEP)
    times = observation["timestamps_seconds"]
    return f"Glasses preview from source t={times[0]:g}–{times[-1]:g} s:"


def release_label_step(event: Mapping[str, Any]) -> int:
    """Step before which this event is already in the model context.

    A clarification answer is produced during the step that asked, so the next
    step is the first one that sees it. Scheduled speech and images arrive at
    the start of their recorded step.
    """

    step = event.get("step")
    _require_step(step)
    if event.get("type") == "clarification_answer":
        return step + 1
    return step


def release_text(event: Mapping[str, Any], step: int) -> str:
    """One factual label for a mid-run release. The payload stays verbatim."""

    if event.get("type") == "visual_context_update":
        return glasses_received(step)
    if event.get("type") == "instruction_injection":
        return wearer_said(str(event.get("payload")), step)
    if event.get("type") == "clarification_answer":
        return wearer_answered(str(event.get("payload")), step)
    raise ValueError(f"unknown release event type {event.get('type')!r}")


def invalid_action_feedback(error: BaseException) -> str:
    """Action-feedback sentence for one malformed model response."""

    return f"Your last response was not a valid action: {error}"


def _require_step(step: object) -> None:
    if type(step) is not int or step < 1:
        raise ValueError("release step must be a positive integer")


def release_frames(event: Mapping[str, Any]) -> list[str]:
    payload = event.get("payload")
    if isinstance(payload, Mapping):
        frames = payload.get("frame_paths", [])
        return [str(frame) for frame in frames]
    return []


def initial_context(
    initial: Mapping[str, Any], initial_site_state: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Combine released task context with the registered website starting state."""
    context = deepcopy(dict(initial))
    if initial_site_state is not None:
        context["initial_site_state"] = deepcopy(dict(initial_site_state))
    return context
