"""Shared deterministic runtime for the local benchmark websites."""

from __future__ import annotations

import mimetypes
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from flask import Flask, Response
from werkzeug.serving import BaseWSGIServer, WSGIRequestHandler, make_server

from egocua.protocol import initial_visible_packets
from egocua.workspace import resolve_within


class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        pass


@dataclass
class SiteState:
    """State shared by all sites and serialized into one normalized final state."""

    task: dict[str, Any]
    arm: str
    data: dict[str, Any]
    asset_root: Path = field(default_factory=Path.cwd)
    current_step: int = 0
    visible_packet_ids: set[str] = field(default_factory=set)
    selected_entity_ids: list[str] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    submitted_entity_ids: list[str] = field(default_factory=list)
    submitted_fields: dict[str, str] = field(default_factory=dict)
    submitted: bool = False
    successful_search_seen: bool = False
    actions: list[dict[str, Any]] = field(default_factory=list)
    event_gate: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if not self.visible_packet_ids:
            self.visible_packet_ids = initial_visible_packets(self.task, self.arm)

    def reveal(self, packet_id: str) -> None:
        self.visible_packet_ids.add(packet_id)

    def log(self, action_type: str, **payload: Any) -> None:
        self.actions.append({"type": action_type, "step": self.current_step, **payload})

    def select(self, entity_id: str) -> None:
        self.selected_entity_ids.append(entity_id)
        self.log("select_entity", entity_id=entity_id)

    def deselect(self, entity_id: str) -> None:
        self.selected_entity_ids = [
            selected for selected in self.selected_entity_ids if selected != entity_id
        ]
        self.log("deselect_entity", entity_id=entity_id)

    def set_field(self, name: str, value: str) -> None:
        self.fields[name] = value
        self.log("set_field", field=name, value=value)

    def commit(self) -> None:
        self.submitted_entity_ids = list(self.selected_entity_ids)
        self.submitted_fields = dict(self.fields)
        self.submitted = True
        self.log(
            "submit",
            entity_ids=list(self.selected_entity_ids),
            fields=dict(self.fields),
        )

    def result_record(
        self, *, valid: bool = True, reason: str | None = None,
    ) -> dict[str, Any]:
        return {
            "task_id": self.task["id"],
            "track": self.task["track"],
            "arm": self.arm,
            "run_status": {"valid": valid, "reason": reason},
            "initial_state": {
                "selected_entity_ids": [],
                "fields": {},
                "submission_completed": False,
            },
            "steps": self.current_step,
            "actions": self.actions,
            "final_state": {
                "selected_entity_ids": list(self.submitted_entity_ids),
                "fields": dict(self.submitted_fields),
                "submission_completed": self.submitted,
            },
        }


class RuntimeState(Protocol):
    task: dict[str, Any]
    arm: str
    asset_root: Path
    current_step: int
    visible_packet_ids: set[str]
    submitted: bool
    successful_search_seen: bool
    actions: list[dict[str, Any]]
    event_gate: Callable[[], bool] | None

    def reveal(self, packet_id: str) -> None: ...

    def result_record(
        self, *, valid: bool = True, reason: str | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SiteRuntime:
    state: RuntimeState
    server: "SiteServer"


class SiteServer:
    def __init__(
        self, state: RuntimeState, app: Flask, host: str = "127.0.0.1", port: int = 0,
    ) -> None:
        self.state = state
        self.httpd: BaseWSGIServer = make_server(
            host,
            port,
            app,
            threaded=True,
            request_handler=QuietRequestHandler,
        )
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )

    @property
    def base_url(self) -> str:
        return f"http://{self.httpd.host}:{self.httpd.server_port}"

    def start(self) -> "SiteServer":
        self.thread.start()
        return self

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)

    def __enter__(self) -> "SiteServer":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()


def visible_observations(state: RuntimeState) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, packet)
        for index, packet in enumerate(state.task["visual_packets"])
        if packet["id"] in state.visible_packet_ids
    ]


def register_observation_routes(
    app: Flask,
    state: RuntimeState,
    render: Callable[[str, int], Any],
    context_html: Callable[[list[tuple[int, dict[str, Any]]]], str],
    frame_html: Callable[[int, int], str],
) -> None:
    """Register the identical glasses-observation surface on one Flask app."""

    def context() -> Any:
        return render(context_html(visible_observations(state)), 200)

    def observation_image(observation_index: int, image_index: int) -> Any:
        packets = state.task["visual_packets"]
        packet = packets[observation_index] if observation_index < len(packets) else None
        if (
            packet is None
            or packet["id"] not in state.visible_packet_ids
            or image_index >= len(packet["frame_paths"])
        ):
            return render("<h1>Frame unavailable</h1>", 404)
        try:
            path = resolve_within(
                state.asset_root,
                packet["frame_paths"][image_index],
                context="Released Observation frame",
            )
        except (ValueError, FileNotFoundError):
            return render("<h1>Frame missing</h1>", 404)
        return Response(
            path.read_bytes(),
            mimetype=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )

    def observation_page(observation_index: int, image_index: int) -> Any:
        return render(frame_html(observation_index, image_index), 200)

    app.add_url_rule("/", "context_root", context, methods=["GET"])
    app.add_url_rule("/context", "context", context, methods=["GET"])
    app.add_url_rule(
        "/observations/<int:observation_index>/images/<int:image_index>",
        "observation_image",
        observation_image,
        methods=["GET"],
    )
    app.add_url_rule(
        "/observation/<int:observation_index>/image/<int:image_index>",
        "observation_page",
        observation_page,
        methods=["GET"],
    )
