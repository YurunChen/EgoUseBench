"""Shared deterministic runtime for the local benchmark websites.

A site knows nothing about tasks, tracks, arms, or the evidence timeline. It owns
a selection set, field values, and a submission flag, and it records every
state-changing action into an `ActionTrace` that the executor also writes
protocol events into.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from flask import Flask
from werkzeug.serving import BaseWSGIServer, WSGIRequestHandler, make_server


class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        pass


@dataclass
class ActionTrace:
    """Ordered record of site actions and protocol events, stamped with the run step."""

    current_step: int = 0
    actions: list[dict[str, Any]] = field(default_factory=list)
    on_record: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    def record(self, action_type: str, **payload: Any) -> None:
        self.actions.append({"type": action_type, "step": self.current_step, **payload})
        if self.on_record is not None:
            self.on_record()

    def record_event(self, event: dict[str, Any]) -> None:
        self.actions.append(event)


@dataclass
class SiteState:
    """Selection, fields, and submission state shared by all sites."""

    data: dict[str, Any]
    asset_root: Path = field(default_factory=Path.cwd)
    trace: ActionTrace = field(default_factory=ActionTrace)
    selected_entity_ids: list[str] = field(default_factory=list)
    fields: dict[str, str] = field(default_factory=dict)
    submitted_entity_ids: list[str] = field(default_factory=list)
    submitted_fields: dict[str, str] = field(default_factory=dict)
    submitted: bool = False
    _logged_state: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logged_state = self.live_state()

    def initialize(self, state: dict[str, Any]) -> None:
        """Set the registered starting state before any model action."""
        if self.trace.actions:
            raise RuntimeError("site initialization must precede model actions")
        self.selected_entity_ids = list(state["selected_entity_ids"])
        self.fields = dict(state["fields"])
        self._logged_state = self.live_state()

    @property
    def entities(self) -> list[dict[str, Any]]:
        return list(self.data["entities"])

    @property
    def actions(self) -> list[dict[str, Any]]:
        return self.trace.actions

    def log(self, action_type: str, **payload: Any) -> None:
        current = self.live_state()
        payload.update(state_before=self._logged_state, state_after=current)
        self.trace.record(action_type, **payload)
        self._logged_state = current

    def select(self, entity_id: str) -> None:
        self.selected_entity_ids.append(entity_id)
        self.log("select_entity", entity_id=entity_id)

    def add_quantity(self, entity_id: str, quantity: int) -> None:
        self.selected_entity_ids.extend([entity_id] * quantity)
        self.log("select_entity", entity_id=entity_id, quantity=quantity)

    def set_quantity(self, entity_id: str, quantity: int) -> None:
        selected: list[str] = []
        placed = False
        for entry in self.selected_entity_ids:
            if entry == entity_id:
                if not placed:
                    selected.extend([entity_id] * quantity)
                    placed = True
            else:
                selected.append(entry)
        self.selected_entity_ids = selected
        action = "deselect_entity" if quantity == 0 else "select_entity"
        self.log(action, entity_id=entity_id, quantity=quantity)

    def save(self, entity_id: str) -> None:
        if entity_id not in self.selected_entity_ids:
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

    def final_state(self) -> dict[str, Any]:
        return {
            "selected_entity_ids": list(self.submitted_entity_ids),
            "fields": dict(self.submitted_fields),
            "submission_completed": self.submitted,
        }

    def live_state(self) -> dict[str, Any]:
        return {
            "selected_entity_ids": list(self.selected_entity_ids),
            "fields": dict(self.fields),
        }


@dataclass(frozen=True)
class SiteRuntime:
    state: SiteState
    server: "SiteServer"


class SiteServer:
    def __init__(
        self,
        state: SiteState,
        app: Flask,
        host: str = "127.0.0.1",
        port: int = 0,
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
        self._started = False

    @property
    def base_url(self) -> str:
        return f"http://{self.httpd.host}:{self.httpd.server_port}"

    def start(self) -> "SiteServer":
        if self._started:
            raise RuntimeError("site server has already started")
        try:
            self.thread.start()
        except BaseException:
            self.httpd.server_close()
            raise
        self._started = True
        return self

    def close(self) -> None:
        if self._started:
            self.httpd.shutdown()
        self.httpd.server_close()
        if self._started:
            self.thread.join(timeout=2)
            self._started = False

    def __enter__(self) -> "SiteServer":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.close()
