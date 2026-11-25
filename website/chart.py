"""Geometry for the homepage slope chart of success by setting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

SETTING_KEYS = ("computer_use", "offline", "online")
WIDTH = 760
HEIGHT = 400
TOP = 36
BOTTOM = 40
COLUMNS = (70, 300, 530)
FLOOR = 30.0
LABEL_GAP = 17


def _y(value: float) -> float:
    span = HEIGHT - TOP - BOTTOM
    return round(TOP + (100 - value) / (100 - FLOOR) * span, 1)


def _spread(positions: list[float]) -> list[float]:
    """Push sorted label positions apart so neighbours do not overlap."""
    placed: list[float] = []
    for position in positions:
        if placed and position - placed[-1] < LABEL_GAP:
            position = placed[-1] + LABEL_GAP
        placed.append(position)
    return placed


def slope_chart(
    rows: Sequence[Mapping[str, Any]], means: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    lines = []
    for row in rows:
        points = [(x, _y(row[key])) for x, key in zip(COLUMNS, SETTING_KEYS, strict=True)]
        lines.append(
            {
                "model": row["model"],
                "group": row["group"],
                "online": row["online"],
                "points": " ".join(f"{x},{y}" for x, y in points),
                "dots": points,
            }
        )
    columns = list(zip(COLUMNS, means, strict=True))
    mean_points = [(x, _y(float(item["value"]))) for x, item in columns]
    mean = {
        "points": " ".join(f"{x},{y}" for x, y in mean_points),
        "dots": mean_points,
        "value": means[-1]["value"],
    }
    labelled = sorted([*lines, mean], key=lambda item: item["dots"][-1][1])
    for item, label_y in zip(
        labelled, _spread([item["dots"][-1][1] for item in labelled]), strict=True
    ):
        item["label_y"] = label_y
    return {
        "width": WIDTH,
        "height": HEIGHT,
        "columns": [
            {"x": x, "label": item["setting"], "value": item["value"]}
            for x, item in columns
        ],
        "grid": [{"y": _y(value), "label": f"{value:.0f}"} for value in (40, 60, 80, 100)],
        "lines": lines,
        "mean": mean,
        "label_x": COLUMNS[-1] + 16,
        "axis_y": HEIGHT - BOTTOM + 26,
    }
