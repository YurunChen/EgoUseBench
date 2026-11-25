"""Find HD-EPIC clips whose object tracks already contain a decision, structurally.

Screening Ego2Web meant looking at contact sheets until something turned up, at a
yield around one in eight. HD-EPIC does not need that: `assoc_info.json` records,
per recording, every tracked object and the time segments where it moved. Two
objects of the same kind, handled one after the other, is the shape a `sequence`
decision has — and that is a fact about the annotation, not a guess about wording.

What this produces is a shortlist with exact seconds, so a contact sheet can be
rendered on the window that matters. It decides nothing: whether those two objects
are really confusable, and whether the frames show the order, is Stage A and B.

Usage:
    PYTHONPATH=. uv run python tools/mine/mine_hdepic_decisions.py --root .
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

ASSOC_INFO = (
    "reference/sources/HD-EPIC/annotations/scene-and-object-movements/assoc_info.json"
)
REPORT = "artifacts/reports/hdepic-decision-candidates.json"

# Ordinals and sides are how the annotators wrote "which one" — they name the
# instance, not the kind, so they are stripped to find objects of the same kind.
INSTANCE_WORDS = re.compile(
    r"\b(first|second|third|fourth|fifth|left|right|top|bottom|upper|lower|"
    r"half|halves|piece|slice|of|the|a|an)\b"
)
MINIMUM_GAP_SECONDS = 1.0
# The window spans both objects, but a packet does not sample across it uniformly —
# it takes frames around each object's own handling segment, the way a Refresh task
# releases V0 and V1. So the span may be minutes; what has to be short is each
# object's own segment, which the annotation gives directly.
MAXIMUM_WINDOW_SECONDS = 600.0


@dataclass(frozen=True)
class Handled:
    name: str
    start: float
    end: float


@dataclass(frozen=True)
class Candidate:
    recording: str
    kind: str
    objects: tuple[Handled, ...]

    @property
    def window(self) -> tuple[float, float]:
        return (self.objects[0].start, self.objects[-1].end)

    @property
    def span(self) -> float:
        start, end = self.window
        return end - start


def kind_of(name: str) -> str:
    """The kind an object belongs to, with the instance words taken out."""

    return " ".join(INSTANCE_WORDS.sub(" ", name.casefold()).split())


def handled_objects(recording: Mapping[str, Any]) -> list[Handled]:
    handled = []
    for entry in recording.values():
        segments = [track["time_segment"] for track in entry.get("tracks", [])]
        if not segments:
            continue
        handled.append(
            Handled(
                name=str(entry["name"]),
                start=min(float(segment[0]) for segment in segments),
                end=max(float(segment[1]) for segment in segments),
            )
        )
    return handled


def candidates_in(recording_id: str, recording: Mapping[str, Any]) -> list[Candidate]:
    """Objects of one kind, handled one clearly after another."""

    by_kind: dict[str, list[Handled]] = {}
    for handled in handled_objects(recording):
        kind = kind_of(handled.name)
        if kind:
            by_kind.setdefault(kind, []).append(handled)

    found = []
    for kind, objects in by_kind.items():
        if len(objects) < 2:
            continue
        ordered = sorted(objects, key=lambda item: item.start)
        sequential = [ordered[0]]
        for handled in ordered[1:]:
            if handled.start - sequential[-1].end >= MINIMUM_GAP_SECONDS:
                sequential.append(handled)
        if len(sequential) < 2:
            continue
        candidate = Candidate(recording_id, kind, tuple(sequential))
        if candidate.span <= MAXIMUM_WINDOW_SECONDS:
            found.append(candidate)
    return found


def mine(root: Path) -> dict[str, Any]:
    assoc = json.loads((root / ASSOC_INFO).read_text())
    found = [
        candidate
        for recording_id, recording in assoc.items()
        for candidate in candidates_in(recording_id, recording)
    ]
    found.sort(key=lambda item: (item.recording, item.window[0]))
    by_participant: dict[str, int] = {}
    for candidate in found:
        participant = candidate.recording.split("-")[0]
        by_participant[participant] = by_participant.get(participant, 0) + 1
    return {
        "schema_version": 1,
        "interpretation": (
            "Each row is a recording window where two or more objects of one kind were "
            "handled in sequence. That is the shape a sequence decision has; whether the "
            "two are confusable and whether the frames show the order is Stage A and B."
        ),
        "summary": {
            "recordings_with_candidates": len({item.recording for item in found}),
            "candidates": len(found),
            "by_participant": dict(sorted(by_participant.items())),
        },
        "candidates": [
            {
                "recording": item.recording,
                "kind": item.kind,
                "window_seconds": [round(value, 2) for value in item.window],
                "objects": [
                    {
                        "name": handled.name,
                        "handled": [round(handled.start, 2), round(handled.end, 2)],
                    }
                    for handled in item.objects
                ],
            }
            for item in found
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=REPORT)
    parser.add_argument("--participant", help="only this participant, e.g. P01")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    report = mine(root)
    if args.participant:
        report["candidates"] = [
            row
            for row in report["candidates"]
            if row["recording"].startswith(args.participant)
        ]
    (root / args.output).parent.mkdir(parents=True, exist_ok=True)
    (root / args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
