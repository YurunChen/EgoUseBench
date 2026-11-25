"""Canonical paths and schema versions for task-synthesis reports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportContract:
    path: Path
    schema_version: int


REPORTS = {
    "acceptance": ReportContract(Path("artifacts/reports/task-acceptance.json"), 3),
    "smoke": ReportContract(Path("artifacts/reports/task-smoke.json"), 7),
    "role": ReportContract(Path("artifacts/reports/role-recoverability.json"), 3),
    "assessment": ReportContract(
        Path("artifacts/reports/coding-agent-task-assessments.json"), 4
    ),
    "confidence": ReportContract(Path("artifacts/reports/task-confidence.json"), 4),
}
