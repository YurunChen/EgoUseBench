# ADR-0001: Separate benchmark definitions from regenerable artifacts

- Status: Superseded by ADR-0006
- Date: 2026-08-06

## Context

The original `data/pilot` directory mixed mining output, mutable task construction, Released Observation definitions, generated frames, provenance, and candidate manifests. Agent adapters also wrote a caller-chosen result JSON only after successful completion. That layout made lifecycle state hard to read and left provider or harness failures without evidence.

The local Ego2Web checkout is about 11 GB and contains its own Git repository. Moving or copying it would add risk without improving the benchmark's scientific interface.

## Decision

Use these top-level responsibilities:

- `data/workbench`: mutable mining, screening, construction, and review records;
- `data/releases`: immutable Benchmark Releases admitted through the Construct-Validity Gate;
- `data/catalogs`: frozen catalog definitions and their source provenance;
- `data/observations`: provenance-bound frames used by tasks or retained candidates;
- `artifacts/reports/source-evidence`: retained visual review material;
- `artifacts/runs`: one self-contained Run Artifact per Experiment Run;
- `artifacts/reports`: checked results, aggregate metrics, confidence intervals, and failure summaries;
- `reference`: read-only third-party checkouts and datasets;
- `configs`: committed experiment plans once an Experiment Runner consumes them.

All task asset paths remain workspace-relative and must resolve inside the declared asset root. Absolute paths and `..` escapes are invalid. The local Ego2Web checkout remains at `reference/Ego2Web` and is ignored by the first-party repository.

Every CLI Agent run creates a Run Artifact even on infrastructure failure. A Run Artifact uses atomic JSON writes and contains `manifest.json`, `runtime.jsonl`, `events.jsonl`, `trace.json`, `run.json`, plus `error.json` when applicable.

## Consequences

- A benchmark definition can be versioned without checking in videos, extracted frames, model traces, or secrets.
- A failure can be reconstructed from one directory instead of terminal scrollback.
- Validator, web environment, and Agent adapters share one path-resolution Module.
- Existing scripts and documentation must use the new workbench and artifact paths.
- The project still needs a root Git repository and dependency lock before a public Benchmark Release; this ADR records the requirement but does not initialize Git implicitly.
