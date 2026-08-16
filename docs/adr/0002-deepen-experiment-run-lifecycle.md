# ADR-0002: Put Experiment Run lifecycle behind one Interface

- Status: Accepted
- Date: 2026-08-06

## Context

The minimal multimodal Adapter and Browser-Use Adapter independently constructed `RunSession`, `WebState`, the local site server and `RunLedger`. They also duplicated event bookkeeping, packet reveal, provenance assembly, duration recording, success finalization and infrastructure-failure handling.

This duplication was scientifically risky: a new Adapter could produce a different Run Artifact or deliver the same intervention with different state transitions. The cluster had two real Adapters, so the Seam was not hypothetical.

## Decision

Introduce an `execution` Module with this Interface:

- `RunConfig`: immutable identity, budget, path and randomness inputs;
- `RunContext`: initial Released Observation, local EgoMart URL, current step, `advance`, `ask_user` and partial-trace capture;
- `run_experiment` / `run_experiment_async`: synchronous and asynchronous Adapter entry points.

The Module owns Experiment Run state, local site lifetime, event recording and reveal, metadata/provenance construction, partial failure records and Run Artifact finalization. Agent Adapters own only provider-specific representations and actions.

Dependencies inside the Module are in-process (`RunSession`, `WebState`) or local-substitutable (`EgoMartServer`, filesystem Run Artifact). True external model and browser dependencies remain outside the Seam in their Adapters.

## Consequences

- All Agent Adapters now produce the same success and failure record shape.
- Event delivery and visible-packet state change in one place.
- A new Agent Adapter crosses one small Interface instead of reimplementing the benchmark lifecycle.
- `execution` is tested directly and through both existing Adapters.
- Sync and async entry points remain separate because the external execution models genuinely differ; their shared implementation stays private.
