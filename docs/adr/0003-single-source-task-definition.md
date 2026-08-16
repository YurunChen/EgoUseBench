# ADR 0003: Keep one authored task definition and derive execution views

## Status

Superseded by ADR-0004, 2026-08-09.

## Context

EgoCUA-Bench needs more information than Ego2Web's compact task rows because Graft,
Refresh and Clarify isolate different interventions and use deterministic entity-level
evaluation. The candidate manifest had nevertheless begun to repeat the same fact in
multiple places: condition instructions were copied into `task_claims`, and final
targets were copied from `expectations` into Refresh and Clarify track blocks.

Those copies could diverge during human annotation. Creating separate editable public
and private task manifests would increase the same risk.

## Decision

The schema-v2 synthesis record is authoritative while a sample is being mined and
screened. Once the sample is adapted into the executable project schema, that dossier
is frozen and referenced; the project-native manifest becomes the only editable
executable task definition.

- Condition instructions are derived from `instructions` by the protocol.
- Accepted final states are stored only in `expectations`.
- Refresh retains only `initial_target_id`; its final target is derived from the shared
  condition expectation.
- Clarify retains the candidate set and answer policy; its hidden target is derived
  from the shared condition expectation.
- Agent-visible payloads and checker-facing evaluator details remain runtime-derived
  views and must not be maintained as independent task data.
- Source evidence, the frozen schema-v2 construction dossier, frame provenance, human
  review and run artifacts remain separate audit records. They have different ownership
  and are not alternative executable task definitions.

The schema rejects the removed derived fields so old or manually edited tasks cannot
silently reintroduce a second source of truth.

## Consequences

Task edits require fewer synchronized changes, while deterministic validation still
checks that Refresh policy matches whether the target actually changes and that a
Clarify target belongs to its candidate set. The design keeps necessary research
entities but does not add a generic event hierarchy, global product service or a second
editable release schema.
