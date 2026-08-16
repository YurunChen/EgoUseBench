# ADR-0004: Separate tasks, experiment arms, environments, and validation evidence

## Status

Superseded by ADR-0006, 2026-08-10.

## Context

The executable candidate rows mixed four independent responsibilities. `track` and
`conditions` both appeared to classify tasks; each task copied a 16-item website catalog;
the same final goal was copied once per condition; and mutable construction, human, and
model review records were embedded in the task. A review update therefore changed the
executable task identity, and changing an experiment matrix required editing every task.

Graft, Refresh, and Clarify are task families. `a_only`, `stream`, and `interactive` are
controlled ways of running those tasks. They are orthogonal concepts.

## Decision

- Store tasks under `tasks/graft`, `tasks/refresh`, and `tasks/clarify`. The directory and
  `track` must agree.
- A task stores its natural instructions, Released Observations, track-specific state, and
  one canonical final `goal`. Graft also stores `initial_goal` because the user's goal
  genuinely changes after B; this is task truth, not an experiment-arm copy.
- Store enabled arms and replicate IDs once in an Experiment Plan. Runtime records call
  this field `arm`.
- Store deterministic EgoMart products in catalog files. A task keeps only the catalog
  path under `environment`.
- Store construction review, independent-human results, and real-model run evidence in
  one external validation record keyed by task ID. The Construct-Validity Gate joins this
  evidence with the immutable task and catalog.
- The Checker derives the accepted outcome for an arm from `goal` and, only for Graft's
  `a_only` arm, `initial_goal`. No per-arm expectation table is authored.

## Consequences

Task definitions can be reviewed without also editing the experimental matrix, website
snapshot, or gate history. The 60 pilot tasks have one physical classification and expand
to 540 Experiment Runs through `configs/pilot.json`. Loading and release validation now
join several files deliberately, but each fact has one owner and each boundary is checked.
