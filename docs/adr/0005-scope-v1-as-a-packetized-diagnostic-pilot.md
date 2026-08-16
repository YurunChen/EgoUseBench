# ADR-0005: Scope v1 as a packetized diagnostic pilot

## Status

Accepted, 2026-08-09.

## Context

EgoCUA-Bench had implemented three task families, three arms per family, a dynamic-event protocol, a frozen shopping environment, deterministic final-state checking, and task-level metrics. The implementation alone did not settle what the pilot could scientifically claim. In particular:

- preselected V0/V1 image packets do not test continuous-video event discovery;
- one slot-typed clarification turn does not test general conversation;
- the current Graft candidates mostly add or integrate information rather than revise or retract an earlier goal;
- not every arm pair has the same causal interpretation;
- 30 candidate tasks with placeholder catalogs cannot support release-quality grounding claims;
- one task per family can validate the closed loop but cannot validate every subtype.

A primary-source design-validity audit is recorded in
`docs/research/benchmark-design-validity-audit.md`. The project owner then confirmed the decisions below through four rounds of design review.

## Decision

### Claim boundary

EgoCUA-Bench v1 is a construct- and protocol-validation pilot for a project-defined diagnostic decomposition. It evaluates web-action adaptation after a Benchmark-supplied update. It does not claim:

- a consensus or exhaustive taxonomy of wearable-agent failures;
- continuous egocentric-video monitoring or autonomous event detection;
- general multi-turn conversational clarification;
- full addition/revision/retraction coverage;
- stable model ranking or broad wearable-agent generalization from the 60-task pilot.

Graft v1 studies addition-like integration. `C`, `P`, and `R` remain Constraint, Parallel, and Relational goal-combination relations, not update operations. Refresh retains the separate `retain` and `replace` constructs. Clarify v1 is a one-turn, slot-typed, deterministic escalation diagnostic.

### Arm roles and primary estimands

Arms are not three equally strong conditions.

- Graft headline contrast: `bundled - injected`. `a_only` is a Diagnostic Baseline.
- Refresh-replace headline benefit: `stream - stale`; online cost: `oracle_latest - stream`.
- Refresh-retain reports false update and robustness separately; it is not pooled into a generic Refresh Gain.
- Clarify headline contrast: `interactive - forced_execute`. `oracle` measures unnecessary asking.

Task-level paired effects are computed after aggregating replicates within each task and arm. Provider seed labels are not interpreted as replicate-level causal pairing unless the provider guarantees them.

### Observation and event protocol

v1 keeps packetized Released Observations. The claim is reaction to supplied evidence, not event discovery. Graft and Refresh deliver the event after the first successful search-result milestone. If it has not occurred by step 8, the event is forced; `trigger_reason` and fallback frequency are reported rather than silently discarding slower runs.

Future packets must be inaccessible through every production read path until release. The leakage audit covers prompts, tool schemas, URLs, filenames, alt text, context pages, HTML, error messages, product IDs, catalog/search ordering, and hidden metadata.

### Target and environment semantics

Diagnostic Ground Truth follows source-supported granularity. Exact item identity is required only when video evidence or the permitted clarification uniquely supports it; otherwise `acceptable_sets` retains every catalog entity consistent with the evidence.

Product IDs and ordering cannot encode target, initial/updated, candidate, subtype, or distractor roles. IDs are stable and opaque; catalog order is deterministically randomized and frozen. Website product images are independent assets with source, access/license note, URL or retrieval method, retrieval time, and file hash. Decisive video frames are not reused as website product images. v1 excludes duplicate-quantity tasks.

### Track admission contracts

Graft tasks require natural A and B instructions, an essential B, identical combined goals for `bundled` and `injected`, and a mechanically auditable C/P/R relation. Refresh requires continuous evidence for the V0-to-V1 sequence; replace requires V0 to be insufficient for the latest target and V0+V1 to be sufficient, while retain requires V0 to remain sufficient. Clarify v1 uses exactly two plausible real-world candidates; the ambiguous input cannot identify one, and one canonical natural-attribute answer must identify exactly one.

Clarify asking is timely only before the first Candidate-Committing Action, not merely before submission. Answer utilization requires the first candidate-specific action after a valid answer to target the hidden target. Later recovery can preserve E2E success while failing the process diagnostic.

### Seed and release gates

The first closed-loop prototype contains three Release-Quality Seeds chosen before model runs:

- one Graft-P seed;
- one Refresh-replace seed;
- one Clarify-variant seed.

Selection uses source-evidence quality, naturalness, and asset feasibility, never evaluated-model success. These three tasks validate the closed loop only. Before claiming subtype coverage, expand to eight seeds: Graft C/P/R, Refresh retain/replace, and Clarify variant/sequence/position.

Every seed requires two agreeing independent constructors, two independent human Run Artifacts per arm using the exact harness, and an independent evaluator reviewer. Information-sufficient arms must succeed; deliberately underinformed controls must terminate validly without leaked answers and are not required to guess correctly. Construction disagreement removes the seed from the pilot. The reviewer replays all seed runs plus constructed PASS, FAIL, UNKNOWN, early-event, missing-event, invalid-question, extra-item, and duplicate-quantity cases. At full pilot scale, audit every UNKNOWN and a track/arm-stratified random sample of at least 10% PASS and 10% FAIL.

Browser-Use with a frozen GPT-5.4-mini snapshot is the primary production Agent stack. The minimal loop is a protocol/API diagnostic and cannot substitute for production model validation. Every arm must produce at least one protocol-valid real-model Run Artifact; model success is not a task admission criterion. An information-complete arm that fails all three replicates triggers a floor-effect diagnosis before scaling, not silent task replacement.

### Frozen 60-task allocation

The complete v1 pilot contains 60 independent base tasks. This scale and allocation are a project choice, not a community-standard sample-size threshold:

- Graft: 30 tasks, with 10 each for C, P, and R;
- Refresh: 16 tasks, with 8 each for retain and replace;
- Clarify: 14 tasks, with 6 variant, 4 sequence, and 4 position tasks.

This doubles the original 15/8/7 pilot proportions while preserving all subtype ratios, so the expansion does not silently reweight the research question. A source video or underlying long-video cluster contributes at most one independent task to the primary analysis. Additional valid constructions from the same cluster may be retained only as reserves, or analyzed with the source cluster as the resampling unit. The target scale is a construction objective, not permission to weaken source evidence, ambiguity, asset, human, evaluator, or real-model gates.

### Statistics and run identity

The pilot reports predeclared task-level effects, task-clustered bootstrap intervals, valid-run rate, evidence bounds, and three-run reliability distributions. It does not use significance tests or produce a cross-track score or stable leaderboard.

Task/arm order is deterministically randomized by a committed scheduler seed. Each run starts a fresh cart, browser context, and model conversation. The actual order and wall-clock time are retained for provider-drift analysis.

Before formal evaluation, establish a root Git history and freeze task, catalog, asset, Experiment Plan, prompt, Adapter, checker, and metric identities. A semantic edit creates a new version and repeats affected construction, human, evaluator, and model gates. Old Run Artifacts remain bound to their original hashes.

## Consequences

- The obsolete 30-task placeholder set has been removed and cannot be relabeled as a release; the current release target is the gated 60-task allocation above.
- Current four-candidate Clarify data must be reconstructed to the two-candidate v1 policy; this is a project decision that supersedes the earlier pilot assumption.
- Current positional product IDs, placeholder catalogs, Clarify submit-only timing, and answer-utilization approximation block release-quality seed approval.
- The three-seed prototype begins only after continuous source review and independent website-asset construction.
- Negative screening and model failures remain reportable results; they do not justify inventing evidence or selecting easier tasks.
