# EgoCUA-Bench Design Validity Audit

Date: 2026-08-10  
Auditor: Codex research pass  
Scope: design-validity evidence audit for the current EgoCUA-Bench scientific plan, using local project evidence plus only primary external sources (papers, official project pages, official code/dataset pages).

## Executive Summary

The current EgoCUA-Bench proposal has a defensible **diagnostic core**, but only if it is explicitly framed as a **project-defined, packetized, supplied-update benchmark** rather than a field-standard theory of wearable computer-use. The strongest external support is for: frozen executable environments with final-state checking, matched interrupted-vs-uninterrupted comparisons, explicit uncertainty reporting, and human verification before release. The weakest support is for: treating `Graft / Refresh / Clarify` as a mature community ontology, claiming natural conversational clarification from the current slot-typed tool, and claiming streaming event-detection competence from preselected image packets.

The biggest blocker is not abstract methodology but benchmark state. The obsolete set of 30 placeholder tasks has been removed rather than grandfathered into v1. After reviewing the exact images visible to the Agent and expanding acquisition across Ego2Web, HD-EPIC, and EgoSchema/Ego4D, the workbench now contains 61 `video_screened` candidates: 60 frozen primaries and 1 reserve, plus 32 explicit rejections. The provenance sidecars verify 504 extracted frames; 488 belong to the current screened pool and 16 are retained as evidence for tasks rejected after construction. All 60 primaries now compile to task/checker artifacts and reference one fixed 705-listing EgoMart site whose normalized names and image hashes are unique. Mechanical completeness still does not establish construct validity: visually or semantically similar listings may still create search collisions that exact deduplication cannot detect. None has passed two isolated construction reviews, fixed-site search review, human execution, evaluator replay, or the formal Browser-Use model gate, so there is still no release task. Candidates sharing one source-video cluster cannot be counted as independent samples without source-clustered analysis.

## Audit Inputs

Local evidence reviewed:

- [BENCHMARK_DESIGN](../BENCHMARK_DESIGN.md)
- [SCIENTIFIC_PROTOCOL_REVIEW](../SCIENTIFIC_PROTOCOL_REVIEW.md)
- [CONTEXT](../../CONTEXT.md)
- [protocol.py](../../egocua/protocol.py)
- [checker.py](../../egocua/checker.py)
- [metrics.py](../../egocua/metrics.py)
- [schema.py](../../egocua/schema.py)
- [webapp.py](../../sites/egomart/app.py)
- [environment.py](../../egocua/environment.py)
- [gates.py](../../egocua/gates.py)
- [minimal_loop.py](../../egocua/agents/minimal_loop.py)
- [configs/pilot.json](../../configs/pilot.json)
- [seed screening manifest](../../data/sources/seed_screening_manifest.json)
- [seed screening report](seed-screening-report.md)

Evidence tiers used in this audit:

| Tier | Meaning in this document | Status here |
|---|---|---|
| Mature consensus | Repeated across established computer-use / web-agent benchmarks | Final-state execution checks; frozen local environments; human verification before release; interval estimates instead of point estimates only |
| Recent method | Strong but recent 2025-2026 benchmark methodology | Dynamic interruption simulation; Ask-F1 style clarification calibration; PASS/FAIL/UNKNOWN evidence bounds; replay-based evaluator audits |
| Project custom assumption | Useful, but not externally established as community consensus | Three-track decomposition; packetized V0/V1 protocol; slot-typed one-turn clarification tool; 3 replicates per arm; 60-task pilot |

## Findings By Question

### 1. Are the three tracks clearly distinguished from prior work?

Verdict: **Yes as a project-defined diagnostic split; no as a mature field taxonomy.**

What external evidence supports:

- Existing work already separates neighboring failure modes rather than treating them as one construct:
  - **mid-task user intent updates** in InterruptBench [S5]
  - **latent wearable-goal inference** in WAGIBench [S6]
  - **egocentric-world-to-web execution under fixed task meaning** in Ego2Web [S7]
  - **streaming no-future state tracking** in Wearable-AI / EgoProactive and TeleEgo [S8][S9]
  - **underspecified egocentric reference** in EgoPoint-Bench and REI-Bench [S10][S11]
- That separation is enough to justify `Graft`, `Refresh`, and `Clarify` as operational benchmark families.

What external evidence does **not** support:

- No primary source treats `Graft / Refresh / Clarify` as a standard, exhaustive, or consensus ontology for wearable computer-use.
- InterruptBench’s `addition / revision / retraction` are **update operations**, not the same thing as EgoCUA’s `Graft / Refresh / Clarify`, nor the same thing as local `C / P / R` task subtypes [S5].

Implication:

- The triad is publishable only as **“our benchmark decomposition for diagnostic attribution”**.
- It should not be written as “the three source-complete classes of goal-belief divergence” without new empirical validation beyond neighboring benchmarks.

Severity: **release-blocking wording issue**

### 2. Do the three experiment arms support the claimed pairwise attribution?

Verdict: **Partly yes, but only for specific pairwise comparisons, and only after human task validation.**

#### Graft

What is supported:

- `bundled` vs `injected` is a valid matched comparison for online interruption cost **if** both correspond to the same final task meaning and B is necessary. This matches InterruptBench’s construction rule that the initial query plus interruption messages must preserve the original intent and each interruption must be essential [S5].

What is not supported:

- `a_only` is **not** the same-strength causal counterfactual for the A+B task. It is a diagnostic baseline for “can the agent do A at all?” and “does B cause A to be dropped?”

Local gap:

- The local schema correctly rejects Graft tasks whose `initial_goal` and final `goal` are identical, but that only enforces “B changes goal semantics” structurally, not that the resulting task is semantically natural or human-solvable [schema.py].
- The retained Graft-P seed remains `video_screened` with unresolved B-naturalness and asset checks; it has no constructor agreement or human validation [seed screening manifest].

#### Refresh

What is supported:

- `stream` vs `oracle_latest` can measure **online refresh cost** if both share the exact same final predicate.
- `stream` vs `stale` can measure the value of the late visual update **only on `replace` tasks where V0 alone is insufficient**.

What is not supported:

- `retain` and `replace` cannot be pooled into one undifferentiated average; they have opposite semantics.
- If `stale` can already solve a claimed `replace` task, the task is shortcut-prone and cannot support adaptation claims.

Local gap:

- The schema enforces that `retain` vs `replace` matches whether `initial_target_id` changes, which is good.
- The protocol does not yet prove by human testing that V1 is necessary in `replace`, or irrelevant in `retain`.

#### Clarify

What is supported:

- `interactive` vs `forced_execute` isolates the value of allowed clarification.
- `oracle` supports over-asking diagnostics, in the same spirit as HiL-Bench’s argument that execution-only success misses the “should have asked” dimension [S12].

What is not supported:

- The current design does **not** justify claims about open-ended natural clarification dialogue. The implemented task is narrower: “detect insufficiency, choose one of five allowed slot types, ask once, then use a deterministic answer.”

Local gap:

- `ask_user(slot, question)` exposes the slot inventory in the tool schema and returns a canonical answer for the one valid slot [protocol.py][minimal_loop.py].
- That is a reasonable **diagnostic simplification**, but it is not the same construct as free-form conversational clarification.

Bottom line:

- Supported causal/paired comparisons are:
  - `bundled` vs `injected`
  - `oracle_latest` vs `stream`
  - `stream` vs `stale` on `replace`
  - `interactive` vs `forced_execute`
- Diagnostic baselines are:
  - `a_only`
  - `oracle`
  - `stale` on `retain`

Severity: **release-blocking**

### 3. Do image packets, dynamic events, and the clarification tool introduce shortcuts or weaken external validity?

Verdict: **Yes. They are acceptable only under a narrowed scope statement.**

#### Image packets

What is methodologically defensible:

- Wearable-AI and TeleEgo both emphasize strict no-future protocols and progressive release of information [S8][S9].
- The local runtime correctly withholds future packets until release and exposes only frame paths, not packet IDs or hidden labels [protocol.py].

What weakens external validity:

- Ego2Web’s contribution is grounded world-to-web execution; TeleEgo and EgoProactive study continuous streaming; neither treats preselected packet release as equivalent to natural event discovery [S7][S8][S9].
- Therefore EgoCUA’s current packet design measures **reaction to supplied updated evidence**, not **continuous first-person monitoring** or **self-triggered event detection**.

Required wording:

- “v1 tests adaptation after benchmark-supplied observation release.”
- Not: “v1 tests online wearable video understanding.”

#### Dynamic event delivery

What is supported:

- InterruptBench injects updates at dynamically determined execution points rather than only at a fixed turn, which supports the general idea of milestone-based delivery [S5].

What still risks shortcutting:

- A fixed fallback step and predictable milestone policy may incentivize waiting strategies or benchmark-specific timing heuristics.
- This is not fatal, but it means reported effects should be interpreted as protocol-relative unless timing sensitivity is tested.

#### Clarification tool

What is supported:

- HiL-Bench supports the idea that asking behavior needs a calibrated metric, not just end-task success [S12].
- REI-Bench and EgoPoint-Bench support the claim that vague or misleading referents are a real, separable problem in egocentric settings [S10][S11].

What weakens external validity:

- The current tool turns clarification into a constrained one-turn slot-selection problem.
- The assistant knows up front that only `position | appearance | brand | variant | sequence` are valid; real users do not expose the ontology this way.
- A canonical answer is much cleaner than human answers in actual wearable interaction.

Most important local gate:

- Runtime placeholder SVG fallback has been removed; a missing product image now fails instead of silently fabricating one [webapp.py].
- Release validation requires opaque IDs, independent product-image provenance and verified hashes, and forbids reuse of released video frames [environment.py].
- No current seed has passed that asset gate, so the project still cannot claim “real-world object grounding to product selection.”

Severity:

- Packetized-update scope mismatch: **release-blocking**
- Slot-typed clarification mismatch: **release-blocking**
- Missing release catalogs/images: **fatal until constructed**

### 4. Do Ground Truth, human validation, checker design, and PASS/FAIL/UNKNOWN match computer-use benchmark practice?

Verdict: **Mostly aligned at the headline level; not yet aligned enough for process claims or release-quality scoring.**

What is strongly aligned with mature consensus:

- WebArena focuses on functional correctness in a reproducible local environment [S1].
- VisualWebArena extends WebArena’s execution-based functional evaluation to visually grounded tasks [S2].
- OSWorld uses initial-state setup plus custom execution-based evaluation scripts for reliable and reproducible scoring [S3].
- AndroidWorld derives durable reward signals from system state rather than brittle surface text [S4].

This supports the current choice to use:

- frozen local environment
- hidden goal truth
- final-state checking by exact acceptable item multiset plus submission status

What is supported only by recent methodology:

- Evidence-supported `PASS / FAIL / UNKNOWN` with lower/upper score bounds comes from recent benchmarking methodology, not long-settled consensus [S14].
- Replay-based evaluator audits and explicit false-negative / broken-task accounting are recent but highly relevant for computer-use benchmarks [S15].

Current local strengths:

- `run_valid`, `goal_state_correct`, `submission_completed`, and `evidence_status` are structurally separated [checker.py].
- `UNKNOWN` is not counted as plain failure [metrics.py].
- Release gates require two independent construction reviews, two human Run Artifacts per arm, visual-necessity and track-specific counterfactual checks, and an independent evaluator reviewer. Information-sufficient arms must succeed; intentionally underinformed controls must terminate validly without answer leakage [gates.py].

Current local weaknesses:

- `policy_pass` is currently hard-coded `True`, so the benchmark cannot yet claim validated process constraints broadly [checker.py].
- Clarify timing and answer utilization have now been tightened to the first Candidate-Committing Action and first post-answer candidate-specific action [checker.py][metrics.py]. Their semantics still require evaluator replay before release.
- The storefront still shows a quantity selector, but submit de-duplicates `selected_item_ids`; duplicate-quantity tasks remain unsupported by actual final-state semantics [webapp.py].
- Seed release gates now replay referenced human, model, and constructed evaluator artifacts. The later full-pilot pipeline for drawing and tracking the preregistered stratified PASS/FAIL audit sample is not implemented yet.

Bottom line:

- Final-state exact-ID checking is on-consensus for shopping-style tasks.
- The Clarify process predicates now match the registered definitions, but no empirical process result should be reported before replay audit and real runs.

Severity: **release-blocking**

### 5. What conclusions do the metrics, 3 replicates, task-level paired bootstrap, and 60-task pilot support?

Verdict: **Reasonable pilot statistics; limited claim horizon.**

What is well-supported:

- RLiable argues that few runs per task create large uncertainty and that interval estimates should accompany aggregate scores [S13].
- The local metric code aggregates over replicates inside each `task × arm`, then computes paired arm differences at task level, then bootstraps across task strata [metrics.py]. That is methodologically sound for a small pilot.

What 3 replicates can support:

- within-task stochasticity estimation
- pass@1 and pass^3 style reliability summaries
- rough task-level arm differences
- whether effects are directionally consistent enough to justify a larger release

What 3 replicates cannot support:

- replicate-level causal pairing unless the provider exposes and matches true stochastic seeds across arms
- precise claims about small between-model differences
- stable leaderboard ordering

What a 60-task pilot can support:

- protocol debugging
- construct-validity red-teaming
- leakage discovery
- floor/ceiling detection
- variance estimation for later sample-size planning
- identifying which track/subtype needs redesign

What a 60-task pilot cannot support:

- strong benchmark-wide difficulty claims
- robust model ranking
- publication-strength generalization claims across wearable computer-use

Why this boundary matters:

- Neighboring primary sources are much larger:
  - VisualWebArena: 910 tasks [S2]
  - OSWorld: 369 tasks [S3]
  - AndroidWorld: 116 tasks with dynamic instantiation [S4]
  - WAGIBench: 3,477 recordings [S6]
  - TeleEgo: 3,291 QA pairs [S9]
  - EgoPoint-Bench: 11,729 QA samples [S10]

Severity: **release-blocking if the paper overclaims; otherwise acceptable as pilot methodology**

### 6. Which design decisions still must be resolved before prototype claims?

Verdict: **Several are still unresolved enough to block any serious prototype claim set.**

Required before prototype conclusions:

These decisions were resolved in [ADR-0005](../adr/0005-scope-v1-as-a-packetized-diagnostic-pilot.md): v1 is packetized, Clarify is one-turn and slot-typed, Graft is addition-like, duplicate quantities are excluded, Browser-Use + frozen GPT-5.4-mini is primary, and a complete three-track seed gate precedes scaling. All eight planned subtypes now have at least one `video_screened` candidate, but this is only structural candidate coverage. The remaining work is blinded input review followed by the asset, human, evaluator, leakage, and real-model gates.

## Issue Register

### Fatal

| ID | Issue | Why it is fatal | Required action |
|---|---|---|---|
| F1 | Sixty compiled candidates share a real fixed site, but none has independent-human approval or fixed-site search-collision review | The benchmark cannot yet support prototype claims about real-world product grounding | Review each task against the actual fixed-site search results, then independently approve and gate a representative seed before full execution |

### Release-Blocking

| ID | Issue | Why it blocks release-quality claims | Required action |
|---|---|---|---|
| R1 | `Graft / Refresh / Clarify` is written too close to an external ontology | No primary source recognizes this triad as community consensus | Reframe as project-defined diagnostic decomposition |
| R2 | Some arm interpretations are too strong | `a_only`, `oracle`, and some `stale` uses are diagnostic baselines, not same-strength causal controls | Restrict causal language to the validated pairwise comparisons only |
| R3 | B/V1/ambiguity necessity is not human-verified per task | Structural schema checks are not enough for construct validity | Add independent task-level validation for essentiality and ambiguity balance |
| R4 | Clarify process semantics require audit | The checker now implements Candidate-Committing Action timing and direct answer use, but has no real replay evidence | Replay constructed positive and adversarial traces before release |
| R5 | Clarify tool narrows the construct more than current prose admits | Slot-typed one-turn escalation is not natural multi-turn clarification | Narrow the claim or redesign the interface |
| R6 | Packet design can be over-read as streaming capability | Preselected packets remove event-discovery burden | Narrow the claim or redesign toward continuous release |
| R7 | Quantity semantics are not fully aligned | UI exposes quantity, but submission state is deduplicated | Freeze “no duplicate-quantity tasks” or change submission semantics |
| R8 | No sampled replay audit of PASS and FAIL | Recent evidence shows evaluator errors materially distort CUA scores [S15] | Add replay audit before release |

### Diagnostic Improvement

| ID | Improvement | Value |
|---|---|---|
| D1 | Timing sensitivity for event injection | Shows whether results are robust beyond the chosen milestone/fallback policy |
| D2 | Post-update `SR(k)` and censored latency plots by subtype | Better reflects online response than success-conditioned mean latency |
| D3 | Separate grounding probe if perception claims are desired | Avoids over-interpreting downstream E2E outcomes as pure perception |
| D4 | Explicit leak audit over filenames, frame order, alt text, and packet metadata | Reduces hidden benchmark shortcuts |

### Future Extension

| ID | Extension | Not needed for v1 if scope is narrowed |
|---|---|---|
| X1 | Continuous video / event discovery | Current packet protocol is enough for supplied-update adaptation |
| X2 | Natural-language multi-turn clarification | Current slot-typed one-turn tool is enough for a narrow diagnostic v1 |
| X3 | Combined interventions and audio/multi-party context | Valuable later, unnecessary for first construct-valid pilot |

## Recommended Claim Boundary For v1

If the project continues without redesign, the strongest defensible claim is:

> EgoCUA-Bench v1 is a frozen, shopping-style, packetized computer-use benchmark that diagnoses whether an agent can adapt web behavior when the benchmark releases new user constraints, new egocentric evidence, or one constrained clarification opportunity during execution.

The following stronger claims are **not** yet defensible:

- “EgoCUA measures the three canonical sources of goal-belief divergence.”
- “EgoCUA measures continuous egocentric streaming adaptation.”
- “EgoCUA measures general conversational clarification ability.”
- “The current 60-task target supports stable model ranking or benchmark-wide difficulty conclusions.”

## First-Round Decisions For The Project Owner

1. Should v1 stay **packetized and narrow** or be redesigned toward **continuous streaming**?
2. Should Clarify v1 be kept as **slot-typed one-turn escalation**, or upgraded to **natural clarification dialogue**?
3. Is Graft v1 only about **addition-like integration**, or do you want explicit **revision/retraction** tasks now?
4. Will prototype claims wait for a **release-quality seed set** with real images and double human validation, or do you intend to show results on candidate tasks?  
Recommendation: wait for the seed set.

## Sources and References

Primary external sources used in this audit:

S1. WebArena paper: https://arxiv.org/abs/2307.13854
S2. VisualWebArena official project page: https://jykoh.com/vwa
S3. OSWorld paper: https://arxiv.org/abs/2404.07972
S4. AndroidWorld official project page: https://google-research.github.io/android_world/
S5. InterruptBench paper: https://arxiv.org/html/2604.00892v1
S6. WAGIBench official project page: https://facebookresearch.github.io/WAGIBench/
S7. Ego2Web official project page: https://ego2web.github.io/
S8. Wearable-AI official dataset card: https://huggingface.co/datasets/facebook/wearable-ai
S9. TeleEgo official project page: https://teleai-uagi.github.io/TeleEgo/
S10. EgoPoint-Bench official project page: https://guyyyug.github.io/EgoPoint-Bench/
S11. REI-Bench paper / project-linked primary source: https://arxiv.org/html/2505.10872v3
S12. HiL-Bench paper: https://arxiv.org/abs/2604.09408
S13. RLiable paper: https://arxiv.org/abs/2108.13264
S14. Evidence-Supported Benchmarking paper: https://arxiv.org/abs/2605.10448
S15. How Benchmarks Mis-Score Computer-Use Agents: https://arxiv.org/abs/2607.28367

Source-role map:

- Mature consensus anchors: [S1], [S2], [S3], [S4], [S13]
- Recent method anchors: [S5], [S12], [S14], [S15]
- Egocentric/wearable construct neighbors: [S6], [S7], [S8], [S9], [S10], [S11]
