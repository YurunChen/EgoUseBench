# ADR-0006: Extend the three diagnostic tracks beyond commerce

## Status

Accepted, 2026-08-10.

## Context

The current 60 candidates isolate Graft, Refresh, and Clarify in one frozen commerce
workflow. That controls the environment, but it risks measuring adaptation only when every
task ends in a shopping cart. Ego2Web shows that the same egocentric evidence can ground
knowledge lookup, media retrieval, maps, and community-search tasks in addition to
e-commerce.

Changing a page theme is not a meaningful extension. A new workflow is useful only when it
changes what the Agent must inspect and which final state the evaluator checks. It must not
replace deterministic state checking with an unconstrained LLM judge.

## Decision

Keep the benchmark at 60 primary tasks. Replace admitted commerce candidates with
cross-domain tasks rather than adding an unbounded second collection. Admit one seed per
track for a workflow before assigning that workflow a quota.

Prioritize workflows in this order:

1. **Knowledge**: search a frozen document collection, distinguish near-neighbor records,
   open the relevant section, save the source, and submit exact fields.
2. **Media**: search frozen video metadata and transcripts, inspect the correct tutorial or
   scene, save it to a list, and submit exact metadata or transcript-backed fields.
3. **Maps**: distinguish place branches, inspect a listing, configure a route, save the
   destination, and submit exact place and route state.

Knowledge is the first implementation target because three already-reviewed input
sequences can be linked to exact source records. Media and maps remain source-mining targets;
they are not to be invented around unrelated frames.

The runtime should eventually use one normalized exact state across sites:

```json
{
  "selected_entity_ids": ["opaque-id"],
  "fields": {"field_name": "exact frozen value"},
  "submission_completed": true
}
```

The environment owns documents, products, videos, and places. A task owns only the accepted
final states. When the first cross-domain seed is implemented, migrate the current
cart-specific schema to this contract; do not maintain two task schemas.

## Track-specific extensions

| Track | Knowledge example | Media example | Maps example |
|---|---|---|---|
| Graft | A later request adds an exact fact, source section, or second reference | A later request adds a transcript, duration, or tutorial constraint | A later request adds a travel-mode or route constraint |
| Refresh | A later view changes or preserves which article is relevant | A later object/action changes or preserves the matching tutorial | A later view changes or preserves the intended branch or landmark |
| Clarify | Two visible objects map to different records | Two activities map to different videos | Two visible places or branches require one distinguishing answer |

The three experiment arms remain properties of a task, not separate task categories. The
cross-domain workflow changes the downstream state machine while preserving the same
Graft, Refresh, or Clarify intervention.

## Difficulty policy

Increase difficulty only through registered, construct-relevant operations:

1. more plausible real-world referents;
2. more near-neighbor search results;
3. decisive evidence hidden on a detail page, section, transcript, or route option;
4. multiple required state changes such as save, fill fields, and submit;
5. a real correction after Graft or Refresh arrives;
6. an irreversible candidate-specific action that Clarify should delay.

Do not add CAPTCHA, unrelated websites, arbitrary long text, or extra clicks merely to raise
step count. Those additions increase nuisance variance rather than the target construct.

## Admission gate

A cross-domain candidate remains outside the primary 60 until all of the following pass:

1. continuous-video review supports its referent or transition;
2. target resources and hard distractors are independently linkable to the visual evidence;
3. every requested field has an exact frozen value and cited source;
4. ignoring a required update, overreacting to a retain event, or committing before a
   clarification changes a checked entity, field, or policy outcome;
5. two humans complete every arm in the exact site workflow;
6. alternative paths and wrong fields fail deterministic evaluator replay;
7. every arm has a protocol-valid Browser-Use run.

## Consequences

- The benchmark gains workflow breadth without pooling unlike domains into one opaque score.
- The current 60 commerce candidates remain the executable baseline until replacements pass
  the same gates; the mined cross-domain queue is not an increased task count.
- Results must be reported by track and workflow, or stratified by both.
- The first implementation replaces one Graft-C, one Refresh-retain, and one
  Clarify-sequence commerce candidate with knowledge-workflow versions built from the same
  reviewed observations.
