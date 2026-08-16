# Cross-domain extension audit

## Outcome

The current EgoCUA-Bench source pool can support a cross-domain extension without
recording new video. Three knowledge-workflow seeds now pass visual-to-resource screening,
but they are not yet constructed or admitted to the primary 60.

Reproducible mining found 28 non-commerce Ego2Web annotations attached to videos already
used by 21 of the current 60 tasks:

| Workflow candidate | Annotations |
|---|---:|
| Media retrieval | 17 |
| Knowledge lookup | 10 |
| Community search | 1 |
| Maps | 0 |

| Current track | Annotations |
|---|---:|
| Graft | 17 |
| Refresh | 5 |
| Clarify | 6 |

These are high-recall candidates, not completed tasks. The original annotations are kept
verbatim in `data/sources/cross_domain_extension_candidate_queue.json`; keyword signals only
describe requested operations and do not determine admission.

## Why knowledge is the first seed workflow

Knowledge tasks introduce a genuinely different operation from shopping while retaining an
exact evaluator. An Agent must identify the real-world referent, find the correct frozen
article, inspect the relevant section, preserve the source selection, and submit structured
fields. Wrong articles, missing fields, wrong values, and extra sources can all be rejected
without an LLM judge.

Media retrieval is the next candidate because it has the largest mined yield. It requires
frozen video metadata and transcript-backed fields before it can be deterministic. Maps has
no same-video candidate among the current 60 tasks, so it should be mined from the full
Ego2Web pool rather than invented around an unrelated current frame.

## Selected screening seeds

The first knowledge workflow uses one existing, continuously reviewed source record per
track:

| Track | Source task | Proposed workflow |
|---|---|---|
| Graft | `seed-graft-c-01` | Save the game article; a later request adds exact piece-count and section fields |
| Refresh | `candidate-refresh-retain-ego2web-08` | Keep the article for the first ingredient after a later packet appears |
| Clarify | `seed-clarify-sequence-poster-01` | Ask which of two posters, then submit the selected record's creator and year |

Their current decision and blockers are recorded once in
`data/sources/cross_domain_extension_seed_selection.json`; exact source facts are recorded
in `data/sources/cross_domain_resource_audit.json`.

- The Jenga sequence maps to the Jenga article, whose Rules section gives a standard set as
  54 blocks ([source](https://en.wikipedia.org/wiki/Jenga)).
- The first ingredient maps to baking powder, described as a dry chemical leavening agent;
  the later yellow Foster Clark's Corn Flour packet maps to corn starch, a thickening agent
  ([baking powder](https://en.wikipedia.org/wiki/Baking_powder),
  [corn starch](https://en.wikipedia.org/wiki/Corn_starch)).
- The second poster maps to the Library of Congress record for *I want you for U.S. Army*,
  created by James Montgomery Flagg around 1917
  ([record](https://www.loc.gov/item/96507165/)).

This resolves the main evidence-linkage risk. A 12-document seed collection containing the
targets and hard near-neighbors is frozen in
`data/sources/egodocs_seed_collection.json`. The seeds remain `resource_screened`, not
`constructed`: normalized final-state replay exists, but no EgoDocs interaction site, human
run, or Browser-Use artifact exists yet.

### Model-assisted visual pre-audit

This pre-audit is a problem-finding pass and does not count as either independent human
review.

- **Graft:** all eight frames show the same plain wooden block-stacking game. The packet does
  not expose a readable brand mark, so the reviewer must explicitly decide whether linking
  the visual referent to the Jenga record is unique enough within the frozen collection.
- **Refresh:** V0 shows the blue packet being opened first; V1 clearly shows a distinct yellow
  Foster Clark's Corn Flour packet later. The temporal retain construct is strong, while the
  small V0 label still requires an independent check that baking powder is identifiable from
  the released four frames.
- **Clarify:** the packet shows a solar-system poster first and an Uncle Sam poster second,
  separated by transition frames. The sequence answer selects the second poster, but the
  reviewer must still confirm that no other wall image is a plausible candidate under the
  ambiguous wording.

## Required task and site contract

The extension should replace cart-specific final state with one normalized state shared by
all sites:

```json
{
  "selected_entity_ids": ["opaque-id"],
  "fields": {"field_name": "exact frozen value"},
  "submission_completed": true
}
```

The task should list complete acceptable states, not separate duplicated target IDs and
answer labels. A commerce task is then the same contract with empty `fields`; a knowledge
task can require both a source and structured values. Site content remains in the site
folder, not in task JSON.

## Complexity added by the workflow

The proposed tasks are not ordinary search questions. They require the Agent to:

1. infer a referent from ordered glasses frames;
2. formulate and revise a search query;
3. distinguish near-neighbor results;
4. open the right record and inspect a decisive section or metadata field;
5. save the source, enter exact fields, and submit a final state;
6. revise, retain, or delay candidate-specific actions under Graft, Refresh, or Clarify.

This is a longer and qualitatively different state transition than selecting a product from
a results grid. Step count is a consequence of the evidence workflow, not a target by
itself.

## Next gate

The normalized task states and negative replay cases are pre-registered in
`data/sources/egodocs_seed_task_specs.json`. The executable replay now checks 17 cases:
correct completion, ignored Graft fields, wrong records, Refresh overreaction, extra or
duplicate sources, missing submission, and Clarify policy failures. All 17 currently produce
their registered decisions. Outcome correctness and clarification policy are reported
separately, so a correct final record reached after an early candidate action is not confused
with a genuinely correct interaction.

The exact released frames, frozen records, and task specifications are bound into
`data/reviews/egodocs_seed_construction_review.json`. All three seeds are now waiting for two
independent construction reviews of instruction naturalness and unique visual linkage.

Only after both reviewers admit a seed should `sites/egodocs/` be implemented and the runtime
be migrated from `selected_item_ids` to `selected_entity_ids`. This avoids building a generic
multi-site framework around a rejected task idea.
