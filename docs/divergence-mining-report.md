# Ego2Web divergence-event mining (2026-08-09)

> Follow-up: this report intentionally preserves the negative result for the **Ego2Web-only** pool. Subsequent external-source passes screened four HD-EPIC Refresh-replace candidates and three EgoSchema/Ego4D Refresh-retain candidates, while explicitly rejecting three HD-EPIC and three EgoSchema events; see [`refresh-source-dataset-audit.md`](research/refresh-source-dataset-audit.md). Those follow-ups do not change the finding that Ego2Web itself supplied no verified natural replacement event.
>
> A separate Ego2Web subtype-expansion pass used the same 500 annotations and 310 videos to screen Graft-C, Graft-R, Clarify-sequence, and Clarify-position candidates. It retained four candidates and rejected two tempting Clarify alternatives; this fills the source-screening matrix but does not change the Refresh finding or create release tasks. See [`subtype_expansion_screening_manifest.json`](../data/sources/subtype_expansion_screening_manifest.json).

## Conclusion

Ego2Web can supply plausible source material for Graft and Clarify, but the current evidence
does **not** establish a usable natural Refresh-replace pool. The distinction matters: seeing a
person interact with A and later with B is not enough to conclude that B replaced A. A valid
replace task must visibly establish all four steps: A is selected, A is returned or relinquished,
B is selected afterwards, and B is the final retained choice.

The reproducible metadata scan found 21 instructions that mention physical selection or contact,
including 4 ordinal pickup instructions across 2 videos. It found 0 instructions that explicitly
describe returning one object, changing one's mind, or selecting another object instead. Dense
continuous-video review of the nine most plausible shopping or sequential-handling videos then
rejected all nine proposed constructions. The final snack-bar sequence was removed after a
9–21 second dense temporal audit showed multiple contacts but no unique return-and-final-choice
event, while its proposed later four-frame packet also failed to identify one final B.

This is a useful negative feasibility result. The project must not assume the planned quota of
four Refresh-replace tasks can be filled from Ego2Web, and it must not weaken replace semantics to
make the quota fit the data.

## Method

`python3 -m tools.mine_divergence_events` scans
`reference/Ego2Web/dataset/ego2web.json` (500 instructions, 310 locally available videos). Its
machine output is
[`divergence_mining.json`](../artifacts/reports/mining/divergence_mining.json). The script
separates three levels of evidence:

1. same-video time intervals, which are only a high-recall source pool;
2. instruction wording about physical handling, which still does not prove replacement;
3. explicit replacement wording, followed by continuous-video review.

Manual screening decisions and original annotations are recorded in the
[`seed_screening_manifest.json`](../data/sources/seed_screening_manifest.json).
Model-assisted inspection is allowed for conservative screening and evidence-gap diagnosis but
cannot admit a task or count as human ground truth. Admission still requires two independent human
constructors; any future yield estimate must also audit a sample of screening rejections rather
than treating model-assisted exclusion as infallible.

## Findings

### Graft: same-video instruction pairs are available, but not automatically valid

- 138 / 310 videos (44.5%) have at least two instructions, producing 258 same-video instruction
  pairs.
- 69 pairs use disjoint time ranges; 58 of those share at least one content token.
- These counts show a useful source pool, not 258 valid Graft tasks. A valid pair must still share
  the intended visual referent, make B natural and essential, and satisfy one registered C/P/R
  relation.
- The current Graft-P seed uses Ego2Web records 482 and 483, but remains `video_screened` until
  two constructors approve the later travel/tin-edition request.

### Refresh: sequential handling is not equivalent to target replacement

| Evidence level | Count | What it supports |
|---|---:|---|
| Same-video disjoint interval pairs | 69 | A broad pool of scenes that change over time |
| Rows mentioning physical selection/contact | 21 | Candidate handling events for review |
| Ordinal pickup rows | 4 rows / 2 videos | Multiple objects handled in an order |
| Rows explicitly describing A-return/change-of-mind/B-instead | 0 | No metadata-level proof of replacement |
| Most plausible videos reviewed continuously | 9 | All nine rejected; none contains a verified complete replacement sequence |
| Refresh-replace tasks admitted | 0 | No release-quality seed yet |

The main rejected patterns were:

- several independent grocery-list items collected in sequence;
- ordered containers used in a laboratory procedure;
- one selected product plus an adjacent comparison item;
- touching or pointing at neighboring snacks without fully selecting either;
- a white coat being returned, followed only by continued browsing rather than a final B choice;
- an automatically paired V1 interval that actually occurred before V0.

The snack-bar sequence is preserved only as a rejection record and audit contact sheet. It is no
longer an Agent-input candidate and cannot be revived by ordinal wording or by relabeling it retain.

### Clarify: textual qualifiers form a source pool, not automatic ambiguity

- 450 / 500 instructions (90.0%) refer deictically to the video.
- 201 / 500 instructions (40.2%) combine deixis with at least one mined qualifier family:
  variant/brand 169, position 45, sequence 24, and color 8. Families overlap.
- Removing a qualifier does not automatically create a valid Clarify task. The source scene must
  visibly contain exactly two plausible candidates, and one natural answer must uniquely select
  one.
- The current rabbit/pig finger-puppet seed meets the initial visual screen after its 8-frame
  interval was expanded to balance pig-focused and rabbit-focused interaction. Three earlier
  Clarify candidates were rejected because the required pair of candidates did not exist or the
  proposed subtype was wrong.

## Claims permitted by the current evidence

- Same-video annotations and qualified visual references make Ego2Web a useful high-recall source
  pool for task construction.
- Natural Refresh-replace events are substantially harder to recover than ordinal or multi-object
  handling events.
- The first reviewed pool has produced no admitted Refresh-replace task; source feasibility must
  be resolved before scaling.

## Claims not permitted

- “44.5% of Ego2Web videos contain natural Graft events.”
- “40.2% of Ego2Web instructions become valid Clarify tasks when a qualifier is removed.”
- “Ego2Web contains verified target switches because shoppers pick up several products.”
- Any acceptance-rate or 60-task coverage estimate before independent human review.
