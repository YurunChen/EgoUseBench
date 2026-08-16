# Domain Docs

This is a single-context research codebase.

## Before exploring

- Read `CONTEXT.md` at the repository root.
- Read the ADRs under `docs/adr/` that affect the area being changed.
- If either is absent, proceed silently rather than creating speculative documentation.

## Current layout

```text
/
├── CONTEXT.md
├── docs/adr/
├── egocua/
├── sites/
├── data/
├── artifacts/
├── tools/
└── tests/
```

## Use the glossary vocabulary

Use the domain terms defined in `CONTEXT.md` in issue titles, refactor proposals, hypotheses, tests, and documentation. Avoid synonyms that the glossary explicitly rejects.

If a required concept is missing, first check whether it is unnecessary new terminology. If it represents a real domain gap, record it for domain modeling.

## Flag ADR conflicts

If a proposed change contradicts an accepted ADR, identify the conflict explicitly instead of silently overriding the decision.
