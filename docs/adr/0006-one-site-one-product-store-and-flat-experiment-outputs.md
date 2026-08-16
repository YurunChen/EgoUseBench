# ADR-0006: One fixed site, task-independent products, and flat experiment outputs

## Status

Accepted, 2026-08-10.

## Context

The candidate workspace first used per-task product files and later replaced them with a global product file plus 60 task-specific lists of 16 visible IDs. Both designs made the website depend on the current task. EgoMart behavior lived outside a site directory, while extracted frames needed by executable tasks lived under disposable synthesis outputs. Run paths repeated task, arm, and replicate fields already present in each manifest.

These layers made the filesystem look more expressive without creating independent domain facts. They also allowed copies to disagree.

## Decision

- Each website is one folder under `sites/`. EgoMart owns its Flask app, page builders, product JSON, and static product images under `sites/egomart/`.
- `sites/egomart/products.json` is the only product source of truth. Every task and arm receives the complete fixed catalog; task JSON does not select, embed, or point to product subsets.
- Executable tasks live directly under `data/tasks/<track>/`. Released frames live under `data/observations/`, with source dossiers and hashes under `data/sources/`.
- Generated evidence, audits, and analyses live under `artifacts/reports/`. One formal run lives under `artifacts/runs/<experiment-id>/<run-id>/`; task, arm, and replicate are manifest fields.
- Old workbench and per-task product layouts are removed. Runtime code does not include a legacy loader.

## Consequences

The runtime has one product-loading interface and one Flask implementation. Adding or editing a site product changes the environment for every task, so construction reviews and Run Artifacts bind the complete fixed store. This makes the environment comparable across tasks and removes the task-to-site selection seam, but it also means catalog changes require global search-collision review and invalidate earlier website-bound evidence. Released tasks no longer depend on disposable synthesis directories or legacy product selectors.
