# Data responsibilities

- `tasks/`: the nine executable seed definitions, grouped by Graft, Refresh, and Clarify.
- `observations/`: extracted frames and retained source-screening evidence.
- `sources/`: source-video screening records, rejection reasons, timestamps, and frame hashes.
- `selection.json`: the current nine-task seed selection.
- `reviews/`: independent construction-review queues and submissions.
- `releases/`: immutable approved releases; currently empty apart from its README.

Product data is not duplicated here. EgoMart's single product store and images live under `sites/egomart/`. Generated runs and reports live under `artifacts/`.
