# Regenerable artifacts

Everything below this directory is ignored by version control and can be recreated from versioned definitions plus external data mounts.

- `runs/`: one auditable Run Artifact per Experiment Run;
- `reports/`: site audits, retained review material, smoke reports, aggregate metrics, confidence intervals, and failure summaries.

Released observation frames are benchmark input, not disposable output, and live under `data/observations/`.

Do not use terminal output or a global log file as the only record of an experiment.
