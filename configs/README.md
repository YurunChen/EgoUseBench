# Experiment configurations

Committed experiment plans own the single formal condition for each track, replicate IDs, production runtime identity, step budget, and scheduler seed. In `pilot.json`, Graft uses `injected`, Refresh uses `stream`, and Clarify uses `interactive`; one task therefore produces one experiment per replicate. Optional control conditions belong to a separate diagnostic plan and are not part of the default benchmark.

Credentials, local absolute paths, and provider secrets remain in environment variables or ignored local files.
