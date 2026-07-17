# Real-call data provenance

`bfcl_live_simple_extract.json` is a derived extract of the "live_simple"
split of the Berkeley Function-Calling Leaderboard (BFCL) dataset:

- Source: `ShishirPatil/gorilla`, `berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_simple.json`
  and the corresponding `possible_answer/BFCL_v4_live_simple.json`.
- License: Apache License 2.0 (same as this project).
- Nature of the data: real, single-turn function-calling queries
  contributed by real developers to the BFCL "live" split (BFCL's own
  naming convention for user-contributed, non-synthetic data), each
  paired with a human-verified ground-truth function call.

Each entry in the extract is `{"id", "tool_name", "args", "n_schema_properties"}`,
derived deterministically from the upstream question + ground-truth files:
`args` is the first acceptable value for each parameter from the ground
truth's value-options list, and `n_schema_properties` is the number of
parameters declared on that call's function schema.

This is used as a **real, non-synthetic pool of known-good tool calls**
(every entry is, by construction, a verified-correct call) for
`scenarios/real_bfcl_calls.py`, so the coverage validation suite is
exercised against the shape of real tool-calling data, not only
synthetic distributions. See PROJECT_SPEC §5.3 / §8.
