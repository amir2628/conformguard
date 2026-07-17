# Changelog

## 0.1.0 — Phase 1

Initial release: the core split-conformal calibration engine and
single-call accept/abstain decision.

- `core/quantile.py`: exact split-conformal quantile
  (Angelopoulos & Bates, arXiv:2107.07511, Theorem 1), verified against
  hand-computed toy examples.
- `core/scores.py`: pluggable nonconformity score interface, with
  built-in `logprob_score`, `make_judge_score`, `schema_validity_score`.
  A scorer error always forces abstain, never a silent accept.
- `storage/calibration_store.py`: local SQLite calibration store with
  staleness and size warnings.
- `core/calibration.py` / `core/decision.py`: `calibrate()` and the
  accept/abstain decision, with the exact, scoped guarantee statement
  attached to every decision.
- `validation/coverage_check.py`: empirical coverage validation suite
  (exact Beta-distribution fluctuation band, not eyeballed).
- `validation/negative_control.py`: proves the coverage-validation
  harness correctly detects deliberately broken exchangeability.
- `core/engine.py` / `integrations/raw_tool_loop.py`: `wrap()` and the
  Anthropic/OpenAI raw tool-calling loop adapter.
- `cli/main.py`: `inspect`, `threshold`, `coverage-check` commands.
- Coverage validation suite run against both synthetic scenarios and a
  real, non-synthetic tool-call dataset (BFCL "live" split).

Phase 2 (multi-check joint calibration), Phase 3 (adaptive thresholds),
and Phase 4 (trajectory-level research module) are not yet started — see
the Roadmap section of `README.md`.
