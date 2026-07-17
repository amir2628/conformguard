# Changelog

## Unreleased

Real-world validation against live, locally-running tool-calling models
(Ollama), not just synthetic/recorded data. See
`docs/real_world_validation.md` for the full write-up, including sample
sizes and what didn't generalize.

- Found and fixed a real adapter gap: `ToolRegistry`'s dispatch methods
  had no way to pass response-level scoring signal (e.g. a chat
  completion's per-token logprobs, which some providers attach to the
  response choice rather than the individual tool call) through to a
  scorer without smuggling it into the tool's own call arguments.
  - `core/engine.py`: added `WrappedTool.call_with_context()`, which
    scores and runs a tool against an already-built `ToolCallContext`,
    bypassing `context_builder`.
  - `integrations/raw_tool_loop.py`: `ToolRegistry.handle_anthropic_tool_use()`
    and `.handle_openai_tool_call()` both gained an optional
    `extra_metadata` parameter that routes through `call_with_context`.
- `integrations/raw_tool_loop.py`: added `mean_completion_logprob()`, a
  small provider-agnostic helper that aggregates an OpenAI-shape chat
  completion choice's per-token logprobs into the single scalar
  `logprob_score` expects. Implements whole-completion averaging
  specifically (not just the function-call argument tokens) based on a
  real, tested finding — see `docs/writing_scorers.md` and
  `docs/real_world_validation.md` for the evidence and its limits (the
  aggregation-strategy difference this addresses was observed clearly on
  one model and did not appear at all on two others tested).
- 12 new tests covering `call_with_context`, `extra_metadata`, and
  `mean_completion_logprob`.
- `examples/ollama_live_demo.py`: new example, live end-to-end run
  against a local Ollama model — real harvested calibration data (not
  synthetic), real accept and abstain decisions with the guarantee
  statement printed for each.

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
