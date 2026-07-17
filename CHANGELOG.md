# Changelog

## 0.2.0 — 2026-07-17

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

Investigated a live model (Command R7B via Ollama) that never produced a
tool call through either the OpenAI-compatible or native chat endpoint.
Root-caused directly (not just inferred): Ollama's default chat template
for this model hardcodes every assistant turn to begin with
`<|START_RESPONSE|>`, which structurally prevents it from ever reaching
the `<|START_ACTION|>` phase its own system preamble instructs it to use
for tool calls — confirmed by bypassing the template via a raw completion
request, which then did produce tool-call-shaped output. This is an
Ollama-side template bug, not fixable from this library's adapter, which
operates on the standard chat-completion surface. Fixed the general
robustness gap this exposed regardless of the specific root cause:
- `integrations/raw_tool_loop.py`: added `NoToolCallProducedError`, and
  `ToolRegistry.handle_openai_choice()` / `.handle_anthropic_message()`,
  which dispatch every tool call in a full response and raise this typed
  error (with `finish_reason`/`stop_reason` and the model's actual
  content attached) when `required=True` and none was produced — this is
  not an abstain, since conformguard never saw a call to score.
- 31 new tests, including a reproduction against the actual real response
  captured live from the failing model.

Phase 2, multi-check joint calibration (PROJECT_SPEC §3 Phase 2 / §4.4):
- `core/multi_check.py`: `calibrate_multi_check()` / `decide_multi_check()`
  support K simultaneous nonconformity checks on one call, calibrated
  jointly via PASC's max-nonconformity-score reduction (arXiv:2605.18812,
  Theorem 6) — a single q_hat over `max(s_1, ..., s_K)` gives
  `P(all K checks pass) >= 1 - alpha`. Verified against a hand-computed
  toy example (K=2, n=5).
- `validation/multi_check_comparison.py`: comparison harness against (a)
  naive independent per-check calibration (shown invalid: good-call
  coverage below target) and (b) Bonferroni correction, replicating
  PASC's own methodology on this project's tool-calling domain. Good-call
  acceptance rate was tried first as the efficiency metric and found
  actively misleading (Bonferroni's acceptance rate climbs above its own
  target as check correlation increases, which looks like an advantage
  but reflects a looser, less discriminating threshold); rejection rate
  on a held-out bad/anomalous pool is the corrected metric, and shows
  joint's efficiency advantage over Bonferroni growing with correlation,
  as PASC predicts.
- `cli/main.py`: `multi-check-threshold` and `multi-check-coverage-check`
  commands, operating on a JSON calibration-data file (the SQLite
  calibration store's one-score-per-row schema has no notion of "these K
  rows are simultaneous checks on one call" yet; retrofitting that
  grouping is separate, not-yet-started storage work).
- Full unit, empirical-coverage, and negative-control tests, same rigor
  as Phase 1 — including a negative control that shifts a SINGLE check's
  test-time distribution (not the already-reduced max score), confirming
  the other K-1 healthy checks don't mask one check's drift.
- PROJECT_SPEC.md §8a: Phase 2 acceptance criteria added and checked
  against what's built (local planning doc, not shipped with the
  package).
- `examples/ollama_multi_check_demo.py`: live end-to-end run of joint
  multi-check calibration against a local Ollama model, calibrating the
  same two real signals as the Phase 1 demo jointly instead of combining
  them by hand into one composite score. Real harvested calibration data,
  including a genuine one-check-only split (`schema_gate` passes,
  `logprob_confidence` fails alone) confirming `failed_checks` correctly
  attributes an abstain to a single check rather than reporting an opaque
  "something failed" — see `docs/real_world_validation.md`'s Round 3.
- `cli/main.py`: `multi-check-coverage-check --compare` gained an
  optional `--bad-data` file (mirroring
  `run_multi_check_comparison()`'s existing `bad_pool` parameter), so it
  now reports each method's actual bad-rejection efficiency metric
  alongside good-call coverage instead of good-call coverage alone.
  Validates that `--bad-data`'s check names match `--data`'s. Without
  `--bad-data`, the command still runs but prints an explicit warning
  that the efficiency metric is being omitted, rather than silently
  reporting a partial comparison. 3 new tests.
- `docs/known_issues.md`: new file tracking real, open gaps that were
  previously only noted in docstrings or commit messages — currently the
  Command R7B / phi4-mini tool-call-production failures and the Phase 2
  multi-check JSON-file storage stopgap described above.
- Extended real-world validation to a live 4-model side-by-side
  comparison (Qwen2.5 7B, Llama 3.1 8B, Mistral Nemo, Hermes 3 — Phi-4
  Mini was tried first and dropped after failing the same smoke test as
  Command R7B) of joint (K=2) calibration, each with its own calibrated
  threshold. Surfaced a one-check-only split reappearing on a different
  model (Hermes 3), a new tool-call-as-text quirk on Hermes 3, and an
  honest finding that smaller calibration samples produced more abstains
  on ordinary, non-adversarial prompts. See `docs/real_world_validation.md`
  Round 4 for the full write-up, including the phi4-mini failure.

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

Phase 3 (adaptive thresholds) and Phase 4 (trajectory-level research
module) are not yet started — see the Roadmap section of `README.md`.
Phase 2 (multi-check joint calibration) is built; see above.
