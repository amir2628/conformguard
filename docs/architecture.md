# Architecture

## Overview

```
                 historical (call, outcome) pairs
                              |
                              v
                    +-------------------+
                    |    calibrate()    |   core/calibration.py
                    |  (uses scorer +   |
                    |   quantile.py)    |
                    +-------------------+
                              |
                              v
                        Calibrator
                    (q_hat, alpha, scorer,
                     labeling_source, ...)
                              |
                              v
   tool_fn  --->    +-------------------+   ---accept--->  tool_fn(**args)
                     |       wrap()      |
   new call  ------> |  (uses decide())  |   ---abstain-->  escalate / raise / callback
                     +-------------------+
                              |
                              v
                          WrapCallResult
                    (decision, score, threshold,
                     guarantee statement, output)
```

Everything downstream of `calibrate()` depends only on the `Calibrator` it
produces (`core/calibration.py:Calibrator`) — a frozen, immutable snapshot
of `q_hat`, `alpha`, the scorer, and the calibration set's metadata. `wrap()`
never recomputes `q_hat`; it only ever compares a new score against the
one baked into the `Calibrator` at calibration time.

## Module map

- **`core/quantile.py`** — the mathematical core. `conformal_quantile()`
  computes the split-conformal threshold as a direct k-th-order-statistic
  lookup (deliberately not via `numpy.quantile`'s interpolation modes,
  which use a different virtual-index convention than the paper's
  definition — see the module docstring and
  `tests/unit/test_quantile.py::test_numpy_method_higher_would_be_wrong_here`
  for the exact discrepancy this avoids).
- **`core/scores.py`** — `NonconformityScore` wraps a raw scoring callable
  with name + error-safe invocation (`.safe()` converts any exception, or
  any non-finite return value, into a forced-abstain `ScoreOutcome` rather
  than letting it propagate or silently pass through). Built-in scorers
  (`logprob_score`, `make_judge_score`, `schema_validity_score`) are thin,
  provider-agnostic wrappers around this interface, not a majority of it —
  a bare user callable is the general case.
- **`core/calibration.py`** — `calibrate()` scores historical
  `(ToolCallContext, outcome)` pairs, keeps only the `outcome=True`
  ("known-good") ones, and computes `q_hat` over their scores. Enforces
  `HARD_MINIMUM_SIZE` (raises `InsufficientCalibrationDataError` below
  it) rather than ever producing a threshold from too little data.
- **`core/decision.py`** — `decide()` is the only place accept/abstain is
  decided, and the only place a `GuaranteeStatement` is built. A scorer
  error or non-finite score always forces abstain here, regardless of
  where `q_hat` happens to fall (including the edge case where `q_hat`
  itself is `+inf`).
- **`core/engine.py`** — `wrap()` composes a `Calibrator` with a plain
  tool function into a `WrappedTool`. On accept, the underlying function
  is called completely unmodified with the original arguments. On
  abstain, behavior is controlled by `on_abstain`: `"escalate"` (default,
  returns a structured abstain result), `"raise"` (raises
  `AbstainedError` carrying the full decision), or a user callback
  (invoked with the `WrapResult`; its return value becomes `.output`).
  `WrappedTool.call_with_context(context)` is the other entry point:
  it bypasses `context_builder` entirely and scores/runs the tool against
  an already-built `ToolCallContext`. This exists for scoring signal that
  isn't part of the tool's own call arguments — for example, a real
  chat-completion API's per-response confidence/logprob data, which is
  attached to the choice/response, not to the individual tool-call
  object. This wasn't in the original design (§4.2 of the original spec
  only described the `context_builder(**kwargs)` path); it was added
  after wiring the raw-loop adapter up to a real local model surfaced the
  gap — see `docs/real_world_validation.md`.
- **`storage/calibration_store.py`** — a local-first SQLite store for
  `(score, outcome, metadata)` records. No network dependency; the store
  also computes staleness/size warnings (not enforcement — enforcement of
  the hard minimum is `calibrate()`'s job, not the store's).
- **`validation/coverage_check.py`** — `theoretical_coverage_band()`
  computes the exact Beta-distribution fluctuation band for observed
  coverage at a given `(n, alpha)` (see the module docstring for the
  derivation: coverage is exactly `Beta(k, n-k+1)`-distributed for
  continuous, exchangeable scores). `run_coverage_validation()` runs `R`
  repeated random calibration/test splits on a fixed score pool and
  reports whether mean observed coverage falls inside that band. This is
  the project's core proof artifact, not a side utility.
- **`validation/negative_control.py`** — the same repeated-split
  methodology, but with the test-time score distribution deliberately
  shifted (`constant_shift`, `distribution_swap`) to break
  exchangeability by construction. Proves the coverage-validation harness
  itself correctly reports degraded coverage rather than papering over a
  broken assumption.
- **`integrations/raw_tool_loop.py`** — `ToolRegistry` dispatches
  Anthropic `tool_use` blocks and OpenAI function tool calls to
  `WrappedTool` instances and formats the result back into each API's
  expected shape (`tool_result` block / `role: tool` message). Phase 1's
  only integration, deliberately — see `PROJECT_SPEC.md` §4.5 for why
  framework breadth is explicitly deferred until the math and this
  adapter are both proven. Both dispatch methods accept an optional
  `extra_metadata` dict; when given, it's merged into the
  `ToolCallContext` built directly from the tool call (via
  `WrappedTool.call_with_context`, bypassing the tool's own
  `context_builder`) rather than being squeezed into the tool's call
  arguments. This is the sanctioned way to pass response-level scoring
  signal — e.g. a real API's per-completion confidence/logprob data —
  through the adapter. `mean_completion_logprob(choice)` is a small
  provider-agnostic helper in this same module that aggregates an
  OpenAI-shape chat completion choice's per-token logprobs into the
  single scalar `logprob_score` expects; see its docstring, and
  `docs/writing_scorers.md`, for why it averages over the whole
  completion rather than just the argument tokens.
- **`cli/main.py`** — `inspect`, `threshold`, `coverage-check` commands
  over a calibration store, for introspecting stored data and re-running
  the coverage proof against it without writing a script. Phase 2 adds
  `multi-check-threshold` and `multi-check-coverage-check`, operating on
  a JSON calibration-data file rather than the SQLite store (which has no
  schema yet for "these K rows are simultaneous checks on one call" --
  see the `multi_check.py` section above). `multi-check-coverage-check
  --compare` currently reports only good-call coverage for each method,
  not the bad-rejection efficiency metric that
  `validation/multi_check_comparison.py` also computes, since there's no
  CLI-level way yet to supply a separate bad/anomalous pool.

## Why the calibration set is restricted to `outcome=True`

This is the least obvious design decision in the codebase, so it's
documented in three places (`core/calibration.py`'s module docstring,
here, and `docs/guarantee_scope.md`) rather than once. Classical split
conformal prediction always calibrates on ground-truth-correct
`(X, Y)` pairs; the guarantee is about how often the *true* label's
nonconformity score exceeds the threshold, not about detecting a wrong
one. Applying that directly to tool calls: the calibration set is past
calls known to have been good, and the guarantee bounds how often a new,
*actually good* call gets wrongly refused — not how often a bad call
sneaks through. `outcome=False` examples are still recorded (for
diagnostics, and because a future score-quality tool might want the
good/bad score overlap), but are not part of the quantile computation
itself. See `tests/unit/test_calibration.py::TestQuantileMatchesDirectComputation`
for a test that pins this behavior directly.

## Extension points intentionally left open, not yet built

- **Weighted / covariate-shift conformal calibration**
  (Angelopoulos & Bates §4.5-4.6) for a *known* calibration-to-deployment
  distribution shift. `CalibrationStore`'s per-record metadata (tool name,
  context bucket, timestamp, version) is structured so a future weighting
  scheme has what it needs without a storage migration, but no weighting
  logic exists yet.
- **Multi-check joint calibration** (PASC's max-nonconformity-score
  reduction) for `K` simultaneous checks on one call — planned as
  `core/multi_check.py` in Phase 2. Nothing in Phase 1's `Calibrator` or
  `decide()` assumes a single check in a way that would need reworking;
  Phase 2 adds alongside, not instead of.
- **Trajectory-level scope** is explicitly out of Phase 1-3's scope; see
  `docs/guarantee_scope.md`'s closing section.
