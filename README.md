# conformguard

Calibrated accept / abstain-and-escalate decisions for agent tool calls,
backed by split conformal prediction's finite-sample coverage guarantee —
not a heuristic confidence threshold.

Every tool-calling agent framework today decides "should I trust this
output / call this tool / hand off to a human" using heuristics: a
hardcoded confidence threshold, a judge-LLM score, a retry counter. None
of these come with a statistical guarantee. `conformguard` wraps a single
tool call with a calibrated threshold, computed via split conformal
prediction (Angelopoulos & Bates, arXiv:2107.07511), and attaches the
exact, scoped guarantee statement to every decision — not just to a docs
page nobody reads.

**What this is not:** a general-purpose conformal prediction library
(MAPIE, puncc, nonconformist already do that well for classifiers and
regressors — this project doesn't reimplement it), a heuristic guardrail
library (Guardrails AI, NeMo Guardrails validate structure/content, with
no statistical coverage guarantee), or a trajectory-level guarantee for
multi-step agent sessions (explicitly out of scope for now — see
"What this does not guarantee" below).

## Install

```
pip install conformguard
```

Requires Python 3.10+.

## Quickstart

```python
from conformguard import calibrate, wrap, ToolCallContext, LabelingSource

# 1. A nonconformity score: higher = more suspicious. Any callable
#    ToolCallContext -> float works; see docs/writing_scorers.md for the
#    built-in logprob_score / judge_score / schema_validity_score.
def query_length_score(context: ToolCallContext) -> float:
    return len(context.args["query"])

# 2. Calibration data: past (call, outcome) pairs, outcome=True means
#    "this call was, in hindsight, correct/safe" (see §"Outcome labeling"
#    in docs/guarantee_scope.md).
calibration_data = [
    (ToolCallContext(tool_name="search", args={"query": q}), True)
    for q in historical_queries  # your own logged, known-good queries
]

calibrator = calibrate(
    scorer=query_length_score,
    calibration_data=calibration_data,
    alpha=0.05,  # wrong at most 5% of the time
    labeling_source=LabelingSource.DETERMINISTIC,
)

# 3. Wrap the tool. The underlying function is never rewritten.
def search(query: str) -> list[str]:
    ...

wrapped_search = wrap(
    search,
    calibrator=calibrator,
    on_abstain="escalate",  # or "raise", or a callback
    context_builder=lambda **kwargs: ToolCallContext(tool_name="search", args=kwargs),
)

result = wrapped_search(query="weather in Berlin")
result.decision    # "accept" | "abstain"
result.guarantee   # the exact, scoped guarantee statement
result.output      # search()'s return value, if accepted
```

See `examples/raw_loop_demo.py` for a complete, runnable, no-API-key-needed
walkthrough, and `docs/writing_scorers.md` for the scorer interface and
built-in scorers.

## The guarantee, precisely

For a single tool call, scored against a threshold computed from `n`
calibration examples via

```
q_hat = k-th smallest calibration score, where k = ceil((n+1)(1-alpha))
```

if the new call is exchangeable with the calibration set:

```
P(this call is wrongly abstained on) <= alpha
```

This is Angelopoulos & Bates' Theorem 1, applied directly to the
tool-calling decision point. It is a **finite-sample** guarantee (holds at
the actual `n` you calibrated on, not just as `n -> infinity`), and it
holds **regardless of how good the nonconformity score is** — score
quality affects usefulness (how often you abstain), not validity.

Every decision returned by `wrap()` carries this as structured data
(`result.guarantee`), not just a log line:

```
Under the assumption that this call is exchangeable with the 1000-example
calibration set (labeling source: deterministic; collected 2026-06-01
through 2026-07-01), this accept/abstain decision is wrong at most 5.0% of
the time (alpha=0.05), for THIS SINGLE CALL ONLY. This is not a guarantee
about any multi-step task this call is part of, and it holds only if the
exchangeability assumption is not violated -- see docs/guarantee_scope.md.
```

**Full detail, including what breaks this guarantee in practice and how
outcome labeling affects it: [`docs/guarantee_scope.md`](docs/guarantee_scope.md).**

## What this does NOT guarantee

- **Multi-step trajectories.** The guarantee is per single call. A
  sequence of individually-accepted calls carries no joint guarantee.
  Trajectory-level coverage is a genuinely open research problem — no
  published result proves it (see `docs/guarantee_scope.md`'s closing
  section) — and nothing in this library should be read as claiming
  otherwise.
- **Catching bad calls.** The guarantee bounds how often a *good* call is
  wrongly refused, not how often a *bad* one is wrongly accepted. That
  depends on your scorer's quality, which the coverage guarantee's
  validity does not depend on.
- **Exchangeability that has already broken.** If your deployment
  distribution has drifted from your calibration data, the guarantee
  simply does not hold until you recalibrate. `conformguard` warns on
  stale/undersized calibration data; it does not yet auto-detect drift
  (Phase 1 scope).

## Real coverage-validation numbers

These are the actual output of `tests/coverage_validation/`, run against
this codebase — not hand-picked. Reproduce with:

```
pytest tests/coverage_validation -v
```

`R = 100` repeated calibration/test splits per row, calibration size
`n = 1000`, against three synthetic score-distribution scenarios designed
to stress different shapes a real nonconformity score might take
(uniform, right-skewed "most calls score confidently," and a logprob-
derived transform matching the built-in `logprob_score`):

| scenario | alpha | target coverage | mean observed | theoretical band (95% CI) | within band |
|---|---|---|---|---|---|
| uniform_scores | 0.10 | 0.90 | 0.8999 | [0.8808, 0.9179] | yes |
| uniform_scores | 0.05 | 0.95 | 0.9492 | [0.9357, 0.9627] | yes |
| skewed_confidence_scores | 0.10 | 0.90 | 0.8995 | [0.8808, 0.9179] | yes |
| skewed_confidence_scores | 0.05 | 0.95 | 0.9501 | [0.9357, 0.9627] | yes |
| logprob_derived_scores | 0.10 | 0.90 | 0.9017 | [0.8808, 0.9179] | yes |
| logprob_derived_scores | 0.05 | 0.95 | 0.9513 | [0.9357, 0.9627] | yes |

The "theoretical band" column is not eyeballed: it's the exact
`Beta(k, n-k+1)` distribution of test-conditional coverage (derived in
`validation/coverage_check.py`'s module docstring), evaluated at each
row's `(n, alpha)`.

### Validated against real (non-synthetic) tool-call data

Per the project's own acceptance criteria, the suite has also been run
against real, non-synthetic data: 258 real, human-contributed
single-function-call queries from the Berkeley Function-Calling
Leaderboard's "live" split (`ShishirPatil/gorilla`, Apache-2.0 — see
`tests/coverage_validation/data/README.md` for full provenance), each
paired with a human-verified ground-truth call, scored by a deterministic
argument-complexity measure:

| scenario | alpha | target coverage | mean observed | theoretical band (95% CI) | within band |
|---|---|---|---|---|---|
| real_bfcl_live_calls (n=150, pool=258) | 0.10 | 0.90 | 0.8986 | [0.8484, 0.9429] | yes |

Reproduce with `pytest tests/coverage_validation/test_real_data_coverage.py -v`.

### Negative controls: proving the validator isn't lying

`tests/negative_controls/` deliberately breaks exchangeability (shifting
the test-time score distribution by a full spread-width) and confirms the
same coverage-validation harness correctly reports degraded coverage,
rather than papering over a broken assumption with a falsely reassuring
number:

| scenario | alpha | no-shift mean coverage | no-shift flagged degraded | shifted mean coverage | shifted flagged degraded |
|---|---|---|---|---|---|
| uniform_scores | 0.10 | 0.8999 | no | 0.0000 | **yes** |
| skewed_confidence_scores | 0.10 | 0.8995 | no | 0.0000 | **yes** |
| logprob_derived_scores | 0.10 | 0.9017 | no | 0.0000 | **yes** |

Reproduce with `pytest tests/negative_controls -v`.

## Real-world validation

The statistical proofs above run on synthetic and recorded data; separately,
this library has also been driven end-to-end against real, locally-running
tool-calling models (Qwen2.5, Llama 3.1, Mistral Nemo, Command R7B via
Ollama) — real HTTP requests, real model-generated tool calls, real accept
and abstain decisions with the full guarantee statement printed for each.
That work surfaced a real gap (no channel existed for response-level
scoring signal like per-completion logprobs) and a real, only
partially-generalizing finding about how to aggregate a model's per-token
logprobs into a single confidence score — both are written up in full,
including what didn't hold up under a larger sample, in
[`docs/real_world_validation.md`](docs/real_world_validation.md).

## CLI

```
conformguard inspect --store .conformguard/calibration.db
conformguard threshold --alpha 0.05 --store .conformguard/calibration.db
conformguard coverage-check --alpha 0.05 --calibration-size 1000 --store .conformguard/calibration.db
```

## Development

```
pip install -e ".[dev]"
pytest tests/unit tests/integration          # fast, default gate
pytest tests/coverage_validation              # required before merging changes to core/quantile.py, core/decision.py
pytest tests/negative_controls                 # required before any release
```

## Documentation

- [`docs/guarantee_scope.md`](docs/guarantee_scope.md) — what is and is
  not proven, read this first.
- [`docs/architecture.md`](docs/architecture.md) — module map and design
  rationale.
- [`docs/writing_scorers.md`](docs/writing_scorers.md) — the nonconformity
  score interface and built-in scorers.
- [`docs/real_world_validation.md`](docs/real_world_validation.md) — live
  runs against real local tool-calling models, what broke, what was fixed.

## Roadmap

Phase 1 (this release) is the single-call calibration engine described
above. Phase 2 (planned) adds joint calibration across a fixed set of `K`
simultaneous checks on one call (PASC's max-nonconformity-score
reduction). Phase 3 (stretch) adds instance-adaptive risk levels. Phase 4,
if it happens, is an explicitly experimental, no-guarantee-claimed
research module investigating trajectory-level coverage — see
`docs/guarantee_scope.md`'s closing section for why that's a genuinely
open problem, not just unimplemented.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
