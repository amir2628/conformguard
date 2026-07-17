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

A separate, related robustness gap was found and fixed the same way:
some models/servers can fail to produce a tool call at all even when one
is required (root-caused to an inference-server-side chat template bug
in one case, not a model limitation). `ToolRegistry.handle_openai_choice()`
/ `.handle_anthropic_message()` now take a `required=True` flag that
raises a typed `NoToolCallProducedError` (with the model's `finish_reason`
and actual response content attached) instead of silently returning
nothing — this is not an abstain, since conformguard never saw a call to
score in the first place.

## Phase 2: multi-check joint calibration

`core/multi_check.py` extends single-call calibration to `K` simultaneous
nonconformity checks on one call (e.g. a schema-validity check + a
model-confidence check + a semantic-similarity check), via PASC's
max-nonconformity-score reduction (Kotte et al., arXiv:2605.18812,
Theorem 6): a single threshold, computed over `max(s_1, ..., s_K)` across
the calibration set, gives `P(ALL K checks pass) >= 1 - alpha` — a genuine
joint guarantee, not `K` separate marginal ones.

```python
from conformguard.core.multi_check import calibrate_multi_check, decide_multi_check

calibrator = calibrate_multi_check(
    [schema_gate_scorer, logprob_confidence_scorer, semantic_similarity_scorer],
    calibration_data=historical_calls,
    alpha=0.1,
)
result = decide_multi_check(calibrator, context)
result.decision          # "accept" | "abstain"
result.failed_checks      # which check(s), if any, exceeded the joint threshold
result.guarantee.text     # the joint, K-check-scoped guarantee statement
```

Compared against two alternatives with no/weaker joint guarantees (naive
independent per-check calibration, and the conservative Bonferroni
correction), replicating PASC's own comparison methodology on this
project's tool-calling domain rather than citing their NER numbers.
Validity (does each method meet its good-call coverage target) and
efficiency (does it still reject bad/anomalous calls at the same target)
are both measured — good-call coverage alone turned out to be a
misleading efficiency proxy (see `validation/multi_check_comparison.py`'s
module docstring) and was replaced with rejection rate on a held-out bad
pool:

| rho (check correlation) | naive good-coverage | joint bad-rejection | bonferroni bad-rejection |
|---|---|---|---|
| 0.0 (independent) | 0.730 (invalid) | 0.982 | 0.980 |
| 0.5 | 0.783 (invalid) | 0.935 | 0.915 |
| 0.9 (highly correlated) | 0.852 (invalid) | 0.889 | 0.812 |

Naive is invalid throughout (its coverage never reaches the 0.9 target).
Joint and Bonferroni are both valid, but joint's efficiency advantage over
Bonferroni grows as the checks become more correlated — exactly PASC's
predicted result, reproduced here rather than assumed. Also validated
against a real, non-synthetic pool (two genuinely different deterministic
scores computed from the same BFCL "live" calls used elsewhere in this
README). Reproduce with `pytest tests/coverage_validation/test_multi_check_comparison.py -v`.

Also validated live against a real, locally-running model:
`examples/ollama_multi_check_demo.py` calibrates the same two real
signals as the Phase 1 Ollama demo (a schema gate and a real logprob
confidence score) jointly instead of combining them by hand. Includes a
real one-check-only split — a schema-valid but hedged-into answer
("Paris", guessed for an intentionally ambiguous prompt) that passes
`schema_gate` but fails `logprob_confidence` alone, with `failed_checks`
correctly attributing the abstain to that one check — see
`docs/real_world_validation.md`'s Round 3.

Extended to a live 4-model side-by-side comparison (Qwen2.5, Llama 3.1,
Mistral Nemo, Hermes 3 — Phi-4 Mini was tried first and dropped after
failing to produce structured tool calls) with an identical prompt set
across all four: same per-check-attributed split reappeared on a
different model, all four models caught both malformed-input prompts
consistently, and a smaller calibration sample produced real, honestly
reported abstain volatility on otherwise-ordinary prompts — see
`docs/real_world_validation.md`'s Round 4.

## CLI

```
conformguard inspect --store .conformguard/calibration.db
conformguard threshold --alpha 0.05 --store .conformguard/calibration.db
conformguard coverage-check --alpha 0.05 --calibration-size 1000 --store .conformguard/calibration.db
conformguard multi-check-threshold --data calibration.json --alpha 0.1
conformguard multi-check-coverage-check --data calibration.json --alpha 0.1 --calibration-size 1000 --compare
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
- [`docs/known_issues.md`](docs/known_issues.md) — tracked, real gaps that
  are open or only partially mitigated, stated plainly.

## Roadmap

Phase 1 is the single-call calibration engine described above. Phase 2,
multi-check joint calibration (PASC's max-nonconformity-score reduction),
is also built — see the section above. Phase 3 (stretch) adds
instance-adaptive risk levels. Phase 4, if it happens, is an explicitly
experimental, no-guarantee-claimed research module investigating
trajectory-level coverage — see `docs/guarantee_scope.md`'s closing
section for why that's a genuinely open problem, not just unimplemented.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
