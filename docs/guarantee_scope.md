# Guarantee scope

This document exists because overclaiming is the single worst failure mode
this project can have. A README, docstring, or log line that says
"guaranteed" next to a number that doesn't actually hold — because the
calibration set wasn't representative, was too small, or the
exchangeability assumption silently broke — is worse than no guarantee at
all, because it's actively misleading. Read this before trusting, or
quoting, any number this library produces.

## What is actually proven

**Claim:** for a single tool call, scored by a fixed nonconformity score
function and compared against a threshold `q_hat` computed from `n`
calibration examples via

```
q_hat = k-th smallest calibration score, where k = ceil((n+1)(1-alpha))
```

if the new call's nonconformity score is exchangeable with the
calibration set's scores, then

```
P(new call is wrongly abstained on) <= alpha
```

This is Theorem 1 of Angelopoulos & Bates ("A Gentle Introduction to
Conformal Prediction and Distribution-Free Uncertainty Quantification",
arXiv:2107.07511), applied directly: the calibration set is the set of
past tool calls known, in hindsight, to have been correct/safe
(`outcome=True`), and the guarantee says that a new call drawn from the
same distribution as those calibration calls will not be wrongly refused
more than `alpha` fraction of the time. `core/quantile.py`'s test suite
(`tests/unit/test_quantile.py`) hand-verifies this formula against
independently worked toy examples, and `tests/coverage_validation/`
empirically confirms it holds across repeated calibration/test splits (see
"Real coverage numbers" below).

**This is a finite-sample guarantee, not an asymptotic one.** It does not
require `n -> infinity`; it holds for the actual `n` you calibrated on
(subject to `n` meeting the hard minimum described below).

## What is explicitly NOT guaranteed

- **Multi-step trajectories.** The guarantee above is per single tool
  call. It says nothing about the probability that an entire multi-step
  agent task completes correctly, or that a sequence of accepted calls is
  jointly safe. A trajectory-level guarantee is an open research question
  (see the "Phase 4" scope note below); nothing in this codebase should be
  read as extending the single-call guarantee to a session.
- **Catching bad calls.** The guarantee bounds how often a *good* call is
  wrongly abstained on. It says nothing about how often a *bad* call is
  wrongly accepted — that depends entirely on the quality of the
  nonconformity score, which the guarantee's validity does not depend on
  (see "Score quality" below).
- **Exchangeability that has actually broken.** If the deployment
  distribution has drifted from the calibration distribution, the
  guarantee simply does not hold, silently, unless you actively check for
  it (see "Exchangeability" below).
- **Anything not scoped in the guarantee statement.** Every decision
  carries a `GuaranteeStatement` with an explicit `alpha`, `scope`,
  calibration set size, date range, and labeling source attached as data,
  not just prose. If a claim about this library's behavior doesn't trace
  to a field on that object, treat the claim as unverified.

## Score quality vs. guarantee validity

A bad nonconformity score does not break the guarantee's *validity* —
coverage still holds at the stated `alpha`, regardless of how good or bad
the score is at distinguishing risky calls from safe ones. What a bad
score breaks is the guarantee's *usefulness*: a score with no real signal
either abstains on almost everything (if it's mostly noise near the
extreme) or accepts almost everything (if its scale doesn't separate risky
calls from safe ones). The empirical coverage validation suite's abstention
rate and set-size numbers are how you judge score quality — not intuition,
and not the coverage number itself, which will look fine either way. See
`docs/writing_scorers.md`.

## Exchangeability, in practice

The mathematical assumption is that the new call's nonconformity score is
exchangeable with the calibration set's scores — informally, that the new
call is "drawn from the same process" as the calibration data. In
practice this fails when:

- the tool, its arguments, or its usage pattern has changed since
  calibration,
- the underlying model has been upgraded or fine-tuned,
- the population of users/callers has shifted,
- enough time has passed that "typical" calls look different now.

`storage/calibration_store.py` timestamps and versions every calibration
set and warns (does not silently proceed) when data is older than a
configurable threshold (default 90 days) or below the recommended minimum
size. `validation/negative_control.py` and
`tests/negative_controls/test_exchangeability_violation_detected.py` prove
that when exchangeability is deliberately broken (by shifting or swapping
the test-time score distribution), the coverage validation harness
correctly reports degraded coverage rather than a falsely reassuring
in-band number — see that suite's own report
(`tests/negative_controls/latest_report.json`) for real numbers, not just
an assertion that it works.

There is no automatic drift *detection* in Phase 1 — only staleness/size
*warnings* and the ability to re-run the coverage validation suite
yourself against fresh data. Angelopoulos & Bates §4.5-4.6 describe real,
proven weighted/covariate-shift conformal techniques for the case where
the calibration-to-deployment shift is known; this library's storage layer
is deliberately architected to not preclude adding that later (see
`docs/architecture.md`), but it is not built in Phase 1.

## Cold start and the hard minimum

`calibrate()` raises `InsufficientCalibrationDataError` below
`HARD_MINIMUM_SIZE` (100) good-outcome examples — it does not silently
produce a wide-open or meaningless threshold from too little data. This is
a deliberate difference from a library that degrades gracefully when
under-provisioned: a calibrator built on a handful of examples is not a
weaker version of the guarantee, it is no guarantee worth trusting at all.
Between the hard minimum and the recommended minimum (1000, matching
Angelopoulos & Bates' own guidance for keeping coverage fluctuation within
roughly +/-2 percentage points at alpha=0.1 — see their Table 1), the
calibration store's `size_warning()` fires but calibration proceeds; treat
a calibrator built in that range as provisional.

## Outcome labeling

The guarantee is only as good as the `outcome` labels in the calibration
set. Three labeling sources are supported (`storage/calibration_store.py`'s
`LabelingSource`), from strongest to weakest:

1. **Deterministic** post-hoc checks (did the call raise, did a
   schema-validated result come back) — the strongest, cleanest signal.
2. **Human** review of a sample of past calls — reliable but expensive,
   and subject to reviewer disagreement/error.
3. **Downstream proxy** (did the overall task eventually succeed) — the
   weakest, most indirect signal; a downstream success can happen despite
   a locally bad call, and vice versa.

Every `GuaranteeStatement` carries its `labeling_source` explicitly. A
guarantee calibrated on human-labeled data should be read with more
confidence than one calibrated on a noisy downstream-success proxy, even
though the numeric `alpha` looks identical in both cases — the number
alone does not tell you this; the labeling source field does.

## Real coverage numbers

These are the actual outputs of `tests/coverage_validation/` and
`tests/negative_controls/` as run against this codebase, not hand-picked
or eyeballed. See `tests/coverage_validation/latest_report.json`,
`tests/coverage_validation/latest_real_data_report.json`, and
`tests/negative_controls/latest_report.json` for the underlying JSON (the
same data rendered as tables in the top-level README).

## Phase 4 (trajectory-level) — explicit non-guarantee

The one open question this library does not attempt to close in Phase 1:
whether any coverage guarantee can be established for a variable-length,
multi-step agent trajectory (as opposed to a single call). Prior art was
checked directly, not just assumed absent:

- PASC's own abstract asserts its max-nonconformity-score reduction
  extends to "agent pipelines," but its experiments never test this —
  they stop at a fixed, known `K` of simultaneous checks on one call,
  the same scope this library's Phase 2 already covers.
- TRACER (arXiv:2602.11409) was investigated specifically because it
  claims a trajectory-level guarantee, and found not to provide one.
  It is not conformal in the first place — its central object is a
  ranking/detection metric, not a calibrated nonconformity score — and
  its stated bound (Theorem A.8) depends on Assumption A.6
  (`λ_t ≤ c·r_t`), a claim about a *latent, unobservable* hazard rate.
  Unlike exchangeability, which can be sanity-checked empirically (e.g.
  via permutation tests on held-out data), this assumption cannot be
  checked against any observable quantity, so the "guarantee" built on
  top of it cannot itself be verified. That's a structural gap, not a
  minor caveat, and it's why TRACER is not cited anywhere in this
  codebase as a source of a real bound.

Any future trajectory-level module in this codebase must carry its own
explicit warning that its coverage numbers, if any, are empirical
observations, not a mathematical bound, until and unless a real theorem
is proven and reviewed the same way this document's claims were.
