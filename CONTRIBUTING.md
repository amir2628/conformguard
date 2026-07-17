# Contributing

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```
pytest tests/unit tests/integration          # fast, run on every commit
pytest tests/coverage_validation              # required before merging any change to
                                                # core/quantile.py, core/decision.py, or core/multi_check.py
pytest tests/negative_controls                 # required before any release
pytest -m live                                  # opt-in, requires a real model provider API key
```

## Changes to the statistical core

`core/quantile.py`, `core/decision.py`, and (once built) `core/multi_check.py`
are the mathematical core of this project. A bug in one of these files
isn't a functional regression — it's a false statistical claim. If you're
changing anything in this area:

1. Add or update the hand-computed toy examples in `tests/unit/test_quantile.py`
   *first*, worked out independently of the implementation, before changing
   the implementation itself.
2. Run `pytest tests/coverage_validation` and `pytest tests/negative_controls`
   and confirm they still pass with real numbers — not just that the fast
   unit tests are green.
3. If the change affects what the guarantee statement actually says
   (scope, assumptions, the formula itself), that's not a refactor, it's a
   new claim — update `docs/guarantee_scope.md` in the same change, and
   re-derive/re-verify the math the same way the original was checked
   (line-by-line against the cited theorem, not "looks right").

## Style

- No unnecessary abstractions; match the existing module boundaries in
  `docs/architecture.md` rather than introducing new ones for a single
  use site.
- Docstrings and comments explain *why*, not *what* — the code already
  says what it does.
- Every claim in a docstring, log line, or docs page that uses the word
  "guarantee" must carry its exact `alpha`, its scope (single-call vs.
  multi-check vs. trajectory), and the exchangeability assumption. This is
  enforced by convention, not by a linter — treat it as a hard rule when
  reviewing.

## Reporting issues

Open an issue with a minimal reproduction. For anything touching the
statistical core, include the calibration set size, alpha, and (if
possible) the score distribution you observed — "the guarantee didn't
hold" reports are only actionable with enough detail to reproduce via
`validation/coverage_check.py`.
