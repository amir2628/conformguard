# Writing nonconformity scores

A nonconformity score is any function `ToolCallContext -> float` that is
**low when a call looks trustworthy and high when it looks risky**. That's
the entire interface contract. Three built-ins ship with the library
(`logprob_score`, `make_judge_score`, `schema_validity_score`), but a
user-supplied callable is the general case, not a fallback.

## The one thing a score's quality does NOT affect

The coverage guarantee (`P(wrongly abstain) <= alpha`) holds **regardless
of how good or bad the score is at telling risky calls from safe ones**.
This is one of conformal prediction's more surprising properties, and it's
worth internalizing before you spend time tuning a score: a bad score
does not produce a false guarantee. What it produces is a *useless* one —
either constant abstention (if the score has no real signal and just
happens to spread wide) or an accept-everything threshold (if the score
doesn't separate risky from safe calls at all). See
`docs/guarantee_scope.md`'s "Score quality vs. guarantee validity"
section.

## How to judge a score, then

Not by intuition, and not by the coverage number (which looks fine either
way — see above). Judge it by running
`validation.coverage_check.run_coverage_validation()` on your own
calibration pool and looking at:

- **Abstention rate** at your chosen `alpha`. If almost everything
  abstains, the score's "risky" tail is too heavy, or too many good calls
  score like risky ones.
- **Separation** between the score distribution on `outcome=True` calls
  and `outcome=False` calls (even though only the `True` ones feed
  `q_hat`, comparing the two distributions tells you whether the score
  carries any signal at all).

## Interface

```python
from conformguard import NonconformityScore, ToolCallContext

def my_score(context: ToolCallContext) -> float:
    # context.tool_name: str
    # context.args: dict[str, Any]
    # context.result: Any | None      -- set if scoring after execution
    # context.metadata: dict[str, Any] -- your extension point
    ...

scorer = NonconformityScore(name="my_score", fn=my_score)
```

Raise inside `fn` (or return `float("inf")`/`float("nan")`) for "I can't
score this call" — `NonconformityScore.safe()` converts that into a
forced-abstain outcome automatically, both at calibration time (where it
becomes a loud `CalibrationScoringError`, since broken historical data
needs fixing, not silent skipping) and at decision time (where it becomes
an abstain, never a silent accept). You do not need to handle this
yourself; just let the exception propagate or return a non-finite value
when you genuinely have no opinion.

## Built-in scorers

### `logprob_score`

```python
from conformguard import logprob_score, ToolCallContext

context = ToolCallContext(
    tool_name="search",
    args={"query": "..."},
    metadata={"model_logprob": -0.12},
)
logprob_score(context)  # 1 - exp(-0.12)
```

Nonconformity = `1 - exp(model_logprob)`. Requires
`context.metadata["model_logprob"]`; raises (and therefore forces
abstain) if it's missing. Useful when your model-calling code already
surfaces token log-probabilities for the generated tool call.

### `make_judge_score`

```python
from conformguard import make_judge_score

def my_judge(context) -> float:
    # Call a cheap secondary model, return a plausibility estimate in [0, 1].
    ...
    return plausibility

scorer = make_judge_score(my_judge)
```

Deliberately provider-agnostic: this library does not make a hardcoded
API call on your behalf. The judge call is a heuristic *input* to
calibration — calibration is what turns it into a checkable guarantee,
not the judge call itself. `judge_fn` must return a value in `[0, 1]`;
the resulting nonconformity score is `1 - plausibility`.

### `schema_validity_score`

```python
from pydantic import BaseModel
from conformguard import schema_validity_score, ToolCallContext

class SearchArgs(BaseModel):
    query: str
    max_results: int = 10

context = ToolCallContext(
    tool_name="search",
    args={"query": "weather"},
    metadata={"schema": SearchArgs},
)
schema_validity_score(context)  # 0.0 if args validate against SearchArgs, else 1.0
```

Deterministic and free (no model call). Requires
`context.metadata["schema"]` to be a pydantic `BaseModel` subclass.

## A realistic composite example

```python
import math
from conformguard import NonconformityScore, ToolCallContext

def composite_score(context: ToolCallContext) -> float:
    schema_penalty = 1.0 if _looks_malformed(context.args) else 0.0
    logprob = context.metadata.get("model_logprob")
    confidence_penalty = (1 - math.exp(logprob)) if logprob is not None else 0.5
    return 0.6 * schema_penalty + 0.4 * confidence_penalty

scorer = NonconformityScore(name="composite_score", fn=composite_score)
```

Phase 1 supports exactly one score per call; combining multiple signals
into one scalar (as above) is the Phase 1 pattern. Genuinely joint,
multi-check calibration with its own coverage guarantee per check (PASC's
max-nonconformity-score reduction) is `core/multi_check.py`, planned for
Phase 2 — see `docs/architecture.md`.
