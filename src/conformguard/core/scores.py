"""Nonconformity score interface and built-in scorers.

A nonconformity score is any function that maps a tool call to a number
that is low when the call looks trustworthy and high when it looks risky.
The scoring function's quality never affects whether the coverage
guarantee holds (Theorem 1 holds regardless of the score's usefulness);
it only affects how *useful* an abstain-or-accept decision is. See
docs/writing_scorers.md.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCallContext(BaseModel):
    """Everything a scorer needs to judge one tool call.

    ``metadata`` is the extension point for scorer-specific inputs (a
    model log-probability, an argument schema, a judge callable's raw
    output, ...) so the built-in scorers and user scorers share one
    context shape instead of each defining their own call signature.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoreOutcome(BaseModel):
    """Result of safely invoking a scorer: either a finite score, or a forced abstain."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: float
    errored: bool
    error: BaseException | None = None

    @property
    def usable(self) -> bool:
        """True only for a finite score produced without error.

        A caller must never treat an errored or non-finite score as
        acceptable input to an accept decision -- both cases are forced
        abstains at the decision layer, regardless of where the threshold
        happens to fall (see core/decision.py).
        """
        return not self.errored and math.isfinite(self.score)


class NonconformityScore:
    """Wraps a raw scoring callable with a name and error-safe invocation.

    ``fn`` receives a single ``ToolCallContext`` and must return a float
    (higher = more suspicious). Any exception raised by ``fn``, or any
    non-finite value it returns, is converted into a forced-abstain
    outcome by :meth:`safe` rather than propagated -- a scorer failure
    must never be interpreted as "no opinion, so accept."
    """

    def __init__(self, name: str, fn: Callable[[ToolCallContext], float]):
        self.name = name
        self.fn = fn

    def __call__(self, context: ToolCallContext) -> float:
        return float(self.fn(context))

    def safe(self, context: ToolCallContext) -> ScoreOutcome:
        try:
            value = float(self.fn(context))
        except Exception as exc:  # deliberate: any scorer failure must force an abstain, never a silent accept
            return ScoreOutcome(score=math.inf, errored=True, error=exc)
        if not math.isfinite(value):
            return ScoreOutcome(score=math.inf, errored=False, error=None)
        return ScoreOutcome(score=value, errored=False, error=None)

    def __repr__(self) -> str:
        return f"NonconformityScore(name={self.name!r})"


def _logprob_score_fn(context: ToolCallContext) -> float:
    if "model_logprob" not in context.metadata:
        raise KeyError(
            "logprob_score requires context.metadata['model_logprob'] "
            "(the model's log-probability for the generated call)"
        )
    logprob = context.metadata["model_logprob"]
    return 1.0 - math.exp(logprob)


logprob_score = NonconformityScore(name="logprob_score", fn=_logprob_score_fn)
"""Built-in scorer: nonconformity = 1 - exp(model_logprob).

Requires ``context.metadata["model_logprob"]``. Missing or malformed
input raises inside ``fn``, which ``safe()`` converts into a forced
abstain rather than a silent accept.
"""


def make_judge_score(judge_fn: Callable[[ToolCallContext], float], name: str = "judge_score") -> NonconformityScore:
    """Build a judge-model scorer from a user-supplied plausibility function.

    ``judge_fn`` must return a plausibility estimate in [0, 1] (e.g. from
    a cheap secondary model call). The resulting nonconformity score is
    ``1 - plausibility``. This is intentionally provider-agnostic: this
    library does not make a hardcoded API call on the user's behalf,
    since the judge is a heuristic input to calibration, not a bypass of
    it -- the calibration step is what turns it into a checkable
    guarantee, not the judge call itself.
    """

    def _fn(context: ToolCallContext) -> float:
        plausibility = float(judge_fn(context))
        if not 0.0 <= plausibility <= 1.0:
            raise ValueError(f"judge_fn must return a plausibility in [0, 1], got {plausibility!r}")
        return 1.0 - plausibility

    return NonconformityScore(name=name, fn=_fn)


def _schema_validity_score_fn(context: ToolCallContext) -> float:
    if "schema" not in context.metadata:
        raise KeyError(
            "schema_validity_score requires context.metadata['schema'] "
            "(a pydantic BaseModel subclass describing valid arguments)"
        )
    schema = context.metadata["schema"]
    try:
        schema(**context.args)
    except Exception:
        return 1.0
    return 0.0


schema_validity_score = NonconformityScore(name="schema_validity_score", fn=_schema_validity_score_fn)
"""Built-in scorer: deterministic argument-schema conformance.

Requires ``context.metadata["schema"]`` to be a pydantic ``BaseModel``
subclass. Returns 0.0 (fully conforming) if ``schema(**context.args)``
validates, 1.0 otherwise. Missing schema raises, which ``safe()``
converts into a forced abstain.
"""
