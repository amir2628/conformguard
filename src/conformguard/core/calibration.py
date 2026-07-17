"""Building a Calibrator from scored, outcome-labeled historical calls.

The calibration set here is restricted to examples labeled ``outcome=True``
("this call, in hindsight, was correct/safe") before the quantile is
computed. This mirrors classical split conformal prediction, where the
calibration set is always ground-truth-correct (X_i, Y_i) pairs and the
quantile controls how often the *true* label's nonconformity score exceeds
the threshold. The analogous guarantee here is: if a new call is actually
a good call drawn from the same distribution as the calibration set's good
calls, it is wrongly abstained on at most alpha of the time. Examples
labeled ``outcome=False`` are recorded (for diagnostics and for whatever a
future score-quality tool wants) but are not fed into the quantile itself,
since Theorem 1's guarantee is about not wrongly rejecting a good example,
not about detecting a bad one -- see docs/guarantee_scope.md.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from conformguard.core.quantile import conformal_quantile
from conformguard.core.scores import NonconformityScore, ToolCallContext
from conformguard.storage.calibration_store import LabelingSource

# Hard floor: calibrate() raises below this many *good*-outcome examples,
# rather than silently producing a threshold from too little data. This is
# a policy choice, not a value derived from the math itself (any n >= 1
# gives a mathematically valid q_hat) -- the point is that a calibrator
# built on a handful of examples is not a weaker version of the guarantee,
# it is no guarantee worth trusting at all (see PROJECT_SPEC §10, "cold-start
# problem"). RECOMMENDED_MINIMUM_SIZE (storage/calibration_store.py) is the
# separate, higher, warn-only threshold for "tight" fluctuation bounds.
HARD_MINIMUM_SIZE = 100


class InsufficientCalibrationDataError(ValueError):
    """Raised when the number of good-outcome calibration examples is below HARD_MINIMUM_SIZE."""


class CalibrationScoringError(RuntimeError):
    """Raised when scoring a historical calibration example fails.

    Unlike a scorer failure at live decision time (which forces an
    abstain, since the guarantee's stakes are per-decision), a scoring
    failure on *historical* calibration data means the calibration set
    itself is broken -- it must be fixed, not silently dropped or
    papered over with an abstain that nobody would ever see.
    """


@dataclass(frozen=True)
class Calibrator:
    """A frozen split-conformal calibration result, ready for decisions.

    Construct via :func:`calibrate`, not directly.
    """

    scorer: NonconformityScore
    alpha: float
    q_hat: float
    n_calibration: int
    n_excluded: int
    tool_names: frozenset[str]
    context_bucket: str
    calibration_set_version: str
    calibration_start: datetime
    calibration_end: datetime
    labeling_source: LabelingSource
    created_at: datetime


def calibrate(
    scorer: NonconformityScore | Callable[[ToolCallContext], float],
    calibration_data: Sequence[tuple[ToolCallContext, bool]],
    alpha: float,
    *,
    labeling_source: LabelingSource = LabelingSource.DETERMINISTIC,
    context_bucket: str = "default",
    calibration_set_version: str | None = None,
    hard_minimum_size: int = HARD_MINIMUM_SIZE,
) -> Calibrator:
    """Build a Calibrator from historical (call, outcome) pairs.

    Args:
        scorer: A NonconformityScore, or a bare callable wrapped into one
            under the name "user_scorer".
        calibration_data: (ToolCallContext, outcome) pairs, where outcome
            is True iff the call was, in hindsight, correct/safe (§4.3).
            Only outcome=True examples are used to compute q_hat.
        alpha: Target miscoverage rate, in (0, 1).
        labeling_source: How outcome labels in this batch were produced.
            Applies uniformly to the whole batch -- if a calibration run
            mixes labeling sources, split it into separate calibrate()
            calls so each Calibrator's guarantee statement is honest
            about its own provenance.
        context_bucket: A caller-chosen tag identifying what this
            calibration applies to (e.g. "prod", "search-tools"). Purely
            descriptive; not enforced against calibration_data.
        calibration_set_version: Version tag for this calibration set.
            Defaults to an auto-generated timestamp-based tag.
        hard_minimum_size: Override for HARD_MINIMUM_SIZE (mainly for
            tests exercising the raise path without 100 examples).

    Raises:
        InsufficientCalibrationDataError: if the number of good-outcome
            examples is below hard_minimum_size.
        CalibrationScoringError: if the scorer raises while scoring any
            historical example.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha!r}")
    if not calibration_data:
        raise InsufficientCalibrationDataError(
            "calibration_data is empty; calibrate() requires at least "
            f"{hard_minimum_size} good-outcome historical examples"
        )

    wrapped_scorer = scorer if isinstance(scorer, NonconformityScore) else NonconformityScore("user_scorer", scorer)

    good_scores: list[float] = []
    tool_names: set[str] = set()
    timestamps: list[datetime] = []
    n_excluded = 0

    for index, (context, outcome) in enumerate(calibration_data):
        tool_names.add(context.tool_name)
        raw_timestamp = context.metadata.get("timestamp")
        if isinstance(raw_timestamp, datetime):
            timestamps.append(raw_timestamp)

        if not outcome:
            n_excluded += 1
            continue

        try:
            score = float(wrapped_scorer(context))
        except Exception as exc:
            raise CalibrationScoringError(
                f"scorer {wrapped_scorer.name!r} raised on calibration_data[{index}] "
                f"(tool_name={context.tool_name!r}): {exc!r}. Fix or remove this "
                f"historical example before calibrating -- a broken calibration "
                f"example must not be silently skipped."
            ) from exc
        good_scores.append(score)

    n_good = len(good_scores)
    if n_good < hard_minimum_size:
        raise InsufficientCalibrationDataError(
            f"only {n_good} good-outcome calibration examples available, below the "
            f"required minimum of {hard_minimum_size}. A calibrator built on too "
            f"little data is not a weaker guarantee, it is no guarantee at all -- "
            f"collect more labeled examples before calibrating."
        )

    q_hat = conformal_quantile(good_scores, alpha=alpha)

    now = datetime.now(timezone.utc)
    if timestamps:
        calibration_start, calibration_end = min(timestamps), max(timestamps)
    else:
        calibration_start = calibration_end = now

    version = calibration_set_version or f"auto-{now:%Y%m%dT%H%M%SZ}"

    return Calibrator(
        scorer=wrapped_scorer,
        alpha=alpha,
        q_hat=q_hat,
        n_calibration=n_good,
        n_excluded=n_excluded,
        tool_names=frozenset(tool_names),
        context_bucket=context_bucket,
        calibration_set_version=version,
        calibration_start=calibration_start,
        calibration_end=calibration_end,
        labeling_source=labeling_source,
        created_at=now,
    )
