"""Phase 2: joint calibration across K simultaneous nonconformity checks on one call.

Implements PASC's (Kotte et al., arXiv:2605.18812) max-nonconformity-score
reduction (their Algorithm 1 / Theorem 6, cited and verified in
PROJECT_SPEC.md §1.1): given K nonconformity scores s_1, ..., s_K on one
call, reduce them to a single score per calibration example by taking
their max, then apply Angelopoulos & Bates' Theorem 1 quantile formula
(core/quantile.py) to that max-score calibration set. The resulting
single threshold q_hat, compared against max(s_1, ..., s_K) of a new
call, gives:

    P(ALL K checks pass simultaneously) >= 1 - alpha

This is a genuine JOINT/simultaneous coverage guarantee across all K
checks at once -- not K separate marginal guarantees, and not the same
thing as calibrating each check independently at level alpha (that
approach, "naive independent calibration", has no joint guarantee at
all: see validation/coverage_check.py's comparison harness for exactly
how much that gap costs in practice, and docs/guarantee_scope.md for why
it's wrong to assume the K individual guarantees compose).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict

from conformguard.core.calibration import CalibrationScoringError, InsufficientCalibrationDataError
from conformguard.core.decision import Decision
from conformguard.core.quantile import conformal_quantile
from conformguard.core.scores import NonconformityScore, ToolCallContext
from conformguard.storage.calibration_store import LabelingSource

# Same rationale as core/calibration.py's HARD_MINIMUM_SIZE: a policy
# choice, not derived from the math, and applied identically here so a
# multi-check calibrator isn't held to a lower bar than a single-check one.
HARD_MINIMUM_SIZE = 100


@dataclass(frozen=True)
class MultiCheckCalibrator:
    """A frozen joint-calibration result for K simultaneous checks. Construct via :func:`calibrate_multi_check`."""

    scorers: tuple[NonconformityScore, ...]
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

    @property
    def k(self) -> int:
        return len(self.scorers)

    @property
    def check_names(self) -> tuple[str, ...]:
        return tuple(scorer.name for scorer in self.scorers)


def calibrate_multi_check(
    scorers: Sequence[NonconformityScore | Callable[[ToolCallContext], float]],
    calibration_data: Sequence[tuple[ToolCallContext, bool]],
    alpha: float,
    *,
    labeling_source: LabelingSource = LabelingSource.DETERMINISTIC,
    context_bucket: str = "default",
    calibration_set_version: str | None = None,
    hard_minimum_size: int = HARD_MINIMUM_SIZE,
) -> MultiCheckCalibrator:
    """Build a MultiCheckCalibrator via the max-nonconformity-score reduction.

    For each good-outcome (``outcome=True``) calibration example, computes
    all K scores and takes their max; q_hat is the standard split-conformal
    quantile (core/quantile.py) of that max-score set. See core/calibration.py's
    module docstring for why only ``outcome=True`` examples feed the quantile
    -- the same reasoning applies here, per-check-max instead of per-check.

    Args:
        scorers: At least 2 NonconformityScore instances (or bare
            callables, wrapped under names "check_0", "check_1", ...).
            Use core.calibration.calibrate() instead for a single check.
        calibration_data: (ToolCallContext, outcome) pairs, same contract
            as core.calibration.calibrate().
        alpha: Target joint miscoverage rate, in (0, 1).

    Raises:
        ValueError: if fewer than 2 scorers are given, or alpha is out of range.
        InsufficientCalibrationDataError: if the number of good-outcome
            examples is below hard_minimum_size.
        CalibrationScoringError: if any scorer raises while scoring any
            historical example.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha!r}")
    if len(scorers) < 2:
        raise ValueError(
            f"calibrate_multi_check requires at least 2 simultaneous checks, got {len(scorers)}; "
            f"use core.calibration.calibrate() for a single check"
        )
    if not calibration_data:
        raise InsufficientCalibrationDataError(
            "calibration_data is empty; calibrate_multi_check() requires at least "
            f"{hard_minimum_size} good-outcome historical examples"
        )

    wrapped_scorers = tuple(
        scorer if isinstance(scorer, NonconformityScore) else NonconformityScore(f"check_{i}", scorer)
        for i, scorer in enumerate(scorers)
    )

    max_scores: list[float] = []
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

        per_check_scores = []
        for scorer in wrapped_scorers:
            try:
                per_check_scores.append(float(scorer(context)))
            except Exception as exc:
                raise CalibrationScoringError(
                    f"scorer {scorer.name!r} raised on calibration_data[{index}] "
                    f"(tool_name={context.tool_name!r}): {exc!r}. Fix or remove this "
                    f"historical example before calibrating -- a broken calibration "
                    f"example must not be silently skipped."
                ) from exc
        max_scores.append(max(per_check_scores))

    n_good = len(max_scores)
    if n_good < hard_minimum_size:
        raise InsufficientCalibrationDataError(
            f"only {n_good} good-outcome calibration examples available, below the "
            f"required minimum of {hard_minimum_size}. A calibrator built on too "
            f"little data is not a weaker guarantee, it is no guarantee at all -- "
            f"collect more labeled examples before calibrating."
        )

    q_hat = conformal_quantile(max_scores, alpha=alpha)

    now = datetime.now(timezone.utc)
    if timestamps:
        calibration_start, calibration_end = min(timestamps), max(timestamps)
    else:
        calibration_start = calibration_end = now

    version = calibration_set_version or f"auto-{now:%Y%m%dT%H%M%SZ}"

    return MultiCheckCalibrator(
        scorers=wrapped_scorers,
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


class CheckResult(BaseModel):
    """One check's contribution to a multi-check decision."""

    model_config = ConfigDict(frozen=True)

    name: str
    score: float
    errored: bool
    error: str | None = None
    passed: bool


class MultiCheckGuaranteeStatement(BaseModel):
    """The exact, scoped guarantee statement for a joint, K-check decision."""

    model_config = ConfigDict(frozen=True)

    alpha: float
    scope: str = "multi_check"
    k: int
    check_names: tuple[str, ...]
    calibration_set_size: int
    calibration_set_version: str
    calibration_start: datetime
    calibration_end: datetime
    labeling_source: LabelingSource
    exchangeability_assumed: bool = True
    text: str


class MultiCheckWrapResult(BaseModel):
    """The full outcome of one joint, K-check accept/abstain decision."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    decision: Decision
    max_score: float
    threshold: float
    checks: tuple[CheckResult, ...]
    guarantee: MultiCheckGuaranteeStatement

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.ACCEPT

    @property
    def failed_checks(self) -> tuple[str, ...]:
        """Names of checks that individually exceeded q_hat (or errored) -- the per-check breakdown."""
        return tuple(c.name for c in self.checks if not c.passed)


def build_multi_check_guarantee_statement(calibrator: MultiCheckCalibrator) -> MultiCheckGuaranteeStatement:
    """Construct the exact, scoped guarantee statement for a MultiCheckCalibrator."""
    pct = (1 - calibrator.alpha) * 100
    error_pct = calibrator.alpha * 100
    start = calibrator.calibration_start.date().isoformat()
    end = calibrator.calibration_end.date().isoformat()
    checks_list = ", ".join(calibrator.check_names)

    text = (
        f"Under the assumption that this call is exchangeable with the "
        f"{calibrator.n_calibration}-example calibration set (labeling source: "
        f"{calibrator.labeling_source.value}; collected {start} through {end}), "
        f"the joint event 'all {calibrator.k} checks pass' ({checks_list}) is wrong at "
        f"most {error_pct:.1f}% of the time (alpha={calibrator.alpha}), for THIS SINGLE "
        f"CALL'S {calibrator.k} SIMULTANEOUS CHECKS ONLY. This is a joint guarantee across "
        f"exactly these {calibrator.k} checks via the max-nonconformity-score reduction "
        f"(PASC, arXiv:2605.18812, Theorem 6) -- it is not a guarantee about any other "
        f"check, any multi-step task this call is part of, and it holds only if the "
        f"exchangeability assumption is not violated -- see docs/guarantee_scope.md. "
        f"[{pct:.1f}% target joint coverage]"
    )

    return MultiCheckGuaranteeStatement(
        alpha=calibrator.alpha,
        scope="multi_check",
        k=calibrator.k,
        check_names=calibrator.check_names,
        calibration_set_size=calibrator.n_calibration,
        calibration_set_version=calibrator.calibration_set_version,
        calibration_start=calibrator.calibration_start,
        calibration_end=calibrator.calibration_end,
        labeling_source=calibrator.labeling_source,
        exchangeability_assumed=True,
        text=text,
    )


def decide_multi_check(calibrator: MultiCheckCalibrator, context: ToolCallContext) -> MultiCheckWrapResult:
    """Score ``context`` against all of ``calibrator``'s checks and produce a joint accept/abstain decision.

    Every check is compared against the SAME threshold ``calibrator.q_hat``
    (that's what the max-score reduction buys: one threshold, applied
    uniformly, rather than K separately-calibrated ones). Accept requires
    every check to be usable (no scorer error, no non-finite score --
    exactly core/decision.py's "never accept without a score" rule,
    applied per-check) AND every check's score to be <= q_hat. Any single
    check failing (or erroring) forces abstain for the whole call, and
    ``.failed_checks`` on the result reports which one(s).
    """
    checks: list[CheckResult] = []
    all_usable = True

    for scorer in calibrator.scorers:
        outcome = scorer.safe(context)
        if not outcome.usable:
            all_usable = False
        checks.append(
            CheckResult(
                name=scorer.name,
                score=outcome.score,
                errored=outcome.errored,
                error=repr(outcome.error) if outcome.error is not None else None,
                passed=outcome.usable and outcome.score <= calibrator.q_hat,
            )
        )

    max_score = max(check.score for check in checks)
    decision = Decision.ACCEPT if (all_usable and max_score <= calibrator.q_hat) else Decision.ABSTAIN

    return MultiCheckWrapResult(
        tool_name=context.tool_name,
        decision=decision,
        max_score=max_score,
        threshold=calibrator.q_hat,
        checks=tuple(checks),
        guarantee=build_multi_check_guarantee_statement(calibrator),
    )


__all__ = [
    "HARD_MINIMUM_SIZE",
    "CheckResult",
    "MultiCheckCalibrator",
    "MultiCheckGuaranteeStatement",
    "MultiCheckWrapResult",
    "build_multi_check_guarantee_statement",
    "calibrate_multi_check",
    "decide_multi_check",
]
