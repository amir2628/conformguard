"""Empirical coverage validation: the artifact that proves the math is right.

Per Angelopoulos & Bates §3.3, and the exact Beta-distribution result for
split conformal coverage (Vovk 2012; also given in Angelopoulos & Bates'
own appendix): conditional on a calibration draw of size n, and letting
k = ceil((n+1)(1-alpha)), the true (test-conditional) coverage

    C = P(s_new <= q_hat | calibration set)

is itself a random variable, distributed as

    C ~ Beta(k, n - k + 1)

This is an exact result for continuous, exchangeable scores (no ties),
derived from the classical fact that F(S) for order statistics of iid
continuous random variables are themselves Beta-distributed order
statistics of Uniform(0,1). It gives a theoretically-predicted band for
observed coverage across repeated calibration/test splits -- not an
eyeballed one -- which is what this module checks empirical coverage
against.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats

from conformguard.core.quantile import conformal_quantile


@dataclass(frozen=True)
class CoverageBand:
    """Theoretically-predicted fluctuation band for observed coverage."""

    n: int
    alpha: float
    k: int
    mean: float
    std: float
    low: float
    high: float
    confidence: float


def theoretical_coverage_band(n: int, alpha: float, confidence: float = 0.95) -> CoverageBand:
    """Compute the exact Beta-distribution coverage band for calibration size n at level alpha.

    Raises:
        ValueError: if alpha/n are out of range, or if k = ceil((n+1)(1-alpha))
            exceeds n -- in that regime q_hat is +inf and coverage is
            deterministically 1.0 (not usefully described by a fluctuation
            band), which callers should special-case rather than treat as
            a Beta-distributed quantity.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha!r}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n!r}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in the open interval (0, 1), got {confidence!r}")

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        raise ValueError(
            f"k=ceil((n+1)(1-alpha))={k} exceeds n={n}: q_hat is deterministically "
            f"+inf at this (n, alpha), so coverage is deterministically 1.0, not a "
            f"Beta-distributed random variable. Use a larger n or larger alpha."
        )

    a, b = k, n - k + 1
    dist = stats.beta(a, b)
    tail = (1.0 - confidence) / 2.0
    return CoverageBand(
        n=n,
        alpha=alpha,
        k=k,
        mean=float(dist.mean()),
        std=float(dist.std()),
        low=float(dist.ppf(tail)),
        high=float(dist.ppf(1.0 - tail)),
        confidence=confidence,
    )


@dataclass(frozen=True)
class CoverageValidationResult:
    """Result of R repeated calibration/test splits on a fixed score pool."""

    n_calibration: int
    alpha: float
    n_trials: int
    observed_coverages: tuple[float, ...]
    mean_observed_coverage: float
    band: CoverageBand

    @property
    def within_band(self) -> bool:
        return self.band.low <= self.mean_observed_coverage <= self.band.high

    def histogram(self, bins: int = 10) -> tuple[list[int], list[float]]:
        counts, edges = np.histogram(self.observed_coverages, bins=bins, range=(0.0, 1.0))
        return counts.tolist(), edges.tolist()


def run_coverage_validation(
    pool: Sequence[float],
    alpha: float,
    calibration_size: int,
    n_trials: int = 100,
    seed: int | None = None,
) -> CoverageValidationResult:
    """Run R repeated random calibration/test splits and measure observed coverage.

    Args:
        pool: A fixed pool of nonconformity scores for known-good calls,
            large enough to repeatedly split into a calibration set of
            ``calibration_size`` and a disjoint test set of the remainder.
        alpha: Target miscoverage rate.
        calibration_size: Number of pool elements used as the calibration
            set on each trial; the rest of the pool is the test set for
            that trial.
        n_trials: Number of repeated splits (R). Per PROJECT_SPEC §7.3,
            R >= 100 is required before trusting the resulting numbers.
        seed: Optional seed for reproducible splits.

    Returns:
        A CoverageValidationResult whose ``within_band`` property is the
        pass/fail signal the test suite asserts on.
    """
    pool_array = np.asarray(pool, dtype=float)
    n_pool = len(pool_array)
    if calibration_size <= 0 or calibration_size >= n_pool:
        raise ValueError(
            f"calibration_size must leave a non-empty test set: got calibration_size="
            f"{calibration_size} against a pool of {n_pool}"
        )
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    rng = np.random.default_rng(seed)
    observed: list[float] = []

    for _ in range(n_trials):
        permuted = rng.permutation(n_pool)
        calibration_indices = permuted[:calibration_size]
        test_indices = permuted[calibration_size:]

        calibration_scores = pool_array[calibration_indices].tolist()
        test_scores = pool_array[test_indices]

        q_hat = conformal_quantile(calibration_scores, alpha=alpha)
        covered = np.mean(test_scores <= q_hat)
        observed.append(float(covered))

    band = theoretical_coverage_band(n=calibration_size, alpha=alpha)

    return CoverageValidationResult(
        n_calibration=calibration_size,
        alpha=alpha,
        n_trials=n_trials,
        observed_coverages=tuple(observed),
        mean_observed_coverage=float(np.mean(observed)),
        band=band,
    )
