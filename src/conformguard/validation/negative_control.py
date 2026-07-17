"""Negative-control / permutation tests: proving the validator isn't lying.

Modeled on PASC's own sanity check (arXiv:2605.18812, Appendix A.1): inject
synthetic non-exchangeability between the calibration set and the test set,
then confirm that the coverage-validation machinery in validation/coverage_check.py
correctly reports *degraded* coverage rather than a falsely reassuring
number that happens to still land inside the theoretical band. This module
is the statistics equivalent of "never fail silently" -- it exists to prove
the library can't be fooled into reporting a guarantee that doesn't hold.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from conformguard.core.quantile import conformal_quantile
from conformguard.validation.coverage_check import CoverageBand, theoretical_coverage_band

ShiftFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def constant_shift(amount: float) -> ShiftFn:
    """Shift every test-time score up by a fixed amount (calls look systematically riskier)."""

    def _shift(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return scores + amount

    return _shift


def distribution_swap(other_pool: Sequence[float]) -> ShiftFn:
    """Replace the test-time scores with a same-size draw from a different pool entirely.

    Models a deployment distribution that has drifted to something
    structurally different from the calibration distribution (the
    exchangeability assumption's real-world failure mode), rather than
    just a uniform shift.
    """
    other_array = np.asarray(other_pool, dtype=float)

    def _shift(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(other_array, size=len(scores), replace=True)

    return _shift


def no_shift() -> ShiftFn:
    """Identity shift: the negative control's own negative control.

    If ``run_negative_control`` reported "degraded" under a no-op shift,
    that would mean the detector cries wolf on perfectly exchangeable
    data -- this shift function exists so a test can rule that out.
    """

    def _shift(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return scores

    return _shift


@dataclass(frozen=True)
class NegativeControlResult:
    """Result of repeated calibration/test splits where the test set was deliberately shifted."""

    alpha: float
    n_calibration: int
    n_trials: int
    observed_coverages: tuple[float, ...]
    mean_observed_coverage: float
    band: CoverageBand
    """The band that WOULD apply if the test set were still exchangeable
    with the calibration set -- i.e. what a well-behaved run should look
    like. Under a real shift, mean_observed_coverage should fall visibly
    below band.low.
    """

    @property
    def degraded(self) -> bool:
        """True iff mean observed coverage falls below the exchangeable-case band.

        This is the pass condition for a negative-control test: a
        deliberately broken exchangeability assumption MUST produce
        degraded coverage, or the validation harness itself would be
        capable of papering over a false guarantee.
        """
        return self.mean_observed_coverage < self.band.low


def run_negative_control(
    pool: Sequence[float],
    alpha: float,
    calibration_size: int,
    shift_fn: ShiftFn,
    n_trials: int = 100,
    seed: int | None = None,
) -> NegativeControlResult:
    """Like coverage_check.run_coverage_validation, but the test set is deliberately shifted.

    On each of ``n_trials`` splits, the calibration set is drawn unshifted
    from ``pool``; the disjoint test set is drawn from ``pool`` and then
    passed through ``shift_fn``, breaking exchangeability between the two
    by construction. The theoretical band attached to the result is the
    band that *would* hold if the test set were exchangeable -- comparing
    observed coverage against it is what makes the degradation visible and
    checkable, not just asserted.
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
        test_scores = shift_fn(pool_array[test_indices], rng)

        q_hat = conformal_quantile(calibration_scores, alpha=alpha)
        covered = np.mean(test_scores <= q_hat)
        observed.append(float(covered))

    band = theoretical_coverage_band(n=calibration_size, alpha=alpha)

    return NegativeControlResult(
        alpha=alpha,
        n_calibration=calibration_size,
        n_trials=n_trials,
        observed_coverages=tuple(observed),
        mean_observed_coverage=float(np.mean(observed)),
        band=band,
    )


def run_multi_check_negative_control(
    good_pool: np.ndarray,
    alpha: float,
    calibration_size: int,
    shift_column: int,
    shift_fn: ShiftFn,
    n_trials: int = 100,
    seed: int | None = None,
) -> NegativeControlResult:
    """Like run_negative_control, but for joint (max-score) multi-check calibration.

    Unlike test_multi_check_exchangeability_violation_detected.py's first
    version (which shifted the already-max-reduced 1-D score -- a
    different, weaker check), this operates on the K-dimensional pool
    directly: on each trial, the calibration set's max-scores are computed
    from the UNSHIFTED K-dimensional calibration rows; the disjoint test
    set has ONLY column ``shift_column`` shifted (via ``shift_fn``)
    *before* taking the row-wise max, leaving the other K-1 checks
    untouched. This tests specifically whether a single check drifting
    out of distribution -- while the others stay exchangeable -- is still
    caught by the joint coverage-validation machinery, rather than being
    diluted away by the other, still-exchangeable checks in the max.

    Args:
        good_pool: shape (N, K) array of known-good nonconformity score
            vectors.
        shift_column: index (0-based) of the single check to shift in the
            test set only.
        shift_fn: applied to that one column's test-set values (same
            shift functions as run_negative_control: constant_shift,
            distribution_swap, no_shift).
    """
    n_pool, k = good_pool.shape
    if not 0 <= shift_column < k:
        raise ValueError(f"shift_column must be in [0, {k}), got {shift_column}")
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
        cal = good_pool[permuted[:calibration_size]]
        test = good_pool[permuted[calibration_size:]].copy()

        q_hat = conformal_quantile(cal.max(axis=1).tolist(), alpha=alpha)

        test[:, shift_column] = shift_fn(test[:, shift_column], rng)
        covered = np.mean(test.max(axis=1) <= q_hat)
        observed.append(float(covered))

    band = theoretical_coverage_band(n=calibration_size, alpha=alpha)

    return NegativeControlResult(
        alpha=alpha,
        n_calibration=calibration_size,
        n_trials=n_trials,
        observed_coverages=tuple(observed),
        mean_observed_coverage=float(np.mean(observed)),
        band=band,
    )
