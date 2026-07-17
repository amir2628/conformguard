"""Split-conformal calibration quantile.

Implements the quantile from Angelopoulos & Bates, "A Gentle Introduction
to Conformal Prediction and Distribution-Free Uncertainty Quantification"
(arXiv:2107.07511), Theorem 1:

    q_hat = Quantile({s_1, ..., s_n}, ceil((n+1)(1-alpha)) / n)

which the paper defines directly as the k-th smallest calibration score,
where k = ceil((n+1)(1-alpha)). Under exchangeability of the calibration
scores and the test-time score, this gives:

    P(s_new <= q_hat) >= 1 - alpha
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class InvalidAlphaError(ValueError):
    """Raised when alpha is not in the open interval (0, 1)."""


class EmptyCalibrationSetError(ValueError):
    """Raised when the calibration score set is empty."""


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """Compute the split-conformal calibration threshold q_hat.

    Args:
        scores: Nonconformity scores from the calibration set. Order does
            not matter; the function sorts internally.
        alpha: Target miscoverage rate, in (0, 1). The resulting q_hat
            satisfies P(s_new <= q_hat) >= 1 - alpha under exchangeability.

    Returns:
        q_hat, the k-th smallest calibration score, where
        k = ceil((n+1)(1-alpha)). If k exceeds n (alpha too small for the
        given calibration size), returns math.inf: the correction is
        mathematically valid at that alpha, but the resulting threshold
        accepts nothing, i.e. every call abstains. This is a real,
        checkable consequence of too little calibration data for the
        requested alpha, not an error to hide.

    Raises:
        InvalidAlphaError: if alpha is not strictly between 0 and 1.
        EmptyCalibrationSetError: if scores is empty.
    """
    if not 0.0 < alpha < 1.0:
        raise InvalidAlphaError(f"alpha must be in the open interval (0, 1), got {alpha!r}")
    n = len(scores)
    if n == 0:
        raise EmptyCalibrationSetError("cannot compute a conformal quantile from an empty calibration set")

    # k is a 1-indexed rank into the sorted scores. This is the exact
    # correction that makes the finite-sample guarantee hold: naively
    # calling numpy.quantile(scores, 1 - alpha) (or even numpy.quantile
    # with method="higher") is *not* equivalent to this k-th-order-statistic
    # definition -- numpy's virtual index for "higher" uses (n - 1) in the
    # denominator, not n, and is off by one from what Angelopoulos & Bates
    # define. We therefore compute the order statistic directly rather
    # than going through any quantile-interpolation routine.
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return math.inf

    sorted_scores = sorted(scores)
    return sorted_scores[k - 1]


def quantile_level(n: int, alpha: float) -> float:
    """Return the empirical quantile level ceil((n+1)(1-alpha))/n used by conformal_quantile.

    Exposed separately for logging/introspection (e.g. the CLI's threshold
    inspector) without recomputing the quantile itself.
    """
    if not 0.0 < alpha < 1.0:
        raise InvalidAlphaError(f"alpha must be in the open interval (0, 1), got {alpha!r}")
    if n <= 0:
        raise EmptyCalibrationSetError("n must be a positive number of calibration examples")
    k = math.ceil((n + 1) * (1.0 - alpha))
    return k / n
