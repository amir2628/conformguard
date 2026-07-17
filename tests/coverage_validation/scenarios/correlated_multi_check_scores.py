"""Synthetic K-dimensional nonconformity score pools with a controllable pairwise correlation.

Used by the Phase 2 multi-check comparison harness. Synthetic, and
deliberately so: the comparison's whole point is to demonstrate how the
joint (max-score) method's advantage over Bonferroni depends on how
correlated the K checks are -- a property that can only be cleanly
demonstrated by controlling the correlation directly, which real,
already-recorded data does not let you do. See
tests/coverage_validation/scenarios/real_bfcl_multi_check_scores.py for a
supplementary real-data comparison, whose correlation is whatever it
naturally is (uncontrolled, and disclosed as such).

Generated via a Gaussian copula: draw from a K-dimensional multivariate
normal with the requested pairwise correlation, transform to Uniform(0,1)
marginals via the normal CDF, then to Beta(2, 5) marginals via the Beta
quantile function -- the same "most calls score confidently, some don't"
shape used by scenarios/skewed_confidence_scores.py in Phase 1's own
coverage-validation suite, extended here to K correlated dimensions
instead of one.
"""

import numpy as np
from scipy import stats

SCENARIO_NAME = "correlated_multi_check_scores"


def _correlated_beta_pool(size: int, seed: int, k: int, rho: float, a: float, b: float) -> np.ndarray:
    if not -1.0 < rho < 1.0:
        raise ValueError(f"rho must be in (-1, 1), got {rho!r}")
    rng = np.random.default_rng(seed)
    cov = np.full((k, k), rho)
    np.fill_diagonal(cov, 1.0)
    z = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=size)
    u = stats.norm.cdf(z)
    return stats.beta.ppf(u, a, b)


def generate_pool(size: int, seed: int, k: int = 3, rho: float = 0.0) -> np.ndarray:
    """Return an (size, k) array of correlated Beta(2, 5)-marginal nonconformity scores.

    Represents known-GOOD calls: most score low (confident/conforming),
    a minority score higher, matching the shape used throughout this
    project's other coverage-validation scenarios.

    Args:
        size: number of (K-dimensional) calibration examples.
        seed: RNG seed.
        k: number of simultaneous checks.
        rho: pairwise correlation between every pair of the K checks'
            underlying Gaussian copula (not the exact correlation of the
            resulting Beta marginals, which is a monotonic but not
            identical function of rho -- close enough for this harness's
            purpose of sweeping "independent" through "highly correlated").
    """
    return _correlated_beta_pool(size, seed, k, rho, a=2, b=5)


def generate_bad_pool(size: int, seed: int, k: int = 3, rho: float = 0.0) -> np.ndarray:
    """Return an (size, k) array of correlated Beta(5, 2)-marginal nonconformity scores.

    Represents known-BAD/anomalous calls: the mirror image of
    generate_pool's shape (most score HIGH, a minority score lower), used
    only to measure each calibration method's efficiency (rejection rate
    on genuinely bad calls) -- never fed into calibration itself, which
    only ever sees known-good examples (see core/multi_check.py's module
    docstring for why).
    """
    return _correlated_beta_pool(size, seed, k, rho, a=5, b=2)
