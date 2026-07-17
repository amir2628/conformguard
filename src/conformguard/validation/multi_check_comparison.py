"""Phase 2 comparison harness: joint (max-score) calibration vs. two alternatives.

Replicates PASC's own comparison methodology (Kotte et al., arXiv:2605.18812)
on this project's own tool-calling domain rather than citing their NER
numbers: for a fixed pool of K-dimensional nonconformity score vectors for
known-good historical calls (one vector per call), repeatedly split into
calibration/test sets and measure, for each of three methods:

- **validity**: observed coverage on held-out GOOD calls, against the
  shared target 1 - alpha.
- **efficiency**: rejection rate on a separate pool of BAD/anomalous score
  vectors, at the same alpha.

Both matter, and neither alone is the whole story. A method that just
accepts everything trivially "covers" 100% of good calls but rejects
nothing bad -- useless. A method that rejects everything trivially
catches all bad calls but covers no good ones -- also useless (and, if it
claims a coverage guarantee while doing this, invalid). The efficiency
metric is deliberately NOT "how often does this method accept a good
call" in isolation: an earlier version of this harness used exactly that
proxy and it was actively misleading -- Bonferroni's good-call acceptance
rate climbs *above* its own target as the K checks become more
correlated (because the union bound over-corrects less when the
checks' extreme-value events overlap), which looks like an advantage
but is really just a looser effective threshold that would also admit
more bad calls. Measuring rejection on an actual bad pool is what
surfaces that correctly -- see docs/real_world_validation.md for the
investigation this correction is based on.

Three methods compared:

- **joint** (core/multi_check.py's max-nonconformity-score reduction):
  a single q_hat computed over max(s_1, ..., s_K) across the calibration
  set. Has a valid joint-coverage guarantee (PASC Theorem 6).
- **naive**: each of the K checks calibrated independently at the FULL
  alpha, with no multiplicity correction; accept requires all K checks to
  individually pass their own threshold. Has NO valid joint guarantee --
  included specifically to demonstrate what that costs empirically.
- **bonferroni**: each of the K checks calibrated independently at
  alpha/K (the standard union-bound correction); accept requires all K to
  pass. Has a valid, but typically conservative, joint guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from conformguard.core.quantile import conformal_quantile


@dataclass(frozen=True)
class MultiCheckMethodResult:
    """One method's repeated-split result for a fixed (good pool, alpha, K), with optional bad-pool efficiency."""

    method: str
    alpha: float
    k: int
    n_calibration: int
    n_trials: int
    observed_good_coverages: tuple[float, ...]
    mean_good_coverage: float
    observed_bad_rejection_rates: tuple[float, ...] | None
    mean_bad_rejection_rate: float | None


def _joint_thresholds(cal: np.ndarray, alpha: float) -> float:
    return conformal_quantile(cal.max(axis=1).tolist(), alpha=alpha)


def _joint_accept(scores: np.ndarray, q_hat: float) -> np.ndarray:
    return scores.max(axis=1) <= q_hat


def _naive_thresholds(cal: np.ndarray, alpha: float) -> np.ndarray:
    k = cal.shape[1]
    return np.array([conformal_quantile(cal[:, j].tolist(), alpha=alpha) for j in range(k)])


def _bonferroni_thresholds(cal: np.ndarray, alpha: float) -> np.ndarray:
    k = cal.shape[1]
    alpha_per_check = alpha / k
    return np.array([conformal_quantile(cal[:, j].tolist(), alpha=alpha_per_check) for j in range(k)])


def _per_check_accept(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.all(scores <= thresholds, axis=1)


_THRESHOLD_FNS = {
    "joint": _joint_thresholds,
    "naive": _naive_thresholds,
    "bonferroni": _bonferroni_thresholds,
}


def _accept(method: str, scores: np.ndarray, thresholds) -> np.ndarray:
    if method == "joint":
        return _joint_accept(scores, thresholds)
    return _per_check_accept(scores, thresholds)


def run_multi_check_comparison(
    good_pool: np.ndarray,
    alpha: float,
    calibration_size: int,
    n_trials: int = 100,
    seed: int | None = None,
    bad_pool: np.ndarray | None = None,
) -> dict[str, MultiCheckMethodResult]:
    """Run all three methods over the same repeated calibration/test splits.

    Args:
        good_pool: shape (N, K) array of nonconformity score vectors for
            known-good historical calls (K simultaneous checks per call).
        alpha: target miscoverage rate for the good-call event.
        calibration_size: calibration set size per trial; the rest of
            good_pool is the held-out good test set for that trial.
        n_trials: number of repeated random splits (R). The SAME splits
            are used across all three methods (a paired comparison, not
            three independent runs), so differences between methods
            reflect the method, not sampling noise between them.
        seed: optional seed for reproducible splits.
        bad_pool: optional shape (M, K) array of nonconformity score
            vectors for known-bad/anomalous calls. If given, each
            trial's calibrated thresholds are also used to measure the
            rejection rate on this fixed pool -- the efficiency metric.
            If omitted, only the validity (good-coverage) metric is
            computed.

    Returns:
        {"joint": ..., "naive": ..., "bonferroni": ...}
    """
    n_pool, k = good_pool.shape
    if calibration_size <= 0 or calibration_size >= n_pool:
        raise ValueError(
            f"calibration_size must leave a non-empty test set: got calibration_size="
            f"{calibration_size} against a pool of {n_pool}"
        )
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    if bad_pool is not None and bad_pool.shape[1] != k:
        raise ValueError(f"bad_pool must have the same K={k} columns as good_pool, got {bad_pool.shape[1]}")

    rng = np.random.default_rng(seed)
    good_coverages: dict[str, list[float]] = {name: [] for name in _THRESHOLD_FNS}
    bad_rejections: dict[str, list[float]] = {name: [] for name in _THRESHOLD_FNS}

    for _ in range(n_trials):
        permuted = rng.permutation(n_pool)
        cal = good_pool[permuted[:calibration_size]]
        good_test = good_pool[permuted[calibration_size:]]

        for name, threshold_fn in _THRESHOLD_FNS.items():
            thresholds = threshold_fn(cal, alpha)
            good_coverages[name].append(float(np.mean(_accept(name, good_test, thresholds))))
            if bad_pool is not None:
                bad_rejections[name].append(float(np.mean(~_accept(name, bad_pool, thresholds))))

    results = {}
    for name in _THRESHOLD_FNS:
        has_bad = bad_pool is not None
        results[name] = MultiCheckMethodResult(
            method=name,
            alpha=alpha,
            k=k,
            n_calibration=calibration_size,
            n_trials=n_trials,
            observed_good_coverages=tuple(good_coverages[name]),
            mean_good_coverage=float(np.mean(good_coverages[name])),
            observed_bad_rejection_rates=tuple(bad_rejections[name]) if has_bad else None,
            mean_bad_rejection_rate=float(np.mean(bad_rejections[name])) if has_bad else None,
        )
    return results
