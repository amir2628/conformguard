"""Empirical coverage validation for Phase 2's joint (max-score) calibration.

The max-nonconformity-score reduction (core/multi_check.py) turns a
K-dimensional joint-calibration problem into an ordinary 1-D
split-conformal quantile problem on max(s_1, ..., s_K) -- so Phase 1's
own coverage-validation machinery (validation/coverage_check.py) applies
here directly and exactly, with no new validation code needed: reduce
each calibration example's K scores to their max, then run the same
R-repeated-split procedure and check against the same exact
Beta(k, n-k+1) theoretical band used throughout Phase 1.
"""

from __future__ import annotations

import json
from pathlib import Path

from conformguard.validation.coverage_check import run_coverage_validation
from tests.coverage_validation.scenarios import correlated_multi_check_scores

CALIBRATION_SIZE = 1000
N_TRIALS = 100
SEED = 20260701
ALPHAS = [0.1, 0.05]
K = 3
RHOS = [0.0, 0.5, 0.9]

REPORT_PATH = Path(__file__).parent / "latest_multi_check_coverage_report.json"


def _max_reduced_pool(rho: float) -> list[float]:
    matrix = correlated_multi_check_scores.generate_pool(size=20_000, seed=SEED, k=K, rho=rho)
    return matrix.max(axis=1).tolist()


def test_joint_calibration_coverage_within_theoretical_band_across_rho_and_alpha():
    rows = []
    for rho in RHOS:
        pool = _max_reduced_pool(rho)
        for alpha in ALPHAS:
            result = run_coverage_validation(
                pool, alpha=alpha, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED
            )
            rows.append(
                {
                    "rho": rho,
                    "alpha": alpha,
                    "k": K,
                    "n_calibration": CALIBRATION_SIZE,
                    "n_trials": N_TRIALS,
                    "mean_observed_coverage": round(result.mean_observed_coverage, 4),
                    "theoretical_band_low": round(result.band.low, 4),
                    "theoretical_band_high": round(result.band.high, 4),
                    "within_band": result.within_band,
                }
            )
            assert result.within_band, (
                f"[rho={rho}, alpha={alpha}] joint calibration's max-reduced coverage "
                f"{result.mean_observed_coverage:.4f} fell outside the theoretical band "
                f"[{result.band.low:.4f}, {result.band.high:.4f}] -- this is exactly Phase 1's "
                f"quantile machinery applied to a 1-D reduced score, so a failure here would "
                f"indicate a bug in the max-reduction wiring itself, not just noise."
            )

    REPORT_PATH.write_text(json.dumps(rows, indent=2) + "\n")
