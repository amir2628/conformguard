"""Proves the multi-check coverage-validation harness detects a SINGLE drifting check.

Stronger, more realistic test than shifting the already-max-reduced
score: here only ONE of the K checks (e.g. check index 1 of 3) has its
test-time distribution shifted, while the other K-1 checks remain
perfectly exchangeable with the calibration set. This confirms the joint
(max-score) method still catches localized drift in a single check,
rather than that drift being diluted away by the other, still-healthy
checks feeding into the same max.
"""

from __future__ import annotations

import json
from pathlib import Path

from conformguard.validation.negative_control import constant_shift, no_shift, run_multi_check_negative_control
from tests.coverage_validation.scenarios import correlated_multi_check_scores

CALIBRATION_SIZE = 1000
N_TRIALS = 100
SEED = 20260701
ALPHA = 0.1
K = 3
RHOS = [0.0, 0.5, 0.9]
SHIFT_COLUMN = 1  # the middle of the 3 checks; arbitrary but fixed

REPORT_PATH = Path(__file__).parent / "latest_multi_check_negative_control_report.json"


def test_no_shift_control_and_single_check_shift_detection_across_rho():
    rows = []
    for rho in RHOS:
        pool = correlated_multi_check_scores.generate_pool(size=20_000, seed=SEED, k=K, rho=rho)
        column_spread = pool[:, SHIFT_COLUMN].max() - pool[:, SHIFT_COLUMN].min()

        no_shift_result = run_multi_check_negative_control(
            pool,
            alpha=ALPHA,
            calibration_size=CALIBRATION_SIZE,
            shift_column=SHIFT_COLUMN,
            shift_fn=no_shift(),
            n_trials=N_TRIALS,
            seed=SEED,
        )
        shifted_result = run_multi_check_negative_control(
            pool,
            alpha=ALPHA,
            calibration_size=CALIBRATION_SIZE,
            shift_column=SHIFT_COLUMN,
            shift_fn=constant_shift(column_spread),
            n_trials=N_TRIALS,
            seed=SEED,
        )

        rows.append(
            {
                "rho": rho,
                "alpha": ALPHA,
                "k": K,
                "shift_column": SHIFT_COLUMN,
                "no_shift_mean_coverage": round(no_shift_result.mean_observed_coverage, 4),
                "no_shift_degraded": no_shift_result.degraded,
                "shifted_mean_coverage": round(shifted_result.mean_observed_coverage, 4),
                "shifted_degraded": shifted_result.degraded,
            }
        )

        assert no_shift_result.degraded is False, (
            f"[rho={rho}] no-shift control (check {SHIFT_COLUMN} untouched) falsely reported "
            f"degraded joint coverage"
        )
        assert shifted_result.degraded is True, (
            f"[rho={rho}] a full-spread-width shift on check {SHIFT_COLUMN} alone (the other "
            f"{K - 1} checks left exchangeable) was NOT detected as degraded joint coverage -- "
            f"the other, still-healthy checks may be masking the one that drifted."
        )

    REPORT_PATH.write_text(json.dumps(rows, indent=2) + "\n")


def test_shift_on_each_individual_check_is_detected():
    """Confirms detection doesn't depend on which of the K checks drifts."""
    pool = correlated_multi_check_scores.generate_pool(size=20_000, seed=SEED, k=K, rho=0.5)
    for column in range(K):
        column_spread = pool[:, column].max() - pool[:, column].min()
        result = run_multi_check_negative_control(
            pool,
            alpha=ALPHA,
            calibration_size=CALIBRATION_SIZE,
            shift_column=column,
            shift_fn=constant_shift(column_spread),
            n_trials=N_TRIALS,
            seed=SEED,
        )
        assert result.degraded, f"shift on check index {column} (of {K}) was not detected"
