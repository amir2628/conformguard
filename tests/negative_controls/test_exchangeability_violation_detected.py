"""Proves the coverage-validation harness detects a broken exchangeability assumption.

Required before any release (PROJECT_SPEC §7.6). This is not a test that
the *library's decisions* behave a certain way under drift -- conformal
prediction offers no guarantee once exchangeability is broken, by
definition. It is a test that the *validation harness itself* (the tool a
user would run to sanity-check their own calibration set) correctly
reports degraded coverage rather than a falsely reassuring in-band number,
across a range of realistic score distributions and shift magnitudes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conformguard.validation.negative_control import (
    constant_shift,
    distribution_swap,
    no_shift,
    run_negative_control,
)
from tests.coverage_validation.scenarios import (
    logprob_derived_scores,
    skewed_confidence_scores,
    uniform_scores,
)

SCENARIOS = [uniform_scores, skewed_confidence_scores, logprob_derived_scores]
ALPHA = 0.1
CALIBRATION_SIZE = 1000
POOL_SIZE = 20_000
N_TRIALS = 100
SEED = 20260701

REPORT_PATH = Path(__file__).parent / "latest_report.json"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.SCENARIO_NAME)
class TestExchangeabilityViolationIsDetected:
    def test_no_shift_control_does_not_falsely_flag_degradation(self, scenario):
        """The negative control on the negative control, per scenario."""
        pool = scenario.generate_pool(size=POOL_SIZE, seed=SEED)
        result = run_negative_control(
            pool, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, shift_fn=no_shift(), n_trials=N_TRIALS, seed=SEED
        )
        assert result.degraded is False, (
            f"[{scenario.SCENARIO_NAME}] no_shift control falsely reported degraded coverage "
            f"-- the detector is crying wolf on perfectly exchangeable data."
        )

    def test_constant_shift_is_detected(self, scenario):
        pool = scenario.generate_pool(size=POOL_SIZE, seed=SEED)
        pool_spread = max(pool) - min(pool)
        shift_amount = pool_spread  # shift by a full spread-width: unambiguous, not a boundary case
        result = run_negative_control(
            pool,
            alpha=ALPHA,
            calibration_size=CALIBRATION_SIZE,
            shift_fn=constant_shift(shift_amount),
            n_trials=N_TRIALS,
            seed=SEED,
        )
        assert result.degraded, (
            f"[{scenario.SCENARIO_NAME}] a full-spread-width upward shift in test-time scores "
            f"was NOT detected as degraded coverage (mean={result.mean_observed_coverage:.4f}, "
            f"band_low={result.band.low:.4f}) -- the validation harness is failing to catch a "
            f"broken exchangeability assumption."
        )

    def test_distribution_swap_is_detected(self, scenario):
        pool = scenario.generate_pool(size=POOL_SIZE, seed=SEED)
        # A disjoint, structurally different distribution: everything shifted
        # well above the calibration pool's own range.
        pool_max = max(pool)
        risky_pool = [pool_max + 1.0 + v for v in scenario.generate_pool(size=POOL_SIZE, seed=SEED + 1)]
        result = run_negative_control(
            pool,
            alpha=ALPHA,
            calibration_size=CALIBRATION_SIZE,
            shift_fn=distribution_swap(risky_pool),
            n_trials=N_TRIALS,
            seed=SEED,
        )
        assert result.degraded, (
            f"[{scenario.SCENARIO_NAME}] a disjoint, structurally different test-time "
            f"distribution was NOT detected as degraded coverage."
        )


def test_report_table_generation_and_persistence():
    rows = []
    for scenario in SCENARIOS:
        pool = scenario.generate_pool(size=POOL_SIZE, seed=SEED)
        pool_spread = max(pool) - min(pool)
        no_shift_result = run_negative_control(
            pool, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, shift_fn=no_shift(), n_trials=N_TRIALS, seed=SEED
        )
        shifted_result = run_negative_control(
            pool,
            alpha=ALPHA,
            calibration_size=CALIBRATION_SIZE,
            shift_fn=constant_shift(pool_spread),
            n_trials=N_TRIALS,
            seed=SEED,
        )
        rows.append(
            {
                "scenario": scenario.SCENARIO_NAME,
                "alpha": ALPHA,
                "no_shift_mean_coverage": round(no_shift_result.mean_observed_coverage, 4),
                "no_shift_degraded": no_shift_result.degraded,
                "shifted_mean_coverage": round(shifted_result.mean_observed_coverage, 4),
                "shifted_degraded": shifted_result.degraded,
                "theoretical_band_low": round(shifted_result.band.low, 4),
                "theoretical_band_high": round(shifted_result.band.high, 4),
            }
        )
        assert no_shift_result.degraded is False
        assert shifted_result.degraded is True

    REPORT_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    assert REPORT_PATH.exists()
