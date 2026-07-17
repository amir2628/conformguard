"""The project's core proof: observed coverage matches the theoretical band.

Per PROJECT_SPEC §7.3 / §8: for at least two alpha values, across R >= 100
repeated calibration/test splits, the mean observed coverage must fall
within the theoretically-predicted Beta-distribution fluctuation band
(validation/coverage_check.py). This is the artifact a skeptical reader
should be able to reproduce and check, not a number quoted on faith.

Running this suite also regenerates the stable report table consumed by
the README (see report_table() below) -- treat its output format as a
versioned artifact, per PROJECT_SPEC §7.3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conformguard.validation.coverage_check import run_coverage_validation
from tests.coverage_validation.scenarios import (
    logprob_derived_scores,
    skewed_confidence_scores,
    uniform_scores,
)

SCENARIOS = [uniform_scores, skewed_confidence_scores, logprob_derived_scores]
ALPHAS = [0.1, 0.05]
CALIBRATION_SIZE = 1000
POOL_SIZE = 20_000
N_TRIALS = 100
SEED = 20260701

REPORT_PATH = Path(__file__).parent / "latest_report.json"


def _all_cases():
    for scenario in SCENARIOS:
        for alpha in ALPHAS:
            yield scenario, alpha


@pytest.mark.parametrize("scenario,alpha", list(_all_cases()), ids=lambda v: getattr(v, "SCENARIO_NAME", v))
def test_observed_coverage_within_theoretical_band(scenario, alpha):
    pool = scenario.generate_pool(size=POOL_SIZE, seed=SEED)
    result = run_coverage_validation(
        pool, alpha=alpha, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED
    )
    assert result.within_band, (
        f"[{scenario.SCENARIO_NAME}, alpha={alpha}] mean observed coverage "
        f"{result.mean_observed_coverage:.4f} fell outside the theoretical band "
        f"[{result.band.low:.4f}, {result.band.high:.4f}] (target {result.band.mean:.4f}) "
        f"across {N_TRIALS} trials -- this indicates a bug in the calibration math, "
        f"not just an unlucky run, since R >= 100 trials is the threshold PROJECT_SPEC "
        f"§7.3 sets for treating this as signal rather than noise."
    )


def test_report_table_generation_and_persistence():
    """Regenerates the stable report table consumed by the README."""
    rows = []
    for scenario, alpha in _all_cases():
        pool = scenario.generate_pool(size=POOL_SIZE, seed=SEED)
        result = run_coverage_validation(
            pool, alpha=alpha, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED
        )
        rows.append(
            {
                "scenario": scenario.SCENARIO_NAME,
                "alpha": alpha,
                "target_coverage": round(1 - alpha, 4),
                "n_calibration": CALIBRATION_SIZE,
                "n_trials": N_TRIALS,
                "mean_observed_coverage": round(result.mean_observed_coverage, 4),
                "theoretical_band_low": round(result.band.low, 4),
                "theoretical_band_high": round(result.band.high, 4),
                "within_band": result.within_band,
            }
        )
        assert result.within_band

    REPORT_PATH.write_text(json.dumps(rows, indent=2) + "\n")
    assert REPORT_PATH.exists()
