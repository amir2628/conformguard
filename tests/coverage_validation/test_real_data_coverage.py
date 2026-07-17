"""Coverage validation run against real, non-synthetic tool-call data.

Per PROJECT_SPEC §5.3 / §8: the coverage validation suite must be run at
least once against real historical tool-call data, not only synthetic
scenarios. tests/coverage_validation/scenarios/real_bfcl_calls.py loads a
fixed extract of 258 real, human-contributed tool-calling queries from the
Berkeley Function-Calling Leaderboard's "live" split (see
tests/coverage_validation/data/README.md for provenance/license), each
paired with a human-verified ground-truth call.

This pool is small (258 examples) and has more repeated score values than
the synthetic scenarios' continuous distributions (real argument
complexity clusters at a handful of common shapes), so the exact
continuous-score Beta band is a looser approximation here than for the
synthetic scenarios in test_empirical_coverage.py -- this file exists
specifically to surface that difference, not to hide it.
"""

from __future__ import annotations

import json
from pathlib import Path

from conformguard.validation.coverage_check import run_coverage_validation
from tests.coverage_validation.scenarios import real_bfcl_calls

ALPHA = 0.1
CALIBRATION_SIZE = 150
N_TRIALS = 100
SEED = 20260701

REPORT_PATH = Path(__file__).parent / "latest_real_data_report.json"


def test_real_bfcl_data_coverage_is_reasonably_close_to_target():
    pool = real_bfcl_calls.load_scores()
    assert len(pool) > CALIBRATION_SIZE, "real data pool must be larger than the calibration size"

    result = run_coverage_validation(pool, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED)

    target = 1 - ALPHA
    # A wider tolerance than the synthetic suite's exact-band assertion:
    # real, tied, small-n data is not the continuous distribution the exact
    # Beta band assumes. Documenting the gap honestly (rather than silently
    # loosening test_empirical_coverage.py's real assertion) is the point.
    assert abs(result.mean_observed_coverage - target) < 0.08, (
        f"mean observed coverage on real BFCL data ({result.mean_observed_coverage:.4f}) "
        f"deviated from the target ({target}) by more than the documented tolerance for "
        f"small, tied, real-world data."
    )

    report = {
        "scenario": real_bfcl_calls.SCENARIO_NAME,
        "data_source": "ShishirPatil/gorilla berkeley-function-call-leaderboard BFCL_v4_live_simple (Apache-2.0)",
        "pool_size": len(pool),
        "alpha": ALPHA,
        "n_calibration": CALIBRATION_SIZE,
        "n_trials": N_TRIALS,
        "target_coverage": round(target, 4),
        "mean_observed_coverage": round(result.mean_observed_coverage, 4),
        "theoretical_band_low": round(result.band.low, 4),
        "theoretical_band_high": round(result.band.high, 4),
        "within_exact_band": result.within_band,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
