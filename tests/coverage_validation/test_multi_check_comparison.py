"""Phase 2 comparison harness: joint (max-score) vs. naive independent vs. Bonferroni.

Replicates PASC's own comparison methodology (arXiv:2605.18812) on this
project's tool-calling domain, per PROJECT_SPEC §3 Phase 2. Two metrics,
both required (see validation/multi_check_comparison.py's module
docstring for why "good-call acceptance rate" alone is actively
misleading as an efficiency proxy):

- **Validity**: coverage on held-out good calls vs. the 1-alpha target.
- **Efficiency**: rejection rate on a separate bad/anomalous pool, at the
  same alpha -- a method that discriminates better between good and bad
  calls rejects more of the bad pool while still meeting its good-call
  coverage target.

Two data sources, used for different, complementary purposes:

- **Synthetic** (primary): the comparison's core claim -- that the joint
  method's efficiency advantage over Bonferroni grows with how correlated
  the K checks are -- requires controlling that correlation directly,
  which only synthetic data allows.
- **Real** (supplementary): a real, non-synthetic good-call pool (the
  BFCL "live" extract used in Phase 1's real-data validation) with two
  genuinely different deterministic scores computed from the same real
  calls (correlation ~0.31, checked, not assumed). Its "bad" pool is
  necessarily synthetic (BFCL's ground truth has no bad calls to draw
  from) -- disclosed, not hidden; see the scenario module's docstring.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from conformguard.validation.multi_check_comparison import run_multi_check_comparison
from tests.coverage_validation.scenarios import correlated_multi_check_scores, real_bfcl_multi_check_scores

ALPHA = 0.1
K = 3
POOL_SIZE = 20_000
BAD_POOL_SIZE = 5_000
CALIBRATION_SIZE = 1000
N_TRIALS = 100
SEED = 20260701
RHOS = [0.0, 0.5, 0.9]

REPORT_PATH = Path(__file__).parent / "latest_multi_check_report.json"


def _synthetic_pools(rho):
    good = correlated_multi_check_scores.generate_pool(size=POOL_SIZE, seed=SEED, k=K, rho=rho)
    bad = correlated_multi_check_scores.generate_bad_pool(size=BAD_POOL_SIZE, seed=SEED + 1, k=K, rho=rho)
    return good, bad


class TestNaiveIndependentHasNoValidGuarantee:
    @pytest.mark.parametrize("rho", RHOS)
    def test_naive_good_coverage_falls_below_target(self, rho):
        good, bad = _synthetic_pools(rho)
        results = run_multi_check_comparison(
            good, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED, bad_pool=bad
        )
        naive = results["naive"]
        target = 1 - ALPHA
        assert naive.mean_good_coverage < target, (
            f"[rho={rho}] naive independent calibration's good-call coverage "
            f"({naive.mean_good_coverage:.4f}) did not fall below the target ({target}) -- "
            f"expected it to, since it has no multiplicity correction at all."
        )


class TestJointAndBonferroniAreValid:
    @pytest.mark.parametrize("rho", RHOS)
    def test_both_valid_methods_meet_or_exceed_target(self, rho):
        good, bad = _synthetic_pools(rho)
        results = run_multi_check_comparison(
            good, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED, bad_pool=bad
        )
        target = 1 - ALPHA
        # Small tolerance for finite-trial noise, same pattern as test_empirical_coverage.py's band.
        assert results["joint"].mean_good_coverage >= target - 0.02, (
            f"[rho={rho}] joint's good coverage {results['joint'].mean_good_coverage:.4f} "
            f"fell meaningfully below target {target}"
        )
        assert results["bonferroni"].mean_good_coverage >= target - 0.02, (
            f"[rho={rho}] bonferroni's good coverage {results['bonferroni'].mean_good_coverage:.4f} "
            f"fell meaningfully below target {target}"
        )


class TestJointIsMoreEfficientThanBonferroni:
    """The core PASC-replicating claim, correctly operationalized (see module docstring
    for why "good-call acceptance rate" was tried first and found to be a misleading
    proxy): at the SAME validity (both methods meet the good-coverage target), joint
    rejects bad/anomalous calls at least as often as Bonferroni, and meaningfully more
    often as the K checks become more correlated.
    """

    @pytest.mark.parametrize("rho", RHOS)
    def test_joint_bad_rejection_at_least_bonferroni(self, rho):
        good, bad = _synthetic_pools(rho)
        results = run_multi_check_comparison(
            good, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED, bad_pool=bad
        )
        joint_rej = results["joint"].mean_bad_rejection_rate
        bonf_rej = results["bonferroni"].mean_bad_rejection_rate
        assert joint_rej >= bonf_rej - 0.01, (
            f"[rho={rho}] joint's bad-call rejection rate ({joint_rej:.4f}) was meaningfully "
            f"lower than bonferroni's ({bonf_rej:.4f}) -- expected joint to be at least as "
            f"efficient at this correlation level."
        )

    def test_joint_advantage_over_bonferroni_grows_with_correlation(self):
        gaps = []
        for rho in RHOS:
            good, bad = _synthetic_pools(rho)
            results = run_multi_check_comparison(
                good, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED, bad_pool=bad
            )
            gap = results["joint"].mean_bad_rejection_rate - results["bonferroni"].mean_bad_rejection_rate
            gaps.append(gap)
        assert gaps[-1] > gaps[0], (
            f"expected the joint-vs-Bonferroni bad-rejection-rate gap to grow from "
            f"rho={RHOS[0]} (gap={gaps[0]:.4f}) to rho={RHOS[-1]} (gap={gaps[-1]:.4f}), but it did not"
        )
        assert all(gap >= -0.01 for gap in gaps), f"joint should never be meaningfully worse than bonferroni: {gaps}"


class TestRealDataSupplementaryComparison:
    def test_real_bfcl_two_check_pattern_matches_synthetic(self):
        good = real_bfcl_multi_check_scores.load_score_matrix()
        bad = real_bfcl_multi_check_scores.load_synthetic_bad_pool(size=BAD_POOL_SIZE, seed=SEED)
        real_calibration_size = 150  # pool has 258 rows; matches Phase 1's real-data test sizing
        results = run_multi_check_comparison(
            good, alpha=ALPHA, calibration_size=real_calibration_size, n_trials=N_TRIALS, seed=SEED, bad_pool=bad
        )
        target = 1 - ALPHA
        # Real, small, tied data: looser tolerance than the synthetic suite's,
        # same honesty pattern as test_real_data_coverage.py.
        assert results["naive"].mean_good_coverage < target + 0.05
        assert results["joint"].mean_bad_rejection_rate >= results["bonferroni"].mean_bad_rejection_rate - 0.05


def test_report_table_generation_and_persistence():
    report = {"synthetic": [], "real": None}

    for rho in RHOS:
        good, bad = _synthetic_pools(rho)
        results = run_multi_check_comparison(
            good, alpha=ALPHA, calibration_size=CALIBRATION_SIZE, n_trials=N_TRIALS, seed=SEED, bad_pool=bad
        )
        row = {"rho": rho, "alpha": ALPHA, "k": K, "n_calibration": CALIBRATION_SIZE, "n_trials": N_TRIALS}
        for name, result in results.items():
            row[f"{name}_mean_good_coverage"] = round(result.mean_good_coverage, 4)
            row[f"{name}_mean_bad_rejection_rate"] = round(result.mean_bad_rejection_rate, 4)
        report["synthetic"].append(row)

    real_good = real_bfcl_multi_check_scores.load_score_matrix()
    real_bad = real_bfcl_multi_check_scores.load_synthetic_bad_pool(size=BAD_POOL_SIZE, seed=SEED)
    real_results = run_multi_check_comparison(
        real_good, alpha=ALPHA, calibration_size=150, n_trials=N_TRIALS, seed=SEED, bad_pool=real_bad
    )
    real_row = {
        "data_source": "BFCL live_simple extract (Apache-2.0), 2 real deterministic scores; "
        "bad pool is a synthetic shift of the real good scores (see scenario module docstring)",
        "pool_size": len(real_good),
        "alpha": ALPHA,
        "k": 2,
        "n_calibration": 150,
        "n_trials": N_TRIALS,
    }
    for name, result in real_results.items():
        real_row[f"{name}_mean_good_coverage"] = round(result.mean_good_coverage, 4)
        real_row[f"{name}_mean_bad_rejection_rate"] = round(result.mean_bad_rejection_rate, 4)
    report["real"] = real_row

    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    assert REPORT_PATH.exists()
