import pytest

from conformguard.validation.coverage_check import (
    run_coverage_validation,
    theoretical_coverage_band,
)


class TestTheoreticalCoverageBand:
    def test_matches_hand_computed_n1000_alpha01(self):
        # k = ceil(1001 * 0.9) = 901; a=901, b=100.
        # Values below independently computed via scipy.stats.beta(901, 100).
        band = theoretical_coverage_band(n=1000, alpha=0.1)
        assert band.k == 901
        assert band.mean == pytest.approx(0.9001, abs=1e-3)
        assert band.low == pytest.approx(0.8808, abs=1e-3)
        assert band.high == pytest.approx(0.9179, abs=1e-3)

    def test_matches_hand_computed_n10_alpha05(self):
        # k = ceil(11 * 0.5) = 6; a=6, b=5.
        band = theoretical_coverage_band(n=10, alpha=0.5)
        assert band.k == 6
        assert band.mean == pytest.approx(0.5455, abs=1e-3)

    def test_matches_hand_computed_n500_alpha005(self):
        # k = ceil(501 * 0.95) = 476; a=476, b=25.
        band = theoretical_coverage_band(n=500, alpha=0.05)
        assert band.k == 476
        assert band.mean == pytest.approx(0.9501, abs=1e-3)

    def test_band_narrows_as_n_grows(self):
        small = theoretical_coverage_band(n=100, alpha=0.1)
        large = theoretical_coverage_band(n=10000, alpha=0.1)
        assert (large.high - large.low) < (small.high - small.low)

    def test_raises_when_k_exceeds_n(self):
        with pytest.raises(ValueError):
            theoretical_coverage_band(n=5, alpha=0.01)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            theoretical_coverage_band(n=100, alpha=1.5)


class TestRunCoverageValidation:
    def test_uniform_pool_mean_coverage_within_band(self):
        pool = [i / 10000 for i in range(10000)]
        result = run_coverage_validation(pool, alpha=0.1, calibration_size=1000, n_trials=100, seed=0)
        assert result.within_band, (
            f"observed mean coverage {result.mean_observed_coverage} outside "
            f"band [{result.band.low}, {result.band.high}]"
        )

    def test_result_reports_correct_trial_count(self):
        pool = [i / 5000 for i in range(5000)]
        result = run_coverage_validation(pool, alpha=0.2, calibration_size=500, n_trials=50, seed=1)
        assert result.n_trials == 50
        assert len(result.observed_coverages) == 50

    def test_seed_gives_reproducible_result(self):
        pool = [i / 5000 for i in range(5000)]
        r1 = run_coverage_validation(pool, alpha=0.1, calibration_size=500, n_trials=20, seed=42)
        r2 = run_coverage_validation(pool, alpha=0.1, calibration_size=500, n_trials=20, seed=42)
        assert r1.observed_coverages == r2.observed_coverages

    def test_calibration_size_must_leave_test_set(self):
        with pytest.raises(ValueError):
            run_coverage_validation([1.0, 2.0, 3.0], alpha=0.1, calibration_size=3, n_trials=10)

    def test_histogram_shape(self):
        pool = [i / 2000 for i in range(2000)]
        result = run_coverage_validation(pool, alpha=0.1, calibration_size=200, n_trials=30, seed=2)
        counts, edges = result.histogram(bins=10)
        assert len(counts) == 10
        assert len(edges) == 11
        assert sum(counts) == 30
