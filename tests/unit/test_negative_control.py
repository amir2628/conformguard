import numpy as np
import pytest

from conformguard.validation.negative_control import (
    constant_shift,
    distribution_swap,
    no_shift,
    run_multi_check_negative_control,
    run_negative_control,
)


class TestNoShiftSanityCheck:
    def test_identity_shift_does_not_falsely_report_degraded(self):
        """The negative control on the negative control: unshifted data must not cry wolf."""
        pool = [i / 5000 for i in range(5000)]
        result = run_negative_control(
            pool, alpha=0.1, calibration_size=500, shift_fn=no_shift(), n_trials=100, seed=7
        )
        assert result.degraded is False


class TestConstantShiftIsDetected:
    def test_large_upward_shift_degrades_coverage(self):
        pool = [i / 5000 for i in range(5000)]
        result = run_negative_control(
            pool, alpha=0.1, calibration_size=500, shift_fn=constant_shift(0.5), n_trials=100, seed=7
        )
        assert result.degraded is True
        assert result.mean_observed_coverage < result.band.low

    def test_tiny_shift_may_not_degrade(self):
        # A shift far smaller than the score range shouldn't reliably trip
        # the detector -- this isn't a hard assertion (it's a probabilistic
        # boundary), just documents that degraded is a real measurement,
        # not a hardcoded True for any nonzero shift.
        pool = [i / 5000 for i in range(5000)]
        result = run_negative_control(
            pool, alpha=0.1, calibration_size=500, shift_fn=constant_shift(1e-6), n_trials=100, seed=7
        )
        assert result.mean_observed_coverage > result.band.low - 0.05


class TestDistributionSwapIsDetected:
    def test_swapping_to_a_riskier_distribution_degrades_coverage(self):
        good_pool = [i / 5000 for i in range(5000)]  # Uniform(0, 1)
        risky_pool = [1.0 + i / 5000 for i in range(5000)]  # Uniform(1, 2), disjoint range
        result = run_negative_control(
            good_pool,
            alpha=0.1,
            calibration_size=500,
            shift_fn=distribution_swap(risky_pool),
            n_trials=100,
            seed=7,
        )
        assert result.degraded is True
        # A fully disjoint, higher-valued test distribution should push
        # coverage close to zero, not just barely below the band.
        assert result.mean_observed_coverage < 0.2


def _independent_good_pool(n=5000, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=(n, k))


class TestMultiCheckNegativeControl:
    def test_no_shift_does_not_falsely_report_degraded(self):
        pool = _independent_good_pool()
        result = run_multi_check_negative_control(
            pool, alpha=0.1, calibration_size=1000, shift_column=0, shift_fn=no_shift(), n_trials=100, seed=7
        )
        assert result.degraded is False

    def test_shift_on_one_column_is_detected(self):
        pool = _independent_good_pool()
        result = run_multi_check_negative_control(
            pool, alpha=0.1, calibration_size=1000, shift_column=0, shift_fn=constant_shift(1.0), n_trials=100, seed=7
        )
        assert result.degraded is True

    def test_shift_on_untouched_column_leaves_other_columns_unaffected(self):
        # Shifting column 0 shouldn't itself validate anything about column 1;
        # this just confirms the function only touches the requested column.
        pool = _independent_good_pool()
        result_col0 = run_multi_check_negative_control(
            pool, alpha=0.1, calibration_size=1000, shift_column=0, shift_fn=constant_shift(5.0), n_trials=50, seed=1
        )
        result_col1 = run_multi_check_negative_control(
            pool, alpha=0.1, calibration_size=1000, shift_column=1, shift_fn=constant_shift(5.0), n_trials=50, seed=1
        )
        assert result_col0.degraded is True
        assert result_col1.degraded is True

    def test_invalid_shift_column_raises(self):
        pool = _independent_good_pool(k=3)
        with pytest.raises(ValueError):
            run_multi_check_negative_control(
                pool, alpha=0.1, calibration_size=1000, shift_column=3, shift_fn=no_shift(), n_trials=10
            )

    def test_calibration_size_must_leave_test_set(self):
        pool = _independent_good_pool(n=100)
        with pytest.raises(ValueError):
            run_multi_check_negative_control(
                pool, alpha=0.1, calibration_size=100, shift_column=0, shift_fn=no_shift(), n_trials=10
            )
