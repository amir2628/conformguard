"""Hand-computed toy examples for the split-conformal quantile.

This file is the crux of the whole project: an off-by-one error here isn't
a functional regression, it's a false statistical claim in every downstream
accept/abstain decision. Every expected value below is worked out by hand
in the comment above it, independent of the implementation.
"""

import math

import pytest

from conformguard.core.quantile import (
    EmptyCalibrationSetError,
    InvalidAlphaError,
    conformal_quantile,
    quantile_level,
)


class TestHandComputedToyExamples:
    def test_five_scores_alpha_one_fifth(self):
        # scores = [1, 2, 3, 4, 5], n = 5, alpha = 0.2
        # k = ceil((n + 1)(1 - alpha)) = ceil(6 * 0.8) = ceil(4.8) = 5
        # q_hat = 5th smallest of [1, 2, 3, 4, 5] = 5
        assert conformal_quantile([1, 2, 3, 4, 5], alpha=0.2) == 5

    def test_five_scores_alpha_two_fifths(self):
        # scores = [1, 2, 3, 4, 5], n = 5, alpha = 0.4
        # k = ceil(6 * 0.6) = ceil(3.6) = 4
        # q_hat = 4th smallest of [1, 2, 3, 4, 5] = 4
        assert conformal_quantile([1, 2, 3, 4, 5], alpha=0.4) == 4

    def test_five_scores_alpha_too_small_gives_infinity(self):
        # scores = [1, 2, 3, 4, 5], n = 5, alpha = 0.1
        # k = ceil(6 * 0.9) = ceil(5.4) = 6, which exceeds n = 5.
        # The correction is still mathematically valid: it means no finite
        # threshold at this n achieves this alpha, so q_hat = +inf (abstain
        # on everything) rather than silently returning a wrong, finite
        # number computed off a truncated/misapplied index.
        assert conformal_quantile([1, 2, 3, 4, 5], alpha=0.1) == math.inf

    def test_ten_scores_alpha_one_half(self):
        # scores = [1..10], n = 10, alpha = 0.5
        # k = ceil(11 * 0.5) = ceil(5.5) = 6
        # q_hat = 6th smallest of [1..10] = 6
        assert conformal_quantile(list(range(1, 11)), alpha=0.5) == 6

    def test_ten_scores_alpha_three_tenths(self):
        # scores = [1..10], n = 10, alpha = 0.3
        # k = ceil(11 * 0.7) = ceil(7.7) = 8
        # q_hat = 8th smallest of [1..10] = 8
        assert conformal_quantile(list(range(1, 11)), alpha=0.3) == 8

    def test_ten_scores_alpha_one_tenth(self):
        # scores = [1..10], n = 10, alpha = 0.1
        # k = ceil(11 * 0.9) = ceil(9.9) = 10
        # q_hat = 10th smallest of [1..10] = 10 (exactly the max, not inf,
        # since k == n here rather than k > n)
        assert conformal_quantile(list(range(1, 11)), alpha=0.1) == 10

    def test_order_of_input_scores_does_not_matter(self):
        # Same toy example as above, scores given out of order:
        # k = ceil(6 * 0.8) = 5, q_hat = 5th smallest = 5, regardless of
        # the order the scores were passed in.
        assert conformal_quantile([5, 1, 4, 2, 3], alpha=0.2) == 5

    def test_numpy_method_higher_would_be_wrong_here(self):
        """Documents the exact pitfall the spec calls out.

        numpy.quantile(scores, k/n, method="higher") is NOT the same as
        the k-th order statistic Angelopoulos & Bates define, because
        numpy's virtual index for "higher" uses (n - 1) in the
        denominator rather than n. For n=10, alpha=0.5, k=6, level=0.6:
        numpy's method="higher" gives the 7th smallest value (7), not the
        6th (6). This test pins the CORRECT value so a future refactor
        that reaches for numpy.quantile with method="higher" fails loudly.
        """
        scores = list(range(1, 11))
        correct = conformal_quantile(scores, alpha=0.5)
        assert correct == 6
        numpy_higher_would_give = 7
        assert correct != numpy_higher_would_give


class TestQuantileLevel:
    def test_matches_conformal_quantile_rank(self):
        # n=10, alpha=0.3 -> k=8 -> level = 8/10 = 0.8
        assert quantile_level(10, alpha=0.3) == pytest.approx(0.8)

    def test_invalid_n_raises(self):
        with pytest.raises(EmptyCalibrationSetError):
            quantile_level(0, alpha=0.1)


class TestInputValidation:
    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_alpha_out_of_range_raises(self, alpha):
        with pytest.raises(InvalidAlphaError):
            conformal_quantile([1, 2, 3], alpha=alpha)

    def test_empty_scores_raises(self):
        with pytest.raises(EmptyCalibrationSetError):
            conformal_quantile([], alpha=0.1)

    def test_single_score(self):
        # n=1, alpha=0.5: k = ceil(2 * 0.5) = 1, q_hat = 1st smallest = only score.
        assert conformal_quantile([42.0], alpha=0.5) == 42.0
