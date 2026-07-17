"""Unit tests for Phase 2's multi-check joint calibration (core/multi_check.py).

Same discipline as test_quantile.py: the max-score reduction plus quantile
formula is worked out by hand for a toy example before trusting the
implementation on anything larger.
"""

import math

import pytest

from conformguard.core.calibration import CalibrationScoringError, InsufficientCalibrationDataError
from conformguard.core.decision import Decision
from conformguard.core.multi_check import (
    calibrate_multi_check,
    decide_multi_check,
)
from conformguard.core.quantile import conformal_quantile
from conformguard.core.scores import NonconformityScore, ToolCallContext
from conformguard.storage.calibration_store import LabelingSource


def _ctx(**metadata) -> ToolCallContext:
    return ToolCallContext(tool_name="search", args={}, metadata=metadata)


def _check_a(context: ToolCallContext) -> float:
    return context.metadata["s1"]


def _check_b(context: ToolCallContext) -> float:
    return context.metadata["s2"]


class TestHandComputedToyExample:
    def test_max_score_quantile_matches_hand_computation(self):
        # 5 calibration examples, K=2 checks each:
        # (s1, s2) pairs: (1,2) (3,1) (2,2) (4,5) (5,4)
        # max per example: 2, 3, 2, 5, 5 -> sorted: [2, 2, 3, 5, 5]
        # alpha=0.4: k = ceil((5+1)*0.6) = ceil(3.6) = 4 -> q_hat = 4th smallest = 5
        pairs = [(1, 2), (3, 1), (2, 2), (4, 5), (5, 4)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]

        calibrator = calibrate_multi_check(
            [_check_a, _check_b], data, alpha=0.4, hard_minimum_size=5
        )

        assert calibrator.q_hat == 5
        # Independently re-derive via conformal_quantile directly, to pin
        # that calibrate_multi_check's own max-reduction wiring (not just
        # the underlying quantile formula, already covered by
        # test_quantile.py) is what's under test here.
        max_scores = [max(s1, s2) for s1, s2 in pairs]
        assert calibrator.q_hat == conformal_quantile(max_scores, alpha=0.4)

    def test_smaller_alpha_gives_larger_or_equal_q_hat(self):
        pairs = [(1, 2), (3, 1), (2, 2), (4, 5), (5, 4)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]
        loose = calibrate_multi_check([_check_a, _check_b], data, alpha=0.4, hard_minimum_size=5)
        strict = calibrate_multi_check([_check_a, _check_b], data, alpha=0.1, hard_minimum_size=5)
        assert strict.q_hat >= loose.q_hat


class TestKRequirement:
    def test_raises_with_fewer_than_two_scorers(self):
        data = [(_ctx(s1=float(i)), True) for i in range(10)]
        with pytest.raises(ValueError):
            calibrate_multi_check([_check_a], data, alpha=0.1, hard_minimum_size=10)

    def test_k_property_matches_number_of_scorers(self):
        pairs = [(float(i), float(i) + 1) for i in range(20)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]
        calibrator = calibrate_multi_check([_check_a, _check_b], data, alpha=0.1, hard_minimum_size=20)
        assert calibrator.k == 2
        assert calibrator.check_names == ("check_0", "check_1")

    def test_named_scorers_preserve_names(self):
        pairs = [(float(i), float(i) + 1) for i in range(20)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]
        scorer_a = NonconformityScore("schema_gate", _check_a)
        scorer_b = NonconformityScore("logprob_confidence", _check_b)
        calibrator = calibrate_multi_check([scorer_a, scorer_b], data, alpha=0.1, hard_minimum_size=20)
        assert calibrator.check_names == ("schema_gate", "logprob_confidence")


class TestHardMinimum:
    def test_raises_below_hard_minimum(self):
        data = [(_ctx(s1=float(i), s2=float(i)), True) for i in range(5)]
        with pytest.raises(InsufficientCalibrationDataError):
            calibrate_multi_check([_check_a, _check_b], data, alpha=0.1, hard_minimum_size=100)

    def test_bad_outcome_examples_excluded_from_hard_minimum_count(self):
        good = [(_ctx(s1=float(i), s2=float(i)), True) for i in range(10)]
        bad = [(_ctx(s1=1000.0, s2=1000.0), False) for _ in range(90)]
        with pytest.raises(InsufficientCalibrationDataError):
            calibrate_multi_check([_check_a, _check_b], good + bad, alpha=0.1, hard_minimum_size=100)

    def test_empty_calibration_data_raises(self):
        with pytest.raises(InsufficientCalibrationDataError):
            calibrate_multi_check([_check_a, _check_b], [], alpha=0.1)


class TestScoringErrorOnHistoricalData:
    def test_erroring_scorer_raises_calibration_scoring_error(self):
        def _boom(ctx: ToolCallContext) -> float:
            raise RuntimeError("bad historical record")

        data = [(_ctx(), True) for _ in range(20)]
        with pytest.raises(CalibrationScoringError):
            calibrate_multi_check([_boom, _check_a], data, alpha=0.1, hard_minimum_size=20)


class TestDecideMultiCheck:
    def _calibrator(self, alpha=0.1, n=20):
        pairs = [(float(i), float(i)) for i in range(n)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]
        return calibrate_multi_check([_check_a, _check_b], data, alpha=alpha, hard_minimum_size=n)

    def test_all_checks_pass_accepts(self):
        calibrator = self._calibrator()
        result = decide_multi_check(calibrator, _ctx(s1=0.0, s2=0.0))
        assert result.decision is Decision.ACCEPT
        assert result.accepted
        assert result.failed_checks == ()
        assert all(c.passed for c in result.checks)

    def test_one_check_failing_abstains_and_is_reported(self):
        calibrator = self._calibrator()
        # s1 is way over q_hat, s2 is fine.
        result = decide_multi_check(calibrator, _ctx(s1=calibrator.q_hat + 1000, s2=0.0))
        assert result.decision is Decision.ABSTAIN
        assert "check_0" in result.failed_checks
        assert "check_1" not in result.failed_checks

    def test_max_score_is_reported(self):
        calibrator = self._calibrator()
        result = decide_multi_check(calibrator, _ctx(s1=3.0, s2=7.0))
        assert result.max_score == 7.0

    def test_both_checks_failing_abstains(self):
        calibrator = self._calibrator()
        result = decide_multi_check(calibrator, _ctx(s1=calibrator.q_hat + 100, s2=calibrator.q_hat + 200))
        assert result.decision is Decision.ABSTAIN
        assert set(result.failed_checks) == {"check_0", "check_1"}

    def test_erroring_check_forces_abstain(self):
        # A scorer that raises at calibration time makes calibrate_multi_check
        # itself raise (CalibrationScoringError, tested above) -- broken
        # historical data must not be silently skipped. To test a scorer
        # failing at *decision* time (a live failure, independent of
        # calibration), build a calibrator with two working scorers, then
        # swap one out for a broken one afterward, mirroring
        # test_decision.py's TestErroringScorerForcesAbstain pattern.
        def _boom(ctx: ToolCallContext) -> float:
            raise RuntimeError("scorer down")

        calibrator = self._calibrator()
        broken_scorers = (calibrator.scorers[0], NonconformityScore("check_1", _boom))
        broken_calibrator = calibrator.__class__(**{**calibrator.__dict__, "scorers": broken_scorers})

        result = decide_multi_check(broken_calibrator, _ctx(s1=0.0, s2=0.0))
        assert result.decision is Decision.ABSTAIN
        errored_check = next(c for c in result.checks if c.name == "check_1")
        assert errored_check.errored is True
        assert errored_check.score == math.inf

    def test_non_finite_score_forces_abstain_even_if_q_hat_is_infinite(self):
        pairs = [(float(i), float(i)) for i in range(10)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]
        calibrator = calibrate_multi_check([_check_a, _check_b], data, alpha=0.01, hard_minimum_size=10)
        assert calibrator.q_hat == math.inf

        inf_scorer = NonconformityScore("inf_check", lambda ctx: math.inf)
        pairs2 = [(float(i), float(i)) for i in range(10)]
        data2 = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs2]
        calibrator2 = calibrate_multi_check([_check_a, inf_scorer], data2, alpha=0.01, hard_minimum_size=10)
        result = decide_multi_check(calibrator2, _ctx(s1=0.0))
        assert result.decision is Decision.ABSTAIN


class TestGuaranteeStatement:
    def test_contains_k_and_check_names_and_alpha(self):
        pairs = [(float(i), float(i)) for i in range(20)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]
        scorer_a = NonconformityScore("schema_gate", _check_a)
        scorer_b = NonconformityScore("logprob_confidence", _check_b)
        calibrator = calibrate_multi_check([scorer_a, scorer_b], data, alpha=0.05, hard_minimum_size=20)
        result = decide_multi_check(calibrator, _ctx(s1=0.0, s2=0.0))

        assert result.guarantee.k == 2
        assert result.guarantee.check_names == ("schema_gate", "logprob_confidence")
        assert result.guarantee.alpha == 0.05
        assert result.guarantee.scope == "multi_check"
        assert "2 SIMULTANEOUS CHECKS" in result.guarantee.text
        assert "schema_gate" in result.guarantee.text
        assert "logprob_confidence" in result.guarantee.text
        assert "alpha=0.05" in result.guarantee.text

    def test_labeling_source_reflected(self):
        pairs = [(float(i), float(i)) for i in range(20)]
        data = [(_ctx(s1=s1, s2=s2), True) for s1, s2 in pairs]
        calibrator = calibrate_multi_check(
            [_check_a, _check_b], data, alpha=0.1, hard_minimum_size=20, labeling_source=LabelingSource.HUMAN
        )
        result = decide_multi_check(calibrator, _ctx(s1=0.0, s2=0.0))
        assert result.guarantee.labeling_source == LabelingSource.HUMAN
        assert "human" in result.guarantee.text
