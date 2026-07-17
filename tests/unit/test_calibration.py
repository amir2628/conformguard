import math
from datetime import datetime, timedelta, timezone

import pytest

from conformguard.core.calibration import (
    CalibrationScoringError,
    InsufficientCalibrationDataError,
    calibrate,
)
from conformguard.core.quantile import conformal_quantile
from conformguard.core.scores import NonconformityScore, ToolCallContext
from conformguard.storage.calibration_store import LabelingSource


def _ctx(tool_name: str = "search", **metadata) -> ToolCallContext:
    return ToolCallContext(tool_name=tool_name, args={}, metadata=metadata)


def _score_from_metadata(context: ToolCallContext) -> float:
    return context.metadata["score"]


def _dataset(n_good: int, n_bad: int = 0, base_score: float = 0.0):
    scorer_fn = _score_from_metadata
    data = []
    for i in range(n_good):
        data.append((_ctx(score=base_score + i), True))
    for i in range(n_bad):
        data.append((_ctx(score=base_score + 1000 + i), False))
    return NonconformityScore("test_scorer", scorer_fn), data


class TestHardMinimum:
    def test_raises_below_hard_minimum(self):
        scorer, data = _dataset(n_good=5)
        with pytest.raises(InsufficientCalibrationDataError):
            calibrate(scorer, data, alpha=0.1, hard_minimum_size=100)

    def test_succeeds_at_hard_minimum(self):
        scorer, data = _dataset(n_good=10)
        calibrator = calibrate(scorer, data, alpha=0.1, hard_minimum_size=10)
        assert calibrator.n_calibration == 10

    def test_raises_on_empty_calibration_data(self):
        scorer = NonconformityScore("s", _score_from_metadata)
        with pytest.raises(InsufficientCalibrationDataError):
            calibrate(scorer, [], alpha=0.1)

    def test_bad_outcome_examples_excluded_from_hard_minimum_count(self):
        # 10 good + 90 bad = 100 total, but only 10 are usable for calibration.
        scorer, data = _dataset(n_good=10, n_bad=90)
        with pytest.raises(InsufficientCalibrationDataError):
            calibrate(scorer, data, alpha=0.1, hard_minimum_size=100)


class TestQuantileMatchesDirectComputation:
    def test_q_hat_matches_conformal_quantile_on_good_scores_only(self):
        scorer, data = _dataset(n_good=20, n_bad=5, base_score=1.0)
        calibrator = calibrate(scorer, data, alpha=0.2, hard_minimum_size=20)
        good_scores = [1.0 + i for i in range(20)]
        expected = conformal_quantile(good_scores, alpha=0.2)
        assert calibrator.q_hat == expected

    def test_bad_outcome_scores_do_not_affect_q_hat(self):
        scorer, data_without_bad = _dataset(n_good=20, n_bad=0, base_score=1.0)
        cal_without_bad = calibrate(scorer, data_without_bad, alpha=0.2, hard_minimum_size=20)

        scorer2, data_with_bad = _dataset(n_good=20, n_bad=50, base_score=1.0)
        cal_with_bad = calibrate(scorer2, data_with_bad, alpha=0.2, hard_minimum_size=20)

        assert cal_without_bad.q_hat == cal_with_bad.q_hat
        assert cal_with_bad.n_excluded == 50


class TestAlphaValidation:
    def test_invalid_alpha_raises(self):
        scorer, data = _dataset(n_good=10)
        with pytest.raises(ValueError):
            calibrate(scorer, data, alpha=0.0, hard_minimum_size=10)


class TestScoringErrorOnHistoricalData:
    def test_erroring_scorer_raises_calibration_scoring_error(self):
        def _boom(ctx: ToolCallContext) -> float:
            raise RuntimeError("bad historical record")

        scorer = NonconformityScore("boom", _boom)
        data = [(_ctx(), True) for _ in range(10)]
        with pytest.raises(CalibrationScoringError):
            calibrate(scorer, data, alpha=0.1, hard_minimum_size=10)

    def test_bare_callable_is_wrapped(self):
        data = [(_ctx(score=float(i)), True) for i in range(10)]
        calibrator = calibrate(_score_from_metadata, data, alpha=0.1, hard_minimum_size=10)
        assert calibrator.scorer.name == "user_scorer"


class TestMetadata:
    def test_tool_names_collected(self):
        data = [
            (_ctx(tool_name="search", score=1.0), True),
            (_ctx(tool_name="write_file", score=2.0), True),
        ] * 5
        calibrator = calibrate(_score_from_metadata, data, alpha=0.1, hard_minimum_size=10)
        assert calibrator.tool_names == frozenset({"search", "write_file"})

    def test_labeling_source_recorded(self):
        data = [(_ctx(score=float(i)), True) for i in range(10)]
        calibrator = calibrate(
            _score_from_metadata,
            data,
            alpha=0.1,
            hard_minimum_size=10,
            labeling_source=LabelingSource.HUMAN,
        )
        assert calibrator.labeling_source == LabelingSource.HUMAN

    def test_explicit_calibration_set_version(self):
        data = [(_ctx(score=float(i)), True) for i in range(10)]
        calibrator = calibrate(
            _score_from_metadata, data, alpha=0.1, hard_minimum_size=10, calibration_set_version="release-42"
        )
        assert calibrator.calibration_set_version == "release-42"

    def test_auto_generated_version_when_not_specified(self):
        data = [(_ctx(score=float(i)), True) for i in range(10)]
        calibrator = calibrate(_score_from_metadata, data, alpha=0.1, hard_minimum_size=10)
        assert calibrator.calibration_set_version.startswith("auto-")

    def test_timestamps_from_metadata_used_when_present(self):
        now = datetime.now(timezone.utc)
        earlier = now - timedelta(days=10)
        data = [
            (_ctx(score=float(i), timestamp=earlier if i == 0 else now), True) for i in range(10)
        ]
        calibrator = calibrate(_score_from_metadata, data, alpha=0.1, hard_minimum_size=10)
        assert calibrator.calibration_start == earlier
        assert calibrator.calibration_end == now


class TestQHatCanBeInfinite:
    def test_too_small_alpha_for_n_gives_infinite_threshold(self):
        # n=10 good examples, alpha so small that k > n -> q_hat = inf.
        data = [(_ctx(score=float(i)), True) for i in range(10)]
        calibrator = calibrate(_score_from_metadata, data, alpha=0.01, hard_minimum_size=10)
        assert calibrator.q_hat == math.inf
