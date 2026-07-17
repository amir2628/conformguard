import math

from conformguard.core.calibration import calibrate
from conformguard.core.decision import Decision, build_guarantee_statement, decide
from conformguard.core.scores import NonconformityScore, ToolCallContext
from conformguard.storage.calibration_store import LabelingSource


def _ctx(tool_name: str = "search", **metadata) -> ToolCallContext:
    return ToolCallContext(tool_name=tool_name, args={}, metadata=metadata)


def _score_from_metadata(context: ToolCallContext) -> float:
    return context.metadata["score"]


def _calibrator(alpha=0.1, n=20, labeling_source=LabelingSource.DETERMINISTIC):
    data = [(_ctx(score=float(i)), True) for i in range(n)]
    return calibrate(_score_from_metadata, data, alpha=alpha, hard_minimum_size=n, labeling_source=labeling_source)


class TestAcceptAbstainBoundary:
    def test_score_at_or_below_threshold_accepts(self):
        calibrator = _calibrator(alpha=0.1, n=20)
        result = decide(calibrator, _ctx(score=calibrator.q_hat))
        assert result.decision is Decision.ACCEPT
        assert result.accepted is True

    def test_score_above_threshold_abstains(self):
        calibrator = _calibrator(alpha=0.1, n=20)
        result = decide(calibrator, _ctx(score=calibrator.q_hat + 1000))
        assert result.decision is Decision.ABSTAIN
        assert result.accepted is False

    def test_low_score_well_within_threshold_accepts(self):
        calibrator = _calibrator(alpha=0.1, n=20)
        result = decide(calibrator, _ctx(score=-999))
        assert result.decision is Decision.ACCEPT


class TestErroringScorerForcesAbstain:
    def test_scorer_exception_never_silently_accepts(self):
        def _boom(ctx: ToolCallContext) -> float:
            raise RuntimeError("scorer down")

        scorer = NonconformityScore("boom", _boom)
        # Build calibrator with a working scorer, then swap in a broken one
        # to simulate a live scorer failure independent of calibration data.
        good_scorer_data = [(_ctx(score=float(i)), True) for i in range(20)]
        calibrator = calibrate(_score_from_metadata, good_scorer_data, alpha=0.1, hard_minimum_size=20)
        broken_calibrator = calibrator.__class__(
            **{**calibrator.__dict__, "scorer": scorer}
        )
        result = decide(broken_calibrator, _ctx())
        assert result.decision is Decision.ABSTAIN
        assert result.scorer_errored is True
        assert result.scorer_error is not None
        assert result.score == math.inf

    def test_non_finite_score_forces_abstain_even_if_threshold_is_infinite(self):
        # Construct a calibrator whose q_hat is +inf (alpha too small for n),
        # then confirm a scorer returning +inf still abstains rather than
        # exploiting inf <= inf.
        data = [(_ctx(score=float(i)), True) for i in range(10)]
        calibrator = calibrate(_score_from_metadata, data, alpha=0.01, hard_minimum_size=10)
        assert calibrator.q_hat == math.inf

        inf_scorer = NonconformityScore("inf_scorer", lambda ctx: math.inf)
        inf_calibrator = calibrator.__class__(**{**calibrator.__dict__, "scorer": inf_scorer})
        result = decide(inf_calibrator, _ctx())
        assert result.decision is Decision.ABSTAIN


class TestGuaranteeStatement:
    def test_contains_exact_alpha_and_scope(self):
        calibrator = _calibrator(alpha=0.05, n=100)
        guarantee = build_guarantee_statement(calibrator)
        assert guarantee.alpha == 0.05
        assert guarantee.scope == "single_call"
        assert guarantee.calibration_set_size == 100
        assert "5.0%" in guarantee.text
        assert "SINGLE CALL ONLY" in guarantee.text

    def test_matches_calibrator_timestamps_and_version(self):
        calibrator = _calibrator(alpha=0.1, n=20)
        guarantee = build_guarantee_statement(calibrator)
        assert guarantee.calibration_set_version == calibrator.calibration_set_version
        assert guarantee.calibration_start == calibrator.calibration_start
        assert guarantee.calibration_end == calibrator.calibration_end

    def test_labeling_source_reflected(self):
        calibrator = _calibrator(alpha=0.1, n=20, labeling_source=LabelingSource.HUMAN)
        guarantee = build_guarantee_statement(calibrator)
        assert guarantee.labeling_source == LabelingSource.HUMAN
        assert "human" in guarantee.text

    def test_every_decision_carries_a_guarantee(self):
        calibrator = _calibrator(alpha=0.1, n=20)
        result = decide(calibrator, _ctx(score=0.0))
        assert result.guarantee.alpha == calibrator.alpha
        assert result.guarantee.calibration_set_version == calibrator.calibration_set_version

    def test_never_says_guaranteed_without_alpha_and_scope(self):
        # A crude but meaningful check against the project's hard overclaiming
        # rule (PROJECT_SPEC §2.1): the rendered text must always carry both
        # the numeric alpha and an explicit single-call scope marker.
        calibrator = _calibrator(alpha=0.2, n=20)
        guarantee = build_guarantee_statement(calibrator)
        assert "alpha=0.2" in guarantee.text
        assert "SINGLE CALL ONLY" in guarantee.text
