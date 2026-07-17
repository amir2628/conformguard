"""Full calibrate() -> wrap() -> accept/abstain paths, per PROJECT_SPEC §7.2."""

from unittest.mock import MagicMock

from conformguard.core.calibration import calibrate
from conformguard.core.decision import Decision
from conformguard.core.engine import wrap
from conformguard.core.scores import NonconformityScore, ToolCallContext


def _context_builder(**kwargs) -> ToolCallContext:
    return ToolCallContext(tool_name="send_email", args=kwargs)


def _score_from_risk_arg(context: ToolCallContext) -> float:
    return context.args["risk"]


class TestFullAcceptPath:
    def test_calibrate_then_wrap_accepts_low_risk_call(self):
        mock_scorer_fn = MagicMock(side_effect=_score_from_risk_arg)
        scorer = NonconformityScore("mock_scorer", mock_scorer_fn)

        data = [
            (ToolCallContext(tool_name="send_email", args={"risk": float(i) / 20}), True)
            for i in range(20)
        ]
        calibrator = calibrate(scorer, data, alpha=0.2, hard_minimum_size=20)

        sent = []

        def send_email(to: str, body: str, risk: float = 0.0) -> str:
            sent.append((to, body))
            return "sent"

        wrapped = wrap(send_email, calibrator, context_builder=_context_builder)
        result = wrapped(to="a@example.com", body="hi", risk=0.0)

        assert result.decision is Decision.ACCEPT
        assert result.output == "sent"
        assert sent == [("a@example.com", "hi")]
        # The scorer was actually invoked at decision time (not bypassed).
        assert mock_scorer_fn.called


class TestFullAbstainPath:
    def test_calibrate_then_wrap_abstains_high_risk_call(self):
        mock_scorer_fn = MagicMock(side_effect=_score_from_risk_arg)
        scorer = NonconformityScore("mock_scorer", mock_scorer_fn)

        data = [
            (ToolCallContext(tool_name="send_email", args={"risk": float(i) / 20}), True)
            for i in range(20)
        ]
        calibrator = calibrate(scorer, data, alpha=0.2, hard_minimum_size=20)

        sent = []

        def send_email(to: str, body: str, risk: float = 0.0) -> str:
            sent.append((to, body))
            return "sent"

        wrapped = wrap(send_email, calibrator, context_builder=_context_builder)
        result = wrapped(to="a@example.com", body="hi", risk=999.0)

        assert result.decision is Decision.ABSTAIN
        assert result.output is None
        assert sent == []  # underlying tool never called on abstain
