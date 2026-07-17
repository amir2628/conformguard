import pytest

from conformguard.core.calibration import calibrate
from conformguard.core.decision import Decision
from conformguard.core.engine import AbstainedError, wrap
from conformguard.core.scores import ToolCallContext


def _score_from_metadata(context: ToolCallContext) -> float:
    return context.metadata["score"]


def _calibrator(alpha=0.1, n=20):
    data = [(ToolCallContext(tool_name="search", args={}, metadata={"score": float(i)}), True) for i in range(n)]
    return calibrate(_score_from_metadata, data, alpha=alpha, hard_minimum_size=n)


def _context_builder(**kwargs) -> ToolCallContext:
    return ToolCallContext(tool_name="search", args=kwargs, metadata={"score": kwargs.get("score", 0.0)})


class TestAcceptCallsUnderlyingFunctionUnmodified:
    def test_accepted_call_invokes_tool_fn_with_original_args(self):
        calls = []

        def tool_fn(query: str, score: float) -> str:
            calls.append((query, score))
            return f"results for {query}"

        calibrator = _calibrator()
        wrapped = wrap(tool_fn, calibrator, context_builder=_context_builder)
        result = wrapped(query="weather", score=0.0)

        assert result.accepted
        assert result.output == "results for weather"
        assert calls == [("weather", 0.0)]

    def test_wrapping_does_not_modify_the_original_function(self):
        def tool_fn(x: int) -> int:
            return x * 2

        calibrator = _calibrator()
        wrap(tool_fn, calibrator, context_builder=lambda **kw: ToolCallContext(tool_name="t", metadata={"score": 0}))
        assert tool_fn(21) == 42  # unwrapped behavior untouched


class TestAbstainEscalate:
    def test_abstain_does_not_call_underlying_function(self):
        calls = []

        def tool_fn(score: float) -> str:
            calls.append(score)
            return "should not happen"

        calibrator = _calibrator(alpha=0.1, n=20)
        wrapped = wrap(tool_fn, calibrator, on_abstain="escalate", context_builder=_context_builder)
        result = wrapped(score=calibrator.q_hat + 1000)

        assert result.decision is Decision.ABSTAIN
        assert result.accepted is False
        assert result.output is None
        assert calls == []
        assert result.guarantee.alpha == calibrator.alpha


class TestAbstainRaise:
    def test_raises_abstained_error_with_result_attached(self):
        def tool_fn(score: float) -> str:
            return "should not happen"

        calibrator = _calibrator(alpha=0.1, n=20)
        wrapped = wrap(tool_fn, calibrator, on_abstain="raise", context_builder=_context_builder)

        with pytest.raises(AbstainedError) as exc_info:
            wrapped(score=calibrator.q_hat + 1000)
        assert exc_info.value.result.decision is Decision.ABSTAIN
        assert exc_info.value.result.guarantee.alpha == calibrator.alpha


class TestAbstainCallback:
    def test_callback_receives_wrap_result_and_return_becomes_output(self):
        received = []

        def fallback(wrap_result):
            received.append(wrap_result)
            return "fallback value"

        def tool_fn(score: float) -> str:
            return "should not happen"

        calibrator = _calibrator(alpha=0.1, n=20)
        wrapped = wrap(tool_fn, calibrator, on_abstain=fallback, context_builder=_context_builder)
        result = wrapped(score=calibrator.q_hat + 1000)

        assert result.output == "fallback value"
        assert result.decision is Decision.ABSTAIN
        assert len(received) == 1
        assert received[0].decision is Decision.ABSTAIN


class TestScorerErrorForcesAbstainThroughWrap:
    def test_erroring_scorer_abstains_and_does_not_call_tool(self):
        calls = []

        def tool_fn(**kwargs) -> str:
            calls.append(kwargs)
            return "should not happen"

        calibrator = _calibrator(alpha=0.1, n=20)

        def broken_context_builder(**kwargs) -> ToolCallContext:
            # metadata deliberately missing "score" -> scorer's dict lookup raises
            return ToolCallContext(tool_name="search", args=kwargs, metadata={})

        wrapped = wrap(tool_fn, calibrator, context_builder=broken_context_builder)
        result = wrapped(query="x")

        assert result.decision is Decision.ABSTAIN
        assert result.scorer_errored is True
        assert calls == []


class TestPositionalArgsRequireContextBuilder:
    def test_positional_args_without_context_builder_raises_type_error(self):
        def tool_fn(x: int) -> int:
            return x

        calibrator = _calibrator()
        wrapped = wrap(tool_fn, calibrator)
        with pytest.raises(TypeError):
            wrapped(5)


class TestCallWithContext:
    def test_bypasses_context_builder_and_uses_given_context_directly(self):
        calls = []

        def tool_fn(city: str) -> str:
            calls.append(city)
            return f"weather for {city}"

        calibrator = _calibrator(alpha=0.1, n=20)
        wrapped = wrap(tool_fn, calibrator, context_builder=_context_builder)

        context = ToolCallContext(tool_name="search", args={"city": "Lisbon"}, metadata={"score": 0.0})
        result = wrapped.call_with_context(context)

        assert result.accepted
        assert result.output == "weather for Lisbon"
        assert calls == ["Lisbon"]

    def test_tool_fn_called_with_only_context_args_not_metadata(self):
        received = {}

        def tool_fn(**kwargs) -> str:
            received.update(kwargs)
            return "ok"

        calibrator = _calibrator(alpha=0.1, n=20)
        wrapped = wrap(tool_fn, calibrator)

        context = ToolCallContext(
            tool_name="search", args={"city": "Lisbon"}, metadata={"score": 0.0, "extra_signal": 42}
        )
        wrapped.call_with_context(context)

        assert received == {"city": "Lisbon"}

    def test_abstains_when_context_score_exceeds_threshold(self):
        calibrator = _calibrator(alpha=0.1, n=20)
        wrapped = wrap(lambda **kw: "should not run", calibrator)

        context = ToolCallContext(tool_name="search", args={}, metadata={"score": calibrator.q_hat + 1000})
        result = wrapped.call_with_context(context)

        assert result.decision is Decision.ABSTAIN
