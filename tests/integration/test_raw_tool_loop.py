import json
from types import SimpleNamespace

import pytest

from conformguard.core.calibration import calibrate
from conformguard.core.engine import wrap
from conformguard.core.scores import ToolCallContext
from conformguard.integrations.raw_tool_loop import (
    NoToolCallProducedError,
    ToolRegistry,
    mean_completion_logprob,
)


def _score_from_args(context: ToolCallContext) -> float:
    # Deliberately simple deterministic scorer for reproducible test behavior:
    # nonconformity == the "risk" argument passed to the tool.
    return context.args.get("risk", 0.0)


def _context_builder(**kwargs) -> ToolCallContext:
    return ToolCallContext(tool_name="get_weather", args=kwargs)


def _calibrator(alpha=0.2, n=20):
    data = [
        (ToolCallContext(tool_name="get_weather", args={"risk": float(i) / n}), True) for i in range(n)
    ]
    return calibrate(_score_from_args, data, alpha=alpha, hard_minimum_size=n)


@pytest.fixture
def registry():
    def get_weather(city: str, risk: float = 0.0) -> str:
        return f"sunny in {city}"

    calibrator = _calibrator()
    wrapped = wrap(get_weather, calibrator, context_builder=_context_builder)
    return ToolRegistry({"get_weather": wrapped})


class TestAnthropicToolUseAdapter:
    def test_accept_path_returns_tool_result_with_content(self, registry):
        block = {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"city": "Berlin", "risk": 0.0},
        }
        result = registry.handle_anthropic_tool_use(block)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_123"
        assert result["is_error"] is False
        assert "sunny in Berlin" in result["content"]

    def test_abstain_path_returns_error_tool_result_with_guarantee_text(self, registry):
        block = {
            "type": "tool_use",
            "id": "toolu_456",
            "name": "get_weather",
            "input": {"city": "Berlin", "risk": 999.0},
        }
        result = registry.handle_anthropic_tool_use(block)
        assert result["is_error"] is True
        assert "conformguard" in result["content"]
        assert "alpha=" in result["content"]

    def test_unregistered_tool_raises_key_error(self, registry):
        block = {"type": "tool_use", "id": "x", "name": "not_registered", "input": {}}
        with pytest.raises(KeyError):
            registry.handle_anthropic_tool_use(block)

    def test_underlying_tool_behavior_unchanged_when_accepted(self, registry):
        block = {
            "type": "tool_use",
            "id": "toolu_789",
            "name": "get_weather",
            "input": {"city": "Tokyo", "risk": 0.0},
        }
        result = registry.handle_anthropic_tool_use(block)
        assert result["content"] == "sunny in Tokyo"


class TestOpenAIToolCallAdapter:
    def test_accept_path_returns_tool_role_message(self, registry):
        tool_call = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris", "risk": 0.0})},
        }
        message = registry.handle_openai_tool_call(tool_call)
        assert message["role"] == "tool"
        assert message["tool_call_id"] == "call_abc"
        assert "sunny in Paris" in message["content"]

    def test_abstain_path_returns_guarantee_text_in_content(self, registry):
        tool_call = {
            "id": "call_def",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Paris", "risk": 999.0})},
        }
        message = registry.handle_openai_tool_call(tool_call)
        assert "conformguard" in message["content"]

    def test_empty_arguments_string_handled(self, registry):
        def no_arg_tool() -> str:
            return "ok"

        calibrator = _calibrator()
        wrapped = wrap(no_arg_tool, calibrator, context_builder=lambda **kw: ToolCallContext(tool_name="no_arg_tool", args={"risk": 0.0}))
        reg = ToolRegistry({"no_arg_tool": wrapped})
        tool_call = {"id": "call_ghi", "type": "function", "function": {"name": "no_arg_tool", "arguments": ""}}
        message = reg.handle_openai_tool_call(tool_call)
        assert message["content"] == "ok"


class TestNoCodeChangesRequiredToUnderlyingTool:
    def test_wrapping_requires_only_a_scorer_and_calibrator(self, registry):
        # The fixture's get_weather function is a plain, unmodified function;
        # nothing about its definition changed to make it wrap()-able.
        block = {"type": "tool_use", "id": "x", "name": "get_weather", "input": {"city": "Oslo", "risk": 0.0}}
        result = registry.handle_anthropic_tool_use(block)
        assert result["content"] == "sunny in Oslo"


class TestExtraMetadataBypassesContextBuilder:
    """Covers the gap found wiring this adapter up to a real local model:

    some OpenAI-compatible servers (Ollama included) attach per-token
    logprobs to the response *choice*, not to the individual tool_call
    object, so a scorer that wants that signal needs a channel that
    doesn't depend on it being one of the tool's own call arguments.
    """

    def _metadata_scorer(self, context: ToolCallContext) -> float:
        return context.metadata["risk_from_response"]

    def _build_registry(self, alpha=0.2, n=20):
        data = [
            (ToolCallContext(tool_name="get_weather", metadata={"risk_from_response": float(i) / n}), True)
            for i in range(n)
        ]
        calibrator = calibrate(self._metadata_scorer, data, alpha=alpha, hard_minimum_size=n)

        def get_weather(city: str) -> str:
            return f"sunny in {city}"

        wrapped = wrap(get_weather, calibrator)  # no context_builder: extra_metadata must bypass it
        return ToolRegistry({"get_weather": wrapped}), calibrator

    def test_openai_extra_metadata_reaches_the_scorer(self):
        registry, calibrator = self._build_registry()
        tool_call = {
            "id": "call_x",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})},
        }
        accepted = registry.handle_openai_tool_call(tool_call, extra_metadata={"risk_from_response": 0.0})
        assert "sunny in Lisbon" in accepted["content"]

        abstained = registry.handle_openai_tool_call(tool_call, extra_metadata={"risk_from_response": 999.0})
        assert "conformguard" in abstained["content"]

    def test_anthropic_extra_metadata_reaches_the_scorer(self):
        registry, calibrator = self._build_registry()
        block = {"type": "tool_use", "id": "x", "name": "get_weather", "input": {"city": "Lisbon"}}

        accepted = registry.handle_anthropic_tool_use(block, extra_metadata={"risk_from_response": 0.0})
        assert accepted["is_error"] is False

        abstained = registry.handle_anthropic_tool_use(block, extra_metadata={"risk_from_response": 999.0})
        assert abstained["is_error"] is True

    def test_underlying_tool_still_called_with_only_its_own_args(self):
        registry, _ = self._build_registry()
        calls = []

        def spy_get_weather(city: str) -> str:
            calls.append(city)
            return "ok"

        # Rebuild with a spy so we can assert exactly what tool_fn received.
        data = [
            (ToolCallContext(tool_name="get_weather", metadata={"risk_from_response": float(i) / 20}), True)
            for i in range(20)
        ]
        calibrator = calibrate(self._metadata_scorer, data, alpha=0.2, hard_minimum_size=20)
        wrapped = wrap(spy_get_weather, calibrator)
        registry = ToolRegistry({"get_weather": wrapped})

        tool_call = {
            "id": "call_y",
            "type": "function",
            "function": {"name": "get_weather", "arguments": json.dumps({"city": "Lisbon"})},
        }
        registry.handle_openai_tool_call(tool_call, extra_metadata={"risk_from_response": 0.0})
        assert calls == ["Lisbon"]  # risk_from_response never leaked into the tool's own call


class TestMeanCompletionLogprob:
    def _sdk_shaped_choice(self, logprobs_values):
        content = [SimpleNamespace(token=f"t{i}", logprob=lp) for i, lp in enumerate(logprobs_values)]
        return SimpleNamespace(logprobs=SimpleNamespace(content=content))

    def _dict_shaped_choice(self, logprobs_values):
        content = [{"token": f"t{i}", "logprob": lp} for i, lp in enumerate(logprobs_values)]
        return {"logprobs": {"content": content}}

    def test_averages_sdk_shaped_choice(self):
        choice = self._sdk_shaped_choice([-0.1, -0.2, -0.3])
        assert mean_completion_logprob(choice) == pytest.approx(-0.2)

    def test_averages_dict_shaped_choice(self):
        choice = self._dict_shaped_choice([-0.1, -0.2, -0.3])
        assert mean_completion_logprob(choice) == pytest.approx(-0.2)

    def test_none_when_logprobs_missing(self):
        choice = SimpleNamespace(logprobs=None)
        assert mean_completion_logprob(choice) is None

    def test_none_when_content_empty(self):
        choice = SimpleNamespace(logprobs=SimpleNamespace(content=[]))
        assert mean_completion_logprob(choice) is None

    def test_none_when_content_missing_entirely(self):
        choice = {"logprobs": {}}
        assert mean_completion_logprob(choice) is None

    def test_averages_over_all_tokens_not_a_subset(self):
        # Pins the "aggregate over everything, don't restrict to argument
        # tokens" decision documented in the function's own docstring and
        # in docs/real_world_validation.md: a single low-confidence
        # reasoning token buried among many confident ones should visibly
        # move the mean, not be diluted away entirely.
        confident = [-0.0001] * 20
        one_uncertain = confident + [-3.5]
        assert mean_completion_logprob(self._sdk_shaped_choice(one_uncertain)) < mean_completion_logprob(
            self._sdk_shaped_choice(confident)
        )
