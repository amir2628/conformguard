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
