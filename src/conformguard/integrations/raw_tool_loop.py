"""Adapter for the raw Anthropic Messages API tool-use format, with a thin shim for OpenAI's.

This is Phase 1's only integration, deliberately: the quantile math and
the coverage/negative-control validation suites have to be proven correct
before framework breadth is worth building (PROJECT_SPEC §4.5).
"""

from __future__ import annotations

import json
from typing import Any

from conformguard.core.engine import WrapCallResult, WrappedTool
from conformguard.core.scores import ToolCallContext


def _stringify_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except TypeError:
        return str(value)



class ToolRegistry:
    """Maps tool names to wrap()-ped tools and executes tool-use blocks against them.

    Construct once per agent loop with the wrapped tools it should
    dispatch to, then feed it tool-use blocks/calls as they arrive from
    the model's response.
    """

    def __init__(self, tools: dict[str, WrappedTool]):
        self.tools = tools

    def _get_tool(self, name: str) -> WrappedTool:
        tool = self.tools.get(name)
        if tool is None:
            raise KeyError(
                f"no wrap()-ped tool registered for {name!r}; registered tools: "
                f"{sorted(self.tools)}"
            )
        return tool

    def handle_anthropic_tool_use(self, block: dict[str, Any]) -> dict[str, Any]:
        """Execute an Anthropic ``tool_use`` content block, return a ``tool_result`` block.

        ``block`` is expected in the shape returned by the Messages API:
        ``{"type": "tool_use", "id": ..., "name": ..., "input": {...}}``.
        """
        tool = self._get_tool(block["name"])
        outcome = tool(**block.get("input", {}))
        return self._to_anthropic_tool_result(block["id"], outcome)

    def handle_openai_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute an OpenAI-style function tool call, return a ``role: tool`` message.

        ``tool_call`` is expected in the shape returned by the Chat
        Completions / Responses API:
        ``{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json>"}}``.
        """
        function = tool_call["function"]
        args = json.loads(function["arguments"]) if function.get("arguments") else {}
        tool = self._get_tool(function["name"])
        outcome = tool(**args)
        return self._to_openai_tool_message(tool_call["id"], outcome)

    @staticmethod
    def _to_anthropic_tool_result(tool_use_id: str, outcome: WrapCallResult) -> dict[str, Any]:
        if outcome.accepted:
            content = _stringify_output(outcome.output)
            is_error = False
        else:
            content = f"[conformguard: abstained] {outcome.guarantee.text}"
            is_error = True
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
            "is_error": is_error,
        }

    @staticmethod
    def _to_openai_tool_message(tool_call_id: str, outcome: WrapCallResult) -> dict[str, Any]:
        if outcome.accepted:
            content = _stringify_output(outcome.output)
        else:
            content = f"[conformguard: abstained] {outcome.guarantee.text}"
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        }
