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


class NoToolCallProducedError(RuntimeError):
    """Raised when a tool call was required but the model did not produce one.

    This is a different failure mode from an abstain, and must not be
    confused with one: an abstain means the model DID attempt a tool
    call and conformguard's calibrated threshold rejected it -- there was
    a score, and the score said no. This error means the model never
    attempted a tool call at all, so there was nothing for conformguard
    to score in the first place. Silently treating "no tool call" as an
    accept, folding it into an abstain, or just ignoring it, would defeat
    this library's "never accept without a score" principle just as
    surely as a silent pass-through would -- so ``ToolRegistry`` raises
    this loudly instead, whenever a caller has said a tool call was
    required (see ``handle_openai_choice``/``handle_anthropic_message``,
    ``required=True``) and none arrived.

    Real-world trigger, confirmed directly (not just anticipated): some
    models, served through some inference servers, can fail to produce a
    tool call at all even when explicitly asked and even when the
    request sets a "you must call a tool" flag -- see
    docs/real_world_validation.md's Command R7B findings for a concrete,
    root-caused example (an inference-server-side chat template bug, not
    a model limitation) discovered while validating this library against
    real local models.
    """

    def __init__(self, message: str, *, finish_reason: str | None = None, content: str | None = None):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.content = content


def _get(obj: Any, key: str) -> Any:
    """Duck-typed field access: works on a dict or an SDK object alike."""
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _extract_openai_tool_calls(choice: Any) -> list[dict[str, Any]]:
    message = _get(choice, "message")
    tool_calls = _get(message, "tool_calls") or []
    return [tc.model_dump() if hasattr(tc, "model_dump") else tc for tc in tool_calls]


def _extract_anthropic_tool_use_blocks(message: Any) -> list[dict[str, Any]]:
    content = _get(message, "content") or []
    blocks = []
    for block in content:
        if _get(block, "type") == "tool_use":
            blocks.append(block.model_dump() if hasattr(block, "model_dump") else block)
    return blocks


def _extract_anthropic_text(message: Any) -> str | None:
    content = _get(message, "content") or []
    texts = [_get(block, "text") for block in content if _get(block, "type") == "text"]
    texts = [t for t in texts if t]
    return "\n".join(texts) if texts else None


def mean_completion_logprob(choice: Any) -> float | None:
    """Aggregate an OpenAI-shape chat completion choice's per-token logprobs into one scalar.

    Returns the mean of ``choice.logprobs.content[*].logprob`` (or the
    equivalent dict-shaped path, if ``choice`` is a raw JSON dict rather
    than an SDK object), or ``None`` if the choice has no logprobs at all
    (server didn't return them, or ``logprobs=True`` wasn't requested).
    The result is meant to be fed directly into
    ``ToolCallContext.metadata["model_logprob"]`` for the built-in
    ``logprob_score``.

    This averages over every token in the raw completion -- including any
    provider-specific tool-call wrapper syntax (e.g. Ollama/Qwen's
    ``<tool_call>...</tool_call>`` text) and the JSON structural tokens
    around the arguments -- rather than restricting to just the tokens
    inside the function-call arguments. That's a deliberate, empirically
    checked choice, not an oversight: when a model reasons or hedges in
    plain text before committing to a tool call (e.g. guessing at an
    ambiguous request), that uncertainty shows up as low per-token
    logprobs in the *reasoning* tokens, not in the argument-value tokens
    themselves -- by the time the model writes out its already-decided
    answer as a JSON string, it typically does so confidently regardless
    of how uncertain the decision leading up to it was. Restricting
    aggregation to just the argument tokens was tried and found to
    silently lose exactly this signal; see docs/real_world_validation.md
    for the real, repeated trials this conclusion is based on.
    """
    logprobs = choice.get("logprobs") if isinstance(choice, dict) else getattr(choice, "logprobs", None)
    if logprobs is None:
        return None
    content = logprobs.get("content") if isinstance(logprobs, dict) else getattr(logprobs, "content", None)
    if not content:
        return None

    values: list[float] = []
    for entry in content:
        logprob = entry.get("logprob") if isinstance(entry, dict) else getattr(entry, "logprob", None)
        if logprob is None:
            return None
        values.append(float(logprob))
    return sum(values) / len(values)


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

    def handle_anthropic_tool_use(
        self, block: dict[str, Any], extra_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute an Anthropic ``tool_use`` content block, return a ``tool_result`` block.

        ``block`` is expected in the shape returned by the Messages API:
        ``{"type": "tool_use", "id": ..., "name": ..., "input": {...}}``.

        ``extra_metadata``, if given, is merged into the ToolCallContext's
        metadata and the tool's own context_builder (if any) is bypassed.
        This is the escape hatch for scoring signal that a real API
        attaches to the response/choice rather than to the individual
        tool-use block -- see handle_openai_tool_call for why this exists.
        """
        tool = self._get_tool(block["name"])
        args = block.get("input", {})
        if extra_metadata is not None:
            context = ToolCallContext(tool_name=block["name"], args=args, metadata=extra_metadata)
            outcome = tool.call_with_context(context)
        else:
            outcome = tool(**args)
        return self._to_anthropic_tool_result(block["id"], outcome)

    def handle_openai_tool_call(
        self, tool_call: dict[str, Any], extra_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute an OpenAI-style function tool call, return a ``role: tool`` message.

        ``tool_call`` is expected in the shape returned by the Chat
        Completions / Responses API:
        ``{"id": ..., "type": "function", "function": {"name": ..., "arguments": "<json>"}}``.

        ``extra_metadata``, if given, is merged into the ToolCallContext's
        metadata and the tool's own context_builder (if any) is bypassed
        in favor of a context built directly from ``tool_call`` + this
        metadata. This matters in practice: some real OpenAI-compatible
        servers (Ollama's included) return per-token logprobs at the
        *choice* level, not attached to the individual tool_call object,
        so a scorer that wants that signal needs a way to receive it that
        does not depend on it being part of the tool's own arguments.
        """
        function = tool_call["function"]
        args = json.loads(function["arguments"]) if function.get("arguments") else {}
        tool = self._get_tool(function["name"])
        if extra_metadata is not None:
            context = ToolCallContext(tool_name=function["name"], args=args, metadata=extra_metadata)
            outcome = tool.call_with_context(context)
        else:
            outcome = tool(**args)
        return self._to_openai_tool_message(tool_call["id"], outcome)

    def handle_openai_choice(
        self,
        choice: Any,
        extra_metadata: dict[str, Any] | None = None,
        required: bool = False,
    ) -> list[dict[str, Any]]:
        """Dispatch every tool call in an OpenAI-shape chat completion choice.

        Accepts a full ``choice`` (SDK object or raw dict), extracts its
        ``tool_calls`` (there may be zero, one, or several -- some models
        emit more than one tool call per response when uncertain, e.g.
        one per candidate answer), and dispatches each through
        :meth:`handle_openai_tool_call`.

        Args:
            required: if True and the choice contains no tool calls,
                raises :class:`NoToolCallProducedError` instead of
                silently returning an empty list. Use this whenever your
                calling code has already decided a tool call was
                mandatory here (e.g. because the request set
                ``tool_choice="required"``, or this step of your workflow
                has no valid non-tool-call response) -- do not rely on
                the request-level flag alone, since some servers accept
                it without honoring it (see the class docstring on
                :class:`NoToolCallProducedError`).

        Returns:
            One ``role: tool`` message per tool call (possibly empty, if
            ``required`` is False and none were produced).
        """
        tool_calls = _extract_openai_tool_calls(choice)
        if required and not tool_calls:
            raise NoToolCallProducedError(
                f"expected a tool call but the model produced none "
                f"(finish_reason={_get(choice, 'finish_reason')!r}). This is not an abstain -- "
                f"conformguard never saw a call to score, so there is no calibrated decision to "
                f"report here. Inspect .content for what the model said instead.",
                finish_reason=_get(choice, "finish_reason"),
                content=_get(_get(choice, "message"), "content"),
            )
        return [self.handle_openai_tool_call(tc, extra_metadata=extra_metadata) for tc in tool_calls]

    def handle_anthropic_message(
        self,
        message: Any,
        extra_metadata: dict[str, Any] | None = None,
        required: bool = False,
    ) -> list[dict[str, Any]]:
        """Dispatch every ``tool_use`` block in an Anthropic Messages API response.

        Accepts a full ``message`` (SDK object or raw dict), extracts its
        ``tool_use`` content blocks (there may be zero, one, or several),
        and dispatches each through :meth:`handle_anthropic_tool_use`.

        Args:
            required: if True and the message contains no ``tool_use``
                blocks, raises :class:`NoToolCallProducedError` instead
                of silently returning an empty list. See
                :meth:`handle_openai_choice` for the same argument's
                rationale.

        Returns:
            One ``tool_result`` block per ``tool_use`` block (possibly
            empty, if ``required`` is False and none were produced).
        """
        tool_use_blocks = _extract_anthropic_tool_use_blocks(message)
        if required and not tool_use_blocks:
            raise NoToolCallProducedError(
                f"expected a tool_use block but the model produced none "
                f"(stop_reason={_get(message, 'stop_reason')!r}). This is not an abstain -- "
                f"conformguard never saw a call to score, so there is no calibrated decision to "
                f"report here. Inspect .content for what the model said instead.",
                finish_reason=_get(message, "stop_reason"),
                content=_extract_anthropic_text(message),
            )
        return [self.handle_anthropic_tool_use(block, extra_metadata=extra_metadata) for block in tool_use_blocks]

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
