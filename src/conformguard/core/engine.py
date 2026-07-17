"""wrap(): the accept/abstain gate around a single tool call.

Design constraints (PROJECT_SPEC §4.2):
- Wrapping never requires rewriting the underlying tool function; it is
  called completely unmodified on accept.
- A scorer error, or a non-finite score, always forces abstain -- this is
  enforced once, centrally, in core/decision.py's decide(), and simply
  inherited here rather than re-implemented.
- Calibration cannot be skipped or defaulted to empty: wrap() requires an
  already-built Calibrator (see core/calibration.py's own hard-minimum
  enforcement), never a bare alpha or an empty calibration set.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict

from conformguard.core.calibration import Calibrator
from conformguard.core.decision import Decision, GuaranteeStatement, WrapResult, decide
from conformguard.core.scores import ToolCallContext

OnAbstain = Union[Literal["escalate", "raise"], Callable[[WrapResult], Any]]


class AbstainedError(RuntimeError):
    """Raised when on_abstain="raise" and the calibrator abstains on a call.

    Carries the full WrapResult (score, threshold, guarantee statement) so
    a caller that catches this can still inspect exactly why the call was
    refused, not just that it was.
    """

    def __init__(self, result: WrapResult):
        super().__init__(
            f"conformguard abstained on tool_name={result.tool_name!r} "
            f"(score={result.score}, threshold={result.threshold}). {result.guarantee.text}"
        )
        self.result = result


class WrapCallResult(BaseModel):
    """Full outcome of one wrap()-ped call: the accept/abstain decision plus the tool's own output.

    ``output`` is the underlying tool function's return value when
    accepted, the ``on_abstain`` callback's return value when a callback
    is used and abstains, or None otherwise.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    tool_name: str
    decision: Decision
    score: float
    threshold: float
    scorer_name: str
    scorer_errored: bool
    scorer_error: str | None
    guarantee: GuaranteeStatement
    output: Any = None

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.ACCEPT

    @classmethod
    def _from_decision(cls, decision_result: WrapResult, output: Any = None) -> WrapCallResult:
        return cls(
            tool_name=decision_result.tool_name,
            decision=decision_result.decision,
            score=decision_result.score,
            threshold=decision_result.threshold,
            scorer_name=decision_result.scorer_name,
            scorer_errored=decision_result.scorer_errored,
            scorer_error=decision_result.scorer_error,
            guarantee=decision_result.guarantee,
            output=output,
        )


class WrappedTool:
    """Callable produced by wrap(). Construct via wrap(), not directly."""

    def __init__(
        self,
        tool_fn: Callable[..., Any],
        calibrator: Calibrator,
        on_abstain: OnAbstain = "escalate",
        tool_name: str | None = None,
        context_builder: Callable[..., ToolCallContext] | None = None,
    ):
        self.tool_fn = tool_fn
        self.calibrator = calibrator
        self.on_abstain = on_abstain
        self.tool_name = tool_name or getattr(tool_fn, "__name__", "tool")
        self.context_builder = context_builder

    def _build_context(self, *args: Any, **kwargs: Any) -> ToolCallContext:
        if self.context_builder is not None:
            return self.context_builder(*args, **kwargs)
        if args:
            raise TypeError(
                "wrap()-ped tools must be called with keyword arguments only, unless a "
                "context_builder is supplied to interpret positional arguments into a "
                "ToolCallContext"
            )
        return ToolCallContext(tool_name=self.tool_name, args=dict(kwargs))

    def __call__(self, *args: Any, **kwargs: Any) -> WrapCallResult:
        context = self._build_context(*args, **kwargs)
        decision_result = decide(self.calibrator, context)

        if decision_result.decision is Decision.ACCEPT:
            output = self.tool_fn(*args, **kwargs)
            return WrapCallResult._from_decision(decision_result, output=output)

        if self.on_abstain == "raise":
            raise AbstainedError(decision_result)
        if self.on_abstain == "escalate":
            return WrapCallResult._from_decision(decision_result, output=None)
        if callable(self.on_abstain):
            fallback_output = self.on_abstain(decision_result)
            return WrapCallResult._from_decision(decision_result, output=fallback_output)
        raise TypeError(f"on_abstain must be 'escalate', 'raise', or a callable, got {self.on_abstain!r}")


def wrap(
    tool_fn: Callable[..., Any],
    calibrator: Calibrator,
    on_abstain: OnAbstain = "escalate",
    tool_name: str | None = None,
    context_builder: Callable[..., ToolCallContext] | None = None,
) -> WrappedTool:
    """Wrap a single tool function with a calibrated accept/abstain gate.

    Args:
        tool_fn: The tool function to wrap, called unmodified on accept.
        calibrator: A Calibrator built by core.calibration.calibrate().
        on_abstain: "escalate" (default) returns a WrapCallResult with
            decision="abstain" and output=None, leaving escalation to the
            caller; "raise" raises AbstainedError; or a callable invoked
            with the WrapResult, whose return value becomes .output.
        tool_name: Overrides the tool name used for scoring/logging
            (defaults to tool_fn.__name__).
        context_builder: Builds a ToolCallContext from the call's
            arguments. Defaults to treating all keyword arguments as
            ToolCallContext.args; required if the wrapped tool is called
            positionally.
    """
    return WrappedTool(
        tool_fn=tool_fn,
        calibrator=calibrator,
        on_abstain=on_abstain,
        tool_name=tool_name,
        context_builder=context_builder,
    )
