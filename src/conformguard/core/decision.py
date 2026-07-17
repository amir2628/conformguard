"""Accept/abstain decision logic and the guarantee statement attached to it.

Every decision carries the guarantee statement as data, not just as a log
line -- see PROJECT_SPEC §4.2.2. This is treated as the single most
important design decision in the project: a bare "confidence: 0.87" with
no attached scope is exactly the failure mode this library exists to
replace.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict

from conformguard.core.calibration import Calibrator
from conformguard.core.scores import ToolCallContext
from conformguard.storage.calibration_store import LabelingSource


class Decision(str, Enum):
    ACCEPT = "accept"
    ABSTAIN = "abstain"


class GuaranteeStatement(BaseModel):
    """Machine-readable guarantee scope, plus a human-readable rendering."""

    model_config = ConfigDict(frozen=True)

    alpha: float
    scope: str = "single_call"
    calibration_set_size: int
    calibration_set_version: str
    calibration_start: datetime
    calibration_end: datetime
    labeling_source: LabelingSource
    exchangeability_assumed: bool = True
    text: str


class WrapResult(BaseModel):
    """The full outcome of one accept/abstain decision."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    decision: Decision
    score: float
    threshold: float
    scorer_name: str
    scorer_errored: bool
    scorer_error: str | None = None
    guarantee: GuaranteeStatement

    @property
    def accepted(self) -> bool:
        return self.decision is Decision.ACCEPT


def build_guarantee_statement(calibrator: Calibrator) -> GuaranteeStatement:
    """Construct the exact, scoped guarantee statement for a Calibrator."""
    pct = (1 - calibrator.alpha) * 100
    error_pct = calibrator.alpha * 100
    start = calibrator.calibration_start.date().isoformat()
    end = calibrator.calibration_end.date().isoformat()

    text = (
        f"Under the assumption that this call is exchangeable with the "
        f"{calibrator.n_calibration}-example calibration set (labeling source: "
        f"{calibrator.labeling_source.value}; collected {start} through {end}), "
        f"this accept/abstain decision is wrong at most {error_pct:.1f}% of the "
        f"time (alpha={calibrator.alpha}), for THIS SINGLE CALL ONLY. This is not "
        f"a guarantee about any multi-step task this call is part of, and it holds "
        f"only if the exchangeability assumption is not violated -- see "
        f"docs/guarantee_scope.md. [{pct:.1f}% target coverage]"
    )

    return GuaranteeStatement(
        alpha=calibrator.alpha,
        scope="single_call",
        calibration_set_size=calibrator.n_calibration,
        calibration_set_version=calibrator.calibration_set_version,
        calibration_start=calibrator.calibration_start,
        calibration_end=calibrator.calibration_end,
        labeling_source=calibrator.labeling_source,
        exchangeability_assumed=True,
        text=text,
    )


def decide(calibrator: Calibrator, context: ToolCallContext) -> WrapResult:
    """Score ``context`` against ``calibrator`` and produce an accept/abstain decision.

    A scorer error, or a non-finite score, always forces abstain -- never
    a silent accept -- regardless of where the threshold happens to fall
    (including the edge case where q_hat itself is +inf, see
    core/quantile.py).
    """
    outcome = calibrator.scorer.safe(context)

    if outcome.usable and outcome.score <= calibrator.q_hat:
        decision = Decision.ACCEPT
    else:
        decision = Decision.ABSTAIN

    return WrapResult(
        tool_name=context.tool_name,
        decision=decision,
        score=outcome.score,
        threshold=calibrator.q_hat,
        scorer_name=calibrator.scorer.name,
        scorer_errored=outcome.errored,
        scorer_error=repr(outcome.error) if outcome.error is not None else None,
        guarantee=build_guarantee_statement(calibrator),
    )


__all__ = [
    "Decision",
    "GuaranteeStatement",
    "WrapResult",
    "build_guarantee_statement",
    "decide",
]
