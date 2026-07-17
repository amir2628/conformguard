"""conformguard: calibrated accept / abstain-and-escalate decisions for agent tool calls.

Backed by split conformal prediction's finite-sample coverage guarantee
(Angelopoulos & Bates, arXiv:2107.07511) rather than a heuristic
confidence threshold. See docs/guarantee_scope.md for exactly what is,
and is not, guaranteed.
"""

from importlib.metadata import version as _version

from conformguard.core.calibration import (
    Calibrator,
    CalibrationScoringError,
    InsufficientCalibrationDataError,
    calibrate,
)
from conformguard.core.decision import Decision, GuaranteeStatement, WrapResult, decide
from conformguard.core.engine import AbstainedError, WrapCallResult, WrappedTool, wrap
from conformguard.core.multi_check import (
    CheckResult,
    MultiCheckCalibrator,
    MultiCheckGuaranteeStatement,
    MultiCheckWrapResult,
    calibrate_multi_check,
    decide_multi_check,
)
from conformguard.core.scores import (
    NonconformityScore,
    ToolCallContext,
    logprob_score,
    make_judge_score,
    schema_validity_score,
)
from conformguard.storage.calibration_store import (
    CalibrationExample,
    CalibrationStore,
    LabelingSource,
)

__version__ = _version("conformguard")

__all__ = [
    "AbstainedError",
    "CalibrationExample",
    "CalibrationScoringError",
    "CalibrationStore",
    "Calibrator",
    "CheckResult",
    "Decision",
    "GuaranteeStatement",
    "InsufficientCalibrationDataError",
    "LabelingSource",
    "MultiCheckCalibrator",
    "MultiCheckGuaranteeStatement",
    "MultiCheckWrapResult",
    "NonconformityScore",
    "ToolCallContext",
    "WrapCallResult",
    "WrapResult",
    "WrappedTool",
    "calibrate",
    "calibrate_multi_check",
    "decide",
    "decide_multi_check",
    "logprob_score",
    "make_judge_score",
    "schema_validity_score",
    "wrap",
]
