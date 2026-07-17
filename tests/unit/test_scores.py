import math

import pytest
from pydantic import BaseModel

from conformguard.core.scores import (
    NonconformityScore,
    ToolCallContext,
    logprob_score,
    make_judge_score,
    schema_validity_score,
)


def _ctx(**metadata) -> ToolCallContext:
    return ToolCallContext(tool_name="dummy_tool", args={"x": 1}, metadata=metadata)


class TestNonconformityScoreWrapper:
    def test_call_returns_raw_value(self):
        scorer = NonconformityScore(name="const", fn=lambda ctx: 0.5)
        assert scorer(_ctx()) == 0.5

    def test_safe_returns_usable_outcome_on_success(self):
        scorer = NonconformityScore(name="const", fn=lambda ctx: 0.5)
        outcome = scorer.safe(_ctx())
        assert outcome.usable
        assert outcome.errored is False
        assert outcome.score == 0.5

    def test_safe_forces_abstain_on_exception(self):
        def _boom(ctx: ToolCallContext) -> float:
            raise RuntimeError("scorer blew up")

        scorer = NonconformityScore(name="broken", fn=_boom)
        outcome = scorer.safe(_ctx())
        assert not outcome.usable
        assert outcome.errored is True
        assert outcome.score == math.inf
        assert isinstance(outcome.error, RuntimeError)

    def test_safe_forces_abstain_on_non_finite_return(self):
        scorer = NonconformityScore(name="nan_scorer", fn=lambda ctx: math.nan)
        outcome = scorer.safe(_ctx())
        assert not outcome.usable
        assert outcome.errored is False  # did not raise, but still unusable
        assert outcome.score == math.inf

    def test_safe_forces_abstain_on_infinite_return(self):
        scorer = NonconformityScore(name="inf_scorer", fn=lambda ctx: math.inf)
        outcome = scorer.safe(_ctx())
        assert not outcome.usable
        assert outcome.score == math.inf


class TestLogprobScore:
    def test_computes_one_minus_exp_logprob(self):
        # logprob = 0 -> exp(0) = 1 -> nonconformity = 0
        assert logprob_score(_ctx(model_logprob=0.0)) == pytest.approx(0.0)

    def test_negative_logprob_gives_positive_score(self):
        import math as _m

        logprob = -0.5
        expected = 1.0 - _m.exp(logprob)
        assert logprob_score(_ctx(model_logprob=logprob)) == pytest.approx(expected)

    def test_missing_metadata_raises_and_safe_abstains(self):
        with pytest.raises(KeyError):
            logprob_score(_ctx())
        outcome = logprob_score.safe(_ctx())
        assert outcome.errored is True
        assert outcome.score == math.inf


class TestJudgeScore:
    def test_high_plausibility_gives_low_nonconformity(self):
        scorer = make_judge_score(lambda ctx: 0.9)
        assert scorer(_ctx()) == pytest.approx(0.1)

    def test_low_plausibility_gives_high_nonconformity(self):
        scorer = make_judge_score(lambda ctx: 0.1)
        assert scorer(_ctx()) == pytest.approx(0.9)

    def test_out_of_range_plausibility_raises_and_safe_abstains(self):
        scorer = make_judge_score(lambda ctx: 1.5)
        with pytest.raises(ValueError):
            scorer(_ctx())
        outcome = scorer.safe(_ctx())
        assert outcome.errored is True

    def test_underlying_judge_exception_forces_abstain(self):
        def _judge(ctx: ToolCallContext) -> float:
            raise ConnectionError("model API unreachable")

        scorer = make_judge_score(_judge)
        outcome = scorer.safe(_ctx())
        assert outcome.errored is True
        assert outcome.score == math.inf


class _ArgsSchema(BaseModel):
    x: int
    y: str = "default"


class TestSchemaValidityScore:
    def test_valid_args_score_zero(self):
        ctx = ToolCallContext(tool_name="t", args={"x": 1, "y": "ok"}, metadata={"schema": _ArgsSchema})
        assert schema_validity_score(ctx) == 0.0

    def test_invalid_args_score_one(self):
        ctx = ToolCallContext(tool_name="t", args={"x": "not an int"}, metadata={"schema": _ArgsSchema})
        assert schema_validity_score(ctx) == 1.0

    def test_missing_required_field_score_one(self):
        ctx = ToolCallContext(tool_name="t", args={}, metadata={"schema": _ArgsSchema})
        assert schema_validity_score(ctx) == 1.0

    def test_missing_schema_metadata_raises_and_safe_abstains(self):
        ctx = _ctx()
        with pytest.raises(KeyError):
            schema_validity_score(ctx)
        outcome = schema_validity_score.safe(ctx)
        assert outcome.errored is True
        assert outcome.score == math.inf


class TestUserSuppliedCallable:
    def test_arbitrary_user_scorer_is_the_general_case(self):
        def custom(ctx: ToolCallContext) -> float:
            return len(ctx.args)

        scorer = NonconformityScore(name="custom", fn=custom)
        ctx = ToolCallContext(tool_name="t", args={"a": 1, "b": 2})
        assert scorer(ctx) == 2
