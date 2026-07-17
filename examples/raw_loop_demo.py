"""Minimal end-to-end demo: calibrate a scorer, wrap a tool, run it through
the raw Anthropic tool-use adapter.

Runs with no API key and no network access -- the "model response" here is
a hand-written stand-in for what the Anthropic Messages API would return
when the model decides to call a tool, so the demo is self-contained and
reproducible. Swap `fake_model_response_with_tool_use()` for a real
`client.messages.create(...)` call to use this against a live agent loop.
"""

from __future__ import annotations

import random

from conformguard import (
    ToolCallContext,
    calibrate,
    wrap,
)
from conformguard.integrations.raw_tool_loop import ToolRegistry
from conformguard.storage.calibration_store import LabelingSource

# --- 1. The tool being protected. Unmodified, ordinary Python. -------------


def send_slack_message(channel: str, text: str) -> str:
    return f"posted to #{channel}: {text!r}"


# --- 2. A nonconformity score. Higher = more suspicious. -------------------
#
# This one is deliberately simple: it treats messages with an exclamation
# mark or the word "urgent" as riskier, since those are the messages most
# likely to be an over-eager or mistaken tool call in a real workflow.


def blast_radius_score(context: ToolCallContext) -> float:
    text = context.args.get("text", "")
    score = 0.05 * len(text.split())
    if "!" in text:
        score += 1.0
    if "urgent" in text.lower():
        score += 1.0
    return score


# --- 3. Calibration data: past (call, outcome) pairs. -----------------------
#
# In a real system these come from logged tool calls plus a post-hoc
# outcome label (§4.3): did the message get flagged/deleted/complained
# about (bad) or was it fine (good)? Here we synthesize a plausible-looking
# history so the demo is runnable standalone.


def _build_calibration_data(n: int = 200, seed: int = 0):
    rng = random.Random(seed)
    ordinary_texts = [
        "standup notes attached",
        "deploy finished successfully",
        "reminder: meeting at 3pm",
        "PR ready for review",
        "weekly report is up",
    ]
    data = []
    for _ in range(n):
        text = rng.choice(ordinary_texts)
        context = ToolCallContext(
            tool_name="send_slack_message",
            args={"channel": "general", "text": text},
        )
        data.append((context, True))  # all "good" outcomes: ordinary, uneventful messages
    return data


def main() -> None:
    calibration_data = _build_calibration_data(n=200)
    calibrator = calibrate(
        scorer=blast_radius_score,
        calibration_data=calibration_data,
        alpha=0.1,
        labeling_source=LabelingSource.DETERMINISTIC,
        hard_minimum_size=100,
    )
    print(f"Calibrated on {calibrator.n_calibration} examples, q_hat={calibrator.q_hat:.3f}")

    def context_builder(**kwargs) -> ToolCallContext:
        return ToolCallContext(tool_name="send_slack_message", args=kwargs)

    wrapped_send = wrap(send_slack_message, calibrator, context_builder=context_builder)
    registry = ToolRegistry({"send_slack_message": wrapped_send})

    # --- 4. Two "model responses," one ordinary and one that should abstain. --

    ordinary_tool_use = {
        "type": "tool_use",
        "id": "toolu_ordinary",
        "name": "send_slack_message",
        "input": {"channel": "general", "text": "deploy finished successfully"},
    }
    urgent_tool_use = {
        "type": "tool_use",
        "id": "toolu_urgent",
        "name": "send_slack_message",
        "input": {"channel": "general", "text": "URGENT!!! everyone drop what you're doing now!"},
    }

    for block in (ordinary_tool_use, urgent_tool_use):
        result = registry.handle_anthropic_tool_use(block)
        print()
        print(f"tool_use: {block['input']['text']!r}")
        print(f"  is_error={result['is_error']}")
        print(f"  content={result['content']}")


if __name__ == "__main__":
    main()
