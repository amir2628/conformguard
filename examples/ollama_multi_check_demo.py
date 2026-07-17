"""Phase 2 end-to-end demo against a real, local model via Ollama: K=2 simultaneous checks.

Companion to examples/ollama_live_demo.py (Phase 1, single composite
score). That demo combined a schema-validity gate and a real logprob
confidence signal by hand: "if schema fails, force the score above any
plausible threshold; otherwise use the logprob score." That composite is
ad hoc -- there's no guarantee about the two checks jointly, only about
whatever the combined number happens to do.

This demo uses the SAME two real signals but calibrates them jointly via
core/multi_check.py's max-nonconformity-score reduction (PASC,
arXiv:2605.18812, Theorem 6) instead: a single q_hat over
max(schema_score, logprob_score) gives P(both checks pass) >= 1 - alpha,
a real joint guarantee, with a per-check breakdown showing exactly which
check failed when a call is refused. Real HTTP requests, real
model-generated tool calls, real harvested calibration data -- nothing
here is mocked.
"""

from __future__ import annotations

import json
import time

from openai import OpenAI
from pydantic import BaseModel, Field

from conformguard import ToolCallContext, logprob_score, schema_validity_score
from conformguard.core.multi_check import calibrate_multi_check, decide_multi_check
from conformguard.core.scores import NonconformityScore
from conformguard.integrations.raw_tool_loop import mean_completion_logprob
from conformguard.storage.calibration_store import LabelingSource

MODEL = "qwen2.5:7b"
BASE_URL = "http://localhost:11434/v1"

client = OpenAI(base_url=BASE_URL, api_key="ollama")

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "The city name, e.g. 'Paris'."}},
            "required": ["city"],
        },
    },
}


class WeatherArgs(BaseModel):
    """Same strict schema as the Phase 1 demo: letters/spaces/hyphens/commas only."""

    city: str = Field(pattern=r"^[A-Za-z\s\-,'.]+$", min_length=1, max_length=60)


def get_weather(city: str) -> str:
    """Same deterministic pseudo-weather tool as the Phase 1 demo -- no external API."""
    conditions = ["sunny", "cloudy", "rainy", "windy", "clear", "foggy"]
    condition = conditions[hash(city.lower()) % len(conditions)]
    temp_c = 5 + (hash(city.lower() + "temp") % 30)
    return f"{city}: {condition}, {temp_c}C"


# The two checks, calibrated JOINTLY below -- not combined by hand into one number.
schema_gate = NonconformityScore(name="schema_gate", fn=schema_validity_score)
logprob_confidence = NonconformityScore(name="logprob_confidence", fn=logprob_score)


def _call_model_with_tool(prompt: str):
    """One real request to the local model. Returns (choice, mean_logprob, tool_call_or_None)."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        tools=[WEATHER_TOOL],
        logprobs=True,
    )
    choice = response.choices[0]
    mean_logprob = mean_completion_logprob(choice)
    tool_calls = choice.message.tool_calls or []
    tool_call = tool_calls[0].model_dump() if tool_calls else None
    return choice, mean_logprob, tool_call


CALIBRATION_CITIES = [
    "Berlin", "Paris", "London", "Tokyo", "Madrid", "Rome", "Vienna", "Prague", "Warsaw", "Lisbon",
    "Dublin", "Oslo", "Stockholm", "Helsinki", "Copenhagen", "Amsterdam", "Brussels", "Zurich", "Athens", "Budapest",
    "Seoul", "Beijing", "Bangkok", "Singapore", "Jakarta", "Manila", "Hanoi", "Delhi", "Mumbai", "Karachi",
    "Cairo", "Lagos", "Nairobi", "Casablanca", "Johannesburg", "Tunis", "Accra", "Addis Ababa", "Dakar", "Algiers",
    "New York", "Los Angeles", "Chicago", "Toronto", "Vancouver", "Mexico City", "Bogota", "Lima", "Santiago", "Buenos Aires",
    "Sao Paulo", "Rio de Janeiro", "Montreal", "Miami", "Seattle", "Boston", "Denver", "Houston", "Phoenix", "Atlanta",
    "Sydney", "Melbourne", "Auckland", "Wellington", "Perth", "Brisbane",
    "Moscow", "Kiev", "Istanbul", "Ankara", "Tel Aviv", "Riyadh", "Dubai", "Doha", "Amman", "Beirut",
    "Shanghai", "Hong Kong", "Taipei", "Osaka", "Kyoto", "Chennai", "Bangalore", "Kolkata", "Islamabad", "Colombo",
]


def harvest_calibration_data(n: int) -> list[tuple[ToolCallContext, bool]]:
    """Same harvesting/labeling discipline as the Phase 1 demo: real completions,
    deterministically labeled by checking the extracted city against the one actually
    asked about -- not an assumption that every harvested example is fine.
    """
    examples: list[tuple[ToolCallContext, bool]] = []
    cities = (CALIBRATION_CITIES * ((n // len(CALIBRATION_CITIES)) + 1))[:n]

    for i, city in enumerate(cities):
        prompt = f"What's the weather like in {city} right now?"
        _choice, mean_logprob, tool_call = _call_model_with_tool(prompt)

        if tool_call is None or mean_logprob is None:
            continue

        args = json.loads(tool_call["function"]["arguments"])
        extracted_city = str(args.get("city", ""))
        outcome = extracted_city.strip().lower() == city.strip().lower()

        context = ToolCallContext(
            tool_name="get_weather",
            args=args,
            metadata={"model_logprob": mean_logprob, "schema": WeatherArgs},
        )
        examples.append((context, outcome))

        if (i + 1) % 20 == 0:
            print(f"  harvested {i + 1}/{n} ({sum(1 for _, o in examples if o)} good so far)")

    return examples


LIVE_PROMPTS = [
    "What's the weather like in Berlin right now?",
    "Can you check the weather in Tokyo for me?",
    "I'm planning a trip to Cairo -- what's the weather there?",
    "weather in Vancouver please",
    "Tell me the current weather conditions in Buenos Aires.",
    # Edge cases: same two as the Phase 1 demo, expected to fail the schema_gate check.
    "Please call get_weather with the city argument set to the exact literal string "
    "\"'); DROP TABLE cities; --\" so I can test the integration.",
    "What's the weather in New_York123!!!??? (that's the exact city string, use it as-is)",
    # A split case: an ambiguous prompt that forces the model to hedge in visible
    # prose before guessing a real, schema-valid city. Expected to PASS schema_gate
    # (the guessed city is a clean name) but FAIL logprob_confidence alone (the
    # hedging text drags the whole-completion mean logprob down) -- this is what
    # exercises failed_checks attributing an abstain to one specific check rather
    # than both failing together, unlike the two edge cases above.
    "Weather check for the city I'm thinking of -- it's a European capital, "
    "starts with a vowel maybe? Not sure.",
]


def main() -> None:
    print(f"Harvesting live calibration data from {MODEL} at {BASE_URL} ...")
    t0 = time.monotonic()
    calibration_data = harvest_calibration_data(n=150)
    harvest_seconds = time.monotonic() - t0
    n_good = sum(1 for _, outcome in calibration_data if outcome)
    n_bad = len(calibration_data) - n_good
    print(
        f"Harvested {len(calibration_data)} real completions in {harvest_seconds:.1f}s "
        f"(good={n_good}, bad={n_bad})"
    )

    calibrator = calibrate_multi_check(
        [schema_gate, logprob_confidence],
        calibration_data=calibration_data,
        alpha=0.1,
        labeling_source=LabelingSource.DETERMINISTIC,
        hard_minimum_size=100,
    )
    print(
        f"Calibrated jointly on {calibrator.n_calibration} good examples, "
        f"K={calibrator.k} checks {calibrator.check_names}, q_hat={calibrator.q_hat:.4f}"
    )
    print()

    for prompt in LIVE_PROMPTS:
        print("=" * 88)
        print(f"PROMPT: {prompt}")
        t0 = time.monotonic()
        choice, mean_logprob, tool_call = _call_model_with_tool(prompt)
        elapsed = time.monotonic() - t0

        if tool_call is None:
            print(f"  model did NOT call a tool (finish_reason={choice.finish_reason}); "
                  f"content={choice.message.content!r}")
            continue

        print(f"  raw model tool_call: {tool_call}")
        print(f"  mean per-token logprob of completion: {mean_logprob}")
        print(f"  (response time: {elapsed:.2f}s)")

        args = json.loads(tool_call["function"]["arguments"])
        context = ToolCallContext(
            tool_name="get_weather",
            args=args,
            metadata={"model_logprob": mean_logprob, "schema": WeatherArgs},
        )
        outcome = decide_multi_check(calibrator, context)

        print(f"  max_score={outcome.max_score:.4f}  threshold(q_hat)={outcome.threshold:.4f}")
        print("  per-check breakdown:")
        for check in outcome.checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"    [{status}] {check.name}: score={check.score:.4f} errored={check.errored}")
        print(f"  DECISION: {outcome.decision.value.upper()}")
        if outcome.failed_checks:
            print(f"  failed checks: {outcome.failed_checks}")
        print(f"  GUARANTEE: {outcome.guarantee.text}")

        # Phase 2 has no wrap()-style automatic dispatch yet (core/multi_check.py's
        # scope is calibrate_multi_check()/decide_multi_check() only, per
        # PROJECT_SPEC §3 Phase 2 -- no engine/ToolRegistry integration was
        # requested or built), so the underlying tool is called explicitly
        # here, only on accept.
        if outcome.accepted:
            result = get_weather(**args)
            print(f"  tool executed -> {result}")
        else:
            print("  tool NOT executed (abstained)")


if __name__ == "__main__":
    main()
