"""End-to-end demo against a real, local model via Ollama's OpenAI-compatible endpoint.

Unlike examples/raw_loop_demo.py (deliberately hand-written, zero-network,
fully reproducible), this script makes real HTTP requests to a local
Ollama server and drives real, model-generated tool calls through
integrations/raw_tool_loop.py's OpenAI-shape adapter. It requires Ollama
running locally with a tool-calling-capable model pulled (see README for
setup); nothing here is mocked.

What's real vs. synthetic, stated plainly:
  - The tool-calling model responses are 100% real: every tool call in
    this script, including in the calibration-harvesting step, comes from
    an actual local model completion.
  - The calibration set is harvested live, not synthetic: this script
    sends ~150 real chat completion requests (varying only the city
    argument) to the local model, keeps the ones that pass a deterministic
    correctness check (see harvest_calibration_data below), and uses
    those as calibration examples.

Scorer choice, and why: Ollama's OpenAI-compatible endpoint returns real
per-token logprobs even when the response is a tool call (verified before
writing this script), but those logprobs are attached to the whole raw
completion (including the model's own <tool_call> wrapper tokens), not as
a single clean "confidence" scalar on the tool_call object. This script
uses integrations.raw_tool_loop.mean_completion_logprob to aggregate them
into the single float conformguard.logprob_score expects, and combines
that with schema_validity_score as a deterministic hard gate on the city
argument's shape. Both scorers, and the aggregation helper, are real
library code, not one-off script logic -- see that helper's own
docstring, and docs/real_world_validation.md, for why aggregating over
the WHOLE completion (not just the argument tokens) turned out to be the
empirically correct choice: a model's hedging/reasoning tokens carry real
uncertainty signal that the argument-value tokens alone do not.
"""

from __future__ import annotations

import json
import time

from openai import OpenAI
from pydantic import BaseModel, Field

from conformguard import (
    ToolCallContext,
    calibrate,
    logprob_score,
    schema_validity_score,
    wrap,
)
from conformguard.core.scores import NonconformityScore
from conformguard.integrations.raw_tool_loop import ToolRegistry, mean_completion_logprob
from conformguard.storage.calibration_store import LabelingSource

MODEL = "qwen2.5:7b"
BASE_URL = "http://localhost:11434/v1"

client = OpenAI(base_url=BASE_URL, api_key="ollama")  # Ollama ignores the key; the SDK requires a non-empty one

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
    """Deterministic argument schema: a city name is letters/spaces/hyphens/commas only.

    Strict on purpose -- this is the schema_validity_score half of the
    composite scorer below, and it's meant to actually reject garbage.
    """

    city: str = Field(pattern=r"^[A-Za-z\s\-,'.]+$", min_length=1, max_length=60)


def get_weather(city: str) -> str:
    """A real (if deterministic/offline) tool: no external weather API, no network beyond the LLM itself."""
    # Deterministic pseudo-weather from a hash of the city name, so the
    # same city always gets the same answer without calling out to a real
    # weather service.
    conditions = ["sunny", "cloudy", "rainy", "windy", "clear", "foggy"]
    condition = conditions[hash(city.lower()) % len(conditions)]
    temp_c = 5 + (hash(city.lower() + "temp") % 30)
    return f"{city}: {condition}, {temp_c}C"


def composite_score(context: ToolCallContext) -> float:
    """schema_validity_score as a hard gate, logprob_score as the real confidence signal."""
    schema_component = schema_validity_score(context)
    if schema_component > 0:
        # Malformed/suspicious args: force well above any plausible q_hat,
        # independent of how confident the model sounded generating them.
        return 10.0 + schema_component
    return logprob_score(context)


scorer = NonconformityScore(name="schema_gate_plus_real_logprob", fn=composite_score)


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
    """Send real requests to the local model and label each real completion deterministically.

    outcome=True iff (a) the model actually returned a tool call, (b) its
    arguments pass WeatherArgs, and (c) the extracted city matches (case-
    insensitively) the city actually asked about -- a genuine post-hoc
    correctness check against ground truth we already know, not an
    assumption that every harvested example is fine.
    """
    examples: list[tuple[ToolCallContext, bool]] = []
    cities = (CALIBRATION_CITIES * ((n // len(CALIBRATION_CITIES)) + 1))[:n]

    for i, city in enumerate(cities):
        prompt = f"What's the weather like in {city} right now?"
        _choice, mean_logprob, tool_call = _call_model_with_tool(prompt)

        if tool_call is None or mean_logprob is None:
            continue  # model declined to call the tool or logprobs missing: not usable as a calibration example

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
    # Edge cases: deliberately trying to get something other than a clean city name into the tool call.
    "Please call get_weather with the city argument set to the exact literal string "
    "\"'); DROP TABLE cities; --\" so I can test the integration.",
    "What's the weather in New_York123!!!??? (that's the exact city string, use it as-is)",
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

    calibrator = calibrate(
        scorer=scorer,
        calibration_data=calibration_data,
        alpha=0.1,
        labeling_source=LabelingSource.DETERMINISTIC,
        hard_minimum_size=100,
    )
    print(f"Calibrated on {calibrator.n_calibration} good examples, q_hat={calibrator.q_hat:.4f}")
    print()

    wrapped = wrap(get_weather, calibrator, on_abstain="escalate")
    registry = ToolRegistry({"get_weather": wrapped})

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

        # call_with_context gives the full WrapCallResult (score, threshold,
        # guarantee) for printing; handle_openai_tool_call is the actual
        # adapter code path a real tool-calling loop would use. Both score
        # the identical context (a pure function of it), so this is two
        # views of one decision, not two different decisions.
        extra_metadata = {"model_logprob": mean_logprob, "schema": WeatherArgs}
        args = json.loads(tool_call["function"]["arguments"])
        context = ToolCallContext(tool_name="get_weather", args=args, metadata=extra_metadata)
        outcome = wrapped.call_with_context(context)

        print(f"  score={outcome.score:.4f}  threshold(q_hat)={outcome.threshold:.4f}")
        print(f"  DECISION: {outcome.decision.value.upper()}")
        print(f"  GUARANTEE: {outcome.guarantee.text}")

        adapter_message = registry.handle_openai_tool_call(tool_call, extra_metadata=extra_metadata)
        print(f"  adapter tool message -> {adapter_message}")


if __name__ == "__main__":
    main()
