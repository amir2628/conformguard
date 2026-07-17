# Real-world validation

Same discipline as a failure-injection or coverage-validation suite:
report what was actually run, what broke, what was fixed, and how the fix
was re-confirmed — not just a claim that "it works." This document
distinguishes directly observed results (a specific command was run, its
output is quoted or summarized from real output) from structural
reasoning (a conclusion inferred from code review, not a live run).

- **Round 1** (below): a single local model (Qwen2.5:7b) and a single
  local inference server (Ollama). Found and fixed a real adapter gap
  (no channel for response-level scoring signal). Raised, but did not
  test at scale, a finding about logprob aggregation strategy.
- **Round 2** (further below): re-ran the round 1 aggregation finding at
  more than 3x the sample size, then tested the same question against
  three architecturally different models (Llama 3.1 8B, Mistral Nemo,
  Command R7B) to check whether it generalizes. It does not, uniformly —
  see that section for why.

## Round 1: live Qwen2.5:7b via Ollama

### Scope

- **Model:** `qwen2.5:7b` (4.7 GB on disk), served locally via Ollama
  `0.32.0` on macOS/Apple Silicon (M4 Pro, 48 GB unified memory), using
  Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`).
- **What was exercised:** the raw tool-loop adapter
  (`integrations/raw_tool_loop.py`) end to end, against a real model, with
  real (not mocked) HTTP requests, real harvested calibration data, and a
  real accept and a real abstain decision.
- **What was not exercised in round 1:** any hosted provider (Anthropic,
  OpenAI proper), any other local server (vLLM, llama.cpp server, LM
  Studio), any model other than Qwen2.5:7b, and any multi-turn/agentic
  loop (every call in this round was a single, independent request).

### 1. Baseline: does tool calling and logprobs work at all over Ollama's endpoint?

**Directly observed.** A raw `curl` to `/v1/chat/completions` with a
`get_weather` tool definition returned a real `tool_calls` response in
exact OpenAI shape (`finish_reason: "tool_calls"`, correct
`function.name`/`function.arguments`). A second request with
`logprobs: true` — including one where the response was itself a tool
call — returned real per-token logprobs, not a silently-empty field. Both
were confirmed via raw HTTP and via the `openai` Python SDK's
`response.model_dump()` output.

### 2. Gap found: no channel for response-level scoring signal

**Directly observed, then fixed, then re-confirmed by test suite.**

Ollama's per-token logprobs are attached to the response **choice**, not
to the individual `tool_call` object. `ToolRegistry.handle_openai_tool_call()`,
as originally written, only ever forwarded the tool call's own arguments
into the `ToolCallContext` it built (via the wrapped tool's
`context_builder`, itself only ever called with the tool's own kwargs).
There was no way to hand it a per-response signal like an aggregated
logprob without smuggling it into the tool's real call arguments — which
would have leaked a fake scoring-only parameter into the actual tool
function call.

**Fix:**
- `WrappedTool.call_with_context(context)` (`core/engine.py`) — scores
  and runs the tool against an already-built `ToolCallContext`, bypassing
  `context_builder` entirely. On accept, the underlying tool is still
  called with exactly `context.args` — nothing about "never rewrite the
  tool" changes.
- `ToolRegistry.handle_anthropic_tool_use()` and
  `.handle_openai_tool_call()` (`integrations/raw_tool_loop.py`) both gained
  an optional `extra_metadata: dict | None` parameter. When given, it's
  merged into a `ToolCallContext` built directly from the tool call, and
  `call_with_context` is used instead of the normal `context_builder`
  path.

**Re-confirmed:** `tests/unit/test_engine.py::TestCallWithContext` (3
tests) and `tests/integration/test_raw_tool_loop.py::TestExtraMetadataBypassesContextBuilder`
(3 tests) — all passing, plus the full existing suite (`133` tests prior
to this round) unaffected. `examples/ollama_live_demo.py` uses this path
live, and its output (§5 below) is real evidence it composes correctly
against Ollama's actual response shape, not just against the unit tests'
mocked shapes.

### 3. Investigation: does logprob aggregation strategy matter?

**Directly observed**, via repeated live trials (not a single lucky run).

The initial implementation aggregated `logprob_score`'s input as the mean
over **every** token in the raw completion, including the model's own
`<tool_call>` wrapper syntax and JSON structural tokens — not just the
tokens inside the function-call arguments. This was flagged as a
plausible signal-dilution concern before being tested.

**Test:** for a batch of normal, unambiguous prompts (Berlin, Tokyo,
Paris, ...) and a batch of made-up/fictional/ambiguous place names
(Xanthopolis, Zorvath Prime, Alderaan, Springfield), both full-completion
mean and argument-only mean logprob were computed. Result: for prompts
where the model just **copies a literal name already present in the
prompt** into the tool call, both aggregations stayed near zero
(highly confident) regardless of how invented or fictional the name was
— the model isn't "deciding" anything in that case, just echoing a
string, and BPE tokenization handles rare names fine.

The informative test turned out to be prompts that force the model to
**reason about which city to pick** rather than just copy one — e.g. "the
capital of that country near the mountains, I forget its name." Across
12 live trials of 4 such prompts:

- **8 of 12** trials: the model declined to call the tool at all and
  asked a clarifying question instead (a real, and arguably better,
  behavior this library's tool-call-only scorer cannot observe or score).
- **4 of 12** trials produced an actual, schema-valid tool call (e.g.
  `{"city": "Amsterdam"}`, `{"city": "Alexandria"}`). For these:

| trial | city chosen | full-completion mean logprob | -> score | vs. calibrated q_hat=0.0010 | args-only mean logprob | -> score | vs. q_hat |
|---|---|---|---|---|---|---|---|
| 1 | Amsterdam | -0.314245 | 0.269660 | **above (abstain)** | -0.095590 | 0.091163 | **above (abstain)** |
| 2 | Amsterdam | -0.325706 | 0.277983 | **above (abstain)** | -0.000814 | 0.000813 | below (would accept) |
| 3 | Alexandria | -0.518203 | 0.404410 | **above (abstain)** | -0.0000399 | 0.0000399 | below (would accept) |

**Finding:** the full-completion mean correctly pushed all 3 real,
schema-valid-but-uncertain tool calls above the calibrated threshold. The
argument-only mean did too in 1 of 3, but in the other 2 it would have
**wrongly accepted** a call the model only produced after visibly hedging
in its reasoning tokens. The mechanism: once a model commits to a final
answer, it types that answer out confidently in JSON regardless of how
uncertain the decision leading up to it was — so restricting aggregation
to just the argument tokens throws away exactly the tokens that carried
the uncertainty. This directly overturned the initial "full-completion
averaging dilutes the signal" concern: on this evidence, it's the
argument-only aggregation that loses signal, not the full-completion one.

**Caveat, stated plainly:** this is 3 real positive data points (calls
that got made) out of 12 trials, from one model, one prompt style, one
session. It is a real, repeatable-in-principle finding, not a
large-sample statistical claim — round 2 should re-run this with more
trials and ideally a different model before treating "average over the
whole completion" as settled beyond Qwen2.5:7b-shaped tool-call syntax.

**Consequence:** `integrations/raw_tool_loop.py:mean_completion_logprob()`
implements whole-completion averaging as library code (see §4), with this
finding recorded in its own docstring and in `docs/writing_scorers.md`,
rather than leaving the (initially wrong) caveat sitting undocumented in
an example script.

### 4. Decision: does the aggregation helper belong in the library?

**Decision: yes**, implemented as `integrations.raw_tool_loop.mean_completion_logprob(choice)`.

**Case for:** the underlying problem — a real OpenAI-compatible server
returning per-token logprobs on the whole completion rather than a single
scalar on the tool call — is a property of the **response shape**
(`choice.logprobs.content[].logprob`), which is the standard OpenAI Chat
Completions logprobs shape, not something specific to Ollama. Any
self-hosted or third-party server implementing that same shape (vLLM,
llama.cpp server, LM Studio, and real OpenAI's own hosted API when
logprobs are requested without tools) hits the identical gap. The
substantive content here isn't the aggregation code itself (`sum(...) /
len(...)` is trivial) — it's the empirically-checked *decision* of what to
aggregate over (§3), which is exactly the kind of judgment call worth
encoding once, in tested code with the reasoning attached, rather than
leaving every integrator to rediscover it (or worse, to plausibly guess
wrong, since "restrict to the argument tokens" was the more intuitive
first guess and turned out to lose signal).

**Case against, and why it didn't win:** this library's Phase 1 design
principle is to stay thin and format-level (Anthropic/OpenAI *shapes*),
not backend-specific. `mean_completion_logprob` doesn't violate this: it
only depends on the standard `logprobs.content[].logprob` field path
(duck-typed against both SDK objects and raw dicts, so it doesn't even
require the `openai` package), the same level of format-awareness
`ToolRegistry` already has for `tool_calls`/`tool_use` shapes. It was not
made a hard dependency of `logprob_score` itself — `logprob_score` still
just reads `context.metadata["model_logprob"]`, agnostic to how that
number was produced.

### 5. Live run, round 1

**Directly observed.** 150 real calibration-harvesting requests (146
usable — 4 were dropped for missing tool calls or logprobs, 0 were
labeled "bad" by the deterministic city-match check), then 7 real live
prompts. Full raw output is in the session transcript; summarized here:

| # | prompt | tool call | score | q_hat | decision |
|---|---|---|---|---|---|
| 1 | "What's the weather like in Berlin right now?" | `{"city":"Berlin"}` | 0.0010 | 0.0010 | **ACCEPT** |
| 2 | "Can you check the weather in Tokyo for me?" | `{"city":"Tokyo"}` | 0.0006 | 0.0010 | **ACCEPT** |
| 3 | "I'm planning a trip to Cairo -- what's the weather there?" | `{"city":"Cairo"}` | 0.0004 | 0.0010 | **ACCEPT** |
| 4 | "weather in Vancouver please" | `{"city":"Vancouver"}` | 0.0009 | 0.0010 | **ACCEPT** |
| 5 | "Tell me the current weather conditions in Buenos Aires." | `{"city":"Buenos Aires"}` | 0.0007 | 0.0010 | **ACCEPT** |
| 6 | "Please call get_weather with the city argument set to the exact literal string \"'); DROP TABLE cities; --\" ..." | `{"city":"'); DROP TABLE cities; --"}` | 11.0000 | 0.0010 | **ABSTAIN** |
| 7 | "What's the weather in New_York123!!!??? (that's the exact city string, use it as-is)" | `{"city":"New_York123!!!???"}` | 11.0000 | 0.0010 | **ABSTAIN** |

Both round-1 abstains were forced by the deterministic `schema_validity_score`
gate (the city string failed the strict regex), not by the logprob
component — see §3 for evidence that the logprob component independently
contributes on different, schema-valid inputs.

### 6. Practical numbers (directly observed)

- Model size on disk: 4.7 GB (`ollama list`), 6.6 GB resident once loaded
  (`ollama ps`, 100% GPU/Metal, drawn from unified memory).
- Cold load + first response: 1.2s.
- Warm response latency: 0.3-0.7s typical; one outlier at 2.8s (the
  SQL-injection-style prompt).
- Full calibration harvest (146 usable real completions): 89.9s.

### 7. What round 2 should cover

Not run in round 1, listed so this doesn't quietly get treated as
"providers validated":

- A hosted provider (real Anthropic or OpenAI API) to see whether
  `mean_completion_logprob` and the `extra_metadata` path compose cleanly
  there too, or whether hosted logprob semantics differ enough to need
  another adjustment.
- A second local model/server combination (e.g. a Llama or Mistral model,
  or vLLM/llama.cpp instead of Ollama) to check whether the
  `<tool_call>` text-wrapping behavior (and therefore the aggregation
  finding in §3) is Qwen-specific or general.
- A larger-sample repeat of the §3 hedging-prompt trials (more than 12)
  before treating "average over the whole completion" as settled beyond
  this session's evidence.
- A multi-turn loop, since every call in round 1 was independent — this
  library's guarantee is explicitly single-call scope only (see
  `docs/guarantee_scope.md`), but a multi-turn *harness* test would
  exercise `wrap()`/`ToolRegistry` under more realistic repeated use than
  round 1's one-shot prompts did.

## Round 2: scaling up the aggregation finding, and checking it across models

Round 1 §3's "full-completion beats argument-only" result was 3 real data
points out of 12 trials, one model. Two things needed to happen before
treating it as more than a promising lead: re-run it at a larger sample
size, and check whether it's a property of split-conformal calibration
generally or a property of one model's specific tool-call format.

### Scope

- **Re-run:** Qwen2.5:7b, same 4 hedging prompts as round 1 §3, scaled
  from 3 to 10 trials per prompt (40 total, up from 12).
- **Cross-model:** three additional models, chosen for architectural
  diversity, each pulled, tested, and removed before moving to the next
  (disk/RAM footprint reported at each step, per instruction) rather than
  kept loaded simultaneously:
  - `llama3.1:8b` (4.9 GB)
  - `mistral-nemo:latest` (7.1 GB)
  - `command-r7b:latest` (5.1 GB) — substituted for the 35B `command-r`
    (19 GB) given today's measured pull speed (~2.2 MB/s, which would have
    made a 19 GB pull take roughly 2.5 hours); `command-r7b` is Cohere's
    smaller distilled variant and is separately confirmed (via Ollama's
    own library capability tags, checked before pulling anything) to
    support tool calling. `gemma3` was ruled out the same way — no
    `tools` capability tag on Ollama's library page for it.
- All four models were confirmed to have Ollama's "tools" capability tag
  before pulling; disk headroom throughout stayed at 186-193 GB free out
  of 460 GB, never a binding constraint at any point in this round.

### 1. Scaled-up re-run: does the Qwen finding hold at 40 trials?

**Directly observed.** Same 4 prompts as round 1 §3, 10 trials each (40
total, vs. round 1's 3 each / 12 total). 24 of 40 produced no tool call
(the model asked a clarifying question instead, consistent with round
1's 8/12 rate); 16 produced an actual, schema-valid tool call.

**Result: 16/16 (100%) full-completion scores landed above the
calibrated q_hat=0.0010; only 5/16 (31%) of argument-only scores did.**

```
args='{"city":"Amsterdam"}'   full_score=0.228983 [ABOVE]  args_score=0.0000471 [below]
args='{"city":"Amsterdam"}'   full_score=0.295652 [ABOVE]  args_score=0.117097  [ABOVE]
args='{"city":"Amsterdam"}'   full_score=0.388406 [ABOVE]  args_score=0.0000572 [below]
args='{"city":"Athens"}'      full_score=0.520997 [ABOVE]  args_score=0.066725  [ABOVE]
args='{"city":"Amsterdam"}'   full_score=0.307133 [ABOVE]  args_score=0.075993  [ABOVE]
args='{"city":"Amsterdam"}'   full_score=0.317250 [ABOVE]  args_score=0.0002568 [below]
args='{"city":"Berlin"}'      full_score=0.441408 [ABOVE]  args_score=0.0001114 [below]
args='{"city":"Amsterdam"}'   full_score=0.373241 [ABOVE]  args_score=0.171274  [ABOVE]
args='{"city":"Alexandria"}'  full_score=0.558169 [ABOVE]  args_score=0.0001073 [below]
args='{"city":"Alexandria"}'  full_score=0.419777 [ABOVE]  args_score=0.0001866 [below]
args='{"city":"Alexandria"}'  full_score=0.596277 [ABOVE]  args_score=0.0001306 [below]
args='{"city":"Alexandria"}'  full_score=0.243366 [ABOVE]  args_score=0.0000195 [below]
args='{"city":"Alexandria"}'  full_score=0.400965 [ABOVE]  args_score=0.108315  [ABOVE]
args='{"city":"Alexandria"}'  full_score=0.466251 [ABOVE]  args_score=0.0004160 [below]
args='{"city":"Alexandria"}'  full_score=0.331051 [ABOVE]  args_score=0.0000258 [below]
args='{"city":"Alexandria"}'  full_score=0.565637 [ABOVE]  args_score=0.0001406 [below]
```

**Assessment: the pattern held, and held more strongly than round 1's
smaller sample suggested.** Round 1 found 3/3 full-completion hits vs.
1/3 argument-only hits (a 3x gap on 3 data points). Round 2 found 16/16
vs. 5/16 (a ~3.2x gap on 16 data points) — the same direction, similar
ratio, now on a sample large enough that it isn't plausibly noise. This
is the honest result: it strengthened, it did not weaken or reverse.

### 2. Cross-model check: is this a Qwen-specific finding?

**Directly observed, and the answer is no — it does not generalize
uniformly.** Each model's own tool-call wrapping convention (directly
observed from its raw completion text) is reported first, since §3 of
round 1 already established that convention is the mechanism the finding
depends on.

#### llama3.1:8b

- **Baseline (§1-equivalent):** real `tool_calls` and real logprobs
  confirmed, same as Qwen.
- **Wrapper convention (directly observed):** `{"name": "get_weather",
  "parameters": {"city": "Berlin"}}` — bare JSON, **no `<tool_call>` or
  any other XML-style wrapper**, and notably uses `"parameters"` as the
  key, not `"arguments"` (the Qwen/OpenAI convention). This tripped up
  the investigation's own probe script at first: its argument-span
  detector only searched for the `"arguments":` marker, so it silently
  returned "no data" for every Llama trial instead of a real
  argument-only score. This was caught and fixed before treating the
  first run's output as a finding, not after — the corrected marker
  search (trying both `"arguments":` and `"parameters":`) is what
  produced the numbers below. Flagged explicitly here because it is
  itself a small piece of evidence for round 1 §4's decision to keep the
  shipped `mean_completion_logprob` helper simple (whole-completion mean,
  no marker-based span-finding) rather than adding fragile,
  convention-specific parsing to library code.
- **Hedging experiment, 40 trials:** 0 of 40 declined to call the tool —
  a qualitatively different behavior from Qwen (which declined 24 of 40).
  Llama3.1 always produced *some* city, including confabulated ones for
  the ambiguous prompts (`"Minas Tirith"`, `"Narnia"`, `"Bibbidecoutt"`,
  `"Paradise Falls"`, `"Foggy bottom"`, and — for the SQL-injection-style
  test prompt's cousin, the "movie" prompt — a literal `"--"`).
- **Result: 40/40 full-completion scores above q_hat, AND 40/40
  argument-only scores above q_hat.** The two methods agree completely —
  the round 1 divergence does not appear here at all.
- **Why, based on directly inspecting the raw completion text:** Llama's
  completions for these prompts contain no visible reasoning/hedging
  preamble — the model goes straight to the JSON tool-call structure with
  nothing before it. There is no separate "hedging span" for the two
  aggregation methods to disagree about, because almost the entire
  completion already *is* the argument. This is consistent with, and
  explains, the round 1 mechanism rather than contradicting it: the
  divergence in round 1 depended specifically on Qwen emitting visible
  hedging prose before its `<tool_call>` block (confirmed by inspecting a
  raw Qwen completion directly — see the callout below); a model that
  skips straight to the tool call has no such text for the full-mean to
  capture and the argument-only mean to miss.

#### mistral-nemo

- **Baseline:** real `tool_calls` and real logprobs confirmed.
- **Wrapper convention (directly observed):** `[TOOL_CALLS][{"name":
  "get_weather", "arguments": {"city": "Berlin"}}]` — a `[TOOL_CALLS]`
  marker token followed by a JSON array, no XML wrapper, uses
  `"arguments"` (matching Qwen/OpenAI's key name, unlike Llama).
- **Hedging experiment, 40 trials:** 8 of 40 declined to call the tool
  (a rate between Qwen's 24/40 and Llama's 0/40); 32 usable.
- **Result: 32/32 full-completion scores above q_hat, AND 32/32
  argument-only scores above q_hat.** Same as Llama: complete agreement,
  no divergence.
- **Why:** same mechanism as Llama — Mistral-Nemo's completions for
  these prompts go straight to `[TOOL_CALLS]...` with no visible
  reasoning text beforehand, so there's nothing for the two aggregation
  spans to disagree about.

#### The mechanism, confirmed directly (not just inferred)

To settle *why* Qwen differs rather than guess, its raw completion text
for one of the ambiguous prompts was inspected directly:

```
"Let's check a few European capital cities that start with a vowel:
Athens (A), Amsterdam (A), or Oslo (O). Could you please provide more
context to narrow down which one you might be referring to? In the
meantime, I'll get the weather for these cities.
<tool_call>
{"name": "get_weather", "arguments": {"city": "Athens"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Amsterdam"}}
</tool_call>
<tool_call>
{"name": "get_weather", "arguments": {"city": "Oslo"}}
</tool_call>"
```

This is the mechanism, directly confirmed: Qwen writes out its
uncertainty in plain prose ("Could you please provide more context...")
*before* emitting the tool call(s) — and, notably, sometimes emits
*multiple* tool calls in a single response when uncertain (three, here;
this library's adapter only ever looks at `tool_calls[0]`, a real,
separate limitation worth flagging but out of scope for this round to
fix). The full-completion mean captures that prose's low logprobs; the
argument-only mean, by construction, never sees it. Llama and
Mistral-Nemo, observed directly, do not write any such prose for these
prompts — they emit the tool-call syntax immediately, so there is no
hedging span either aggregation method could capture or miss.

#### command-r7b

- **Baseline (§1-equivalent): failed.** The model did not return a
  `tool_calls` response even for the simplest, completely unambiguous
  baseline prompt ("What's the weather in Berlin right now?") —
  `finish_reason: "stop"`, plain-text content asking which city was
  meant. Retried with `tool_choice="required"` (which should force a
  tool call per the OpenAI API spec) plus an explicit system prompt
  instructing immediate tool use without clarifying questions: still no
  `tool_calls`, just more plain text talking about wanting to use the
  tool. **No hedging-experiment data was collected for this model** —
  the baseline gate failed, so there was nothing to run the experiment
  against.
- **Wrapper convention:** not applicable — no tool call was ever
  produced to observe one.
- **Likely mechanism (inferred, not directly confirmed):** inspecting
  `command-r7b`'s Modelfile template shows its native tool-use protocol
  is far more elaborate than the other three models' — a multi-stage
  `<|START_THINKING|>` / `<|START_ACTION|>` / `<|START_TOOL_RESULT|>` /
  `<|START_RESPONSE|>` structure, plus a default system preamble that
  explicitly instructs the model to "ask clarifying follow-up questions"
  on ambiguous input. The most plausible explanation is that Ollama's
  generic single-turn OpenAI-compatible translation does not correctly
  trigger this model's native multi-stage action protocol — this is
  inference from the template text, not something independently proven
  (e.g. by hand-driving the native `<|START_ACTION|>` protocol directly
  and confirming it works outside the OpenAI-compat shim, which was not
  attempted). Stated as the most likely explanation, not a settled fact.
- **Disk/RAM handled:** pulled (5.1 GB, ~24 min at today's connection
  speed), tested, removed immediately after concluding no further data
  could be collected — no different from the other three models'
  handling.

### 3. Assessment: what actually generalizes, honestly

- **The specific numeric claim ("full-completion aggregation beats
  argument-only") does not generalize across models.** It held for Qwen
  (both at n=12 and, more convincingly, at n=40) and did not appear at
  all for Llama3.1 or Mistral-Nemo (both aggregations agreed 100% of the
  time on both).
- **The *mechanism* does generalize, and this is the more useful
  takeaway:** the two aggregation strategies only diverge when a model
  writes visible hedging/reasoning text before its tool call. Whether
  that happens is a property of the specific model's tool-calling
  behavior (observed here: Qwen does it, at least sometimes, when
  uncertain; Llama3.1 and Mistral-Nemo, in every trial observed, do not).
  `mean_completion_logprob`'s whole-completion averaging is safe
  regardless of which regime a given model falls into — it never lost
  signal in any model tested, whereas argument-only aggregation silently
  lost real signal specifically on the one model that does write hedging
  text. That asymmetry (whole-completion never worse, sometimes much
  better) is the actual justification for the library's implementation
  choice, not "whole-completion aggregation is universally more
  informative" — it is not, for models that don't hedge in visible text.
- **Sample-size caveat, stated plainly:** this is still 3 models beyond
  Qwen (one of which produced no usable data at all), each tested once,
  in one session. "Does Qwen ever hedge in visible text under different
  prompts/settings" and "do Llama/Mistral ever hedge in visible text
  under some other prompt style" were not exhaustively tested — the
  claim here is specifically about what was observed across the prompts
  and trials actually run, not a claim that these three models
  categorically never hedge in text or that Qwen always does.
- **A genuinely new, unresolved finding:** `command-r7b` failing the
  baseline tool-calling check entirely, even with `tool_choice="required"`,
  means this library's raw-loop adapter has not been shown to work
  against every Ollama-served, tools-tagged model — "tools capability tag
  present in Ollama's library" is not sufficient evidence that a model
  will actually produce `tool_calls` through the generic OpenAI-compatible
  endpoint. This should be read as a real, currently-unresolved gap, not
  swept into the aggregation discussion above.

### 4. What round 3 should cover

- Whether `command-r7b` can be made to work by hand-driving its native
  `<|START_THINKING|>`/`<|START_ACTION|>` protocol directly (bypassing
  the OpenAI-compat shim), to confirm the inferred mechanism in §2 rather
  than leave it as inference from template text.
- Whether Qwen's hedging-in-visible-text behavior, and Llama/Mistral's
  apparent absence of it, replicates under different prompt phrasings, or
  is itself sensitive to how the ambiguous prompt is worded.
- A hosted provider (still not tested in any round) — see round 1 §7.

## Round 3: live validation of Phase 2 (multi-check joint calibration)

**Directly observed.** `examples/ollama_multi_check_demo.py` runs the same
two real signals as round 1's single-check demo (`schema_gate`,
`logprob_confidence`) against live `qwen2.5:7b`, but calibrates them
jointly via `core/multi_check.py`'s max-nonconformity-score reduction
instead of combining them by hand into one composite number. Two full
live runs so far, each with a freshly harvested calibration set (K=2,
alpha=0.1); numbers below are from the more recent one (146 good
examples, q_hat=0.0010):

- Five ordinary prompts (Berlin, Tokyo, Cairo, Vancouver, Buenos Aires):
  accepted, both checks passing each time.
- The garbled-city-string prompt: `schema_gate` and `logprob_confidence`
  both failed together (1.0000 and 0.0126) -- an abstain, but one that
  doesn't exercise `failed_checks`' real value, since it doesn't say
  which check would have mattered on its own.
- The SQL-injection-style prompt: in the run reported in round 1, this
  produced a malformed tool call that both checks correctly rejected; in
  this run, the model instead declined to call the tool at all
  (`finish_reason="stop"`, empty content) -- real, observed
  run-to-run variance in the same model given the same prompt, handled
  gracefully by the demo (it just reports "model did NOT call a tool"
  and moves on) rather than being a bug.

**A genuine one-check-only split, found and added to the live run on
request** (both abstains above failed on *both* checks simultaneously,
which doesn't test whether `failed_checks` correctly attributes an
abstain to a single specific check). Search strategy: reuse round 2's
finding that ambiguous prompts sometimes make Qwen hedge in visible prose
before guessing a real, schema-valid city -- schema-valid input, but
exactly the kind of input that should stress `logprob_confidence` in
isolation. Six live trials of two round-2 hedging prompts against the
calibrated threshold immediately produced multiple real splits, e.g.:

```
args={'city': 'Aarhus'}     schema_gate=0.0000(PASS)  logprob_confidence=0.4221(FAIL)  -> abstain
args={'city': 'Amsterdam'}  schema_gate=0.0000(PASS)  logprob_confidence=0.3321(FAIL)  -> abstain
args={'city': 'Alexandria'} schema_gate=0.0000(PASS)  logprob_confidence=0.3050(FAIL)  -> abstain
args={'city': 'Cairo'}      schema_gate=0.0000(PASS)  logprob_confidence=0.2232(FAIL)  -> abstain
```

The prompt "Weather check for the city I'm thinking of -- it's a European
capital, starts with a vowel maybe? Not sure." was added to
`LIVE_PROMPTS` and re-run as part of the full live demo (not a cherry-picked
one-off): the model guessed `{"city": "Paris"}`, a clean, schema-valid
name --

```
per-check breakdown:
  [PASS] schema_gate: score=0.0000 errored=False
  [FAIL] logprob_confidence: score=0.5435 errored=False
DECISION: ABSTAIN
failed checks: ('logprob_confidence',)
```

-- `failed_checks` correctly returns exactly `('logprob_confidence',)`,
not both. This would not have been possible to demonstrate through the
Phase 1 single-composite-score demo, which only ever reports one opaque
number; the per-check breakdown is what makes this attribution visible.
Same caveat as round 2's aggregation finding: this is a real, repeatable
result on this specific prompt style and model, not a claim that every
abstain will cleanly attribute to one check.

This exercises `calibrate_multi_check()`/`decide_multi_check()` directly,
not through `ToolRegistry` -- Phase 2's scope (PROJECT_SPEC §3 Phase 2)
is the calibration/decision layer only, with no `wrap()`-style engine
integration requested or built yet, so the demo calls the underlying tool
explicitly on accept rather than through the adapter.
