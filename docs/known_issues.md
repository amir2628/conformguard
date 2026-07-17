# Known Issues

Tracked, real gaps and limitations, kept discoverable here rather than
buried in a docstring or scattered across commit messages. See
`CHANGELOG.md` for what's already been fixed; this file is for what
hasn't. Each entry states its status plainly -- confirmed root cause vs.
observed-but-not-explained, fixed-as-mitigation vs. actually fixed -- and
is not to be softened into sounding more resolved than it is.

## 1. Some Ollama-served models never produce structured tool calls

**Status:** confirmed root cause for Command R7B. A second model
(`phi4-mini`) shows the same symptom; its mechanism was not independently
root-caused the way Command R7B's was.

**What happens:** the model is sent a standard OpenAI-compatible chat
completion request with `tools=[...]` (and, for Command R7B, even with
`tool_choice="required"`, which should force a call per the OpenAI API
spec) and never returns a `tool_calls` response -- `finish_reason` comes
back `"stop"`, and the model either declines conversationally or emits a
garbled, non-structured approximation of a tool call as plain text
content.

**Root cause, confirmed directly for Command R7B** (`docs/real_world_validation.md`
round 2 §2): Ollama's default chat template for this model hardcodes
every assistant turn to begin with `<|START_RESPONSE|>`, even though the
model's own system preamble instructs it to begin with
`<|START_THINKING|>`/`<|START_ACTION|>` when a tool call is needed. This
was confirmed by bypassing the template entirely via a raw completion
request with a corrected turn prefix, which then did produce
tool-call-shaped output from the same model. This is an inference-server
template bug, not a model limitation, and not something fixable from
this library's adapter, which operates on the standard chat-completion
surface and has no access to Ollama's internal prompt templating.

**`phi4-mini`, observed but not root-caused** (`docs/real_world_validation.md`
round 4): same symptom (never returns `tool_calls`, narrates and echoes a
garbled pseudo-call as text instead), confirmed repeatable across 3
trials, but the template-bypass investigation done for Command R7B was
not repeated for this model. Listed here as the same category of failure,
not confirmed to be the same mechanism.

**Impact / mitigation, not a fix:** `ToolRegistry.handle_openai_choice()`
and `.handle_anthropic_message()` raise a typed `NoToolCallProducedError`
when `required=True` and no tool call was produced, instead of silently
returning nothing. This makes the failure loud and attributable
(`finish_reason`/`stop_reason` and the model's actual content are
attached to the exception) -- it does not make the affected model able to
call tools through this adapter. There is no workaround in this library
for actually getting a tool call out of an affected model/template
combination; doing so would require hand-driving the model's native
tool-use protocol outside the standard OpenAI-compatible endpoint, the
way the Command R7B root cause was confirmed.

## 2. Phase 2 multi-check calibration has no SQLite storage integration

**Status:** open, not started. Deliberately deferred, not an oversight.

**What's missing:** `calibrate_multi_check()` / `decide_multi_check()`
(`core/multi_check.py`) take in-memory `(ToolCallContext, outcome)` pairs
directly, the same as Phase 1's `calibrate()`. But the CLI's
`multi-check-threshold` / `multi-check-coverage-check` commands
(`cli/main.py`) read a JSON file (`_load_multi_check_data`) rather than
`storage/calibration_store.py`'s SQLite store, because that store's
schema is one score per row with no notion of "these K rows are
simultaneous checks on the same call" -- there is no shared identifier
linking a call's K scores together, and no `check_name` column
distinguishing them.

**Why not built now:** retrofitting that grouping is real, separate
storage-layer work (a schema migration: at minimum a shared call
identifier and a check-name column, or a dedicated multi-check table),
not something to bolt on speculatively onto a CLI convenience feature.
Phase 1's single-check storage schema and Phase 2's calibration/decision
logic were both built and proven correct before this gap needed to be
closed, and closing it isn't required for Phase 2's own acceptance
criteria (PROJECT_SPEC.md §8a, local planning doc).

**What it would take:** a versioned schema migration to
`storage/calibration_store.py` adding the call-grouping and check-name
columns, `CalibrationStore` methods to write/read K-row groups atomically,
and a CLI change to read from the store instead of (or in addition to)
the JSON file format. Not started.

**Workaround:** use the JSON file format directly, or call
`calibrate_multi_check()`/`decide_multi_check()` from a script that reads
whatever calibration data source you already have and constructs
`ToolCallContext` objects itself.
