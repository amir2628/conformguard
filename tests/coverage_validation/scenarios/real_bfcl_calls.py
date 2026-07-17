"""Real (non-synthetic) tool-call scenario, sourced from BFCL's "live" split.

See tests/coverage_validation/data/README.md for provenance and license.
Unlike the other scenarios in this directory, this one does not generate
synthetic data: it loads a fixed extract of 258 real, human-contributed
tool-calling queries, each paired with a human-verified ground-truth
function call (i.e. every entry is, by construction, a known-good call).

The nonconformity score is a deterministic, post-hoc complexity measure
computed directly from each real call's arguments and schema -- not a
model confidence signal (none is available offline), but a legitimate,
reproducible score with real variance across genuinely different queries.
"""

import json
from pathlib import Path

SCENARIO_NAME = "real_bfcl_live_calls"

_DATA_PATH = Path(__file__).parent.parent / "data" / "bfcl_live_simple_extract.json"


def _complexity_score(entry: dict) -> float:
    args = entry["args"]
    n_properties = max(entry["n_schema_properties"], 1)
    arg_fraction = len(args) / n_properties
    serialized_length = len(json.dumps(args, default=str))
    length_component = min(serialized_length / 200.0, 1.0)
    return 0.5 * arg_fraction + 0.5 * length_component


def load_scores() -> list[float]:
    """Return the fixed pool of nonconformity scores for the real call extract.

    No ``size``/``seed`` parameters (unlike the synthetic scenarios): this
    is real, fixed-size data, not something we generate more of on demand.
    """
    entries = json.loads(_DATA_PATH.read_text())
    return [_complexity_score(entry) for entry in entries]
