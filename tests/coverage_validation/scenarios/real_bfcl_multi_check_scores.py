"""Real (non-synthetic), two-check score pool derived from the BFCL "live" extract.

Supplementary to the primary synthetic multi-check comparison
(correlated_multi_check_scores.py): this pool is real data, but its
correlation structure is whatever it naturally is, not something this
module controls. Both scores are deterministic functions of the same
real call's arguments, so some correlation between them should be
expected -- that is disclosed here, not hidden, and is itself a
realistic illustration of PASC's point: real-world simultaneous checks
on the same call are rarely independent, which is exactly the regime
where the joint (max-score) method's advantage over Bonferroni is
largest. See tests/coverage_validation/data/README.md for provenance.
"""

import json
from pathlib import Path

import numpy as np

SCENARIO_NAME = "real_bfcl_two_check"

_DATA_PATH = Path(__file__).parent.parent / "data" / "bfcl_live_simple_extract.json"


def _complexity_score(entry: dict) -> float:
    """check_a: argument verbosity (same score used in Phase 1's real_bfcl_calls.py)."""
    args = entry["args"]
    n_properties = max(entry["n_schema_properties"], 1)
    arg_fraction = len(args) / n_properties
    serialized_length = len(json.dumps(args, default=str))
    length_component = min(serialized_length / 200.0, 1.0)
    return 0.5 * arg_fraction + 0.5 * length_component


def _type_diversity_score(entry: dict) -> float:
    """check_b: a structurally different signal -- fraction of argument values that
    are not plain strings (bool/int/float/list/dict), as a proxy for argument-shape
    complexity distinct from check_a's verbosity measure.
    """
    args = entry["args"]
    if not args:
        return 0.0
    non_string = sum(1 for v in args.values() if not isinstance(v, str))
    return non_string / len(args)


def load_score_matrix() -> np.ndarray:
    """Return the fixed (258, 2) real score matrix: columns are [complexity, type_diversity]."""
    entries = json.loads(_DATA_PATH.read_text())
    rows = [[_complexity_score(e), _type_diversity_score(e)] for e in entries]
    return np.asarray(rows, dtype=float)


def load_synthetic_bad_pool(size: int, seed: int) -> np.ndarray:
    """A synthetic "bad" pool for efficiency measurement, paired with the real good pool above.

    Unlike load_score_matrix() (100% real), this is NOT real data -- BFCL's
    "live_simple" split is ground-truth-correct calls by construction, so
    there is no natural pool of "bad" real calls to draw from here.
    Constructed by resampling real good scores and shifting them up by a
    full spread-width (same shift-based construction used by
    validation/negative_control.py), so its correlation structure is
    inherited from the real data rather than invented from scratch.
    """
    real = load_score_matrix()
    rng = np.random.default_rng(seed)
    resampled = real[rng.integers(0, len(real), size=size)]
    spread = real.max(axis=0) - real.min(axis=0)
    return resampled + spread
