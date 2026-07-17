"""Scenario: most good calls score very low nonconformity, a long right tail.

Modeled on how a real model-confidence-derived score tends to look on
successful calls: most successful calls are scored confidently (low
nonconformity), but a minority still score moderately high even though the
outcome was fine. Beta(2, 5) produces exactly this shape while remaining
continuous (no ties), which the coverage band's exact formula assumes.
"""

import numpy as np

SCENARIO_NAME = "skewed_confidence_scores"


def generate_pool(size: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.beta(2.0, 5.0, size=size).tolist()
