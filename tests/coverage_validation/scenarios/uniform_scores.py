"""Baseline scenario: nonconformity scores for good calls are Uniform(0, 1).

This is the simplest continuous, tie-free distribution and serves as a
sanity-check scenario -- if the coverage validator doesn't pass here, it
won't pass anywhere.
"""

import numpy as np

SCENARIO_NAME = "uniform_scores"


def generate_pool(size: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=size).tolist()
