"""Scenario: scores derived from simulated model log-probabilities.

Mirrors the built-in logprob_score transform (core/scores.py):
nonconformity = 1 - exp(logprob). Log-probabilities for a successful call
are simulated as small negative numbers clustered near zero (the model was
generally confident), with a spread that occasionally produces a
moderately unconfident but still-successful call.
"""

import math

import numpy as np

SCENARIO_NAME = "logprob_derived_scores"


def generate_pool(size: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    logprobs = -rng.exponential(scale=0.3, size=size)
    return [1.0 - math.exp(lp) for lp in logprobs]
