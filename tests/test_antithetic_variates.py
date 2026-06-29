"""Focused tests for antithetic-variates implementation details.

This file verifies that antithetic Monte Carlo reports uncertainty using
pair means (the correct effective sampling unit) rather than raw path count.
"""

import numpy as np

from src.payoffs import european_payoff
from src.variance_reduction import (
    mc_price_european_antithetic,
    simulate_gbm_paths_antithetic,
)


def test_antithetic_european_stderr_uses_pair_means(european_market, make_rng):
    # Build one antithetic result through the public wrapper.
    params = dict(european_market)
    steps = 64
    n_paths = 100_000

    rng_for_wrapper = make_rng(123)
    result = mc_price_european_antithetic(
        params["S0"], params["K"], params["r"], params["T"], params["sigma"], steps, n_paths, rng_for_wrapper, "call"
    )

    # Recreate the same antithetic sample manually and compute paired stderr
    # directly to confirm implementation-level correctness.
    rng_for_manual = make_rng(123)
    paths = simulate_gbm_paths_antithetic(
        params["S0"], params["r"], params["sigma"], params["T"], steps, n_paths, rng_for_manual
    )
    ST = paths[:, -1]
    discounted = np.exp(-params["r"] * params["T"]) * european_payoff(ST, params["K"], "call")
    n_pairs = n_paths // 2
    pair_means = 0.5 * (discounted[:n_pairs] + discounted[n_pairs:])
    expected_stderr = pair_means.std(ddof=1) / np.sqrt(n_pairs)

    # Wrapper stderr should match manual paired estimate exactly.
    assert np.isclose(result["stderr"], expected_stderr, rtol=0, atol=1e-12)
