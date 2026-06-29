"""Unit and statistical sanity tests for GBM path simulation.

These tests validate structural properties (shape, initial level, input
validation) and a distribution-level check against GBM terminal moments.
"""

import numpy as np
import pytest

from src.gbm import simulate_gbm_paths
from src.rng import RNG


def test_simulate_gbm_paths_shape_and_initial_level():
    # Basic contract: output is (n_paths, steps + 1) and starts at S0.
    S0 = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    steps = 8
    n_paths = 16

    rng = RNG(seed=7)
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)

    assert paths.shape == (n_paths, steps + 1)
    assert np.allclose(paths[:, 0], S0)


def test_simulate_gbm_paths_with_explicit_shocks_matches_formula():
    # With zero shocks, each path follows deterministic drift-only evolution
    # in log-space, so rows should match a known closed-form sequence.
    S0 = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    steps = 4
    n_paths = 3
    shocks = np.zeros((n_paths, steps))

    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, shocks=shocks)

    dt = T / steps
    drift = (r - 0.5 * sigma * sigma) * dt
    expected = S0 * np.exp(np.arange(steps + 1) * drift)

    assert np.allclose(paths, expected[None, :])


def test_simulate_gbm_paths_rejects_float_steps():
    # steps must be an integer; passing a float should raise TypeError.
    with pytest.raises(TypeError, match="steps must be an integer"):
        simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, steps=10.0, n_paths=100, rng=RNG(seed=1))


def test_simulate_gbm_paths_rejects_float_n_paths():
    with pytest.raises(TypeError, match="n_paths must be an integer"):
        simulate_gbm_paths(100.0, 0.05, 0.2, 1.0, steps=10, n_paths=100.0, rng=RNG(seed=1))


def test_simulate_gbm_paths_rejects_invalid_shocks_shape():
    # Shock matrix must match (n_paths, steps) exactly.
    with pytest.raises(ValueError, match=r"shocks must have shape"):
        simulate_gbm_paths(
            S0=100.0,
            r=0.03,
            sigma=0.25,
            T=1.0,
            steps=5,
            n_paths=10,
            shocks=np.zeros((10, 4)),
        )


def test_simulate_gbm_paths_terminal_moments_match_gbm_theory():
    # Large-sample Monte Carlo check: empirical terminal mean/variance should
    # be close to GBM theoretical moments.
    S0 = 100.0
    r = 0.03
    sigma = 0.2
    T = 1.0
    steps = 252
    n_paths = 200_000

    rng = RNG(seed=123)
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)
    ST = paths[:, -1]

    empirical_mean = ST.mean()
    empirical_var = ST.var()
    theoretical_mean = S0 * np.exp(r * T)
    theoretical_var = (S0 ** 2) * np.exp(2 * r * T) * (np.exp(sigma * sigma * T) - 1.0)

    assert abs(empirical_mean - theoretical_mean) / theoretical_mean < 0.01
    assert abs(empirical_var - theoretical_var) / theoretical_var < 0.03
