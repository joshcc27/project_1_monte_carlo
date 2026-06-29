"""Tests for the QMCGenerator and its variance-reduction benefit."""

import numpy as np
import pytest
from src.bs_analytics import bs_price
from src.gbm import simulate_gbm_paths
from src.mc import mc_price_european
from src.rng import QMCGenerator, RNG


def test_qmc_generator_output_shape():
    qmc = QMCGenerator(dim=8, seed=0)
    out = qmc.normal(size=(16, 8))
    assert out.shape == (16, 8)


def test_qmc_generator_approximately_standard_normal():
    qmc = QMCGenerator(dim=64, seed=1)
    out = qmc.normal(size=(4096, 64))
    assert abs(out.mean()) < 0.05
    assert abs(out.std() - 1.0) < 0.05


def test_qmc_generator_rejects_wrong_dim():
    qmc = QMCGenerator(dim=8, seed=0)
    with pytest.raises(ValueError, match="dim=8"):
        qmc.normal(size=(16, 16))


def test_qmc_reduces_pricing_error_vs_mc():
    # At the same modest path count, QMC RMSE across scramble seeds should
    # beat MC RMSE across pseudo-random seeds for a smooth European call.
    S0, K, r, T, sigma = 100.0, 100.0, 0.02, 1.0, 0.2
    steps = 64
    n_paths = 1024  # power of 2 gives optimal Sobol coverage
    bs_call = bs_price(S0, K, T, r, sigma, "call")

    mc_errors = []
    for seed in range(15):
        paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=RNG(seed=seed))
        mc_errors.append(mc_price_european(paths[:, -1], K, r, T, "call")["price"] - bs_call)

    qmc_errors = []
    for seed in range(15):
        qmc = QMCGenerator(dim=steps, seed=seed)
        paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=qmc)
        qmc_errors.append(mc_price_european(paths[:, -1], K, r, T, "call")["price"] - bs_call)

    mc_rmse = np.sqrt(np.mean(np.square(mc_errors)))
    qmc_rmse = np.sqrt(np.mean(np.square(qmc_errors)))
    assert qmc_rmse < mc_rmse
