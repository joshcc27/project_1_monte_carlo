"""Tests for barrier option payoffs and Monte Carlo barrier pricer."""

import numpy as np
import pytest
from src.gbm import simulate_gbm_paths
from src.mc import mc_price_barrier, mc_price_european
from src.payoffs import barrier_payoff, european_payoff
from src.rng import RNG


# ---------------------------------------------------------------------------
# barrier_payoff unit tests
# ---------------------------------------------------------------------------

def test_down_out_knocked_path_gets_zero_payoff():
    # A path that dips below the barrier should receive zero payoff.
    # Shape: (2 paths, 3 times including S0)
    paths = np.array([
        [100.0, 80.0, 110.0],   # crosses barrier=90 → knocked out
        [100.0, 95.0, 115.0],   # stays above barrier → survives
    ])
    payoffs = barrier_payoff(paths, K=100.0, barrier=90.0, barrier_type="down_out", option_type="call")
    assert payoffs[0] == 0.0                        # knocked out
    assert payoffs[1] == pytest.approx(15.0)        # vanilla call payoff


def test_down_in_only_pays_when_crossed():
    paths = np.array([
        [100.0, 80.0, 115.0],   # crosses barrier=90 → activated
        [100.0, 95.0, 115.0],   # never crosses → zero
    ])
    payoffs = barrier_payoff(paths, K=100.0, barrier=90.0, barrier_type="down_in", option_type="call")
    assert payoffs[0] == pytest.approx(15.0)
    assert payoffs[1] == 0.0


def test_up_out_knocked_path_gets_zero():
    paths = np.array([
        [100.0, 130.0, 105.0],  # crosses barrier=120 → knocked out
        [100.0, 110.0, 108.0],  # stays below → survives
    ])
    payoffs = barrier_payoff(paths, K=100.0, barrier=120.0, barrier_type="up_out", option_type="call")
    assert payoffs[0] == 0.0
    assert payoffs[1] == pytest.approx(8.0)


def test_up_in_only_pays_when_crossed():
    paths = np.array([
        [100.0, 130.0, 108.0],  # crosses barrier=120 → activated
        [100.0, 110.0, 108.0],  # never crosses → zero
    ])
    payoffs = barrier_payoff(paths, K=100.0, barrier=120.0, barrier_type="up_in", option_type="call")
    assert payoffs[0] == pytest.approx(8.0)
    assert payoffs[1] == 0.0


def test_knock_in_plus_knock_out_equals_vanilla():
    # In-out parity: knock-in payoff + knock-out payoff = vanilla payoff (same paths).
    rng = RNG(seed=42)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=64, n_paths=10_000, rng=rng)
    K, barrier = 100.0, 85.0
    out_payoffs = barrier_payoff(paths, K, barrier, "down_out", "call")
    in_payoffs = barrier_payoff(paths, K, barrier, "down_in", "call")
    vanilla = european_payoff(paths[:, -1], K, "call")
    np.testing.assert_allclose(out_payoffs + in_payoffs, vanilla)


def test_barrier_payoff_rejects_invalid_barrier_type():
    paths = np.ones((5, 3))
    with pytest.raises(ValueError, match="barrier_type"):
        barrier_payoff(paths, K=100.0, barrier=90.0, barrier_type="sideways_out", option_type="call")


def test_barrier_payoff_rejects_single_column_path():
    with pytest.raises(ValueError, match="monitoring date"):
        barrier_payoff(np.ones((5, 1)), K=100.0, barrier=90.0, barrier_type="down_out", option_type="call")


# ---------------------------------------------------------------------------
# mc_price_barrier integration tests
# ---------------------------------------------------------------------------

def test_mc_price_barrier_result_schema(make_rng):
    rng = make_rng(1)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=32, n_paths=10_000, rng=rng)
    result = mc_price_barrier(paths, K=100.0, r=0.02, T=1.0, barrier=85.0,
                              barrier_type="down_out", option_type="call")
    assert set(result.keys()) == {"price", "stderr", "ci_low", "ci_high", "n_paths", "extra"}
    assert result["price"] >= 0
    assert result["ci_low"] < result["price"] < result["ci_high"]


def test_mc_price_barrier_rejects_non_2d_input():
    with pytest.raises(ValueError, match="2D"):
        mc_price_barrier(np.array([100.0, 101.0]), K=100.0, r=0.02, T=1.0,
                         barrier=85.0, barrier_type="down_out", option_type="call")


def test_down_out_call_cheaper_than_vanilla(make_rng):
    # A down-and-out call is strictly less valuable than a vanilla call
    # (it can be knocked out, so it can only be worth less or equal).
    rng = make_rng(7)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=128, n_paths=200_000, rng=rng)
    vanilla = mc_price_european(paths[:, -1], K=100.0, r=0.02, T=1.0, option_type="call")
    barrier_result = mc_price_barrier(paths, K=100.0, r=0.02, T=1.0, barrier=80.0,
                                      barrier_type="down_out", option_type="call")
    assert barrier_result["price"] < vanilla["price"]


def test_down_out_with_zero_barrier_matches_vanilla(make_rng):
    # A barrier of 0 can never be breached under GBM, so the knock-out price
    # should match the plain European price.
    rng = make_rng(99)
    paths = simulate_gbm_paths(100.0, 0.02, 0.2, 1.0, steps=64, n_paths=100_000, rng=rng)
    vanilla = mc_price_european(paths[:, -1], K=100.0, r=0.02, T=1.0, option_type="call")
    barrier_result = mc_price_barrier(paths, K=100.0, r=0.02, T=1.0, barrier=0.0,
                                      barrier_type="down_out", option_type="call")
    assert abs(barrier_result["price"] - vanilla["price"]) < 1e-10
