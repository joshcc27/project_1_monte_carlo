"""Tests for control-variate variance reduction behaviour.

The primary goal is to validate that the control-variate helper reduces
standard error in a realistic Asian-option setting and handles invalid input.
"""

import numpy as np
import pytest
from src.asian_geometric import price_geometric_asian
from src.gbm import simulate_gbm_paths
from src.payoffs import asian_arithmetic_payoff
from src.variance_reduction import control_variate


def test_control_variate_reduces_standard_error(asian_market, sim_medium, make_rng):
    # Use a realistic Asian-option scenario where geometric and arithmetic
    # payoffs are strongly correlated, making control variates effective.
    params = dict(asian_market)
    steps = sim_medium["steps"]
    n_paths = sim_medium["n_paths"]

    rng = make_rng(2024)
    paths = simulate_gbm_paths(params["S0"], params["r"], params["sigma"], params["T"], steps, n_paths, rng=rng)

    discount = np.exp(-params["r"] * params["T"])
    # Target estimator X: discounted arithmetic Asian payoff samples.
    arithmetic = discount * asian_arithmetic_payoff(paths, params["K"], 'call')

    # Control estimator Y: discounted geometric Asian payoff samples.
    # EY is provided by the closed-form geometric Asian formula.
    geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    geo_payoffs = discount * np.maximum(geo_avg - params["K"], 0.0)
    geo_price = price_geometric_asian(
        params["S0"], params["K"], params["r"], params["sigma"], params["T"], steps, 'call'
    )

    est_cv, stderr_cv, _ = control_variate(arithmetic, geo_payoffs, geo_price)
    plain_stderr = arithmetic.std(ddof=1) / np.sqrt(n_paths)

    # Expect material error reduction from control variates.
    assert plain_stderr / stderr_cv > 1.5
    assert abs(est_cv - geo_price) < 10  # sanity bound


def test_control_variate_rejects_empty_inputs():
    # Empty series should fail fast with a clear validation error.
    with pytest.raises(ValueError, match="non-empty"):
        control_variate([], [], 0.0)


def test_mc_price_asian_arithmetic_cv_result_schema(asian_market, make_rng):
    from src.variance_reduction import mc_price_asian_arithmetic_cv
    params = dict(asian_market)
    result = mc_price_asian_arithmetic_cv(
        params["S0"], params["K"], params["r"], params["T"], params["sigma"],
        steps=32, n_paths=10_000, rng=make_rng(1), option_type="call",
    )
    assert set(result.keys()) == {"price", "stderr", "ci_low", "ci_high", "n_paths", "extra"}
    assert result["extra"]["variance_reduction"] == "control_variate"
    assert "cv_beta" in result["extra"]
    assert "geo_asian_price" in result["extra"]
    assert result["ci_low"] < result["price"] < result["ci_high"]


def test_mc_price_asian_arithmetic_cv_reduces_stderr(asian_market, sim_medium, make_rng):
    from src.variance_reduction import mc_price_asian_arithmetic_cv
    from src.mc import mc_price_asian_arithmetic
    from src.gbm import simulate_gbm_paths
    params = dict(asian_market)
    steps = sim_medium["steps"]
    n_paths = sim_medium["n_paths"]

    rng_plain = make_rng(77)
    paths = simulate_gbm_paths(
        params["S0"], params["r"], params["sigma"], params["T"], steps, n_paths, rng=rng_plain
    )
    plain = mc_price_asian_arithmetic(paths, params["K"], params["r"], params["T"], "call")

    rng_cv = make_rng(77)
    cv = mc_price_asian_arithmetic_cv(
        params["S0"], params["K"], params["r"], params["T"], params["sigma"],
        steps, n_paths, rng_cv, "call",
    )

    # CV should materially reduce standard error.
    assert plain["stderr"] / cv["stderr"] > 1.5


def test_mc_price_asian_arithmetic_cv_put_branch(asian_market, make_rng):
    from src.variance_reduction import mc_price_asian_arithmetic_cv
    params = dict(asian_market)
    result = mc_price_asian_arithmetic_cv(
        params["S0"], params["K"], params["r"], params["T"], params["sigma"],
        steps=64, n_paths=50_000, rng=make_rng(99), option_type="put",
    )
    assert result["price"] >= 0
    assert result["stderr"] >= 0
