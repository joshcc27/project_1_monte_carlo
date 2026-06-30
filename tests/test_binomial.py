"""Tests for the CRR binomial tree pricer."""
import math
import numpy as np
import pytest
from src.binomial import price_binomial

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

# Black-Scholes ATM call reference (Hull Figure 13.3).
BS_CALL = 10.450583572185565
BS_PUT  = 5.573526022256971


def test_call_converges_to_bs_at_high_steps():
    # At 2000 steps the CRR price should be within 2 cents of Black-Scholes.
    price = price_binomial(S0, K, r, T, sigma, steps=2000, option_type="call")
    assert abs(price - BS_CALL) < 0.02


def test_put_converges_to_bs_at_high_steps():
    price = price_binomial(S0, K, r, T, sigma, steps=2000, option_type="put")
    assert abs(price - BS_PUT) < 0.02


def test_put_call_parity():
    # C - P = S0 - K * exp(-rT) must hold for European options.
    call = price_binomial(S0, K, r, T, sigma, steps=1000, option_type="call")
    put  = price_binomial(S0, K, r, T, sigma, steps=1000, option_type="put")
    parity_rhs = S0 - K * math.exp(-r * T)
    assert abs((call - put) - parity_rhs) < 1e-4


def test_error_decreases_with_more_steps():
    # Absolute errors at 10, 50, 200, 1000 steps should be monotonically decreasing.
    step_counts = [10, 50, 200, 1000]
    errors = [abs(price_binomial(S0, K, r, T, sigma, steps=n) - BS_CALL)
              for n in step_counts]
    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1], (
            f"Error did not decrease: steps={step_counts[i]} err={errors[i]:.6f}, "
            f"steps={step_counts[i+1]} err={errors[i+1]:.6f}"
        )


def test_call_and_put_are_positive(european_market):
    m = european_market
    call = price_binomial(m["S0"], m["K"], m["r"], m["T"], m["sigma"],
                          steps=200, option_type="call")
    put  = price_binomial(m["S0"], m["K"], m["r"], m["T"], m["sigma"],
                          steps=200, option_type="put")
    assert call > 0
    assert put > 0


def test_rejects_non_positive_spot():
    with pytest.raises(ValueError):
        price_binomial(0.0, K, r, T, sigma, steps=100)


def test_rejects_non_positive_strike():
    with pytest.raises(ValueError):
        price_binomial(S0, -1.0, r, T, sigma, steps=100)


def test_rejects_non_positive_maturity():
    with pytest.raises(ValueError):
        price_binomial(S0, K, r, 0.0, sigma, steps=100)


def test_rejects_non_positive_sigma():
    with pytest.raises(ValueError):
        price_binomial(S0, K, r, T, 0.0, steps=100)


def test_rejects_float_steps():
    with pytest.raises(TypeError):
        price_binomial(S0, K, r, T, sigma, steps=100.0)


def test_rejects_invalid_option_type():
    with pytest.raises(ValueError):
        price_binomial(S0, K, r, T, sigma, steps=100, option_type="fwd")
