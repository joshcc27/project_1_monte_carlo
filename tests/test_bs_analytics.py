"""Deterministic regression tests for Black-Scholes analytic functions.

These tests pin known closed-form reference values for price, delta, and vega,
and verify exact put-call parity consistency for the analytic formulas.
"""

import numpy as np
from src.bs_analytics import bs_delta, bs_price, bs_vega

S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.2

CALL_PRICE = 10.450583572185565
PUT_PRICE = 5.573526022256971
CALL_DELTA = 0.6368306511756191
PUT_DELTA = -0.3631693488243809
VEGA = 37.52403469169379


def test_bs_price_matches_known_values():
    # Compare both call and put prices to trusted benchmark values.
    assert np.isclose(bs_price(S0, K, T, r, sigma, 'call'), CALL_PRICE, rtol=0, atol=1e-9)
    assert np.isclose(bs_price(S0, K, T, r, sigma, 'put'), PUT_PRICE, rtol=0, atol=1e-9)


def test_bs_delta_matches_known_values():
    # Delta signs and magnitudes are sensitive; use tight absolute tolerance.
    assert np.isclose(bs_delta(S0, K, T, r, sigma, 'call'), CALL_DELTA, rtol=0, atol=1e-9)
    assert np.isclose(bs_delta(S0, K, T, r, sigma, 'put'), PUT_DELTA, rtol=0, atol=1e-9)


def test_bs_vega_matches_known_values():
    # Vega should match reference to near machine precision for this case.
    assert np.isclose(bs_vega(S0, K, T, r, sigma), VEGA, rtol=0, atol=1e-9)


def test_bs_gamma_matches_known_value():
    from src.bs_analytics import bs_gamma
    gamma = bs_gamma(S0, K, T, r, sigma)
    assert gamma > 0
    assert np.isclose(gamma, 0.018762017345846895, rtol=0, atol=1e-9)


def test_bs_theta_matches_known_values():
    from src.bs_analytics import bs_theta
    call_theta = bs_theta(S0, K, T, r, sigma, "call")
    put_theta = bs_theta(S0, K, T, r, sigma, "put")
    assert call_theta < 0
    assert np.isclose(call_theta, -6.414027546438196, rtol=0, atol=1e-9)
    assert np.isclose(put_theta, -1.6578804239346256, rtol=0, atol=1e-9)


def test_bs_rho_matches_known_values():
    from src.bs_analytics import bs_rho
    call_rho = bs_rho(S0, K, T, r, sigma, "call")
    put_rho = bs_rho(S0, K, T, r, sigma, "put")
    assert call_rho > 0
    assert put_rho < 0
    assert np.isclose(call_rho, 53.232481545376345, rtol=0, atol=1e-9)
    assert np.isclose(put_rho, -41.89046090469506, rtol=0, atol=1e-9)


def test_bs_put_call_parity_holds():
    # Exact Black-Scholes parity: C - P = S0 - K * exp(-rT).
    # This confirms call/put formulas are internally consistent.
    call = bs_price(S0, K, T, r, sigma, "call")
    put = bs_price(S0, K, T, r, sigma, "put")
    parity_rhs = S0 - K * np.exp(-r * T)
    assert np.isclose(call - put, parity_rhs, rtol=0, atol=1e-10)
