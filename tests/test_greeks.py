"""Validation tests for Monte Carlo Greek estimators.

Coverage includes:
- pathwise delta/vega accuracy against analytic benchmarks,
- finite-difference estimator accuracy,
- call and put branches,
- bump-input validation,
- presence of returned uncertainty fields.
"""

from src.greeks import mc_european_greeks
from src.rng import RNG
import pytest


def test_pathwise_delta_close_to_analytic():
    # Pathwise delta should converge closely to Black-Scholes delta.
    params = dict(S0=100.0, K=105.0, r=0.01, T=1.5, sigma=0.3, option_type='call')
    steps = 128
    n_paths = 300_000
    rng = RNG(seed=777)

    result = mc_european_greeks(steps=steps, n_paths=n_paths, rng=rng, **params)
    delta_pw = result['delta']['pathwise']
    delta_analytic = result['delta']['analytic']
    assert abs(delta_pw - delta_analytic) < 1e-2


def test_pathwise_vega_close_to_analytic():
    # Pathwise vega should be highly accurate for smooth European payoff.
    params = dict(S0=95.0, K=90.0, r=0.02, T=1.0, sigma=0.2, option_type='call')
    steps = 128
    n_paths = 300_000
    rng = RNG(seed=31415)

    result = mc_european_greeks(steps=steps, n_paths=n_paths, rng=rng, **params)
    vega_pw = result['vega']['pathwise']
    vega_analytic = result['vega']['analytic']
    rel_error = abs(vega_pw - vega_analytic) / vega_analytic
    assert rel_error < 5e-3


def test_finite_difference_delta_close_to_analytic():
    # CRN finite-difference delta should remain within practical MC tolerance.
    params = dict(S0=100.0, K=95.0, r=0.01, T=1.0, sigma=0.2, option_type='call')
    result = mc_european_greeks(steps=128, n_paths=300_000, rng=RNG(seed=123), **params)
    delta_fd = result['delta']['finite_difference']
    delta_analytic = result['delta']['analytic']
    assert abs(delta_fd - delta_analytic) < 2e-2


def test_finite_difference_vega_close_to_analytic():
    # CRN finite-difference vega is noisier than pathwise but should still
    # track analytic vega to a tight relative-error bound.
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.25, option_type='call')
    result = mc_european_greeks(steps=128, n_paths=300_000, rng=RNG(seed=456), **params)
    vega_fd = result['vega']['finite_difference']
    vega_analytic = result['vega']['analytic']
    rel_error = abs(vega_fd - vega_analytic) / vega_analytic
    assert rel_error < 1e-2


def test_put_branch_pathwise_and_finite_difference_sanity():
    # Exercise put-specific indicator/sign logic in both estimators.
    params = dict(S0=90.0, K=100.0, r=0.01, T=1.0, sigma=0.22, option_type='put')
    result = mc_european_greeks(steps=128, n_paths=300_000, rng=RNG(seed=789), **params)
    assert abs(result['delta']['pathwise'] - result['delta']['analytic']) < 2e-2
    assert abs(result['delta']['finite_difference'] - result['delta']['analytic']) < 2e-2
    vega_rel_pw = abs(result['vega']['pathwise'] - result['vega']['analytic']) / result['vega']['analytic']
    vega_rel_fd = abs(result['vega']['finite_difference'] - result['vega']['analytic']) / result['vega']['analytic']
    assert vega_rel_pw < 1e-2
    assert vega_rel_fd < 1.5e-2


def test_invalid_bump_sizes_raise_clear_errors():
    # Down-bump simulations require S0-h_S > 0 and sigma-h_sigma > 0.
    params = dict(S0=100.0, K=100.0, r=0.01, T=1.0, sigma=0.2, option_type='call')
    with pytest.raises(ValueError, match=r"0 < h_S < S0"):
        mc_european_greeks(steps=64, n_paths=10_000, rng=RNG(seed=1), h_S=100.0, **params)
    with pytest.raises(ValueError, match=r"0 < h_sigma < sigma"):
        mc_european_greeks(steps=64, n_paths=10_000, rng=RNG(seed=1), h_sigma=0.2, **params)


def test_greeks_output_contains_stderr_fields():
    # Contract check: all Greeks expose the expected fields.
    params = dict(S0=100.0, K=100.0, r=0.01, T=1.0, sigma=0.2, option_type='call')
    result = mc_european_greeks(steps=64, n_paths=20_000, rng=RNG(seed=42), **params)
    for greek in ("delta", "vega"):
        assert "pathwise_stderr" in result[greek]
        assert "finite_difference_stderr" in result[greek]
        assert result[greek]["pathwise_stderr"] >= 0
        assert result[greek]["finite_difference_stderr"] >= 0
    for greek in ("gamma", "theta", "rho"):
        assert "finite_difference" in result[greek]
        assert "finite_difference_stderr" in result[greek]
        assert "analytic" in result[greek]
        assert result[greek]["finite_difference_stderr"] >= 0


def test_gamma_fd_close_to_analytic():
    params = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.2, option_type='call')
    result = mc_european_greeks(steps=128, n_paths=300_000, rng=RNG(seed=11), **params)
    gamma_fd = result['gamma']['finite_difference']
    gamma_an = result['gamma']['analytic']
    rel_err = abs(gamma_fd - gamma_an) / gamma_an
    assert rel_err < 0.05


def test_theta_fd_close_to_analytic():
    params = dict(S0=100.0, K=100.0, r=0.05, T=1.0, sigma=0.2, option_type='call')
    result = mc_european_greeks(steps=128, n_paths=300_000, rng=RNG(seed=22), **params)
    theta_fd = result['theta']['finite_difference']
    theta_an = result['theta']['analytic']
    # Both are negative; compare magnitudes via relative error.
    rel_err = abs(theta_fd - theta_an) / abs(theta_an)
    assert rel_err < 0.05


def test_rho_fd_close_to_analytic():
    params = dict(S0=100.0, K=100.0, r=0.05, T=1.0, sigma=0.2, option_type='call')
    result = mc_european_greeks(steps=128, n_paths=300_000, rng=RNG(seed=33), **params)
    rho_fd = result['rho']['finite_difference']
    rho_an = result['rho']['analytic']
    rel_err = abs(rho_fd - rho_an) / abs(rho_an)
    assert rel_err < 0.05


def test_gamma_analytic_same_for_call_and_put():
    # Put-call parity implies gamma is identical for call and put.
    params_call = dict(S0=100.0, K=100.0, r=0.02, T=1.0, sigma=0.2, option_type='call')
    params_put = {**params_call, 'option_type': 'put'}
    r_call = mc_european_greeks(steps=64, n_paths=10_000, rng=RNG(seed=0), **params_call)
    r_put = mc_european_greeks(steps=64, n_paths=10_000, rng=RNG(seed=0), **params_put)
    assert r_call['gamma']['analytic'] == r_put['gamma']['analytic']


def test_h_T_validation():
    params = dict(S0=100.0, K=100.0, r=0.01, T=1.0, sigma=0.2, option_type='call')
    with pytest.raises(ValueError, match="h_T"):
        mc_european_greeks(steps=64, n_paths=1_000, rng=RNG(seed=1), h_T=1.5, **params)

