"""Variance reduction techniques for Monte Carlo pricers.

This module provides two common Monte Carlo variance-reduction methods:

- Antithetic variates: pairs each shock matrix row ``Z`` with ``-Z``.
- Control variates: adjusts an estimator ``X`` using a correlated control ``Y``
  with known expectation ``E[Y]``.

The module focuses on variance-reduction mechanics. Result-schema construction
and confidence-interval conventions are delegated to ``src.mc_results``.
"""
import numpy as np
from .asian_geometric import price_geometric_asian
from .gbm import simulate_gbm_paths
from .mc_results import build_result_antithetic, build_result_iid, discount_payoffs
from .payoffs import asian_arithmetic_payoff, european_payoff


def _antithetic_shocks(rng, n_paths, steps):
    """Generate antithetic standard-normal shocks.

    Parameters
    ----------
    rng : object
        Random source exposing ``normal(size=...)``.
    n_paths : int
        Total number of Monte Carlo paths. Must be even.
    steps : int
        Number of time increments per path.

    Returns
    -------
    numpy.ndarray
        Shock matrix with shape ``(n_paths, steps)`` arranged as
        ``[Z_block, -Z_block]``.

    Raises
    ------
    ValueError
        If ``n_paths`` is odd.
    """
    # Antithetic pairing requires one negative counterpart per base draw.
    if n_paths % 2 != 0:
        raise ValueError("Antithetic variates require an even number of paths")

    # Draw only half the shocks directly, then mirror to create symmetric pairs.
    half = n_paths // 2
    z_half = rng.normal(size=(half, steps))
    # Stack Z and -Z so each path has an explicit antithetic partner.
    return np.concatenate([z_half, -z_half], axis=0)


def simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths, rng):
    """Simulate GBM paths using antithetic shocks.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Volatility (annualised, decimal).
    T : float
        Maturity in years.
    steps : int
        Number of time increments.
    n_paths : int
        Number of total paths (must be even for antithetic pairing).
    rng : object
        Random source exposing ``normal(size=...)``.

    Returns
    -------
    numpy.ndarray
        Simulated GBM path matrix of shape ``(n_paths, steps + 1)``, ordered
        by paired blocks ``[Z-block, -Z-block]``.
    """

    # Build paired shocks first, then reuse standard GBM simulator.
    shocks = _antithetic_shocks(rng, n_paths, steps)
    return simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, shocks=shocks)


def mc_price_european_antithetic(S0, K, r, T, sigma, steps, n_paths, rng, option_type):
    """Price a European option with antithetic Monte Carlo sampling.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Maturity in years.
    sigma : float
        Volatility (annualised, decimal).
    steps : int
        Number of time increments.
    n_paths : int
        Number of total paths (must be even).
    rng : object
        Random source exposing ``normal(size=...)``.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard MC summary with antithetic-aware standard error and
        metadata ``extra["variance_reduction"] = "antithetic"``.

    Notes
    -----
    Standard error is computed from antithetic pair means via
    ``src.mc_results.build_result_antithetic``.
    """

    # 1) Generate antithetic paths with explicit Z/-Z pairing.
    paths = simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths, rng)
    # 2) Compute terminal European payoff for each path.
    ST = paths[:, -1]
    payoffs = european_payoff(ST, K, option_type)
    # 3) Discount to t=0 and summarise using pair-mean statistics.
    discounted = discount_payoffs(payoffs, r, T)
    return build_result_antithetic(discounted, extra={"variance_reduction": "antithetic"})


def mc_price_asian_arithmetic_antithetic(S0, K, r, T, sigma, steps, n_paths, rng, option_type):
    """Price an arithmetic-average Asian option with antithetic sampling.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Maturity in years.
    sigma : float
        Volatility (annualised, decimal).
    steps : int
        Number of time increments.
    n_paths : int
        Number of total paths (must be even).
    rng : object
        Random source exposing ``normal(size=...)``.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard MC summary with antithetic-aware standard error and
        metadata ``extra["variance_reduction"] = "antithetic"``.

    Notes
    -----
    Standard error is computed from antithetic pair means via
    ``src.mc_results.build_result_antithetic``.
    """

    # 1) Generate antithetic GBM paths with explicit Z/-Z pairing.
    paths = simulate_gbm_paths_antithetic(S0, r, sigma, T, steps, n_paths, rng)
    # 2) Compute arithmetic-average Asian payoff path by path.
    payoffs = asian_arithmetic_payoff(paths, K, option_type)
    # 3) Discount and summarise with antithetic pair-aware statistics.
    discounted = discount_payoffs(payoffs, r, T)
    return build_result_antithetic(discounted, extra={"variance_reduction": "antithetic"})


def control_variate(X, Y, EY):
    """Apply a control-variate adjustment to estimator samples.

    Parameters
    ----------
    X : array-like
        Sample values of target estimator (for example discounted arithmetic
        Asian payoff paths).
    Y : array-like
        Sample values of correlated control estimator, same shape as ``X``.
    EY : float
        Known expectation of the control estimator ``Y``.

    Returns
    -------
    tuple[float, float, float]
        ``(est, stderr, b)`` where:
        - ``est`` is adjusted estimator mean,
        - ``stderr`` is standard error of adjusted samples,
        - ``b`` is the estimated optimal control coefficient.

        The adjusted sample is
        ``X_cv = X - b * (Y - EY)``
        with ``b`` estimated from sample covariance/variance.

    Raises
    ------
    ValueError
        If ``X`` and ``Y`` have different shapes, are not one-dimensional, or
        are empty.
    """

    # Convert once to float arrays so moments/covariance are consistent.
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    if X.ndim != 1:
        raise ValueError("X and Y must be 1D arrays")
    if X.size == 0:
        raise ValueError("X and Y must be non-empty")

    # Centre both series to compute sample covariance and control variance.
    X_mean = X.mean()
    Y_mean = Y.mean()
    centred_X = X - X_mean
    centred_Y = Y - Y_mean
    if X.size > 1:
        # Unbiased sample covariance and variance (ddof=1 form).
        cov = np.sum(centred_X * centred_Y) / (X.size - 1)
        var_Y = np.sum(centred_Y**2) / (X.size - 1)
    else:
        # Single sample: no variance information available.
        cov = 0.0
        var_Y = 0.0
    if var_Y == 0:
        # Degenerate control (or one sample): no adjustment possible.
        b = 0.0
    else:
        # Estimated optimal coefficient minimising adjusted variance.
        b = cov / var_Y

    # Apply control-variate correction to each sample, then estimate mean/SE.
    X_cv = X - b * (Y - EY)
    est = X_cv.mean()
    stderr = X_cv.std(ddof=1) / np.sqrt(X_cv.size) if X_cv.size > 1 else 0.0
    return est, stderr, b


def mc_price_asian_arithmetic_cv(S0, K, r, T, sigma, steps, n_paths, rng, option_type):
    """Price an arithmetic-average Asian option using geometric Asian control variates.

    Uses the closed-form geometric Asian price as the control variate expectation,
    exploiting the high correlation between arithmetic and geometric averages under GBM
    to materially reduce standard error vs plain Monte Carlo.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Maturity in years.
    sigma : float
        Volatility (annualised, decimal).
    steps : int
        Number of equally spaced monitoring intervals.
    n_paths : int
        Number of Monte Carlo paths.
    rng : object
        Random source exposing ``normal(size=...)``.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard MC result dictionary with ``price``, ``stderr``, ``ci_low``,
        ``ci_high``, ``n_paths``, and ``extra``. ``extra`` contains:
        - ``variance_reduction``: ``"control_variate"``
        - ``cv_beta``: estimated optimal control coefficient
        - ``geo_asian_price``: closed-form geometric Asian price used as EY
    """
    paths = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, rng=rng)
    discount = np.exp(-r * T)

    # Target: discounted arithmetic Asian payoff samples.
    arithmetic_payoffs = discount * asian_arithmetic_payoff(paths, K, option_type)

    # Control: discounted geometric Asian payoff samples with known expectation.
    geo_avg = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
    if option_type.lower() == "call":
        geo_payoffs = discount * np.maximum(geo_avg - K, 0.0)
    else:
        geo_payoffs = discount * np.maximum(K - geo_avg, 0.0)

    geo_price = price_geometric_asian(S0, K, r, sigma, T, steps, option_type)

    est, stderr, b = control_variate(arithmetic_payoffs, geo_payoffs, geo_price)

    result = build_result_iid(arithmetic_payoffs)
    # Overwrite price/stderr with CV-adjusted estimates; CI uses adjusted stderr.
    from .mc_results import Z_95
    result["price"] = est
    result["stderr"] = stderr
    result["ci_low"] = est - Z_95 * stderr
    result["ci_high"] = est + Z_95 * stderr
    result["extra"] = {
        "variance_reduction": "control_variate",
        "cv_beta": b,
        "geo_asian_price": geo_price,
    }
    return result
