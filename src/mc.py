"""Basic Monte Carlo pricing helpers.

This module owns baseline Monte Carlo pricing flow:
- transform simulated market outputs to pathwise payoffs,
- discount payoffs to valuation time,
- build standard result summaries with uncertainty diagnostics.

Result-schema construction is delegated to ``src.mc_results``.
"""
import numpy as np
from .mc_results import build_result_iid, discount_payoffs
from .payoffs import asian_arithmetic_payoff, barrier_payoff, european_payoff


def _mc_price(data, payoff_fn, K, r, T, option_type):
    """Generic Monte Carlo pricing routine parameterised by payoff function.

    Parameters
    ----------
    data : array-like
        Simulation output consumed by ``payoff_fn``.
        For European options this is terminal prices ``ST``; for arithmetic
        Asian options this is the full path matrix.
    payoff_fn : callable
        Function with signature ``payoff_fn(data, K, option_type)`` that
        returns pathwise intrinsic payoffs.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary from
        ``src.mc_results.build_result_iid``.
    """
    # Ensure numeric array input before passing to payoff routines.
    arr = np.asarray(data, dtype=float)
    # Compute intrinsic payoff path by path using selected payoff function.
    payoffs = payoff_fn(arr, K, option_type)
    # Convert maturity payoffs to present values.
    discounted = discount_payoffs(payoffs, r, T)
    # Build unified pricing and uncertainty summary.
    return build_result_iid(discounted)


def mc_price_european(ST, K, r, T, option_type):
    """Price a European option from simulated terminal prices.

    Parameters
    ----------
    ST : array-like
        Terminal prices at maturity for all simulated paths.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary containing ``price``,
        ``stderr``, confidence interval bounds, sample count, and ``extra``.

    Raises
    ------
    ValueError
        If ``ST`` is not a one-dimensional array-like.
    """
    # Enforce 1D terminal-price convention for European payoffs.
    st_arr = np.asarray(ST, dtype=float)
    if st_arr.ndim != 1:
        raise ValueError("ST must be a 1D array-like of terminal prices")

    return _mc_price(st_arr, european_payoff, K, r, T, option_type)


def mc_price_barrier(paths, K, r, T, barrier, barrier_type, option_type):
    """Price a barrier option from full simulated paths.

    Parameters
    ----------
    paths : array-like
        Simulated path matrix of shape ``(n_paths, n_times)``.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    barrier : float
        Barrier level.
    barrier_type : str
        One of ``"up_out"``, ``"up_in"``, ``"down_out"``, ``"down_in"``.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary.
    """
    path_arr = np.asarray(paths, dtype=float)
    if path_arr.ndim != 2:
        raise ValueError("paths must be a 2D array-like with shape (n_paths, n_times)")
    if path_arr.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    payoffs = barrier_payoff(path_arr, K, barrier, barrier_type, option_type)
    discounted = discount_payoffs(payoffs, r, T)
    return build_result_iid(discounted)


def mc_price_asian_arithmetic(paths, K, r, T, option_type):
    """Price an arithmetic-average Asian option from full simulated paths.

    Parameters
    ----------
    paths : array-like
        Simulated path matrix where rows are paths and columns are times.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    dict
        Standard Monte Carlo result dictionary containing ``price``,
        ``stderr``, confidence interval bounds, sample count, and ``extra``.

    Raises
    ------
    ValueError
        If ``paths`` is not a two-dimensional array-like with at least two
        time columns (initial value + at least one monitoring point).
    """
    # Enforce matrix input: rows are paths, columns are observation times.
    path_arr = np.asarray(paths, dtype=float)
    if path_arr.ndim != 2:
        raise ValueError("paths must be a 2D array-like with shape (n_paths, n_times)")
    # Require at least initial value and one monitored value.
    if path_arr.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    return _mc_price(path_arr, asian_arithmetic_payoff, K, r, T, option_type)
