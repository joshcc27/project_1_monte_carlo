"""Payoff helpers for European and Asian options.

This module contains vectorised payoff functions that transform simulated
price outputs into cashflows before discounting.
"""
import numpy as np
from .validation import normalise_option_type


def european_payoff(ST, K, option_type):
    """Return vectorised European option payoff from terminal prices.

    Parameters
    ----------
    ST : array-like
        Terminal asset prices at maturity for one or more simulated paths.
    K : float
        Strike price.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    numpy.ndarray
        Pathwise intrinsic values at maturity:
        - call: ``max(ST - K, 0)``
        - put: ``max(K - ST, 0)``

    Raises
    ------
    ValueError
        If ``option_type`` is not ``"call"`` or ``"put"``.
    """

    t = normalise_option_type(option_type)
    if t == "call":
        # Call payoff increases with terminal price above strike.
        return np.maximum(ST - K, 0.0)
    # Put payoff increases as terminal price falls below strike.
    return np.maximum(K - ST, 0.0)


def barrier_payoff(paths, K, barrier, barrier_type, option_type):
    """Return vectorised barrier option payoff from full simulated paths.

    The barrier is checked at every monitoring date (columns 1 onward).
    Knock-out/knock-in is determined by whether the barrier is breached at
    any monitored time step.

    Parameters
    ----------
    paths : numpy.ndarray
        Simulated path matrix of shape ``(n_paths, n_times)`` where column 0
        is the initial spot and remaining columns are monitoring observations.
    K : float
        Strike price.
    barrier : float
        Barrier level.
    barrier_type : str
        One of ``"up_out"``, ``"up_in"``, ``"down_out"``, ``"down_in"``.
        ``"up"`` means the barrier is above the initial spot; ``"out"`` means
        the option is extinguished when the barrier is touched.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    numpy.ndarray
        Pathwise payoffs after applying the barrier condition.

    Raises
    ------
    ValueError
        If ``barrier_type`` is not one of the four recognised strings, or if
        the path matrix has fewer than two columns.
    """
    if paths.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    bt = barrier_type.lower()
    valid = {"up_out", "up_in", "down_out", "down_in"}
    if bt not in valid:
        raise ValueError(f"barrier_type must be one of {sorted(valid)}, got '{barrier_type}'")

    monitoring = paths[:, 1:]
    if "up" in bt:
        crossed = np.any(monitoring >= barrier, axis=1)
    else:
        crossed = np.any(monitoring <= barrier, axis=1)

    vanilla = european_payoff(paths[:, -1], K, option_type)

    if "out" in bt:
        return np.where(crossed, 0.0, vanilla)
    return np.where(crossed, vanilla, 0.0)


def asian_arithmetic_payoff(paths, K, option_type):
    """Return vectorised arithmetic-average Asian option payoff.

    Parameters
    ----------
    paths : numpy.ndarray
        Simulated path matrix of shape ``(n_paths, n_times)`` where column 0 is
        the initial spot and remaining columns are monitoring observations.
    K : float
        Strike price.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    numpy.ndarray
        Pathwise payoff based on arithmetic average of monitored prices:
        - call: ``max(avg_price - K, 0)``
        - put: ``max(K - avg_price, 0)``

    Raises
    ------
    ValueError
        If paths do not include at least one monitoring date beyond ``S0`` or
        if ``option_type`` is invalid.
    """

    # Require at least two columns: initial value + one monitored value
    if paths.shape[1] < 2:
        raise ValueError("paths must include at least one monitoring date beyond S0")

    t = normalise_option_type(option_type)
    # Averaging starts at the first monitoring date and runs to maturity.
    avg_prices = paths[:, 1:].mean(axis=1)

    if t == "call":
        # Positive only when average price exceeds strike.
        return np.maximum(avg_prices - K, 0.0)
    # Positive only when strike exceeds average price.
    return np.maximum(K - avg_prices, 0.0)
