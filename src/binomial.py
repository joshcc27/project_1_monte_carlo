"""CRR binomial tree pricing for European options.

Implements the Cox-Ross-Rubinstein (CRR) lattice model:
- Up factor u = exp(sigma * sqrt(dt)), down factor d = 1/u.
- Risk-neutral probability p = (exp(r*dt) - d) / (u - d).
- Backward induction from terminal payoffs to present value.
"""
import math
import numpy as np
from .validation import normalise_option_type, validate_positive, validate_positive_int


def price_binomial(S0, K, r, T, sigma, steps, option_type="call"):
    """Return the CRR binomial tree price for a European option.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike price.
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time to maturity in years.
    sigma : float
        Volatility (annualized, decimal).
    steps : int
        Number of binomial time steps. More steps → closer to Black-Scholes.
    option_type : str
        Option side, case-insensitive: ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Present value of the European option under the CRR binomial model.

    Raises
    ------
    ValueError
        If numeric inputs are not strictly positive or ``option_type`` is invalid.
    TypeError
        If ``steps`` is not an integer.
    """
    validate_positive(S0, "S0")
    validate_positive(K, "K")
    validate_positive(T, "T")
    validate_positive(sigma, "sigma")
    validate_positive_int(steps, "steps")
    option_type = normalise_option_type(option_type)

    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # Terminal asset prices at all leaf nodes (vectorised).
    j = np.arange(steps, -1, -1)
    ST = S0 * (d ** j) * (u ** (steps - j))

    if option_type == "call":
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    # Backward induction through the tree.
    for _ in range(steps):
        payoff = disc * (p * payoff[1:] + (1 - p) * payoff[:-1])

    return float(payoff[0])
