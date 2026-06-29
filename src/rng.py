"""Random number generator wrappers used across the project.

This module provides two generators that both satisfy the same interface
(a ``normal(size)`` method), so either can be passed to ``simulate_gbm_paths``:

- ``RNG``: pseudo-random wrapper around ``numpy.random.default_rng``.
- ``QMCGenerator``: quasi-random Sobol sequences for lower variance estimates.
"""
import numpy as np


class RNG:
    """Lightweight wrapper around ``numpy.random.default_rng``.

    Parameters
    ----------
    seed : int | None, optional
        Seed for reproducible random streams. If ``None``, NumPy entropy is
        used to initialise the generator.
    """

    def __init__(self, seed=None):
        self.seed(seed)

    def seed(self, seed=None):
        """Reset the underlying generator with a new seed.

        Parameters
        ----------
        seed : int | None, optional
            Seed value passed to ``numpy.random.default_rng``.
        """
        self._generator = np.random.default_rng(seed)

    def normal(self, size, mean=0.0, std=1.0):
        """Draw normal random variates.

        Parameters
        ----------
        size : int | tuple[int, ...]
            Output shape of the draw.
        mean : float, optional
            Mean of the normal distribution (default ``0.0``).
        std : float, optional
            Standard deviation of the normal distribution (default ``1.0``).

        Returns
        -------
        numpy.ndarray
            Array of normal random variates with requested shape.
        """
        return self._generator.normal(loc=mean, scale=std, size=size)

    def uniform(self, size=None):
        """Draw uniform random variates on ``[0, 1)``.

        Parameters
        ----------
        size : int | tuple[int, ...] | None, optional
            Output shape. If ``None``, returns a scalar float.

        Returns
        -------
        float | numpy.ndarray
            Uniform draw(s) in the half-open interval ``[0, 1)``.
        """
        return self._generator.random(size=size)


class QMCGenerator:
    """Quasi-random generator using scrambled Sobol sequences.

    Produces low-discrepancy draws that cover the sample space more uniformly
    than pseudo-random numbers, giving lower pricing error at the same path
    count — especially for smooth payoffs.

    Parameters
    ----------
    dim : int
        Dimensionality of each point. Must equal the number of time steps
        when used with ``simulate_gbm_paths``.
    seed : int | None, optional
        Scramble seed for the Sobol engine. Different seeds yield independent
        scrambled variants of the same base sequence, useful for uncertainty
        estimation across multiple QMC runs.

    Notes
    -----
    Powers of two for ``n_paths`` give optimal Sobol coverage. The generator
    is stateful: successive ``normal`` calls advance the sequence.
    """

    def __init__(self, dim, seed=None):
        from scipy.stats.qmc import Sobol
        self._engine = Sobol(d=dim, scramble=True, seed=seed)
        self._dim = dim

    def normal(self, size):
        """Draw quasi-random standard normal variates via inverse-CDF transform.

        Parameters
        ----------
        size : tuple[int, int]
            Output shape ``(n, dim)`` where ``dim`` must match the generator's
            dimensionality.

        Returns
        -------
        numpy.ndarray
            Array of shape ``size`` with quasi-normal variates.

        Raises
        ------
        ValueError
            If the requested second dimension does not match ``dim``.
        """
        from scipy.stats import norm as scipy_norm
        if isinstance(size, int):
            n, d = size, 1
        else:
            n = size[0]
            d = size[1] if len(size) > 1 else 1
        if d != self._dim:
            raise ValueError(
                f"QMCGenerator has dim={self._dim} but size requests {d} columns. "
                "Initialise QMCGenerator(dim=steps) to match the path step count."
            )
        uniform = self._engine.random(n)
        # Clamp away from 0/1 to avoid ±inf from the inverse-CDF at boundaries.
        uniform = np.clip(uniform, 1e-10, 1 - 1e-10)
        return scipy_norm.ppf(uniform)
