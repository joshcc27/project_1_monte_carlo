"""Shared input validation helpers used across pricing modules."""

import numpy as np


def validate_positive(value, name):
    """Require a scalar value to be strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_positive_int(value, name):
    """Require a value to be a strictly positive integer."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def validate_non_empty_1d_array(values, name):
    """Return values as a float array and require it is a non-empty 1D vector."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D array")
    return arr


def normalise_option_type(option_type):
    """Normalise and validate option side."""
    t = option_type.lower()
    if t not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    return t
