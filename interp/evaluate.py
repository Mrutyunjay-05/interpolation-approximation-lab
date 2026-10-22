"""Input validation and the public interpolate() function."""

import numpy as np

from interp.core import divided_differences, newton_eval


def _to_float_array(values, name):
    """Convert values to a float array, with a clear error if that fails."""
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain only numbers")


def validate_inputs(x, y):
    """Check the interpolation nodes and return them as float arrays.

    Raises ValueError if x or y is not 1-D, is empty, contains NaN or
    infinity, if the lengths differ, or if x has repeated values.
    """
    x = _to_float_array(x, "x")
    y = _to_float_array(y, "y")

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1-D arrays")
    if x.size == 0 or y.size == 0:
        raise ValueError("x and y must not be empty")
    if x.size != y.size:
        raise ValueError(
            f"x and y must have the same length (got {x.size} and {y.size})"
        )
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("x and y must not contain NaN or infinity")
    if np.unique(x).size != x.size:
        raise ValueError("x contains repeated nodes; nodes must be distinct")

    return x, y


def interpolate(x_nodes, y_nodes, xq):
    """Evaluate the Newton interpolating polynomial through the given nodes.

    xq can be a scalar or an array of any shape. The result has the same
    shape as xq (a plain float if xq is a scalar).
    """
    x, y = validate_inputs(x_nodes, y_nodes)

    xq = _to_float_array(xq, "xq")
    if not np.all(np.isfinite(xq)):
        raise ValueError("xq must not contain NaN or infinity")

    coef = divided_differences(x, y)
    result = newton_eval(x, coef, xq)

    if result.ndim == 0:
        return float(result)
    return result
