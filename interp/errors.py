"""Node generators and error measurement for the interpolation experiments."""

import numpy as np

from interp.evaluate import interpolate


def _check_interval(a, b, n):
    """Validate the interval [a, b] and the number of nodes n."""
    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError("a and b must be finite numbers")
    if a >= b:
        raise ValueError("a must be smaller than b")
    if int(n) != n or n < 2:
        raise ValueError("n must be an integer >= 2")


def equispaced_nodes(a, b, n):
    """Return n equally spaced nodes on [a, b], including both endpoints."""
    _check_interval(a, b, n)
    return np.linspace(a, b, int(n))


def chebyshev_nodes(a, b, n):
    """Return n Chebyshev nodes (first kind) on [a, b], in increasing order.

    x_k = (a+b)/2 + (b-a)/2 * cos((2k+1) * pi / (2n)),  k = 0, ..., n-1
    """
    _check_interval(a, b, n)
    n = int(n)
    k = np.arange(n)
    x = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k + 1) * np.pi / (2 * n))
    return np.sort(x)


def fine_grid(a, b, num=2001):
    """Return a fine grid of num points on [a, b] for measuring the error."""
    if a >= b:
        raise ValueError("a must be smaller than b")
    if num < 2:
        raise ValueError("num must be at least 2")
    return np.linspace(a, b, int(num))


def max_abs_error(f, x_nodes, grid):
    """Max |f(x) - p(x)| over grid, where p interpolates f at x_nodes.

    f must accept a NumPy array and return an array of the same shape.
    """
    x_nodes = np.asarray(x_nodes, dtype=float)
    grid = np.asarray(grid, dtype=float)
    y_nodes = f(x_nodes)
    p = interpolate(x_nodes, y_nodes, grid)
    return float(np.max(np.abs(f(grid) - p)))


def error_table(f, a, b, n_values, num_grid=2001):
    """Compare equispaced and Chebyshev nodes for each n in n_values.

    Returns a list of dicts with keys 'n', 'equispaced' and 'chebyshev',
    where the last two are the maximum absolute errors on a fine grid.
    """
    grid = fine_grid(a, b, num_grid)
    rows = []
    for n in n_values:
        rows.append(
            {
                "n": int(n),
                "equispaced": max_abs_error(f, equispaced_nodes(a, b, n), grid),
                "chebyshev": max_abs_error(f, chebyshev_nodes(a, b, n), grid),
            }
        )
    return rows


def format_error_table(rows):
    """Return the rows of error_table as a readable text table."""
    lines = [f"{'n':>4} {'equispaced':>14} {'chebyshev':>14}"]
    for r in rows:
        lines.append(f"{r['n']:>4} {r['equispaced']:>14.3e} {r['chebyshev']:>14.3e}")
    return "\n".join(lines)
