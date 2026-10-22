import numpy as np


def divided_differences(x, y):
    """Return Newton divided-difference coefficients for nodes x and values y."""
    x = np.asarray(x, dtype=float)
    coef = np.array(y, dtype=float)
    n = len(x)
    for j in range(1, n):
        coef[j:] = (coef[j:] - coef[j - 1:-1]) / (x[j:] - x[:n - j])
    return coef


def newton_eval(x, coef, xq):
    """Evaluate the Newton polynomial at query points xq (Horner-style)."""
    x = np.asarray(x, dtype=float)
    xq = np.asarray(xq, dtype=float)
    result = np.full_like(xq, coef[-1], dtype=float)
    for k in range(len(coef) - 2, -1, -1):
        result = result * (xq - x[k]) + coef[k]
    return result