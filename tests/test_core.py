import numpy as np
from interp.core import divided_differences, newton_eval


def test_reproduces_nodes():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 5.0])
    coef = divided_differences(x, y)
    assert np.allclose(newton_eval(x, coef, x), y)


def test_exact_for_quadratic():
    x = np.array([0.0, 1.0, 2.0])
    y = x**2 + 1
    coef = divided_differences(x, y)
    xq = np.linspace(0, 2, 7)
    assert np.allclose(newton_eval(x, coef, xq), xq**2 + 1)