import numpy as np
import pytest

from interp.errors import (
    chebyshev_nodes,
    equispaced_nodes,
    error_table,
    fine_grid,
    max_abs_error,
)


def runge(x):
    return 1.0 / (1.0 + 25.0 * x**2)


def test_equispaced_count_and_endpoints():
    x = equispaced_nodes(-1, 1, 5)
    assert len(x) == 5
    assert x[0] == -1 and x[-1] == 1
    assert np.allclose(np.diff(x), 0.5)


def test_chebyshev_count_inside_interval_sorted():
    x = chebyshev_nodes(-1, 1, 7)
    assert len(x) == 7
    assert np.all(x > -1) and np.all(x < 1)
    assert np.all(np.diff(x) > 0)


def test_chebyshev_symmetric_on_symmetric_interval():
    x = chebyshev_nodes(-1, 1, 8)
    assert np.allclose(x, -x[::-1])


def test_chebyshev_known_values():
    # n = 2 on [-1, 1]: nodes are -cos(pi/4) and +cos(pi/4)
    x = chebyshev_nodes(-1, 1, 2)
    assert np.allclose(x, [-np.cos(np.pi / 4), np.cos(np.pi / 4)])


def test_nodes_on_shifted_interval():
    x = chebyshev_nodes(2, 6, 5)
    assert np.all(x > 2) and np.all(x < 6)


@pytest.mark.parametrize("bad_n", [1, 0, -3, 2.5])
def test_bad_n_raises(bad_n):
    with pytest.raises(ValueError):
        equispaced_nodes(0, 1, bad_n)
    with pytest.raises(ValueError):
        chebyshev_nodes(0, 1, bad_n)


def test_bad_interval_raises():
    with pytest.raises(ValueError):
        equispaced_nodes(1, 1, 5)
    with pytest.raises(ValueError):
        chebyshev_nodes(2, 1, 5)


def test_error_is_tiny_for_polynomial_of_low_degree():
    grid = fine_grid(-1, 1)
    err = max_abs_error(lambda x: x**3 - x, equispaced_nodes(-1, 1, 6), grid)
    assert err < 1e-12


def test_runge_chebyshev_beats_equispaced():
    grid = fine_grid(-1, 1)
    err_eq = max_abs_error(runge, equispaced_nodes(-1, 1, 20), grid)
    err_ch = max_abs_error(runge, chebyshev_nodes(-1, 1, 20), grid)
    assert err_ch < err_eq


def test_error_table_structure():
    rows = error_table(np.sin, 0, 2 * np.pi, [4, 8, 12])
    assert [r["n"] for r in rows] == [4, 8, 12]
    assert all(set(r) == {"n", "equispaced", "chebyshev"} for r in rows)
    # for a smooth function the error shrinks as n grows
    assert rows[-1]["chebyshev"] < rows[0]["chebyshev"]
