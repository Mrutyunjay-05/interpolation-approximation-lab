import numpy as np
import pytest

from interp.evaluate import interpolate, validate_inputs


def test_different_lengths():
    with pytest.raises(ValueError):
        validate_inputs([0, 1, 2], [1, 2])


def test_repeated_nodes():
    with pytest.raises(ValueError):
        validate_inputs([0, 1, 1], [1, 2, 3])


def test_empty_input():
    with pytest.raises(ValueError):
        validate_inputs([], [])


def test_not_1d():
    with pytest.raises(ValueError):
        validate_inputs([[0, 1], [2, 3]], [[1, 2], [3, 4]])


def test_non_numeric():
    with pytest.raises(ValueError):
        validate_inputs(["a", "b"], [1, 2])


def test_nan_or_inf():
    with pytest.raises(ValueError):
        validate_inputs([0, 1, np.nan], [1, 2, 3])
    with pytest.raises(ValueError):
        validate_inputs([0, 1, 2], [1, np.inf, 3])


def test_valid_input_returns_float_arrays():
    x, y = validate_inputs([0, 1, 2], [1, 2, 5])
    assert x.dtype == float and y.dtype == float


def test_interpolate_scalar_returns_float():
    value = interpolate([0, 1, 2], [1, 2, 5], 1.5)
    assert isinstance(value, float)
    assert np.isclose(value, 1.5**2 + 1)


def test_interpolate_array_keeps_shape():
    xq = np.linspace(0, 2, 6).reshape(2, 3)
    out = interpolate([0, 1, 2], [1, 2, 5], xq)
    assert out.shape == xq.shape
    assert np.allclose(out, xq**2 + 1)


def test_interpolate_reproduces_nodes():
    x = [0.0, 1.0, 2.0, 3.0]
    y = [1.0, 3.0, 2.0, 5.0]
    assert np.allclose(interpolate(x, y, x), y)


def test_interpolate_rejects_bad_query():
    with pytest.raises(ValueError):
        interpolate([0, 1, 2], [1, 2, 5], np.nan)
