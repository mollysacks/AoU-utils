import numpy as np
import pytest

from aou_utils.bq import _param_spec


def test_scalar_types():
    assert _param_spec(5) == ("INT64", False, 5)
    assert _param_spec(np.int64(5)) == ("INT64", False, 5)
    assert _param_spec(0.5) == ("FLOAT64", False, 0.5)
    assert _param_spec("x") == ("STRING", False, "x")
    assert _param_spec(True) == ("BOOL", False, True)


def test_array_of_numpy_ints_becomes_python_ints():
    bq_type, is_array, values = _param_spec(np.array([1, 2, 3]))
    assert (bq_type, is_array, values) == ("INT64", True, [1, 2, 3])
    assert all(type(v) is int for v in values)


def test_empty_array_raises():
    with pytest.raises(ValueError):
        _param_spec([])
