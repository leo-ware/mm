from mm.covariance import *
from mm.exceptions import LinAlgError
import pytest

def test_cv_lookup():
    assert cv_lookup("uniform") == UniformVariance
    with pytest.raises(ValueError):
        cv_lookup("unknown")

def test_parse_cv_arg():
    assert parse_cv_arg(None) == Identity
    assert parse_cv_arg("uniform") == UniformVariance
    assert parse_cv_arg([[1, 0], [0, 1]]).dim == 2

    with pytest.raises(LinAlgError):
        parse_cv_arg([[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        parse_cv_arg(3)
    with pytest.raises(ValueError):
        parse_cv_arg([1, 2, 3])
    with pytest.raises(ValueError):
        parse_cv_arg([[1, 0], [0, 1]], 3)
    with pytest.raises(ValueError):
        parse_cv_arg([])

def test_uniform_variance():
    uv = UniformVariance(3)
    assert uv.dim == 3
    assert uv.params == [uv.scale]
    assert uv.matrix.shape == (3, 3)
    assert uv.matrix.diag().tolist() == [1, 1, 1]

    uv = UniformVariance(3, sigma=2.0)
    assert uv.dim == 3
    assert uv.params == [uv.scale]
    assert uv.matrix.shape == (3, 3)
    assert uv.matrix.diag().tolist() == [2, 2, 2]

    uv = UniformVariance(3, fixed=True)
    assert uv.dim == 3
    assert uv.params == []
    assert uv.matrix.shape == (3, 3)
    assert uv.matrix.diag().tolist() == [1, 1, 1]
