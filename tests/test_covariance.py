from mm.covariance import *
from mm.exceptions import LinAlgError
import pytest

def test_parse_cv_arg():
    assert isinstance(parse_cv_arg(None), Identity)
    assert isinstance(parse_cv_arg([[1, 0], [0, 1]]), Covariance)
    assert not parse_cv_arg([[1, 0], [0, 1]]).params
    assert parse_cv_arg(3).get_matrix(3).tolist() == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]

    with pytest.raises(LinAlgError):
        parse_cv_arg([[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        parse_cv_arg([1, 2, 3])
    with pytest.raises(ValueError):
        parse_cv_arg([])

def test_identity():
    id = Identity()
    assert id.get_matrix(3).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert id.params == {}

def test_unknown_variance():
    cv = UniformVariance()
    assert cv.get_matrix(3).shape == (3, 3)
    assert cv.get_matrix_from_params(3, sigma_2=1).tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert len(cv.params) == 1
    assert list(cv.params.keys()) == ["sigma_2"]
    assert cv.param_types == {"sigma_2": tuple()}

    cv = UniformVariance(sigma_2=2.0)
    assert cv.get_matrix(3).tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

def test_explicit():
    cv = Explicit([
        [2, 1],
        [1, 2]
    ])
    assert cv.get_matrix(2).tolist() == [[2, 1], [1, 2]]
    assert cv.params == {}

    with pytest.raises(ValueError):
        cv.get_matrix(3)
    with pytest.raises(ValueError):
        Explicit([[0, 0], [0, 0]])
    with pytest.raises(ValueError):
        Explicit([[1, 2], [3, 4], [5, 6]])
