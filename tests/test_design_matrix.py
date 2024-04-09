from mm.design_matrix import DesignMatrix, REDM, FEDM
from mm.infix_api import make_vars

import pandas as pd
import numpy as np
import pytest

def test_random_effect():
    data = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
        "z": [1, 1, 0],
    })
    redm = REDM(data=data[["z"]], intercept=True)
    assert redm.design_matrix.shape == (3, 2)
    assert redm.design_matrix[0, 0] == 0
    assert redm.intercept_names == [0, 1]
    
    redm = REDM(data=data[["z"]], slope_data=data[["x"]], intercept=True)
    assert redm.design_matrix.shape == (3, 4)
    assert redm.design_matrix.numpy().tolist() == [
        [0, 1, 0, 1],
        [0, 1, 0, 2],
        [1, 0, 3, 0],
    ]

    with pytest.raises(ValueError):
        redm = REDM(data=data[["z"]], intercept=False)
    with pytest.raises(ValueError):
        redm = REDM(data=data[["z"]], slope_data=data[["x", "y"]], intercept=False)

def test_fixed_effect():
    data = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
        "z": [1, 1, 0],
    })
    fedm = FEDM(data[["x"]], intercept=True)
    assert fedm.design_matrix.shape == (3, 2)
    assert fedm.design_matrix.numpy().tolist() == [
        [1, 1],
        [1, 2],
        [1, 3],
    ]

    fedm = FEDM(data[["x"]], intercept=False)
    assert fedm.design_matrix.shape == (3, 1)
    assert fedm.design_matrix.numpy().tolist() == [
        [1],
        [2],
        [3],
    ]

    fedm = FEDM(data[["x", "y"]], intercept=True)
    assert fedm.design_matrix.shape == (3, 3)
    assert fedm.design_matrix.numpy().tolist() == [
        [1, 1, 4],
        [1, 2, 5],
        [1, 3, 6],
    ]

def test_design_matrix():
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": [4, 5, 6],
        "z": [1, 1, 0],
    })
    x, y, z = make_vars(df.columns)

    dm = DesignMatrix(df, y.hat() == x + 1)
    assert dm.outcome.design_matrix.numpy().tolist() == [[4],[5],[6],]
    assert dm.fixed_effects.design_matrix.numpy().tolist() == [
        [1, 1],
        [1, 2],
        [1, 3],
    ]
    assert dm.random_effects == {}

    dm = DesignMatrix(df, y.hat() == x + 1 + (1 | z))
    assert dm.outcome.design_matrix.numpy().tolist() == [[4],[5],[6],]
    assert dm.fixed_effects.design_matrix.numpy().tolist() == [
        [1, 1],
        [1, 2],
        [1, 3],
    ]
    assert len(list(dm.random_effects.items()))
    zrf = list(dm.random_effects.values())[0]
    assert zrf.design_matrix.numpy().tolist() == [
        [0, 1],
        [0, 1],
        [1, 0],
    ]

    dm = DesignMatrix(df, y.hat() == (x + 1 | z))
    zrf = list(dm.random_effects.values())[0]
    assert zrf.design_matrix.numpy().tolist() == [
        [0, 1, 0, 1],
        [0, 1, 0, 2],
        [1, 0, 3, 0],
    ]
