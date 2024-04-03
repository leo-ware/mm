from mm.api import make_vars
from mm.design_matrix import DesignMatrix
import pandas as pd
import numpy as np
import pytest

def test_design_matrix():
    x, y, z = make_vars("xyz")
    data = pd.DataFrame({
        x.name: [1, 2, 3],
        y.name: [4, 5, 6],
    })
    model = y.hat() == x + 1
    dm = DesignMatrix(model, data)
    assert dm.outcome().equals(data[y.name])
    assert dm.fixed_effects().equals(data[[x.name]])

    beta, intercept = dm.fe_regress()
    assert len(beta) == 1
    assert beta.index[0] == x.name
    assert beta.values[0] == 1
    assert intercept == pytest.approx(3)

    assert dm.outcome_fe_resid().values == pytest.approx(np.array([0, 0, 0]))