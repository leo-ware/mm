import pandas as pd
from pytest import approx

from mm.models import GLS
from mm.infix_api import make_vars
from mm.utils import listify1d

def test_gls_opt():
    x, y, z = make_vars('xyz')
    data = pd.DataFrame({
        "x": [1, 3, 5, 7, 9],
        "y": [2, 4, 6, 8, 10],
        "z": [1, 2, 3, 4, 5],
    })
    y_on_x = GLS(y.hat() == x + 1)
    y_on_x.fit(data)
    assert listify1d(y_on_x.beta_hat) == approx([1, 1])
    assert listify1d(y_on_x.predict(data)) == approx([2, 4, 6, 8, 10])

    y_on_z = GLS(y.hat() == z + 1)
    y_on_z.fit(data)
    assert listify1d(y_on_z.beta_hat) == approx([0, 2])
    assert listify1d(y_on_z.predict(data)) == approx([2, 4, 6, 8, 10])
