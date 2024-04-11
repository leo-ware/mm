from mm.models import MixedModel
from mm.infix_api import make_vars
from mm.covariance import Identity, UniformVariance
import pandas as pd

def test_total_covariance():
    df = pd.DataFrame({
        'x': [0, 1, 0, 1, 0],
        'z': [0, 0, 0, 1, 1],
        'y': [1, 2, 3, 4, 5]
    })
    x, y, z = make_vars(df.columns)
    model = MixedModel(y.dist(x), residual_cv=Identity())
    assert model._total_covariance(df[df.x == 1]).numpy().tolist() == [[1, 0], [0, 1]]

    model = MixedModel(
        y.dist(x + (1 | z)),
        residual_cv=Identity(),
        effect_cv={(1 | z): UniformVariance(sigma_2=2)}
        )
    assert model._total_covariance(df[df.z == 0]).numpy().tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]
    assert model._total_covariance(df).numpy().tolist() == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 3]
    ]


# def test_runs():
#     df = pd.DataFrame({
#         'x': [0, 1, 0, 1, 0],
#         'z': [0, 1, 2, 3, 4],
#         'y': [1, 2, 3, 4, 5]
#     })
#     x, y, z = make_vars(df.columns)
#     model = MixedModel(y.dist(x + 1))
#     model.fit(df, max_epochs=1)
