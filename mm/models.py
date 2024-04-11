import torch
import pandas as pd
from typing import Union, Dict

from mm.design_matrix import DesignMatrix
from mm.exceptions import UninitializedError, LinAlgError
from mm.infix_api import ModelEquation, RandomEffect
from mm.utils import is_invertible
from mm.covariance import parse_cv_arg, Covariance
from mm.gaussian import Gaussian
from mm.optimizers import parse_optimizer_param, adam

class GLS:
    def __init__(self, eq: ModelEquation, eps_cov=None):
        if eq.has_random:
            raise ValueError("GLS does not support random effects")
        self.eq = eq
        self.eps_cov = parse_cv_arg(eps_cov)
        self.beta_hat: Union[None, torch.Tensor] = None
    
    def _get_dm(self, data: Union[DesignMatrix, pd.DataFrame]) -> DesignMatrix:
        if isinstance(data, DesignMatrix):
            return data
        return DesignMatrix(data=data, eq=self.eq)
    
    # if we get passed a newly initialized CV instance this will return garbage
    def fit(self, data: pd.DataFrame) -> None:
        dm = self._get_dm(data)
        X = dm.fixed_effects.design_matrix
        y = dm.outcome.design_matrix
        V = self.eps_cov.get_matrix_from_params(dim=dm.n)
        _X_T_V_inv = X.T @ torch.inverse(V)
        self.beta_hat = torch.inverse(_X_T_V_inv @ X) @ _X_T_V_inv @ y

    def predict(self, data: pd.DataFrame) -> torch.Tensor:
        X = self._get_dm(data).fixed_effects.design_matrix
        return X @ self.beta_hat

class MixedModel:
    pass
    # def __init__(self, eq: ModelEquation, residual_cv: Union[None, Covariance] = None, effect_cv: Union[None, Dict[RandomEffect, Covariance]] = None):
    #     self.eq = eq
    #     self.residual_cv: Covariance = parse_cv_arg(residual_cv)
    #     self.random_cv: Dict[RandomEffect, Covariance] = {}
    #     for r_effect in eq.effect.random_effects:
    #         self.random_cv[r_effect] = parse_cv_arg((effect_cv or {}).get(r_effect, None))
    #     self._fixed_coef: Union[None, torch.Tensor] = None
    
    # @property
    # def fixed_coef(self):
    #     if self._fixed_coef is None:
    #         raise UninitializedError("Model has not been fit yet")
    #     return self._fixed_coef
    
    # @property
    # def params(self):
    #     return {
    #         'residual_cv': self.residual_cv.params,
    #         'random_cv': {k: v.params for k, v in self.random_cv.items()},
    #         'fixed_coef': self.fixed_coef
    #     }
    
    # @property
    # def params_list(self):
    #     return list({
    #         *self.residual_cv.params.values(),
    #         *[rcv.params.values() for rcv in self.random_cv.values()],
    #         self.fixed_coef
    #     })
    
    # def _total_covariance(self, data: Union[DesignMatrix, pd.DataFrame]):
    #     dm = data if isinstance(data, DesignMatrix) else DesignMatrix(data, self.eq)
    #     V = self.residual_cv.get_matrix(dim=dm.n)
    #     for r_effect, redm in dm.random_effects.items():
    #         rdm = redm.design_matrix
    #         rcv = self.random_cv[r_effect].get_matrix(dim=rdm.shape[1])
    #         V += rdm @ rcv @ rdm.T
    #     return V
    
    # def _restricted_log_likelihood(self, data: Union[DesignMatrix, pd.DataFrame]):
    #     dm = data if isinstance(data, DesignMatrix) else DesignMatrix(data, self.eq)
    #     total_covariance = self._total_covariance(dm)

    #     gls_est = GLS(self.eq.outcome.dist(self.eq.effect.fixed_effects), total_covariance)
    #     gls_est.fit(dm)
    #     fixed_coefs_hat = gls_est.beta_hat

    #     outcome_model = Gaussian(mean=dm.fixed_effects.design_matrix @ fixed_coefs_hat, covariance=total_covariance)
    #     restricted_log_likelihood = (
    #         outcome_model.log_likelihood(dm.outcome.design_matrix) # is this scaled by the right factor?
    #         -0.5*torch.log(torch.det(
    #             dm.fixed_effects.design_matrix.T @
    #             torch.inverse(total_covariance) @
    #             dm.fixed_effects.design_matrix
    #         ))
    #     )
    #     return restricted_log_likelihood
    
    # def fit(self, data: Union[DesignMatrix, pd.DataFrame], optimizer='adam', **kwargs):
    #     dm = data if isinstance(data, DesignMatrix) else DesignMatrix(data, self.eq)
    #     optimize = adam#parse_optimizer_param(optimizer)

    #     self._fixed_coef = torch.randn(dm.fixed_effects.design_matrix.shape[1], 1, requires_grad=True)
    #     return optimize(
    #         params=self.params_list,
    #         loss_fn=lambda: self._restricted_log_likelihood(dm),
    #         **kwargs
    #     )

    # def predict(self, data: pd.DataFrame):
    #     pass
