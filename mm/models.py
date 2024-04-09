import torch
import pandas as pd
from typing import Union

from mm.design_matrix import DesignMatrix
from mm.exceptions import UninitializedError, LinAlgError
from mm.infix_api import ModelEquation
from mm.utils import is_invertible
from mm.covariance import parse_cv_arg, Covariance

class GLS:
    def __init__(self, eq: ModelEquation, eps_cov=None):
        if eq.has_random:
            raise ValueError("GLS does not support random effects")
        self.eq = eq
        self.eps_cov = parse_cv_arg(eps_cov)
        self.beta_hat = None
    
    def _get_dm(self, data: pd.DataFrame):
        return DesignMatrix(data=data, eq=self.eq)
    
    def fit(self, data: pd.DataFrame):
        dm = self._get_dm(data)
        if not isinstance(self.eps_cov, Covariance):
            self.eps_cov = self.eps_cov(dm.n)
        
        X = dm.fixed_effects.design_matrix
        y = dm.outcome.design_matrix
        V = self.eps_cov.matrix

        _X_T_V_inv = X.T @ torch.inverse(V)
        self.beta_hat = torch.inverse(_X_T_V_inv @ X) @ _X_T_V_inv @ y

    def predict(self, data: pd.DataFrame):
        X = self._get_dm(data).fixed_effects.design_matrix
        return X @ self.beta_hat

class MixedModel:
    def __init__(self, eq: ModelEquation):
        self.eq = eq
    
    def _init_params(self):
        pass

    def log_likelihood(self, data: Union[DesignMatrix, pd.DataFrame]):
        pass
    
    def fit(self, data: pd.DataFrame):
        pass

    def predict(self, data: pd.DataFrame):
        pass
