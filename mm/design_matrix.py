import torch
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from mm.infix_api import ModelEquation
from mm.covariance import Covariance, cv_lookup, UniformVariance

class SingleEffectDesignMatrix(ABC):
    def _get_covariance_matrix(self, dim, cv) -> Covariance:
        if isinstance(cv, Covariance):
            if cv.d != dim:
                raise ValueError(f"Dimension mismatch: {cv.d} != {dim}")
            else:
                return cv
        elif isinstance(cv, str):
            return cv_lookup(cv)(dim)
        elif cv is None:
            return UniformVariance(dim)
    
    def _make_tensor(self, data, column=False):
        t = torch.tensor(np.array(data)).to(float)
        if len(t.shape) == 1:
            t = t.reshape(-1, 1)
        elif len(t.shape) != 2:
            raise ValueError(f"Expected 1D or 2D array, got shape {t.shape}")
        if column and t.shape[1] != 1:
            raise ValueError(f"Expected column vector, got shape {t.shape}")
        return t
    
    @property
    @abstractmethod
    def design_matrix(self) -> torch.Tensor:
        pass
    
    @property
    @abstractmethod
    def covariance_model(self) -> Covariance:
        pass


class REDM(SingleEffectDesignMatrix):
    def __init__(self, data, intercept = True, slope_data=None, covariance=None):
        self.has_intercept = bool(intercept)
        self.has_slope = slope_data is not None
        if not self.has_intercept and not self.has_slope:
            raise ValueError("random effect requires at least one of intercept and slope")
        
        random_df = pd.get_dummies(self._make_tensor(data, column=True).reshape(-1), dtype=float)
        if self.has_slope:
            slope_data = self._make_tensor(slope_data, column=True)
        
        self.intercept_names = list(random_df.columns)
        self.intercept_dm = torch.tensor(random_df.values)
        if self.has_slope:
            self.slope_dm = slope_data * self.intercept_dm
        else:
            self.slope_dm = None
        self._covariance = self._get_covariance_matrix(self.intercept_dm.shape[1], covariance)
    
    @property
    def design_matrix(self):
        if not self.has_slope:
            return self.intercept_dm
        elif not self.has_intercept:
            return self.slope_dm
        else:
            return torch.cat([self.intercept_dm, self.slope_dm], dim=1)
    
    @property
    def covariance_model(self):
        return self._covariance

class FEDM(SingleEffectDesignMatrix):
    def __init__(self, data, intercept=True, covariance=None):
        self._data = self._make_tensor(data)
        if intercept:
            self._data = torch.cat([torch.ones(self._data.shape[0], 1), self._data], dim=1)
        self._covariance = self._get_covariance_matrix(self._data.shape[1], covariance)
        if not isinstance(self._covariance, UniformVariance):
            raise ValueError("Fixed effects must be independent")
    
    @property
    def design_matrix(self):
        return self._data
    
    @property
    def covariance_model(self):
        return self._covariance

class ODM(SingleEffectDesignMatrix):
    def __init__(self, data, covariance=None):
        self._data = self._make_tensor(data)
        self._covariance = self._get_covariance_matrix(self._data.shape[1], covariance)
    
    @property
    def design_matrix(self):
        return self._data
    
    @property
    def covariance_model(self):
        return self._covariance

class DesignMatrix:
    def __init__(self, data: pd.DataFrame, eq: ModelEquation):
        self.data = data
        self.eq = eq
        self.n = data.shape[0]

        self.outcome = ODM(data[[eq.outcome.name]])
        self.fixed_effects = FEDM(
            data[[v.variable.name for v in eq.effect.fixed_effects]],
            intercept=eq.effect.intercept
        )
        self.random_effects = {}
        for re in eq.effect.random_effects:
            self.random_effects[re] = REDM(
                data=data[[re.group.name]],
                intercept=re.intercept,
                slope_data=data[[re.slope.name]] if re.slope is not None else None,
            )
