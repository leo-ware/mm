import torch
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from warnings import warn
from typing import Union, Dict

from mm.infix_api import ModelEquation, RandomEffect
from mm.covariance import Covariance, cv_lookup, UniformVariance
from mm.exceptions import DataTypeWarning
from mm.config import fp_type

UnknownDataType = Union[list, np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

class SingleEffectDesignMatrix(ABC):
    def _make_tensor(self, data: UnknownDataType, column: bool = False):
        t = torch.tensor(np.array(data)).to(fp_type)
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


class REDM(SingleEffectDesignMatrix):
    def __init__(self, data: UnknownDataType, intercept: bool = True, slope_data=None):
        self.has_intercept = bool(intercept)
        self.has_slope = slope_data is not None
        if not self.has_intercept and not self.has_slope:
            raise ValueError("random effect requires at least one of intercept and slope")
        
        if np.array(data).dtype.kind == 'f':
            warn(DataTypeWarning("Random effect received float, this is unadvisable"))
        random_df = pd.get_dummies(self._make_tensor(data, column=True).reshape(-1), dtype=float)
        if self.has_slope:
            slope_data = self._make_tensor(slope_data, column=True)
        
        self.intercept_names = list(random_df.columns)
        self.intercept_dm = torch.tensor(random_df.values)
        if self.has_slope:
            self.slope_dm = slope_data * self.intercept_dm
        else:
            self.slope_dm = None
    
    @property
    def design_matrix(self) -> torch.Tensor:
        if not self.has_slope:
            return self.intercept_dm
        elif not self.has_intercept:
            return self.slope_dm
        else:
            return torch.cat([self.intercept_dm, self.slope_dm], dim=1)

class FEDM(SingleEffectDesignMatrix):
    def __init__(self, data: UnknownDataType, intercept: bool = True):
        self._data = self._make_tensor(data)
        if intercept:
            self._data = torch.cat([torch.ones(self._data.shape[0], 1), self._data], dim=1)
    
    @property
    def design_matrix(self) -> torch.Tensor:
        return self._data

class ODM(SingleEffectDesignMatrix):
    def __init__(self, data, covariance=None):
        self._data = self._make_tensor(data)
    
    @property
    def design_matrix(self) -> torch.Tensor:
        return self._data

class DesignMatrix:
    def __init__(self, data: pd.DataFrame, eq: ModelEquation):
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Expected DataFrame, got {type(data)}")
        
        self.data: DesignMatrix = data
        self.eq: ModelEquation = eq
        self.n: int = data.shape[0]

        self.outcome = ODM(data[[eq.outcome.name]])
        self.fixed_effects = FEDM(
            data[[v.variable.name for v in eq.effect.fixed_effects]],
            intercept=eq.effect.intercept
        )
        self.random_effects: Dict[RandomEffect, REDM] = {}
        for re in eq.effect.random_effects:
            self.random_effects[re] = REDM(
                data=data[[re.group.name]],
                intercept=re.intercept,
                slope_data=data[[re.slope.name]] if re.slope is not None else None,
            )
