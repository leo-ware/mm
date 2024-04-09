from abc import ABC, abstractmethod
from typing import List, Union, Callable
import torch
import numpy as np

from mm.exceptions import LinAlgError
from mm.utils import is_invertible

class Covariance(ABC):
    def __init__(self, dim, fixed=False):
        self.dim = dim
        self._fixed = fixed
    
    def _loudly_check_invertible(self):
        if not is_invertible(self.matrix):
            raise LinAlgError("Covariance matrix must be invertible")
    
    @property
    def fixed(self):
        return self._fixed
    
    @property
    @abstractmethod
    def params(self) -> List[torch.Tensor]:
        pass

    @property
    @abstractmethod
    def matrix(self) -> torch.Tensor:
        pass

def cv_lookup(cv):
    _cv_registry = {
        "uniform": UniformVariance
    }
    if not isinstance(cv, str):
        raise TypeError("cv_lookup expects a string")
    else:
        try:
            return _cv_registry[cv]
        except KeyError:
            raise ValueError(f"Unknown covariance type: {cv}")

def parse_cv_arg(cv_arg, dim=None) -> Union[Covariance, Callable[int, Covariance]]:
    if isinstance(cv_arg, str):
        cv_class = cv_lookup(cv_arg)
    elif cv_arg is None:
        cv_class = Identity
    elif (
        isinstance(cv_arg, torch.Tensor) or
        isinstance(cv_arg, np.ndarray) or
        isinstance(cv_arg, list)
        ):
        if isinstance(cv_arg, list):
            cv_arg = np.array(cv_arg)
        if not (len(cv_arg.shape) == 2):
            raise ValueError("Covariance matrix must be 2D")
        cv = CustomCovariance(cv_arg)
        if dim and cv.dim != dim:
            raise ValueError(f"Dimension mismatch: {cv.dim} vs {dim}")
        return cv
    else:
        raise ValueError("Invalid covariance argument")
    
    if dim is None:
        return cv_class
    else:
        return cv_class(dim)

class CustomCovariance(Covariance):
    def __init__(self, matrix):
        super().__init__(matrix.shape[0], fixed=True)
        self._matrix = torch.tensor(matrix)
        self._loudly_check_invertible()
    
    @property
    def params(self):
        return []
    
    @property
    def matrix(self):
        return self._matrix.to(float)

class Identity(Covariance):
    @property
    def params(self):
        return []
    
    @property
    def matrix(self):
        return torch.eye(self.dim).to(float)

class UniformVariance(Covariance):
    def __init__(self, dim, sigma=1.0, fixed=False):
        super().__init__(dim, fixed=fixed)
        self.scale = torch.tensor(sigma, requires_grad=not fixed)

    @property
    def params(self):
        return [self.scale] * (not self.fixed)

    @property
    def matrix(self):
        return (self.scale * torch.eye(self.dim)).to(float)
