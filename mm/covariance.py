from abc import ABC, abstractmethod
from typing import List, Union, Callable, Dict, Tuple
import torch
import numpy as np

from mm.exceptions import LinAlgError
from mm.utils import is_invertible

def _check_param_type_valid(param_types):
    for key, value in param_types.items():
        if isinstance(value, list):
            param_types[key] = value = tuple(value)
        elif value is int or value is float:
            param_types[key] = value = tuple()
        
        if not isinstance(value, tuple):
            raise TypeError(f"param annotations type(int), type(float), Tuple[int], or List[int], not type {type(value)}")
        for el in value:
            if not isinstance(el, int):
                raise TypeError(f"params must be int, Tuple[int], or List[int], not type {type(el)}")
    return param_types

class Covariance(ABC):
    def __init__(self, **fixed):
        # what do the params look like
        self.__param_types = self.__get_param_types()
        self.__param_names = set(self.__param_types.keys())

        # what are the params values
        self.__fixed = self.__typecheck_params(dict(fixed), require_all=False)
        self.__params = {k: self.__param_initializer(v)
            for k, v in self.__param_types.items()
            if k not in self.__fixed
            }
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(" +
            ", ".join(f"{k}={v.numpy().tolist()}" for k, v in {**self.params, **self.fixed_params}) +
            ")")
    
    def __param_initializer(self, ptype):
        return torch.randn(ptype, requires_grad=True)
    
    @classmethod
    def __get_param_types(cls):
        return _check_param_type_valid(cls.__annotations__)
    
    def __typecheck_params(self, params, require_all=True):
        if params is None and not require_all:
            return {}
        if not isinstance(params, dict):
            raise TypeError(f"fixed must be a dictionary, not type {type(fixed)}")
        
        pks = set(params.keys())
        if (extra := pks - self.__param_names):
            raise ValueError(f"unexpected parameters {list(extra)}")
        if require_all and (missing := self.__param_names - pks):
            raise ValueError(f"missing parameters {list(missing)}")
        
        new_params = {}
        for key, value in params.items():
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            if not self.__param_types[key] == value.shape:
                raise ValueError(f"parameter did not match expected shape, expected {self.__param_types[key]}, got {value.shape}")
            new_params[key] = value
        
        return new_params
    
    @property
    def params(self):
        """Returns current values of fitted params"""
        return self.__params
    
    @property
    def param_types(self) -> Dict[str, Union[Tuple[int]]]:
        """Returns the sizes of all parameters"""
        return self.__param_types
    
    @property
    def fixed_params(self):
        """Returns values of fixed params"""
        return self.__fixed
    
    def get_matrix(self, dim):
        """Constructs a covariance matrix of size (dim, dim) using the current parameters"""
        return self._derive_matrix(dim, **self.params, **self.fixed_params).to(float)
    
    def get_matrix_from_params(self, dim, params=None, **kwarg_params):
        """Constructs a covariance matrix of size (dim, dim) using the given parameters"""
        if not isinstance(dim, int):
            raise TypeError(f"dim must be an int, not type {type(dim)}")
        if params is not None and not isinstance(params, dict):
            raise TypeError(f"params must be a dictionary or None, not type {type(params)}")
        params = (params or {}) | dict(kwarg_params)
        if (pf_overlap := set(params.keys()) & self.fixed_params.keys()):
            raise ValueError(f"parameters {list(pf_overlap)} are fixed and cannot be changed")
        params = self.__typecheck_params(params | self.fixed_params, require_all=True)
        return self._derive_matrix(dim, **params).to(float)

    @abstractmethod
    def _derive_matrix(self, dim: int, **params) -> torch.Tensor:
        """Covariance subclasses must define this method to create a covariance matrix of size (dim, dim) from the params"""
        pass

def cv_lookup(cv):
    _cv_registry = {
        "identity": Identity,
        "uniform": UniformVariance
    }
    if not isinstance(cv, str):
        raise TypeError("cv_lookup expects a string")
    else:
        try:
            return _cv_registry[cv]()
        except KeyError:
            raise ValueError(f"Unknown covariance type {cv}, supply one of {list(_cv_registry.keys())}")

def parse_cv_arg(cv_arg) -> Covariance:
    """Tries to guess what kind of covariance matrix the user wants

    - if an instance of Covariance is passed, it is returned
    - if a string is passed, it looks for a covariance matrix with that name
    - if an iterable is passed, interprets it as an explicit covariance matrix
    - if a number is passed, interprets it as a variance
    - if None is passed, assumes variance is 1
    """
    if isinstance(cv_arg, Covariance):
        return cv_arg
    elif isinstance(cv_arg, str):
        return cv_lookup(cv_arg)
    elif cv_arg is None:
        return Identity()
    elif isinstance(cv_arg, (torch.Tensor, np.ndarray, list)):
        return Explicit(cv_arg)
    elif isinstance(cv_arg, (int, float)):
        return UniformVariance(sigma_2=cv_arg)
    else:
        raise TypeError(f"cv_arg expected str, Covariance, or None, not type {type(cv_arg)}")

class Explicit(Covariance):
    def __init__(self, matrix):
        super().__init__()
        if not isinstance(matrix, torch.Tensor):
            matrix = torch.tensor(matrix)
        matrix = matrix.to(float)
        self._matrix = matrix
        
        # spot check for validity
        if not is_invertible(self._matrix) or not (self._matrix == self._matrix.T).all():
            raise LinAlgError("Covariance matrix must be symmetric and invertible")
    
    def _derive_matrix(self, dim):
        if self._matrix.shape != (dim, dim):
            raise ValueError(f"Dimension mismatch {self._matrix.shape} vs {dim}")
        return self._matrix

class Identity(Covariance):
    def _derive_matrix(self, dim):
        return torch.eye(dim)

class UniformVariance(Covariance):
    sigma_2: float
    def _derive_matrix(self, dim, sigma_2):
        return torch.eye(dim) * sigma_2
