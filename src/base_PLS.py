from typing import Optional
import numpy as np


class _PLS:
    def __init__(self, n_components=2, *,
                 scale=True,
                 center=True,
                 flip=False,
                 max_iter=500,
                 tol=1e-06,
                 copy=True,
                 dtype=np.float64,
                 deflation_mode="regression"):
        self.n_components = n_components
        self.scale = scale
        self.center = center
        self.flip = flip
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy
        self.dtype = dtype
        self.deflation_mode = deflation_mode
        self.fitted = False

    def _validate_array(self, x: np.ndarray, *,
                        variable_name="x",
                        copy: Optional[bool] = None,
                        ensure_min_samples: int = 2,
                        ensure_2d: bool = True):
        if not isinstance(x, np.ndarray):
            raise ValueError(
                f"{variable_name} should be numpy.ndarray, is {type(x)}"
            )
        if (ensure_2d):
            if x.ndim != 2:
                raise ValueError(
                    f"{variable_name}.shape={x.shape}, should be 2d when ensure_2d=True"
                )
        else:
            if x.ndim not in (1, 2):
                raise ValueError(
                    f"{variable_name}.shape={x.shape}, should be 1d or 2d when ensure_2d=False"
                )

        if (x.shape[0] < ensure_min_samples):
            raise ValueError(
                f"{variable_name}.shape[0]={x.shape[0]}, should be at least {ensure_min_samples}"
            )

        if (copy is None):
            copy = self.copy

        dtype = self.dtype
        if (dtype is None):
            dtype = x.dtype

        return np.array(x, dtype=dtype, copy=copy)

    def validate_input(self, x: np.ndarray, y: np.ndarray, *,
                       copy: Optional[bool] = None,
                       ensure_min_samples: int = 2,
                       allow_univariate_y=True):
        x = self._validate_array(x, variable_name="x",
                                 copy=copy,
                                 ensure_min_samples=ensure_min_samples,
                                 ensure_2d=True)
        y = self._validate_array(y, variable_name="y",
                                 copy=copy,
                                 ensure_min_samples=ensure_min_samples,
                                 ensure_2d=not allow_univariate_y)
        if (x.shape[0] != y.shape[0]):
            raise ValueError(
                f"First dimension of x and y do not match, {x.shape[0]}!={y.shape[0]}"
            )

        return x, y

    def check_is_fitted(self):
        if not self.fitted:
            raise ValueError(
                f"You should run {type(self)}.fit before any prediction functions."
            )
