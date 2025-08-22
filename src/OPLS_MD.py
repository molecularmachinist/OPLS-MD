"""
MD simulation wrappers for the PLS/OPLS objects in OPLS.py
"""
from typing import TYPE_CHECKING, Literal, Optional, overload
import numpy as np

try:
    from MDAnalysis import Universe, AtomGroup  # type: ignore
except ImportError as e:
    _reason = e
    if (not TYPE_CHECKING):
        Universe = None
        AtomGroup = None

from .OPLS import OPLS
from .PLS import PLS
from .OPLS_PLS import OPLS_PLS

# A dirty hack to make the super functions be loaded only when type checking
if TYPE_CHECKING:
    _Base_PLS = PLS
else:
    _Base_PLS = object


class _MD_PLS_WRAPPER(_Base_PLS):
    """
    A wrapper class to wrap PLS object to accept MDAnalysis universes and atomg groups,
    as well as unflattened MD trajectories. Should NOT be instanciated as such, but through the PLS_MD class.
    The final class needs to first inherit this class, and only then the class which is wrapped.
    """
    # This "Mixin" class is used instead of inheritance to be able to use he same wrappers for PLS and OPLS

    def _get_dims(self, crd: np.ndarray | Universe | AtomGroup):
        if (type(crd) == Universe or type(crd) == AtomGroup):
            self.natoms = crd.atoms.n_atoms
            self.ndim = 3
        else:
            if (crd.ndim == 2):
                self.ndim = 1
            elif (crd.ndim == 3):
                self.ndim = crd.shape[2]
            else:
                raise ValueError(
                    f"X is {crd.ndim} dimensional, should be 2 or 3")
            self.natoms = crd.shape[1]

    def _from_crd(self, crd: np.ndarray | Universe | AtomGroup) -> np.ndarray:
        if (type(crd) == Universe):
            u = crd
            sel = crd.atoms
        elif (type(crd) == AtomGroup):
            u = crd.universe
            sel = crd
        else:
            if (crd.ndim == 2):
                return crd
            if (crd.ndim != 3):
                raise ValueError(f"X is {crd.ndim} dimensional, "
                                 "should be 2 or 3")

            if (crd.shape[1:] != (self.natoms, self.ndim)):
                raise ValueError("Wrong shape in input, the model was trained "
                                 f"with shape{(-1, self.natoms, self.ndim)}, "
                                 f"input is {crd.shape}")

            return crd.reshape((crd.shape[0], self.natoms*self.ndim))

        if (sel.n_atoms != self.natoms):
            raise ValueError("Wrong number of atoms, the model was trained "
                             f"with {self.natoms}, input has {sel.n_atoms}")

        X = np.empty((len(u.trajectory), self.natoms*self.ndim))
        for i, ts in enumerate(u.trajectory):
            X[i] = sel.positions.ravel()
        return X

    def _to_crd(self, X: np.ndarray) -> np.ndarray:
        return X.reshape(X.shape[0], self.natoms, self.ndim)

    def fit(self, crd: np.ndarray | Universe | AtomGroup, y: np.ndarray):
        self._get_dims(crd)
        return super().fit(self._from_crd(crd), y)

    @overload
    def transform(self,
                  crd: np.ndarray | Universe | AtomGroup,
                  Y: np.ndarray,
                  *,
                  copy: bool = True) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def transform(self,
                  crd: np.ndarray | Universe | AtomGroup,
                  Y: None = None,
                  *,
                  copy: bool = True) -> np.ndarray:
        ...

    def transform(self,
                  crd: np.ndarray | Universe | AtomGroup,
                  Y: Optional[np.ndarray] = None,
                  *,
                  copy: bool = True):
        X = self._from_crd(crd)
        return super().transform(X, Y, copy=copy)

    @overload
    def inverse_transform(self, X: np.ndarray, Y: None = None) -> np.ndarray | Universe | AtomGroup:
        ...

    @overload
    def inverse_transform(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray | Universe | AtomGroup, np.ndarray]:
        ...

    def inverse_transform(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> \
            np.ndarray | Universe | AtomGroup | tuple[np.ndarray | Universe | AtomGroup, np.ndarray]:
        if Y is None:
            return self._from_crd(super().inverse_transform(X))
        else:
            X_new, Y_new = super().inverse_transform(X, Y)
            return self._from_crd(X_new), Y_new

    def inverse_predict(self,
                        Y: np.ndarray,
                        *,
                        ndim: Optional[int] = None,
                        copy=True):
        return self._to_crd(super().inverse_predict(Y, ndim=ndim, copy=copy))

    def predict(self,
                crd: np.ndarray | Universe | AtomGroup,
                *,
                ndim: Optional[int] = None,
                copy: bool = True):
        X = self._from_crd(crd)
        return super().predict(X, ndim=ndim, copy=copy)

    def score(self,
              crd: np.ndarray | Universe | AtomGroup,
              y: np.ndarray,
              sample_weight: Optional[np.ndarray] = None,
              *,
              ndim: Optional[int] = None,
              copy: bool = True) -> float:
        X = self._from_crd(crd)
        return super().score(X, y, sample_weight=sample_weight, ndim=ndim, copy=copy)


# A dirty hack to make the super functions be loaded only when type checking
if TYPE_CHECKING:
    class _Base_OPLS(OPLS, _MD_PLS_WRAPPER):
        ...
else:
    _Base_OPLS = _MD_PLS_WRAPPER


class _MD_OPLS_WRAPPER(_Base_OPLS):
    """
    A wrapper class to wrap OPLS object to accept MDAnalysis universes and atom groups,
    as well as unflattened MD trajectories. Should NOT be instanciated as such, but through the OPLS_MD class.
    The final class needs to first inherit this class, and only then the class which is wrapped.

    Only adds the wrapping for correct on top of _MD_PLS_WRAPPER
    """

    @overload
    def correct(self, crd: np.ndarray | Universe | AtomGroup, y: None = None, *,
                copy: bool = True,
                return_ortho: Literal[False] = False) -> np.ndarray:
        ...

    @overload
    def correct(self, crd: np.ndarray | Universe | AtomGroup, y: None = None, *,
                copy: bool = True,
                return_ortho: Literal[True] = True) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def correct(self, crd: np.ndarray | Universe | AtomGroup, y: np.ndarray, *,
                copy: bool = True,
                return_ortho: Literal[False] = False) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def correct(self, crd: np.ndarray | Universe | AtomGroup, y: np.ndarray, *,
                copy: bool = True,
                return_ortho: Literal[True] = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...

    @overload
    def correct(self, crd: np.ndarray | Universe | AtomGroup, y: Optional[np.ndarray] = None, *,
                copy: bool = True,
                return_ortho: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray] |\
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...

    def correct(self,
                crd: np.ndarray | Universe | AtomGroup,
                y: Optional[np.ndarray] = None,
                *,
                copy: bool = True,
                return_ortho: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
        X = self._from_crd(crd)
        return super().correct(X, y, copy=copy, return_ortho=return_ortho)


class PLS_MD(_MD_PLS_WRAPPER, PLS):
    """
    Partial Least Squares (PLS), wrapped such that every method accepts an MDAnalysis
    Universe or AtomGroup, or an unflattened coordinate array of shape(n, n_atoms, ndim) instead of X-coordinates.
    In case of MDAnalysis, all frames in the trajectory of Universe (or AtomGroup.universe) will be read into memory.

    The output of the inverse transformations will also be reshaped back to the orginal dimensionality.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to fit.
    scale : bool, default=True
        Whether to scale X and Y to unit variance
    center : bool, default=True
        Whether to centre X and Y to zero mean
    flip : bool, default=False
        Whether to flip the singular vectors for compatibility with different solvers.
        With flip=True and deflation_mode="regression" the PLS model will be the same
        (up to machine precision) as with sklearn.cross_decomposition.PLSRegression.
        With flip=False and univariate y, the x_scores_ are guaranteed to be positively
        correlated to y.
    max_iter : int, default=500
        Maximum number of NIPALS iterations.
    tol : float, default=1e-06
        Tolerance for stopping NIPALS iterations.
    copy : bool, default=True
        Whether to make copies of X and Y. False does not guarantee calculations are in place, but
        True does guarantee copying.
    dtype : numpy.dtype, default=float64
        numpy dtype to cast input to. If None calculations will be done with whatever dtype the input arrays have.
    deflation_mode : str, default=None
        Whether to calculate y deflation with x_scores ("regression") or y_scores ("canonical").
        The latter only is reliable with n_components<=Y.shape[1]. The first will make this the same as
        sklearn.cross_decomposition.PLSRegression and the latter the same as
        sklearn.cross_decomposition.PLSCanonical.
        In either case the first PLS component is the same.
    """
    pass


class OPLS_MD(_MD_OPLS_WRAPPER, OPLS):
    """
    Orthogonal Projection on Latent Structure (O-PLS), wrapped such that every method accepts an MDAnalysis
    Universe or AtomGroup, or an unflattened coordinate array of shape(n, n_atoms, ndim) instead of X-coordinates.
    In case of MDAnalysis, all frames in the trajectory of Universe (or AtomGroup.universe) will be read into memory.

    The output of the inverse transformations will also be reshaped back to the orginal dimensionality.

    If the input X is a numpy array with ndim=2, it is assumed to be already flattened.

    Parameters
    ----------
    n_components : int, default=1
        Number of components to fit.
    scale : bool, default=True
        Whether to scale X and Y to unit variance
    center : bool, default=True
        Whether to centre X and Y to zero mean
    flip : bool, default=False
        Whether to flip the singular vectors for compatibility with different solvers.
        With flip=True and deflation_mode="regression" the first PLS component will be
        exactly (up to machine precision) the same as with sklearn.cross_decomposition.PLSRegression.
        Does not affect the results in any other meaningful way.
    max_iter : int, default=500
        Maximum number of NIPALS iterations.
    tol : float, default=1e-06
        Tolerance for stopping NIPALS iterations.
    copy : bool, default=True
        Whether to make copies of X and Y. False does not guarantee everything is in place, but
        True does guarantee copying.
    algorithm : str, default="OPLS"
        The algorithm to use. Acceptable values are "OPLS" and "O2PLS".
        NOTE: "O2PLS" is not yet well tested, and OPLS is only tested with univariate y.
    deflation_mode : str, default=None
        Whether to calculate y deflation with x_scores ("regression") or y_scores ("canonical").
        The latter only is reliable with n_components<=Y.shape[1].
        With OPLS algorithm this only changes the y-loadings, y-rotations and the final regressor, so
        it does not affect the corrected coordinates.
        With O2PLS algorithm "canonical" should be used.
        If None, "regression" is used for OPLS and "canonical" with O2PLS.

    Attributes
    ----------
    predictive_scores: np.ndarray
        Predictive x-scores.
    predictive_loadings: np.ndarray
        Predictive x-loadings.
    orthogonal_scores: np.ndarray
        Orthogonal x-scores.
    orthogonal_loadings: np.ndarray
        Orthogonal x-loadings.
    """
    pass


class OPLS_PLS_MD(_MD_OPLS_WRAPPER, OPLS_PLS):
    """
    Orthogonal Projection on Latent Structure (O-PLS) wrapper for a Partial Least Squares (PLS)  model,
    wrapped such that every method accepts an MDAnalysis Universe or AtomGroup, or an unflattened coordinate
    array of shape(n, n_atoms, ndim) instead of X-coordinates. In case of MDAnalysis, all frames in the
    trajectory of Universe (or AtomGroup.universe) will be read into memory.

    The output of the inverse transformations will also be reshaped back to the orginal dimensionality.

    If the input X is a numpy array with ndim=2, it is assumed to be already flattened.

    Parameters
    ----------
    n_components : int, default=1
        Number of components to fit.
    pls_components : int, default=2
        Number of components to fit.
    scale : bool, default=True
        Whether to scale X and Y to unit variance
    center : bool, default=True
        Whether to centre X and Y to zero mean
    flip : bool, default=False
        Whether to flip the singular vectors for compatibility with different solvers.
        With flip=True and deflation_mode="regression" the first PLS component will be
        exactly (up to machine precision) the same as with sklearn.cross_decomposition.PLSRegression.
        Does not affect the results in any other meaningful way.
    max_iter : int, default=500
        Maximum number of NIPALS iterations.
    tol : float, default=1e-06
        Tolerance for stopping NIPALS iterations.
    copy : bool, default=True
        Whether to make copies of X and Y. False does not guarantee everything is in place, but
        True does guarantee copying.
    algorithm : str, default="OPLS"
        The algorithm to use. Acceptable values are "OPLS" and "O2PLS".
        NOTE: "O2PLS" is not yet well tested, and OPLS is only tested with univariate y.
    deflation_mode : str, default=None
        Whether to calculate y deflation with x_scores ("regression") or y_scores ("canonical").
        The latter only is reliable with n_components<=Y.shape[1].
        With OPLS algorithm this only changes the y-loadings, y-rotations and the final regressor, so
        it does not affect the corrected coordinates.
        With O2PLS algorithm "canonical" should be used.
        If None, "regression" is used for OPLS and "canonical" with O2PLS.

    Attributes
    ----------
    predictive_scores: np.ndarray
        Predictive x-scores.
    predictive_loadings: np.ndarray
        Predictive x-loadings.
    orthogonal_loadings: np.ndarray
        Orthogonal x-loadings.
    orthogonal_scores: np.ndarray
        Orthogonal x-scores.
    pls_: PLS
        The wrapped PLS model
    """

    @overload
    def transform_ortho(self, crd: np.ndarray | Universe | AtomGroup, Y: np.ndarray, *, copy: bool = True) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def transform_ortho(self, crd: np.ndarray | Universe | AtomGroup, Y: None = None, *, copy: bool = True) -> np.ndarray:
        ...

    def transform_ortho(self,
                        crd: np.ndarray | Universe | AtomGroup,
                        Y: Optional[np.ndarray] = None, *,
                        copy: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        X = self._from_crd(crd)
        return super().transform_ortho(X, Y, copy=copy)

    @overload
    def inverse_transform_ortho(self, X: np.ndarray, Y: None = None) -> np.ndarray:
        ...

    @overload
    def inverse_transform_ortho(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def inverse_transform_ortho(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if Y is None:
            return self._from_crd(super().inverse_transform_ortho(X))
        else:
            X_new, Y_new = super().inverse_transform_ortho(X, Y)
            return self._from_crd(X_new), Y_new
