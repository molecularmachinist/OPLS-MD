"""
MD simulation wrappers for the PLS/OPLS objects in OPLS.py
"""
import numpy as np

try:
    from MDAnalysis import Universe, AtomGroup
except ImportError as e:
    Universe = None
    AtomGroup = None

from .OPLS import PLS, OPLS, OPLS_PLS

from typing import Union, Tuple


class _MD_PLS_WRAPPER():
    """
    A wrapper class to wrap PLS object to accept MDAnalysis universes and atomg groups,
    as well as unflattened MD trajectories. Should NOT be instanciated as such, but through the PLS_MD class.
    The final class needs to first inherit this class, and only then the class which is wrapped.
    """

    def _get_dims(self, crd: Union[np.ndarray, Universe, AtomGroup]):
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

    def _from_crd(self, crd: Union[np.ndarray, Universe, AtomGroup]) -> np.ndarray:
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

    def fit(self, crd: Union[np.ndarray, Universe, AtomGroup], y):
        self._get_dims(crd)
        return super().fit(self._from_crd(crd), y)

    def transform(self,
                  crd: Union[np.ndarray, Universe, AtomGroup],
                  Y=None,
                  copy=True):
        X = self._from_crd(crd)
        return super().transform(X, Y, copy)

    def inverse_transform(self, X: np.ndarray, Y: np.ndarray = None):
        if Y is None:
            return self._from_crd(super().inverse_transform(X))
        else:
            X_new, Y_new = super().inverse_transform(X, Y)
            return self._from_crd(X_new), Y_new

    def inverse_predict(self,
                        Y: np.ndarray,
                        ndim: int = None,
                        copy=True):
        return self._to_crd(super().inverse_predict(Y, ndim, copy))

    def predict(self,
                crd: Union[np.ndarray, Universe, AtomGroup],
                ndim: int = None,
                copy=True):
        X = self._from_crd(crd)
        return super().predict(X, ndim, copy)


class _MD_OPLS_WRAPPER(_MD_PLS_WRAPPER):
    """
    A wrapper class to wrap OPLS object to accept MDAnalysis universes and atomg groups,
    as well as unflattened MD trajectories. Should NOT be instanciated as such, but through the OPLS_MD class.
    The final class needs to first inherit this class, and only then the class which is wrapped.

    Only adds the wrapping for correct on top of _MD_PLS_WRAPPER
    """

    def correct(self,
                crd: Union[np.ndarray, Universe, AtomGroup],
                y: np.ndarray = None,
                copy: bool = True,
                return_ortho: bool = False) -> Union[
            Tuple[np.ndarray, np.ndarray], np.ndarray]:
        X = self._from_crd(crd)
        return super().correct(X, y, copy, return_ortho)


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

    def transform_ortho(self,
                        crd: Union[np.ndarray, Universe, AtomGroup],
                        Y: np.ndarray = None,
                        copy=True) -> Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        X = self._from_crd(crd)
        return super().transform_ortho(X, Y, copy)

    def inverse_transform_ortho(self, X: np.ndarray, Y: np.ndarray = None) -> Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if Y is None:
            return self._from_crd(super().inverse_transform_ortho(X))
        else:
            X_new, Y_new = super().inverse_transform_ortho(X, Y)
            return self._from_crd(X_new), Y_new
