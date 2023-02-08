from typing import Union, Tuple

import numpy as np
from sklearn.utils.validation import check_is_fitted

from .PLS import PLS
from .OPLS import OPLS


class OPLS_PLS(OPLS):
    """
    Orthogonal Projection on Latent Structure (O-PLS) wrapper for a Partial Least Squares (PLS)  model.

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

    def __init__(self, n_components=1, pls_components=2, *, scale=True, center=True, flip=False, max_iter=500, tol=1e-06, copy=True, algorithm="OPLS", deflation_mode=None):

        super().__init__(
            n_components=n_components,
            scale=scale,
            center=center,
            deflation_mode=deflation_mode,
            algorithm=algorithm,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
            flip=flip
        )
        self.pls_components = pls_components

    def __str__(self):
        return f"{type(self).__name__}(n_components={self.n_components}, pls_components={self.pls_components})"

    def __repr__(self):
        return f"{type(self).__name__}(n_components={self.n_components}, pls_components={self.pls_components}, algorithm={repr(self.algorithm)}, deflation_mode={repr(self.deflation_mode)})"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OPLS_PLS":
        """
        Fit OPLS and PLS models.
        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by xd variables.
        y: np.ndarray
            Dependent matrix with size n samples by yd, or a vector. For now only t==1 is tested.
        n_comp: int
            Number of components, default is None, which indicates that
            largest dimension which is smaller value between n and p
            will be used.

        Returns
        -------
        Fitted OPLS_PLS object (reference to self)

        Reference
        ---------
        [1] Trygg J, Wold S. Orthogonal projection on Latent Structure (OPLS).
            J Chemometrics. 2002, 16, 119-128.
        [2] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable
            regression (LVR) method with a integral OSC filter.
            J Chemometrics. 2003, 17, 53-64.
        [3] https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition
        """

        n, xd = x.shape

        if (y.ndim == 1):
            yd = 1
        else:
            yd = y.shape[1]

        if (self.deflation_mode == "regression"):
            npc = min(n, xd)
        elif (self.deflation_mode == "canonical"):
            npc = min(n, xd, yd)
        else:
            raise ValueError(
                f"deflation_mode=\"{self.deflation_mode}\" is not supported. Supported values are \"canonical\" and \"regression\"")

        if (self.n_components+self.pls_components > npc):
            raise ValueError(
                f"Number of components is too large for X=shape{x.shape}, Y=shape{y.shape} and deflation_mode={self.deflation_mode}")

        super().fit(x, y)

        self._pls = PLS(
            n_components=self.pls_components,
            scale=self.scale,
            center=self.center,
            flip=self.flip,
            max_iter=self.max_iter,
            tol=self.tol,
            copy=self.copy,
            deflation_mode=self.deflation_mode,
        ).fit(*self.correct(x, y))
        self.pls_ = self._pls

        return self

    def transform(self, X: np.ndarray, Y: np.ndarray = None, copy=True) -> Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Remove the orthogonal components and apply the dimension reduction.

        Parameters
        ----------
        X : np.ndarray
            shape(n, xd) coordinates to transform.
        Y : np.ndarray, default=None
            shape(n,yd) targets to transform (optional)
        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        x_scores : np.ndarray
            shape(n, pls_comp) x-scores.
        y_scores : np.ndarray
            shape(n, pls_comp) y-scores, only returned if Y is not None.
        """
        if (Y is None):
            x_filt = super().correct(X, copy=copy)
            return self._pls.transform(x_filt, copy=copy)
        x_filt, y_filt = super().correct(X, Y, copy=copy)
        return self._pls.transform(x_filt, y_filt, copy=copy)

    def inverse_transform(self, X: np.ndarray, Y: np.ndarray = None) -> Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """calculate inverse of the dimension reduction.

        Parameters
        ----------
        X : np.ndarray
            shape(n, n_comp) scores to inverse transform.
        Y : np.ndarray, default=None
            shape(n, n_comp) scores to inverse transform (optional)

        Returns
        -------
        x_scores : np.ndarray
            shape(n, xd) estimate of X-coordinates.
        y_scores : np.ndarray
            shape(n, yd) estimate of y-targets, only returned if Y is not None.
        """
        return self._pls.inverse_transform(X, Y)

    def transform_ortho(self, X: np.ndarray, Y: np.ndarray = None, copy=True) -> Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Apply the dimension reduction to get the orthogonal components.

        Parameters
        ----------
        X : np.ndarray
            shape(n, xd) coordinates to transform.
        Y : np.ndarray, default=None
            shape(n,yd) targets to transform (optional)
        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        x_scores : np.ndarray
            shape(n, n_comp) orthogonal x-scores.
        y_scores : np.ndarray
            shape(n, n_comp) orthogonal y-scores OR unchanged Y if algorithm=="OPLS"
            only returned if Y is not None.
        """
        return super().transform(X, Y, copy=copy)

    def inverse_transform_ortho(self, X: np.ndarray, Y: np.ndarray = None) -> Union[
            np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """calculate inverse of the dimension reduction.

        Parameters
        ----------
        X : np.ndarray
            shape(n, n_comp) scores to inverse transform.
        Y : np.ndarray, default=None
            shape(n, n_comp) scores to inverse transform (optional)

        Returns
        -------
        x_scores : np.ndarray
            shape(n, xd) estimate of X-coordinates.
        y_scores : np.ndarray
            shape(n, yd) estimate of y-targets OR unchanged Y if algorithm=="OPLS"
            only returned if Y is not None.
        """
        return super().inverse_transform(X, Y)

    def predict(self, X: np.ndarray, ndim: int = None, copy=True) -> np.ndarray:
        check_is_fitted(self)
        X_new = self.correct(X, copy=copy)
        return self.pls_.predict(X_new, ndim, copy=copy)

    def inverse_predict(self, Y: np.ndarray, ndim: int = None, copy=True) -> np.ndarray:
        check_is_fitted(self)
        return self.pls_.inverse_predict(Y, ndim, copy=copy)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, ndim: int = None) -> float:
        check_is_fitted(self)
        return self.pls_.score(*self.correct(X, y), sample_weight=sample_weight, ndim=ndim)
