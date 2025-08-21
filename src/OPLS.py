#!/usr/bin/env python3
from typing import Literal, Optional, overload
import numpy as np
from numpy import linalg as la
from scipy.linalg import pinv

from .base_PLS import _PLS
from .utils import center_scale_data, nipals, flip_scores_by_absolute_value


class OPLS(
    _PLS
):
    """
    Orthogonal Projection on Latent Structure (O-PLS).

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
    dtype : numpy.dtype, default=float64
        numpy dtype to cast input to. If None calculations will be done with whatever dtype the input arrays have.
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

    """

    def __init__(
        self, n_components=1, *,
        scale=True,
        center=True,
        flip=False,
        max_iter=500,
        tol=1e-06,
        copy=True,
        dtype=np.float64,
        algorithm="OPLS",
        deflation_mode=None,
    ):
        if (deflation_mode is None):
            if (algorithm == "OPLS"):
                deflation_mode = "regression"
            elif (algorithm == "O2PLS"):
                deflation_mode = "canonical"

        super().__init__(
            n_components=n_components,
            scale=scale,
            center=center,
            flip=flip,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
            dtype=dtype,
            deflation_mode=deflation_mode
        )
        self.algorithm = algorithm

    def __str__(self):
        return f"{type(self).__name__}(n_components={self.n_components})"

    def __repr__(self):
        return f"{type(self).__name__}(n_components={self.n_components}, algorithm={repr(self.algorithm)}, deflation_mode={repr(self.deflation_mode)})"

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit OPLS model.

        Parameters
        ----------
        x: np.ndarray
            Variable matrix with shape (n_samples, n_features)
        y: np.ndarray
            Dependent matrix with shape (n_samples, n_targets) or (n_samples,). For now only n_targets==1 is tested.

        Returns
        -------
        Fitted OPLS object (reference to self)

        Reference
        ---------
        [1] Trygg J, Wold S. Orthogonal projection on Latent Structure (OPLS).
            J Chemometrics. 2002, 16, 119-128.
        [2] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable
            regression (LVR) method with a integral OSC filter.
            J Chemometrics. 2003, 17, 53-64.
        [3] https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition
        """
        X, Y = self.validate_input(x, y)
        if (Y.ndim == 1):
            Y = Y[:, np.newaxis]

        n, xd = X.shape
        _, yd = Y.shape

        if (self.deflation_mode not in ("canonical", "regression")):
            raise ValueError(
                f"deflation_mode=\"{self.deflation_mode}\" is not supported. Supported values are \"canonical\" and \"regression\"")

        if (self.algorithm == "OPLS"):
            npc = min(n, xd)
        elif (self.algorithm == "O2PLS"):
            npc = min(n, xd, yd)
        else:
            raise ValueError(
                f"algorithm=\"{self.algorithm}\" is not supported. Supported values are \"OPLS\" and \"O2PLS\"")

        if self.n_components > npc:
            raise ValueError(
                f"Number of components is too large for X=shape{X.shape}, Y=shape{Y.shape} and deflation_mode={self.deflation_mode}")

        n_comp = self.n_components

        if (self.n_components <= 0):
            raise ValueError(
                f"n_components should be positive nonzero integer, is {self.n_components}")

        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = center_scale_data(
            X, Y, scale=self.scale, center=self.center)
        self.intercept_ = self._y_mean

        #  Variable              | name       |    variable in sklearn user guide
        W = np.empty((xd, n_comp))  # X-weights  |     U
        C = np.empty((yd, n_comp))  # Y-weights  |     V
        T = np.empty((n, n_comp))   # X-scores   |     Xi
        U = np.empty((n, n_comp))   # Y-scores   |     Omega
        P = np.empty((xd, n_comp))  # X-loadings |     Gamma
        Q = np.empty((yd, n_comp))  # Y-loadings |     Delta

        # Orthogonal variables
        Wortho = np.empty((xd, n_comp))  # X-weights
        Tortho = np.empty((n, n_comp))   # X-scores
        Portho = np.empty((xd, n_comp))  # X-loadings
        if (self.algorithm == "O2PLS"):
            Cortho = np.empty((yd, n_comp))  # Y-weights
            Uortho = np.empty((n, n_comp))   # Y-scores
            Qortho = np.empty((yd, n_comp))  # Y-loadings

        # Y_eps = np.finfo(Y.dtype).eps

        for k in range(n_comp):
            # Replace columns that are all close to zero with zeros
            # Y_mask = np.all(np.abs(Y) < 10 * Y_eps, axis=0)
            # Y[:, Y_mask] = 0.0

            # Run nipals to get first singular vectors
            w, c, t, u = nipals(X, Y, tol=self.tol, max_iter=self.max_iter)

            if (self.flip):
                # Flip for consistency across solvers
                flip_scores_by_absolute_value(w, c)
                # recalculate scores after flip
                t = X @ w
                u = Y @ c / (c.T @ c)

            # Regress p to minimize error in Xhat = t p^T
            p = (X.T @ t) / (t.T @ t)

            # find orthogonal weights
            w_ortho = p-((w.T @ p) / (w.T @ w))*w
            w_ortho /= la.norm(w_ortho)

            # with the orthogonal weights, calculate orthogonal scores and loadings
            t_ortho = (X @ w_ortho) / (w_ortho.T @ w_ortho)
            p_ortho = (X.T @ t_ortho) / (t_ortho.T @ t_ortho)

            # orthogonal deflation of X
            X -= t_ortho @ p_ortho.T

            if (self.deflation_mode == "canonical"):
                # Regress q to minimize error in Yhat = u q^T
                q = (Y.T @ u) / (u.T @ u)
            elif (self.deflation_mode == "regression"):
                # In regression mode only x score (t) is used
                # Regress q to minimize error in Yhat = t q^T
                q = (Y.T @ t) / (t.T @ t)

            # In O2PLS we also calculate y-loadings and deflate y
            if (self.algorithm == "O2PLS"):
                # find orthogonal y-weights
                c_ortho = q-((c.T @ q) / (c.T @ c))*c
                c_ortho /= la.norm(c_ortho)

                # with the orthogonal weights, calculate orthogonal scores and loadings
                u_ortho = (Y @ c_ortho) / (c_ortho.T @ c_ortho)
                if (self.deflation_mode == "canonical"):
                    q_ortho = (Y.T @ u_ortho) / (u_ortho.T @ u_ortho)
                    # deflate y
                    Y -= u_ortho @ q_ortho.T
                elif (self.deflation_mode == "regression"):
                    # In regression mode only x score (t_ortho) is used
                    q_ortho = (Y.T @ t_ortho) / (t_ortho.T @ t_ortho)
                    # deflate y
                    Y -= t_ortho @ q_ortho.T

            W[:, k] = w.squeeze(axis=1)
            U[:, k] = u.squeeze(axis=1)
            C[:, k] = c.squeeze(axis=1)
            T[:, k] = t.squeeze(axis=1)
            P[:, k] = p.squeeze(axis=1)
            Q[:, k] = q.squeeze(axis=1)

            Wortho[:, k] = w_ortho.squeeze(axis=1)
            Tortho[:, k] = t_ortho.squeeze(axis=1)
            Portho[:, k] = p_ortho.squeeze(axis=1)
            if (self.algorithm == "O2PLS"):
                Uortho[:, k] = u_ortho.squeeze(axis=1)
                Cortho[:, k] = c_ortho.squeeze(axis=1)
                Qortho[:, k] = q_ortho.squeeze(axis=1)

        self._x_weights = W
        self._y_weights = C
        self._x_scores = T
        self._y_scores = U
        self._x_loadings = P
        self._y_loadings = Q

        self._Wortho = Wortho
        self._Tortho = Tortho
        self._Portho = Portho

        self.x_rotations_ = W @ pinv(P.T @ W, check_finite=False)
        self.y_rotations_ = C @ pinv(Q.T @ C, check_finite=False)

        # orthogonal x rotations
        self._Rortho = Wortho @ pinv(Portho.T @ Wortho, check_finite=False)

        if (self.algorithm == "O2PLS"):
            self._Uortho = Uortho
            self._Cortho = Cortho
            self._Qortho = Qortho
            # orthogonal y rotations
            self._Sortho = Cortho @ pinv(Qortho.T @ Cortho, check_finite=False)

        # "expose" all the weights, scores and loadings
        self.x_weights_ = self._x_weights
        self.y_weights_ = self._y_weights
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        self.x_loadings_ = self._x_loadings
        self.y_loadings_ = self._y_loadings

        self.fitted = True
        return self

    def predict(self, X: np.ndarray, *, ndim: Optional[int] = None, copy: bool = True) -> np.ndarray:
        raise NotImplementedError("predict not available with OPLS")

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None, *,
              ndim: Optional[int] = None,
              copy: bool = True) -> float:
        raise NotImplementedError("score not available with OPLS")

    def inverse_predict(self, Y, *, ndim: Optional[int] = None, copy: bool = True):
        raise NotImplementedError("inverse_predict not available with OPLS")

    @overload
    def transform(self, X: np.ndarray, Y: np.ndarray, *, copy: bool = True) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def transform(self, X: np.ndarray, Y: None = None, *, copy: bool = True) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray, Y: Optional[np.ndarray] = None, *, copy: bool = True) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Apply the dimension reduction to get the orthogonal scores.

        Parameters
        ----------
        X : np.ndarray
            shape(n_samples, n_features) coordinates to transform.
        Y : np.ndarray, default=None
            shape(n_samples, n_targets) targets to transform (optional)
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
        self.check_is_fitted()
        X = self._validate_array(X, variable_name="Y")
        # Normalize
        X -= self._x_mean
        X /= self._x_std
        # calculate scores
        x_scores = X @ self._Rortho
        if Y is not None:
            if (self.algorithm == "OPLS"):
                return x_scores, Y
            Y = self._validate_array(
                Y, variable_name="Y", ensure_2d=False, copy=copy
            )
            if Y.ndim == 1:
                Y = Y[:, np.newaxis]
            Y -= self._y_mean
            Y /= self._y_std
            y_scores = Y @ self._Sortho
            return x_scores, y_scores

        return x_scores

    @overload
    def inverse_transform(self, X: np.ndarray, Y: None = None) -> np.ndarray:
        ...

    @overload
    def inverse_transform(self, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def inverse_transform(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """calculate inverse of the dimension reduction.

        Parameters
        ----------
        X : np.ndarray
            shape(n_samples, n_comp) scores to inverse transform.
        Y : np.ndarray, default=None
            shape(n_samples, n_comp) scores to inverse transform (optional)

        Returns
        -------
        x_scores : np.ndarray
            shape(n_samples, n_features) estimate of X-coordinates.
        y_scores : np.ndarray
            shape(n_samples, n_targets) estimate of y-targets OR unchanged Y if algorithm=="OPLS"
            only returned if Y is not None.
        """
        self.check_is_fitted()
        X = self._validate_array(X, variable_name="X")
        # calculate scores
        x_new = X @ self._Portho.T
        x_new *= self._x_std
        x_new += self._x_mean

        if Y is not None:
            if (self.algorithm == "OPLS"):
                return x_new, Y
            Y = self._validate_array(Y, variable_name="Y")
            y_new = Y @ self._Qortho.T
            y_new *= self._y_std
            y_new += self._y_mean
            return x_new, y_new

        return x_new

    @overload
    def correct(self, X: np.ndarray, y: None = None, *,
                copy: bool = True,
                return_ortho: Literal[False] = False) -> np.ndarray:
        ...

    @overload
    def correct(self, X: np.ndarray, y: None = None, *,
                copy: bool = True,
                return_ortho: Literal[True] = True) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def correct(self, X: np.ndarray, y: np.ndarray, *,
                copy: bool = True,
                return_ortho: Literal[False] = False) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def correct(self, X: np.ndarray, y: np.ndarray, *,
                copy: bool = True,
                return_ortho: Literal[True] = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...

    @overload
    def correct(self, X: np.ndarray, y: Optional[np.ndarray] = None, *,
                copy: bool = True,
                return_ortho: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray] |\
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ...

    def correct(self, X: np.ndarray, y: Optional[np.ndarray] = None, *,
                copy: bool = True,
                return_ortho: bool = False) -> np.ndarray | tuple[np.ndarray, ...]:
        """
        Remove orthogonal components from X (and possibly y)

        Parameters
        ----------
        X: np.ndarray
            Data matrix with shape(n_samples, n_features)
        y: np.ndarray | None
            Data matrix shape(n_samples) or shape(n_samples, n_targets).
            If None, only X is corrected and returned.
            If the algorithm was OPLS, the corrected y is simply a copy of y.
        copy: bool
            Wether to work on and return copies of data. Default is True.
        return_ortho: bool
            Return orthogonal components of X (and possibly y). Default is False.

        Returns
        -------
        x_new: np.ndarray
            Corrected data, shape(n_samples, n_features).
        x_ortho: np.ndarray
            Orthogonal score, shape(n_samples, n_features). Returned if return _ortho=True.
        y_new: np.ndarray
            Corrected data, shape(n_samples, n_targets). Returned if y is not None.
        y_ortho: np.ndarray
            Orthogonal score, shape(n_samples, n_targets). Returned if y is not None and return _ortho=True.
        """
        self.check_is_fitted()
        if (not y is None):
            x_new, y_new = self.validate_input(X, y, copy=copy)
        else:
            x_new = self._validate_array(X, variable_name="X", copy=copy)
        x_new -= self._x_mean
        x_new /= self._x_std
        n, xd = x_new.shape

        if (xd != self._x_loadings.shape[0]):
            raise ValueError("Dimension mismatch in X, "
                             f"the model has been trained with nd={self._x_loadings.shape[0]},"
                             f" but X is shape{X.shape}")

        if (not y is None):
            if (len(y_new.shape) == 1):
                y_new = y_new[:, np.newaxis]
            if (y_new.shape[1] != self._y_loadings.shape[0]):
                raise ValueError("Dimension mismatch in y, "
                                 f"the model has been trained with nf={self._y_loadings.shape[0]},"
                                 f" but y is shape{y.shape}")

        correct_y = (not y is None) and self.algorithm == "O2PLS"

        if (correct_y):
            y_new -= self._y_mean
            y_new /= self._y_std

        # The division is not needed as Wortho should be orthonormal, so Wortho.T @ Wortho = I
        # If the normalisation is removed, the division will be needed
        T_ortho_new = (x_new @ self._Rortho)
        # / np.diag(self._Wortho.T @ self._Wortho)

        x_ortho = T_ortho_new @ self._Portho.T

        x_new -= x_ortho
        x_new *= self._x_std
        x_new += self._x_mean

        if correct_y:
            # The division is not needed as Cortho should be orthonormal, so Cortho.T @ Cortho = I
            U_ortho_new = (y_new @ self._Sortho)
            # / np.diag(self._Cortho.T @ self._Cortho)

            y_ortho = U_ortho_new @ self._Qortho.T

            y_new -= y_ortho

            y_new *= self._y_std
            y_new += self._y_mean

        if not return_ortho:
            if (not y is None):
                if (len(y.shape) == 1):
                    return x_new, y_new.squeeze(axis=1)
                return x_new, y_new
            return x_new

        x_ortho *= self._x_std
        x_ortho += self._x_mean
        if (not y is None):
            if (correct_y):
                y_ortho *= self._y_std
                y_ortho += self._y_mean
            else:
                y_ortho = np.zeros_like(y_new)
            if (len(y.shape) == 1):
                return x_new, x_ortho, y_new.squeeze(axis=1), y_ortho.squeeze(axis=1)
            return x_new, x_ortho, y_new, y_ortho
        return x_new, x_ortho

    @property
    def predictive_scores(self) -> np.ndarray:
        """ Orthogonal loadings. """
        return self._x_scores

    @property
    def predictive_loadings(self) -> np.ndarray:
        """ Predictive loadings. """
        return self._x_loadings

    @property
    def weights_y(self) -> np.ndarray:
        """ y scores. """
        return self._y_weights

    @property
    def orthogonal_loadings(self) -> np.ndarray:
        """ Orthogonal loadings. """
        return self._Portho

    @property
    def orthogonal_scores(self) -> np.ndarray:
        """ Orthogonal scores. """
        return self._Tortho
