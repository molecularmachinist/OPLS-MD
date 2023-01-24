#!/usr/bin/env python3
import typing
from typing import Union, Tuple

import numpy as np
from numpy import linalg as la
from scipy.linalg import pinv
from sklearn.cross_decomposition._pls import (_PLS, _svd_flip_1d)
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


def center_scale_data(X: np.ndarray, Y: np.ndarray, center=True, scale=True) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Centering and scaling of 2d arrays X and Y, along the first dimension.
    The input array will not be copied and the returned references should match the input

    Parameters
    ----------
    X,Y: np.ndarray
        2d arrays of shape(n,d), shape(m,f).
    center: bool, default=True
        Whether to do the mean centering. If False also scaling is not done.
    scale: bool, default=True
        Whether to scale the data to unit variance. Effectively divides the data with the
        respective (corrected) sample standard deviations.

    Returns
    -------
    X: np.ndarray
        The corrected X of shape (n,d). Should be a reference to the input X.
    Y: np.ndarray
        The corrected Y of shape (n,f). Should be a reference to the input Y.
    x_mean: np.ndarray
        shape(d) array of the means of X. If center=False, it is all zeroes.
    y_mean: np.ndarray
        shape(f) array of the means of Y. If center=False, it is all zeroes.
    x_std: np.ndarray
        shape(d) array of the standard deviations of X. Any zero values are replaced by 1.
        if center=False or scale=False it is all ones.
    y_std: np.ndarray
        shape(f) array of the standard deviations of Y. Any zero values are replaced by 1.
        if center=False or scale=False it is all ones.
    """
    if (center):
        x_mean = X.mean(axis=0)
        y_mean = Y.mean(axis=0)
        X -= x_mean
        Y -= y_mean
    else:
        x_mean = np.zeros(X.shape[1])
        y_mean = np.zeros(Y.shape[1])

    if (scale and center):
        x_std = X.std(axis=0, ddof=1)
        y_std = Y.std(axis=0, ddof=1)
        x_std = np.where(x_std != 0, x_std, 1.0)
        y_std = np.where(y_std != 0, y_std, 1.0)
        X /= x_std
        Y /= y_std
    else:
        x_std = np.ones(X.shape[1])
        y_std = np.ones(Y.shape[1])

    return X, Y, x_mean, y_mean, x_std, y_std


def nipals(x: np.ndarray, y: np.ndarray,
           tol: float = 1e-10,
           max_iter: int = 10000) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Non-linear Iterative Partial Least Squares

    Parameters
    ----------
    x: np.ndarray
        Variable matrix with size n by d, where n number of samples,
        p number of variables.
    y: np.ndarray
        Dependent variable with size n by t.
    tol: float
        Tolerance for the convergence.
    max_iter: int
        Maximal number of iterations.

    Returns
    -------
    w: np.ndarray
        X-weights with size d by 1.
    c: np.ndarray
        Y-weight with size t by 1
    t: np.ndarray
        X-scores with size n by 1
    u: np.ndarray
        Y-scores with size n by 1.

    References
    ----------
    [1] Wold S, et al. PLS-regression: a basic tool of chemometrics.
        Chemometr Intell Lab Sys 2001, 58, 109–130.
    [2] Bylesjo M, et al. Model Based Preprocessing and Background
        Elimination: OSC, OPLS, and O2PLS. in Comprehensive Chemometrics.
    """
    u = y[:, 0]
    u = u[:, np.newaxis]
    i = 0
    d = tol * 10
    while d > tol and i <= max_iter:
        w = (x.T @ u) / (u.T @ u)
        w /= la.norm(w)
        t = x @ w
        c = y.T @ t / (t.T @ t)
        u_new = y @ c / (c.T @ c)
        d = la.norm(u_new - u) / la.norm(u_new)
        u = u_new
        i += 1

    return w, c, t, u


class PLS(
    _PLS
):
    """
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

    def __init__(
        self, n_components=2, *, scale=True, center=True, flip=False, max_iter=500, tol=1e-06, copy=True, deflation_mode="regression",
    ):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode=deflation_mode,
            mode="A",
            algorithm="mynipals",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )
        self.flip = flip
        self.center = center

    def __str__(self):
        return f"{type(self).__name__}(n_components={self.n_components})"

    def __repr__(self):
        return f"{type(self).__name__}(n_components={self.n_components}, deflation_mode={self.deflation_mode})"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PLS":
        """
        Fit PLS model.
        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by xd variables.
        y: np.ndarray
            Dependent matrix with size n samples by yd, or a vector. For now only t==1 is implemented.
        n_comp: int
            Number of components, default is None, which indicates that
            largest dimension which is smaller value between n and p
            will be used.
        Returns
        -------
        PLS object
        Reference
        ---------
        [1] Trygg J, Wold S. Orthogonal projection on Latent Structure (OPLS).
            J Chemometrics. 2002, 16, 119-128.
        [2] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable
            regression (LVR) method with a integral OSC filter.
            J Chemometrics. 2003, 17, 53-64.
        [3] https://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition
        """
        check_consistent_length(x, y)
        X = self._validate_data(
            x, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        Y = check_array(
            y, input_name="Y", dtype=np.float64, copy=self.copy, ensure_2d=False
        )
        if (Y.ndim == 1):
            Y = Y[:, np.newaxis]

        n, xd = X.shape
        _, yd = Y.shape

        if (self.deflation_mode == "regression"):
            npc = min(n, xd)
        elif (self.deflation_mode == "canonical"):
            npc = min(n, xd, yd)
        else:
            raise ValueError(
                f"deflation_mode=\"{self.deflation_mode}\" is not supported. Supported values are \"canonical\" and \"regression\"")

        if self.n_components > npc:
            raise ValueError(
                f"Number of components is too large for X=shape{X.shape}, Y=shape{Y.shape} and deflation_mode={self.deflation_mode}. Maximum value is {npc}.")

        if (self.n_components <= 0):
            raise ValueError(
                f"n_components should be positive nonzero integer, is {self.n_components}")

        n_comp = self.n_components

        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = center_scale_data(
            X, Y, center=self.center, scale=self.scale)
        self.intercept_ = self._y_mean

        #  Variable            | name       |    variable in sklearn user guide
        W = np.empty((xd, n_comp))  # X-weights  |     U
        C = np.empty((yd, n_comp))  # Y-weights  |     V
        T = np.empty((n,  n_comp))  # X-scores   |     Xi
        U = np.empty((n,  n_comp))  # Y-scores   |     Omega
        P = np.empty((xd, n_comp))  # X-loadings |     Gamma
        Q = np.empty((yd, n_comp))  # Y-loadings |     Delta

        #Y_eps = np.finfo(Y.dtype).eps

        for k in range(n_comp):
            # Replace columns that are all close to zero with zeros
            #Y_mask = np.all(np.abs(Y) < 10 * Y_eps, axis=0)
            #Y[:, Y_mask] = 0.0

            # Run nipals to get first singular vectors
            w, c, t, u = nipals(X, Y, tol=self.tol, max_iter=self.max_iter)

            if (self.flip):
                # Flip for consistency across solvers
                _svd_flip_1d(w, c)
                # recalculate scores after flip
                t = X @ w
                u = Y @ c / (c.T @ c)

            # Regress p to minimize error in Xhat = t p^T
            p = (X.T @ t) / (t.T @ t)
            # deflation of X
            X -= t @ p.T

            if (self.deflation_mode == "canonical"):
                # Regress q to minimize error in Yhat = u q^T
                q = (Y.T @ u) / (u.T @ u)
                # deflate y
                Y -= u @ q.T
            elif (self.deflation_mode == "regression"):
                # In regression mode only x score (u) is used
                # Regress q to minimize error in Yhat = t q^T
                q = (Y.T @ t) / (t.T @ t)
                # deflate y
                Y -= t @ q.T

            W[:, k] = w.squeeze(axis=1)
            U[:, k] = u.squeeze(axis=1)
            C[:, k] = c.squeeze(axis=1)
            T[:, k] = t.squeeze(axis=1)
            P[:, k] = p.squeeze(axis=1)
            Q[:, k] = q.squeeze(axis=1)

        self._x_weights = W
        self._y_weights = C
        self._x_scores = T
        self._y_scores = U
        self._x_loadings = P
        self._y_loadings = Q

        self.x_rotations_ = W @ pinv(P.T @ W, check_finite=False)
        self.y_rotations_ = C @ pinv(Q.T @ C, check_finite=False)

        self._coef_ = self.x_rotations_ @ Q.T
        self._coef_ *= self._y_std
        self._coef_ = self._coef_.T

        #self.coef_ = self._coef_

        # "expose" all the weights, scores and loadings
        self.x_weights_ = self._x_weights
        self.y_weights_ = self._y_weights
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        self.x_loadings_ = self._x_loadings
        self.y_loadings_ = self._y_loadings

        return self

    def inverse_predict(self, Y, copy=True):
        """Predict samples of given targets.
        With univariate y, this is a great way to visualize the final regression model, as this is just a linear interpolation of the coefficient vector
        along the given y-coordinates. For example:

        >>> x_interp = pls.inverse_predict(np.linspace(y.min(), y.max(), 101))

        With multivariate y this can still be used to generate interpolated structures, but it becomes a more complex combination of the coefficient vectors.
        In such a case it might be more meaningful to manually interpolate along each of the n_components coefficient vector individually. 

        Parameters
        ----------
        Y : np.ndarray
            Array of shape(n_samples) or shape(n_samples, yd) targets. In the first case this is done as
            linear interpolatio along the coefficient vector. In the latter the pseudo inverse of the coeficient matrix is calculated.
            When yd=1 these two methods are equal (up to machine precision).

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        X_pred : np.ndarray
            shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.

        """
        check_is_fitted(self)
        Y = check_array(
            Y, input_name="Y", ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES
        )
        # Center the Y values. _coef_ already has
        Y -= self.intercept_

        if Y.ndim == 1:
            # This is technically equal to the below with univariate y, but doesn't require the pseudo inversing
            scaledcoef = self._coef_ / (self._coef_**2).sum()
            X_pred = Y[:, np.newaxis] * scaledcoef
        else:
            invcoef = pinv(self._coef_).T
            X_pred = Y @ invcoef

        X_pred *= self._x_std
        X_pred += self._x_mean
        return X_pred


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

    """

    def __init__(
        self, n_components=1, *, scale=True, center=True, flip=False, max_iter=500, tol=1e-06, copy=True, algorithm="OPLS", deflation_mode=None,
    ):
        if (deflation_mode is None):
            if (algorithm == "OPLS"):
                deflation_mode = "regression"
            elif (algorithm == "O2PLS"):
                deflation_mode = "canonical"

        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode=deflation_mode,
            mode="A",
            algorithm=algorithm,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )
        self.flip = flip
        self.center = center

    def __str__(self):
        return f"{type(self).__name__}(n_components={self.n_components})"

    def __repr__(self):
        return f"{type(self).__name__}(n_components={self.n_components}, algorithm={self.algorithm}, deflation_mode={self.deflation_mode})"

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OPLS":
        """
        Fit OPLS model.

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
        check_consistent_length(x, y)
        X = self._validate_data(
            x, dtype=np.float64, copy=self.copy, ensure_min_samples=2
        )
        Y = check_array(
            y, input_name="Y", dtype=np.float64, copy=self.copy, ensure_2d=False
        )
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

        #Y_eps = np.finfo(Y.dtype).eps

        for k in range(n_comp):
            # Replace columns that are all close to zero with zeros
            #Y_mask = np.all(np.abs(Y) < 10 * Y_eps, axis=0)
            #Y[:, Y_mask] = 0.0

            # Run nipals to get first singular vectors
            w, c, t, u = nipals(X, Y, tol=self.tol, max_iter=self.max_iter)

            if (self.flip):
                # Flip for consistency across solvers
                _svd_flip_1d(w, c)
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

            if (self.algorithm == "O2PLS"):
                # In O2PLS we also calculate y-loadings and deflate y
                if (self.deflation_mode == "canonical"):
                    # Regress q to minimize error in Yhat = u q^T
                    q = (Y.T @ u) / (u.T @ u)
                    # find orthogonal weights
                    c_ortho = q-((c.T @ q) / (c.T @ c))*c
                    c_ortho /= la.norm(c_ortho)

                    # with the orthogonal weights, calculate orthogonal scores and loadings
                    u_ortho = (Y @ c_ortho) / (c_ortho.T @ c_ortho)
                    q_ortho = (Y.T @ u_ortho) / (u_ortho.T @ u_ortho)

                    # deflate y
                    Y -= u_ortho @ q_ortho.T
                elif (self.deflation_mode == "regression"):
                    # In regression mode only x score (u) is used
                    # Regress q to minimize error in Yhat = t q^T
                    q = (Y.T @ t) / (t.T @ t)
                    # find orthogonal weights
                    c_ortho = q-((c.T @ q) / (c.T @ c))*c
                    c_ortho /= la.norm(c_ortho)

                    # with the orthogonal weights, calculate orthogonal scores and loadings
                    u_ortho = (Y @ c_ortho) / (c_ortho.T @ c_ortho)
                    q_ortho = (Y.T @ t_ortho) / (t_ortho.T @ t_ortho)
                    # deflate y
                    Y -= t_ortho @ q_ortho.T
            elif (self.algorithm == "OPLS"):
                if (self.deflation_mode == "canonical"):
                    # Regress q to minimize error in Yhat = u q^T
                    q = (Y.T @ u) / (u.T @ u)
                elif (self.deflation_mode == "regression"):
                    # In regression mode only x score (u) is used
                    # Regress q to minimize error in Yhat = t q^T
                    q = (Y.T @ t) / (t.T @ t)

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

        self._coef_ = self.x_rotations_ @ Q.T
        self._coef_ *= self._y_std
        self._coef_ = self._coef_.T

        #self.coef_ = self._coef_

        # "expose" all the weights, scores and loadings
        self.x_weights_ = self._x_weights
        self.y_weights_ = self._y_weights
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        self.x_loadings_ = self._x_loadings
        self.y_loadings_ = self._y_loadings

        return self

    def inverse_predict(self, Y, copy=True):
        """Predict samples of given targets.
        With univariate y, this is a great way to visualize the final regression model, as this is just a linear interpolation of the coefficient vector
        along the given y-coordinates. For example:

        >>> x_interp = pls.inverse_predict(np.linspace(y.min(), y.max(), 101))

        With multivariate y this can still be used to generate interpolated structures, but it becomes a more complex combination of the coefficient vectors.
        In such a case it might be more meaningful to manually interpolate along each of the n_components coefficient vector individually. 

        Parameters
        ----------
        Y : np.ndarray
            Array of shape(n_samples) or shape(n_samples, yd) targets. In the first case this is done as
            linear interpolatio along the coefficient vector. In the latter the pseudo inverse of the coeficient matrix is calculated.
            When yd=1 these two methods are equal (up to machine precision).

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        X_pred : np.ndarray
            shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.

        """
        check_is_fitted(self)
        Y = check_array(
            Y, input_name="Y", ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES
        )
        # Center the Y values. _coef_ already has
        Y -= self.intercept_

        if Y.ndim == 1:
            # This is technically equal to the below with univariate y, but doesn't require the pseudo inversing
            scaledcoef = self._coef_ / (self._coef_**2).sum()
            X_pred = Y[:, np.newaxis] * scaledcoef
        else:
            invcoef = pinv(self._coef_).T
            X_pred = Y @ invcoef

        X_pred *= self._x_std
        X_pred += self._x_mean
        return X_pred

    def transform(self, X: np.ndarray, Y: np.ndarray = None, copy=True):
        """Apply the dimension reduction to get the orthogonal scores.

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
        check_is_fitted(self)
        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Normalize
        X -= self._x_mean
        X /= self._x_std
        # calculate scores
        x_scores = X @ self._Rortho
        if Y is not None:
            if (self.algorithm == "OPLS"):
                return x_scores, Y
            Y = check_array(
                Y, input_name="Y", ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES
            )
            if Y.ndim == 1:
                Y = Y[:, np.newaxis]
            Y -= self._y_mean
            Y /= self._y_std
            y_scores = Y @ self._Sortho
            return x_scores, y_scores

        return x_scores

    def inverse_transform(self, X: np.ndarray, Y: np.ndarray = None):
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
        check_is_fitted(self)
        X = check_array(X, input_name="X", dtype=FLOAT_DTYPES)
        # calculate scores
        x_new = X @ self._Portho.T
        x_new *= self._x_std
        x_new += self._x_mean

        if Y is not None:
            if (self.algorithm == "OPLS"):
                return x_new, Y
            Y = check_array(Y, input_name="Y", dtype=FLOAT_DTYPES)
            y_new = Y @ self._Qortho.T
            y_new *= self._y_std
            y_new += self._y_mean
            return x_new, y_new

        return x_new

    def correct(self, X: np.ndarray, y: np.ndarray = None, copy: bool = True, return_ortho: bool = False) -> Union[
            Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Remove orthogonal components from X (and possibly y)

        Parameters
        ----------
        X: np.ndarray
            Data matrix with shape(n, c), where n is number of
            samples, and c is number of variables
        y: np.ndarray | None
            Data matrix shape(n) or shape(n, t), where n is the number of samples
            and t the number of features in y. If None, only X is corrected and returned.
            If the algorithm was OPLS, the corrected y is simply a copy of y.
        copy: bool
            Wether to work on and return copies of data. Default is True.
        return_ortho: bool
            Return orthogonal components of X (and possibly y). Default is False.

        Returns
        -------
        x_new: np.ndarray
            Corrected data, shape(n, c).
        x_ortho: np.ndarray
            Orthogonal score, shape(n, c). Returned if return _ortho=True.
        y_new: np.ndarray
            Corrected data, shape(n, t). Returned if y is not None.
        y_ortho: np.ndarray
            Orthogonal score, shape(n, t). Returned if y is not None and return _ortho=True.
        """
        check_is_fitted(self)
        x_new = self._validate_data(
            X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        x_new -= self._x_mean
        x_new /= self._x_std
        n, xd = x_new.shape

        if (xd != self._x_loadings.shape[0]):
            raise ValueError("Dimension mismatch in X, "
                             f"the model has been trained with nd={self._x_loadings.shape[0]},"
                             f" but X is shape{X.shape}")

        if (not y is None):
            check_consistent_length(X, y)
            y_new = Y = check_array(
                y, input_name="Y", dtype=FLOAT_DTYPES, copy=copy, ensure_2d=False
            )
            if (len(y_new.shape) == 1):
                y_new = y_new[:, np.newaxis]
            if (y_new.shape[1] != self._y_loadings.shape[0]):
                raise ValueError("Dimension mismatch in y, "
                                 f"the model has been trained with nf={self._x_loadings.shape[0]},"
                                 f" but y is shape{X.shape}")

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

    def score(self, X, y=None):
        return super().score(*self.correct(X, y))

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
        return f"{type(self).__name__}(n_components={self.n_components})"

    def __repr__(self):
        return f"{type(self).__name__}(n_components={self.n_components}, pls_components={self.pls_components}, algorithm={self.algorithm}, deflation_mode={self.deflation_mode})"

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
        if(Y is None):
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

    def predict(self, X: np.ndarray, copy=True) -> np.ndarray:
        check_is_fitted(self)
        X_new = self.correct(X, copy=copy)
        return self.pls_.predict(X_new, copy=copy)

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None) -> float:
        check_is_fitted(self)
        return self.pls_.score(*self.correct(X, y), sample_weight=sample_weight)
