#!/usr/bin/env python3
import typing
from typing import Tuple, Union

import numpy as np
from numpy import linalg as la
from scipy.linalg import pinv
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition._pls import (_PLS, _center_scale_xy,
                                              _svd_flip_1d)
from sklearn.utils import check_array, check_consistent_length


def nipals(x: np.ndarray, y: np.ndarray,
           tol: float = 1e-10,
           max_iter: int = 10000) -> typing.Tuple:
    """
    Non-linear Iterative Partial Least Squares
    Parameters
    ----------
    x: np.ndarray
        Variable matrix with size n by d, where n number of samples,
        p number of variables.
    y: np.ndarray
        Dependent variable with size n by t. For now only t==1 is implemented.
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
        # TODO: remove
        #print(f"Iteration {i}: d={d}")
        u = u_new
        i += 1

    return w, c, t, u


class PLS(
    _PLS
):

    def __init__(
        self, n_components=2, *, scale=True, flip=False, max_iter=500, tol=1e-06, copy=True, deflation_mode="regression",
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

        npc = min(n, xd)
        if self.n_components is not None and self.n_components < npc:
            npc = self.n_components
        else:
            self.n_components = npc

        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, Y, scale=self.scale)
        self.intercept_ = self._y_mean

        #  Variable            | name       |    variable in sklearn user guide
        W = np.empty((xd, npc))  # X-weights  |     U
        C = np.empty((yd, npc))  # Y-weights  |     V
        T = np.empty((n, npc))  # X-scores   |     Xi
        U = np.empty((n, npc))  # Y-scores   |     Omega
        P = np.empty((xd, npc))  # X-loadings |     Gamma
        Q = np.empty((yd, npc))  # Y-loadings |     Delta

        Y_eps = np.finfo(Y.dtype).eps

        for k in range(npc):
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
                # Regress q to minimize error in Yhat = u p^T
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


class OPLS(
    _PLS
):

    def __init__(
        self, n_components=2, *, scale=True, flip=False, max_iter=500, tol=1e-06, copy=True, deflation_mode="regression",
    ):
        super().__init__(
            n_components=n_components,
            scale=scale,
            deflation_mode=deflation_mode,
            mode="A",
            algorithm="OPLS",
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )
        self.flip = flip

    def fit(self, x: np.ndarray, y: np.ndarray) -> "OPLS":
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

        npc = min(n, xd)
        if self.n_components is not None and self.n_components < npc:
            npc = self.n_components
        else:
            self.n_components = npc

        X, Y, self._x_mean, self._y_mean, self._x_std, self._y_std = _center_scale_xy(
            X, Y, scale=self.scale)
        self.intercept_ = self._y_mean

        #  Variable              | name       |    variable in sklearn user guide
        W = np.empty((xd, npc))  # X-weights  |     U
        C = np.empty((yd, npc))  # Y-weights  |     V
        T = np.empty((n, npc))   # X-scores   |     Xi
        U = np.empty((n, npc))   # Y-scores   |     Omega
        P = np.empty((xd, npc))  # X-loadings |     Gamma
        Q = np.empty((yd, npc))  # Y-loadings |     Delta
        # Orthogonal variables
        Wortho = np.empty((xd, npc))  # X-weights
        Tortho = np.empty((n, npc))   # X-scores
        Portho = np.empty((xd, npc))  # X-loadings
        if (self.algorithm == "O2PLS"):
            Cortho = np.empty((yd, npc))  # Y-weights  |     V
            Uortho = np.empty((n, npc))   # Y-scores   |     Omega
            Qortho = np.empty((yd, npc))  # Y-loadings |     Delta

        #Y_eps = np.finfo(Y.dtype).eps

        for k in range(npc):
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

            w_ortho = p-((w.T @ p) / (w.T @ w))*w
            w_ortho /= la.norm(w_ortho)

            t_ortho = (X @ w_ortho) / (w_ortho.T @ w_ortho)
            p_ortho = (X.T @ t_ortho) / (t_ortho.T @ t_ortho)

            # orthogonal deflation of X
            X -= t_ortho @ p_ortho.T

            if (self.algorithm == "O2PLS"):
                if (self.deflation_mode == "canonical"):
                    # Regress q to minimize error in Yhat = u q^T
                    q = (Y.T @ u) / (u.T @ u)
                    c_ortho = q-((c.T @ q) / (c.T @ c))*c
                    c_ortho /= la.norm(c_ortho)

                    u_ortho = (Y @ c_ortho) / (c_ortho.T @ c_ortho)
                    q_ortho = (Y.T @ u_ortho) / (u_ortho.T @ u_ortho)

                    # deflate y
                    Y -= u_ortho @ q_ortho.T
                elif (self.deflation_mode == "regression"):
                    # In regression mode only x score (u) is used
                    # Regress q to minimize error in Yhat = u p^T
                    q = (Y.T @ t) / (t.T @ t)
                    c_ortho = q-((c.T @ q) / (c.T @ c))*c
                    c_ortho /= la.norm(c_ortho)

                    u_ortho = (Y @ c_ortho) / (c_ortho.T @ c_ortho)
                    q_ortho = (Y.T @ t_ortho) / (t_ortho.T @ t_ortho)
                    # deflate y
                    Y -= t_ortho @ q_ortho.T
            else:
                if (self.deflation_mode == "canonical"):
                    # Regress q to minimize error in Yhat = u q^T
                    q = (Y.T @ u) / (u.T @ u)
                elif (self.deflation_mode == "regression"):
                    # In regression mode only x score (u) is used
                    # Regress q to minimize error in Yhat = u p^T
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
        if (self.algorithm == "O2PLS"):
            self._Uortho = Uortho
            self._Cortho = Cortho
            self._Qortho = Qortho

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

    def transform(self, X):
        return self.correct(X, return_ortho=False)

    def correct(self, X, y=None, return_ortho=False):
        x_new = X.copy()
        x_new -= self._x_mean
        x_new /= self._x_std
        n, xd = x_new.shape

        if (not y is None):
            y_new = y.copy()

        correct_y = (not y is None) and self.algorithm == "O2PLS"

        if (correct_y):
            y_new -= self._y_mean
            y_new /= self._y_std

        T_ortho_new = np.empty((n, self.n_components))
        for k in range(self.n_components):
            wk_ortho = self._Wortho[:, k:k+1]
            T_ortho_new[:, k:k+1] = (x_new @ wk_ortho) /   \
                                    (wk_ortho.T @ wk_ortho)

        x_ortho = T_ortho_new @ self._Portho.T

        x_new -= x_ortho
        x_new *= self._x_std
        x_new += self._x_mean

        if correct_y:
            U_ortho_new = np.empty((n, self.n_components))
            for k in range(self.n_components):
                ck_ortho = self._Cortho[:, k:k+1]
                U_ortho_new[:, k:k+1] = (y_new @ ck_ortho) /   \
                                        (ck_ortho.T @ ck_ortho)

            y_ortho = U_ortho_new @ self._Qortho.T

            y_new -= y_ortho

            y_new *= self._y_std
            y_new += self._y_mean

        if not return_ortho:
            if (not y is None):
                return x_new, y_new
            return x_new

        x_ortho *= self._x_std
        x_ortho += self._x_mean
        if (not y is None):
            if (correct_y):
                y_ortho *= self._y_std
                y_ortho += self._y_mean
            else:
                y_ortho = np.zeros_like(y)
            return x_new, x_ortho, y_new, y_ortho
        return x_new, x_ortho

    def score(self, X, y=None):
        return super().score(self.correct(X), y)

    @property
    def predictive_scores(self):
        """ Orthogonal loadings. """
        return self._x_scores

    @property
    def predictive_loadings(self):
        """ Predictive loadings. """
        return self._x_loadings

    @property
    def weights_y(self):
        """ y scores. """
        return self._y_weights

    @property
    def orthogonal_loadings(self):
        """ Orthogonal loadings. """
        return self._Portho

    @property
    def orthogonal_scores(self):
        """ Orthogonal scores. """
        return self._Tortho
