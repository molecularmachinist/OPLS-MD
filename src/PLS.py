import numpy as np
from scipy.linalg import pinv
from sklearn.metrics import r2_score
from sklearn.cross_decomposition._pls import (_PLS, _svd_flip_1d)
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES

from .utils import center_scale_data, nipals


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
        return f"{type(self).__name__}(n_components={self.n_components}, deflation_mode={repr(self.deflation_mode)})"

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

        all_coefs = np.empty((n_comp, yd, xd))

        # Y_eps = np.finfo(Y.dtype).eps

        for k in range(n_comp):
            # Replace columns that are all close to zero with zeros
            # Y_mask = np.all(np.abs(Y) < 10 * Y_eps, axis=0)
            # Y[:, Y_mask] = 0.0

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

            _x_rot = W[:, :k+1] @ pinv(P[:, :k+1].T @
                                       W[:, :k+1], check_finite=False)
            _y_rot = C[:, :k+1] @ pinv(Q[:, :k+1].T @
                                       C[:, :k+1], check_finite=False)
            coef = _x_rot @ Q[:, :k+1].T
            coef *= self._y_std

            all_coefs[k] = coef.T

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

        self._all_coefs = all_coefs

        self.coef_ = self._coef_

        # "expose" all the weights, scores and loadings
        self.x_weights_ = self._x_weights
        self.y_weights_ = self._y_weights
        self.x_scores_ = self._x_scores
        self.y_scores_ = self._y_scores
        self.x_loadings_ = self._x_loadings
        self.y_loadings_ = self._y_loadings

        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None, ndim: int = None) -> float:
        y_pred = self.predict(X, ndim=ndim)
        return r2_score(y, y_pred, sample_weight=sample_weight)

    def predict(self, X: np.ndarray, ndim: int = None, copy=True) -> np.ndarray:
        """Predict targets of given samples.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        ndim : int|None, default None
            Number of PLS dimension to use for the prediction. None uses all
        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.
        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        if (ndim is None):
            ndim = self.n_components

        check_is_fitted(self)

        if (ndim > self.n_components):
            raise ValueError(
                f"ndim is bigger than the number of components this object was trained with")

        X = self._validate_data(X, copy=copy, dtype=FLOAT_DTYPES, reset=False)
        # Normalize
        X -= self._x_mean
        X /= self._x_std

        Ypred = X @ self._all_coefs[ndim-1].T
        return Ypred + self.intercept_

    def inverse_predict(self, Y: np.ndarray, ndim: int = None, copy=True) -> np.ndarray:
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

        if (ndim is None):
            ndim = self.n_components

        if (ndim > self.n_components):
            raise ValueError(
                f"ndim is bigger than the number of components this object was trained with")

        Y = check_array(
            Y, input_name="Y", ensure_2d=False, copy=copy, dtype=FLOAT_DTYPES
        )
        # Center the Y values. _coef_ already has
        Y -= self.intercept_

        if Y.ndim == 1:
            # This is technically equal to the below with univariate y, but doesn't require the pseudo inversing
            scaledcoef = self._all_coefs[ndim-1] / \
                (self._all_coefs[ndim-1]**2).sum()
            X_pred = Y[:, np.newaxis] * scaledcoef
        else:
            invcoef = pinv(self._all_coefs[ndim-1]).T
            X_pred = Y @ invcoef

        X_pred *= self._x_std
        X_pred += self._x_mean
        return X_pred
