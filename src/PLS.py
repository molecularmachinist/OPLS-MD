from typing import Optional, overload, Literal
import numpy as np
from scipy.linalg import pinv

from .base_PLS import _PLS
from .utils import center_scale_data, nipals, r2_score, flip_scores_by_absolute_value


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

    def __str__(self):
        return f"{type(self).__name__}(n_components={self.n_components})"

    def __repr__(self):
        return f"{type(self).__name__}(n_components={self.n_components}, deflation_mode={repr(self.deflation_mode)})"

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Fit PLS model.
        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by n_features variables.
        y: np.ndarray
            Dependent matrix with size n samples by n_targets, or a vector. For now only t==1 is implemented.
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
        X, Y = self.validate_input(x, y)
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
                flip_scores_by_absolute_value(w, c)
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
                # In regression mode only x score (t) is used
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

        self.fitted = True
        return self

    def score(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None, *,
              ndim: Optional[int] = None,
              copy: bool = True) -> float:
        """Predict targets of given samples.
        Parameters
        ----------
        X : array-like of shape(n_samples, n_features)
            Samples.
        y : array-like of shape shape (n_samples) or (n_samples, n_targets)
            True y-values
        ndim : int|None, default None
            Number of PLS dimension to use for the prediction. None uses all.
        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.
            If `None`, uses value of self.copy.
        Returns
        -------
        r2_score : float
            Returns r2 score of predictions.
        """
        y = self._validate_array(y, variable_name="y", ensure_2d=False)
        if (y.shape[0] != X.shape[0]):
            raise ValueError(
                f"Inconsistent lengths between X and y ({X.shape=}, {y.shape=})"
            )
        yd = 1 if y.ndim == 1 else y.shape[1]
        if (yd != self.y_loadings_.shape[0]):
            raise ValueError(
                f"Wrong number of features in y ({y.shape=}, should be ({X.shape[0], self.y_loadings_.shape[0]}))"
            )
        if (sample_weight is not None and sample_weight.shape != (y.shape[0],)):
            sample_weight = self._validate_array(sample_weight,
                                                 variable_name="sample_weight",
                                                 ensure_2d=False)
            raise ValueError(
                f"Inconsistent lengths between sample_weight and y ({sample_weight.shape=}, {y.shape=})"
            )
        y_pred = self.predict(X, ndim=ndim, copy=copy)
        return r2_score(y, y_pred, sample_weights=sample_weight)

    @overload
    def transform(self, X: np.ndarray, y: np.ndarray, *, copy: bool = True) -> tuple[np.ndarray, np.ndarray]:
        ...

    @overload
    def transform(self, X: np.ndarray, y: None = None, *, copy: bool = True) -> np.ndarray:
        ...

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, *, copy=True) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict latent space of given samples.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            X-samples.
        y : array-like of shape (n_samples) or (n_samples, n_targets), default=None
            Y-samples. Ignored if None.
        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.
            If `None`, uses value of self.copy.
        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_features)
            Returns predicted latent space of X.
        y_scores : ndarray of shape (n_samples, n_targets), only if y is not None
            Returns predicted latent space of y.
        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        self.check_is_fitted()
        if (y is not None):
            X, y = self.validate_input(X, y, copy=copy)
        else:
            X = self._validate_array(X, variable_name="X", copy=copy)
        X -= self._x_mean
        X /= self._x_std

        x_scores = X @ self.x_rotations_

        if (y is not None):
            y -= self._y_mean
            y /= self._y_std
            y_scores = y @ self.y_rotations_
            return x_scores, y_scores

        return x_scores

    @overload
    def inverse_transform(self, x_scores: np.ndarray, y_scores: None = None) -> np.ndarray:
        ...

    @overload
    def inverse_transform(self, x_scores: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ...

    def inverse_transform(self, x_scores: np.ndarray, y_scores: Optional[np.ndarray] = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict latent space of given samples.
        Parameters
        ----------
        x_scores : array-like of shape (n_samples, n_components)
            X-scores (latent space).
        y_scores : array-like of shape(n_samples, n_components), default=None
            y-scores (latent space). Ignored if None.
        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.
            If `None`, uses value of self.copy.
        Returns
        -------
        X_hat : ndarray of shape (n_samples, n_features)
            Returns predicted real space X.
        y_hat : ndarray of shape (n_samples, n_targets), only if y_scores is not None
            Returns predicted real space y.
        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        self.check_is_fitted()
        if (y_scores is not None):
            x_scores, y_scores = self.validate_input(x_scores,
                                                     y_scores,
                                                     copy=False)
        else:
            x_scores = self._validate_array(x_scores,
                                            variable_name="X",
                                            copy=False)

        X_hat = x_scores @ self.x_loadings_.T
        X_hat *= self._x_std
        X_hat += self._x_mean

        if (y_scores is not None):
            y_hat = y_scores @ self.y_loadings_.T
            y_hat *= self._y_std
            y_hat += self._y_mean
            return X_hat, y_hat

        return X_hat

    def predict(self, X: np.ndarray, *, ndim: Optional[int] = None, copy=True) -> np.ndarray:
        """Predict targets of given samples.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        ndim : int|None, default None
            Number of PLS dimension to use for the prediction. None uses all
        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.
            If `None`, uses value of self.copy.
        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_targets)
            Returns predicted values.
        Notes
        -----
        This call requires the estimation of a matrix of shape
        `(n_features, n_targets)`, which may be an issue in high dimensional
        space.
        """
        if (ndim is None):
            ndim = self.n_components

        self.check_is_fitted()

        if (ndim > self.n_components):
            raise ValueError(
                f"ndim is bigger than the number of components this object was trained with")

        X = self._validate_array(X, variable_name="X", copy=copy)
        # Normalize
        X -= self._x_mean
        X /= self._x_std

        Ypred = X @ self._all_coefs[ndim-1].T
        return Ypred + self.intercept_

    def inverse_predict(self, Y: np.ndarray, *, ndim: Optional[int] = None, copy=True) -> np.ndarray:
        """Predict samples of given targets.
        With univariate y, this is a great way to visualize the final regression model, as this is just a linear interpolation of the coefficient vector
        along the given y-coordinates. For example:

        >>> x_interp = pls.inverse_predict(np.linspace(y.min(), y.max(), 101))

        With multivariate y this can still be used to generate interpolated structures, but it becomes a more complex combination of the coefficient vectors.
        In such a case it might be more meaningful to manually interpolate along each of the n_components coefficient vector individually. 

        Parameters
        ----------
        Y : np.ndarray
            Array of shape(n_samples) or shape(n_samples, n_targets) targets. In the first case this is done as
            linear interpolation along the coefficient vector. In the latter the pseudo inverse of the coeficient matrix is calculated.
            When n_targets=1 these two methods are equal (up to machine precision).

        copy : bool, default=True
            Whether to copy `X` and `Y`, or perform in-place normalization.

        Returns
        -------
        X_pred : np.ndarray
            shape (n_samples, n_features)
            Returns predicted values.

        """
        self.check_is_fitted()

        if (ndim is None):
            ndim = self.n_components

        if (ndim > self.n_components):
            raise ValueError(
                f"ndim is bigger than the number of components this object was trained with")

        Y = self._validate_array(
            Y, variable_name="Y", ensure_2d=False, copy=copy
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
