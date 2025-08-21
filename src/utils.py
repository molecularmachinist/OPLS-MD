import numpy as np
from numpy import linalg as la

from typing import Optional


def center_scale_data(X: np.ndarray, Y: np.ndarray, center=True, scale=True) -> tuple[
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
           max_iter: int = 10000) -> tuple[
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
    d: np.floating | float = tol * 10
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


def r2_score(y_true: np.ndarray, y_hat: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
    if (y_true.ndim == 1):
        y_true = y_true[:, None]
    if (y_hat.ndim == 1):
        y_hat = y_hat[:, None]
    if sample_weights is None:
        weights: float | np.ndarray = 1.0
    else:
        weights = sample_weights[:, None]

    numerator = (weights*(y_true - y_hat) ** 2).sum(axis=0, dtype=np.float64)
    y_mean = np.average(y_true, axis=0,
                        weights=sample_weights)
    denominator = (
        weights * (y_true - y_mean) ** 2
    ).sum(axis=0, dtype=np.float64)
    # Both zero: perfect prediction of zero variance
    both_zero = denominator == 0 * numerator == 0
    denominator[both_zero] = 1.0
    numerator[both_zero] = 1.0
    return np.mean(1-numerator/denominator)


def flip_scores_by_absolute_value(u: np.ndarray, v: np.ndarray):
    """Flips u and v such that the biggest absolute value in u will have a positive sign.

    Flipping makes the PLS model the same (up to machine precision) as with sklearn.

    The flipping is done in place.
    """
    biggest_abs_val_idx = np.argmax(np.abs(u))
    sign = np.sign(u[biggest_abs_val_idx])
    u *= sign
    v *= sign
