#!/usr/bin/env python3
import typing
from typing import Tuple, Union

import numpy as np
from numpy import linalg as la


def nipals(x: np.ndarray, y: np.ndarray,
           tol: float = 1e-10,
           max_iter: int = 10000,
           dot=np.dot) -> typing.Tuple:
    """
    Non-linear Iterative Partial Least Squares
    Parameters
    ----------
    x: np.ndarray
        Variable matrix with size n by p, where n number of samples,
        p number of variables.
    y: np.ndarray
        Dependent variable with size n by 1.
    tol: float
        Tolerance for the convergence.
    max_iter: int
        Maximal number of iterations.
    Returns
    -------
    w: np.ndarray
        Weights with size p by 1.
    u: np.ndarray
        Y-scores with size n by 1.
    c: np.ndarray
        Y-weight with size 1 by 1
    t: np.ndarray
        Scores with size n by 1
    References
    ----------
    [1] Wold S, et al. PLS-regression: a basic tool of chemometrics.
        Chemometr Intell Lab Sys 2001, 58, 109–130.
    [2] Bylesjo M, et al. Model Based Preprocessing and Background
        Elimination: OSC, OPLS, and O2PLS. in Comprehensive Chemometrics.
    """
    u = y
    i = 0
    d = tol * 10
    while d > tol and i <= max_iter:
        w = (x.T @ u) / (u.T @ u)
        w /= la.norm(w)
        t = x @ w
        c = t.T @ y / (t.T @ t)
        u_new = y @ c / (c.T @ c)
        d = la.norm(u_new - u) / la.norm(u_new)
        # TODO: remove
        print(f"Iteration {i}: d={d}")
        u = u_new
        i += 1

    return w, u, c, t
    
def center_scale_x_y(
        x: np.ndarray, y: np.ndarray, scale: bool=False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_mean = x.mean(axis=0)
    y_mean = y.mean(axis=0)
    x-=x_mean
    y-=y_mean
    if(scale):
        x_std = x.std(axis=0, ddof=1)
        y_std = y.std(axis=0, ddof=1)
        x /= x_std
        y /= y_std
    else:
        x_std = np.ones(x.shape[1])
        y_std = np.ones(y.shape[1])

    
    return x, y, x_mean, y_mean, x_std, y_std



class OPLS:
    """
    Orthogonal Projection on Latent Structure (O-PLS).
    Methods
    ----------
    predictive_scores: np.ndarray
        First predictive score.
    predictive_loadings: np.ndarray
        Predictive loadings.
    weights_y: np.ndarray
        y weights.
    orthogonal_loadings: np.ndarray
        Orthogonal loadings.
    orthogonal_scores: np.ndarray
        Orthogonal scores.
    """
    def __init__(self, n_components: int = None, copy: bool = True, scale: bool = True):
        """
        TODO:
            1. add arg for specifying the method for performing PLS
        """
        # Data centering and scaling
        self.x_mean: np.ndarray = None
        self.y_mean: np.ndarray = None
        self.x_std:  np.ndarray = None
        self.y_std:  np.ndarray = None
        
        # orthogonal score matrix
        self._Tortho: np.ndarray = None
        # orthogonal loadings
        self._Portho: np.ndarray = None
        # loadings
        self._Wortho: np.ndarray = None
        # covariate weights
        self._w: np.ndarray = None

        # predictive scores
        self._T: np.ndarray = None
        self._P: np.ndarray = None
        self._C: np.ndarray = None
        # coefficients
        self.coef: np.ndarray = None
        # total number of components
        self.npc: int = None

        # Parameters
        self.scale: bool = scale
        self.copy:  bool = copy
        self.n_components: int = n_components


    def fit(self, x: np.ndarray, y: np.ndarray) -> "OPLS":
        """
        Fit PLS model.
        Parameters
        ----------
        x: np.ndarray
            Variable matrix with size n samples by p variables.
        y: np.ndarray
            Dependent matrix with size n samples by 1, or a vector
        n_comp: int
            Number of components, default is None, which indicates that
            largest dimension which is smaller value between n and p
            will be used.
        Returns
        -------
        OPLS object
        Reference
        ---------
        [1] Trygg J, Wold S. Projection on Latent Structure (OPLS).
            J Chemometrics. 2002, 16, 119-128.
        [2] Trygg J, Wold S. O2-PLS, a two-block (X-Y) latent variable
            regression (LVR) method with a integral OSC filter.
            J Chemometrics. 2003, 17, 53-64.
        """
        if(self.copy):
            X = x.copy()
            Y = y.copy()
        else:
            X = x
            Y = y
        n, p = x.shape
        npc = min(n, p)
        n_comp = self.n_components
        if n_comp is not None and n_comp < npc:
            npc = n_comp
        
        if(Y.ndim==1):
            Y = Y[:,np.newaxis]
        if(Y.shape[1]!=1):
            raise NotImplementedError(f"Multivariate OPLS is not yet implemented, y should be shape(n,1), or shape(n), is shape{y.shape}")
        
        X, Y, self.x_mean, self.y_mean, self.x_std, self.y_std = center_scale_x_y(X, Y,scale=self.scale)

        # initialization
        Tortho = np.empty((n, npc))
        Portho = np.empty((p, npc))
        Wortho = np.empty((p, npc))
        T, P, C = np.empty((n, npc)), np.empty((p, npc)), np.empty(npc)

        # X-y variations
        tw = (X.T @ Y) / (Y.T @ Y)
        print(tw.shape)
        tw /= la.norm(tw)
        # predictive scores
        tp = X @ tw
        # initial component
        w, u, _, t = nipals(X, Y)
        p = (X.T @ t) / (t.T @ t)
        print(p.shape)
        for nc in range(npc):
            # orthoganol weights
            w_ortho = p - ((tw.T @ p) * tw)
            w_ortho /= la.norm(w_ortho)
            # orthoganol scores
            t_ortho = X @ w_ortho
            # orthoganol loadings
            p_ortho = (X.T @ t_ortho) / (t_ortho.T @ t_ortho)
            # update X to the residue matrix
            X -= t_ortho @ p_ortho.T # in pace change
            # save to matrix
            Tortho[:, nc] = t_ortho.squeeze()
            Portho[:, nc] = p_ortho.squeeze()
            Wortho[:, nc] = w_ortho.squeeze()
            # predictive scores
            tp -= t_ortho * (p_ortho.T @ tw)
            T[:, nc] = tp.squeeze()
            C[nc] = (y.T @ tp) / (tp.T @ tp)

            # next component
            w, u, _, t = nipals(X, Y)
            p = (X.T @ t) / (t.T @ t)
            P[:, nc] = p.squeeze()

        self._Tortho = Tortho
        self._Portho = Portho
        self._Wortho = Wortho
        # covariate weights
        self._w = tw.squeeze()

        # coefficients and predictive scores
        self._T = T
        self._P = P
        self._C = C
        self.coef = self._w * C[:, np.newaxis]

        self.npc = npc

        return self

    def predict(
            self, X, n_component=None, return_scores=False
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """ Predict the new coming data matrx. """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        coef = self.coef[n_component - 1]

        y = np.dot(X, coef)
        if return_scores:
            return y*self.y_std+self.y_mean, np.dot(X, self._w)

        return y*self.y_std+self.y_mean

    def correct(
            self, x, n_component=None, return_scores=False, dot=np.dot
    ) -> Union[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Correction of X
        Parameters
        ----------
        x: np.ndarray
            Data matrix with size n by c, where n is number of
            samples, and c is number of variables
        n_component: int | None
            Number of components. If is None, the number of components
            used in fitting the model is used. Default is None.
        return_scores: bool
            Return orthogonal scores. Default is False.
        Returns
        -------
        xc: np.ndarray
            Corrected data, with same matrix size with input X.
        t: np.ndarray
            Orthogonal score, n by n_component.
        """
        # TODO: Check X type and dimension consistencies between X and
        #       scores in model.
        xc = x.copy()
        if n_component is None:
            n_component = self.npc

        if xc.ndim == 1:
            t = np.empty(n_component)
            for nc in range(n_component):
                t_ = dot(xc, self._Wortho[:, nc])
                xc -= t_ * self._Portho[:, nc]
                t[nc] = t_
        else:
            n, c = xc.shape
            t = np.empty((n, n_component))
            # scores
            for nc in range(n_component):
                t_ = dot(xc, self._Wortho[:, nc])
                xc -= t_[:, np.newaxis] * self._Portho[:, nc]
                t[:, nc] = t_

        if return_scores:
            return xc*self.x_std+self.x_mean, t

        return xc*self.x_std+self.x_mean

    def predictive_score(self, n_component=None) -> np.ndarray:
        """
        Parameters
        ----------
        n_component: int
            The component number.
        Returns
        -------
        np.ndarray
            The first predictive score.
        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._T[:, n_component-1]

    def ortho_score(self, n_component=None) -> np.ndarray:
        """
        Parameters
        ----------
        n_component: int
            The component number.
        Returns
        -------
        np.ndarray
            The first orthogonal score.
        """
        if n_component is None or n_component > self.npc:
            n_component = self.npc
        return self._Tortho[:, n_component-1]

    @property
    def predictive_scores(self):
        """ Orthogonal loadings. """
        return self._T

    @property
    def predictive_loadings(self):
        """ Predictive loadings. """
        return self._P

    @property
    def weights_y(self):
        """ y scores. """
        return self._C

    @property
    def orthogonal_loadings(self):
        """ Orthogonal loadings. """
        return self._Portho

    @property
    def orthogonal_scores(self):
        """ Orthogonal scores. """
        return self._Tortho
