from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy import transpose, linalg

from ...metrics import mean_square_error


# todo check intersepts what to do in that case
class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """

        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        """
        P, D, Q = np.linalg.svd(X, full_matrices=False)
        for i in range(D.size):
            if D[i] != 0:
                D[i] = 1 / D[i]

        X_a = Q @ np.diag(D) @ P.transpose()
        self.coefs_ = X_a @ y
        """
        if self.include_intercept_:
            X = self.reshapeX(X)
        self.coefs_ = transpose(linalg.pinv(transpose(X))) @ y

    def reshapeX(self , X):
        array = np.ndarray(shape=(len(X), 1))
        for i in range(len(array)):
            array[i][0] = 1
        X = np.append(array, X, axis=1)
        return X

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            X = self.reshapeX(X)
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """

        y_pred = self.predict(X)
        return mean_square_error(y_pred, y)
