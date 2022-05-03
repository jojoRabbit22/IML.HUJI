import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # start with calcilating mean Mu,k
        k = int(np.amax(y)) +1  # number of diffrent values
        Nk = [np.count_nonzero(y == i) for i in range(k)]
        mu_MLE = [0 for i in range(k)]

        for i in range(k):
            for j in range(len(y)):
                if y[j] == i:
                    mu_MLE[i] += X[j]
            mu_MLE[i] = mu_MLE[i] * (1 / Nk[i])
        sigma_MLU = np.zeros((len(X[0]), len(X[0])))
        for i in range(len(y)):
            a = X[i] - mu_MLE[int(y[i])]
            b = a.reshape(1, len(a))
            a = a.reshape(len(a), 1)
            sigma_MLU += a @ b

        sigma_MLU = sigma_MLU * (1 / len(X))
        self.mu_ = np.array(mu_MLE)
        self.cov_ = sigma_MLU
        pi_k = [Nk[i] / len(y) for i in range(k)]
        self.pi_ = pi_k

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
        y_pred = [0 for i in X]
        y_L = self.likelihood(X)
        for j in range(len(y_L)):
            max = np.amax(y_L[j])
            for i in range(len(y_L[j])):
                if y_L[j][i] == max:
                    y_pred[j] = i
        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        a_k = [0 for i in self.mu_]
        b_k = [0 for i in self.mu_]

        sigma_inverse = inv(self.cov_)
        for i in range(len(a_k)):
            a_k[i] = sigma_inverse @ self.mu_[i]
            b_k[i] = math.log(self.pi_[i]) - 0.5 * (self.mu_[i] @ sigma_inverse @ self.mu_[i])

        y_L = np.zeros((len(X), len(self.pi_)))
        for j in range(len(X)):
            for i in range(len(self.pi_)):
                a = a_k[i].reshape(1, len(a_k[i]))
                b = X[j].reshape(len(X[j]),1)
                y_L[j][i] = a @ b + b_k[i]
        return y_L

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return loss_functions.misclassification_error(y, self._predict(X))
