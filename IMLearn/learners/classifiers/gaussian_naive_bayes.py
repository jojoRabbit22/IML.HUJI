import math
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import loss_functions


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # start with calcilating mean Mu,k
        self.classes_, amount_in_clas = np.unique(y, return_counts=True)
        self.mu_ = []
        self.vars_ = []
        for clas in self.classes_:
            a = [X[y == clas].mean(axis=0)]
            self.mu_.append(a[0])
            b = [X[y == clas].var(axis=0)]
            self.vars_.append(b[0])

        self.vars_ = np.array(self.vars_)
        self.mu_ = np.array(self.mu_)
        self.pi_ = np.array([(i / y.shape[0]) for i in amount_in_clas])

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

        y_L = np.zeros((len(X), len(self.pi_)))
        mult = 1

        for j in range(len(X)):
            x_j = X[j]
            for k in range(len(self.pi_)):
                for d in range(len(X[0])):
                    var_k = self.vars_[k].reshape(1 , len(self.vars_[k]) )
                    var_k_transposed = var_k.reshape(len(self.vars_[k]) ,1)
                    vars_squared = var_k@var_k_transposed
                    x_sub_mu_j_squared = ((x_j - self.mu_[k]).reshape(1 , len(x_j)) @(x_j - self.mu_[k]).reshape(len(x_j) ,1))[0][0]

                    tempA = 1 / (vars_squared * math.sqrt(2 * math.pi))
                    tempB = math.exp(-0.5 * x_sub_mu_j_squared) / (vars_squared * self.pi_[k])
                    a = tempA * tempB
                    for i in range(len(X[0])):
                        b = 0
                        x_sub_mu_i_squared = \
                        ((X[i] - self.mu_[k]).reshape(1, len(x_j)) @ (X[i] - self.mu_[k]).reshape(len(x_j), 1))[0][0]

                        b += (1 / (vars_squared * math.sqrt(2 * math.pi))) * math.exp(
                            (-0.5 * x_sub_mu_i_squared) / vars_squared)
                    mult *= (a / b)
                y_L[j][k] = mult
                mult = 1
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
