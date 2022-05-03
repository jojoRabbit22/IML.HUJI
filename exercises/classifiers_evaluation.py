from math import atan2, pi

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.metrics import loss_functions
from utils import custom

pio.templates.default = "simple_white"


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    raise NotImplementedError()


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    path = "datasets/" # todo this wont run on schools computer!
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        pathA = path + f
        arr = np.load(pathA)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def myCallback(model: Perceptron, arr: np.ndarray, response):
            loss = model._loss(arr, response)
            losses.append(loss)

        perceptron = Perceptron(callback=myCallback).fit(arr[:, :-1], arr[:, -1])

        # Plot figure
        line = list(range(1, perceptron.max_iter_))
        line = np.array(line)
        go.Figure(data=[go.Scatter(x=line, y=losses, mode="lines", showlegend=False)],
                  layout=go.Layout(title=f"{n} data",
                                   xaxis=dict(title="normilsed loss"),
                                   yaxis=dict(title="number of iterations"),
                                   showlegend=False)).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    symbols = np.array(["circle", "x", "square"])
    path = "datasets/"  # todo this wont run on schools computer!
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        pathA = path + f
        arr = np.load(pathA)

        # Fit models and predict over training set
        lda = LDA()
        X = arr[:, :-1]
        y = arr[:, -1]
        lda.fit(X, y)
        y_predicted_by_LDA = lda.predict(X)
        y_predicted_by_LDA = np.array(y_predicted_by_LDA)

        gaussian_bayes = GaussianNaiveBayes()
        gaussian_bayes.fit(X, y)
        y_predicted_by_gaussian_bayes = gaussian_bayes.predict(X)
        y_predicted_by_gaussian_bayes = np.array(y_predicted_by_gaussian_bayes)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        line = list(range(0, len(lda.pi_)))
        line = np.array(line)
        y = y.astype(int)
        print(loss_functions.accuracy(y, y_predicted_by_gaussian_bayes))

        fig = make_subplots(rows=1, cols=2 , subplot_titles=(
            rf"$\textbf{{ LDA with accuracy{loss_functions.accuracy(y, y_predicted_by_LDA)} }}$",
            rf"$\textbf{{ naive gaussian bayes with accuracy{loss_functions.accuracy(y, y_predicted_by_gaussian_bayes)} }}$"))

        # left graph
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=y_predicted_by_LDA, symbol=symbols[y],
                                             colorscale=[custom[2], custom[4]])), row=1, col=1)
        # left elipses
        for i in range(3):
            fig.add_trace(get_ellipse(lda.mu_[i],lda.cov_), row=1, col=1)


        # left markers
        fig.add_trace(
            go.Scatter(x=lda.mu_.transpose()[0], y=lda.mu_.transpose()[1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x")), row=1, col=1)

        # right graph
        fig.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                 marker=dict(color=y_predicted_by_gaussian_bayes, symbol=symbols[y],
                                             colorscale=[custom[2], custom[4]])), row=1, col=2)
        # left elipses
        for i in range(3):
            fig.add_trace(get_ellipse(gaussian_bayes.mu_[i], np.diag(gaussian_bayes.vars_[i])), row=1, col=2)

        # right markers
        fig.add_trace(
            go.Scatter(x=gaussian_bayes.mu_.transpose()[0], y=gaussian_bayes.mu_.transpose()[1], mode="markers",
                       showlegend=False,
                       marker=dict(color="black", symbol="x")), row=1, col=2)

        fig.update_layout(
            title_text=rf"$\textbf{{(1) {f} Dataset}}$")
        fig.show()

        # figb = go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
        #                                   marker=dict(color=y_predicted_by_gaussian_bayes, symbol=symbols[y],
        #                                               colorscale=[custom[2], custom[4]])),
        #                        get_ellipse(gaussian_bayes.mu_[0], np.diag(gaussian_bayes.vars_[0])),
        #                        get_ellipse(gaussian_bayes.mu_[1], np.diag(gaussian_bayes.vars_[1])),
        #                        get_ellipse(gaussian_bayes.mu_[2], np.diag(gaussian_bayes.vars_[2]))],
        #                  layout=go.Layout(
        #                      title=rf"$\textbf{{(1) {f} Gassien Dataset with accuracy{loss_functions.accuracy(y, y_predicted_by_gaussian_bayes)} }}$"))
        # figb.add_trace(
        #     go.Scatter(x=gaussian_bayes.mu_.transpose()[0], y=gaussian_bayes.mu_.transpose()[1], mode="markers",
        #                showlegend=False,
        #                marker=dict(color="black", symbol="x"))).show()
        # fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
