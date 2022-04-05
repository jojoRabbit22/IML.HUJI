from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
from numpy.linalg import inv, det, slogdet
import numpy as np
import sys

sys.path.append('/Users/joelschreiber/Documents/uni/שנה ג/סימסטר ב/iml/IML.HUJI')
from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
import sys

sys.path.append("../")
from utils import *
from scipy.stats import norm

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu1 = [np.mean(np.random.normal(10, 1)) for _ in range(1000)]
    uni = UnivariateGaussian(False)
    uni.__init__(uni)
    uni.fit(mu1)
    print("(", uni.mu_, ",", uni.var_, ")")

    # Question 2 - Empirically showing sample mean is consistent
    uni2 = UnivariateGaussian(False)
    uni2.__init__(uni)
    uni2.fit(mu1)
    predictions = []
    for i in range(100):
        uni2.fit(mu1[:i * 10])
        predictions.append(np.abs(uni2.mu_ - 10))
    predictions = predictions[1:]
    ms = np.linspace(0, 1000, 100)
    # we can see the grapgh converging to zero as the sample size increases

    go.Figure([go.Scatter(x=ms, y=predictions, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{ Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=300)).show()


    # Question 3
    samples = np.sort(mu1)
    pdf_arr = uni.pdf(samples)

    # we expect a bell curve

    go.Figure([go.Scatter(x=samples, y=pdf_arr, mode='markers', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{ samples and their pdf scatter plot}$",
                               xaxis_title="sample value",
                               yaxis_title=" pdf of sample ",
                               height=700)).show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    sigma = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    muu = np.array([0, 0, 4, 0])

    muModel = (np.random.multivariate_normal(muu, sigma, 1000))
    uni = MultivariateGaussian()
    uni.fit(muModel)
    print(uni.mu_)
    print("-------")
    print(uni.cov_)

    # Question 5 - Likelihood evaluation
    line = np.linspace(-10, 10, 200)  # change back to 200
    line2 = np.linspace(-10, 10, 200)  # change back to 200
    s = np.random.multivariate_normal(muu, sigma, 1000)
    exp = [0, 0, 0, 0]
    logs = []
    logs = np.array([[uni.log_likelihood
                      (np.array([x, 0, y, 0]), sigma, s)
                      for x in line] for y in line2])

    go.Figure(data=go.Heatmap(x=line, y=line, z=logs,
                              colorscale='blues'),
              layout=go.Layout(title='Heatmap for log-likelihood of samples across line',
                               xaxis_title='x axis scale',
                               yaxis_title='y axis scale ')).show()

    # Question 6 - Maximum likelihood

    max_vals = np.unravel_index(np.argmax(logs),
                                np.shape(logs))
    max_val = logs[139][99]
    f1_max = line[max_vals[0]]
    f3_max = line2[max_vals[1]]
    print(max_val, f1_max, f3_max)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
