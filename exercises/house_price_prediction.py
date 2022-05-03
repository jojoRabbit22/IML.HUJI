from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import os
import warnings


warnings.filterwarnings("ignore")
pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(str)
    df = df.dropna(axis=0).drop_duplicates()
    df = df.drop("date", 1)
    df = df.drop("id", 1)
    df = df.drop("lat", 1)
    df = df.drop("long", 1)
    df = df[df["bedrooms"] > 0]
    df = df[df["bedrooms"] < 20]
    df = df[df["bathrooms"] >= 0]
    df = df[df["sqft_living"] > 0]
    df = df[df["sqft_lot"] < 1500000]
    df = df[df["sqft_lot15"] < 600000]
    df = df[df["sqft_lot"] >= 0]
    df = df[df["floors"] > 0]
    df = df[df["floors"] < 400]
    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["grade"].isin(range(1, 15)) &
            df["condition"].isin(range(1, 6))]

    df = df[df["floors"] < 400]
    df = pd.get_dummies(df, prefix='zipcode_dum', columns=['zipcode'])
    df["yr_built"] = (np.where(df["yr_renovated"] > df["yr_built"], df["yr_renovated"], df["yr_built"]))
    df = df.drop("yr_renovated", 1)
    df["yr_built"] = round(df["yr_built"] / 10)
    pd.set_option('display.max_columns', None)
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    X = df.drop("price", 1)
    y = df["price"]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in X:
        if not "zipcode" in i:
            path = output_path + i + ".png"
            cor = np.cov(X[i], y)[0, 1] / (np.std(X[i]) * np.std(y))
            fig = px.scatter(pd.DataFrame({'x': X[i], 'y': y}), x="x", y="y", trendline="ols",
                             title=f"scatter plot of Correlation of {i} Vals and Res <br>Pearson Correlation {cor}",
                             labels={"x": i + " Values", "y": "Response Values"}).write_image(path)


if __name__ == '__main__':
    np.random.seed(0)

    y_true = np.array([279000, 432000, 326000, 333000, 437400, 555950])
    y_pred = np.array([199000.37562541, 452589.25533196, 345267.48129011, 345856.57131275, 563867.1347574, 395102.94362135])
    print(mean_square_error(y_pred,y_true))
    # mod = LinearRegression()
    # line = np.linspace(0, 100, 101)
    # yy = np.linspace(0, 100, 101) * 5
    # yy = mod.fit_predict([line,line], yy)
    # print(mod.loss(line, yy))
    #
    # fig = px.scatter(pd.DataFrame({'x': line, 'y': yy}), x="x", y="y", trendline="ols",
    #                  title=f"persetage of data vs loss_mean over 10 sampls",
    #                  labels={"x": "%of data sampled", "y": "loss"}).show()

    # Question 1 - Load and preprocessing of housing prices dataset
    str = "/Users/joelschreiber/Documents/uni/שנה ג/סימסטר ב/iml/IML.HUJI/datasets/house_prices.csv"
    df = load_data(str)

    # Question 2 - Feature evaluation with respect to response
    X = df.drop("price", 1)
    y = df["price"]
    # feature_evaluation(X, y, 'graph_images/') #todo remove

    # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    model = LinearRegression()
    loss = []
    loss_means = []
    std = []
    persenteges = np.linspace(10, 100, 91)
    training_set = X_train
    training_set["price"] = y_train
    df_length = len(training_set["price"])
    for i in persenteges:
        loss = []
        n = round(i * df_length / 100)
        for j in range(0, 10):
            sample = training_set.sample(n)
            X_sampls = sample.drop("price", 1)
            y_samples = sample["price"]
            model.fit_predict(X_sampls, y_samples)
            loss.append(model.loss(X_test, y_test))
        std.append(np.std(loss, axis=0))
        meanLoss = np.mean(loss)
        loss_means.append(meanLoss)

    confidenceP = [0 for a in range(len(std))]
    confidenceN = [0 for a in range(len(std))]
    for j in range(len(std)):
        confidenceP[j] = loss_means[j] + 2 * std[j]
        confidenceN[j] = loss_means[j] - 2 * std[j]

    df_var = (pd.DataFrame({'x': persenteges, 'y': loss_means}))
    fig = px.scatter(pd.DataFrame({'x': persenteges, 'y': loss_means}), x="x", y="y",
                     title=f"persetage of data vs loss_mean over 10 sampls",
                     labels={"x": "%of data sampled", "y": "loss"})
    fig.add_trace(go.Scatter(x=persenteges, y=confidenceN, fill=None, mode="lines", line=dict(color="lightgrey"),showlegend=False))
    fig.add_trace(go.Scatter(x=persenteges, y=confidenceP, fill='tonexty', mode="lines", line=dict(color="lightgrey"), showlegend=False))
fig.show()
