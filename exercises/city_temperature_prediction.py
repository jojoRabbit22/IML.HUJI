import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 31, 31, 30, 31]
    df = pd.read_csv(string_path, parse_dates=[2])
    df = df.dropna(axis=0).drop_duplicates()
    df = df[df["Year"] < 2023]
    df = df[df["Year"] > 1900]
    df = df[df["Month"] <= 12]
    df = df[df["Month"] >= 1]
    df = df[df["Day"] <= 31]
    df = df[df["Day"] >= 1]
    df = df[df["Temp"] <= 60]
    df = df[df["Temp"] >= -40]
    df = df.reset_index()
    df["DayOfYear"] = df["Date"].apply(lambda x: x.day_of_year)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    string_path = "/Users/joelschreiber/Documents/uni/שנה ג/סימסטר ב/iml/IML.HUJI/datasets/City_Temperature.csv"
    df = load_data(string_path)

    # Question 2 - Exploring data for specific country
    df_israel = df[df["Country"] == "Israel"]

    df_israel["Year"] =df_israel["Year"].astype(str)
    fig = px.scatter(df_israel , x="DayOfYear", y="Temp",color="Year",
                     title=f"temperature via day of Year Graph ").show()

    months_std = df_israel.groupby('Month')["Temp"].agg(['std'])
    months_std = np.array(months_std).reshape(12)
    months_array = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                    "November", "December"]

    d = pd.DataFrame({'x': months_array, 'y': months_std})
    fog = px.bar(d, x="x", y="y", labels={'x': 'month', 'y': 'std of temps per month over years'}).show()

    # Question 3 - Exploring differences between countries
    df_counts_std = []
    df_counts_mean = []

    for all_countries in df["Country"].unique():
        d = (df[df["Country"] == all_countries]).groupby('Month')["Temp"].agg(['std'])
        df_counts_std.append(np.array(d).reshape(12))
        d = (df[df["Country"] == all_countries]).groupby('Month')["Temp"].agg(['mean'])
        df_counts_mean.append(np.array(d).reshape(12))

    all_std = np.array(df_counts_std).reshape(12 * 4)
    all_means = np.array(df_counts_mean).reshape(12 * 4)
    all_countries = ["" for x in range(12 * 4)]
    all_months = strs = ["" for x in range(12 * 4)]
    unique_countries = df["Country"].unique()
    for i in range(len(df_counts_std)):
        for j in range(12):
            all_countries[j + i * 12] = unique_countries[i]
            all_months[j + i * 12] = months_array[j]

    df_all_counts = pd.DataFrame({'x': all_months, 'y': all_means , 'color' : all_countries})
    fig = px.line(df_all_counts, x="x", y="y", color="color",
                     title=f"temperature via month of Year and county grapg with std bars" , error_y = all_std,
                     labels={"x": "months" , "y": "avg temperature"})
    fig.update_traces(marker=dict(size=20,
                                  line=dict(width=2,)),
                      selector=dict(mode='markers')).show()

    # Question 4 - Fitting model for different values of `k`


    # msk = np.random.rand(len(df)) < 0.75
    # msk.sort()
    # train = df[msk]
    # test = df[~msk]
    n = np.round(0.75 * len(df)).astype(int)
    train = df.sample(n)
    print(train)
    test = df.drop(train.index)
    print(test)
    X_train = train["DayOfYear"]
    y_train = train["Temp"]
    X_test = test["DayOfYear"]
    y_test = test["Temp"]
    losses_for_k_deg = [0 for i in range(10)]
    for i in range(10):
        model = PolynomialFitting(i)
        model.fit(X_train, y_train)
        losses_for_k_deg[i] = round(model.loss(X_test, y_test), 2)
    d = pd.DataFrame({'x': [i for i in range(10)], 'y': losses_for_k_deg})
    bars = px.bar(d, x="x", y="y", labels={'x': 'degree of poly', 'y': 'loss of model'},
                  title=f"lloss of model per degrre of poly").show()
    for i in range(10):
        print("the loss for degree", i, " is ", losses_for_k_deg[i])

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(4)
    model.fit(df_israel["DayOfYear"], df_israel["Temp"])
    country_losses = []
    for s in unique_countries:
        temp_df = df[df["Country"] == s]
        country_losses.append(model.loss(temp_df["DayOfYear"], temp_df["Temp"]))

    d = pd.DataFrame({'x': unique_countries, 'y': country_losses})
    bars = px.bar(d, x="x", y="y", labels={'x': 'country', 'y': 'error'}
                  ,title = f"loss of israel model for country").show()

