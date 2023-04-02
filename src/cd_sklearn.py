#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 13:45:36 2022

@author: green-machine
"""

import os
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def transform_cobb_douglas(df: DataFrame, year_base: int) -> tuple[DataFrame, tuple[float]]:
    """
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Capital
        df.iloc[:, 1]      Labor
        df.iloc[:, 2]      Product
        ================== =================================
    """
    df = df.div(df.loc[year_base, :])
    # =========================================================================
    # Labor Capital Intensity
    # =========================================================================
    df['lab_cap_int'] = df.iloc[:, 0].div(df.iloc[:, 1])
    # =========================================================================
    # Labor Productivity
    # =========================================================================
    df['lab_product'] = df.iloc[:, 2].div(df.iloc[:, 1])
    # =========================================================================
    # Original: k=0.25, b=1.01
    # =========================================================================
    k, b = np.polyfit(
        np.log(df.iloc[:, -2].astype(float)),
        np.log(df.iloc[:, -1].astype(float)),
        deg=1
    )
    # =========================================================================
    # Scipy Signal Median Filter, Non-Linear Low-Pass Filter
    # =========================================================================
    # =========================================================================
    # k, b = np.polyfit(
    #     np.log(signal.medfilt(df.iloc[:, -2])),
    #     np.log(signal.medfilt(df.iloc[:, -1])),
    #     deg=1
    # )
    # =========================================================================
    # =========================================================================
    # Description
    # =========================================================================
    df['cap_to_lab'] = df.iloc[:, 1].div(df.iloc[:, 0])
    # =========================================================================
    # Fixed Assets Turnover
    # =========================================================================
    df['c_turnover'] = df.iloc[:, 2].div(df.iloc[:, 0])
    # =========================================================================
    # Product Trend Line=3 Year Moving Average
    # =========================================================================
    df['prod_roll'] = df.iloc[:, 2].rolling(window=3, center=True).mean()
    df['prod_roll_sub'] = df.iloc[:, 2].sub(df.iloc[:, -1])
    # =========================================================================
    # Computed Product
    # =========================================================================
    df['prod_comp'] = df.iloc[:, 0].pow(k).mul(
        df.iloc[:, 1].pow(1-k)).mul(np.exp(b))
    # =========================================================================
    # Computed Product Trend Line=3 Year Moving Average
    # =========================================================================
    df['prod_comp_roll'] = df.iloc[:, -1].rolling(window=3, center=True).mean()
    df['prod_comp_roll_sub'] = df.iloc[:, -2].sub(df.iloc[:, -1])
    # =========================================================================
    #     print(f"R**2: {r2_score(df.iloc[:, 2], df.iloc[:, 3]):,.4f}")
    #     print(df.iloc[:, 3].div(df.iloc[:, 2]).sub(1).abs().mean())
    # =========================================================================
    return df, (k, np.exp(b))


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return


@cache
def get_data_frame(path_src: str = "/home/green-machine/data_science/data/interim") -> tuple[np.ndarray]:
    os.chdir(path_src)
    return stockpile_cobb_douglas()


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray]:
    df = df.pipe(
        transform_cobb_douglas,
        year_base=1899
    )[0].iloc[:, [3, 4]].applymap(np.log)
    return df.iloc[:, 0].values[:, np.newaxis], df.iloc[:, 1].values


if __name__ == '__main__':
    df = get_data_frame()
    X, y = get_data_frame().pipe(get_X_y)

    lr = LinearRegression()
    pr = LinearRegression()
    quadratic = PolynomialFeatures(degree=2)
    X_quad = quadratic.fit_transform(X)

    lr.fit(X, y)
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    y_lin_fit = lr.predict(X_fit)

    pr.fit(X_quad, y)
    y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

    plt.scatter(X, y, label="Trained")
    plt.plot(X_fit, y_lin_fit, label="Linear", linestyle="--")
    plt.plot(X_fit, y_quad_fit, label="Quadratic")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    df = get_data_frame()
    X, y = get_data_frame().pipe(get_X_y)

    regr = LinearRegression()

    quadratic = PolynomialFeatures(degree=2)
    cubic = PolynomialFeatures(degree=3)
    X_quad = quadratic.fit_transform(X)
    X_cubic = cubic.fit_transform(X)

    # =========================================================================
    # Linear Fit
    # =========================================================================
    X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
    regr = regr.fit(X, y)
    y_lin_fit = regr.predict(X_fit)
    linear_r2 = r2_score(y, regr.predict(X))

    # =========================================================================
    # Quadratic Fit
    # =========================================================================
    regr = regr.fit(X_quad, y)
    y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
    quadratic_r2 = r2_score(y, regr.predict(X_quad))

    # =========================================================================
    # Cubic Fit
    # =========================================================================
    regr = regr.fit(X_cubic, y)
    y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
    cubic_r2 = r2_score(y, regr.predict(X_cubic))

    # =========================================================================
    # Plot the Results
    # =========================================================================
    plt.scatter(X, y, label="Train", color="lightgray")
    plt.plot(
        X_fit, y_lin_fit,
        label=f"Linear (d=1), $R^2={linear_r2:,.4f}$",
        color="blue",
        lw=2,
        linestyle=":"
    )
    plt.plot(
        X_fit, y_quad_fit,
        label=f"Linear (d=2), $R^2={quadratic_r2:,.4f}$",
        color="red", lw=2, linestyle="-"
    )
    plt.plot(
        X_fit, y_cubic_fit,
        label=f"Linear (d=2), $R^2={cubic_r2:,.4f}$",
        color="green",
        lw=2,
        linestyle="--"
    )
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    X, y = get_X_y()

    tree = DecisionTreeRegressor(max_depth=1)
    tree.fit(X, y)
    sort_idx = X.flatten().argsort()

    lin_regplot(X[sort_idx], y[sort_idx], tree)
    plt.xlabel("Labor Capital Intensity")
    plt.ylabel("Labor Productivity")
    plt.grid()
    plt.show()


if __name__ == '__main__':
    df = get_data_frame()
    X, y = get_data_frame().pipe(get_X_y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.45, random_state=1
    )

    forest = RandomForestRegressor(
        n_estimators=1000, criterion="squared_error", random_state=1, n_jobs=-1
    )
    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    plt.scatter(
        y_train_pred, y_train_pred - y_train,
        c="black",
        marker="o",
        s=35,
        alpha=.5,
        label="Train"
    )
    plt.scatter(
        y_test_pred, y_test_pred - y_test,
        c="lightgreen",
        marker="s",
        s=35,
        alpha=.7,
        label="Test"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.legend(loc="upper right")
    plt.hlines(y=0, xmin=y.min(), xmax=y.max(), lw=2, color="red")
    plt.xlim([y.min(), y.max()])
    plt.grid()
    plt.show()
    print(
        f"MSE on Train Data: {mean_squared_error(y_train, y_train_pred):,.4f}")
    print(f"MSE on Test Data: {mean_squared_error(y_test, y_test_pred):,.4f}")
    print(f"R**2 on Train Data: {r2_score(y_train, y_train_pred):,.4f}")
    print(f"R**2 on Test Data: {r2_score(y_test, y_test_pred):,.4f}")


if __name__ == '__main__':
    df = get_data_frame()
    X, y = get_data_frame().pipe(get_X_y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.45, random_state=1
    )

    regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
    regr.fit(X_train, y_train)

    y_train_pred = regr.predict(X_train)
    y_test_pred = regr.predict(X_test)

    plt.scatter(
        y_train_pred, y_train_pred - y_train,
        c="black",
        marker="o",
        s=35,
        alpha=.5,
        label="Train"
    )
    plt.scatter(
        y_test_pred, y_test_pred - y_test,
        c="lightgreen",
        marker="s",
        s=35,
        alpha=.7,
        label="Test"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.legend(loc="upper right")
    plt.hlines(y=0, xmin=y.min(), xmax=y.max(), lw=2, color="red")
    plt.xlim([y.min(), y.max()])
    plt.grid()
    plt.show()
    print(
        f"MSE on Train Data: {mean_squared_error(y_train, y_train_pred):,.4f}")
    print(f"MSE on Test Data: {mean_squared_error(y_test, y_test_pred):,.4f}")
    print(f"R**2 on Train Data: {r2_score(y_train, y_train_pred):,.4f}")
    print(f"R**2 on Test Data: {r2_score(y_test, y_test_pred):,.4f}")


if __name__ == '__main__':
    df = get_data_frame()
    X, y = get_data_frame().pipe(get_X_y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=1
    )

    ransac = RANSACRegressor(LinearRegression(),
                             max_trials=100,
                             min_samples=10,
                             residual_threshold=.05,
                             random_state=0)
    ransac.fit(X, y)

    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    line_X = np.arange(X.min(), X.max(), 1)
    line_y_ransac = ransac.predict(line_X[:, np.newaxis])

    plt.scatter(X[inlier_mask], y[inlier_mask],
                c="blue", marker="o", label="Inliers")
    plt.scatter(X[outlier_mask], y[outlier_mask],
                c="lightgreen", marker="s", label="Outliers")
    plt.plot(line_X, line_y_ransac, color="red")
    plt.xlabel("Labor Capital Intensity")
    plt.ylabel("Labor Productivity")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
