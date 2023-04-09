#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:20:01 2023

@author: green-machine
"""


import matplotlib.pyplot as plt
import numpy as np
from data.make_dataset import get_data_frame, get_X_y
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score


def calculate_graph_k_folds_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 6
) -> None:
    solver = LinearRegression().fit(X, y)
    # =========================================================================
    # K-Folds cross-validator
    # =========================================================================
    kf = KFold(n_splits=n_splits)

    y_container = np.zeros_like(y)

    for idx_train, idx_test in kf.split(X, y):
        solver.fit(X[idx_train], y[idx_train])
        y_test_pred = solver.predict(X[idx_test])
        y_container[idx_test] = y_test_pred
        y_pred = solver.predict(X)

    plt.figure()
    plt.scatter(X, y, label='Original')
    plt.scatter(X, y_container, label='Linear Fit, K-Folds cross-validator')
    plt.scatter(X, y_pred, label='Linear Fit, Cumulative')
    plt.title('`Labor Productivity` over `Labor Capital Intensity`, 1899--1922')
    plt.xlabel('Labor Capital Intensity')
    plt.ylabel('Labor Productivity')
    plt.legend()
    plt.grid()
    plt.show()
    print('Figure Has Been Plotted')


def compare_r2s(X: np.ndarray, y: np.ndarray) -> None:
    solver = LinearRegression().fit(X, y)
    y_pred = solver.predict(X)

    r2_solver = solver.score(X, y)
    r2_metrics = r2_score(y, y_pred)

    print(
        f'R**2 Powered by sklearn.linear_model.LinearRegression: {r2_solver:.6}'
    )
    print(
        f'R**2 Powered by sklearn.metrics.r2_score: {r2_metrics:.6}'
    )


def get_neg_mean_squared_error_leave_one_out(X: np.ndarray, y: np.ndarray) -> None:
    """
    Cross Validation

    Returns
    -------
    None.

    """
    solver = LinearRegression().fit(X, y)

    loo = LeaveOneOut()

    scores = cross_val_score(
        solver, X, y, scoring='neg_mean_squared_error', cv=loo
    )
    print(f'Mean of `neg_mean_squared_error`: {scores.mean():,.6f}')


if __name__ == '__main__':
    # =========================================================================
    # Make Dataset
    # =========================================================================
    X, y = get_data_frame().pipe(get_X_y)

    calculate_graph_k_folds_linear_regression(X, y)
    compare_r2s(X, y)
    get_neg_mean_squared_error_leave_one_out(X, y)
