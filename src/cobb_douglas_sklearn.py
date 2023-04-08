# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:32:51 2020

@author: Alexander Mikhailov
"""


import os

import matplotlib.pyplot as plt
import numpy as np
# =============================================================================
# Kolmogorov-Smirnov Test for Goodness of Fit
# =============================================================================
# from scipy.stats import kstest
from data.combine import combine_cobb_douglas
from data.make_dataset import (get_data_frame, get_data_frame_transformed,
                               get_X_y)
from data.transform import transform_cobb_douglas
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import (KFold, LeaveOneOut, LeavePOut,
                                     RepeatedKFold, ShuffleSplit,
                                     TimeSeriesSplit, cross_val_score)

DIR_SRC = "../data/interim"
MAP_FIG = {
    'fg_a': 'Chart I Progress in Manufacturing {}$-${} ({}=100)',
    'fg_b': 'Chart II Theoretical and Actual Curves of Production {}$-${} ({}=100)',
    'fg_c': 'Chart III Percentage Deviations of $P$ and $P\'$ from Their Trend Lines\nTrend Lines=3 Year Moving Average',
    'fg_d': 'Chart IV Percentage Deviations of Computed from Actual Product {}$-${}',
    'fg_e': 'Chart V Relative Final Productivities of Labor and Capital',
    'year_base': 1899,
}

os.chdir(DIR_SRC)

plot_cobb_douglas(
    *combine_cobb_douglas().pipe(transform_cobb_douglas),
    MAP_FIG
)


get_data_frame().pipe(get_data_frame_transformed)

X, y = get_data_frame().pipe(get_X_y)

# =============================================================================
# TODO: Discrete Laplace Transform
# =============================================================================


# =============================================================================
# Cross Validation
# =============================================================================
# =============================================================================
# K-Fold
# =============================================================================
kf = KFold(n_splits=4)

# =============================================================================
# Repeated K-Fold
# =============================================================================
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)

# =============================================================================
# Leave One Out (LOO)
# =============================================================================
loo = LeaveOneOut()

# =============================================================================
# Leave P Out (LPO)
# =============================================================================
lpo = LeavePOut(p=2)

# =============================================================================
# Random Permutations Cross-Validation a.k.a. Shuffle & Split
# =============================================================================
ss = ShuffleSplit(n_splits=2, test_size=.25, random_state=0)


# =============================================================================
# Time Series Split
# =============================================================================
X = np.column_stack(np.log(df.iloc[:, -2]))
tscv = TimeSeriesSplit(n_splits=3)
plt.figure()
plt.scatter(X, y)
# =============================================================================
# for _, (train, test) in enumerate(kf.split(X), start=1):
#     k, b = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_pred = np.add(np.multiply(X, k), b)
#     plt.plot(X, y_pred, label=f'Test {_:02d}')
#
# for _, (train, test) in enumerate(rkf.split(X), start=1):
#     k, b = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_pred = np.add(np.multiply(X, k), b)
#     plt.plot(X, y_pred, label=f'Test {_:02d}')
#
# for _, (train, test) in enumerate(loo.split(X), start=1):
#     k, b = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_pred = np.add(np.multiply(X, k), b)
#     plt.plot(X, y_pred, label=f'Test {_:02d}')
#
# for _, (train, test) in enumerate(lpo.split(X), start=1):
#     k, b = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_pred = np.add(np.multiply(X, k), b)
#     plt.plot(X, y_pred, label=f'Test {_:02d}')
#
# for _, (train, test) in enumerate(ss.split(X), start=1):
#     k, b = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_pred = np.add(np.multiply(X, k), b)
#     plt.plot(X, y_pred, label=f'Test {_:02d}')
# =============================================================================

for _, (train, test) in enumerate(tscv.split(X), start=1):
    k, b = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_pred = np.add(np.multiply(X, k), b)
    plt.plot(X, y_pred, label=f'Test {_:02d}')

    # =========================================================================
    # _b = np.exp(b)
    # =========================================================================

k, b = np.polyfit(X.flatten(), y, deg=1)
y_pred = np.add(np.multiply(X, k), b)
plt.plot(X, y_pred, label='Test {:02d}'.format(0))
plt.legend()
plt.grid()
plt.show()


# =============================================================================
# Cross Validation Alternative
# =============================================================================
# =============================================================================
# Required
# =============================================================================
X = np.transpose(np.atleast_2d(X))


# =============================================================================
# X = np.log(X)
# =============================================================================
y = np.log(y)
loo = LeaveOneOut(y.shape[0])
regr = LinearRegression()
scores = cross_val_score(
    regr, X, y, scoring='mean_squared_error', cv=loo,)
print(scores.mean())


lr = LinearRegression()
lr.fit(X, y)

r2 = r2_score(y, lr.predict(X))
# =============================================================================
# r2 = lr.score(X, y)
# =============================================================================
print('R2 (test data): {:.2}'.format(r2))

kf = KFold(len(X), n_folds=4)
# =============================================================================
# p = np.zeros_like(y)
# =============================================================================
for train, test in kf:
    lr.fit(X[train], y[train])
    p[test] = lr.predict(X[test])
    print(lr.predict(X))

plt.figure(1)
plt.scatter(X, y, label='Original')
plt.scatter(p, y, label='Linear Fit')
plt.title('Labor Capital Intensity & Labor Productivity, 1899--1922')
plt.xlabel('Labor Capital Intensity')
plt.ylabel('Labor Productivity')
plt.legend()
plt.grid()
plt.show()
