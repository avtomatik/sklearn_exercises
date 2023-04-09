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

# =============================================================================
# Make Dataset
# =============================================================================
X, y = get_data_frame().pipe(get_X_y)
print(X)

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
tscv = TimeSeriesSplit(n_splits=3)

plt.figure()
plt.scatter(X, y)
# =============================================================================
# for _, (train, test) in enumerate(kf.split(X), start=1):
#     polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_train_pred = np.poly1d(polyfit_linear)(X[train])
#     plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
#
# for _, (train, test) in enumerate(rkf.split(X), start=1):
#     polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_train_pred = np.poly1d(polyfit_linear)(X[train])
#     plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
#
# for _, (train, test) in enumerate(loo.split(X), start=1):
#     polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_train_pred = np.poly1d(polyfit_linear)(X[train])
#     plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
#
# for _, (train, test) in enumerate(lpo.split(X), start=1):
#     polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_train_pred = np.poly1d(polyfit_linear)(X[train])
#     plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
#
# for _, (train, test) in enumerate(ss.split(X), start=1):
#     polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
#     y_train_pred = np.poly1d(polyfit_linear)(X[train])
#     plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
# =============================================================================

for _, (train, test) in enumerate(tscv.split(X), start=1):
    polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_train_pred = np.poly1d(polyfit_linear)(X[train])
    plt.plot(X[train], y_train_pred, label=f'Split {_:02}')

    # =========================================================================
    # _b = np.exp(polyfit_linear[1])
    # =========================================================================

polyfit_linear = np.polyfit(X.flatten(), y, deg=1)
y_pred = np.poly1d(polyfit_linear)(X)
plt.plot(X, y_pred, label='Test {:02d}'.format(0))
plt.legend()
plt.grid()
plt.show()
