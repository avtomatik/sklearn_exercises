#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:32:51 2020

@author: green-machine
"""


import matplotlib.pyplot as plt
import numpy as np
from data.make_dataset import get_data_frame, get_X_y
from sklearn.model_selection import (KFold, LeaveOneOut, LeavePOut,
                                     RepeatedKFold, ShuffleSplit,
                                     TimeSeriesSplit)

# =============================================================================
# Make Dataset
# =============================================================================
X, y = get_data_frame().pipe(get_X_y)

# =============================================================================
# Cross Validation
# =============================================================================
# =============================================================================
# K-Fold
# =============================================================================
kf = KFold(n_splits=4)
plt.figure()
plt.scatter(X, y)
for _, (train, test) in enumerate(kf.split(X), start=1):
    polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_train_pred = np.poly1d(polyfit_linear)(X[train])
    plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
    _b = np.exp(polyfit_linear[1])

# =============================================================================
# Repeated K-Fold
# =============================================================================
random_state = 12883823
rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
plt.figure()
plt.scatter(X, y)
for _, (train, test) in enumerate(rkf.split(X), start=1):
    polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_train_pred = np.poly1d(polyfit_linear)(X[train])
    plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
    _b = np.exp(polyfit_linear[1])

# =============================================================================
# Leave One Out (LOO)
# =============================================================================
loo = LeaveOneOut()
plt.figure()
plt.scatter(X, y)
for _, (train, test) in enumerate(loo.split(X), start=1):
    polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_train_pred = np.poly1d(polyfit_linear)(X[train])
    plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
    _b = np.exp(polyfit_linear[1])

# =============================================================================
# Leave P Out (LPO)
# =============================================================================
lpo = LeavePOut(p=2)
plt.figure()
plt.scatter(X, y)
for _, (train, test) in enumerate(lpo.split(X), start=1):
    polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_train_pred = np.poly1d(polyfit_linear)(X[train])
    plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
    _b = np.exp(polyfit_linear[1])

# =============================================================================
# Random Permutations Cross-Validation a.k.a. Shuffle & Split
# =============================================================================
ss = ShuffleSplit(n_splits=2, test_size=.25, random_state=0)
plt.figure()
plt.scatter(X, y)
for _, (train, test) in enumerate(ss.split(X), start=1):
    polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_train_pred = np.poly1d(polyfit_linear)(X[train])
    plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
    _b = np.exp(polyfit_linear[1])

# =============================================================================
# Time Series Split
# =============================================================================
tscv = TimeSeriesSplit(n_splits=3)
plt.figure()
plt.scatter(X, y)
for _, (train, test) in enumerate(tscv.split(X), start=1):
    polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
    y_train_pred = np.poly1d(polyfit_linear)(X[train])
    plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
    _b = np.exp(polyfit_linear[1])

polyfit_linear = np.polyfit(X.flatten(), y, deg=1)
y_pred = np.poly1d(polyfit_linear)(X)
plt.plot(X, y_pred, label='Test {:02d}'.format(0))
plt.legend()
plt.grid()
plt.show()
