# =============================================================================
# Cross-Validation on Cobb--Douglas Dataset
# =============================================================================


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
# cross_validator
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
# =============================================================================
# Random Permutations Cross-Validation a.k.a. Shuffle & Split
# =============================================================================

lpo = LeavePOut(p=2)

ss = ShuffleSplit(n_splits=2, test_size=.25, random_state=0)
# =============================================================================
# Time Series Split
# =============================================================================
tscv = TimeSeriesSplit(n_splits=3)

plt.figure(0)
plt.scatter(X, y)
polyfit_linear = np.polyfit(X.flatten(), y, deg=1)
y_pred = np.poly1d(polyfit_linear)(X)

# =============================================================================
# https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html
# =============================================================================
# =============================================================================
# https://numpy.org/doc/stable/reference/generated/numpy.polyval.html
# =============================================================================


plt.plot(X.flatten(), y_pred)
_b = np.exp(polyfit_linear[1])
print(_b)
plt.legend()
plt.grid()
plt.show()
for _num, cross_validator in enumerate((kf, rkf, loo, lpo, ss, tscv), start=1):
    plt.figure(_num)
    for _, (train, test) in enumerate(cross_validator.split(X), start=1):
        polyfit_linear = np.polyfit(X[train].flatten(), y[train], deg=1)
        y_train_pred = np.poly1d(polyfit_linear)(X[train])
        plt.plot(X[train], y_train_pred, label=f'Split {_:02}')
        _b = np.exp(polyfit_linear[1])
        print(_b)
    plt.legend()
    plt.grid()
    plt.show()

