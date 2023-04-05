# =============================================================================
# Cross-Validation Test on Cobb-Douglas Dataset
# =============================================================================


import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import (KFold, LeaveOneOut, LeavePOut,
                                     RepeatedKFold, ShuffleSplit,
                                     TimeSeriesSplit)

from data.make_dataset import get_data_frame, get_X_y

X, y = get_data_frame().pipe(get_X_y)
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

ss = ShuffleSplit(n_splits=2, test_size=0.25, random_state=0)
# =============================================================================
# Time Series Split
# =============================================================================
tscv = TimeSeriesSplit(n_splits=3)
plt.figure()
plt.scatter(X, y)
f1p = np.polyfit(X, y, deg=1)
k, b = f1p
Z = b+k*X
plt.plot(X, Z)
_ = 0
# for train,test in kf.split(X):
# for train,test in rkf.split(X):
# for train,test in loo.split(X):
# for train,test in lpo.split(X):
# for train,test in ss.split(X):
for train, test in tscv.split(X):
    _ += 1
    f1p = np.polyfit(X[train], y[train], deg=1)
    k, b = f1p
    Z = b+k*X
    plt.plot(X, Z, label='Test %02d' % _)
# b=np.exp(b)

plt.legend()
plt.grid()
plt.show()
