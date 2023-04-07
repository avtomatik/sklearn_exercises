#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 01:09:35 2023

@author: green-machine
"""


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import (ShuffleSplit, cross_val_score,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# =============================================================================
# Support Vector Machine
# =============================================================================
from sklearn.svm import SVC

# =============================================================================
# Cross Validation: Here
# =============================================================================
# =============================================================================
# http://scikit-learn.org/stable/modules/cross_validation.html
# =============================================================================

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=.4, random_state=0
)
# =============================================================================
# SVC: Support Vector Classification
# =============================================================================
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)

# =============================================================================
# Option 1
# =============================================================================
scores = cross_val_score(clf, iris.data, iris.target, cv=5)

# =============================================================================
# Option 2
# =============================================================================
scores = cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_macro')
print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), 2 * scores.std()))

# =============================================================================
# Option 3
# =============================================================================
print(iris.data.shape[0])
cv = ShuffleSplit(n_splits=5, test_size=.3, random_state=0)
result = cross_val_score(clf, iris.data, iris.target, cv=cv)

# =============================================================================
# Option 4
# =============================================================================


def custom_cv_2folds(X):
    n = X.shape[0]
    _ = 1
    while _ <= 2:
        idx = np.range(n * (_ - 1) / 2, n * _ / s, dtype=int)
        yield idx, idx
        _ += 1


custom_cv = custom_cv_2folds(iris.data)
cross_val_score(clf, iris.data, iris.target, cv=custom_cv)

# =============================================================================
# Option 5
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=.4, random_state=0
)
scaler = StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
result = clf.score(X_test_transformed, y_test)

# =============================================================================
# Option 6
# =============================================================================
cv = ShuffleSplit(n_splits=5, test_size=.3, random_state=0)
clf = make_pipeline(StandardScaler(), SVC(C=1))
result = cross_val_score(clf, iris.data, iris.target, cv=cv)
