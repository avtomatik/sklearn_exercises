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
from sklearn.svm import SVC


def custom_cv_2folds(X: np.ndarray) -> tuple[np.ndarray]:
    """
    http://scikit-learn.org/stable/modules/cross_validation.html

    Parameters
    ----------
    X : np.ndarray
        DESCRIPTION.

    Yields
    ------
    idx : TYPE
        DESCRIPTION.
    idx : TYPE
        DESCRIPTION.

    """
    n = X.shape[0]
    _ = 1
    while _ <= 2:
        idx = np.arange(n * (_ - 1) / 2, n * _ / 2, dtype=int)
        yield idx, idx
        _ += 1


def main() -> None:
    # =========================================================================
    # Make Dataset
    # =========================================================================
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.4, random_state=0
    )
    # =========================================================================
    # Support Vector Machine: Support Vector Classification
    # =========================================================================
    estimator = SVC(kernel='linear', C=1).fit(X_train, y_train)

    scores = cross_val_score(estimator, X, y, cv=5)
    print('Cross Validation: Base Scoring')
    print(f'Accuracy: {scores.mean():,.4f} (+/- {2 * scores.std():,.4f})')

    scores = cross_val_score(estimator, X, y, cv=5, scoring='f1_macro')
    print('Cross Validation: F1 Scoring')
    print(f'Accuracy: {scores.mean():,.4f} (+/- {2 * scores.std():,.4f})')

    cross_validator = ShuffleSplit(n_splits=5, test_size=.3, random_state=0)
    scores = cross_val_score(estimator, X, y, cv=cross_validator)
    print('Cross Validation: Shuffle Split')
    print(f'Accuracy: {scores.mean():,.4f} (+/- {2 * scores.std():,.4f})')

    cross_validator = custom_cv_2folds(X)
    scores = cross_val_score(estimator, X, y, cv=cross_validator)
    print('Cross Validation: Custom')
    print(f'Accuracy: {scores.mean():,.4f} (+/- {2 * scores.std():,.4f})')

    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    estimator = SVC(C=1).fit(X_train_transformed, y_train)
    X_test_transformed = scaler.transform(X_test)
    scores = estimator.score(X_test_transformed, y_test)
    print('Cross Validation: Standard Scaler')
    print(f'Accuracy: {scores.mean():,.4f} (+/- {2 * scores.std():,.4f})')

    estimator = make_pipeline(StandardScaler(), SVC(C=1))
    cross_validator = ShuffleSplit(n_splits=5, test_size=.3, random_state=0)
    scores = cross_val_score(estimator, X, y, cv=cross_validator)
    print('Cross Validation: Composite Estimator')
    print(f'Accuracy: {scores.mean():,.4f} (+/- {2 * scores.std():,.4f})')


if __name__ == '__main__':
    main()
