#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:32:51 2020

@author: green-machine
"""


from data.collect import stockpile_cobb_douglas

X, y = get_data_frame().pipe(get_X_y)

solver = LassoCV(cv=4, random_state=0).fit(X, y)
print(solver.score(X, y))
print(solver.predict(X[:1, ]))

solver = LinearRegression().fit(X, y)
solver.score(X, y)
print(solver.coef_)
print(solver.intercept_)

solver = Lasso(alpha=.000001)
solver.fit([[0, 0], [1, 2], [2, 4]], [0, 2, 4])
print(solver.coef_)
print(solver.intercept_)
print(np.polyfit(range(3), [0, 2, 4], deg=1))


# =============================================================================
# Cross Validation
# =============================================================================
# =============================================================================
# K-Fold
# =============================================================================

# kf = KFold(n_splits=4)
# =============================================================================
# Repeated K-Fold
# =============================================================================

# random_state = 12883823
# rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
# =============================================================================
# Leave One Out (LOO)
# =============================================================================

# loo = LeaveOneOut()
# =============================================================================
# Leave P Out (LPO)
# =============================================================================
# =============================================================================
# Random Permutations Cross-Validation a.k.a. Shuffle & Split
# =============================================================================

# lpo = LeavePOut(p=2)

# ss = ShuffleSplit(n_splits=2, test_size=0.25, random_state=0)
# =============================================================================
# Time Series Split
# =============================================================================

# tscv = TimeSeriesSplit(n_splits=3)
# plt.figure()
# plt.scatter(X, y)
# _ = 0
# for train, test in kf.split(X):
# for train, test in rkf.split(X):
# for train, test in loo.split(X):
# for train, test in lpo.split(X):
# for train, test in ss.split(X):
# for train, test in tscv.split(X):
#    _ += 1
#    f1p = np.polyfit(X[train], y[train], deg=1)
#    k, b = f1p
#    Z = b+k*X
#    plt.plot(X, Z, label='Test {:02d}'.format(_))
# b = np.exp(b)
# f1p = np.polyfit(X, y, deg=1)
# k, b = f1p
# Z = b+k*X
# plt.plot(X, Z, label='Test {:02d}'.format(0))
# plt.legend()
# plt.grid()
# plt.show()
# =============================================================================
# Cross Validation Alternative
# =============================================================================
# =============================================================================
#  Required
# =============================================================================
# X = np.transpose(np.atleast_2d(X))


# X = np.log(X)
# y = np.log(y)
# loo = LeaveOneOut(y.shape[0])
# regr = LinearRegression()
# scores = cross_val_score(regr, X, y, scoring='mean_squared_error', cv=loo, )
# print(scores.mean())

# lr = LinearRegression()
# lr.fit(X, y)

# =============================================================================
# r2 = lr.score(X, y)
# =============================================================================
# r2 = r2_score(y, lr.predict(X))
# print('R2 (test data): {:.2}'.format(r2))

# kf = KFold(len(X), n_folds=4)
# p = np.zeros_like(y)
# for train, test in kf:
# lr.fit(X[train], y[train])
# p[test] = lr.predict(X[test])
# print(lr.predict(X))
# plt.figure(1)
# plt.scatter(X, y, label='Original')
# plt.scatter(p, y, label='Linear Fit')
# plt.title('Labor Capital Intensity & Labor Productivity, 1899--1922')
# plt.xlabel('Labor Capital Intensity')
# plt.ylabel('Labor Productivity')
# plt.legend()
# plt.grid()
# plt.show()

# ================================================================================
# future_projects.py
# ================================================================================
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:32:51 2020

@author: Alexander Mikhailov
"""
plot_cobb_douglas_new_features(df)

X, y = get_data_frame().pipe(get_X_y)

solver = LassoCV(cv=4, random_state=0).fit(X, y)
print(solver.score(X, y))
print(solver.predict(X[:1, ]))

solver = LinearRegression().fit(X, y)
solver.score(X, y)
print(solver.coef_)
print(solver.intercept_)

solver = Lasso(alpha=.000001)
solver.fit([[0, 0], [1, 2], [2, 4]], [0, 2, 4])
print(solver.coef_)
print(solver.intercept_)
print(np.polyfit(range(3), [0, 2, 4], deg=1))


# # =============================================================================
# # Cross Validation
# # =============================================================================
# # =============================================================================
# # K-Fold
# # =============================================================================
# kf = KFold(n_splits=4)
# plt.figure()
# plt.scatter(X, y)
# _ = 0
# for train, test in kf.split(X):
#     _ += 1
# k, b = np.polyfit(X[train],
#                   y[train],
#                   1)
#     Z = b + k*X
#     plt.plot(X, Z, label='Test {:02d}'.format(_))
#     b = np.exp(b)

# # =============================================================================
# # Repeated K-Fold
# # =============================================================================
# random_state = 12883823
# rkf = RepeatedKFold(n_splits=2, n_repeats=2, random_state=random_state)
# plt.figure()
# plt.scatter(X, y)
# _ = 0
# for train, test in rkf.split(X):
#     _ += 1
# k, b = np.polyfit(X[train],
#                   y[train],
#                   1)
#     Z = b + k*X
#     plt.plot(X, Z, label='Test {:02d}'.format(_))
#     b = np.exp(b)

# # =============================================================================
# # Leave One Out (LOO)
# # =============================================================================
# loo = LeaveOneOut()
# plt.figure()
# plt.scatter(X, y)
# _ = 0
# for train, test in loo.split(X):
#     _ += 1
# k, b = np.polyfit(X[train],
#                   y[train],
#                   1)
#     Z = b + k*X
#     plt.plot(X, Z, label='Test {:02d}'.format(_))
#     b = np.exp(b)

# # =============================================================================
# # Leave P Out (LPO)
# # =============================================================================
# lpo = LeavePOut(p=2)
# plt.figure()
# plt.scatter(X, y)
# _ = 0
# for train, test in lpo.split(X):
#     _ += 1
# k, b = np.polyfit(X[train],
#                   y[train],
#                   1)
#     Z = b + k*X
#     plt.plot(X, Z, label='Test {:02d}'.format(_))
#     b = np.exp(b)

# # =============================================================================
# # Random Permutations Cross-Validation a.k.a. Shuffle & Split
# # =============================================================================
# ss = ShuffleSplit(n_splits=2, test_size=0.25, random_state=0)
# plt.figure()
# plt.scatter(X, y)
# _ = 0
# for train, test in ss.split(X):
#     _ += 1
# k, b = np.polyfit(X[train],
#                   y[train],
#                   1)
#     Z = b + k*X
#     plt.plot(X, Z, label='Test {:02d}'.format(_))
#     b = np.exp(b)

# # =============================================================================
# # Time Series Split
# # =============================================================================
# tscv = TimeSeriesSplit(n_splits=3)
# plt.figure()
# plt.scatter(X, y)
# _ = 0
# for train, test in tscv.split(X):
#     _ += 1
#     k, b = np.polyfit(X
#                       [train], y[train], 1)
#     Z = b + k*X
#     plt.plot(X, Z, label='Test {:02d}'.format(_))
#     b = np.exp(b)

# k, b = np.polyfit(X, y, deg=1)
# Z = b + k*X
# plt.plot(X, Z, label = 'Test {:02d}'.format(0))
# plt.legend()
# plt.grid()
# plt.show()
# =============================================================================
# Cross Validation Alternative
# =============================================================================
# # # X = np.transpose(np.atleast_2d(X)) # # Required


# # # # # X = np.log(X)
# # # y = np.log(y)
# # # loo = LeaveOneOut(y.shape[0])
# # # regr = LinearRegression()
# # # scores = cross_val_score(regr, X, y, scoring = 'mean_squared_error', cv=loo)
# # # print(scores.mean())
# # # # # lr = LinearRegression()
# # # # # lr.fit(X, y)

# # # # # r2 = r2_score(y, lr.predict(X)) # # r2 = lr.score(X, y)
# # # # # print('R2 (test data): {:.2}'.format(r2))

# # # # # kf = KFold(len(X), n_folds = 4)
# # # # # # # p = np.zeros_like(y)
# # # # # for train, test in kf:
# # # # #    lr.fit(X[train], y[train])
# # # # #    p[test] = lr.predict(X[test])
# # # print(lr.predict(X))
# # # # # plt.figure(1)
# # # # # plt.scatter(X, y, label = 'Original')
# # # # # plt.scatter(p, y, label = 'Linear Fit')
# # # # # plt.title('Labor Capital Intensity & Labor Productivity, 1899--1922')
# # # # # plt.xlabel('Labor Capital Intensity')
# # # # # plt.ylabel('Labor Productivity')
# # # # # # # plt.legend()
# # # # # plt.grid()
# # # # # # # plt.show()
# =============================================================================
# Cross Validation: Here
# =============================================================================
