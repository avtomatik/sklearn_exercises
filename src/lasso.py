#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 21:07:24 2023

@author: green-machine
"""


import numpy as np
from sklearn.linear_model import Lasso

solver = Lasso(alpha=.000001)
solver.fit([[0, 0], [1, 2], [2, 4]], [0, 2, 4])
print(solver.coef_)
print(solver.intercept_)
print(np.polyfit(range(3), [0, 2, 4], deg=1))
