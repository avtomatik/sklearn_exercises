#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 21:24:18 2023

@author: green-machine
"""
# =============================================================================
# Elastic Net
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Lasso

# =============================================================================
# Labor Capital Intensity
# =============================================================================
# =============================================================================
# Labor Productivity
# =============================================================================
X, y = get_data_frame().pipe(get_X_y)


solver = Lasso(normalize=1)
alphas = np.logspace(-5, 2, 1000)
alphas, coefs, _ = solver.path(X, y, alphas=alphas)
fig, ax = plt.subplots()
ax.plot(alphas, coefs.T)
ax.set_xscale('log')
ax.set_xlim(alphas.max(), alphas.min())
plt.legend()
plt.grid()
plt.show()
