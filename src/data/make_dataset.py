#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:11:07 2023

@author: alexandermikhailov
"""


import os
from functools import cache

import numpy as np
import pandas as pd
from data.combine import combine_cobb_douglas
from data.transform import transform_cobb_douglas
from src.config import DATA_DIR


@cache
def get_data_frame() -> pd.DataFrame:
    os.chdir(DATA_DIR)
    return combine_cobb_douglas()


def get_X_y(df: pd.DataFrame) -> tuple[np.ndarray]:
    """


    Parameters
    ----------
    df : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    TYPE
        Labor Capital Intensity.
    TYPE
        Labor Productivity.

    """
    df = df.pipe(
        transform_cobb_douglas,
        year_base=1899
    )[0].iloc[:, [3, 4]].applymap(np.log)
    return df.iloc[:, 0].values[:, np.newaxis], df.iloc[:, 1].values
