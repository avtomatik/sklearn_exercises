#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:11:07 2023

@author: alexandermikhailov
"""


import os
from functools import cache
from pathlib import Path

import numpy as np
from data.combine import combine_cobb_douglas
from data.transform import transform_cobb_douglas
from pandas import DataFrame


@cache
def get_data_frame(path_src: str = "data/interim") -> DataFrame:
    os.chdir(Path(__file__).parent.parent.parent.resolve().joinpath(path_src))
    return combine_cobb_douglas()


def get_data_frame_transformed(df: DataFrame) -> DataFrame:
    return df.pipe(
        transform_cobb_douglas,
        year_base=1899
    )[0]


def get_X_y(df: DataFrame) -> tuple[np.ndarray]:
    """


    Parameters
    ----------
    df : DataFrame
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
