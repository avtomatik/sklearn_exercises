#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:11:07 2023

@author: alexandermikhailov
"""


import os
from functools import cache

import numpy as np
from collect import stockpile_cobb_douglas
from pandas import DataFrame
from transform import transform_cobb_douglas


@cache
def get_data_frame(path_src: str = "../../data/interim") -> DataFrame:
    os.chdir(path_src)
    return stockpile_cobb_douglas()


def get_X_y(df: DataFrame) -> tuple[np.ndarray]:
    df = df.pipe(
        transform_cobb_douglas,
        year_base=1899
    )[0].iloc[:, [3, 4]].applymap(np.log)
    return df.iloc[:, 0].values[:, np.newaxis], df.iloc[:, 1].values


print(get_data_frame())
