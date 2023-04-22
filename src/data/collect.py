#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:14:17 2023

@author: alexandermikhailov
"""


import pandas as pd
from data.pull import pull_by_series_id
from data.read import read_usa_hist
from pandas import DataFrame


def stockpile_usa_hist(series_ids: dict[str, str]) -> DataFrame:
    """
    Parameters
    ----------
    series_ids : dict[str, str]
        DESCRIPTION.
    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        ...                ...
        df.iloc[:, -1]     Values
        ================== =================================
    """
    return pd.concat(
        map(
            lambda _: read_usa_hist(_[1]).sort_index().pipe(
                pull_by_series_id, _[0]
            ),
            series_ids.items()
        ),
        axis=1,
        sort=True
    )
