#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:04:16 2023

@author: alexandermikhailov
"""


from functools import cache

import pandas as pd
from pandas import DataFrame


@cache
def read_usa_hist(filepath_or_buffer: str) -> DataFrame:
    """
    Retrieves Data from Enumerated Historical Datasets
    Parameters
    ----------
    filepath_or_buffer : str
    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Series IDs
        df.iloc[:, 1]      Values
        ================== =================================
    """
    MAP = {
        'dataset_douglas.zip': {'series_id': 4, 'period': 5, 'value': 6},
        'dataset_usa_brown.zip': {'series_id': 5, 'period': 6, 'value': 7},
        'dataset_usa_cobb-douglas.zip': {'series_id': 5, 'period': 6, 'value': 7},
        'dataset_usa_kendrick.zip': {'series_id': 4, 'period': 5, 'value': 6},
        'dataset_usa_mc_connell_brue.zip': {'series_id': 1, 'period': 2, 'value': 3},
        'dataset_uscb.zip': {'series_id': 9, 'period': 10, 'value': 11},
    }
    kwargs = {
        'filepath_or_buffer': filepath_or_buffer,
        'header': 0,
        'names': tuple(MAP.get(filepath_or_buffer).keys()),
        'index_col': 1,
        'skiprows': (0, 4)[filepath_or_buffer == 'dataset_usa_brown.zip'],
        'usecols': tuple(MAP.get(filepath_or_buffer).values()),
    }
    return pd.read_csv(**kwargs)
