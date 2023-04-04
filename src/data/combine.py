#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 22:02:35 2023

@author: alexandermikhailov
"""


from operator import itemgetter

from collect import stockpile_usa_hist
from pandas import DataFrame


def combine_cobb_douglas(series_number: int = 3) -> DataFrame:
    """
    Original Cobb--Douglas Data Collection Extension
    Parameters
    ----------
    series_number : int, optional
        DESCRIPTION. The default is 3.
    Returns
    -------
    DataFrame
        ================== =================================
        df.index           Period
        df.iloc[:, 0]      Capital
        df.iloc[:, 1]      Labor
        df.iloc[:, 2]      Product
        ================== =================================
    """
    SERIES_IDS_EXT = {
        # =====================================================================
        # Cobb C.W., Douglas P.H. Capital Series: Total Fixed Capital in 1880 dollars (4)
        # =====================================================================
        'CDT2S4': ('dataset_usa_cobb-douglas.zip', 'capital'),
        # =====================================================================
        # Cobb C.W., Douglas P.H. Labor Series: Average Number Employed (in thousands)
        # =====================================================================
        'CDT3S1': ('dataset_usa_cobb-douglas.zip', 'labor'),
        # =====================================================================
        # Bureau of the Census, 1949, Page 179, J14: Warren M. Persons, Index of Physical Production of Manufacturing
        # =====================================================================
        'J0014': ('dataset_uscb.zip', 'product'),
        # =====================================================================
        # Bureau of the Census, 1949, Page 179, J13: National Bureau of Economic Research Index of Physical Output, All Manufacturing Industries.
        # =====================================================================
        'J0013': ('dataset_uscb.zip', 'product_nber'),
        # =====================================================================
        # The Revised Index of Physical Production for All Manufacturing In the United States, 1899--1926
        # =====================================================================
        'DT24AS01': ('dataset_douglas.zip', 'product_rev'),
    }
    SERIES_IDS = dict(zip(
        SERIES_IDS_EXT, map(itemgetter(0), SERIES_IDS_EXT.values())
    ))
    df = stockpile_usa_hist(SERIES_IDS)
    df.columns = map(itemgetter(1), SERIES_IDS_EXT.values())
    return df.iloc[:, range(series_number)].dropna(axis=0)
