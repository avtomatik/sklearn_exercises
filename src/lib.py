#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:30:49 2023

@author: green-machine
"""


import os

import pandas as pd


def collect_excel_colums(path_src: str = "../../data/interim") -> list[tuple[str, tuple[str]]]:
    """
    Returns Excel Files' Columns

    Parameters
    ----------
    path_src : TYPE, optional
        DESCRIPTION. The default is "../../data/interim".

    Returns
    -------
    list[tuple[str, tuple[str]]]
        DESCRIPTION.

    """
    return [
        (file_name, tuple(pd.read_excel(file_name).columns))
        for file_name in os.listdir(path_src)
    ]
