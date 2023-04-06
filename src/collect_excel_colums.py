#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 21:30:49 2023

@author: green-machine
"""


import os

import pandas as pd

DIR_SRC = "../../data/interim"

columns = [
    (file_name, tuple(pd.read_excel(file_name).columns))
    for file_name in os.listdir(DIR_SRC)
]
