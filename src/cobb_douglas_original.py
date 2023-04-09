# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:32:51 2020

@author: Alexander Mikhailov
"""


import os

from data.combine import combine_cobb_douglas
from data.make_dataset import get_data_frame
from data.transform import transform_cobb_douglas

DIR_SRC = "../data/interim"
MAP_FIG = {
    'fg_a': 'Chart I Progress in Manufacturing {}$-${} ({}=100)',
    'fg_b': 'Chart II Theoretical and Actual Curves of Production {}$-${} ({}=100)',
    'fg_c': 'Chart III Percentage Deviations of $P$ and $P\'$ from Their Trend Lines\nTrend Lines=3 Year Moving Average',
    'fg_d': 'Chart IV Percentage Deviations of Computed from Actual Product {}$-${}',
    'fg_e': 'Chart V Relative Final Productivities of Labor and Capital',
    'year_base': 1899,
}

os.chdir(DIR_SRC)

plot_cobb_douglas(
    *combine_cobb_douglas().pipe(transform_cobb_douglas),
    MAP_FIG
)

df = get_data_frame().pipe(transform_cobb_douglas, year_base=1899)[0]
