# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:45:47 2020

@author: Alexander Mikhailov
"""


from pathlib import Path

import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier

DIR = "/home/green-machine/Downloads"
FILE_NAME = "trips_data.xlsx"


trips = pd.read_excel(Path(DIR).joinpath(FILE_NAME))
trips_processed = pd.get_dummies(
    trips,
    columns=[
        "city",
        "vacation_preference",
        "transport_preference",
    ]
)
classifier = GradientBoostingClassifier()
input_data = trips_processed.drop("target", axis=1)
output_data = trips_processed.target
classifier.fit(input_data, output_data)
print({col: 0 for col in trips_processed.columns})
