# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 20:45:47 2020

@author: Alexander Mikhailov
"""


import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from src.config import DATA_DIR


def main(
    file_name: str = 'trips_data.xlsx'
) -> None:
    trips = pd.read_excel(DATA_DIR.joinpath(file_name))
    trips_processed = pd.get_dummies(
        trips,
        columns=[
            'city',
            'vacation_preference',
            'transport_preference',
        ]
    )
    classifier = GradientBoostingClassifier()
    input_data = trips_processed.drop('target', axis=1)
    output_data = trips_processed.target
    classifier.fit(input_data, output_data)
    print({col: 0 for col in trips_processed.columns})


if __name__ == '__main__':
    main()
