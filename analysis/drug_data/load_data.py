import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


def prepare_data():
    data = pd.read_csv('drugs.csv')
    data.drop(['yhat', 'a'], axis=1, inplace=True)
    # data.loc[data['gender'] == 0.48246, 'gender'] = 1
    # data.loc[data['gender'] == -0.48246, 'gender'] = 0
    oe = OrdinalEncoder()
    data['gender'] = oe.fit_transform(np.reshape(data['gender'].values, (-1, 1)))
    data['y'].replace({
        'never': 0,
        'not last year': 1,
        'last year': 2}, inplace=True)
    data['race'].replace({
        'non-white': 0,
        'white': 1}, inplace=True)
    string_cols = data.dtypes[data.dtypes == 'object'].index.values
    data.drop(string_cols, axis=1, inplace=True)
    return data
