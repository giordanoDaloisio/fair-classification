import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_data():
    data = pd.read_csv('bar_pass_prediction.csv', index_col='Unnamed: 0')
    col_to_drop = ['ID', 'decile1b', 'decile3', 'decile1', 'cluster', 'bar1', 'bar2',
                   'sex', 'male', 'race1', 'race2', 'other', 'asian', 'black', 'hisp', 'bar', 'index6040', 'indxgrp',
                   'indxgrp2', 'dnn_bar_pass_prediction', 'grad', 'bar1_yr', 'bar2_yr', 'ugpa']
    data.drop(col_to_drop, axis=1, inplace=True)
    data.loc[data['Dropout'] == 'NO', 'Dropout'] = 0
    data.loc[data['Dropout'] == 'YES', 'Dropout'] = 1
    data.dropna(inplace=True)
    data.loc[data['gender'] == 'female', 'gender'] = 1
    data.loc[data['gender'] == 'male', 'gender'] = 0
    data.loc[data['race'] == 7.0, 'race'] = 0
    data.loc[data['race'] != 0, 'race'] = 1
    data['gpa'] = pd.qcut(data['gpa'], 3, labels=['a', 'b', 'c'])
    enc = LabelEncoder()
    data['gpa'] = enc.fit_transform(data['gpa'].values)
    return data
