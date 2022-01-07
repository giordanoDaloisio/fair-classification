import pandas as pd
import numpy as np


def prepare_data():
    """Load the data and apply some pre-processing steps"""

    data = pd.read_csv('crime_data_normalized_csv.csv', na_values='?', delimiter=';')
    data.drop(['state', 'county', 'community', 'communityname',
               'fold', 'OtherPerCap'], axis=1, inplace=True)
    na_cols = data.isna().any()[data.isna().any() == True].index
    data.drop(na_cols, axis=1, inplace=True)
    y_classes = np.quantile(data['ViolentCrimesPerPop'].values, [
        0, 0.2, 0.4, 0.6, 0.8, 1])
    i = 0
    data['ViolentCrimesClass'] = data['ViolentCrimesPerPop']
    for cl in y_classes:
        data.loc[data['ViolentCrimesClass'] <= cl, 'ViolentCrimesClass'] = i * 100
        i += 1
    data.drop('ViolentCrimesPerPop', axis=1, inplace=True)
    data['black_people'] = data['racepctblack'] > -0.45
    data['hisp_people'] = data['racePctHisp'] > -0.4
    data['black_people'] = data['black_people'].astype(int)
    data['hisp_people'] = data['hisp_people'].astype(int)
    data.drop('racepctblack', axis=1, inplace=True)
    data.drop('racePctHisp', axis=1, inplace=True)
    return data


df = prepare_data()
df.head()
