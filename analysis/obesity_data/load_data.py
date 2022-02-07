import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_data():
    data = pd.read_csv('obesity.csv')
    data.drop(['NObeyesdad', 'weight_cat', 'yhat', 'a'], axis=1, inplace=True)
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'].values)
    data['Gender'] = data['Gender'].astype(np.int64)
    data['y'].replace({
        'Normal_Weight': 0,
        'Overweight_Level_I': 1,
        'Overweight_Level_II': 2,
        'Obesity_Type_I': 3,
        'Insufficient_Weight': 4
    }, inplace=True)
    data['family_history_with_overweight'] = le.fit_transform(data['family_history_with_overweight'].values)
    data['FAVC'] = le.fit_transform(data['FAVC'].values)
    data['CAEC'] = le.fit_transform(data['CAEC'].values)
    data['SMOKE'] = le.fit_transform(data['SMOKE'].values)
    data['SCC'] = le.fit_transform(data['SCC'].values)
    data['CALC'] = le.fit_transform(data['CALC'].values)
    data['MTRANS'] = le.fit_transform(data['MTRANS'].values)
    data.loc[data['Age'] < 22, 'Age'] = 0
    data.loc[data['Age'] >= 22, 'Age'] = 1
    return data
