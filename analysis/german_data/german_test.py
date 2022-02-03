import pandas as pd

import test_utils as tu

if __name__ == '__main__':
    data = pd.read_csv('german.csv')
    data.loc[data['credit']==0, 'credit'] = -1
    label = 'credit'
    positive_label = 1
    sensitive_features = ['sex', 'age']
    unpriv_group = {'sex': 0, 'age': 0}

    tu.test_binary(data, label, sensitive_features, unpriv_group, 'german_bias.csv', 'german_fair.csv', positive_label)
