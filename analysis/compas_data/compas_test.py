import pandas as pd

import test_utils as tu

if __name__ == '__main__':
    label = 'two_year_recid'
    sensitive_vars = ['sex', 'race']
    protected_group = {'sex': 0, 'race': 0}
    positive_label = 1
    data = pd.read_csv('compas.csv')
    data.loc[data['two_year_recid'] == 0, 'two_year_recid'] = -1

    tu.test_binary(data, label, sensitive_vars, protected_group, 'compas_bias.csv', 'compas_fair.csv',
                   positive_label)
