import test_utils as tu
from load_data import *

if __name__ == '__main__':
    data = load_dataset()
    label = 'income'
    sensitive_features = ['race', 'sex']
    unpriv_group = {'sex': 0, 'race': 0}
    positive_label = 1
    tu.test_binary(data, label, sensitive_features, unpriv_group, 'adult_bias.csv', 'adult_fair.csv',
                   positive_label)
