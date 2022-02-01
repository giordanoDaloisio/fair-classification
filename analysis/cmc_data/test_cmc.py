import pandas as pd

import test_utils as tu
from fairclass import loss_funcs as lf


if __name__ == '__main__':
    data = pd.read_csv('cmc.data',
                       names=['wife_age', 'wife_edu', 'hus_edu', 'num_child', 'wife_religion', 'wife_work',
                              'hus_occ', 'living', 'media', 'contr_use'])
    label = 'contr_use'
    sensitive_features = ['wife_religion', 'wife_work']
    unpriv_group = {'wife_religion': 1, 'wife_work': 1}
    data[label] = data[label]-1
    loss = lf._logistic_loss

    tu.test(data, label, sensitive_features, unpriv_group, loss, 'cmc_bias.csv', 'cmc_fair.csv')
