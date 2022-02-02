from load_data import *
import test_utils as tu
from fairclass import loss_funcs as lf

if __name__ == '__main__':
    data = load_dataset()
    label = 'income'
    sensitive_features = ['race', 'sex']
    unpriv_group = {'sex': 0, 'race': 0}
    positive_label = 1
    loss = lf._logistic_loss
    tu.test(data, label, sensitive_features, unpriv_group, loss, 'adult_bias.csv', 'adult_fair.csv')
