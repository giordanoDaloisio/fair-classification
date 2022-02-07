from load_data import *
import test_utils as tu
from fairclass import loss_funcs

if __name__ == '__main__':
    data = prepare_data()
    label = 'y'
    protected_group = {'race': 1, 'gender': 0}
    positive_label = 0
    sensitive_features = ['race', 'gender']
    loss = loss_funcs._logistic_loss
    tu.test(data, label, sensitive_features, protected_group, loss, 'drug_bias.csv', 'drug_fair.csv', positive_label)
