import load_data as ld
import test_utils as tu
from fairclass import loss_funcs as lf

if __name__ == '__main__':
    data = ld.prepare_data()
    protected_group = {'race': 1, 'gender': 1}
    label = 'gpa'
    sensitive_features = ['race', 'gender']
    positive_label = 2
    loss = lf._logistic_loss

    tu.test(data, label, sensitive_features, protected_group, loss, 'law_bias.csv', 'law_fair.csv')
