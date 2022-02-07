import load_data as ld
import test_utils as tu
from fairclass import loss_funcs as lf

if __name__ == '__main__':
    data = ld.prepare_data()
    label = 'y'
    positive_label = 0
    protected_group = {'Gender': 1, 'Age': 1}
    sensitive_vars = ['Gender', 'Age']
    loss = lf._logistic_loss

    tu.test(data, label, sensitive_vars, protected_group, loss, 'obesity_bias.csv', 'obesity_fair.csv', positive_label)
