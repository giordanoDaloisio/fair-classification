import load_data as ld
import test_utils as tu
from fairclass import loss_funcs as lf

if __name__ == '__main__':
    data = ld.prepare_data()
    label = 'POLITICAL_VIEW'
    protected_group = {'GENDER': 0, 'RELIGION': 0}
    sensitive_variables = ['GENDER', 'RELIGION']
    positive_label = 3
    loss = lf._logistic_loss
    tu.test(data, label, sensitive_variables, protected_group, loss, 'trump_bias.csv', 'trump_fair.csv', positive_label)
