import load_data as ld
from fairclass import loss_funcs as lf
import test_utils as tu

if __name__ == '__main__':
    data = ld.prepare_data()
    label = 'score_cut'
    sensitive_vars = ['age', 'sex']
    protected_group = {'age': 1, 'sex': 0}
    positive_label = 0
    loss = lf._logistic_loss
    tu.test(data, label, sensitive_vars, protected_group, loss, 'park_bias.csv', 'park_fair.csv',positive_label)
