import load_data as ld
from fairclass import loss_funcs as lf
import test_utils as tu

if __name__ == '__main__':
    data = ld.load_data()
    label = 'quality'
    sensitive_variables = ['alcohol', 'type']
    protected_group = {'alcohol': 0, 'type': 1}
    positive_label = 2
    loss = lf._logistic_loss
    tu.test(data, label, sensitive_variables, protected_group, loss, 'law_bias.csv', 'law_fair.csv')
