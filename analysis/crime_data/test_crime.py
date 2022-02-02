import load_data as ld
from fairclass import loss_funcs as lf
import test_utils as t

if __name__ == '__main__':
    data = ld.prepare_data()
    label = 'ViolentCrimesClass'
    groups_condition = {'black_people': 1, 'hisp_people': 1}
    sensitive_features = ['black_people', 'hisp_people']
    positive_label = 1
    loss = lf._logistic_loss
    t.test(data, label, sensitive_features, groups_condition, loss, 'crime_bias.csv', 'crime_fair.csv', positive_label)
