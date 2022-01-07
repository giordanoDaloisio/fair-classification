import pandas as pd
import numpy as np

from fairclass import utils as ut, loss_funcs as lf, metrics as m

data = pd.read_csv('cmc.data', names=['wife_age', 'wife_edu', 'hus_edu', 'num_child', 'wife_religion', 'wife_work',
                                      'hus_occ', 'living', 'media', 'contr_use'])

label = 'contr_use'
sensitive_features = ['wife_religion', 'wife_work']
unpriv_group = {'wife_religion': 1, 'wife_work': 1}
x = data.drop(label, axis=1).values
x = ut.add_intercept(x)
y = data[label].values.ravel()
x_control = {
    'wife_religion': data['wife_religion'].values.ravel(),
    'wife_work': data['wife_work'].values.ravel()
}
x_train, y_train, x_control_train, \
    x_test, y_test, x_control_test = ut.split_into_train_test(x, y, x_control, 0.7)

loss_function = lf._logistic_loss
apply_fairness_constraints = 1
apply_accuracy_constraint = 0
sep_constraint = 0
gamma = 0.5

w = ut.train_model(x_train, y_train, x_control_train, loss_function,
                   apply_fairness_constraints, apply_accuracy_constraint,
                   sep_constraint, sensitive_features,
                   {'wife_religion': 0, 'wife_work': 0}, gamma)

train_score, test_score, \
    correct_answers_train, correct_answers_test = ut.check_accuracy(
        w, x_train, y_train, x_test, y_test, None, None)
distances_boundary_test = (np.dot(x_test, w)).tolist()
all_class_labels_assigned_test = np.sign(distances_boundary_test)

pred = data.copy()
# pred[label] = all_class_labels_assigned_test

# correlation_dict_test = ut.get_correlations(None, None,
#                                             all_class_labels_assigned_test,
#                                             x_control_test,
#                                             sensitive_features)
# cov_dict_test = ut.print_covariance_sensitive_attrs(None, x_test,
#                                                     distances_boundary_test,
#                                                     x_control_test,
#                                                     sensitive_features)
# p_rule = ut.print_classifier_fairness_stats([test_score],
#                                             [correlation_dict_test],
#                                             [cov_dict_test],
#                                             sensitive_features[1])
# di = m.disparate_impact(pred, unpriv_group, label, 2)
# sp = m.statistical_parity(pred, unpriv_group, label, 2)
# print 'Disparate Impact: ', di
# print 'Statistical Parity: ', sp
