import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairclass import loss_funcs as lf, utils as ut
from fairclass.one_vs_rest import OneVsRest

if __name__ == '__main__':
    data = pd.read_csv('cmc.data',
                       names=['wife_age', 'wife_edu', 'hus_edu', 'num_child', 'wife_religion', 'wife_work',
                              'hus_occ', 'living', 'media', 'contr_use'])
    label = 'contr_use'
    sensitive_features = ['wife_religion', 'wife_work']
    unpriv_group = {'wife_religion': 1, 'wife_work': 1}
    data[label] = data[label]-1
    loss = lf._logistic_loss

    print '##### BIASED CLASSIFIER'
    fairness_constr = 0
    acc_constr = 0
    ovr = OneVsRest(loss, fairness_constr, acc_constr, 0.5, sensitive_features)
    model, metrics = ut.cross_val(ovr, data, label, unpriv_group, 1)
    ut.print_metrics(metrics)

    print '\n##### FAIR CLASSIFIER'
    fairness_constr = 1
    acc_constr = 0
    ovr = OneVsRest(loss, fairness_constr, acc_constr, 0.5, sensitive_features)
    model, metrics = ut.cross_val(ovr, data, label, unpriv_group, 1)
    ut.print_metrics(metrics)

    # BIASED
    # fairness_constr = 1
    # acc_constr = 0
    #
    # x_train = d_train.drop(label, axis=1).values
    # y_train = d_train[label].values.ravel()
    # ovr.fit(x_train, y_train)
    #
    # x_test = d_test.drop(label, axis=1).values
    # pred = d_test.copy()
    # pred[label] = ovr.pred(x_test)
    # print('Accuracy: ', accuracy_score(d_test[label].values.ravel(), pred[label].values))
    # print('Disparate Impact: ', mt.disparate_impact(pred, group_condition=unpriv_group, label_name=label,
    #                                                  positive_label=1))
    # print('Statistical Parity: ', mt.statistical_parity(pred, unpriv_group, label, 1))

    # FAIRNESS CONSTR
    # ovr = OneVsRest(loss, 1, 0, 0.5)
    # ovr.train(d_train, label, sensitive_features)
    # x_test = d_test.drop(label, axis=1).values
    # pred = d_test.copy()
    # pred[label] = ovr.pred(x_test)
    # print('Accuracy: ', accuracy_score(d_test[label].values.ravel(), pred[label].values))
    # print('Disparate Impact: ', mt.disparate_impact(pred, group_condition=unpriv_group, label_name=label,
    #                                                 positive_label=1))
    # print('Statistical Parity: ', mt.statistical_parity(pred, unpriv_group, label, 1))
