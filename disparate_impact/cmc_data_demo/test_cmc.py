import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fairclass import loss_funcs as lf, metrics as mt
from fairclass.one_vs_rest import OneVsRest

if __name__ == '__main__':
    data = pd.read_csv('cmc.data',
                       names=['wife_age', 'wife_edu', 'hus_edu', 'num_child', 'wife_religion', 'wife_work',
                              'hus_occ', 'living', 'media', 'contr_use'])
    label = 'contr_use'
    sensitive_features = ['wife_religion', 'wife_work']
    unpriv_group = {'wife_religion': 1, 'wife_work': 1}
    data[label] = data[label]-1
    d_train, d_test = train_test_split(data, test_size=0.3, shuffle=True)
    loss = lf._logistic_loss
    x_control = {s: d_train[s] for s in sensitive_features}

    # BIASED
    fairness_constr = 0
    acc_constr = 0
    ovr = OneVsRest(loss, fairness_constr, acc_constr, 0.5, x_control, sensitive_features)
    x_train = d_train.drop(label, axis=1).values
    y_train = d_train[label].values.ravel()
    ovr.fit(x_train, y_train)

    x_test = d_test.drop(label, axis=1).values
    pred = d_test.copy()
    pred[label] = ovr.pred(x_test)
    print(pred)
    print('Accuracy: ', accuracy_score(d_test[label].values.ravel(), pred[label].values))
    # print('Disparate Impact: ', mt.disparate_impact(pred, group_condition=unpriv_group, label_name=label,
    #                                                 positive_label=1))
    print('Statistical Parity: ', mt.statistical_parity(pred, unpriv_group, label, 1))

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
