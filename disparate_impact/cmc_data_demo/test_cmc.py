import pandas as pd

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

    fairness_constr = 0
    acc_constr = 0
    ovr = OneVsRest(loss, fairness_constr, acc_constr, 0.5, sensitive_features)
    model, metrics = ut.cross_val(ovr, data, label, unpriv_group, 1)
    mt_bias = pd.DataFrame(metrics)
    mt_bias.to_csv('cmc_bias.csv', index=0)

    fairness_constr = 1
    acc_constr = 0
    ovr = OneVsRest(loss, fairness_constr, acc_constr, 0.5, sensitive_features)
    model, metrics = ut.cross_val(ovr, data, label, unpriv_group, 1)
    mt_fair = pd.DataFrame(metrics)
    mt_fair.to_csv('cmc_fair.csv', index=0)
