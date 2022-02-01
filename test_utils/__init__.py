from fairclass.one_vs_rest import OneVsRest
from fairclass import utils as ut
import pandas as pd
import os

path = '.\\ris'


def test(data, label, sensitive_features, groups_condition, loss, bias_df_name, fair_df_name, gamma=0.5):
    # print '####### BIASED CLASSIFIER'
    # fairness_constr = 0
    # acc_constr = 0
    # ovr = OneVsRest(loss, fairness_constr, acc_constr, gamma, sensitive_features)
    # model, metrics = ut.cross_val(ovr, data, label, groups_condition, 1)
    # mt_bias = pd.DataFrame(metrics)
    # if not os.path.exists(path):
    #     os.mkdir(path)
    # mt_bias.to_csv(os.path.join(path, bias_df_name), index=0)
    # ut.print_metrics(metrics)

    print '\n######## FAIR CLASSIFIER'
    fairness_constr = 1
    acc_constr = 0
    ovr = OneVsRest(loss, fairness_constr, acc_constr, gamma, sensitive_features)
    model, metrics = ut.cross_val(ovr, data, label, groups_condition, 1)
    mt_fair = pd.DataFrame(metrics)
    mt_fair.to_csv(os.path.join(path, fair_df_name), index=0)
    ut.print_metrics(metrics)
