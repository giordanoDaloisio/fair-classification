from __future__ import division
from sklearn.metrics import accuracy_score, f1_score, zero_one_loss


def disparate_impact(data_pred, group_condition, label_name, positive_label):
    unpriv_group_prob, priv_group_prob = _compute_probs(data_pred, label_name, positive_label, group_condition, )
    return min(unpriv_group_prob / priv_group_prob,
               priv_group_prob / unpriv_group_prob) if unpriv_group_prob != 0 else \
        unpriv_group_prob / priv_group_prob


def statistical_parity(data_pred, group_condition, label_name, positive_label):
    unpriv_group_prob, priv_group_prob = _compute_probs(data_pred, label_name, positive_label, group_condition)
    return unpriv_group_prob - priv_group_prob


def zero_one_loss_diff(df_true, df_pred, group_condition, label_name, positive_label):
    unpriv_group_true, _, priv_group_true, _ = _get_groups(df_true, label_name, positive_label, group_condition)
    unpriv_group_pred, _, priv_group_pred, _ = _get_groups(df_pred, label_name, positive_label, group_condition)
    zo_loss_unpr = zero_one_loss(unpriv_group_true[label_name].values.ravel(), unpriv_group_pred[label_name]
                                 .values.ravel(), normalize=True)
    zo_loss_priv = zero_one_loss(priv_group_true[label_name].values.ravel(), priv_group_pred[label_name]
                                 .values.ravel(), normalize=True)
    return zo_loss_unpr - zo_loss_priv


def _get_groups(data, label_name, positive_label, group_condition):
    query = '&'.join([str(k) + '==' + str(v) for k, v in group_condition.items()])
    label_query = label_name + '==' + str(positive_label)
    unpriv_group = data.query(query)
    unpriv_group_pos = data.query(query + '&' + label_query)
    priv_group = data.query('~(' + query + ')')
    priv_group_pos = data.query('~(' + query + ')&' + label_query)
    return unpriv_group, unpriv_group_pos, priv_group, priv_group_pos


def _compute_probs(data_pred, label_name, positive_label, group_condition):
    unpriv_group, unpriv_group_pos, priv_group, priv_group_pos = _get_groups(data_pred, label_name, positive_label,
                                                                             group_condition)
    unpriv_group_prob = (len(unpriv_group_pos)
                         / len(unpriv_group))
    priv_group_prob = (len(priv_group_pos)
                       / len(priv_group))
    return unpriv_group_prob, priv_group_prob
