def disparate_impact(data_pred, group_condition, label_name, positive_label):
    unpriv_group_prob, priv_group_prob = _compute_probs(data_pred, group_condition, label_name, positive_label)
    return min(unpriv_group_prob / priv_group_prob,
               priv_group_prob / unpriv_group_prob) if unpriv_group_prob != 0 else \
        unpriv_group_prob / priv_group_prob


def statistical_parity(data_pred, group_condition, label_name, positive_label):
    unpriv_group_prob, priv_group_prob = _compute_probs(data_pred, group_condition, label_name, positive_label)
    return unpriv_group_prob - priv_group_prob


def _compute_probs(data_pred, label_name, positive_label, group_condition):
    query = '&'.join(['{'+k+'}=={'+v+'}' for k, v in group_condition.items()])
    label_query = label_name + '==' + str(positive_label)
    unpriv_group_prob = (len(data_pred.query(query + '&' + label_query))
                         / len(data_pred.query(query)))
    priv_group_prob = (len(data_pred.query('~(' + query + ')&' + label_query))
                       / len(data_pred.query('~(' + query + ')')))
    return unpriv_group_prob, priv_group_prob
