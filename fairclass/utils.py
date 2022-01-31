from collections import defaultdict
from copy import deepcopy
from multiprocessing import Process, Queue
from random import seed, shuffle

import numpy as np
from scipy.optimize import minimize  # for loss func minimization
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer
import metrics as mt

SEED = 1122334455
seed(SEED)  # set the random seed so that the random permutations can be reproduced again
np.random.seed(SEED)


def train_model(x, y, x_control, loss_function, apply_fairness_constraints, apply_accuracy_constraint, sep_constraint,
                sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma=None, alpha=None):
    """

    Function that trains the model subject to various fairness constraints.
    If no constraints are given, then simply trains an unaltered classifier.
    Example usage in: "synthetic_data_demo/decision_boundary_demo.py"

    ----

    Inputs:

    X: (n) x (d+1) numpy array -- n = number of examples, d = number of features, one feature is the intercept
    y: 1-d numpy array (n entries)
    x_control: dictionary of the type {"s": [...]}, key "s" is the sensitive feature name, and the value is a 1-d list with n elements holding the sensitive feature values
    loss_function: the loss function that we want to optimize -- for now we have implementation of logistic loss, but other functions like hinge loss can also be added
    apply_fairness_constraints: optimize accuracy subject to fairness constraint (0/1 values)
    apply_accuracy_constraint: optimize fairness subject to accuracy constraint (0/1 values)
    sep_constraint: apply the fine grained accuracy constraint
        for details, see Section 3.3 of arxiv.org/abs/1507.05259v3
        For examples on how to apply these constraints, see "synthetic_data_demo/decision_boundary_demo.py"
    Note: both apply_fairness_constraints and apply_accuracy_constraint cannot be 1 at the same time
    sensitive_attrs: ["s1", "s2", ...], list of sensitive features for which to apply fairness constraint, all of these sensitive features should have a corresponding array in x_control
    sensitive_attrs_to_cov_thresh: the covariance threshold that the classifier should achieve (this is only needed when apply_fairness_constraints=1, not needed for the other two constraints)
    gamma: controls the loss in accuracy we are willing to incur when using apply_accuracy_constraint and sep_constraint

    ----

    Outputs:

    w: the learned weight vector for the classifier

    """

    assert ((
                        apply_accuracy_constraint == 1 and apply_fairness_constraints == 1) == False)  # both constraints cannot be applied at the same time

    max_iter = 100000  # maximum number of iterations for the minimization algorithm

    if apply_fairness_constraints == 0:
        constraints = []
    else:
        constraints = get_constraint_list_cov(x, y, x_control, sensitive_attrs, sensitive_attrs_to_cov_thresh)

    if apply_accuracy_constraint == 0:  # its not the reverse problem, just train w with cross cov constraints

        if alpha:
            f_args = (x, LabelBinarizer().fit_transform(y), alpha, np.ones(x.shape[0]))
        else:
            f_args = (x, y)
        w = minimize(fun=loss_function,
                     x0=np.random.rand(x.shape[1], ),
                     args=f_args,
                     method='SLSQP',
                     options={"maxiter": max_iter},
                     constraints=constraints)

    else:

        if alpha:
            f_args = (x, LabelBinarizer().fit_transform(y), alpha, np.ones(x.shape[0]))
        else:
            f_args = (x, y)
        # train on just the loss function
        w = minimize(fun=loss_function,
                     x0=np.random.rand(x.shape[1], ),
                     args=f_args,
                     method='SLSQP',
                     options={"maxiter": max_iter},
                     constraints=[]
                     )

        old_w = deepcopy(w.x)

        def constraint_gamma_all(w, x, y, initial_loss_arr):

            gamma_arr = np.ones_like(y) * gamma  # set gamma for everyone
            new_loss = loss_function(w, x, y)
            old_loss = sum(initial_loss_arr)
            return ((1.0 + gamma) * old_loss) - new_loss

        def constraint_protected_people(w, x,
                                        y):  # dont confuse the protected here with the sensitive feature protected/non-protected values -- protected here means that these points should not be misclassified to negative class
            return np.dot(w, x.T)  # if this is positive, the constraint is satisfied

        def constraint_unprotected_people(w, ind, old_loss, x, y):

            new_loss = loss_function(w, np.array([x]), np.array(y))
            return ((1.0 + gamma) * old_loss) - new_loss

        constraints = []
        predicted_labels = np.sign(np.dot(w.x, x.T))
        unconstrained_loss_arr = loss_function(w.x, x, y, return_arr=True)

        if sep_constraint:  # separate gemma for different people
            for i in range(0, len(predicted_labels)):
                if predicted_labels[i] == 1.0 and x_control[sensitive_attrs[0]][
                    i] == 1.0:  # for now we are assuming just one sensitive attr for reverse constraint, later,
                    # extend the code to take into account multiple sensitive attrs
                    c = ({'type': 'ineq', 'fun': constraint_protected_people, 'args': (x[i], y[
                        i])})  # this constraint makes sure that these people stay in the positive class even in the
                    # modified classifier
                    constraints.append(c)
                else:
                    c = ({'type': 'ineq', 'fun': constraint_unprotected_people,
                          'args': (i, unconstrained_loss_arr[i], x[i], y[i])})
                    constraints.append(c)
        else:  # same gamma for everyone
            c = ({'type': 'ineq', 'fun': constraint_gamma_all, 'args': (x, y, unconstrained_loss_arr)})
            constraints.append(c)

        def cross_cov_abs_optm_func(weight_vec, x_in, x_control_in_arr):
            cross_cov = (x_control_in_arr - np.mean(x_control_in_arr)) * np.dot(weight_vec, x_in.T)
            return float(abs(sum(cross_cov))) / float(x_in.shape[0])

        w = minimize(fun=cross_cov_abs_optm_func,
                     x0=old_w,
                     args=(x, x_control[sensitive_attrs[0]]),
                     method='SLSQP',
                     options={"maxiter": 100000},
                     constraints=constraints
                     )

    try:
        assert (w.success == True)
    except:
        print "Optimization problem did not converge.. Check the solution returned by the optimizer."
        print "Returned solution is:"
        print w

    return w.x


def add_intercept(x):
    """ Add intercept to the data before linear classification """
    m, n = x.shape
    intercept = np.ones(m).reshape(m, 1)  # the constant b
    return np.concatenate((intercept, x), axis=1)


def check_binary(arr):
    "give an array of values, see if the values are only 0 and 1"
    s = sorted(set(arr))
    if s[0] == 0 and s[1] == 1:
        return True
    else:
        return False


def get_one_hot_encoding(in_arr):
    """
        input: 1-D arr with int vals -- if not int vals, will raise an error
        output: m (ndarray): one-hot encoded matrix
                d (dict): also returns a dictionary original_val -> column in encoded matrix
    """

    for k in in_arr:
        if str(type(k)) != "<type 'numpy.float64'>" and type(k) != int and type(k) != np.int64:
            print str(type(k))
            print "************* ERROR: Input arr does not have integer types"
            return None

    in_arr = np.array(in_arr, dtype=int)
    assert (len(in_arr.shape) == 1)  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    index_dict = {}  # value to the column number
    for i in range(0, len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []
    for i in range(0, len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict


def check_accuracy(model, x_train, y_train, x_test, y_test, y_train_predicted, y_test_predicted):
    """
    returns the train/test accuracy of the model
    we either pass the model (w)
    else we pass y_predicted
    """
    if model is not None and y_test_predicted is not None:
        print "Either the model (w) or the predicted labels should be None"
        raise Exception("Either the model (w) or the predicted labels should be None")

    if model is not None:
        y_test_predicted = np.sign(np.dot(x_test, model))
        y_train_predicted = np.sign(np.dot(x_train, model))

    def get_accuracy(y, Y_predicted):
        correct_answers = (Y_predicted == y).astype(int)  # will have 1 when the prediction and the actual label match
        accuracy = float(sum(correct_answers)) / float(len(correct_answers))
        return accuracy, sum(correct_answers)

    train_score, correct_answers_train = get_accuracy(y_train, y_train_predicted)
    test_score, correct_answers_test = get_accuracy(y_test, y_test_predicted)

    return train_score, test_score, correct_answers_train, correct_answers_test


def test_sensitive_attr_constraint_cov(model, x_arr, y_arr_dist_boundary, x_control, thresh, verbose):
    """
    The covariance is computed b/w the sensitive attr val and the distance from the boundary
    If the model is None, we assume that the y_arr_dist_boundary contains the distace from the decision boundary
    If the model is not None, we just compute a dot product or model and x_arr
    for the case of SVM, we pass the distace from bounday becase the intercept in internalized for the class
    and we have compute the distance using the project function

    this function will return -1 if the constraint specified by thresh parameter is not satifsified
    otherwise it will reutrn +1
    if the return value is >=0, then the constraint is satisfied
    """

    assert (x_arr.shape[0] == x_control.shape[0])
    if len(x_control.shape) > 1:  # make sure we just have one column in the array
        assert (x_control.shape[1] == 1)

    arr = []
    if model is None:
        arr = y_arr_dist_boundary  # simply the output labels
    else:
        arr = np.dot(model, x_arr.T)  # the product with the weight vector -- the sign of this is the output label

    arr = np.array(arr, dtype=np.float64)

    cov = np.dot(x_control - np.mean(x_control), arr) / float(len(x_control))

    ans = thresh - abs(
        cov)  # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    # ans = thresh - cov # will be <0 if the covariance is greater than thresh -- that is, the condition is not satisfied
    if verbose is True:
        print "Covariance is", cov
        print "Diff is:", ans
        print
    return ans


def get_correlations(model, x_test, y_predicted, x_control_test, sensitive_attrs):
    """
    returns the fraction in positive class for sensitive feature values
    """

    if model is not None:
        y_predicted = np.sign(np.dot(x_test, model))

    y_predicted = np.array(y_predicted)

    out_dict = {}
    for attr in sensitive_attrs:

        attr_val = []
        for v in x_control_test[attr]: attr_val.append(v)
        assert (len(attr_val) == len(y_predicted))

        total_per_val = defaultdict(int)
        attr_to_class_labels_dict = defaultdict(lambda: defaultdict(int))

        for i in range(0, len(y_predicted)):
            val = attr_val[i]
            label = y_predicted[i]

            # val = attr_val_int_mapping_dict_reversed[val] # change values from intgers to actual names
            total_per_val[val] += 1
            attr_to_class_labels_dict[val][label] += 1

        class_labels = set(y_predicted.tolist())

        local_dict_1 = {}
        for k1, v1 in attr_to_class_labels_dict.items():
            total_this_val = total_per_val[k1]

            local_dict_2 = {}
            for k2 in class_labels:  # the order should be the same for printing
                v2 = v1[k2]

                f = float(v2) * 100.0 / float(total_this_val)

                local_dict_2[k2] = f
            local_dict_1[k1] = local_dict_2
        out_dict[attr] = local_dict_1

    return out_dict


def get_constraint_list_cov(x_train, y_train, x_control_train, sensitive_attrs, sensitive_attrs_to_cov_thresh):
    """
    get the list of constraints to be fed to the minimizer
    """

    constraints = []

    for attr in sensitive_attrs:

        attr_arr = x_control_train[attr]
        attr_arr_transformed, index_dict = get_one_hot_encoding(attr_arr)

        if index_dict is None:  # binary attribute
            thresh = sensitive_attrs_to_cov_thresh[attr]
            c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                  'args': (x_train, y_train, attr_arr_transformed, thresh, False)})
            constraints.append(c)
        else:  # otherwise, its a categorical attribute, so we need to set the cov thresh for each value separately

            for attr_val, ind in index_dict.items():
                attr_name = attr_val
                thresh = sensitive_attrs_to_cov_thresh[attr][attr_name]

                t = attr_arr_transformed[:, ind]
                c = ({'type': 'ineq', 'fun': test_sensitive_attr_constraint_cov,
                      'args': (x_train, y_train, t, thresh, False)})
                constraints.append(c)

    return constraints


def _train_test_split(df_train, df_test, label):
    x_train = df_train.drop(label, axis=1).values
    y_train = df_train[label].values.ravel()
    x_test = df_test.drop(label, axis=1).values
    y_test = df_test[label].values.ravel()
    return x_train, x_test, y_train, y_test


def cross_val(classifier, data, label, groups_condition, positive_label, n_splits=10):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=2)
    metrics = {
        'stat_par': [],
        'zero_one_loss': [],
        'disp_imp': [],
        'acc': [],
        'f1': []
    }
    for train, test in fold.split(data):
        data = data.copy()
        df_train = data.iloc[train]
        df_test = data.iloc[test]
        model = deepcopy(classifier)
        run_metrics = _model_train(df_train, df_test, label, model, defaultdict(
            list), groups_condition, positive_label)
        for k in metrics.keys():
            metrics[k].append(run_metrics[k])
    return model, metrics


def _model_train(df_train, df_test, label, classifier, metrics, groups_condition, positive_label):
    x_train, x_test, y_train, y_test = _train_test_split(
        df_train, df_test, label)
    model = deepcopy(classifier)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    df_pred = df_test.copy()
    df_pred[label] = pred
    metrics['stat_par'].append(mt.statistical_parity(
        df_pred, groups_condition, label, positive_label))
    metrics['disp_imp'].append(mt.disparate_impact(
        df_pred, groups_condition, label, positive_label=positive_label))
    metrics['zero_one_loss'].append(mt.zero_one_loss_diff(df_test, df_pred, groups_condition, label, positive_label))
    metrics['acc'].append(mt.accuracy_score(y_test, pred))
    metrics['f1'].append(mt.f1_score(y_test, pred, average='weighted'))
    return metrics
