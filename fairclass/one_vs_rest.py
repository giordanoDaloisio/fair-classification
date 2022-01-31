import numpy as np
from fairclass import utils as u


class OneVsRest:
    def __init__(self, loss, fairness_constr, acc_constr, gamma):
        self.loss = loss
        self.class_weights = []
        self.fairness_constr = fairness_constr
        self.acc_constr = acc_constr
        self.gamma = gamma

    def train(self, df_train, label, sensitive_vars):
        for l in df_train[label].sort_values().unique():
            df_copy = df_train.copy()
            df_copy.loc[df_copy[label] != l, label] = -1
            x = df_copy.drop(label, axis=1).values
            y = df_copy[label].values.ravel()
            x = u.add_intercept(x)
            x_control = {s: df_copy[s].values for s in sensitive_vars}
            thresh = {s: 0 for s in sensitive_vars}
            w = u.train_model(x, y, x_control, self.loss, self.fairness_constr, self.acc_constr, 0, sensitive_vars,
                              thresh, self.gamma)
            self.class_weights.append(w)

    def pred(self, x_test):
        x_test = u.add_intercept(x_test)
        scores = []
        for w in self.class_weights:
            score = np.dot(x_test, w)
            scores.append(score)
        return np.argmax(scores, axis=0)
