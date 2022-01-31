import numpy as np
from fairclass import utils as u


class OneVsRest:
    def __init__(self, loss, fairness_constr, acc_constr, gamma, sensitive_vars):
        self.loss = loss
        self.fairness_constr = fairness_constr
        self.acc_constr = acc_constr
        self.gamma = gamma
        self.sensitive_vars = sensitive_vars
        self.class_weights = []

    def fit(self, x, x_control, y):
        x = u.add_intercept(x)
        for l in np.sort(np.unique(y)):
            y_c = np.copy(y)
            y_c[y_c != l] = -1
            thresh = {s: 0 for s in self.sensitive_vars}
            w = u.train_model(x, y_c, x_control, self.loss, self.fairness_constr, self.acc_constr, 0,
                              self.sensitive_vars, thresh, self.gamma)
            self.class_weights.append(w)

    def pred(self, x_test):
        x_test = u.add_intercept(x_test)
        scores = []
        for w in self.class_weights:
            score = np.dot(x_test, w)
            scores.append(score)
        return np.argmax(scores, axis=0)
