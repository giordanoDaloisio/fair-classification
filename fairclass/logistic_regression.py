import utils as ut
import loss_funcs as lf
import numpy as np


class LogisticRegression:
    def __init__(self, fairness_const, acc_constr, sensitive_attr, gamma):
        self.gamma = gamma
        self.sensitive_attr = sensitive_attr
        self.acc_constr = acc_constr
        self.weights = []
        self.fairness_constr = fairness_const

    def fit(self, x, x_control, y):
        x = ut.add_intercept(x)
        loss = lf._logistic_loss
        sensitive_cov = {s: 0 for s in self.sensitive_attr}
        w = ut.train_model(x, y, x_control, loss, self.fairness_constr, self.acc_constr, 0, self.sensitive_attr,
                           sensitive_cov, self.gamma)
        self.weights = w

    def pred(self, x):
        x = ut.add_intercept(x)
        scores = np.dot(x, self.weights)
        ris = []
        for s in scores:
            ris += [1] if s > 0.5 else [0]
        return ris
