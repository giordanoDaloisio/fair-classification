import sys
import os
import numpy as np
from scipy.special import logsumexp
from extmath import safe_sparse_dot, squared_norm
from scipy import sparse
from collections import defaultdict
import traceback
from copy import deepcopy


def _hinge_loss(w, X, y):
    yz = y * np.dot(X, w)  # y * (x.w)
    yz = np.maximum(np.zeros_like(yz), (1 - yz))  # hinge function

    return sum(yz)


def _logistic_loss(w, X, y, return_arr=None):
    """Computes the logistic loss.
    This function is used from scikit-learn source code

    Parameters
    ----------
    w : ndarray, shape (n_features,) or (n_features + 1,)
        Coefficient vector.

    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data.

    y : ndarray, shape (n_samples,)
        Array of labels.

    """

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    if return_arr:
        out = -(log_logistic(yz))
    else:
        out = -np.sum(log_logistic(yz))
    return out


def _logistic_loss_l2_reg(w, X, y, lam=None):
    if lam is None:
        lam = 1.0

    yz = y * np.dot(X, w)
    # Logistic loss is the negative of the log of the logistic function.
    logistic_loss = -np.sum(log_logistic(yz))
    l2_reg = (float(lam) / 2.0) * np.sum([elem * elem for elem in w])
    out = logistic_loss + l2_reg
    return out


def log_logistic(X):
    """ This function is used from scikit-learn source code. Source link below """

    """Compute the log of the logistic function, ``log(1 / (1 + e ** -x))``.
    This implementation is numerically stable because it splits positive and
    negative values::
        -log(1 + exp(-x_i))     if x_i > 0
        x_i - log(1 + exp(x_i)) if x_i <= 0

    Parameters
    ----------
    X: array-like, shape (M, N)
        Argument to the logistic function

    Returns
    -------
    out: array, shape (M, N)
        Log of the logistic function evaluated at every point in x
    Notes
    -----
    Source code at:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/extmath.py
    -----

    See the blog post describing this implementation:
    http://fa.bianp.net/blog/2013/numerical-optimizers-for-logistic-regression/
    """
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X)  # same dimensions and data types

    idx = X > 0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out


def _multinomial_loss(w, X, Y, alpha, sample_weight):
    """Computes multinomial loss and class probabilities.
    Parameters
    ----------
    w : ndarray of shape (n_classes * n_features,) or
        (n_classes * (n_features + 1),)
        Coefficient vector.
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        Training data.
    Y : ndarray of shape (n_samples, n_classes)
        Transformed labels according to the output of LabelBinarizer.
    alpha : float
        Regularization parameter. alpha is equal to 1 / C.
    sample_weight : array-like of shape (n_samples,)
        Array of weights that are assigned to individual samples.
    Returns
    -------
    loss : float
        Multinomial loss.
    p : ndarray of shape (n_samples, n_classes)
        Estimated class probabilities.
    w : ndarray of shape (n_classes, n_features)
        Reshaped param vector excluding intercept terms.
    Reference
    ---------
    Bishop, C. M. (2006). Pattern recognition and machine learning.
    Springer. (Chapter 4.3.4)
    """
    n_classes = Y.shape[1]
    n_features = X.shape[1]
    fit_intercept = w.size == (n_classes * (n_features + 1))
    w = w.reshape(n_classes, -1)
    sample_weight = sample_weight[:, np.newaxis]
    if fit_intercept:
        intercept = w[:, -1]
        w = w[:, :-1]
    else:
        intercept = 0
    p = safe_sparse_dot(X, w.T)
    p += intercept
    p -= logsumexp(p, axis=1)[:, np.newaxis]
    loss = -(sample_weight * Y * p).sum()
    loss += 0.5 * alpha * squared_norm(w)
    p = np.exp(p, p)
    return loss, p, w
