from __future__ import absolute_import
import numpy as np
from scipy import sparse
import warnings


def safe_sparse_dot(a, b, **_3to2kwargs):
    if 'dense_output' in _3to2kwargs: dense_output = _3to2kwargs['dense_output']; del _3to2kwargs['dense_output']
    else: dense_output = False
    u"""Dot product that handle the sparse matrix case correctly.
    Parameters
    ----------
    a : {ndarray, sparse matrix}
    b : {ndarray, sparse matrix}
    dense_output : bool, default=False
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.
    Returns
    -------
    dot_product : {ndarray, sparse matrix}
        Sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = np.matmul(a, b_2d)
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = np.matmul(a_2d, b)
            ret = ret.reshape(a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = np.matmul(a, b)

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, u"toarray")
    ):
        return ret.toarray()
    return ret


def squared_norm(x):
    u"""Squared Euclidean or Frobenius norm of x.
    Faster than norm(x) ** 2.
    Parameters
    ----------
    x : array-like
    Returns
    -------
    float
        The Euclidean norm when x is a vector, the Frobenius norm when x
        is a matrix (2-d array).
    """
    x = np.ravel(x, order="K")
    if np.issubdtype(x.dtype, np.integer):
        warnings.warn(
            "Array type is integer, np.dot may overflow. "
            "Data should be float type to avoid this issue",
            UserWarning,
        )
    return np.dot(x, x)
