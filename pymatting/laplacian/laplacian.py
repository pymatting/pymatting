from pymatting.util.util import trimap_split
import scipy.sparse
import numpy as np


def make_linear_system(L, trimap, lambda_value=100.0, return_c=False):
    """This function constructs a linear system from a matting Laplacian and a trimap as described in []

    Parameters
    ----------
    L: scipy.sparse.spmatrix
        Matting matrix
    trimap: numpy.ndarray
        Trimap
    lambda_value: float
        Constraint penalty, defaults to 100
    return_c: bool
        Whether to return the constraint matrix C, defaults to false
    
    Returns
    -------
    A: scipy.sparse.spmatrix
        Matrix describing the system of linear equations
    b: numpy.ndarray
        Vector describing the right-hand side of the system
    C: numpy.ndarray
        Vector describing the diagonal entries of the matrix C, only returned if `return_c` is set to True
    """
    h, w = trimap.shape[:2]

    is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

    c = lambda_value * is_known
    b = lambda_value * is_fg

    C = scipy.sparse.diags(c)

    A = L + C

    A = A.tocsr()

    A.sum_duplicates()

    if return_c:
        return A, b, c

    return A, b
