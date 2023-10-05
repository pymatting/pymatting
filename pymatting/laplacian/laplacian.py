from pymatting.util.util import trimap_split
import scipy.sparse


def make_linear_system(L, trimap, lambda_value=100.0, return_c=False):
    """This function constructs a linear system from a matting Laplacian by
    constraining the foreground and background pixels with a diagonal matrix
    `C` to values in the right-hand-side vector `b`. The constraints are
    weighted by a factor :math:`\lambda`. The linear system is given as

    .. math::

      A = L + \lambda C,

    where :math:`C=\mathop{Diag}(c)` having :math:`c_i = 1` if pixel i is known
    and :math:`c_i = 0` otherwise.
    The right-hand-side :math:`b` is a vector with entries :math:`b_i = 1` is
    pixel is is a foreground pixel and :math:`b_i = 0` otherwise.


    Parameters
    ----------
    L: scipy.sparse.spmatrix
        Laplacian matrix, e.g. calculated with :code:`lbdm_laplacian` function
    trimap: numpy.ndarray
        Trimap with shape :math:`h\\times w`
    lambda_value: float
        Constraint penalty, defaults to 100
    return_c: bool
        Whether to return the constraint matrix `C`, defaults to False

    Returns
    -------
    A: scipy.sparse.spmatrix
        Matrix describing the system of linear equations
    b: numpy.ndarray
        Vector describing the right-hand side of the system
    C: numpy.ndarray
        Vector describing the diagonal entries of the matrix `C`, only returned
        if `return_c` is set to True
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
