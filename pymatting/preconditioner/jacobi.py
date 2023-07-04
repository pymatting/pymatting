def jacobi(A):
    """
    Compute the Jacobi preconditioner function for the matrix A.

    Parameters
    ----------
    A: np.array
        Input matrix to compute the Jacobi preconditioner for.

    Returns
    -------
    precondition_matvec: function
        Function which applies the Jacobi preconditioner to a vector

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> A = np.array([[2, 3], [3, 5]])
    >>> preconditioner = jacobi(A)
    >>> preconditioner(np.array([1, 2]))
    array([0.5, 0.4])
    """
    diagonal = A.diagonal()

    inverse_diagonal = 1.0 / diagonal

    def precondition_matvec(x):
        return x * inverse_diagonal

    return precondition_matvec
