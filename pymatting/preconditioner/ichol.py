import numpy as np
import scipy.sparse
from pymatting_aot.aot import _ichol, _backsub_L_csc_inplace, _backsub_LT_csc_inplace


class CholeskyDecomposition(object):
    """Cholesky Decomposition

    Calling this object applies the preconditioner to a vector by forward and back substitution.

    Parameters
    ----------
    Ltuple: tuple of numpy.ndarrays
        Tuple of array describing values, row indices and row pointers for Cholesky factor in the compressed sparse comlumn format (csc)
    """

    def __init__(self, Ltuple):
        self.Ltuple = Ltuple

    @property
    def L(self):
        """Returns the Cholesky factor

        Returns
        -------
        L: scipy.sparse.csc_matrix
            Cholesky factor
        """
        Lv, Lr, Lp = self.Ltuple
        n = len(Lp) - 1
        return scipy.sparse.csc_matrix(self.Ltuple, (n, n))

    def __call__(self, b):
        Lv, Lr, Lp = self.Ltuple
        n = len(b)
        x = b.copy()
        _backsub_L_csc_inplace(Lv, Lr, Lp, x, n)
        _backsub_LT_csc_inplace(Lv, Lr, Lp, x, n)
        return x


def ichol(
    A,
    discard_threshold=1e-4,
    shifts=[0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 100, 1e3, 1e4, 1e5],
    max_nnz=int(4e9 / 16),
):
    """Implements the thresholded incomplete Cholesky decomposition

    Parameters
    ----------
    A: scipy.sparse.csc_matrix
        Matrix for which the preconditioner should be calculated
    discard_threshold: float
        Values having an absolute value smaller than this theshold will be discarded while calculating the cholesky decompositions
    shifts: array of floats
        Values to try for regularizing the matrix of interest in case it is not positive definite after discarding the small values
    max_nnz: int
        Maximum number of non-zero entries in the Cholesky decomposition. Defaults to 250 million, which should usually be around 4 GB.

    Returns
    -------
    chol: CholeskyDecomposition
        Preconditioner or solver object.

    Raises
    ------
    ValueError
        If inappropriate parameter values were passed

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> A = np.array([[2.0, 3.0], [3.0, 5.0]])
    >>> cholesky_decomposition = ichol(csc_matrix(A))
    >>> cholesky_decomposition(np.array([1.0, 2.0]))
    array([-1.,  1.])
    """

    if isinstance(A, scipy.sparse.csr_matrix):
        A = A.T

    if not isinstance(A, scipy.sparse.csc_matrix):
        raise ValueError("Matrix A must be a scipy.sparse.csc_matrix")

    if not A.has_canonical_format:
        A.sum_duplicates()

    m, n = A.shape

    assert m == n

    Lv = np.empty(max_nnz, dtype=np.float64)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int64)  # Row indices of non-zero elements of L
    Lp = np.zeros(
        n + 1, dtype=np.int64
    )  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv

    for shift in shifts:
        nnz = _ichol(
            n,
            A.data,
            A.indices.astype(np.int64),
            A.indptr.astype(np.int64),
            Lv,
            Lr,
            Lp,
            discard_threshold,
            shift,
            max_nnz,
        )

        if nnz >= 0:
            break

        if nnz == -1:
            print("PERFORMANCE WARNING:")
            print(
                "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A with parameters:"
            )
            print("    discard_threshold = %e" % discard_threshold)
            print("    shift = %e" % shift)
            print("Try decreasing discard_threshold or start with a larger shift")
            print("")

        if nnz == -2:
            raise ValueError(
                "Thresholded incomplete Cholesky decomposition failed because more than max_nnz non-zero elements were created. Try increasing max_nnz or discard_threshold."
            )

    if nnz < 0:
        raise ValueError(
            "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A and diagonal shifts did not help."
        )

    Lv = Lv[:nnz]
    Lr = Lr[:nnz]

    return CholeskyDecomposition((Lv, Lr, Lp))
