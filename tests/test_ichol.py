import numpy as np
import scipy.sparse
from pymatting import ichol


def test_ichol():
    np.random.seed(0)

    for _ in range(10):
        n = 50
        A = np.random.rand(n, n)
        A[np.random.rand(n, n) < 0.8] = 0
        A += A.T + n * np.eye(n)

        A = scipy.sparse.csc_matrix(A)

        decomposition = ichol(A, discard_threshold=0.0, max_nnz=100000)

        x_true = np.random.rand(n)
        b = A.dot(x_true)

        x = decomposition(b)

        error = np.linalg.norm(x - x_true)

        assert error < 1e-10
