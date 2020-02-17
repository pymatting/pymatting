from pymatting import cg, CounterCallback, ProgressCallback
import numpy as np


def test_cg():
    np.random.seed(0)

    atol = 1e-10

    for n in [1, 5, 10]:
        x_true = np.random.rand(n)

        # make positive definite matrix
        A = np.random.rand(n, n)
        A += A.T + n * np.eye(n)

        b = A.dot(x_true)

        M_jacobi = np.diag(1.0 / np.diag(A))

        def precondition(x):
            return M_jacobi.dot(x)

        for x0 in [None, np.random.rand(n)]:
            for M in [None, M_jacobi, precondition]:
                for callback in [None, CounterCallback(), ProgressCallback()]:
                    x = cg(A, b, x0=x0, M=M, rtol=0, atol=atol, callback=callback)

                    r = np.linalg.norm(A.dot(x) - b)

                    assert np.linalg.norm(r) < 1e-10

                    if callback is not None:
                        assert 0 < callback.n < 20
