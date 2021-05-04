from ctypes import CDLL, c_int, c_double, POINTER
import numpy as np
from config import get_library_path

library = CDLL(get_library_path("amgcl"))

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

_solve_amgcl_csr = library.solve_amgcl_csr
_solve_amgcl_csr.restype = c_int
_solve_amgcl_csr.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_int,
    c_double_p,
    c_double_p,
    c_int,
    c_double,
    c_double,
    c_int,
]


def solve_amgcl_csr(
    csr_values,
    csr_indices,
    csr_indptr,
    b,
    x=None,
    atol=1e-10,
    rtol=0,
    maxiter=10000,
):
    assert csr_values.flags["C_CONTIGUOUS"]
    assert csr_indices.flags["C_CONTIGUOUS"]
    assert csr_indptr.flags["C_CONTIGUOUS"]
    assert b.flags["C_CONTIGUOUS"]

    assert csr_values.dtype == np.float64
    assert b.dtype == np.float64
    assert csr_indices.dtype == np.int32
    assert csr_indptr.dtype == np.int32
    assert csr_values.shape[0] == csr_indices.shape[0]
    assert csr_indptr.shape[0] == b.shape[0] + 1

    if x is None:
        x = b.copy()
    else:
        assert x.dtype == np.float64
        assert x.flags["C_CONTIGUOUS"]

    n = x.shape[0]
    nnz = csr_values.shape[0]

    niter = _solve_amgcl_csr(
        np.ctypeslib.as_ctypes(csr_values),
        np.ctypeslib.as_ctypes(csr_indices.ravel()),
        np.ctypeslib.as_ctypes(csr_indptr.ravel()),
        nnz,
        np.ctypeslib.as_ctypes(b),
        np.ctypeslib.as_ctypes(x),
        n,
        atol,
        rtol,
        maxiter,
    )

    if niter < 0:
        raise ValueError("Matrix could not be inverted")

    if niter == maxiter:
        raise ValueError("Did not converge")

    return x


def main():
    import scipy.sparse
    import time

    np.random.seed(0)

    n = 10000
    k = 20 * n

    i = np.random.randint(n, size=k)
    j = np.random.randint(n, size=k)
    v = np.random.rand(k)
    i_inds = np.concatenate([i, j, np.arange(n)]).astype(np.int32)
    j_inds = np.concatenate([j, i, np.arange(n)]).astype(np.int32)
    coo_values = np.concatenate([v, v, np.random.rand(n) + n])

    A = scipy.sparse.csr_matrix((coo_values, (i_inds, j_inds)), (n, n))
    A.sum_duplicates()

    x_true = np.random.rand(n)

    b = A.dot(x_true)

    for _ in range(10):
        t = time.perf_counter()

        atol = 1e-10

        x = solve_amgcl_csr(A.data, A.indices, A.indptr, b, atol=atol)

        dt = time.perf_counter() - t

        err = np.linalg.norm(x - x_true)
        norm_r = np.linalg.norm(b - A.dot(x))
        assert norm_r <= atol
        print(
            "%f seconds - norm(r) = %.20f - norm(x - x_true) = %e" % (dt, norm_r, err)
        )
        assert err < 1e-4

    print("test passed")


if __name__ == "__main__":
    main()
