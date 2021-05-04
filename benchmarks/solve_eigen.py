from ctypes import CDLL, c_int, c_double, POINTER
import numpy as np
from config import get_library_path

library = CDLL(get_library_path("eigen"))

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

_solve_eigen_icholt_coo = library.solve_eigen_icholt_coo
_solve_eigen_icholt_coo.restype = c_int
_solve_eigen_icholt_coo.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_int,
    c_double_p,
    c_double_p,
    c_int,
    c_double,
    c_double,
]

_solve_eigen_cholesky_coo = library.solve_eigen_cholesky_coo
_solve_eigen_cholesky_coo.restype = c_int
_solve_eigen_cholesky_coo.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_int,
    c_double_p,
    c_double_p,
    c_int,
]


def solve_eigen_icholt_coo(
    coo_data,
    row,
    col,
    b,
    rtol=1e-10,
    initial_shift=0.01,
):
    assert coo_data.flags["C_CONTIGUOUS"]
    assert row.flags["C_CONTIGUOUS"]
    assert col.flags["C_CONTIGUOUS"]
    assert b.flags["C_CONTIGUOUS"]
    assert coo_data.dtype == np.float64
    assert b.dtype == np.float64
    assert row.dtype == np.int32
    assert col.dtype == np.int32
    assert coo_data.shape[0] == row.shape[0]
    assert coo_data.shape[0] == col.shape[0]

    n = b.shape[0]
    nnz = coo_data.shape[0]

    x = np.empty(n)

    err = _solve_eigen_icholt_coo(
        np.ctypeslib.as_ctypes(coo_data),
        np.ctypeslib.as_ctypes(row.ravel()),
        np.ctypeslib.as_ctypes(col.ravel()),
        nnz,
        np.ctypeslib.as_ctypes(b),
        np.ctypeslib.as_ctypes(x),
        n,
        rtol,
        initial_shift,
    )

    if err:
        raise ValueError("Linear system could not be solved")

    return x


def solve_eigen_cholesky_coo(coo_data, row, col, b):
    assert coo_data.flags["C_CONTIGUOUS"]
    assert row.flags["C_CONTIGUOUS"]
    assert col.flags["C_CONTIGUOUS"]
    assert b.flags["C_CONTIGUOUS"]
    assert coo_data.dtype == np.float64
    assert b.dtype == np.float64
    assert row.dtype == np.int32
    assert col.dtype == np.int32
    assert coo_data.shape[0] == row.shape[0]
    assert coo_data.shape[0] == col.shape[0]

    n = b.shape[0]
    nnz = coo_data.shape[0]

    x = np.empty(n)

    err = _solve_eigen_cholesky_coo(
        np.ctypeslib.as_ctypes(coo_data),
        np.ctypeslib.as_ctypes(row.ravel()),
        np.ctypeslib.as_ctypes(col.ravel()),
        nnz,
        np.ctypeslib.as_ctypes(b),
        np.ctypeslib.as_ctypes(x),
        n,
    )

    if err:
        raise ValueError("Linear system could not be solved")

    return x


def main():
    import scipy.sparse

    np.random.seed(0)

    n = 100
    k = 20 * n

    i = np.random.randint(n, size=k)
    j = np.random.randint(n, size=k)
    v = np.random.rand(k)
    i_inds = np.concatenate([i, j, np.arange(n)]).astype(np.int32)
    j_inds = np.concatenate([j, i, np.arange(n)]).astype(np.int32)
    coo_values = np.concatenate([v, v, n * np.ones(n)])

    A = scipy.sparse.coo_matrix((coo_values, (i_inds, j_inds)), (n, n))
    A.sum_duplicates()

    x_true = np.random.rand(n)
    b = A.dot(x_true)

    solvers = [
        solve_eigen_icholt_coo,
        solve_eigen_cholesky_coo,
    ]

    for solver in solvers:
        x = solver(A.data, A.row, A.col, b)

        err = np.linalg.norm(x - x_true)

        print("norm(x - x_true) = %e " % err)

        assert err < 1e-5

        print("test passed")


if __name__ == "__main__":
    main()
