from ctypes import CDLL, c_int, c_double, POINTER
import numpy as np
from config import get_library_path

library = CDLL(get_library_path("mumps"))

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

init_mpi = library.init_mpi
init_mpi.restype = c_int

init_mpi()

finalize_mpi = library.finalize_mpi
finalize_mpi.restype = c_int

_solve_mumps_coo = library.solve_mumps_coo
_solve_mumps_coo.restype = c_int
_solve_mumps_coo.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_int,
    c_double_p,
    c_int,
    c_int,
    c_int,
]


def solve_mumps_coo(
    coo_values, i_inds, j_inds, b, x=None, is_symmetric=False, print_info=False
):
    assert coo_values.flags["C_CONTIGUOUS"]
    assert i_inds.flags["C_CONTIGUOUS"]
    assert j_inds.flags["C_CONTIGUOUS"]
    assert b.flags["C_CONTIGUOUS"]
    assert coo_values.dtype == np.float64
    assert b.dtype == np.float64
    assert i_inds.dtype == np.int32
    assert j_inds.dtype == np.int32
    assert coo_values.shape[0] == i_inds.shape[0]
    assert coo_values.shape[0] == j_inds.shape[0]

    if x is None:
        x = b.copy()
    else:
        assert x.dtype == np.float64
        assert x.flags["C_CONTIGUOUS"]
        x[:] = b

    n = x.shape[0]
    nnz = coo_values.shape[0]

    err = _solve_mumps_coo(
        np.ctypeslib.as_ctypes(coo_values),
        np.ctypeslib.as_ctypes((i_inds + 1).ravel()),
        np.ctypeslib.as_ctypes((j_inds + 1).ravel()),
        nnz,
        np.ctypeslib.as_ctypes(x),
        n,
        is_symmetric,
        print_info,
    )

    if err:
        raise ValueError("Matrix could not be inverted")

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
    coo_values = np.concatenate([v, v, np.random.rand(n)])

    A = scipy.sparse.coo_matrix((coo_values, (i_inds, j_inds)), (n, n))
    A.sum_duplicates()

    x_true = np.random.rand(n)
    b = A.dot(x_true)

    x = solve_mumps_coo(A.data, A.row, A.col, b, is_symmetric=False)

    assert np.linalg.norm(x - x_true) < 1e-10

    A = scipy.sparse.tril(A)

    x = solve_mumps_coo(A.data, A.row, A.col, b, is_symmetric=True)

    assert np.linalg.norm(x - x_true) < 1e-10

    print("test passed")


if __name__ == "__main__":
    main()
