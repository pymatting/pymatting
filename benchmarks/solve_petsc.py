from ctypes import CDLL, c_int, c_double, POINTER
import numpy as np
from config import get_library_path

library = CDLL(get_library_path("petsc"))

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

init_petsc = library.init_petsc
init_petsc.res_type = c_int

finalize_petsc = library.finalize_petsc
finalize_petsc.res_type = c_int

init_petsc()

_solve_petsc_coo = library.solve_petsc_coo
_solve_petsc_coo.restype = c_int
_solve_petsc_coo.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_int,
    c_double_p,
    c_double_p,
    c_int,
    c_double,
    c_double,
    c_double,
    c_int,
]


def solve_petsc_coo(
    coo_values,
    i_inds,
    j_inds,
    b,
    x=None,
    atol=1e-10,
    rtol=0,
    gamg_threshold=0.1,
    maxiter=10000,
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

    n = x.shape[0]
    nnz = coo_values.shape[0]

    err = _solve_petsc_coo(
        np.ctypeslib.as_ctypes(coo_values),
        np.ctypeslib.as_ctypes(i_inds.ravel()),
        np.ctypeslib.as_ctypes(j_inds.ravel()),
        nnz,
        np.ctypeslib.as_ctypes(b),
        np.ctypeslib.as_ctypes(x),
        n,
        atol,
        rtol,
        gamg_threshold,
        maxiter,
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
    coo_values = np.concatenate([v, v, np.random.rand(n) + n])

    A = scipy.sparse.coo_matrix((coo_values, (i_inds, j_inds)), (n, n))
    A.sum_duplicates()

    # indices must be sorted
    pairs = list(zip(A.col, A.row))
    assert all(a < b for a, b in zip(pairs, pairs[1:]))

    # rows and columns must not contain more than 'n' elements
    count = np.zeros(n)
    for i in A.row:
        count[i] += 1
    assert np.all(count < n)
    count = np.zeros(n)
    for i in A.col:
        count[i] += 1
    assert np.all(count < n)

    seen = np.zeros((n, n), dtype=np.bool8)

    for i, j in zip(A.row, A.col):
        assert not seen[i, j]
        seen[i, j] = True

    x_true = np.random.rand(n)
    b = A.dot(x_true)

    x = solve_petsc_coo(A.data, A.row, A.col, b)

    err = np.linalg.norm(x - x_true)
    print("norm(x - x_true) = %e" % err)
    assert err < 1e-4

    print("test passed")


if __name__ == "__main__":
    main()
