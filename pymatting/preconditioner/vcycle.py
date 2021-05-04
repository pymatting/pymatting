import numpy as np
import scipy.sparse
import scipy.sparse.linalg


def make_P(shape):
    h, w = shape
    n = h * w
    h2 = h // 2
    w2 = w // 2
    n2 = w2 * h2
    weights = np.float64([1, 2, 1, 2, 4, 2, 1, 2, 1]) / 16

    x2 = np.repeat(np.tile(np.arange(w2), h2), 9)
    y2 = np.repeat(np.repeat(np.arange(h2), w2), 9)

    x = x2 * 2 + np.tile([-1, 0, 1, -1, 0, 1, -1, 0, 1], n2)
    y = y2 * 2 + np.tile([-1, -1, -1, 0, 0, 0, 1, 1, 1], n2)

    mask = (0 <= x) & (x < w) & (0 <= y) & (y <= h)

    i_inds = (x2 + y2 * w2)[mask]
    j_inds = (x + y * w)[mask]
    values = np.tile(weights, n2)[mask]

    downsample = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), (n2, n))
    upsample = downsample.T

    return upsample, downsample


def jacobi_step(A, A_diag, b, x, num_iter, omega):
    if x is None:
        if num_iter > 0:
            x = omega * b / A_diag
            num_iter -= 1
        else:
            x = np.zeros_like(b)

    for _ in range(num_iter):
        x = x + omega * (b - A.dot(x)) / A_diag

    return x


def _vcycle_step(
    A,
    b,
    shape,
    cache,
    num_pre_iter,
    num_post_iter,
    omega,
    direct_solve_size,
):
    h, w = shape
    n = h * w

    if n <= direct_solve_size:
        return scipy.sparse.linalg.spsolve(A, b)

    if shape not in cache:
        upsample, downsample = make_P(shape)

        coarse_A = downsample.dot(A).dot(upsample)

        A_diag = A.diagonal()

        cache[shape] = (upsample, downsample, coarse_A, A_diag)
    else:
        upsample, downsample, coarse_A, A_diag = cache[shape]

    # smooth error
    x = jacobi_step(A, A_diag, b, None, num_pre_iter, omega)

    # calculate residual error to perfect solution
    residual = b - A.dot(x)

    # downsample residual error
    coarse_residual = downsample.dot(residual)

    # calculate coarse solution for residual
    coarse_x = _vcycle_step(
        coarse_A,
        coarse_residual,
        (h // 2, w // 2),
        cache,
        num_pre_iter,
        num_post_iter,
        omega,
        direct_solve_size,
    )

    # apply coarse correction
    x += upsample.dot(coarse_x)

    # smooth error
    x = jacobi_step(A, A_diag, b, x, num_post_iter, omega)

    return x


def vcycle(
    A,
    shape,
    num_pre_iter=1,
    num_post_iter=1,
    omega=0.8,
    direct_solve_size=64,
    cache=None,
):
    """
    Implements the V-Cycle preconditioner.
    The V-Cycle solver was recommended by :cite:`lee2014scalable` to solve the alpha matting problem.

    Parameters
    ----------
    A: numpy.ndarray
        Input matrix
    shape: tuple of ints
        Describing the height and width of the image
    num_pre_iter: int
        Number of Jacobi iterations before each V-Cycle, defaults to 1
    num_post_iter: int
        Number of Jacobi iterations after each V-Cycle, defaults to 1
    omega: float
        Weight parameter for the Jacobi method. If method fails to converge, try different values.

    Returns
    -------
    precondition: function
        Function which applies the V-Cycle preconditioner to a vector

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> A = np.array([[2, 3], [3, 5]])
    >>> preconditioner = vcycle(A, (2, 2))
    >>> preconditioner(np.array([1, 2]))
    array([-1.,  1.])
    """

    if cache is None:
        cache = {}

    def precondition(r):
        return _vcycle_step(
            A, r, shape, cache, num_pre_iter, num_post_iter, omega, direct_solve_size
        )

    return precondition
