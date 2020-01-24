import numpy as np
from numba import njit
import scipy.sparse
from pymatting.util.util import weights_to_laplacian


@njit("Tuple((f8[:], i4[:], i4[:]))(f8[:,:,:], f8, i4)", cache=True)
def _rw(image, sigma, r):
    h, w = image.shape[:2]
    n = h * w

    m = n * (2 * r + 1) ** 2

    i_inds = np.empty(m, dtype=np.int32)
    j_inds = np.empty(m, dtype=np.int32)
    values = np.empty(m)

    k = 0

    for y in range(h):
        for x in range(w):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x2 = x + dx
                    y2 = y + dy

                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    i = x + y * w
                    j = x2 + y2 * w

                    zi = image[y, x]
                    zj = image[y2, x2]

                    wij = np.exp(-900 * np.linalg.norm(zi - zj) ** 2)

                    i_inds[k] = i
                    j_inds[k] = j

                    values[k] = wij

                    k += 1

    return values, i_inds, j_inds


def rw_laplacian(image, sigma=0.033, radius=1, regularization=1e-8):
    """
    This function implements the alpha estimator for random walk alpha matting as described in :cite:`grady2005random`.

    Parameters
    ------------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    sigma: float
        Sigma used to calculate the weights (see Equation 4 in :cite:`grady2005random`), defaults to 0.033
    radius: int
        Local window size, defaults to 1
    regularitaion: float
        Regularization strength, defaults to 1e-8

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian
    """
    h, w = image.shape[:2]
    n = h * w

    values, i_inds, j_inds = _rw(image, sigma, radius)

    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))

    return weights_to_laplacian(W, regularization=regularization)
