import numpy as np
import scipy.sparse
from pymatting.util.util import weights_to_laplacian
from numba import njit

@njit("Tuple((f8[:], i4[:], i4[:]))(f8[:,:,:], f8, i4)", cache=True, nogil=True)
def _rw_laplacian(image, sigma, r):
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
    This function implements the alpha estimator for random walk alpha matting
    as described in :cite:`grady2005random`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    sigma: float
        Sigma used to calculate the weights (see Equation 4 in
        :cite:`grady2005random`), defaults to :math:`0.033`
    radius: int
        Radius of local window size, defaults to :math:`1`, i.e. only adjacent
        pixels are considered. The size of the local window is given as
        :math:`(2 r + 1)^2`, where :math:`r` denotes the radius. A larger radius
        might lead to violated color line constraints, but also favors further
        propagation of information within the image.
    regularization: float
        Regularization strength, defaults to :math:`10^{-8}`. Strong
        regularization improves convergence but results in smoother alpha matte.

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian
    """
    h, w = image.shape[:2]
    n = h * w

    values, i_inds, j_inds = _rw_laplacian(image, sigma, radius)

    W = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))

    return weights_to_laplacian(W, regularization=regularization)
