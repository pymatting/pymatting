import numpy as np
import scipy.sparse
from numba import njit

@njit("f8[:, :](f8[:, :], f8)", cache=True, nogil=True)
def calculate_kernel_matrix(X, v):
    n, m = X.shape
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = np.exp(-np.sqrt(v) * np.sum(np.square(X[i] - X[j])))
    return K


@njit("Tuple((f8[:], i4[:], i4[:]))(f8[:, :, :], f8, i4)", cache=True, nogil=True)
def _lbdm_laplacian(image, epsilon, r):
    h, w = image.shape[:2]
    n = h * w

    area = (2 * r + 1) ** 2

    indices = np.arange(n).reshape(h, w)

    values = np.zeros((n, area ** 2))
    i_inds = np.zeros((n, area ** 2), dtype=np.int32)
    j_inds = np.zeros((n, area ** 2), dtype=np.int32)

    # gray = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3.0
    # v = np.std(gray)

    for y in range(r, h - r):
        for x in range(r, w - r):
            i = x + y * w

            X = np.ones((area, 3 + 1))

            k = 0
            for y2 in range(y - r, y + r + 1):
                for x2 in range(x - r, x + r + 1):
                    for c in range(3):
                        X[k, c] = image[y2, x2, c]
                    k += 1

            window_indices = indices[y - r : y + r + 1, x - r : x + r + 1].flatten()

            # does not produce better results than no kernel
            # K = calculate_kernel_matrix(X, v)

            K = np.dot(X, X.T)

            f = np.linalg.solve(K + epsilon * np.eye(area), K)

            tmp2 = np.eye(f.shape[0]) - f
            tmp3 = tmp2.dot(tmp2.T)

            for k in range(area):
                i_inds[i, k::area] = window_indices
                j_inds[i, k * area : k * area + area] = window_indices
            values[i] = tmp3.ravel()

    return values.ravel(), i_inds.ravel(), j_inds.ravel()


def lbdm_laplacian(image, epsilon=1e-7, radius=1):
    """
    Calculate a Laplacian matrix based on :cite:`zheng2009learning`.

    Parameters
    ----------
    image: numpy.ndarray
       Image with shape :math:`h\\times w \\times 3`
    epsilon: float
       Regularization strength, defaults to :math:`10^{-7}`. Strong
       regularization improves convergence but results in smoother alpha mattes.
    radius: int
       Radius of local window size, defaults to :math:`1`, i.e. only adjacent
       pixels are considered. The size of the local window is given as
       :math:`(2 r + 1)^2`, where :math:`r` denotes the radius. A larger radius
       might lead to violated color line constraints, but also favors further
       propagation of information within the image.

    Returns
    -------
    L: scipy.sparse.csr_matrix
        Matting Laplacian
    """
    h, w = image.shape[:2]
    n = h * w

    values, i_inds, j_inds = _lbdm_laplacian(image, epsilon, radius)

    L = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))

    return L
