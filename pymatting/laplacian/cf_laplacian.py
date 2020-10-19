import numpy as np
import scipy.sparse
from pymatting_aot.aot import _cf_laplacian


def cf_laplacian(image, epsilon=1e-7, radius=1):
    """
    This function implements the alpha estimator for closed-form alpha matting as proposed by :cite:`levin2007closed`.

    Parameters
    ------------
    image: numpy.ndarray
       Image with shape :math:`h\\times w \\times 3`
    epsilon: float
       Regularization strength, defaults to :math:`10^{-7}`. Strong regularization improves convergence but results in smoother alpha mattes.
    radius: int
       Radius of local window size, defaults to :math:`1`, i.e. only adjacent pixels are considered.
       The size of the local window is given as :math:`(2 r + 1)^2`, where :math:`r` denotes         the radius. A larger radius might lead to violated color line constraints, but also
       favors further propagation of information within the image.

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian
    """
    h, w, d = image.shape
    n = h * w

    # Data for matting laplacian in csr format
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(n * (4 * radius + 1) ** 2, dtype=np.int64)
    values = np.zeros((n, 4 * radius + 1, 4 * radius + 1), dtype=np.float64)

    _cf_laplacian(image, epsilon, radius, values, indices, indptr)

    L = scipy.sparse.csr_matrix((values.ravel(), indices, indptr), (n, n))

    return L
