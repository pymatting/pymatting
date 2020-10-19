import numpy as np
import scipy.sparse
from pymatting_aot.aot import _lbdm_laplacian


def lbdm_laplacian(image, epsilon=1e-7, radius=1):
    """
    Calculate a Laplacian matrix based on :cite:`zheng2009learning`.

    Parameters
    ----------
    image: numpy.ndarray
       Image with shape :math:`h\\times w \\times 3`
    epsilon: float
       Regularization strength, defaults to :math:`10^{-7}`. Strong regularization improves convergence but results in smoother alpha mattes.
    radius: int
       Radius of local window size, defaults to :math:`1`, i.e. only adjacent pixels are considered. The size of the local window is given as :math:`(2 r + 1)^2`, where :math:`r` denotes the radius. A larger radius might lead to violated color line constraints, but also favors further propagation of information within the image.

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
