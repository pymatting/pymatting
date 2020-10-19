import numpy as np
import scipy.sparse
from pymatting.util.util import weights_to_laplacian
from pymatting_aot.aot import _rw_laplacian


def rw_laplacian(image, sigma=0.033, radius=1, regularization=1e-8):
    """
    This function implements the alpha estimator for random walk alpha matting as described in :cite:`grady2005random`.

    Parameters
    ------------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    sigma: float
        Sigma used to calculate the weights (see Equation 4 in :cite:`grady2005random`), defaults to :math:`0.033`
    radius: int
        Radius of local window size, defaults to :math:`1`, i.e. only adjacent pixels are considered. The size of the local window is given as :math:`(2 r + 1)^2`, where :math:`r` denotes the radius. A larger radius might lead to violated color line constraints, but also favors further propagation of information within the image.
    regularization: float
        Regularization strength, defaults to :math:`10^{-8}`. Strong regularization improves convergence but results in smoother alpha mattes.

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
