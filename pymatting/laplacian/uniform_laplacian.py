from pymatting.util.util import (
    grid_coordinates,
    sparse_conv_matrix,
    weights_to_laplacian,
)
import numpy as np


def uniform_laplacian(image, radius=1):
    """This function returns a Laplacian matrix with all weights equal to one.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    radius: int
        Radius of local window size, defaults to 1, i.e. only adjacent pixels are considered.
       The size of the local window is given as :math:`(2 r + 1)^2`, where :math:`r` denotes         the radius. A larger radius might lead to violated color line constraints, but also
       favors further propagation of information within the image.

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian
    """
    height, width = image.shape[:2]
    window_size = 2 * radius + 1

    W = sparse_conv_matrix(width, height, np.ones((window_size, window_size)))

    return weights_to_laplacian(W)
