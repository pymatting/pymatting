from pymatting.util.util import (
    grid_coordinates,
    sparse_conv_matrix,
    weights_to_laplacian,
)
import numpy as np


def uniform_laplacian(image, radius=1):
    """This function returns a Laplacian matrix with all weights equal to one.

    Parameters
    ------------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    radius: int
        Local window size, defaults to 1

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian
    """
    height, width = image.shape[:2]
    window_size = 2 * radius + 1

    W = sparse_conv_matrix(width, height, np.ones((window_size, window_size)))

    return weights_to_laplacian(W)
