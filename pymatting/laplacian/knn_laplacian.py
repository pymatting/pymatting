import numpy as np
import scipy.sparse
from pymatting.util.kdtree import knn
from pymatting.util.util import normalize_rows


def knn_laplacian(
    image, n_neighbors=[20, 10], distance_weights=[2.0, 0.1],
):
    """
    This function calculates the KNN matting Laplacian matrix as described in :cite:`chen2013knn`.

    Parameters
    ----------
    image: numpy.ndarray
        Image
    n_neighbors: list of ints
        Number of neighbors to consider
    distance_weights: list of floats
        Weight of distance in feature vector

    Returns
    ---------
    L: scipy.sparse.spmatrix
        Matting Laplacian matrix
    """
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    i, j = [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        distances, neighbors = knn(f, f, k=k)

        i.append(np.repeat(np.arange(n), k))
        j.append(neighbors.flatten())

    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.ones(2 * sum(n_neighbors) * n)

    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))

    W = normalize_rows(W)

    I = scipy.sparse.identity(n)

    L = I - W

    return L
