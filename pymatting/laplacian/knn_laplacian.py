import numpy as np
import scipy.sparse
from pymatting.util.kdtree import knn
from pymatting.util.util import normalize_rows


def knn_laplacian(
    image,
    n_neighbors=[20, 10],
    distance_weights=[2.0, 0.1],
    kernel="binary",
):
    """
    This function calculates the KNN matting Laplacian matrix similar to
    :cite:`chen2013knn`.
    We use a kernel of 1 instead of a soft kernel by default since the former is
    faster to compute and both produce almost identical results in all our
    experiments, which is to be expected as the soft kernel is very close to 1
    in most cases.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h\\times w \\times 3`
    n_neighbors: list of ints
        Number of neighbors to consider. If :code:`len(n_neighbors)>1` multiple
        nearest neighbor calculations are done and merged, defaults to
        `[20, 10]`, i.e. first 20 neighbors are considered and in the second run
        :math:`10` neighbors. The pixel distances are then weighted by the
        :code:`distance_weights`.
    distance_weights: list of floats
        Weight of distance in feature vector, defaults to `[2.0, 0.1]`.
    kernel: str
        Must be either "binary" or "soft". Default is "binary".

    Returns
    -------
    L: scipy.sparse.spmatrix
        Matting Laplacian matrix
    """
    h, w = image.shape[:2]
    r, g, b = image.reshape(-1, 3).T
    n = w * h

    if kernel not in ["binary", "soft"]:
        raise ValueError("kernel must be binary/soft, but not " + kernel + ".")

    x = np.tile(np.linspace(0, 1, w), h)
    y = np.repeat(np.linspace(0, 1, h), w)

    # Store weight matrix indices and values in sparse coordinate form.
    i, j, coo_data = [], [], []

    for k, distance_weight in zip(n_neighbors, distance_weights):
        # Features consist of RGB color values and weighted spatial coordinates.
        f = np.stack(
            [r, g, b, distance_weight * x, distance_weight * y],
            axis=1,
            out=np.zeros((n, 5), dtype=np.float32),
        )

        # Find indices of nearest neighbors in feature space.
        _, neighbor_indices = knn(f, f, k=k)

        # [0 0 0 0 0 (k times) 1 1 1 1 1 2 2 2 2 2 ...]
        i.append(np.repeat(np.arange(n), k))
        j.append(neighbor_indices.ravel())

        W_ij = np.ones(k * n)

        if kernel == "soft":
            W_ij -= np.abs(f[i[-1]] - f[j[-1]]).sum(axis=1) / f.shape[1]

        coo_data.append(W_ij)

    # Add matrix to itself to get a symmetric matrix.
    # The '+' here is list concatenation and not addition.
    # The csr_matrix constructor will do the addition later.
    ij = np.concatenate(i + j)
    ji = np.concatenate(j + i)
    coo_data = np.concatenate(coo_data + coo_data)

    # Assemble weights from coordinate sparse matrix format.
    W = scipy.sparse.csr_matrix((coo_data, (ij, ji)), (n, n))

    W = normalize_rows(W)

    I = scipy.sparse.identity(n)

    L = I - W

    return L
