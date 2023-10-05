import numpy as np
from pymatting.util.util import vec_vec_dot, mat_vec_dot, vec_vec_outer
from pymatting.util.boxfilter import boxfilter


def lkm_laplacian(image, epsilon=1e-7, radius=10, return_diagonal=True):
    """Calculates the Laplacian for large kernel matting :cite:`he2010fast`

    Parameters
    ----------
    image: numpy.ndarray
        Image of shape :math:`h\\times w \\times 3`
    epsilons: float
        Regularization strength, defaults to :math:`10^{-7}`
    radius: int
        Radius of local window size, defaults to :math:`10`, i.e. only adjacent
        pixels are considered. The size of the local window is given as
        :math:`(2 r + 1)^2`, where :math:`r` denotes the radius. A larger radius
        might lead to violated color line constraints, but also favors further
        propagation of information within the image.
    return_diagonal: bool
        Whether to also return the diagonal of the laplacian, defaults to True

    Returns
    -------
    L_matvec: function
        Function that applies the Laplacian matrix to a vector
    diag_L: numpy.ndarray
        Diagonal entries of the matting Laplacian, only returns if
        `return_diagonal` is True
    """
    image = image.astype(np.float64)

    window_size = 2 * radius + 1
    window_area = window_size * window_size
    h, w, depth = image.shape

    # means over neighboring pixels
    means = boxfilter(image, radius, mode="valid") / window_area

    # color covariance over neighboring pixels
    covs = boxfilter(vec_vec_outer(image, image), radius, mode="valid") / window_area
    covs -= vec_vec_outer(means, means)

    # precompute values which do not depend on p
    V = np.linalg.inv(covs + epsilon / window_area * np.eye(depth)) / window_area
    Vm = mat_vec_dot(V, means)
    mVm = 1 / window_area + vec_vec_dot(means, Vm)
    c = boxfilter(np.ones((h - 2 * radius, w - 2 * radius)), radius, mode="full")

    def L_matvec(p):
        p = p.reshape(h, w)

        p_sums = boxfilter(p, radius, mode="valid")
        pI_sums = boxfilter(p[:, :, np.newaxis] * image, radius, mode="valid")

        p_L = c * p

        temp = p_sums[:, :, np.newaxis] * Vm - mat_vec_dot(V, pI_sums)
        p_L += vec_vec_dot(image, boxfilter(temp, radius, mode="full"))

        temp = p_sums * mVm - vec_vec_dot(pI_sums, Vm)
        p_L -= boxfilter(temp, radius, mode="full")

        return p_L.flatten()

    if return_diagonal:
        # compute diagonal of L
        diag_L = boxfilter(1.0 - mVm, radius, mode="full")
        temp = 2 * boxfilter(Vm, radius, mode="full")
        temp -= mat_vec_dot(boxfilter(V, radius, mode="full"), image)
        diag_L += vec_vec_dot(image, temp)

        return L_matvec, diag_L.flatten()

    return L_matvec
