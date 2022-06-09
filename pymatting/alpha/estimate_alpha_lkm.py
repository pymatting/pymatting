from pymatting.util.util import sanity_check_image
from pymatting.laplacian.lkm_laplacian import lkm_laplacian
from pymatting.util.util import trimap_split
from pymatting.solver.cg import cg
import numpy as np


def estimate_alpha_lkm(image, trimap, laplacian_kwargs={}, cg_kwargs={}):
    """
    Estimate alpha from an input image and an input trimap as described in Fast Matting Using Large Kernel Matting Laplacian Matrices by :cite:`he2010fast`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h \\times  w \\times d` for which the alpha matte should be estimated
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times  w` of the image
    laplacian_kwargs: dictionary
        Arguments passed to the :code:`lkm_laplacian` function
    cg_kwargs: dictionary
        Arguments passed to the :code:`cg` solver

    Returns
    -------
    alpha: numpy.ndarray
        Estimated alpha matte

    Example
    -------
    >>> from pymatting import *
    >>> image = load_image("data/lemur/lemur.png", "RGB")
    >>> trimap = load_image("data/lemur/lemur_trimap.png", "GRAY")
    >>> alpha = estimate_alpha_lkm(
    ...     image,
    ...     trimap,
    ...     laplacian_kwargs={"epsilon": 1e-6, "radius": 15},
    ...     cg_kwargs={"maxiter":2000})

    """

    sanity_check_image(image)

    L_matvec, diag_L = lkm_laplacian(image, **laplacian_kwargs)

    is_fg, is_bg, is_known, is_unknown = trimap_split(trimap)

    lambda_value = 100.0

    c = lambda_value * is_known
    b = lambda_value * is_fg

    inv_diag_A = 1.0 / (diag_L + c)

    def A_matvec(x):
        return L_matvec(x) + c * x

    def jacobi(x):
        return inv_diag_A * x

    x = cg(A_matvec, b, M=jacobi, **cg_kwargs)

    alpha = np.clip(x, 0, 1).reshape(trimap.shape)

    return alpha
