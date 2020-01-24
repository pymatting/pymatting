from pymatting.laplacian.lkm_laplacian import lkm_laplacian
from pymatting.util.util import trimap_split
from pymatting.solver.cg import cg
import numpy as np


def estimate_alpha_lkm(image, trimap, laplacian_kwargs={}, cg_kwargs={}):
    """
    Estimate alpha from an input image and an input trimap using Learning Based Digital Matting as proposed by :cite:`he2010fast`.

    Parameters
    -----------------
    image: numpy.ndarray
        Image with shape :math:`h \\times  w \\times d` for which the foreground should be estimated
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times  w \\times 1` of the image
    laplacian_kwargs: dictionary
        Arguments passed to the :code:`lkm_laplacian` function
    cg_kwargs: dictionary
        Arguments passed to the :code:`cg`

    Returns
    ----------------
    alpha: numpy.ndarray
        Estimated alpha matte
    """
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
