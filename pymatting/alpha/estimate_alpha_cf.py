from pymatting.laplacian.laplacian import make_linear_system
from pymatting.laplacian.cf_laplacian import cf_laplacian
from pymatting.preconditioner.ichol import ichol
from pymatting.solver.cg import cg
import numpy as np


def estimate_alpha_cf(
    image, trimap, preconditioner=None, laplacian_kwargs={}, cg_kwargs={}
):
    """
    Estimate alpha from an input image and an input trimap using Closed-Form Alpha Matting as proposed by :cite:`levin2007closed`.

    Parameters
    -----------------
    image: numpy.ndarray
        Image with shape :math:`h \\times  w \\times d` for which the foreground should be estimated
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times  w \\times 1` of the image
    preconditioner: function or scipy.sparse.linalg.LinearOperator
        Function or sparse matrix that applies the preconditioner to a vector (default: ichol)
    laplacian_kwargs: dictionary
        Arguments passed to the :code:`cf_laplacian` function
    cg_kwargs: dictionary
        Arguments passed to the :code:`cg`

    Returns
    ----------------
    alpha: numpy.ndarray
        Estimated alpha matte
    """
    if preconditioner is None:
        preconditioner = ichol

    A, b = make_linear_system(cf_laplacian(image, **laplacian_kwargs), trimap)

    x = cg(A, b, M=preconditioner(A), **cg_kwargs)

    alpha = np.clip(x, 0, 1).reshape(trimap.shape)

    return alpha
