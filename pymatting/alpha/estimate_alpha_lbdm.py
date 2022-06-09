from pymatting.util.util import sanity_check_image
from pymatting.laplacian.laplacian import make_linear_system
from pymatting.laplacian.lbdm_laplacian import lbdm_laplacian
from pymatting.preconditioner.ichol import ichol
from pymatting.solver.cg import cg
import numpy as np


def estimate_alpha_lbdm(
    image, trimap, preconditioner=None, laplacian_kwargs={}, cg_kwargs={}
):
    """
    Estimate alpha from an input image and an input trimap using Learning Based Digital Matting as proposed by :cite:`zheng2009learning`.

    Parameters
    ----------
    image: numpy.ndarray
        Image with shape :math:`h \\times  w \\times d` for which the alpha matte should be estimated
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times  w` of the image
    preconditioner: function or scipy.sparse.linalg.LinearOperator
        Function or sparse matrix that applies the preconditioner to a vector (default: ichol)
    laplacian_kwargs: dictionary
        Arguments passed to the :code:`lbdm_laplacian` function
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
    >>> alpha = estimate_alpha_lbdm(
    ...     image,
    ...     trimap,
    ...     laplacian_kwargs={"epsilon": 1e-6},
    ...     cg_kwargs={"maxiter":2000})

    """
    if preconditioner is None:
        preconditioner = ichol

    sanity_check_image(image)

    A, b = make_linear_system(lbdm_laplacian(image, **laplacian_kwargs), trimap)

    x = cg(A, b, M=preconditioner(A), **cg_kwargs)

    alpha = np.clip(x, 0, 1).reshape(trimap.shape)

    return alpha
