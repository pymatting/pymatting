from pymatting.laplacian.laplacian import make_linear_system
from pymatting.laplacian.knn_laplacian import knn_laplacian
from pymatting.preconditioner.jacobi import jacobi
from pymatting.solver.cg import cg
import numpy as np


def estimate_alpha_knn(
    image, trimap, preconditioner=None, laplacian_kwargs={}, cg_kwargs={}
):
    """
    Estimate alpha from an input image and an input trimap using KNN Matting as proposed by :cite:`chen2013knn`.

    Parameters
    -----------------
    image: numpy.ndarray
        Image with shape :math:`h \\times  w \\times d` for which the foreground should be estimated
    trimap: numpy.ndarray
        Trimap with shape :math:`h \\times  w \\times 1` of the image
    preconditioner: function or scipy.sparse.linalg.LinearOperator
        Function or sparse matrix that applies the preconditioner to a vector (default: jacobi)
    laplacian_kwargs: dictionary
        Arguments passed to the :code:`knn_laplacian` function
    cg_kwargs: dictionary
        Arguments passed to the :code:`cg`

    Returns
    ----------------
    alpha: numpy.ndarray
        Estimated alpha matte
    """
    if preconditioner is None:
        preconditioner = jacobi

    A, b = make_linear_system(knn_laplacian(image, **laplacian_kwargs), trimap)

    x = cg(A, b, M=preconditioner(A), **cg_kwargs)

    alpha = np.clip(x, 0, 1).reshape(trimap.shape)

    return alpha
