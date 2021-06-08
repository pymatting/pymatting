from pymatting.preconditioner.ichol import ichol
from pymatting.util.util import sparse_conv_matrix_with_offsets
from pymatting.solver.cg import cg
import scipy.sparse
import numpy as np


def estimate_foreground_cf(
    image,
    alpha,
    regularization=1e-5,
    rtol=1e-5,
    neighbors=[(-1, 0), (1, 0), (0, -1), (0, 1)],
    return_background=False,
    foreground_guess=None,
    background_guess=None,
    ichol_kwargs={},
    cg_kwargs={},
):
    """Estimates the foreground of an image given alpha matte and image.

    This method is based on the publication :cite:`levin2007closed`.

    Parameters
    ----------
    image: numpy.ndarray
        Input image with shape :math:`h \\times  w \\times d`
    alpha: numpy.ndarray
        Input alpha matte with shape :math:`h \\times  w`
    regularization: float
        Regularization strength :math:`\\epsilon`, defaults to :math:`10^{-5}`
    neighbors: list of tuples of ints
        List of relative positions that define the neighborhood of a pixel
    return_background: bool
        Whether to return the estimated background in addition to the foreground
    foreground_guess: numpy.ndarray
        An initial guess for the foreground image in order to accelerate convergence.
        Using input image by default.
    background_guess: numpy.ndarray
        An initial guess for the background image.
        Using input image by default.
    ichol_kwargs: dictionary
        Keyword arguments for the incomplete Cholesky preconditioner
    cg_kwargs: dictionary
        Keyword arguments for the conjugate gradient descent solver

    Returns
    -------
    F: numpy.ndarray
        Extracted foreground
    B: numpy.ndarray
        Extracted background (not returned by default)

    Example
    -------
    >>> from pymatting import *
    >>> image = load_image("data/lemur/lemur.png", "RGB")
    >>> alpha = load_image("data/lemur/lemur_alpha.png", "GRAY")
    >>> F = estimate_foreground_cf(image, alpha, return_background=False)
    >>> F, B = estimate_foreground_cf(image, alpha, return_background=True)


    See Also
    --------
    stack_images: This function can be used to place the foreground on a new background.
    """
    h, w, d = image.shape

    assert alpha.shape == (h, w)

    n = w * h

    a = alpha.flatten()

    S = None
    for dx, dy in neighbors:
        # directional derivative
        D = sparse_conv_matrix_with_offsets(w, h, [1.0, -1.0], [0, dx], [0, dy])

        S2 = D.T.dot(scipy.sparse.diags(regularization + np.abs(D.dot(a)))).dot(D)

        S = S2 if S is None else S + S2

        del D, S2

    V = scipy.sparse.bmat([[S, None], [None, S]])

    del S

    U = scipy.sparse.bmat([[scipy.sparse.diags(a), scipy.sparse.diags(1 - a)]])

    A = U.T.dot(U) + V

    A.sum_duplicates()

    del V

    # Compute preconditioner for faster convergence
    precondition = ichol(A, **ichol_kwargs)

    # By default use input image as initialization
    foreground = (image if foreground_guess is None else foreground_guess).copy()
    background = (image if background_guess is None else background_guess).copy()

    # For each image color channel
    for channel in range(d):
        image_channel = image[:, :, channel].flatten()

        b = U.T.dot(image_channel)

        # Initialization vector for conjugate gradient descent
        f0 = foreground[:, :, channel].flatten()
        b0 = background[:, :, channel].flatten()
        fb = np.concatenate([f0, b0])

        # Solve linear system
        # A fb = b
        # for fb using conjugate gradient descent
        fb = cg(A, b, x0=fb, M=precondition, rtol=rtol, **cg_kwargs)

        foreground[:, :, channel] = fb[:n].reshape(h, w)
        background[:, :, channel] = fb[n:].reshape(h, w)

    foreground = np.clip(foreground, 0, 1)
    background = np.clip(background, 0, 1)

    if return_background:
        return foreground, background

    return foreground
