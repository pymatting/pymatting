import numpy as np
from pymatting_aot.aot import _estimate_fb_ml


def estimate_foreground_ml(
    image,
    alpha,
    regularization=1e-5,
    n_small_iterations=10,
    n_big_iterations=2,
    small_size=32,
    return_background=False,
    gradient_weight=1.0,
):
    """Estimates the foreground of an image given its alpha matte.

    See :cite:`germer2020multilevel` for reference.

    Parameters
    ----------
    image: numpy.ndarray
        Input image with shape :math:`h \\times  w \\times d`
    alpha: numpy.ndarray
        Input alpha matte shape :math:`h \\times  w \\times 1`
    regularization: float
        Regularization strength :math:`\\epsilon`, defaults to :math:`10^{-5}`.
        Higher regularization results in smoother colors.
    n_small_iterations: int
        Number of iterations performed on small scale, defaults to :math:`10`
    n_big_iterations: int
        Number of iterations performed on large scale, defaults to :math:`2`
    small_size: int
        Threshold that determines at which size `n_small_iterations` should be used
    return_background: bool
        Whether to return the estimated background in addition to the foreground
    gradient_weight: float
        Larger values enforce smoother foregrounds, defaults to :math:`1`

    Returns
    -------
    F: numpy.ndarray
        Extracted foreground
    B: numpy.ndarray
        Extracted background

    Example
    -------
    >>> from pymatting import *
    >>> image = load_image("data/lemur/lemur.png", "RGB")
    >>> alpha = load_image("data/lemur/lemur_alpha.png", "GRAY")
    >>> F = estimate_foreground_ml(image, alpha, return_background=False)
    >>> F, B = estimate_foreground_ml(image, alpha, return_background=True)

    See Also
    ----
    stack_images: This function can be used to place the foreground on a new background.
    """

    foreground, background = _estimate_fb_ml(
        image.astype(np.float32),
        alpha.astype(np.float32),
        regularization,
        n_small_iterations,
        n_big_iterations,
        small_size,
        gradient_weight,
    )

    if return_background:
        return foreground, background

    return foreground
