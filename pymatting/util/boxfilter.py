from pymatting.util.util import apply_to_channels
import numpy as np
from numba import njit, prange


@njit("f8[:, :](f8[:, :], i8)", cache=True, parallel=True)
def boxfilter_rows_valid(src, r):
    m, n = src.shape

    dst = np.zeros((m, n - 2 * r))

    for i in prange(m):
        for j_dst in range(1):
            s = 0.0
            for j_src in range(j_dst, j_dst + 2 * r + 1):
                s += src[i, j_src]
            dst[i, j_dst] = s

        for j_dst in range(1, dst.shape[1]):
            j_src = j_dst - 1
            s -= src[i, j_src]

            j_src = j_dst + 2 * r
            s += src[i, j_src]

            dst[i, j_dst] = s

    return dst


@njit("f8[:, :](f8[:, :], i8)", cache=True, parallel=True)
def boxfilter_rows_same(src, r):
    m, n = src.shape

    dst = np.zeros((m, n))

    for i in prange(m):
        for j_dst in range(1):
            s = 0.0
            for j_src in range(j_dst + r + 1):
                s += src[i, j_src]
            dst[i, j_dst] = s

        for j_dst in range(1, r + 1):
            s += src[i, j_dst + r]
            dst[i, j_dst] = s

        for j_dst in range(r + 1, n - r):
            s -= src[i, j_dst - r - 1]
            s += src[i, j_dst + r]
            dst[i, j_dst] = s

        for j_dst in range(n - r, n):
            s -= src[i, j_dst - r - 1]
            dst[i, j_dst] = s

    return dst


@njit("f8[:, :](f8[:, :], i8)", cache=True, parallel=True)
def boxfilter_rows_full(src, r):
    m, n = src.shape

    dst = np.zeros((m, n + 2 * r))

    for i in prange(m):
        for j_dst in range(1):
            s = 0.0
            for j_src in range(j_dst + r + 1 - r):
                s += src[i, j_src]
            dst[i, j_dst] = s

        for j_dst in range(1, 2 * r + 1):
            s += src[i, j_dst]
            dst[i, j_dst] = s

        for j_dst in range(2 * r + 1, dst.shape[1] - 2 * r):
            s -= src[i, j_dst - r - r - 1]
            s += src[i, j_dst]
            dst[i, j_dst] = s

        for j_dst in range(dst.shape[1] - 2 * r, dst.shape[1]):
            s -= src[i, j_dst - r - r - 1]
            dst[i, j_dst] = s

    return dst


@apply_to_channels
def boxfilter(src, radius=3, mode="same"):
    """Computes the boxfilter (uniform blur, i.e. blur with kernel :code:`np.ones(radius, radius)`) of an input image.

    Depending on the mode, the input image of size :math:`(h, w)` is either of shape

    * :math:`(h - 2 r, w - 2 r)` in case of 'valid' mode
    * :math:`(h, w)` in case of 'same' mode
    * :math:`(h + 2 r, w + 2 r)` in case of 'full' mode

    .. image:: figures/padding.png

    Parameters
    ----------
    src: numpy.ndarray
        Input image having either shape :math:`h \\times w \\times d`  or :math:`h \\times w`
    radius: int
        Radius of boxfilter, defaults to :math:`3`
    mode: str
        One of 'valid', 'same' or 'full', defaults to 'same'

    Returns
    -------
    dst: numpy.ndarray
        Blurred image

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> boxfilter(np.eye(5), radius=2, mode="valid")
    array([[5.]])
    >>> boxfilter(np.eye(5), radius=2, mode="same")
    array([[3., 3., 3., 2., 1.],
           [3., 4., 4., 3., 2.],
           [3., 4., 5., 4., 3.],
           [2., 3., 4., 4., 3.],
           [1., 2., 3., 3., 3.]])
    >>> boxfilter(np.eye(5), radius=2, mode="full")
    array([[1., 1., 1., 1., 1., 0., 0., 0., 0.],
           [1., 2., 2., 2., 2., 1., 0., 0., 0.],
           [1., 2., 3., 3., 3., 2., 1., 0., 0.],
           [1., 2., 3., 4., 4., 3., 2., 1., 0.],
           [1., 2., 3., 4., 5., 4., 3., 2., 1.],
           [0., 1., 2., 3., 4., 4., 3., 2., 1.],
           [0., 0., 1., 2., 3., 3., 3., 2., 1.],
           [0., 0., 0., 1., 2., 2., 2., 2., 1.],
           [0., 0., 0., 0., 1., 1., 1., 1., 1.]])

    """
    assert radius > 0
    assert mode in ["valid", "same", "full"]
    assert src.shape[0] >= 2 * radius + 1
    assert src.shape[1] >= 2 * radius + 1

    boxfilter_rows = {
        "valid": boxfilter_rows_valid,
        "same": boxfilter_rows_same,
        "full": boxfilter_rows_full,
    }[mode]

    tmp = src.T
    tmp = boxfilter_rows(tmp, radius)
    tmp = tmp.T
    dst = boxfilter_rows(tmp, radius)

    return dst
