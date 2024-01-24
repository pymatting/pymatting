import numpy as np
from numba import njit, prange


@njit("(f8[:],)", cache=True, nogil=True)
def _propagate_1d_first_pass(d):
    n = len(d)
    for i1 in range(1, n):
        i2 = i1 - 1
        d[i1] = min(d[i1], d[i2] + 1)

    for i1 in range(n - 2, -1, -1):
        i2 = i1 + 1
        d[i1] = min(d[i1], d[i2] + 1)


@njit("(f8[:], i4[:], f8[:], f8[:])", cache=True, nogil=True)
def _propagate_1d(d, v, z, f):
    nx = len(d)
    k = -1
    s = -np.inf
    for x in range(nx):
        d_yx = d[x]

        if d_yx == np.inf:
            continue

        fx = d_yx * d_yx

        f[x] = fx

        while k >= 0:
            vk = v[k]

            s = 0.5 * (fx + x * x - f[vk] - vk * vk) / (x - vk)

            if s > z[k]:
                break

            k -= 1

        k += 1
        v[k] = x
        z[k] = s
        z[k + 1] = np.inf

    if k < 0:
        return

    k = 0
    for x in range(nx):
        while z[k + 1] < x:
            k += 1

        vk = v[k]
        dx = x - vk

        d[x] = np.sqrt(dx * dx + f[vk])


@njit("(f8[:, :],)", cache=True, parallel=True, nogil=True)
def _propagate_distance(distance):
    ny, nx = distance.shape

    for x in prange(nx):
        _propagate_1d_first_pass(distance[:, x])

    v = np.zeros((ny, nx), dtype=np.int32)
    z = np.zeros((ny, nx + 1))
    f = np.zeros((ny, nx))

    for y in prange(ny):
        _propagate_1d(distance[y], v[y], z[y], f[y])


def distance_transform(mask):
    """
    For every non-zero value, compute the distance to the closest zero value.
    Based on :cite:`felzenszwalb2012distance`.

    Parameters
    ----------
    mask: numpy.ndarray
        2D matrix of zero and nonzero values.

    Returns
    -------
    distance: numpy.ndarray
        Distance to closest zero-valued pixel.

    Example
    -------
    >>> from pymatting import *
    >>> import numpy as np
    >>> mask = np.random.rand(10, 20) < 0.9
    >>> distance = distance_transform(mask)
    """
    # Ensure that mask.dtype is boolean
    if mask.dtype != np.bool_:
        mask = mask != 0

    ny, nx = mask.shape
    distance = np.zeros((ny, nx), np.float64)
    distance[mask] = np.inf

    _propagate_distance(distance)

    return distance
