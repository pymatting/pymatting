import numpy as np


def _rw_laplacian(image, sigma, r):
    h, w = image.shape[:2]
    n = h * w

    m = n * (2 * r + 1) ** 2

    i_inds = np.empty(m, dtype=np.int32)
    j_inds = np.empty(m, dtype=np.int32)
    values = np.empty(m)

    k = 0

    for y in range(h):
        for x in range(w):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    x2 = x + dx
                    y2 = y + dy

                    x2 = max(0, min(w - 1, x2))
                    y2 = max(0, min(h - 1, y2))

                    i = x + y * w
                    j = x2 + y2 * w

                    zi = image[y, x]
                    zj = image[y2, x2]

                    wij = np.exp(-900 * np.linalg.norm(zi - zj) ** 2)

                    i_inds[k] = i
                    j_inds[k] = j

                    values[k] = wij

                    k += 1

    return values, i_inds, j_inds


exports = {
    "_rw_laplacian": (_rw_laplacian, "Tuple((f8[:], i4[:], i4[:]))(f8[:,:,:], f8, i4)"),
}
