import numpy as np


def boxfilter_rows_valid(src, r):
    m, n = src.shape

    dst = np.zeros((m, n - 2 * r))

    for i in range(m):
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


def boxfilter_rows_same(src, r):
    m, n = src.shape

    dst = np.zeros((m, n))

    for i in range(m):
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


def boxfilter_rows_full(src, r):
    m, n = src.shape

    dst = np.zeros((m, n + 2 * r))

    for i in range(m):
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


exports = {
    "boxfilter_rows_valid": (boxfilter_rows_valid, "f8[:, :](f8[:, :], i8)"),
    "boxfilter_rows_same": (boxfilter_rows_same, "f8[:, :](f8[:, :], i8)"),
    "boxfilter_rows_full": (boxfilter_rows_full, "f8[:, :](f8[:, :], i8)"),
}
