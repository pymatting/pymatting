from pymatting.util.boxfilter import boxfilter
import numpy as np
import scipy.signal
import time


def run_boxfilter(m, n, r, mode, n_runs):
    src = np.random.rand(m, n)
    kernel = np.ones((2 * r + 1, 2 * r + 1))

    dst_ground_truth = scipy.signal.correlate2d(src, kernel, mode=mode)

    for _ in range(n_runs):
        t = time.perf_counter()

        dst = boxfilter(src, r, mode)

        dt = time.perf_counter() - t

        max_error = np.max(np.abs(dst - dst_ground_truth))

        assert np.all(np.isfinite(dst))

        print(
            "%f gbyte/sec, %f seconds, max_error: %.20f %e %d-by-%d, r=%d"
            % (dst.nbytes * 1e-9 / dt, dt, max_error, max_error, m, n, r)
        )

        assert max_error < 1e-10


def test_boxfilter():
    modes = ["valid", "same", "full"]

    for mode in modes:
        print("testing boxfilter mode", mode)
        for r in range(1, 5):
            min_size = 2 * r + 1
            for m in range(min_size, min_size + 10):
                for n in range(min_size, min_size + 10):
                    run_boxfilter(m, n, r, mode, 2)

        run_boxfilter(50, 50, 4, mode, 2)

    for mode in modes:
        print("testing boxfilter mode", mode)
        for r in range(1, 20):
            run_boxfilter(256, 256, r, mode, 1)
