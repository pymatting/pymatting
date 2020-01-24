class CounterCallback(object):
    """Callback to count number of iterations of iterative solvers."""

    def __init__(self):
        self.n = 0

    def __call__(self, A, x, b, norm_b, r, norm_r):
        self.n += 1


class ProgressCallback(object):
    """
    Callback to count number of iterations of iterative solvers.
    Also prints residual error.
    """

    def __init__(self):
        self.n = 0

    def __call__(self, A, x, b, norm_b, r, norm_r):
        self.n += 1

        print("iteration %7d - %e (%.20f)" % (self.n, norm_r, norm_r))
