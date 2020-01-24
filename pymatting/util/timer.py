import time


class Timer(object):
    """Timer for benchmarking"""

    def __init__(self):
        """Starts a timer"""

        self.t = time.perf_counter()

    def stop(self, message=None):
        """Return and print time since last stop-call or initialization.
        Also print elapsed time if message is provided.
        
        Parameters
        ----------
        message: str
            Message to print in front of passed seconds
        """
        delta_time = time.perf_counter() - self.t

        if message is not None:
            print(message, " - %f seconds" % delta_time)

        self.t = time.perf_counter()

        return delta_time
