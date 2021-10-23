import time
from scipy.spatial import cKDTree
from pymatting import knn
import numpy as np


def run_kdtree():
    np.random.seed(0)

    k = 20
    n_data = 100_000
    n_query = n_data
    dimension = 5

    data_points = np.random.rand(n_data, dimension).astype(np.float32)
    query_points = np.random.rand(n_query, dimension).astype(np.float32)

    t0 = time.perf_counter()

    distances1, indices1 = knn(data_points, query_points, k)

    t1 = time.perf_counter()

    tree = cKDTree(data_points)

    distances2, indices2 = tree.query(query_points, k=k)

    t2 = time.perf_counter()

    debug = False

    if debug:
        print("numba knn: %f seconds" % (t1 - t0))
        print("scipy knn: %f seconds" % (t2 - t1))
        print("number of different indices:", np.sum(indices1 != indices2))
        print("norm(dist1 - dist2):", np.linalg.norm(distances1 - distances2))
        print("")

    # might very rarely be false by random chance if two points are the same
    assert 0 == np.sum(indices1 != indices2)
    assert np.linalg.norm(distances1 - distances2) < 1e-5


def test_kdtree():
    for _ in range(10):
        run_kdtree()
