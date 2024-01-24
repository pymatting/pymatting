from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt
from pymatting import distance_transform
import numpy as np


def distance_transform_naive(mask):
    # O(w^2 h^2) complexity, only used for testing
    h, w = mask.shape
    distance = np.empty((h, w))

    for y in range(h):
        for x in range(w):
            # If mask is zero, do nothing
            if not mask[y, x]:
                distance[y, x] = 0
                continue

            # Compute distance to closest point with mask value 0
            d = np.inf
            for y2 in range(h):
                for x2 in range(w):
                    if not mask[y2, x2]:
                        dx = x - x2
                        dy = y - y2
                        d = min(d, dx * dx + dy * dy)

            distance[y, x] = d

    return np.sqrt(distance)


def distance_transform_naive_vectorized(mask):
    # O(w^2 h^2) complexity and memory usage, only used for testing
    h, w = mask.shape
    px, py = np.mgrid[:h, :w]
    points = np.column_stack([px.ravel(), py.ravel()])
    distances = np.zeros((h, w))
    distances.ravel()[mask.ravel() != 0] = cdist(
        points[mask.ravel() != 0], points[mask.ravel() == 0]
    ).min(axis=1)
    return distances


def test_distance():
    mask = np.random.rand(13, 29) < 0.95

    distance_naive = distance_transform_naive(mask)
    distance_vectorized = distance_transform_naive_vectorized(mask)

    assert np.allclose(distance_naive, distance_vectorized)

    for _ in range(5):
        w = np.random.randint(50, 100)
        h = np.random.randint(50, 100)
        mask = np.random.rand(h, w) < 0.95

        distance_scipy = distance_transform_edt(mask)

        distance_numba = distance_transform(mask)

        distance_vectorized = distance_transform_naive_vectorized(mask)

        assert np.allclose(distance_scipy, distance_numba)
        assert np.allclose(distance_vectorized, distance_numba)
