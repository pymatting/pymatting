from pymatting import *
import numpy as np


def test_util():
    @apply_to_channels
    def conv(image, kernel):
        height, width = image.shape
        A = sparse_conv_matrix(width, height, kernel)
        return A.dot(image.flatten()).reshape(image.shape)

    image = np.random.rand(5, 6, 7)
    kernel = np.random.rand(3, 3)

    convolved = conv(image, kernel)

    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            for c in range(image.shape[2]):
                patch = image[y - 1 : y + 2, x - 1 : x + 2, c]

                error = abs(np.sum(patch * kernel) - convolved[y, x, c])
                assert error < 1e-10

    assert np.allclose(normalize([1, 2, 3]), [0.0, 0.5, 1.0])

    a = np.random.rand(13, 7)
    b = np.random.rand(13, 7)
    A = np.random.rand(13, 5, 7)
    assert np.allclose(vec_vec_dot(a, b), [np.inner(ai, bi) for ai, bi in zip(a, b)])
    assert np.allclose(vec_vec_outer(a, b), [np.outer(ai, bi) for ai, bi in zip(a, b)])
    assert np.allclose(mat_vec_dot(A, b), [np.dot(Ai, bi) for Ai, bi in zip(A, b)])
