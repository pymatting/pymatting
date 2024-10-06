import scipy.sparse.linalg
import numpy as np
from pymatting import (
    load_image,
    trimap_split,
    make_linear_system,
    LAPLACIANS,
)


def test_laplacians():
    scale = 0.1

    image_path = "data/lemur/lemur.png"
    trimap_path = "data/lemur/lemur_trimap.png"
    good_alpha_path = "data/lemur/lemur_alpha_fba.png"

    image = load_image(image_path, "rgb", scale, "bilinear")
    trimap = load_image(trimap_path, "gray", scale, "nearest")
    good_alpha = load_image(good_alpha_path, "gray", scale, "bilinear")

    # allow 1% regression
    allowed_error = 0.01

    expected_errors = {
        "cf_laplacian": 1.6383251091411692,
        "knn_laplacian": 2.1748604717689384,
        "lbdm_laplacian": 1.6402653371109384,
        "rw_laplacian": 5.950822534778654,
        "uniform_laplacian": 5.307956449533795,
    }

    for laplacian in LAPLACIANS:
        laplacian_name = laplacian.__name__

        print("Testing", laplacian_name)

        A, b = make_linear_system(laplacian(image), trimap)

        x = scipy.sparse.linalg.spsolve(A, b)

        alpha = np.clip(x, 0, 1).reshape(trimap.shape)

        is_unknown = trimap_split(trimap, flatten=False)[3]

        difference_unknown = np.abs(alpha - good_alpha)[is_unknown]

        error = np.linalg.norm(difference_unknown)

        expected_error = expected_errors[laplacian_name]

        additional_error = (error - expected_error) / expected_error

        print(f'"{laplacian_name}: {error},')

        if additional_error > allowed_error:
            print("Regression:")
            print(laplacian_name)
            print(f"Performance decreased by {100.0 * additional_error:.3f} %")

        assert additional_error < allowed_error
