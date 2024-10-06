import numpy as np
from pymatting import *


def test_alpha():
    scale = 0.1

    image_path = "data/lemur/lemur.png"
    trimap_path = "data/lemur/lemur_trimap.png"
    good_alpha_path = "data/lemur/lemur_alpha_fba.png"

    image = load_image(image_path, "rgb", scale, "bilinear")
    trimap = load_image(trimap_path, "gray", scale, "nearest")
    good_alpha = load_image(good_alpha_path, "gray", scale, "bilinear")

    for estimate_alpha_method, max_error in [
        (estimate_alpha, 1.68),
        (estimate_alpha_cf, 1.68),
        (estimate_alpha_lbdm, 1.68),
        (estimate_alpha_rw, 6.0),
        (estimate_alpha_knn, 2.2),
        (lambda *args: estimate_alpha_lkm(*args, laplacian_kwargs={"radius": 1}), 1.68),
        (estimate_alpha_sm, 2.1),
    ]:
        alpha = estimate_alpha_method(image, trimap)

        error = np.linalg.norm(alpha - good_alpha)

        assert error < max_error
