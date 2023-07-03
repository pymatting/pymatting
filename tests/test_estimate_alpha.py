import numpy as np
from pymatting import *


def test_alpha():
    scale = 0.1
    index = 1
    name = f"GT{index:02d}"
    image_dir = "data"

    image = load_image(
        f"{image_dir}/input_training_lowres/{name}.png", "rgb", scale, "bilinear"
    )
    trimap = load_image(
        f"{image_dir}/trimap_training_lowres/Trimap1/{name}.png",
        "gray",
        scale,
        "nearest",
    )
    true_alpha = load_image(
        f"{image_dir}/gt_training_lowres/{name}.png", "gray", scale, "nearest"
    )

    for estimate_alpha_method, max_error in [
        (estimate_alpha, 4.4),
        (estimate_alpha_cf, 4.4),
        (estimate_alpha_lbdm, 4.4),
        (estimate_alpha_rw, 6.2),
        (estimate_alpha_knn, 3.6),
        (lambda *args: estimate_alpha_lkm(*args, laplacian_kwargs={"radius": 1}), 4.4),
        (estimate_alpha_sm, 5.1),
    ]:
        alpha = estimate_alpha_method(image, trimap)

        error = np.linalg.norm(alpha - true_alpha)

        assert error < max_error
