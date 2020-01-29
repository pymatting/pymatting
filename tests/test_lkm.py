import scipy.sparse.linalg
import numpy as np
import time
import json
import os
from pymatting import (
    load_image,
    show_images,
    trimap_split,
    make_linear_system,
    cf_laplacian,
    lkm_laplacian,
    jacobi,
    ProgressCallback,
    cg,
)


def test_lkm():
    index = 1
    scale = 0.2
    atol = 1e-5
    epsilon = 1e-7
    radius = 2
    image_dir = "data"

    errors = []

    name = f"GT{index:02d}"
    # print(name)

    image = load_image(
        f"{image_dir}/input_training_lowres/{name}.png", "rgb", scale, "bilinear"
    )
    trimap = load_image(
        f"{image_dir}/trimap_training_lowres/Trimap1/{name}.png",
        "gray",
        scale,
        "bilinear",
    )
    true_alpha = load_image(
        f"{image_dir}/gt_training_lowres/{name}.png", "gray", scale, "nearest"
    )

    L_lkm, diag_L_lkm = lkm_laplacian(image, epsilon=epsilon, radius=radius)

    L_cf = cf_laplacian(image, epsilon=epsilon, radius=radius)

    A_cf, b, c = make_linear_system(L_cf, trimap, return_c=True)

    def A_lkm(x):
        return L_lkm(x) + c * x

    inv_diag_A_lkm = 1.0 / (diag_L_lkm + c)

    def jacobi_lkm(r):
        return inv_diag_A_lkm * r

    jacobi_cf = jacobi(A_cf)

    lkm_callback = ProgressCallback()
    cf_callback = ProgressCallback()

    x_lkm = cg(A_lkm, b, M=jacobi_lkm)
    x_cf = cg(A_cf, b, M=jacobi_cf)

    if 0:
        # show result
        show_images(
            [x_lkm.reshape(trimap.shape), x_cf.reshape(trimap.shape),]
        )

    difference = np.linalg.norm(x_lkm - x_cf)

    # print("norm(x_lkm - x_cf):")
    # print(difference)
    # print("iterations:")
    # print("lkm:", lkm_callback.n)
    # print("cf:", cf_callback.n)

    assert difference < 1e-5
    assert abs(lkm_callback.n - cf_callback.n) <= 2

    # print("ok")
