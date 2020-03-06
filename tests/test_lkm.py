import numpy as np
from pymatting import (
    load_image,
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
    epsilon = 1e-7
    radius = 2
    image_dir = "data"

    name = f"GT{index:02d}"

    image = load_image(
        f"{image_dir}/input_training_lowres/{name}.png", "rgb", scale, "bilinear"
    )
    trimap = load_image(
        f"{image_dir}/trimap_training_lowres/Trimap1/{name}.png",
        "gray",
        scale,
        "nearest",
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

    difference = np.linalg.norm(x_lkm - x_cf)

    assert difference < 2e-4
    assert abs(lkm_callback.n - cf_callback.n) <= 2
