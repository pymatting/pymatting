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
    scale = 0.125
    epsilon = 1e-7
    radius = 2

    image_path = "data/lemur/lemur.png"
    trimap_path = "data/lemur/lemur_trimap.png"

    image = load_image(image_path, "rgb", scale, "bilinear")
    trimap = load_image(trimap_path, "gray", scale, "nearest")

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

    x_lkm = cg(A_lkm, b, M=jacobi_lkm, callback=lkm_callback)
    x_cf = cg(A_cf, b, M=jacobi_cf, callback=cf_callback)

    difference = np.linalg.norm(x_lkm - x_cf)

    assert difference < 1e-5
    assert abs(lkm_callback.n - cf_callback.n) <= 2
