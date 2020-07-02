import os
import json
import numpy as np
from collections import defaultdict
from config import IMAGE_DIR, ATOL, SCALES, INDICES
from pymatting import (
    load_image,
    trimap_split,
    make_linear_system,
    cg,
    jacobi,
    ichol,
    LAPLACIANS,
    lkm_laplacian,
)


def compute_alpha(image, trimap, laplacian_name, is_fg, is_bg, is_known):
    laplacians = dict((laplacian.__name__, laplacian) for laplacian in LAPLACIANS)

    atol = ATOL * np.sum(is_known)

    if laplacian_name == "lkm_laplacian":
        L_matvec, diag_L = lkm_laplacian(image)

        lambda_value = 100.0

        c = lambda_value * is_known.ravel()
        b = lambda_value * is_fg.ravel()

        inv_diag_A = 1.0 / (diag_L + c)

        A = lambda x: L_matvec(x) + c * x
        # jacobi preconditioner for lkm_laplacian
        M = lambda x: inv_diag_A * x

    else:
        laplacian = laplacians[laplacian_name]

        A, b = make_linear_system(laplacian(image), trimap)

        preconditioner = {
            "knn_laplacian": jacobi,
            "rw_laplacian": jacobi,
            "cf_laplacian": ichol,
            "lbdm_laplacian": ichol,
            "uniform_laplacian": ichol,
        }[laplacian_name]

        M = preconditioner(A)

    x = cg(A, b, M=M, atol=atol)

    alpha = np.clip(x, 0, 1).reshape(trimap.shape)

    return alpha


def main():
    scale = max(SCALES)

    laplacian_names = [laplacian.__name__ for laplacian in LAPLACIANS]
    laplacian_names.append("lkm_laplacian")

    results = defaultdict(dict)

    for index in INDICES:
        name = f"GT{index:02d}.png"

        image_path = os.path.join(IMAGE_DIR, "input_training_lowres", name)
        trimap_path = os.path.join(IMAGE_DIR, "trimap_training_lowres/Trimap1", name)
        true_alpha_path = os.path.join(IMAGE_DIR, "gt_training_lowres", name)

        image = load_image(image_path, "rgb", scale, "bilinear")
        trimap = load_image(trimap_path, "gray", scale, "nearest")
        true_alpha = load_image(true_alpha_path, "gray", scale, "bilinear")

        is_fg, is_bg, is_known, is_unknown = trimap_split(trimap, flatten=False)

        for laplacian_name in laplacian_names:
            print(f"Processing image {name} with {laplacian_name}")

            alpha = compute_alpha(image, trimap, laplacian_name, is_fg, is_bg, is_known)

            difference_unknown = np.abs(alpha - true_alpha)[is_unknown]

            error = np.linalg.norm(difference_unknown)

            results[laplacian_name][str(index)] = error

    os.makedirs("results/", exist_ok=True)
    with open("results/laplacians.json", "w") as f:
        print(results)
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
